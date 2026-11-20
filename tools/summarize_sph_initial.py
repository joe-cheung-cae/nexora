"""Summarize the four physical-mass, open-tank initial fields from the SPH fixture."""

import argparse
import csv
import gzip
import math
from collections import defaultdict
from itertools import product
from pathlib import Path


HEIGHT, RHO0, C0, GAMMA, G = 6.0, 1000.0, 80.0, 7.0, 9.81


def kernel_integral(depth_over_h):
    """Exact M4 integral over the fluid + wall cuboid for a point inside the tank."""
    # The filled wall extends by >= 2h on every side and below the bottom.
    # The only intersecting exterior is the spherical cap above the top plane.
    # For one truncated cube (a-q)^3 the cap's radial integral is
    # integral_s^a q*(q-s)*(a-q)^3 dq = s*(a-s)^5/20 + (a-s)^6/60.
    def cap_term(a):
        length = max(a - depth_over_h, 0.0)
        return depth_over_h * length**5 / 20 + length**6 / 60

    return 1 - (cap_term(2) - 4 * cap_term(1)) / 2


def region(x, y, z, h):
    if HEIGHT - z < 2 * h:
        return "free_surface"
    near_x = min(x, HEIGHT - x) < 2 * h
    near_y = min(y, HEIGHT - y) < 2 * h
    if z < 2 * h:
        return "bottom_corner" if near_x and near_y else "bottom_other"
    return "side_wall" if near_x or near_y else "interior"


def summarize(group):
    # Equal lattice volume within each case for the kernel and linear-field
    # errors. The hydrostatic pressure/acceleration use physical particle mass.
    count = len(group)
    mass = math.fsum(p["mass"] for p in group)

    def mean(xs):
        return math.fsum(xs) / count

    def rms(xs):
        return math.sqrt(mean(x * x for x in xs))

    def mass_rms(key):
        return math.sqrt(math.fsum(p["mass"] * p[key]**2 for p in group) / mass)

    acceleration_rms = mass_rms("net_acceleration")
    return {
        "count": count, "mass_kg": mass,
        "neighbors_mean": mean(p["neighbors"] for p in group),
        "S_mean": mean(p["S"] for p in group),
        "I_mean": mean(p["I"] for p in group),
        "S_minus_I_mean": mean(p["quadrature"] for p in group),
        "S_minus_I_rms": rms(p["quadrature"] for p in group),
        "I_minus_1_mean": mean(p["truncation"] for p in group),
        "I_minus_1_rms": rms(p["truncation"] for p in group),
        "grad_z_field_error_rms": rms(p["gradient_error"] for p in group),
        "pressure_error_rms_kPa": mass_rms("pressure_error") / 1000,
        "net_acceleration_rms_m_s2": acceleration_rms,
        "net_acceleration_rms_over_g": acceleration_rms / G,
    }


def matched_probes():
    """Evaluate the two h=0.5 lattices at identical positions, including off-particle points."""
    h = 0.5
    for s in (0.5, 1.5, 5.5):
        center = (2.75, 2.75, HEIGHT - h * s)
        for n in (12, 24):
            dx = HEIGHT / n
            # These probes are over 2h from the four sides and the bottom.
            weights = []
            gradients = [[], [], []]
            norm = dx**3 / (4 * math.pi * h**3)
            for index in product(range(n), repeat=3):
                delta = tuple((i + 0.5) * dx - c for i, c in zip(index, center))
                q = math.hypot(*delta) / h
                if q >= 2:
                    continue
                outer, inner = 2 - q, max(1 - q, 0.0)
                weights.append(norm * (outer**3 - 4 * inner**3))
                if q > 0:
                    factor = norm * (3 * outer**2 - 12 * inner**2) / (q * h**2)
                    for axis in range(3):
                        gradients[axis].append(factor * delta[2] * delta[axis])
            S = math.fsum(weights)
            I = kernel_integral(s)
            grad = tuple(math.fsum(g) for g in gradients)
            yield dict(n=n, dx=dx, h=h, x=center[0], y=center[1], z=center[2],
                       depth_over_h=s, S=S, I=I, S_minus_I=S-I,
                       grad_z_field_error=math.dist(grad, (0, 0, 1)))


def read_groups(path):
    groups = defaultdict(list)
    ids = defaultdict(set)
    open_input = gzip.open if path.suffix == ".gz" else open
    with open_input(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            n, dx, h = int(row["n"]), float(row["dx"]), float(row["h"])
            case = (row["case"], n, dx, h)
            particle_id = int(row["id"])
            if particle_id in ids[case]:
                raise ValueError(f"duplicate ID {particle_id} in {case}")
            ids[case].add(particle_id)
            x, y, z, S, gx, gy, gz, p, ax, ay, az = (
                float(row[k]) for k in ("x", "y", "z", "S", "grad_x", "grad_y", "grad_z",
                                        "p", "ap_x", "ap_y", "ap_z")
            )
            s = (HEIGHT - z) / h
            integral = kernel_integral(s)
            log_rho_ratio = math.log1p((GAMMA - 1) * G * (HEIGHT - z) / C0**2) / (GAMMA - 1)
            reference_pressure = RHO0 * C0**2 / GAMMA * math.expm1(GAMMA * log_rho_ratio)
            particle = {
                "mass": RHO0 * math.exp(log_rho_ratio) * dx**3,
                "S": S, "I": integral, "quadrature": S - integral, "truncation": integral - 1,
                "gradient_error": math.hypot(gx, gy, gz - 1),
                "pressure_error": p - reference_pressure,
                "net_acceleration": math.hypot(ax, ay, az - G),
                "neighbors": int(row["neighbor_count"]),
            }
            if not all(math.isfinite(v) for v in particle.values()):
                raise ValueError(f"nonfinite field in {case}, ID {particle_id}")
            groups[case, "region", "all"].append(particle)
            groups[case, "region", region(x, y, z, h)].append(particle)
            if s < 2:
                groups[case, "depth/h", f"{s:g}"].append(particle)
                groups[case, "depth/h band", "[0,1)" if s < 1 else "[1,2)"].append(particle)
    expected = {(6, 1.0, 1.0), (12, .5, .5), (24, .25, .25), (24, .25, .5)}
    if {key[1:] for key in ids} != expected or len(ids) != 4:
        raise ValueError("expected the three fixed h/dx cases and one fixed-h control")
    for (_, n, _, _), case_ids in ids.items():
        if len(case_ids) != n**3:
            raise ValueError(f"incomplete {n}³ field")
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("particles", type=Path, help="exported particle CSV or .csv.gz")
    parser.add_argument("output", type=Path, help="regional and depth-resolved summary CSV")
    parser.add_argument("--matched-probes", type=Path, help="write paired fixed-h probe values")
    args = parser.parse_args()
    groups = read_groups(args.particles)
    with args.output.open("w", newline="") as stream:
        writer = None
        for (case, grouping, selection), particles in groups.items():
            row = dict(zip(("case", "n", "dx_m", "h_m"), case))
            row.update(grouping=grouping, selection=selection, **summarize(particles))
            if writer is None:
                writer = csv.DictWriter(stream, row, lineterminator="\n")
                writer.writeheader()
            writer.writerow(row)
    if args.matched_probes:
        probes = list(matched_probes())
        with args.matched_probes.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, probes[0], lineterminator="\n")
            writer.writeheader()
            writer.writerows(probes)


if __name__ == "__main__":
    main()
