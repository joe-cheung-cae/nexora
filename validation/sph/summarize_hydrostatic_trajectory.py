#!/usr/bin/env python3
"""Recompute sampled water-tank diagnostics and summarize accepted-step logs."""

import argparse
from collections import defaultdict
import csv
from itertools import groupby
import math
from pathlib import Path


FACES = ("x_min", "x_max", "y_min", "y_max", "bottom")
FACE_PLANES = (("x", 0), ("x", 6), ("y", 0), ("y", 6), ("z", 0))
DEPTH = 6
END = 2.34765625
GRAVITY = 9.81
RHO0 = 1000
SOUND_SPEED = 80
GAMMA = 7


def reference_pressure(z):
    relative_depth = (GAMMA - 1) * GRAVITY * max(DEPTH - z, 0) / SOUND_SPEED**2
    return RHO0 * SOUND_SPEED**2 / GAMMA * math.expm1(
        GAMMA / (GAMMA - 1) * math.log1p(relative_depth)
    )


def read_rows(path):
    with path.open(newline="") as stream:
        yield from csv.DictReader(stream)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(actual, expected, label):
    require(
        math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-10),
        f"{label}: computed {actual:.17g}, recorded {expected:.17g}",
    )


def moments(particles):
    mass = math.fsum(p["mass"] for p in particles)
    if not particles:
        return dict(
            particle_count=0, mass_kg=mass, mean_pressure_pa=None,
            mean_reference_pressure_pa=None, pressure_rms_error_pa=None,
        )
    return dict(
        particle_count=len(particles),
        mass_kg=mass,
        mean_pressure_pa=math.fsum(p["mass"] * p["pressure"] for p in particles) / mass,
        mean_reference_pressure_pa=math.fsum(
            p["mass"] * p["reference"] for p in particles
        ) / mass,
        pressure_rms_error_pa=math.sqrt(
            math.fsum(
                p["mass"] * (p["pressure"] - p["reference"])**2 for p in particles
            ) / mass
        ),
    )


def sample_frames(path, steps, n):
    profiles = {}
    samples = {}
    masses_by_id = {}
    marked_by_step = {}
    for step, rows in groupby(read_rows(path), key=lambda row: int(row["step"])):
        particles = []
        bins = [[] for _ in range(DEPTH)]
        seen = set()
        marked = set()
        require(step not in samples, f"{path}: repeated frame {step}")
        for row in rows:
            require(int(row["n"]) == n, f"{path}: wrong resolution")
            compare(float(row["time"]), float(steps[step]["time"]), f"{path}: time")
            pid = int(row["id"])
            require(pid not in seen, f"{path}: duplicate particle ID {pid} at step {step}")
            seen.add(pid)
            if int(row["crossed"]):
                marked.add(pid)
            p = {key: float(row[key]) for key in ("x", "y", "z", "mass", "pressure", "speed", "density")}
            require(all(map(math.isfinite, p.values())), f"{path}: nonfinite particle at step {step}")
            require(p["density"] > 0 and p["mass"] > 0, f"{path}: nonpositive density or mass")
            p["reference"] = reference_pressure(p["z"])
            if step == 0:
                density = RHO0 * (
                    1 + (GAMMA - 1) * GRAVITY * (DEPTH - p["z"]) / SOUND_SPEED**2
                )**(1 / (GAMMA - 1))
                compare(p["mass"], density * (DEPTH / n)**3, f"{path}: initial mass")
                masses_by_id[pid] = p["mass"]
            else:
                compare(p["mass"], masses_by_id[pid], f"{path}: mass change for ID {pid}")
            particles.append(p)
            if all(0 <= p[axis] <= DEPTH for axis in ("x", "y", "z")):
                bins[min(int(p["z"]), DEPTH - 1)].append(p)
        require(len(seen) == n**3, f"{path}: incomplete particle frame {step}")
        all_moments = moments(particles)
        pressure_rms = all_moments["pressure_rms_error_pa"]
        speed_rms = math.sqrt(math.fsum(p["mass"] * p["speed"]**2 for p in particles) / all_moments["mass_kg"])
        compare(pressure_rms, float(steps[step]["pressure_rms"]), f"{path}: pressure RMS at step {step}")
        compare(speed_rms, float(steps[step]["speed_rms"]), f"{path}: speed RMS at step {step}")
        compare(max(p["speed"] for p in particles), float(steps[step]["speed_max"]), f"{path}: max speed")
        occupied = [p for layer in bins for p in layer]
        require(len(occupied) == int(steps[step]["in_tank"]), f"{path}: occupancy at step {step}")
        profiles[step] = [moments(layer) for layer in bins]
        samples[step] = moments(occupied)
        marked_by_step[step] = marked
    require(set(samples) >= {0, len(steps) - 1}, f"{path}: initial or final frame missing")
    return profiles, samples, set(masses_by_id), marked_by_step


def summarize_crossings(path, steps, ids, marked_by_step, n):
    events = defaultdict(list)
    face_ids = {face: set() for face in FACES}
    all_ids = set()
    first_time = {face: None for face in FACES}
    event_counts = dict.fromkeys(FACES, 0)
    for row in read_rows(path):
        events[int(row["step"])].append(row)
    for step, step_row in enumerate(steps):
        time = float(step_row["time"])
        dt = float(step_row["dt"])
        step_events = set()
        for row in events.pop(step, ()):
            pid = int(row["id"])
            face = row["face"]
            fraction = float(row["fraction"])
            require(int(row["n"]) == n, f"{path}: wrong resolution")
            compare(float(row["time"]), time, f"{path}: event step time")
            require(pid in ids and 0 < fraction <= 1 and step > 0, f"{path}: invalid event {row}")
            require((pid, face) not in step_events, f"{path}: duplicate event {row}")
            step_events.add((pid, face))
            axis, coordinate = FACE_PLANES[FACES.index(face)]
            compare(float(row[axis]), coordinate, f"{path}: intersection plane")
            require(
                all(0 <= float(row[other]) <= DEPTH for other in ("x", "y", "z") if other != axis),
                f"{path}: event outside finite face {row}",
            )
            crossing_time = time - dt * (1 - fraction)
            first_time[face] = min(crossing_time, first_time[face] or math.inf)
            face_ids[face].add(pid)
            all_ids.add(pid)
            event_counts[face] += 1
        require(len(all_ids) == int(step_row["unique_any"]), f"{path}: union through step {step}")
        require(
            all(len(face_ids[face]) == int(step_row[face]) for face in FACES),
            f"{path}: distinct face totals through step {step}",
        )
        if step in marked_by_step:
            require(marked_by_step[step] == all_ids, f"{path}: crossed IDs in frame at step {step}")
    require(not events, f"{path}: events outside accepted steps")
    faces = [dict(
        face=face, distinct_particle_count=len(face_ids[face]),
        distinct_fraction=len(face_ids[face]) / len(ids), event_count=event_counts[face],
        first_crossing_time_s=first_time[face],
    ) for face in FACES]
    faces.append(dict(
        face="any", distinct_particle_count=len(all_ids), distinct_fraction=len(all_ids) / len(ids),
        event_count=sum(event_counts.values()),
        first_crossing_time_s=min((t for t in first_time.values() if t is not None), default=None),
    ))
    return faces


def read_steps(path, n):
    steps = list(read_rows(path))
    previous_time = 0
    for index, row in enumerate(steps):
        require((int(row["n"]), int(row["step"])) == (n, index), f"{path}: skipped or invalid step {index}")
        time, dt = float(row["time"]), float(row["dt"])
        compare(time - previous_time, dt, f"{path}: dt at step {index}")
        if index:
            require(dt > 0, f"{path}: nonpositive dt at step {index}")
            cap = min(DEPTH / n / 256, 0.9 * float(row["cfl_bound"]))
            require(dt <= cap + 2e-15, f"{path}: cap exceeded at step {index}")
        for key in ("pressure_rms", "speed_rms", "speed_max"):
            value = float(row[key])
            require(math.isfinite(value) and value >= 0, f"{path}: invalid {key} at step {index}")
        previous_time = time
    compare(previous_time, END, f"{path}: final time")
    return steps


def summarize(directory, n):
    steps = read_steps(directory / "steps.csv", n)
    profiles, samples, ids, marked = sample_frames(directory / "frames.csv", steps, n)
    expected_rows = {(step, layer) for step in profiles for layer in range(DEPTH)}
    for row in read_rows(directory / "profiles.csv"):
        step, layer = int(row["step"]), int(row["bin"])
        expected_rows.remove((step, layer))
        require(int(row["n"]) == n, f"{directory}: wrong profile resolution")
        compare(float(row["time"]), float(steps[step]["time"]), f"{directory}: profile time")
        sample = profiles[step][layer]
        require(sample["particle_count"] == int(row["count"]), f"{directory}: layer count mismatch")
        compare(sample["mass_kg"], float(row["mass"]), f"{directory}: layer mass mismatch")
        if sample["particle_count"]:
            compare(sample["mean_pressure_pa"], float(row["mean_pressure"]), f"{directory}: layer pressure")
            compare(sample["mean_reference_pressure_pa"], float(row["mean_reference_pressure"]), f"{directory}: layer reference")
    require(not expected_rows, f"{directory}: missing profiles")
    crossings = summarize_crossings(directory / "crossings.csv", steps, ids, marked, n)
    initial, final = steps[0], steps[-1]
    final_particle_count = int(final["in_tank"])
    summary = dict(
        resolution=n, particle_count=n**3, accepted_steps=int(final["step"]),
        end_time_s=float(final["time"]), initial_pressure_rms_error_pa=float(initial["pressure_rms"]),
        final_pressure_rms_error_pa=float(final["pressure_rms"]),
        final_in_tank_particle_count=final_particle_count,
        final_in_tank_fraction=final_particle_count / n**3,
        final_in_tank_mass_kg=samples[len(steps) - 1]["mass_kg"],
        final_in_tank_pressure_rms_error_pa=samples[len(steps) - 1]["pressure_rms_error_pa"],
        final_speed_rms_m_s=float(final["speed_rms"]),
    )
    for key, label in (
        ("pressure_rms", "pressure_rms_error_pa"), ("speed_rms", "speed_rms_m_s"),
        ("speed_max", "particle_speed_m_s"),
    ):
        peak = max(steps, key=lambda row: float(row[key]))
        summary[f"peak_{label}"] = float(peak[key])
        summary[f"peak_{key}_time_s"] = float(peak["time"])
    profile = [
        dict(resolution=n, z_lower_m=z, z_upper_m=z + 1, **layer)
        for z, layer in enumerate(profiles[len(steps) - 1])
    ]
    faces = [dict(resolution=n, **crossing) for crossing in crossings]
    return summary, profile, faces


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[6, 12, 24])
    args = parser.parse_args()
    summaries, profiles, crossings = [], [], []
    for n in args.resolutions:
        summary, profile, faces = summarize(args.run_dir / f"n{n}", n)
        summaries.append(summary)
        profiles.extend(profile)
        crossings.extend(faces)
    for name, rows in (("summary", summaries), ("profiles", profiles), ("crossings", crossings)):
        write_csv(args.output_dir / f"hydrostatic-trajectory-{name}.csv", rows)


if __name__ == "__main__":
    main()
