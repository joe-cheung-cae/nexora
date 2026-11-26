"""Compare original, wall-only and continuity-integrated tank trajectories."""
import argparse
import csv
from itertools import groupby, zip_longest
import math
from pathlib import Path

from summarize_hydrostatic_trajectory import (
    DEPTH, GRAVITY, compare, read_rows, require, summarize, target_flags, write_csv,
)


def check_wall_only(original, wall_only):
    """The h=dx shell already had two complete layers. Check every old output field."""
    for name in ("steps.csv", "frames.csv", "profiles.csv", "crossings.csv"):
        for index, (old, wall) in enumerate(zip_longest(read_rows(original/name), read_rows(wall_only/name))):
            require(old is not None and wall is not None, f"{name}: length differs")
            require(all(wall[k] == v for k, v in old.items()), f"{original}/{name}: row {index} changed")


def acceleration_metrics(case_dir):
    steps = list(read_rows(case_dir / "steps.csv"))
    frames = read_rows(case_dir / "frames.csv")
    bins = {}
    for index, particles in groupby(frames, key=lambda r: int(r["step"])):
        mass = squares = 0.
        moments = [[0., 0.] for _ in range(DEPTH)]
        for p in particles:
            x, y, z = (float(p[k]) for k in ("x", "y", "z"))
            m = float(p["mass"])
            ax, ay, az = (float(p[k]) for k in ("ap_x", "ap_y", "ap_z"))
            term = m * (ax*ax + ay*ay + (az-GRAVITY)**2)
            mass += m
            squares += term
            if all(0 <= c <= DEPTH for c in (x, y, z)):
                b = moments[min(int(z), DEPTH-1)]
                b[0] += m
                b[1] += term
        compare(math.sqrt(squares/mass), float(steps[index]["net_acceleration_rms"]), "frame acceleration RMS")
        for layer, (m, square) in enumerate(moments):
            bins[index, layer] = math.sqrt(square/m) if m else None
    for p in read_rows(case_dir/"profiles.csv"):
        value = bins[int(p["step"]), int(p["bin"])]
        if value is not None:
            compare(float(p["net_acceleration_rms"]), value, "profile acceleration RMS")
    final_step = len(steps) - 1
    return (
        [bins[final_step, i] for i in range(DEPTH)],
        dict(initial_net_accel_rms_m_s2=float(steps[0]["net_acceleration_rms"]),
             final_net_accel_rms_m_s2=float(steps[-1]["net_acceleration_rms"]),
             peak_net_accel_rms_m_s2=max(float(s["net_acceleration_rms"]) for s in steps)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("repair", type=Path, help="contains wall-only and continuity run directories")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--require-targets", action="store_true", help="exit 1 if a repaired run misses a physical target")
    args = parser.parse_args()
    summaries, layers, faces = [], [], []
    all_passed = True
    for n in (6, 12, 24):
        cases = [("original", args.original/f"n{n}"),
                 ("wall-only", args.repair/"wall-only"/f"n{n}"),
                 ("continuity", args.repair/"continuity"/f"n{n}")]
        check_wall_only(cases[0][1], cases[1][1])
        for variant, directory in cases:
            summary, profile, crossing = summarize(directory, n)
            # Wall-only recorded extra acceleration components on the identical trajectory.
            acceleration_source = cases[1][1] if variant == "original" else directory
            acceleration_layers, acceleration = acceleration_metrics(acceleration_source)
            summary.update(acceleration)
            flags = target_flags(summary, crossing)
            if variant == "continuity":
                all_passed &= all(flags.values())
            summaries.append(dict(variant=variant, **summary, **flags))
            for row, accel in zip(profile, acceleration_layers, strict=True):
                layers.append(dict(variant=variant, **row, net_accel_rms_m_s2=accel))
            faces.extend(dict(variant=variant, **r) for r in crossing)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary", summaries), ("layers", layers), ("crossings", faces)):
        write_csv(args.output_dir/f"hydrostatic-repair-{name}.csv", rows)
    if args.require_targets and not all_passed:
        raise SystemExit("One or more repaired trajectories exceed the physical targets.")


if __name__ == "__main__":
    main()
