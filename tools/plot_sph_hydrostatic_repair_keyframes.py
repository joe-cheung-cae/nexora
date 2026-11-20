"""Render repaired tank particles at the four historical trajectory keyframes."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from plot_sph_hydrostatic_trajectories import NS, TIMES, at, color, label, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="directory containing continuity/n{6,12,24}")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] /
                        "validation/sph/hydrostatic-repair-keyframes.png")
    args = parser.parse_args()
    image = Image.new("RGB", (2000, 1640), "white")
    draw = ImageDraw.Draw(image)
    ink, muted = "#20323c", "#526671"
    label(draw, (95, 38), "Hydrostatic tank with continuity density", 44, ink)
    label(draw, (95, 108), "The same particle IDs from the two initial midplane layers, projected onto x–z",
          26, muted)
    # Match the existing baseline view, kernel-pressure palette, and initial midplane selection.
    low, high, limit = -1.5, 7.5, 60
    side = 350
    for column, time in enumerate(TIMES):
        left = 150 + 463 * column
        label(draw, (left+side/2, 194), f"t = {time:g} s" if time == 0 else f"t = {time} s",
              27, ink, "mm")
    for row, n in enumerate(NS):
        directory = args.results / "continuity" / f"n{n}"
        frames = read(directory/"frames.csv")
        steps = read(directory/"steps.csv")
        initial = at(frames, 0)
        selected = {int(p["id"]) for p in initial if abs(float(p["y"])-3) <= 3/n + 1e-10}
        if len(selected) != 2*n*n:
            raise ValueError(f"unexpected slice size at n={n}")
        top = 232 + 412 * row
        label(draw, (88, top+side/2), f"{n}³", 34, ink, "mm")
        for column, time in enumerate(TIMES):
            left = 150 + 463*column
            def xy(x, z):
                return left+side*(x-low)/(high-low), top+side*(high-z)/(high-low)

            particles = at(frames, time)
            if len({int(p["id"]) for p in particles}) != n**3 or len(particles) != n**3:
                raise ValueError(f"incomplete frame n={n}, t={time}")
            crossed = sum(int(p["crossed"]) for p in particles)
            if crossed != int(at(steps, time)[0]["unique_any"]):
                raise ValueError(f"inconsistent crossing count n={n}, t={time}")
            draw.rectangle((left, top, left+side, top+side), outline="#c4cfd2", width=1)
            draw.line([xy(0,6), xy(0,0), xy(6,0), xy(6,6)], fill=ink, width=3)
            for q in range(0, 18, 2):
                draw.line([xy(q/3,6), xy((q+1)/3,6)], fill="#a4afb4", width=2)
            offview = 0
            for p in particles:
                if int(p["id"]) not in selected:
                    continue
                x, z = float(p["x"]), float(p["z"])
                if not (low <= x <= high and low <= z <= high):
                    offview += 1
                    continue
                px, py = xy(x, z)
                radius = {6:7, 12:4, 24:3}[n]
                draw.ellipse((px-radius, py-radius, px+radius, py+radius),
                             fill=color(float(p["pressure"])/1000, limit), outline="#829097")
                if int(p["crossed"]):
                    draw.line((px-4, py-4, px+4, py+4), fill=ink, width=2)
                    draw.line((px-4, py+4, px+4, py-4), fill=ink, width=2)
            label(draw, (left+side/2, top+side+20), f"{crossed}/{n**3:,} crossed  ·  {offview} off-view",
                  21, muted, "mm")
    y, bar_left, bar_width = 1497, 520, 745
    label(draw, (95, y+13), "Gauge pressure (kPa)", 27, ink, "lm")
    for dx in range(bar_width):
        draw.line((bar_left+dx, y, bar_left+dx, y+22),
                  fill=color(limit*(2*dx/(bar_width-1)-1), limit))
    for val in (-60,-30,0,30,60):
        x = bar_left+(val+limit)*bar_width/(2*limit)
        label(draw, (x, y+32), str(val), 21, muted, "mt")
    label(draw, (1330, y+10), "Color clips at ±60 kPa.", 24, muted, "lm")
    label(draw, (95, 1604),
          "Tank walls span 6 m. The dashed line marks the open top. Crossings count all fluid IDs. Off-view counts refer to the tracked slice.",
          24, muted, "lm")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)


if __name__ == "__main__":
    main()
