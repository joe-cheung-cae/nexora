"""Plot wall-only and continuity hydrostatic tank validation from raw trajectory CSVs."""
import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw

from plot_sph_hydrostatic_trajectories import font


RESOLUTIONS = (6, 12, 24)
COLORS = {6: "#245b9c", 12: "#c05b15", 24: "#008574"}
INK, GRAY, GRID = "#182d39", "#445967", "#dfe6e8"
END_TIME = 2.34765625
SPEED_LIMIT = .01 * math.sqrt(9.81 * 6)


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def label(draw, x, y, text, size=23, anchor="lm", color=INK):
    draw.text((x, y), text, font=font(size), fill=color, anchor=anchor)


def panel(draw, left, top, title, x_label, y_label, x_range, y_range, xticks, yticks):
    right, bottom = left + 637, top + 260

    def point(x, y):
        return (left + (x - x_range[0]) * (right-left) / (x_range[1]-x_range[0]),
                bottom - (y-y_range[0]) * (bottom-top) / (y_range[1]-y_range[0]))

    label(draw, left, top-75, title, 32)
    label(draw, left, top-30, y_label, 21, color=GRAY)
    for value in yticks:
        x, y = point(x_range[0], value)
        draw.line((x, y, right, y), fill=GRID, width=2)
        label(draw, x-12, y, f"{value:g}", 20, "rm", GRAY)
    for value in xticks:
        x, y = point(value, y_range[0])
        draw.line((x, y, x, y+6), fill=GRAY, width=2)
        label(draw, x, y+14, f"{value:g}", 20, "mt", GRAY)
    draw.line((left, top, left, bottom, right, bottom), fill=GRAY, width=2)
    label(draw, (left+right)/2, bottom+53, x_label, 22, "mt", GRAY)
    return point


def dashed(draw, a, b, color):
    length = math.dist(a, b)
    for i in range(0, math.ceil(length), 14):
        start = min(i/length, 1)
        stop = min((i+8)/length, 1)
        draw.line(tuple((a[j] + t*(b[j]-a[j])) for t in (start, stop) for j in (0, 1)),
                  fill=color, width=2)


def marker(draw, point, color, radius=5):
    x, y = point
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="directory containing wall-only/ and continuity/")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] /
                        "validation/sph/hydrostatic-repair-comparison.png")
    args = parser.parse_args()
    im = Image.new("RGB", (1690, 1110), "white")
    draw = ImageDraw.Draw(im)
    label(draw, 88, 54, "Hydrostatic tank trajectories", 46)
    for i, n in enumerate(RESOLUTIONS):
        x = 970 + 212*i
        draw.line((x, 53, x+45, 53), fill=COLORS[n], width=5)
        label(draw, x+56, 53, f"{n}³ fluid", 24)
    draw.line((85, 102, 1608, 102), fill=GRID, width=2)
    for col, (variant, title, ymax, ticks) in enumerate((
        ("wall-only", "Wall-only control", 16, (0, 4, 8, 12, 16)),
        ("continuity", "Continuity-density repair", .30, (0, .05, .10, .15, .20, .25, .30)),
    )):
        point = panel(draw, 105+825*col, 222, title, "Time (s)", "Mass-weighted RMS speed (m/s)",
                      (0, END_TIME), (0, ymax), (0, .5, 1, 1.5, 2), ticks)
        for n in RESOLUTIONS:
            rows = read(args.results / variant / f"n{n}" / "steps.csv")
            if float(rows[-1]["time"]) != END_TIME:
                raise ValueError(f"incomplete {variant} trajectory at n={n}")
            path = [point(float(row["time"]), float(row["speed_rms"])) for row in rows]
            draw.line(path, fill=COLORS[n], width=3)
        dashed(draw, point(0, SPEED_LIMIT), point(END_TIME, SPEED_LIMIT), GRAY)
        if variant == "continuity":
            x, y = point(END_TIME, SPEED_LIMIT)
            label(draw, x-2, y-8, f"Speed limit {SPEED_LIMIT:.4f} m/s", 19, "rb", GRAY)

    height_ticks = (.5, 1.5, 2.5, 3.5, 4.5, 5.5)
    pressure = panel(draw, 105, 709, "Terminal pressure", "Layer midpoint above bottom (m)",
                     "Layer mean pressure (kPa)", (0, 6), (0, 60), height_ticks, (0, 15, 30, 45, 60))
    error = panel(draw, 930, 709, "Terminal pressure difference", "Layer midpoint above bottom (m)",
                  "Layer mean minus reference (kPa)", (0, 6), (-1.6, 0), height_ticks,
                  (-1.6, -1.2, -.8, -.4, 0))
    for n in RESOLUTIONS:
        path = args.results / "continuity" / f"n{n}" / "profiles.csv"
        rows = [r for r in read(path) if float(r["time"]) == END_TIME]
        if [int(r["bin"]) for r in rows] != list(range(6)):
            raise ValueError(f"expected six terminal layers in {path}")
        mean, reference, residual = [], [], []
        for row in rows:
            z = int(row["bin"]) + .5
            p, ref = (float(row[k])/1000 for k in ("mean_pressure", "mean_reference_pressure"))
            mean.append(pressure(z, p))
            reference.append(pressure(z, ref))
            residual.append(error(z, p-ref))
        for series in (mean, residual):
            draw.line(series, fill=COLORS[n], width=3)
            for xy in series:
                marker(draw, xy, COLORS[n])
        for a, b in zip(reference, reference[1:]):
            dashed(draw, a, b, INK)
    label(draw, 108, 1080,
          f"Terminal profiles use the continuity runs at {END_TIME} s. Dashed pressure curves show each run’s layer reference.",
          23, color=GRAY)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    im.save(args.output)


if __name__ == "__main__":
    main()
