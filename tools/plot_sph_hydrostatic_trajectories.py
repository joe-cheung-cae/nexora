"""Render matched physical-mass hydrostatic trajectories using Pillow.

python3 tools/plot_sph_hydrostatic_trajectories.py results/wcsph-trajectory-resolution
"""
import argparse
import csv
import math
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

NS = (6, 12, 24)
TIMES = (0.0, 0.38671875, 1.5625, 2.34765625)
COLORS = ('#0072b2', '#d55e00', '#009e73')
FONT_CANDIDATES = (
    '/System/Library/Fonts/Helvetica.ttc',
    'DejaVuSans.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
)

@lru_cache(maxsize=None)
def font(size):
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow versions before scalable default fonts.
        return ImageFont.load_default()

def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))

def at(rows, t):
    result = [r for r in rows if abs(float(r['time']) - t) < 1e-09]
    if not result:
        raise ValueError(f'missing time {t}')
    return result

def label(draw, xy, text, size=21, fill='#202831', anchor=None):
    draw.text(xy, text, font=font(size), fill=fill, anchor=anchor)

def color(p, limit):
    v = max(-1.0, min(1.0, p / limit))
    a, b = ((48, 100, 180), (247, 247, 247)) if v < 0 else ((247, 247, 247), (180, 38, 45))
    w = v + 1 if v < 0 else v
    return tuple((round(x * (1 - w) + y * w) for x, y in zip(a, b)))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    data = {n: {k: read(args.results / f'n{n}' / f'{k}.csv') for k in ('frames', 'profiles', 'steps')} for n in NS}
    frames, selected = ({}, {})
    for n in NS:
        selected[n] = {r['id'] for r in at(data[n]['frames'], 0) if abs(float(r['y']) - 3) <= 3 / n + 1e-10}
        assert len(selected[n]) == 2 * n * n
        for t in TIMES:
            rows = at(data[n]['frames'], t)
            assert len(rows) == n ** 3 and len({r['id'] for r in rows}) == n ** 3
            frames[n, t] = rows
            assert sum((int(r['crossed']) for r in rows)) == int(at(data[n]['steps'], t)[0]['unique_any'])
    # A common clipped kPa scale keeps the hydrostatic range legible.
    limit = 60.0
    lo, hi = (-1.5, 7.5)
    im = Image.new('RGB', (2200, 1730), 'white')
    d = ImageDraw.Draw(im)
    label(d, (1100, 26), 'Hydrostatic tank: matched trajectory frames', 36, anchor='mt')
    for i, n in enumerate(NS):
        for j, t in enumerate(TIMES):
            left, top, side = (125 + j * 500, 115 + i * 485, 400)
            xy = lambda x, z: (left + (x - lo) / (hi - lo) * side, top + side - (z - lo) / (hi - lo) * side)
            a, b = (xy(0, 6), xy(6, 0))
            d.line((a[0], a[1], a[0], b[1], b[0], b[1], b[0], a[1]), fill='#404950', width=3)
            for q in range(0, 12, 2):
                d.line((a[0] + (b[0] - a[0]) * q / 12, a[1], a[0] + (b[0] - a[0]) * (q + 1) / 12, a[1]), fill='#8d969c', width=2)
            rows = frames[n, t]
            subset = [r for r in rows if r['id'] in selected[n]]
            visible = [r for r in subset if lo <= float(r['x']) <= hi and lo <= float(r['z']) <= hi]
            for r in visible:
                x, z = xy(float(r['x']), float(r['z']))
                radius = 6 if n == 6 else 4 if n == 12 else 2
                d.ellipse((x - radius, z - radius, x + radius, z + radius), fill=color(float(r['pressure']) / 1000, limit))
            for r in visible:
                if int(r['crossed']):
                    x, z = xy(float(r['x']), float(r['z']))
                    d.line((x - 4, z - 4, x + 4, z + 4), fill='#111111', width=2)
                    d.line((x - 4, z + 4, x + 4, z - 4), fill='#111111', width=2)
            d.rectangle((left, top, left + side, top + side), outline='#9aa2a8', width=1)
            count = sum((int(r['crossed']) for r in rows))
            d.rectangle((left + 3, top + 3, left + 225, top + 55), fill='white')
            label(d, (left + 8, top + 7), f'crossed: {count}/{n ** 3}', 19)
            label(d, (left + 8, top + 31), f'selected off-view: {len(subset) - len(visible)}', 18)
            if i == 0:
                label(d, (left + side / 2, top - 38), f't = {t:.8f} s'.replace('00000000', '0').replace('50000000', '5'), 25, anchor='mt')
            if j == 0:
                label(d, (left - 19, top + side / 2), f'{n}³   z (m)', 23, anchor='rm')
            if i == 2:
                label(d, (left + side / 2, top + side + 12), 'x (m)', 21, anchor='mt')
            for v in (0, 3, 6):
                x, _ = xy(v, 0)
                _, y = xy(0, v)
                if i == 2:
                    label(d, (x, top + side - 4), str(v), 17, anchor='mb')
                if j == 0:
                    label(d, (left + 4, y), str(v), 17, anchor='lm')
    barleft, bartop, barwidth = (410, 1580, 1200)
    for x in range(barwidth):
        d.line((barleft + x, bartop, barleft + x, bartop + 22), fill=color((2 * x / barwidth - 1) * limit, limit))
    for x, txt in ((barleft, f'{-limit:.0f}'), (barleft + barwidth // 2, '0'), (barleft + barwidth, f'{limit:.0f}')):
        label(d, (x, bartop + 28), txt, 19, anchor='mt')
    label(d, (1020, 1538), 'Particle gauge pressure (kPa); shared scale clipped at ±60', 22, anchor='mt')
    label(d, (1100, 1660), 'x–z projection of two initial lattice planes nearest y = 3 m; × marks selected particles that have crossed a wall.', 20, anchor='mt')
    label(d, (1100, 1690), 'Crossing counts include all particles and are cumulative. Dashed top denotes the open surface.', 20, anchor='mt')
    im.save('validation/sph/hydrostatic-resolution-keyframes.png')
    # Show the complete endpoint range and a detail of the finer cases.
    t = TIMES[-1]
    profiles = {}
    for n in NS:
        rows = sorted(at(data[n]['profiles'], t), key=lambda r: int(r['bin']))
        assert len(rows) == 6
        profiles[n] = [r for r in rows if int(r['count'])]
    values = [float(r[k]) / 1000 for rows in profiles.values() for r in rows for k in ('mean_pressure', 'mean_reference_pressure')]
    pmin = math.floor(min(values) / 250) * 250
    pmax = math.ceil(max(values) / 250) * 250
    im = Image.new('RGB', (2200, 1110), 'white')
    d = ImageDraw.Draw(im)
    label(d, (1100, 28), 'Final mass-weighted pressure profiles', 38, anchor='mt')
    label(d, (1100, 78), f't = {t:.8f} s  |  six common 1 m vertical bins', 25, anchor='mt')
    for j, (ns, xmin, xmax, ticks) in enumerate(((NS, pmin, pmax, list(range(int(pmin), int(pmax) + 1, 250))), ((12, 24), 0, 190, [0, 25, 50, 75, 100, 125, 150, 175]))):
        left, top, w, h = (140 + j * 1070, 195, 900, 650)
        xy = lambda p, z: (left + (p - xmin) / (xmax - xmin) * w, top + h - z / 6 * h)
        d.rectangle((left, top, left + w, top + h), outline='#7a858c', width=2)
        for z in range(7):
            _, y = xy(0, z)
            d.line((left, y, left + w, y), fill='#e2e6e8')
            label(d, (left - 12, y), str(z), 21, anchor='rm')
        for p in ticks:
            x, _ = xy(p, 0)
            d.line((x, top, x, top + h), fill='#edf0f1')
            label(d, (x, top + h + 10), str(p), 20, anchor='mt')
        if xmin <= 0 <= xmax:
            x0, _ = xy(0, 0)
            d.line((x0, top, x0, top + h), fill='#a5adb1', width=2)
        for n in ns:
            c = COLORS[NS.index(n)]
            for key, dashed in (('mean_pressure', False), ('mean_reference_pressure', True)):
                points = [xy(float(r[key]) / 1000, int(r['bin']) + 0.5) for r in profiles[n]]
                if len(points) > 1:
                    for a, b in zip(points, points[1:]):
                        if dashed:
                            for q in range(0, 20, 2):
                                d.line((a[0] + (b[0] - a[0]) * q / 20, a[1] + (b[1] - a[1]) * q / 20, a[0] + (b[0] - a[0]) * (q + 1) / 20, a[1] + (b[1] - a[1]) * (q + 1) / 20), fill=c, width=3)
                        else:
                            d.line((*a, *b), fill=c, width=4)
                for x, y in points:
                    if dashed:
                        d.line((x - 7, y - 7, x + 7, y + 7), fill=c, width=3)
                        d.line((x - 7, y + 7, x + 7, y - 7), fill=c, width=3)
                    else:
                        d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=c)
        label(d, (left + w / 2, top - 42), 'All resolutions: full endpoint range' if j == 0 else 'Finer resolutions: 0–190 kPa', 26, anchor='mt')
        label(d, (left + w / 2, top + h + 46), 'Gauge pressure (kPa)', 23, anchor='mt')
        label(d, (left - 70, top + h / 2), 'z (m)', 22, anchor='mm')
    for i, (n, c) in enumerate(zip(NS, COLORS)):
        x = 400 + i * 500
        d.line((x, 947, x + 60, 947), fill=c, width=4)
        d.ellipse((x + 24, 941, x + 36, 953), fill=c)
        label(d, (x + 72, 934), f'{n}³ measured', 23)
    label(d, (1100, 1005), 'Dashed line and ×: analytic pressure evaluated at the same particles, in the matching resolution color.', 22, anchor='mt')
    label(d, (1100, 1043), 'Bins include particles currently inside the 6 m tank; measured and analytic averages are mass weighted.', 22, anchor='mt')
    im.save('validation/sph/hydrostatic-resolution-profiles.png')
if __name__ == '__main__':
    main()
