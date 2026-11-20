"""Reconstruct tank pressure accelerations per ID from Release double trajectory CSVs.

Usage: python hydrostatic_force_balance.py RAW_CONTINUITY_DIRECTORY OUTPUT.csv
Requires NumPy. Regions use instantaneous distances <2h and are disjoint, apart
from the all-row summary. All means and RMS values use physical fluid mass.
The hydrostatic substitution evaluates analytic rho,p at the same observed
fluid coordinates, retaining masses and the same Adami wall reconstruction.
The analytic-wall substitution also sets wall pressure directly from its height.
Both substitutions are instantaneous force evaluations; the positions stay fixed.
"""
import argparse
import csv
import itertools
from pathlib import Path

import numpy as np


HEIGHT, GRAVITY, RHO0, C0, GAMMA = 6., 9.81, 1000., 80., 7.
BULK = RHO0 * C0**2 / GAMMA
TIMES = (0., 1.5625, 2.34765625)


def hydrostatic(z):
    log_ratio = np.log1p((GAMMA - 1) * GRAVITY * (HEIGHT - z) / C0**2) / (GAMMA - 1)
    return RHO0 * np.exp(log_ratio), BULK * np.expm1(GAMMA * log_ratio)


def wall_positions(n, h):
    lattice = np.array(list(itertools.product(range(-2, n + 2), range(-2, n + 2),
                                              range(-2, n))))
    wall = np.any((lattice < 0) | (lattice >= n), axis=1)
    return (lattice[wall] + .5) * h


def neighbor_pairs(x, h, fluid_count):
    # Independent fixed-width buckets. Edges are directed, with a fluid receiver.
    cells = {}
    for slot, key in enumerate(np.floor(x / (2*h)).astype(int)):
        cells.setdefault(tuple(key), []).append(slot)
    pairs_i, pairs_j = [], []
    offsets = list(itertools.product((-1, 0, 1), repeat=3))
    for cell, slots in cells.items():
        owned = np.array([i for i in slots if i < fluid_count], dtype=int)
        if not owned.size:
            continue
        adjacent = []
        for dx, dy, dz in offsets:
            adjacent.extend(cells.get((cell[0]+dx, cell[1]+dy, cell[2]+dz), ()))
        source = np.array(adjacent)
        delta = x[owned, None, :] - x[source]
        rr = np.einsum('ijk,ijk->ij', delta, delta)
        ii, jj = np.where((rr > 0) & (rr < (2*h)**2))
        pairs_i.append(owned[ii])
        pairs_j.append(source[jj])
    return np.concatenate(pairs_i), np.concatenate(pairs_j)


def reconstruct(x, mass, rho, pressure, i, j, h, analytic_wall=False):
    nf = len(rho)
    d = x[i] - x[j]
    q = np.linalg.norm(d, axis=1) / h
    wall = j >= nf
    iw, jw, qw = i[wall], j[wall]-nf, q[wall]
    w = np.where(qw < 1, 1 - 1.5*qw**2 + .75*qw**3, .25*(2-qw)**3)
    nw = len(x) - nf
    wsum = np.bincount(jw, weights=w, minlength=nw)
    numerator = np.bincount(jw, weights=w*(pressure[iw] + rho[iw]*GRAVITY*d[wall, 2]),
                            minlength=nw)
    pw = np.divide(numerator, wsum, where=wsum > 0, out=np.zeros(nw))
    if analytic_wall:
        pw = hydrostatic(x[nf:, 2])[1]
    rhow = RHO0 * np.exp(np.log1p(pw/BULK)/GAMMA)
    coeff = np.r_[pressure/rho**2, pw/rhow**2]
    grad = np.where(q < 1, 3 - 2.25*q, .75*(2-q)**2/q) / (np.pi*h**5)
    pair = (mass[j]*(coeff[i]+coeff[j])*grad)[:, None] * d

    def accumulate(mask):
        return np.column_stack([np.bincount(i[mask], weights=pair[mask, k], minlength=nf)
                                for k in range(3)])

    return accumulate(~wall), accumulate(wall)


def summarize(n, time, mode, mass, x, ff, fw, closure):
    h = HEIGHT/n
    residual = ff + fw + [0, 0, -GRAVITY]
    near = np.minimum(x, HEIGHT-x) < 2*h
    side = near[:, 0] | near[:, 1]
    free, bottom = HEIGHT-x[:, 2] < 2*h, x[:, 2] < 2*h
    corner = near[:, 0] & near[:, 1]
    regions = dict(all=np.ones(len(mass), dtype=bool), free_surface=free,
                   bottom_corner=~free & bottom & corner,
                   bottom_other=~free & bottom & ~corner,
                   side_wall=~free & ~bottom & side, interior=~free & ~bottom & ~side)
    for name, mask in regions.items():
        if not mask.any():
            continue
        def mean(values):
            return np.average(values[mask], weights=mass[mask])

        def rms(vector):
            return np.sqrt(mean(np.sum(vector**2, axis=1)))

        yield dict(n=n, time=time, state=mode, region=name, count=int(mask.sum()),
                   mean_ff_z=mean(ff[:, 2]), mean_fw_z=mean(fw[:, 2]), gravity_z=-GRAVITY,
                   mean_net_z=mean(residual[:, 2]), rms_ff=rms(ff), rms_fw=rms(fw),
                   rms_net_xy=rms(residual[:, :2]), rms_net_z=rms(residual[:, 2:]),
                   rms_net=rms(residual),
                   max_ap_closure_abs=(np.max(closure[mask]) if closure is not None else ''))


def analyze(source):
    for n in (6, 12, 24):
        h = HEIGHT/n
        frames = np.genfromtxt(source/f'n{n}'/'frames.csv', names=True, delimiter=',')
        for time in TIMES:
            rows = frames[frames['time'] == time]
            rows.sort(order='id')
            if not np.array_equal(rows['id'], np.arange(1, n**3+1)):
                raise ValueError(f'n={n} t={time} missing or duplicate particle IDs')
            fluid = np.column_stack([rows[k] for k in 'xyz'])
            walls = wall_positions(n, h)
            x = np.vstack((fluid, walls))
            mass = np.r_[rows['mass'], hydrostatic(walls[:, 2])[0]*h**3]
            i, j = neighbor_pairs(x, h, len(fluid))
            ap = np.column_stack([rows['ap_'+k] for k in 'xyz'])
            analytic = hydrostatic(fluid[:, 2])
            for mode, fields in [('observed', (rows['density'], rows['pressure'])),
                                 ('hydrostatic_substitution', analytic),
                                 ('hydrostatic_analytic_wall', analytic)]:
                ff, fw = reconstruct(x, mass, *fields, i, j, h,
                                     analytic_wall=mode == 'hydrostatic_analytic_wall')
                closure = None
                if mode == 'observed':
                    np.testing.assert_allclose(ff+fw, ap, atol=1e-10, rtol=1e-12)
                    closure = np.max(np.abs(ff+fw-ap), axis=1)
                yield from summarize(n, time, mode, rows['mass'], fluid, ff, fw, closure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    rows = list(analyze(args.source))
    with args.output.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
