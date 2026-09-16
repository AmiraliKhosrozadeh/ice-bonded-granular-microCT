"""Grain tracking for S2 2->3 with a seed that carries the radial flow.

The affine seed of _sand_track_affine.py stretches the column axially by the
measured shortening and nothing else.  On S2 2->3 the head also spreads,
from about 5.1 to 6.5 mm radius over the top 8 mm, and a grain that moved
1.4 mm outward lands well outside the half-spacing search tolerance of a seed
that predicts no radial motion.  The local predictor cannot recover it
either, because it is rebuilt from matches and there are none where the flow
is.  So the reciprocal set was silent about the flow: 1899 tracked grains in
the top 8 mm with a median radial motion of +6 um.

Here the seed is built from the two column outlines, which stage 1 measured
without any correlation.  A grain at height z and radius r in scan A is
predicted at the height the axial stretch gives and at the radius
r * R_B(z_B) / R_A(z_A), with R the equivalent radius of the column section
at that height.  Everything after the seed is unchanged: the same match, the
same tolerance and margin, the same volume test, the same local refinement,
and the same reciprocity test, so the acceptance is no looser than before.

Output: dvc_S2_23/grain_track_flow.npz, same layout as grain_track_recip.npz.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

SC = ('/mnt/c/Users/cak7496/AppData/Local/Temp/claude/D--wsl/'
      'a62987fb-a2df-41c2-a6a3-fd119ddec5c2/scratchpad')
sys.path.insert(0, SC)
from _sand_grain_track import PAIRS, centroids
from _sand_track_affine import _match, K_LOCAL, ROUNDS

OUT = f'{SC}/sand_out'


def outline(com, nb=40):
    """Column axis and outer radius against z, from the grain cloud itself.

    The geometry files carry an axis fit in another frame, and using it put
    a 100-voxel lateral offset into the seed that no grain could survive.
    Taking axis and radius from the centroids keeps everything in one frame:
    per z-band the axis is the median grain position and the radius the 98th
    percentile of the grain distance from it.
    """
    z = com[:, 0]
    edges = np.linspace(z.min(), z.max() + 1e-6, nb + 1)
    zc, cy, cx, rr = [], [], [], []
    for k in range(nb):
        m = (z >= edges[k]) & (z < edges[k + 1])
        if m.sum() < 50:
            continue
        y0, x0 = np.median(com[m, 1]), np.median(com[m, 2])
        r = np.hypot(com[m, 1] - y0, com[m, 2] - x0)
        zc.append(0.5 * (edges[k] + edges[k + 1]))
        cy.append(y0); cx.append(x0); rr.append(np.percentile(r, 98))
    return np.array(zc), np.array(cy), np.array(cx), np.array(rr)


def seed(coma, ga, gb, oa, ob):
    """Predicted scan-B position of every scan-A grain."""
    Fzz = float(gb['z1'] - gb['z0']) / float(ga['z1'] - ga['z0'])
    pred = coma.copy()
    pred[:, 0] = gb['z0'] + (coma[:, 0] - ga['z0']) * Fzz
    za, ya, xa, ra = oa
    zb, yb, xb, rb = ob
    ay, ax = np.interp(coma[:, 0], za, ya), np.interp(coma[:, 0], za, xa)
    by, bx = np.interp(pred[:, 0], zb, yb), np.interp(pred[:, 0], zb, xb)
    f = np.interp(pred[:, 0], zb, rb) / np.maximum(np.interp(coma[:, 0], za, ra), 1.0)
    pred[:, 1] = by + (coma[:, 1] - ay) * f
    pred[:, 2] = bx + (coma[:, 2] - ax) * f
    return pred, Fzz, f


def direction(ta, da, tb, db):
    ida, coma, vola, ga = centroids(ta, da)
    idb, comb, volb, gb = centroids(tb, db)
    oa, ob = outline(coma), outline(comb)
    spacing = float(np.median(cKDTree(coma).query(coma, k=2)[0][:, 1]))
    tol = 0.5 * spacing
    tree_b = cKDTree(comb)
    pred, Fzz, f = seed(coma, ga, gb, oa, ob)
    print(f'     axis shift A->B at mid-height: dy {np.median(ob[1])-np.median(oa[1]):+.1f} '
          f'dx {np.median(ob[2])-np.median(oa[2]):+.1f} vox', flush=True)
    print(f'  {ta}->{tb}: {len(ida)} grains, Fzz {Fzz:.4f}, radial factor '
          f'{np.percentile(f, 5):.3f}..{np.percentile(f, 95):.3f}', flush=True)
    keep = jj = None
    for r in range(ROUNDS):
        keep, jj, best = _match(coma, pred, comb, volb, vola, tol, tree_b)
        print(f'     round {r+1}: {int(keep.sum())} matched, residual median '
              f'{np.median(best[keep]):.2f} vox', flush=True)
        if keep.sum() < 200:
            break
        ax_, au = coma[keep], comb[jj[keep, 0]] - coma[keep]
        nn = cKDTree(ax_).query(coma, k=min(K_LOCAL, len(ax_)))[1]
        if nn.ndim == 1:
            nn = nn[:, None]
        pred = coma + np.median(au[nn], axis=1)
    return dict(ida=ida, idb=idb, coma=coma, comb=comb, vola=vola,
                keep=keep, jj=jj, spacing=spacing)


def run(pair):
    ta, da, tb, db = PAIRS[pair]
    f = direction(ta, da, tb, db)
    r = direction(tb, db, ta, da)
    fa = f['ida'][f['keep']]
    fj = f['jj'][f['keep'], 0]
    fb = f['idb'][fj]
    rev = dict(zip(r['ida'][r['keep']].tolist(),
                   r['idb'][r['jj'][r['keep'], 0]].tolist()))
    rec = np.array([rev.get(int(b), -1) == int(a) for a, b in zip(fa, fb)])
    sel = np.flatnonzero(f['keep'])[rec]
    d = f'{OUT}/dvc_{pair}'
    np.savez(f'{d}/grain_track_flow.npz',
             com_a=f['coma'][sel], com_b=f['comb'][fj[rec]],
             vol_a=f['vola'][sel], id_a=fa[rec], id_b=fb[rec],
             spacing=f['spacing'])
    print(f'{pair}: {int(rec.sum())} reciprocal of {len(fa)} forward '
          f'({100*rec.mean():.1f}%)  -> grain_track_flow.npz', flush=True)


if __name__ == '__main__':
    for p in (sys.argv[1:] or ['S2_23']):
        run(p)
