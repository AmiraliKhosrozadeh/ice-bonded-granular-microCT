"""Cumulative deformation referenced to SCAN 1, for every specimen.

Each per-step figure shows only what happened in that step.  Referencing every
step back to the FIRST scan is more informative, because the state at scan N is
the whole load history, not the last increment of it.

The increments cannot be added row by row: spam-ddic labelled 1->2 with the
scan-1 label image and 2->3 with the scan-2 one, so row i is a different
physical bead in each file.  They are chained by tracking instead -- a bead at
X1 with displacement u12 must end up at X1+u12, which is where its scan-2
counterpart sits:

    u_1N = u_12 + u_23(matched) + ... + u_(N-1)N(matched)   at SCAN 1 positions

Strain is re-fitted from the cumulative field over the same 12-neighbour
stencil rather than summed from the increments -- strain is not additive when
the rotations differ, and re-fitting avoids that assumption entirely.

TOLERANCE.  The original was a hard-coded 35-voxel bead radius, correct only
for Glass_75_1700_T5_HR.  Bead size differs between these specimens, so the
radius is measured per specimen from the median nearest-neighbour spacing of
the correlation nodes: in a packed bed touching beads sit ~2R apart, so
R = 0.5 * median_nn and the match tolerance is 0.5 * R.  Anything further is
not a match and is DROPPED rather than guessed, so the bead count falls as the
chain lengthens -- that attrition is reported, never hidden.
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

from _spec import TAG, SCANS, UM, OUT
from _bead_filter import keep_beads
from _strain_measure import (assign_injective, eq_strain, rigid_out,
                             rotation_deg)

RES = f'/home/amirali_wsl/spam-results-bond/{TAG}'
SC = ('/mnt/c/Users/cak7496/AppData/Local/Temp/claude/D--wsl/'
      'a62987fb-a2df-41c2-a6a3-fd119ddec5c2/scratchpad')
K = 12


def load(t):
    p = f'{RES}/{TAG}_ddic_{t}-ddic.tsv'
    if not os.path.exists(p):
        return None
    rows = open(p).read().strip().split('\n')
    h = rows[0].split()
    d = np.array([[float(x) for x in r.split()] for r in rows[1:]])
    c = {n: i for i, n in enumerate(h)}
    ok = d[:, c['returnStatus']] >= 1
    # Real beads only.  spam-ddic correlates every LABEL, and on the Alumina
    # specimens the segmentation also labels the punch, the support and the
    # wall -- see _bead_filter.py for the measurements.  Those are steel moving
    # with the platen; leaving them in reports the beads beside them as
    # strained when they are not.
    lab = d[:, c.get('Label', c.get('NodeNumber', 0))].astype(int)[ok]
    keep = keep_beads(lab, OUT, TAG, int(t[0]), f'scan {t[0]}->{t[1]}')
    sel = np.where(ok)[0][keep]
    return (d[sel][:, [c['Zpos'], c['Ypos'], c['Xpos']]],
            d[sel][:, [c['Zdisp'], c['Ydisp'], c['Xdisp']]])


def bead_radius(X):
    """R from the median nearest-neighbour spacing: touching beads sit ~2R."""
    d, _ = cKDTree(X).query(X, k=2)
    return 0.5 * float(np.median(d[:, 1]))


TR = [f'{a}{b}' for a, b in zip(SCANS[:-1], SCANS[1:])]
steps = [(t, load(t)) for t in TR]
steps = [(t, v) for t, v in steps if v is not None]
if not steps:
    sys.exit(f'{TAG}: no DDIC results in {RES}')
print(f'{TAG}: {len(steps)} increment(s) {[t for t, _ in steps]}', flush=True)

first = SCANS[0]
X1, U1k = steps[0][1]
R = bead_radius(X1)
tol = 0.5 * R
print(f'  bead radius from node spacing: {R:.1f} vox   '
      f'match tolerance {tol:.1f} vox', flush=True)

out = {'X': X1, 'R_bead': R}
# the first increment IS already referenced to scan 1
out[f'U_1to{steps[0][0][1]}'] = U1k
keep = np.ones(len(X1), bool)
cum = U1k.copy()
made = [f'1to{steps[0][0][1]}']

for t, (Xk, Uk) in steps[1:]:
    to = t[1]
    kidx = np.where(keep)[0]
    pred = X1[keep] + cum[keep]
    # ONE-TO-ONE.  An unconstrained nearest-neighbour query let several scan-1
    # beads claim the SAME scan-k bead and inherit a bitwise-identical
    # displacement -- 350 matches onto only 254 distinct targets on
    # Glass_100_1700_T7 3->4, 340 onto 267 on Glass_75_1700_T5_HR 2->3.  A bead
    # cannot be in two places, and the printed "98% matched" was counting beads
    # that found *a* bead rather than their own.  The assignment is now
    # injective, closest pair first, and a bead with no free target inside the
    # tolerance is DROPPED rather than handed someone else's motion.
    S, T = assign_injective(pred, Xk, tol)
    n_before = len(kidx)
    add = np.zeros_like(cum)
    add[kidx[S]] = Uk[T]
    lost = np.setdiff1d(np.arange(n_before), S)
    keep[kidx[lost]] = False
    cum = cum + add
    gap = np.linalg.norm(pred[S] - Xk[T], axis=1)
    print(f'  chain to scan {to}: matched {len(S)}/{n_before} '
          f'({100*len(S)/max(n_before, 1):.0f}%, one-to-one)  '
          f'median gap {np.median(gap):.1f} vox   '
          f'carried {int(keep.sum())} beads', flush=True)
    out[f'U_1to{to}'] = cum.copy()
    out[f'keep_1to{to}'] = keep.copy()
    made.append(f'1to{to}')

# REPORT LIKE FOR LIKE.  The first row used to be evaluated over every bead
# while the chained rows used only the survivors, so successive lines described
# different POPULATIONS and the progression was not a progression.  All rows are
# now reported on the common set that survives the whole chain.
#
# The magnitude column is |u - u_rigid|, which is what the figure plots.  Raw
# |u| includes the bulk translation of the specimen under the punch: it made
# Glass_100_1700_T7 read 2.05 / 1.84 / 2.07 mm -- non-monotone, entirely from
# rigid-body motion, and nothing to do with deformation.
common = keep.copy()
out['keep_common'] = common
Xc = X1[common]
print(f'\ncommon set carried through the whole chain: {int(common.sum())} beads',
      flush=True)
hdr = ('field'.ljust(14) + 'beads'.rjust(7) + '|u-rigid| mm'.rjust(14)
       + 'eq med'.rjust(9) + 'eq p95'.rjust(9) + 'rot med'.rjust(10))
print(hdr, flush=True)
for m in made:
    to = int(m.split('to')[1])
    U = out[f'U_1to{to}'][common]
    e = eq_strain(Xc, U)                     # Green-Lagrange
    out[f'eq_1to{to}'] = e
    mm = np.linalg.norm(rigid_out(Xc, U), axis=1) * UM / 1000.0
    rot = rotation_deg(Xc, U)
    print(f'{m:<14s}{int(common.sum()):7d}{np.median(mm):14.2f}'
          f'{np.median(e):9.3f}{np.percentile(e, 95):9.3f}'
          f'{np.median(rot):9.1f} deg', flush=True)
print('\nstrain is GREEN-LAGRANGE 0.5(F^T F - I).  The median local rotation is\nprinted beside it because the infinitesimal 0.5(G+G^T) used before is a\nstrain measure only while that angle stays small -- here it reaches 20 deg\nat the median and 66 deg at the 95th percentile, where a bead that merely\nROTATES registers a spurious strain.', flush=True)

np.savez(f'{SC}/cumulative_{TAG}.npz', **out)
print(f'\n-> {SC}/cumulative_{TAG}.npz   fields {made}', flush=True)
