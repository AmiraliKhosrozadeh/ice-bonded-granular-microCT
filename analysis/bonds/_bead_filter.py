"""Keep only real beads in the DIC fields.

spam-ddic correlates every LABEL in the bead image, and the segmentation does
not only label beads.  On the Alumina specimens it also labels the PUNCH, the
SUPPORT and the confining wall, and those entered the displacement and strain
fields as if they were beads.

The evidence that they are not beads, measured on the correlation nodes
themselves (scan 1 -> 2):

    Alumina_75_1000_T5   real beads: radius from axis median 140 vox, 2% in the
                         outer 10% of the height.
                         failing labels: median radius 225 vox (p95 260 = the
                         specimen's own outer radius), 63% in the outer 10% of
                         the height.
    Alumina_175_1800_T5  same picture: 139 vs 204 vox, 0% vs 70% at the ends.

So they sit outside the bead column and against the two loading faces: the
platens and the wall.  They are steel, they move rigidly with the punch, and a
12-neighbour strain stencil that includes one reports the bead beside it as
strained when it is not.

Share of correlation nodes affected, first transition:

    Alumina_175_1800_T5   35%      Glass_100_1700_T5_HR   0%
    Alumina_75_1800_T7    29%      Glass_100_1700_T7      0%
    Alumina_100_1800_T5   26%      Glass_75_1700_T5_HR    1%
    Alumina_75_1000_T5    19%      Glass_100_1800_T5      1%
                                   Glass_75_1000_T6       2%

An Alumina-only problem, consistent with everything else about these two
materials, and with the user's own observation that only Alumina shows bead
failure at all.

THE GATE.  A label is kept if it passes the same sphericity/fill/radius gate
already used for the bond analysis (_bead_quality.py, column shape_ok).  Using
one gate everywhere means the strain field and the bond census describe the
same population.  It also removes broken bead FRAGMENTS, which is correct here
for a different reason: a fragment is not a rigid sphere, so its centroid
displacement is not a material displacement.
"""
import os

import numpy as np
import pandas as pd

_CACHE = {}


def sound_labels(out_dir, tag, scan):
    """Set of label ids at `scan` that pass the bead shape gate."""
    key = (out_dir, tag, scan)
    if key in _CACHE:
        return _CACHE[key]
    p = f'{out_dir}/{tag}_bead_quality.csv'
    if not os.path.exists(p):
        _CACHE[key] = None            # no gate available: keep everything
        return None
    q = pd.read_csv(p)
    s = q[q.scan == scan]
    if not len(s):                    # the table holds FIRST and LAST only
        s = q[q.scan == q.scan.min()]
    _CACHE[key] = set(s[s.shape_ok].bead_id.astype(int))
    return _CACHE[key]


def keep_beads(labels, out_dir, tag, scan, what=''):
    """Boolean mask over `labels`: True where the label is a real bead.

    Returns all-True and says so when no gate is available, rather than
    silently dropping everything.
    """
    ok = sound_labels(out_dir, tag, scan)
    if ok is None:
        print(f'  {what}: no bead-quality table for {tag}, keeping all '
              f'{len(labels)} nodes', flush=True)
        return np.ones(len(labels), bool)
    m = np.array([int(L) in ok for L in labels], bool)
    n = int((~m).sum())
    if n:
        print(f'  {what}: dropped {n} of {len(m)} correlation nodes that are '
              f'not beads ({100*n/max(len(m),1):.0f}%) -- platen, wall or '
              f'broken fragment', flush=True)
    return m
