"""Is the breakage organised into a plane, at ANY clustering scale?

One eps cannot answer this.  Set it from the grain size and it was 0.42 mm
against broken bonds 1.7 mm apart, so DBSCAN linked nothing and the test could
never return a positive.  Set it from the neighbour distance of the broken set
and it linked everything into a single blob, so the test could never return a
negative either.  Both are the same mistake: a scale chosen once, at which the
answer is decided by the choice rather than by the data.

So eps is swept, and at every value the observed clustering is compared against
a null built the same way -- the same number of INTACT bonds, drawn at random
from the same specimen, clustered at the same eps.  The null therefore has the
same point count, the same volume and the same density as the observed set, and
differs from it only in whether the points are the ones that broke.

At each eps the p-value is the fraction of null resamples whose best cluster is
at least as large AND at least as planar as the observed one.  Sweeping eps
means many tests, so the p-values are corrected with Benjamini-Hochberg across
the sweep, exactly as in the bead planar-cracking analysis.

A crack plane has to survive that.  If none does, the honest statement is that
breakage happened -- which the raw rates already show -- but that it is not
organised on a surface, and the "diagonal or horizontal or vertical" question
has no answer to give for these specimens.
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

SC = ('/mnt/c/Users/cak7496/AppData/Local/Temp/claude/D--wsl/'
      'a62987fb-a2df-41c2-a6a3-fd119ddec5c2/scratchpad')
sys.path.insert(0, SC)
from _sand_crack2 import bonds, planes, word, MIN_CLUSTER

OUT = f'{SC}/sand_out'
N_NULL = 60
FACTORS = (0.35, 0.5, 0.7, 0.9, 1.1, 1.3, 1.6, 2.0)
N_BAND = 10        # height bands the null is stratified over


def stratified(mid, broke, intact, rng, nband=N_BAND):
    """Intact bonds drawn with the SAME height distribution as the broken ones.

    A null of uniformly random intact bonds is the wrong null here, and the
    unloaded control proves it: 183 of its bonds pass the threshold purely as
    false positives, yet they beat that null at three of seven scales.  They do
    so because they are not spatially uniform -- tracking is worst near the free
    ends, so the false positives pile up there, and a pile is a cluster.

    Breakage under load piles up near the loaded end too, for a real reason.  A
    null that ignores height therefore rewards BOTH, and cannot tell them
    apart: it would report a horizontal crack plane for any specimen whose
    breakage is concentrated at one end, which is every specimen here.

    Stratifying by height removes exactly that.  What survives is organisation
    WITHIN the height distribution -- a surface -- rather than the height
    distribution itself, which the axial profile already reports and reports
    better.
    """
    z = mid[:, 0]
    edges = np.linspace(z.min(), z.max() + 1e-9, nband + 1)
    bi = np.clip(np.digitize(z, edges) - 1, 0, nband - 1)
    want = np.bincount(bi[broke], minlength=nband)
    pool = [intact[bi[intact] == k] for k in range(nband)]
    out = []
    for k in range(nband):
        if want[k] == 0 or not len(pool[k]):
            continue
        take = min(want[k], len(pool[k]))
        out.append(rng.choice(pool[k], take, replace=False))
    return np.concatenate(out) if out else np.array([], int)


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(o[::-1]):
        k = m - rank
        prev = min(prev, p[i] * m / k)
        adj[i] = prev
    return adj


def sweep(pair, thr, rng):
    c = bonds(pair)
    if c is None:
        return None
    broke = c['sep'] > thr
    nb = int(broke.sum())
    if nb < MIN_CLUSTER:
        print(f'\n{pair}: {nb} broken bonds, too few to cluster')
        return None
    B = c['mid'][broke]
    intact = np.flatnonzero(~broke)
    knn = float(np.median(cKDTree(B).query(B, k=6)[0][:, 5]))

    print(f'\n{pair}   {nb} broken of {len(broke)} bonds '
          f'({100*nb/len(broke):.2f}%)   median 5-NN {knn:.2f} mm')
    print(f'   {"eps mm":>8s}{"clust":>7s}{"planar":>8s}{"dip":>6s}'
          f'{"null n95":>10s}{"null pl95":>11s}{"p":>8s}   orientation')
    rows, pvals = [], []
    for fk in FACTORS:
        eps = fk * knn
        obs = planes(B, eps)
        if not obs:
            rows.append(None); pvals.append(1.0); continue
        o = obs[0]
        ns, np_ = [], []
        for _ in range(N_NULL):
            if len(intact) < nb:
                break
            sel = stratified(c['mid'], broke, intact, rng)
            if len(sel) < MIN_CLUSTER:
                break
            pl = planes(c['mid'][sel], eps)
            ns.append(pl[0]['n'] if pl else 0)
            np_.append(pl[0]['planarity'] if pl else 0.0)
        if not ns:
            rows.append(None); pvals.append(1.0); continue
        ns, np_ = np.array(ns), np.array(np_)
        p = float(np.mean((ns >= o['n']) & (np_ >= o['planarity'])))
        p = max(p, 1.0 / (len(ns) + 1))          # cannot claim below resolution
        rows.append((eps, o, float(np.percentile(ns, 95)),
                     float(np.percentile(np_, 95))))
        pvals.append(p)

    adj = bh([p for p in pvals])
    best = None
    for (r, p, q) in zip(rows, pvals, adj):
        if r is None:
            continue
        eps, o, n95, pl95 = r
        flag = '  ** SIGNIFICANT' if q < 0.05 else ''
        print(f'   {eps:8.2f}{o["n"]:7d}{o["planarity"]:8.2f}{o["dip"]:6.0f}'
              f'{n95:10.0f}{pl95:11.2f}{q:8.3f}   {word(o["dip"]):<11s}{flag}')
        if q < 0.05 and (best is None or q < best[1]):
            best = (o, q)
    if best is None:
        print('   -> no clustering scale gives a surface beyond chance')
    else:
        o, q = best
        print(f'   -> surface at dip {o["dip"]:.0f} deg ({word(o["dip"])}), '
              f'{o["n"]} bonds, planarity {o["planarity"]:.2f}, q={q:.3f}')
    return dict(pair=pair, n_broke=nb, n_bond=int(len(broke)),
                knn=knn, q_min=float(np.min(adj)),
                significant=best is not None,
                dip=float(best[0]['dip']) if best else None,
                orientation=word(best[0]['dip']) if best else None)


def main(pairs):
    rng = np.random.default_rng(0)
    c0 = bonds('S1_12')
    thr = float(np.percentile(c0['sep'], 99))
    print(f'threshold from the unloaded control: {thr*1000:.0f} um '
          f'(its own false-positive rate 1.0% by construction)')
    out = {}
    for p in pairs:
        r = sweep(p, thr, rng)
        if r:
            out[p] = r
    with open(f'{OUT}/crack_sweep.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'\n-> {OUT}/crack_sweep.json')


if __name__ == '__main__':
    main(sys.argv[1:] or ['S1_12', 'S2_12', 'S2_23', 'S3_12'])
