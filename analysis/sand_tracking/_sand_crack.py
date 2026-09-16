"""Crack surfaces in sand, from grain motion rather than from the grey image.

A crack is where neighbouring grains come apart.  Both grains are tracked, so
the separation of a specific pair is measurable, and the surface the separated
pairs lie on is the crack.  That surface has an orientation, which is the
horizontal / diagonal / vertical the figures are supposed to show.

WHY CENTRE-TO-CENTRE AND NOT THE GAP.  The obvious definition -- bond broken
when the gap between the grain surfaces opens -- cannot be used here.  The
measured median gap between neighbouring grains INCREASES at every load step
(S2: 1.00 -> 2.00 -> 2.83 voxels) while the column is being compacted by up to
21%.  Grains cannot move apart while the specimen shortens, so what is growing
is not the gap but the amount of each grain the DL model failed to claim: the
labels shrink between scans.  Any gap threshold would read that shrinkage as
breakage.  The distance between two grain CENTRES does not depend on how much
of each grain was claimed, so that is what is used.

    bond, defined in scan A:  centres closer than (r_i + r_j) * BOND_TOL
    broken, measured in B:    that distance grew by more than THRESH

THE THRESHOLD IS NOT CHOSEN, IT IS CALIBRATED.  S1 1->2 is an unloaded pair --
the column length is identical, 733 voxels in both scans -- so every bond it
calls broken is a false positive.  The threshold is set at the 99th percentile
of the separation distribution on that pair, and the false-positive rate it
leaves is reported with every result.

ORIENTATION.  Broken bonds are clustered with DBSCAN, each cluster's midpoints
are fitted with a plane by PCA, and the angle between that plane's normal and
the loading axis gives the dip: 0 deg normal = a HORIZONTAL crack across the
column, 90 deg = a VERTICAL split, in between = diagonal.  Planarity is the
ratio of the smallest to the middle singular value, so a cluster that is really
a blob is not reported as a plane.

NULL CONTROL.  The same clustering on the same number of randomly chosen INTACT
bonds, repeated, gives the cluster sizes and planarities that this packing
produces by chance.  A crack has to beat that, not merely exist -- the same
test that showed the beads had no planar cracking.
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
from _sand_motion import _load, _shortening, NICE, MM

OUT = f'{SC}/sand_out'
BOND_TOL = 1.25        # centres within this multiple of (r_i + r_j) are bonded
N_NULL = 40            # null resamples
MIN_CLUSTER = 12       # a plane needs at least this many broken bonds


def bonds(pair):
    """Bonded pairs in scan A, and how far their centres moved apart in B."""
    f = f'{OUT}/dvc_{pair}/grain_track.npz'
    if not os.path.exists(f):
        return None
    got = _load(pair)
    if got is None:
        return None
    a, u = got
    d = np.load(f)
    vol = d['vol_a'].astype(float)
    if len(vol) != len(a):                      # _load may drop grains
        vol = np.full(len(a), np.median(vol))
    r = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0) * MM        # mm
    b = a + u

    rmax = float(np.percentile(r, 99))
    tree = cKDTree(a)
    pairs = np.array(sorted(tree.query_pairs(2.0 * rmax * BOND_TOL)), dtype=int)
    if not len(pairs):
        return None
    i, j = pairs[:, 0], pairs[:, 1]
    d0 = np.linalg.norm(a[i] - a[j], axis=1)
    keep = d0 <= (r[i] + r[j]) * BOND_TOL
    i, j, d0 = i[keep], j[keep], d0[keep]
    d1 = np.linalg.norm(b[i] - b[j], axis=1)
    mid = 0.5 * (a[i] + a[j])
    return dict(i=i, j=j, sep=d1 - d0, mid=mid, n_grain=len(a),
                r=r, d0=d0)


def calibrate():
    """Separation threshold from the unloaded pair, where nothing broke."""
    c = bonds('S1_12')
    if c is None:
        return None, None
    s = c['sep']
    thr = float(np.percentile(s, 99.0))
    return thr, s


def planes(mid, eps, min_samples=6):
    """DBSCAN the broken-bond midpoints; fit a plane to each cluster."""
    if len(mid) < MIN_CLUSTER:
        return []
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(mid)
    out = []
    for k in range(lab.max() + 1):
        m = lab == k
        if m.sum() < MIN_CLUSTER:
            continue
        P = mid[m]
        c = P.mean(axis=0)
        sv = np.linalg.svd(P - c, compute_uv=True)
        s, V = sv[1], sv[2]
        if s[1] <= 0:
            continue
        planarity = 1.0 - s[2] / s[1]           # 1 = perfectly flat
        n = V[2]                                # normal = smallest direction
        dip = np.degrees(np.arccos(abs(n[0])))  # 0 = normal along the axis
        out.append(dict(n=int(m.sum()), centre=c, normal=n,
                        planarity=float(planarity), dip=float(dip),
                        extent=float(s[0] / np.sqrt(max(m.sum() - 1, 1)))))
    return sorted(out, key=lambda d: -d['n'])


def analyse(pair, thr, rng=None):
    c = bonds(pair)
    if c is None:
        print(f'{pair}: no bonds'); return None
    rng = rng or np.random.default_rng(0)
    broke = c['sep'] > thr
    nb = int(broke.sum())
    frac = nb / max(len(broke), 1)
    eps = 3.0 * float(np.median(c['r'])) * 2.0        # ~3 grain diameters
    obs = planes(c['mid'][broke], eps)

    # NULL: the same count of intact bonds, clustered the same way
    intact = np.flatnonzero(~broke)
    null_big, null_plan = [], []
    for _ in range(N_NULL):
        if nb < MIN_CLUSTER or len(intact) < nb:
            break
        sel = rng.choice(intact, nb, replace=False)
        pl = planes(c['mid'][sel], eps)
        null_big.append(pl[0]['n'] if pl else 0)
        null_plan.append(pl[0]['planarity'] if pl else 0.0)

    print(f'\n{NICE[pair]}')
    print(f'   bonds defined in scan A      {len(broke)}')
    print(f'   separated beyond threshold   {nb}  ({100*frac:.2f}%)')
    if null_big:
        nb95 = float(np.percentile(null_big, 95))
        np95 = float(np.percentile(null_plan, 95))
        print(f'   null control (n={len(null_big)}): largest chance cluster '
              f'{np.median(null_big):.0f} (p95 {nb95:.0f}), '
              f'planarity p95 {np95:.2f}')
    else:
        nb95 = np95 = np.inf
        print('   null control: too few broken bonds to resample')
    if not obs:
        print('   no cluster reached the minimum size: no crack surface')
        return dict(pair=pair, n_bond=len(broke), n_broke=nb, planes=[])
    print(f'   {"clust":>6s}{"bonds":>7s}{"planar":>8s}{"dip":>7s}'
          f'{"extent mm":>11s}   orientation')
    kept = []
    for k, p in enumerate(obs[:6]):
        beats = p['n'] > nb95 and p['planarity'] > np95
        word = ('horizontal' if p['dip'] < 30 else
                'vertical' if p['dip'] > 60 else 'diagonal')
        print(f'   {k:6d}{p["n"]:7d}{p["planarity"]:8.2f}{p["dip"]:7.0f}'
              f'{p["extent"]:11.2f}   {word}'
              + ('   ** beats the null' if beats else '   (within chance)'))
        p['beats_null'] = bool(beats)
        p['word'] = word
        kept.append(p)
    return dict(pair=pair, n_bond=len(broke), n_broke=nb, planes=kept,
                thr=thr, eps=eps, mid=c['mid'], broke=broke)


def main(pairs):
    thr, s1 = calibrate()
    if thr is None:
        print('cannot calibrate: S1_12 grain track missing'); return
    print(f'threshold from the UNLOADED pair S1 1->2: separation > '
          f'{thr*1000:.0f} um')
    print(f'   its own separation distribution: p50 {np.median(s1)*1000:+.0f}, '
          f'p90 {np.percentile(s1,90)*1000:+.0f}, '
          f'p99 {np.percentile(s1,99)*1000:+.0f} um')
    print(f'   so this threshold calls {100*(s1>thr).mean():.1f}% of bonds '
          f'broken on a specimen that did not shorten at all')
    res = {}
    for p in pairs:
        r = analyse(p, thr)
        if r:
            res[p] = {k: v for k, v in r.items() if k not in ('mid', 'broke')}
    with open(f'{OUT}/crack_summary.json', 'w') as fh:
        json.dump(res, fh, indent=2, default=lambda o: (
            o.tolist() if isinstance(o, np.ndarray) else float(o)))
    print(f'\n-> {OUT}/crack_summary.json')


if __name__ == '__main__':
    main(sys.argv[1:] or ['S1_12', 'S2_12', 'S2_23', 'S3_12'])
