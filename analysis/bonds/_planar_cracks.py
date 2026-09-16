"""Do broken bonds organise into planar features, or are they scattered?

The claim to be tested is that bond-breakage events cluster into co-planar
groups, which would be the signature of a shear band or a tension crack running
through the packing rather than of damage accumulating independently at
scattered contacts.

METHOD.  Each broken bond is one event, positioned at the midpoint of the two
grain centroids in the reference scan.  Events are clustered with DBSCAN, and
each cluster is fitted with a plane by principal component analysis.  Two
descriptors are kept per cluster: the RMS out-of-plane distance expressed in
grain radii (a shear band is physically a few grains thick, so an absolute
measure is more meaningful than a ratio), and the dip of the plane normal from
the loading axis.

WHY THERE IS A NULL CONTROL.  DBSCAN returns clusters from any point set dense
enough, and principal component analysis returns a best-fit plane from any three
or more points -- a handful of scattered events will look "planar" simply
because a small cloud has to be flat in some direction.  Observed planarity is
therefore meaningless on its own.

The null resamples WHICH bonds broke, drawing the observed number of breaks at
random from the bonds that were actually present and bonded in the reference
scan, and repeating the whole clustering and fitting.  Drawing from the real
bond population rather than from uniform random points is what makes the test
fair: it holds the packing geometry, the specimen shape and the spatial density
of bonds fixed, so the only thing being tested is whether the breaks are
arranged more planar-ly than a random subset of the same contacts would be.

A feature is reported as real only where the observed statistic falls outside
the null distribution.

Usage:  SPEC=<tag> python _planar_cracks.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from _spec import TAG, OUT, SCANS, FIRST, LAST, UM

RNG = np.random.default_rng(20260905)
N_NULL = 400
EPS_D = 1.5          # DBSCAN neighbourhood, in grain DIAMETERS
MIN_SAMPLES = 5      # a plane needs 3 points; 5 gives it some support
MIN_CLUSTER = 6      # clusters smaller than this are not called features


def load_events():
    """(broken positions, all bonded positions, grain radius) in voxels."""
    p = f'{OUT}/{TAG}_bond_survival_{FIRST}to{LAST}.csv'
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p)
    s = s[s.connected1]
    if not len(s):
        return None
    c = np.load(f'{OUT}/centroids_scan{FIRST:02d}.npz')
    pos = {int(i): x for i, x in zip(c['ids'], c['com'])}
    R = float(np.median(c['req']))

    def mid(row):
        a, b = pos.get(int(row.b1_a)), pos.get(int(row.b1_b))
        return None if a is None or b is None else 0.5 * (a + b)

    keep, broke = [], []
    for r in s.itertuples():
        m = mid(r)
        if m is None:
            continue
        keep.append(m)
        broke.append(r.outcome in ('detached', 'separated'))
    return np.array(keep), np.array(broke, bool), R


def clusters_of(X, eps):
    lab = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(X)
    out = []
    for k in set(lab) - {-1}:
        pts = X[lab == k]
        if len(pts) < MIN_CLUSTER:
            continue
        c = pts.mean(axis=0)
        u, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
        n = vt[2]                                   # plane normal
        rms = float(np.sqrt(np.mean(((pts - c) @ n) ** 2)))
        span = float(np.sqrt(np.mean(np.sum((pts - c) ** 2, axis=1))))
        dip = float(np.degrees(np.arccos(abs(n[0]))))   # from the loading axis
        out.append(dict(n=len(pts), rms=rms, span=span, dip=dip, centre=c))
    return out, lab


def summarise(cl):
    """Scalars that the null can be compared against."""
    if not cl:
        return dict(n_clusters=0, biggest=0, in_clusters=0, best_rms=np.nan)
    big = max(cl, key=lambda d: d['n'])
    return dict(n_clusters=len(cl), biggest=big['n'],
                in_clusters=sum(d['n'] for d in cl),
                best_rms=min(d['rms'] for d in cl))


def sweep(X_all, broke, R, eps_list):
    """Observed vs null across a range of neighbourhood sizes.

    A single eps is a researcher degree of freedom: too large and DBSCAN
    returns the whole specimen as one cluster (the null then produces a BIGGER
    cluster than the data), too small and it returns nothing.  The whole sweep
    is reported rather than the most favourable point, so that a null result
    cannot be turned into a positive one by choosing eps.
    """
    n_br = int(broke.sum())
    rows = []
    for ed in eps_list:
        eps = ed * 2 * R
        obs = summarise(clusters_of(X_all[broke], eps)[0])
        nul = pd.DataFrame([summarise(clusters_of(
            X_all[RNG.choice(len(X_all), size=n_br, replace=False)], eps)[0])
            for _ in range(N_NULL)])
        rms_o = obs['best_rms'] / R if np.isfinite(obs['best_rms']) else np.nan
        rms_n = (nul.best_rms / R).dropna()
        rows.append(dict(
            eps_d=ed,
            clusters=obs['n_clusters'], clusters_null=float(nul.n_clusters.median()),
            p_clusters=float((nul.n_clusters >= obs['n_clusters']).mean()),
            biggest=obs['biggest'], biggest_null=float(nul.biggest.median()),
            p_biggest=float((nul.biggest >= obs['biggest']).mean()),
            rms_R=rms_o, rms_null=float(rms_n.median()) if len(rms_n) else np.nan,
            p_rms=float((rms_n <= rms_o).mean()) if (len(rms_n) and np.isfinite(rms_o)) else np.nan,
            frac_in=obs['in_clusters'] / max(n_br, 1)))
    return pd.DataFrame(rows)


def main():
    got = load_events()
    if got is None:
        print(f'{TAG}: no survival file', flush=True)
        return
    X_all, broke, R = got
    n_br = int(broke.sum())
    print(f'{TAG}: {len(X_all)} bonds followed, {n_br} broken '
          f'({100*n_br/max(len(X_all),1):.0f}%), grain radius {R:.1f} vox',
          flush=True)
    if n_br < MIN_CLUSTER:
        print('  too few breaks to cluster', flush=True)
        return

    sw = sweep(X_all, broke, R, [0.6, 0.8, 1.0, 1.25, 1.5, 2.0])
    print(f'  {"eps/d":>6s}{"clusters":>10s}{"null":>6s}{"p":>7s}'
          f'{"biggest":>9s}{"null":>6s}{"p":>7s}'
          f'{"RMS/R":>8s}{"null":>7s}{"p":>7s}{"in cl.":>8s}', flush=True)
    for r in sw.itertuples():
        print(f'  {r.eps_d:6.2f}{r.clusters:10d}{r.clusters_null:6.0f}'
              f'{r.p_clusters:7.3f}{r.biggest:9d}{r.biggest_null:6.0f}'
              f'{r.p_biggest:7.3f}{r.rms_R:8.2f}{r.rms_null:7.2f}'
              f'{r.p_rms:7.3f}{100*r.frac_in:7.0f}%', flush=True)
    sw.insert(0, 'specimen', TAG)
    sw.to_csv(f'{OUT}/{TAG}_planar_sweep.csv', index=False)

    best = sw.p_rms.min() if sw.p_rms.notna().any() else np.nan
    hit = sw[(sw.p_biggest <= 0.05) | (sw.p_rms <= 0.05)]
    if len(hit):
        print(f'  -> planar organisation above chance at eps/d = '
              f'{list(hit.eps_d)}', flush=True)
    else:
        print(f'  -> no planar organisation above chance at any eps '
              f'(best p = {best:.2f})', flush=True)
    print(f'  -> {TAG}_planar_sweep.csv', flush=True)


if __name__ == '__main__':
    main()
