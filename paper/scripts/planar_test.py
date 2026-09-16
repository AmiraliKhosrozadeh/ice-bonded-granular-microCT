"""Breakage-event clustering null test (Methods, spatial organisation).

Portable copy of the pipeline's _planar_cracks.py that reads the bond survival
census and the first-scan label image from E: and writes
scripts/data/planar_sweep/<tag>_planar_sweep.csv.  Same parameters and seed as
the pipeline: DBSCAN with min_samples 5, clusters below 6 events ignored,
400 resamplings of WHICH bonds broke from the bonds present in the reference
scan, neighbourhood swept over 0.6-2.0 grain diameters with 1.0 the
pre-specified value.

    python scripts/planar_test.py Alumina_100_1800_T5
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import tifffile
from sklearn.cluster import DBSCAN

BF = "E:/RPTU-images/CT_images/paper_figures/bond_failure"
CT = "E:/RPTU-images/CT_images"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "planar_sweep")
LABELS = {
    "Alumina_100_1800_T5": f"{CT}/Alumina/Alumina_100_1800_T5/Alumina_100_1800_T5_spam/data/bead_labels_scan01.tif",
    "Glass_100_1700_T5_HR": f"{CT}/Glass/Glass_100_1700_T5_HR/Glass_T5_HR_spam/data/bead_labels_scan01_aligned.tif",
}
RNG = np.random.default_rng(20260905)
N_NULL = 400
MIN_SAMPLES = 5
MIN_CLUSTER = 6


def centroids(path):
    """label -> (z, y, x) centroid and equivalent radius, page by page."""
    with tifffile.TiffFile(path) as t:
        n = max(int(p.asarray().max()) for p in t.pages) + 1
        cnt = np.zeros(n); sz = np.zeros(n); sy = np.zeros(n); sx = np.zeros(n)
        for z, p in enumerate(t.pages):
            a = p.asarray().astype(np.int64)
            m = a > 0
            lab = a[m]
            yy, xx = np.nonzero(m)
            cnt += np.bincount(lab, minlength=n)
            sz += np.bincount(lab, weights=np.full(len(lab), float(z)), minlength=n)
            sy += np.bincount(lab, weights=yy.astype(float), minlength=n)
            sx += np.bincount(lab, weights=xx.astype(float), minlength=n)
    ok = cnt > 0
    com = np.stack([sz, sy, sx], 1)[ok] / cnt[ok, None]
    req = (3 * cnt[ok] / (4 * np.pi)) ** (1 / 3)
    return dict(zip(np.nonzero(ok)[0], com)), float(np.median(req))


def clusters_of(X, eps):
    lab = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(X)
    out = []
    for k in set(lab) - {-1}:
        pts = X[lab == k]
        if len(pts) < MIN_CLUSTER:
            continue
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        n = vt[2]
        out.append(dict(n=len(pts), rms=float(np.sqrt(np.mean(((pts - c) @ n) ** 2)))))
    return out


def summarise(cl):
    if not cl:
        return dict(n_clusters=0, biggest=0, in_clusters=0, best_rms=np.nan)
    return dict(n_clusters=len(cl), biggest=max(d["n"] for d in cl),
                in_clusters=sum(d["n"] for d in cl), best_rms=min(d["rms"] for d in cl))


def sweep(X_all, broke, R, eps_list):
    n_br = int(broke.sum())
    rows = []
    for ed in eps_list:
        eps = ed * 2 * R
        obs = summarise(clusters_of(X_all[broke], eps))
        nul = pd.DataFrame([summarise(clusters_of(
            X_all[RNG.choice(len(X_all), size=n_br, replace=False)], eps))
            for _ in range(N_NULL)])
        rms_o = obs["best_rms"] / R if np.isfinite(obs["best_rms"]) else np.nan
        rms_n = (nul.best_rms / R).dropna()
        rows.append(dict(
            eps_d=ed,
            clusters=obs["n_clusters"], clusters_null=float(nul.n_clusters.median()),
            p_clusters=float((nul.n_clusters >= obs["n_clusters"]).mean()),
            biggest=obs["biggest"], biggest_null=float(nul.biggest.median()),
            p_biggest=float((nul.biggest >= obs["biggest"]).mean()),
            rms_R=rms_o, rms_null=float(rms_n.median()) if len(rms_n) else np.nan,
            p_rms=float((rms_n <= rms_o).mean()) if (len(rms_n) and np.isfinite(rms_o)) else np.nan,
            frac_in=obs["in_clusters"] / max(n_br, 1)))
    return pd.DataFrame(rows)


def main(tag):
    s = pd.read_csv(glob.glob(f"{BF}/{tag}/data/{tag}_bond_survival_*.csv")[0])
    s = s[s.connected1]
    pos, R = centroids(LABELS[tag])
    keep, broke = [], []
    for r in s.itertuples():
        a, b = pos.get(int(r.b1_a)), pos.get(int(r.b1_b))
        if a is None or b is None:
            continue
        keep.append(0.5 * (a + b))
        broke.append(r.outcome in ("detached", "separated"))
    X_all, broke = np.array(keep), np.array(broke, bool)
    print(f"{tag}: {len(X_all)} bonds followed, {broke.sum()} broken, R {R:.1f} vox", flush=True)
    sw = sweep(X_all, broke, R, [0.6, 0.8, 1.0, 1.25, 1.5, 2.0])
    sw.insert(0, "specimen", tag)
    print(sw.to_string(index=False))
    sw.to_csv(os.path.join(OUT, f"{tag}_planar_sweep.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1])
