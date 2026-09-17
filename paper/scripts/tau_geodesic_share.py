"""Share of the ice above the unloaded column's range in the ice-path map of
every scan (the rule of fig_tau_geodesic.py, from its cached fields).

    python scripts/tau_geodesic_share.py
        -> supplementary/tab_taumap.tex, scripts/data/tau_geodesic_share.csv
"""
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_tau_geodesic as tg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ceiling(val, vox):
    """the unloaded column's percentile per 1 mm of height, smoothed, floored"""
    Z = val.shape[0]
    dist = np.abs(np.arange(Z) - (Z - 1 if tg.SEED == "support" else 0)) * vox
    nb = max(1, int(round(1.0 / vox)))
    prof = np.full(Z, np.nan)
    for k in range(0, Z, nb):
        v = val[k:k + nb]
        if np.isfinite(v).any():
            prof[k:k + nb] = np.nanpercentile(v, float(tg.FADE_PCT))
    ok = np.isfinite(prof)
    o = np.argsort(dist[ok])
    prof = np.interp(dist, dist[ok][o], prof[ok][o])
    prof = ndi.uniform_filter1d(prof, 2 * nb + 1, mode="nearest")
    return dist, np.maximum(prof, tg.FADE)


def share(val, dist0, thr0, vox, halo):
    Z = val.shape[0]
    dz = np.abs(np.arange(Z) - (Z - 1 if tg.SEED == "support" else 0)) * vox
    o = np.argsort(dist0)
    t = np.interp(dz, dist0[o], thr0[o])[:, None, None]
    v = val[:-halo] if tg.SEED == "support" else val[halo:]
    t = t[:-halo] if tg.SEED == "support" else t[halo:]
    hi = np.nan_to_num(v, nan=-1e9) > t
    hi = ndi.binary_opening(hi, iterations=1)
    lab, n = ndi.label(hi)
    if n:
        sz = np.bincount(lab.ravel()); sz[0] = 0
        hi = (sz * vox ** 3 >= tg.MIN_BODY_MM3)[lab]
    return 100.0 * hi.sum() / np.isfinite(v).sum()


def main():
    rows = []
    for pid, mat, stages, r in tg.SPECS:
        vox = tg.vox_of(pid)
        halo = int(tg.HALO_MM / vox)
        val0, _, _, _ = tg.field(pid, stages[0], r)
        dist0, thr0 = ceiling(val0, vox)
        for st in stages:
            val, big, ice, _ = tg.field(pid, st, r)
            s = share(val, dist0, thr0, vox, halo)
            rows.append((pid, st, s))
            print(f"{pid} {st}: {s:.2f} %", flush=True)
    os.makedirs(os.path.join(ROOT, "supplementary"), exist_ok=True)
    with open(os.path.join(ROOT, "scripts", "data", "tau_geodesic_share.csv"), "w") as f:
        f.write("id,scan,share_pct\n")
        for pid, st, s in rows:
            f.write(f"{pid},{st},{s:.2f}\n")
    # one row per specimen, one column per scan
    by = {}
    for pid, st, s in rows:
        by.setdefault(pid, {})[st] = s
    nmax = max(max(d) for d in by.values())
    L = ["\\begin{tabular}{l" + "r" * nmax + "}", "\\toprule",
         "Specimen & unloaded & " + " & ".join(f"load step {k}" for k in range(2, nmax + 1)) + " \\\\",
         "\\midrule"]
    for pid, _, _, _ in tg.SPECS:
        d = by[pid]
        cells = [f"{d[k]:.2f}" if k in d else "--" for k in range(1, nmax + 1)]
        L.append(pid + " & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(ROOT, "supplementary", "tab_taumap.tex"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("-> supplementary/tab_taumap.tex")


if __name__ == "__main__":
    main()
