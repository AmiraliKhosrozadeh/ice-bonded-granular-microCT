"""Merge the core recompute into one CSV and one LaTeX table body.

eps, S_V and k come from props_core.csv (fixed core, full resolution).
Air connectivity is re-tested on the same core.
tau_ice is unchanged: it is measured on cubes placed wholly inside the sample
hull, so it never depended on where the hull boundary sat.
"""
import csv
import glob
import io
import os
import re

import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "micro")
HB, RF = (0.20, 0.80), 0.85
STRUCT = np.ones((3, 3, 3), bool)
ORDER = ["G1", "G2", "G3", "G4", "G5", "A1", "A2", "A3", "A4", "S1", "S2", "S3"]
MAT = {"G": "glass", "A": r"$\gamma$-alumina", "S": "sand"}


def geom(p):
    z = np.load(p)
    ph = z["phase"]
    inside = ph > 0
    occ = np.where(inside.any(axis=(1, 2)))[0]
    lo, hi = int(occ[0]), int(occ[-1])
    h = hi - lo
    band = slice(lo + int(HB[0] * h), lo + int(HB[1] * h) + 1)
    gr = ph[band] == 3
    if not gr.any():
        gr = ph[band] >= 2
    _, gy, gx = np.nonzero(gr)
    cy, cx = gy.mean(), gx.mean()
    R = float(np.quantile(np.hypot(gy - cy, gx - cx), 0.98))
    return ph, band, (cy, cx), R


def perc(air):
    lab, n = ndi.label(air, structure=STRUCT)
    if n == 0:
        return False, 0.0, 0
    nz = lab.shape[0]
    b = max(1, int(0.05 * nz))
    lo = np.unique(lab[:b])
    hi = np.unique(lab[-b:])
    sz = np.bincount(lab.ravel())
    sz[0] = 0
    return (bool(np.intersect1d(lo[lo > 0], hi[hi > 0]).size),
            float(sz.max() / air.sum()), int(n))


bins = {}
for p in sorted(glob.glob(os.path.join(M, "bin2", "*.npz"))):
    m = re.match(r"([A-Z]\d)_(\d)\.npz$", os.path.basename(p))
    bins.setdefault(m.group(1), []).append((int(m.group(2)), p))

conn = {}
for k, v in bins.items():
    v.sort()
    _, _, _, R0 = geom(v[0][1])
    for st, p in v:
        ph, band, (cy, cx), _ = geom(p)
        sub = ph[band]
        Y, X = np.ogrid[:sub.shape[1], :sub.shape[2]]
        disc = np.hypot(Y - cy, X - cx) <= RF * R0
        z, big, nc = perc((sub == 1) & disc)
        conn[(k, st)] = (z, big, nc)

core = {}
for r in csv.DictReader(io.open(os.path.join(M, "props_core.csv"), encoding="utf-8")):
    core[(r["id"], int(r["stage"]))] = r

tau = {}
for r in csv.DictReader(io.open(os.path.join(M, "section36.csv"), encoding="utf-8")):
    tau[(r["id"], int(r["stage"]))] = (r["tau_ice"], r["n_cubes_ok"], r["n_cubes"])

out = ["id,stage,material,eps,phi_ice,rho,SV_per_mm,k_KC_m2,air_percolates,"
       "largest_air_share,tau_ice,n_cubes_ok,n_cubes"]
tex = []
for k in ORDER:
    stages = sorted(st for (i, st) in core if i == k)
    for j, st in enumerate(stages):
        r = core[(k, st)]
        z, big, _ = conn[(k, st)]
        t, ok, n = tau.get((k, st), ("", "", ""))
        out.append("%s,%d,%s,%.4f,%s,%s,%.3f,%.4g,%s,%.3f,%s,%s,%s" % (
            k, st, r["material"], float(r["eps"]),
            ("%.4f" % float(r["phi_ice"])) if r["phi_ice"] else "",
            ("%.4f" % float(r["rho"])) if r["rho"] else "",
            float(r["SV_per_mm"]), float(r["k_KC_m2"]),
            "True" if z else "False", big, t, ok, n))
        kk = float(r["k_KC_m2"]) * 1e13
        tval = ("$%.2f$ (%s/%s)" % (float(t), ok, n)) if t else "--- (%s/%s)" % (ok, n)
        head = "%s & %s" % (k, MAT[k[0]]) if j == 0 else "   &"
        tex.append("      %s & %d & %.3f & %.2f & %.3g & %s & %s \\\\"
                   % (head, st, float(r["eps"]), float(r["SV_per_mm"]), kk,
                      "yes" if z else "no", tval))
    if k != ORDER[-1]:
        tex.append(r"      \midrule")

io.open(os.path.join(M, "section36_core.csv"), "w", encoding="utf-8",
        newline="\n").write("\n".join(out) + "\n")
io.open(os.path.join(M, "tab_microevo_core.tex"), "w", encoding="utf-8",
        newline="\n").write("\n".join(tex) + "\n")
print("\n".join(tex))
