"""Is the void-fraction dip at load step 2 real, or a specimen-mask artefact?

The mask that defines the specimen interior is not the same size from one scan
to the next -- interior volume moves by -13 to +28 % between consecutive scans
of the SAME specimen, which a column under axial compression cannot do.  So
eps = air / interior is contaminated by whatever the mask happened to include.

Porosity is intensive, so it can be measured on a region that is defined the
same way in every scan instead.  Here: the middle band of the column by
height, and inside a fraction of the fitted radius, so that neither the ragged
outer boundary nor the two ends enter the count.

Prints eps over the whole mask (as published) beside eps on the common core.
"""
import glob
import os
import re

import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "micro", "bin2")
HBAND = (0.20, 0.80)      # keep the middle 60 % of the column height
RFRAC = 0.85              # keep inside 85 % of the fitted radius
ORDER = ["G1", "G2", "G3", "G4", "G5", "A1", "A2", "A3", "A4", "S1", "S2", "S3"]


def core_eps(path):
    z = np.load(path)
    ph = z["phase"]
    vox = float(z["vox_mm"]) ** 3
    inside = ph > 0
    occ = np.where(inside.any(axis=(1, 2)))[0]
    lo, hi = int(occ[0]), int(occ[-1])
    h = hi - lo
    z0 = lo + int(round(HBAND[0] * h))
    z1 = lo + int(round(HBAND[1] * h))
    sub = inside[z0:z1 + 1]
    air = (ph[z0:z1 + 1] == 1)

    # radius from the mask itself, per specimen not per slice, so the crop is
    # the same shape at every height
    yy, xx = np.nonzero(sub.any(axis=0))
    cy, cx = yy.mean(), xx.mean()
    r = np.hypot(yy - cy, xx - cx)
    R = np.quantile(r, 0.98)
    Y, X = np.ogrid[:sub.shape[1], :sub.shape[2]]
    disc = (np.hypot(Y - cy, X - cx) <= RFRAC * R)

    sub = sub & disc
    air = air & disc
    n_in = int(sub.sum())
    return (float(air.sum()) / max(n_in, 1),
            n_in * vox,
            float(inside.sum()) * vox,
            float((ph == 1).sum()) / max(int(inside.sum()), 1),
            (hi - lo + 1) * float(z["vox_mm"]))


rows = {}
for p in sorted(glob.glob(os.path.join(D, "*.npz"))):
    m = re.match(r"([A-Z]\d)_(\d)\.npz$", os.path.basename(p))
    if not m:
        continue
    rows.setdefault(m.group(1), []).append((int(m.group(2)),) + core_eps(p))

print("%-4s %2s %9s %9s %9s %9s %8s" %
      ("id", "st", "eps_all", "eps_core", "V_all", "V_core", "H_mm"))
for k in ORDER:
    if k not in rows:
        continue
    prev = None
    for st, e_core, v_core, v_all, e_all, hmm in sorted(rows[k]):
        flag = ""
        if prev is not None:
            flag = "  all%+.4f core%+.4f" % (e_all - prev[0], e_core - prev[1])
        print("%-4s %2d %9.4f %9.4f %9.1f %9.1f %8.2f%s" %
              (k, st, e_all, e_core, v_all, v_core, hmm, flag))
        prev = (e_all, e_core)
    print()
