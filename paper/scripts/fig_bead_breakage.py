"""Grey-value sections through broken gamma-alumina beads at the last load
step, one panel per specimen, the loaded end at the top of each.

The planes were chosen by eye from the fragment census (labels much smaller
than a bead in the last-scan labelling).  A4 is read from the aligned volume;
A1 and A2 from the raw reconstructions, at the z offset that matches the
aligned frame, because their aligned volumes are masked to the specimen and
the masking shows in a grey-value panel.  Slices are cached under data/ so
the figure rebuilds without E:.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_style as ps

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "bead_breakage.npz")
OUT = os.path.join(os.path.dirname(HERE), "figures", "bead_breakage.png")
VOX = 24.7660229 / 1000.0            # mm
H, HALF = 300, 150                   # window height and half width, voxels

A = "E:/RPTU-images/CT_images/Alumina"
# label, file, plane, (z of window top, fixed coordinate, centre of the free coordinate)
PANELS = [
    ("A4", A + "/Alumina-75-1800-T7/Alumina_75_1800_T7_spam/data/ct_scan03_aligned.tif",
     "yz", (78, 333, 233)),           # punch face at z 93; plane x = 333 through bead 354
    ("A1", A + "/Alumina_100_1800_T5/Alumina_100_1800_T5_spam/data/ct_scan03.tif",
     "xz", (330, 302, 399)),          # aligned z 507 is raw z 426; plane y = 302
    ("A2", A + "/Alumina_175_1800_T5/Alumina_175_1800_T5_spam/data/ct_scan02.tif",
     "yz", (370, 422, 240)),          # aligned z 533 is raw z 500; plane x = 422
]

if not os.path.exists(CACHE):
    import tifffile

    got = {}
    for lab, f, plane, (z0, fixed, c) in PANELS:
        v = tifffile.imread(f)
        lo, hi = np.percentile(v[::8, ::8, ::8], [1, 99.5])
        if plane == "yz":
            img = v[z0:z0 + H, c - HALF:c + HALF, fixed]
        else:
            img = v[z0:z0 + H, fixed, c - HALF:c + HALF]
        got[lab] = ((img.astype(np.float32) - lo) / (hi - lo)).clip(0, 1)
        del v
    np.savez_compressed(CACHE, **got)

d = np.load(CACHE)
ps.apply(10.0)
n = len(PANELS)
fig, ax = plt.subplots(1, n, figsize=(ps.TW, ps.TW / n * H / (2 * HALF) * 1.0))
for k, (lab, *_) in enumerate(PANELS):
    a = ax[k]
    a.imshow(d[lab], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    a.set_xticks([])
    a.set_yticks([])
    for sp in a.spines.values():
        sp.set_linewidth(0.8)
    a.text(0.03, 0.97, "(%s) %s" % ("abc"[k], lab), transform=a.transAxes,
           ha="left", va="top", fontsize=10, color="white")
px = 1.0 / VOX
ax[1].plot([12, 12 + px], [H - 12, H - 12], color="white", linewidth=2.2,
            solid_capstyle="butt")
ax[1].text(12 + px / 2, H - 18, "1 mm", color="white", ha="center",
            va="bottom", fontsize=9)
fig.subplots_adjust(left=0.004, right=0.996, top=0.99, bottom=0.01, wspace=0.03)
ps.save(fig, OUT, pdf=False)
