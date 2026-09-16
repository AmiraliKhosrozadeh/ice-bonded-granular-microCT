"""Crack-orientation renders (failure_classification pipeline) for every loaded
scan of every bead specimen, one sheet per material, with the paper's legend.

    python scripts/fig_crackdir_supp.py  -> supplementary/crackdir_glass.png, crackdir_alumina.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crack_split_tight import REG
import paper_style as ps

FC = "E:/RPTU-images/CT_images/paper_figures/failure_classification"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "supplementary")
LEGEND = os.path.join(ROOT, "figures", "crackdir_legend.png")
GROUPS = {"glass": ["G1", "G2", "G3", "G4", "G5"], "alumina": ["A1", "A2", "A3", "A4"]}


def crop(path):
    a = plt.imread(path)
    if a.max() > 1.0:
        a = a / 255.0
    if a.shape[2] == 4:
        al = a[..., 3:4]
        a = a[..., :3] * al + (1.0 - al)
    a = a[..., :3]
    ink = a.mean(axis=2) < 0.97
    rows = np.where(ink.any(axis=1))[0]
    # first ink block down the page is the render; the baked legend below is dropped
    r0, r1 = rows[0], rows[0]
    for v in rows[1:]:
        if v - r1 > 20:
            break
        r1 = v
    cols = np.where(ink[r0:r1 + 1].any(axis=0))[0]
    return a[r0:r1 + 1, cols[0]:cols[-1] + 1]


def crop_all(path):
    a = plt.imread(path)
    if a.max() > 1.0:
        a = a / 255.0
    a = a[..., :3]
    ink = a.mean(axis=2) < 0.97
    ys, xs = np.nonzero(ink)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def sheet(name, ids):
    rows = []
    for pid in ids:
        tag, spam, scans = REG[pid]
        imgs = []
        for s in scans[1:]:
            p = f"{FC}/{tag}/{tag}_scan{s:02d}_failure_class.png"
            if os.path.exists(p):
                imgs.append((s, crop(p)))
        rows.append((pid, imgs))
    ncol = max(len(r[1]) for r in rows)
    ps.apply(9.0)
    fig = plt.figure(figsize=(ps.TW, 0.38 * ps.TW * len(rows) + 0.8))
    gs = fig.add_gridspec(len(rows) + 1, ncol, height_ratios=[1] * len(rows) + [0.14],
                          hspace=0.16, wspace=0.08, left=0.01, right=0.99, top=0.975, bottom=0.01)
    for i, (pid, imgs) in enumerate(rows):
        for j in range(ncol):
            ax = fig.add_subplot(gs[i, j])
            ax.axis("off")
            if j < len(imgs):
                s, im = imgs[j]
                ax.imshow(im)
                ax.set_title(f"{pid}, load step {s}", fontsize=9, pad=3)
    axl = fig.add_subplot(gs[-1, :])
    axl.axis("off")
    axl.imshow(crop_all(LEGEND))
    out = os.path.join(OUTDIR, f"crackdir_{name}.png")
    fig.savefig(out, dpi=250, facecolor="white")
    plt.close(fig)
    print("->", out)


if __name__ == "__main__":
    for name, ids in GROUPS.items():
        sheet(name, ids)
