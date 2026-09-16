"""Void and crack classes at every scan of every bead specimen, one sheet per
material, for the supplementary material.  Renders from render_crack3d.py
(classes of crack_split_tight.py), composed as in fig_crack3d.py.

    python scripts/fig_crack3d_supp.py   -> supplementary/crack3d_glass.png, crack3d_alumina.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crack_split_tight import REG
import paper_style as ps

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "supplementary")
C_VOID, C_SURF, C_BODY = "#4fc3f7", "#f5a623", "#cc0000"
GROUPS = {"glass": ["G1", "G2", "G3", "G4", "G5"], "alumina": ["A1", "A2", "A3", "A4"]}


def load(path):
    a = plt.imread(path)
    if a.max() > 1.0:
        a = a / 255.0
    if a.shape[2] == 4:
        al = a[..., 3:4]
        a = a[..., :3] * al + (1.0 - al)
    return a[..., :3]


def runs(idx, gap):
    out, s, p = [], idx[0], idx[0]
    for v in idx[1:]:
        if v - p > gap:
            out.append((s, p))
            s = v
        p = v
    out.append((s, p))
    return out


def specimen(img):
    ink = img.mean(axis=2) < 0.97
    r0, r1 = runs(np.where(ink.any(axis=1))[0], 20)[0]
    cols = np.where(ink[r0:r1 + 1].any(axis=0))[0]
    return img[r0:r1 + 1, cols[0]:cols[-1] + 1]


def sheet(name, ids):
    rows = []
    for pid in ids:
        tag, spam, scans = REG[pid]
        d = f"{spam}/results_voidcrack/tight2"
        imgs = [specimen(load(f"{d}/voidcrack3d_scan{s:02d}.png")) for s in scans]
        rows.append((pid, scans, imgs))
    ncol = max(len(r[2]) for r in rows)
    ps.apply(9.0)
    fig = plt.figure(figsize=(ps.TW, 0.42 * ps.TW * len(rows) + 0.6))
    gs = fig.add_gridspec(len(rows) + 1, ncol, height_ratios=[1] * len(rows) + [0.12],
                          hspace=0.16, wspace=0.08, left=0.01, right=0.99, top=0.975, bottom=0.01)
    for i, (pid, scans, imgs) in enumerate(rows):
        H = max(im.shape[0] for im in imgs)
        for j in range(ncol):
            ax = fig.add_subplot(gs[i, j])
            ax.axis("off")
            if j < len(imgs):
                im = imgs[j]
                can = np.ones((H, im.shape[1], 3))
                can[:im.shape[0]] = im
                ax.imshow(can)
                ax.set_title(f"{pid}, {'unloaded' if scans[j] == 1 else f'load step {scans[j]}'}", fontsize=9, pad=3)
    axl = fig.add_subplot(gs[-1, :])
    axl.axis("off")
    axl.legend(handles=[Patch(facecolor=C_VOID, edgecolor="0.3", label="void"),
                        Patch(facecolor=C_SURF, edgecolor="0.3", label="surface crack"),
                        Patch(facecolor=C_BODY, edgecolor="0.3", label="body crack")],
               loc="center", ncol=3, frameon=True, edgecolor="black", fontsize=9,
               handlelength=1.6, columnspacing=1.2, handletextpad=0.5, borderpad=0.45)
    out = os.path.join(OUTDIR, f"crack3d_{name}.png")
    fig.savefig(out, dpi=250, facecolor="white")
    plt.close(fig)
    print("->", out)


if __name__ == "__main__":
    for name, ids in GROUPS.items():
        sheet(name, ids)
