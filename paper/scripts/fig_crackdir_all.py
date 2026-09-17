"""Crack orientation classes at the last scan of every bead specimen, one
panel each, three by three, with the paper's legend.

    python scripts/fig_crackdir_all.py   -> figures/crackdir_all.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crack_split_tight import REG
from fig_crackdir_supp import crop, crop_all, FC, LEGEND
import paper_style as ps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "crackdir_all.png")
ORDER = ["G1", "G2", "G3", "G4", "G5", "A1", "A2", "A3", "A4"]


def main():
    tiles = []
    for pid in ORDER:
        tag, spam, scans = REG[pid]
        s = scans[-1]
        tiles.append((pid, s, crop(f"{FC}/{tag}/{tag}_scan{s:02d}_failure_class.png")))
    H = max(t[2].shape[0] for t in tiles)
    ps.apply(9.0)
    fig = plt.figure(figsize=(ps.TW, 0.36 * ps.TW * 3 + 0.55))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.13], hspace=0.22, wspace=0.05,
                          left=0.01, right=0.99, top=0.965, bottom=0.01)
    for k, (pid, s, im) in enumerate(tiles):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        ax.axis("off")
        can = np.ones((H, im.shape[1], 3))
        can[:im.shape[0]] = im
        ax.imshow(can)
        ax.set_title(f"{pid}, load step {s}", fontsize=9, pad=3)
    axl = fig.add_subplot(gs[3, :])
    axl.axis("off")
    axl.imshow(crop_all(LEGEND))
    fig.savefig(OUT, dpi=300, facecolor="white")
    print("->", OUT)


if __name__ == "__main__":
    main()
