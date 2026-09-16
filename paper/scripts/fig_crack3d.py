"""Void and crack in G3 at every scan, unloaded through the last load step.

G3 is the one specimen carried through four scans, so it is the only one in
which the first small increment of damage is resolved on its own, and the
sequence reads left to right.  The renders come from render_crack3d.py (classes of crack_split_tight.py),
which bounds the specimen by its beads plus the ice skin around them rather
than by the tube bore; the panels are top-aligned at their true relative size
so the compaction is visible, and one legend is drawn beneath in the paper's
font.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# renders re-made inside a bead-bounded specimen mask by rerender_g3_tight.py
SRC = (r"E:/RPTU-images/CT_images/Glass/Glass_100_1700_T7/Glass_1700_spam/"
       r"results_voidcrack/tight2")
OUT = r"C:/Users/cak7496/Desktop/first paper/MicroCT-paper/figures/crack3d.png"
TW = 390.0 / 72.27          # elsarticle preprint text width, inches

PANELS = [("unloaded", "voidcrack3d_scan01.png"),
          ("load step 2", "voidcrack3d_scan02.png"),
          ("load step 3", "voidcrack3d_scan03.png"),
          ("load step 4", "voidcrack3d_scan04.png")]

C_VOID, C_SURF, C_BODY = "#4fc3f7", "#f5a623", "#cc0000"


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
    """First ink block down the page, cropped tight sideways."""
    ink = img.mean(axis=2) < 0.97
    r0, r1 = runs(np.where(ink.any(axis=1))[0], 20)[0]
    cols = np.where(ink[r0:r1 + 1].any(axis=0))[0]
    return img[r0:r1 + 1, cols[0]:cols[-1] + 1]


imgs = [specimen(load(os.path.join(SRC, f))) for _, f in PANELS]
H = max(i.shape[0] for i in imgs)
W = max(i.shape[1] for i in imgs)
canvas = []
for i in imgs:                                  # top-aligned, centred sideways
    c = np.ones((H, W, 3), dtype=i.dtype)
    x = (W - i.shape[1]) // 2
    c[:i.shape[0], x:x + i.shape[1]] = i
    canvas.append(c)
    print("%-12s %s" % (PANELS[len(canvas) - 1][0], i.shape))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
})
n = len(canvas)
colw = (TW * 0.99 - 0.34 * (n - 1)) / n
panel_h = colw * H / W
fig_h = panel_h + 0.72
fig = plt.figure(figsize=(TW, fig_h))
gs = fig.add_gridspec(2, n, height_ratios=[panel_h, 0.40],
                      wspace=0.34, hspace=0.02,
                      left=0.005, right=0.995,
                      top=1 - 0.24 / fig_h, bottom=0.01)
axes = []
for j, (label, _) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(canvas[j])
    ax.axis("off")
    ax.set_title(label, fontsize=10, pad=4)
    axes.append(ax)

# one arrow from each panel to the next, across the gap, a third of the way
# down the column
for a, b in zip(axes[:-1], axes[1:]):
    a.annotate("", xy=(-0.06, 0.66), xycoords=b.transAxes,
               xytext=(1.06, 0.66), textcoords=a.transAxes,
               arrowprops=dict(arrowstyle="-|>", mutation_scale=14,
                               color="black", linewidth=1.5,
                               shrinkA=0, shrinkB=0),
               annotation_clip=False)

axl = fig.add_subplot(gs[1, :])
axl.axis("off")
axl.legend(handles=[Patch(facecolor=C_VOID, edgecolor="0.3", label="void"),
                    Patch(facecolor=C_SURF, edgecolor="0.3", label="surface crack"),
                    Patch(facecolor=C_BODY, edgecolor="0.3", label="body crack")],
           loc="center", ncol=3, frameon=True, edgecolor="black",
           fontsize=9, handlelength=1.6, columnspacing=1.2,
           handletextpad=0.5, borderpad=0.45)

fig.savefig(OUT, dpi=400, facecolor="white")
plt.close(fig)
print("->", OUT, "%.1f MB" % (os.path.getsize(OUT) / 1e6))
