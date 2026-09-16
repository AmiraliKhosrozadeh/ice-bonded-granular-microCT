"""One throat of each kind, in three dimensions and in one grey-value plane.

Columns are a cohesive throat, an adhesive throat that released part of one
face and the one throat in the set that released a face completely.  The top
row is the 3D render of the two beads with the ice bridge and the crack in the
throat, the bottom row one section plane through the same throat with the
beads and the measured throat outlined.  The renders carry a baked legend and
the sections a baked scale bar; the legend is cropped off and redrawn once in
the paper's font, the bar is kept and labelled.

"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "figures")
OUT = os.path.join(FIG, "bond_sections.png")
TW = 390.0 / 72.27
BAR = r"500 $\mu$m"          # stated in the render index, not measured here

B = "E:/RPTU-images/CT_images/paper_figures/bond_failure"
# (column label, sections file, index of the section plane to show)
# All three from A3.  The author chose the throats by eye; the census has the
# first at coverage (0.70, 0.45), the second at (0.73, 0.47) and the third at
# (0.96, 0.15).
PANELS = [
    ("cohesive",
     B + "/Alumina_75_1000_T5/individual_bonds/answerable/mixed/"
         "a067_beads0335-1525_coverHI70_coverLO45_sections.png", 0),
    ("adhesive (mixed mode)",
     B + "/Alumina_75_1000_T5/individual_bonds/answerable/mixed/"
         "a066_beads2091-2262_coverHI73_coverLO47_sections.png", 1),
    ("adhesive",
     B + "/Alumina_75_1000_T5/individual_bonds/answerable/adhesive/"
         "a001_beads0372-1554_coverHI96_coverLO15_sections.png", 0),
]

# swatch colours read back off the original legends
KEYS = [("#1565c0", "bead A"), ("#2e7d32", "bead B"),
        ("#26c6da", "ice bridge in the throat"), ("#e00000", "crack"),
        ("#ab47bc", "measured throat")]


def load(name):
    a = plt.imread(name)
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


def tight(img):
    ink = img.mean(axis=2) < 0.97
    r = np.where(ink.any(axis=1))[0]
    c = np.where(ink.any(axis=0))[0]
    return img[r[0]:r[-1] + 1, c[0]:c[-1] + 1]


def section(name, k):
    """Panel k of a sections strip, cropped away from its baked legend."""
    img = load(name)
    ink = img.mean(axis=2) < 0.97
    rb = runs(np.where(ink.any(axis=1))[0], 20)
    r0, r1 = rb[0]
    cb = runs(np.where(ink[r0:r1 + 1].any(axis=0))[0], 15)
    c0, c1 = cb[k]
    return tight(img[r0:r1 + 1, c0:c1 + 1])


def render(name):
    """The 3D render above its baked legend box.

    The box is framed in black, so its top edge is the first row that is
    more than half black; nothing on a bead surface is that straight.
    """
    img = load(name.replace("_sections.png", "_3d.png"))
    black = (img.max(axis=2) < 0.2).mean(axis=1)
    top = np.where(black > 0.5)[0]
    if top.size:
        img = img[:top[0] - 6]
    return tight(img)


def scalebar(img):
    """Where the baked bar sits, as (row, x0, x1)."""
    from scipy import ndimage

    h, w = img.shape[:2]
    flat = (img.max(axis=2) - img.min(axis=2)) < 0.08
    v = img.mean(axis=2)
    best = None
    for m in (flat & (v > 0.90), flat & (v < 0.14)):
        m = m.copy()
        m[:int(h * 0.55)] = False
        lab, n = ndimage.label(m)
        for sl in ndimage.find_objects(lab):
            hh = sl[0].stop - sl[0].start
            ww = sl[1].stop - sl[1].start
            if hh < 2 or ww < w * 0.10 or ww / hh < 5 or m[sl].mean() < 0.85:
                continue
            if best is None or ww > best[0]:
                best = (ww, (sl[0].start + sl[0].stop) // 2,
                        sl[1].start, sl[1].stop - 1)
    if best is None:
        return None
    return best[1], best[2], best[3]


def crop_corners(img):
    """Cut the panel down to a rectangle free of the black fill outside the
    reconstructed window.

    That fill is pure black and reaches the panel rim; air inside the
    specimen is dark grey and does not.  Edge lines are shaved off greedily,
    each time the one carrying the most fill, until none is left.
    """
    from scipy import ndimage

    black = img.max(axis=2) < 0.03
    lab, n = ndimage.label(black)
    m = 6
    rim = np.zeros_like(black)
    rim[:m], rim[-m:], rim[:, :m], rim[:, -m:] = True, True, True, True
    edge = set(np.unique(lab[rim]))
    edge.discard(0)
    fill = np.isin(lab, list(edge))
    t, b, l, r = 0, img.shape[0], 0, img.shape[1]
    while fill[t:b, l:r].any():
        sub = fill[t:b, l:r]
        cand = [(sub[0].mean(), "t"), (sub[-1].mean(), "b"),
                (sub[:, 0].mean(), "l"), (sub[:, -1].mean(), "r")]
        side = max(cand)[1]
        if side == "t":
            t += 1
        elif side == "b":
            b -= 1
        elif side == "l":
            l += 1
        else:
            r -= 1
    return img[t:b, l:r]


def pad_to(img, ratio):
    """White-pad an image to width/height = ratio, centred.

    Returns the canvas and the (row, column) offset of the image in it.
    """
    h, w = img.shape[:2]
    if w / h < ratio:
        W = int(round(h * ratio))
        out = np.ones((h, W, 3), dtype=img.dtype)
        x = (W - w) // 2
        out[:, x:x + w] = img
        return out, 0, x
    H = int(round(w / ratio))
    out = np.ones((H, w, 3), dtype=img.dtype)
    y = (H - h) // 2
    out[y:y + h] = img
    return out, y, 0


tops, bots, bars, labels = [], [], [], []
for label, name, k in PANELS:
    tops.append(render(name))
    sec = section(name, k)
    r, x0, x1 = scalebar(sec)            # 500 um in pixels, before cropping
    bars.append(x1 - x0)
    bots.append(crop_corners(sec[8:-8, 8:-8]))     # inside the baked frame
    labels.append(label)
    print("%-24s render %s  section %s  bar %d px" % (label, tops[-1].shape,
                                                      bots[-1].shape, bars[-1]))

# every column the same width; the render row square, the section row as tall
# as its tallest panel
tops = [pad_to(t, 1.0)[0] for t in tops]
hs = max(b.shape[0] / b.shape[1] for b in bots)
# where the bar goes on each section: over the baked one if it survived the
# crop, else bottom left; then the panels are padded to a common shape
pos = []
for b in bots:
    found = scalebar(b)
    if found is None:
        found = (int(b.shape[0] * 0.93), int(b.shape[1] * 0.05), 0)
    pos.append(found[:2])
padded = [pad_to(b, 1.0 / hs) for b in bots]
bots = [c for c, _, _ in padded]
pos = [(r + dy, x + dx) for (r, x), (_, dy, dx) in zip(pos, padded)]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
})
n = len(PANELS)
colw = (TW * 0.99 - 0.06 * (n - 1)) / n
h_top, h_bot, lh, gap = colw, colw * hs, 0.34, 0.30
fig_h = h_top + h_bot + lh + 2 * gap
fig = plt.figure(figsize=(TW, fig_h))
gs = fig.add_gridspec(3, n, height_ratios=[h_top, h_bot, lh],
                      wspace=0.04, hspace=gap / ((h_top + h_bot + lh) / 3.0),
                      left=0.005, right=0.995, top=0.995, bottom=0.005)

for j in range(n):
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(tops[j])
    ax.axis("off")
    ax.text(0.5, -0.03, "(%s) %s" % ("abc"[j], labels[j]),
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5)

    ax = fig.add_subplot(gs[1, j])
    ax.imshow(bots[j])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
    # a fresh scale bar of the same length, drawn over the baked one
    r, x = pos[j]
    ax.plot([x, x + bars[j]], [r, r], color="white", linewidth=4.0,
            solid_capstyle="butt")
    ax.plot([x, x + bars[j]], [r, r], color="black", linewidth=2.0,
            solid_capstyle="butt")
    ax.text(x + bars[j] / 2.0, r - 9, BAR, ha="center", va="bottom",
            fontsize=9, color="white",
            path_effects=[pe.withStroke(linewidth=2.2, foreground="black")])
    ax.text(0.5, -0.03, "(%s)" % "def"[j], transform=ax.transAxes,
            ha="center", va="top", fontsize=9.5)

axl = fig.add_subplot(gs[2, :])
axl.axis("off")
axl.legend(handles=[Patch(facecolor=c, edgecolor="0.15", label=t)
                    for c, t in KEYS],
           loc="center", ncol=5, frameon=True, edgecolor="black",
           fontsize=8.0, handlelength=1.4, columnspacing=1.0,
           handletextpad=0.45, borderpad=0.45)

fig.savefig(OUT, dpi=400, facecolor="white")
plt.close(fig)
print("->", OUT, "%.2f MB" % (os.path.getsize(OUT) / 1e6))
