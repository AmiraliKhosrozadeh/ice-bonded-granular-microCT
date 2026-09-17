"""Shared matplotlib style for the paper figures.

Everything is authored at the printed size: the text block of the elsarticle
preprint layout is 390 pt = 5.40 in, so a figure built at TW inches wide and
included with width=\\linewidth is reproduced 1:1 and its 11 pt labels really
are 11 pt on the page.

Times New Roman is set for the text AND for the mathtext, which is the part
that is easy to miss -- without the mathtext block the symbols in an axis
label ($q_3$, $\\varepsilon$, $\\tau$) fall back to DejaVu and sit visibly
apart from the words around them.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TW = 390.0 / 72.27          # \linewidth of the preprint layout, in inches
FONT = "Times New Roman"


def apply(base=11.0):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [FONT, "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT,
        "mathtext.it": FONT + ":italic",
        "mathtext.bf": FONT + ":bold",
        "mathtext.sf": FONT,
        "font.size": base,
        "axes.labelsize": base + 1,
        "axes.titlesize": base + 1,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base - 0.5,
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 400,
    })


def frame(ax, grid=True):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)


def save(fig, path, pdf=True):
    import os
    fig.savefig(path, dpi=400, bbox_inches="tight")
    if pdf and path.endswith(".png"):
        fig.savefig(path[:-4] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote " + os.path.basename(path))


# ---------------------------------------------------------------------------
# axis triad for the 3D renders: the paper's convention, z from the punch
# towards the support (down the panel), x across the panel, y into the page
# (drawn oblique); red, green, blue for x, y, z
TRIAD = (((1.0, 0.0), "$x$", "#d62728", (1.1, 0.0, "left", "center")),
         ((0.62, 0.62), "$y$", "#2ca02c", (0.7, 0.7, "left", "bottom")),
         ((0.0, -1.0), "$z$", "#1f77b4", (0.0, -1.1, "center", "top")))


def draw_triad(ax, fontsize=8, lw=1.2):
    ax.set_xlim(-0.3, 1.5)
    ax.set_ylim(-1.55, 1.05)
    ax.set_axis_off()
    for (dx, dy), name, col, (lx, ly, ha, va) in TRIAD:
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, mutation_scale=9))
        ax.text(lx, ly, name, ha=ha, va=va, fontsize=fontsize, color=col)


def add_triad(fig, left, bottom, width, height, fontsize=8):
    """the triad in a figure-fraction box of a matplotlib figure"""
    ax = fig.add_axes([left, bottom, width, height])
    draw_triad(ax, fontsize=fontsize)
    return ax


def triad_rgba(px, fontsize=8):
    """the triad as an RGBA array px pixels wide, for pasting into a render"""
    import numpy as np
    apply()
    # drawn at its printed size (0.6 in) so the arrows and the 8 pt labels
    # keep the proportions of the matplotlib version, then rasterised to px
    size_in = 0.6
    fig = plt.figure(figsize=(size_in, size_in), dpi=px / size_in)
    ax = fig.add_axes([0, 0, 1, 1])
    draw_triad(ax, fontsize=fontsize, lw=1.2)
    fig.patch.set_alpha(0.0)
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return a


def paste_triad(png, frac=0.10, margin=0.02, fontsize=8, band=False):
    """paste the triad into the lower-left corner of a render PNG, in place;
    frac is the triad width as a fraction of the image width; with band the
    image is first extended by a white band at the bottom for it"""
    from PIL import Image
    im = Image.open(png).convert("RGBA")
    W, H = im.size
    px = int(round(frac * W))
    tri = Image.fromarray(triad_rgba(px, fontsize=fontsize))
    if band:
        tall = Image.new("RGBA", (W, H + px), (255, 255, 255, 255))
        tall.paste(im, (0, 0))
        im, H = tall, H + px
    x0 = int(round(margin * W))
    y0 = H - px - (0 if band else int(round(margin * W)))
    im.alpha_composite(tri, (x0, y0))
    im.convert("RGB").save(png)
    print("  triad -> " + png)
