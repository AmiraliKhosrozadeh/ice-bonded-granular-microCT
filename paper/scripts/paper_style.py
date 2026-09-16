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
