"""Legends for fig:crackdir and its appendix twin, in the paper's font.

The renders carry no legend of their own.  The colours are the ones the
renders use for the three dip bands.  G1 at its second load step, the panel
in the main text, has no horizontal component, so its legend carries only
the two classes that appear; the appendix figure, whose alumina panel does
have one, carries all three.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import paper_style as ps

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "figures")
KEYS = [("#e00000", "vertical crack"), ("#ef6c00", "diagonal crack"),
        ("#1565c0", "horizontal crack")]

ps.apply(10.0)
for name, keys in (("crackdir_legend.png", KEYS),
                   ("crackdir_legend_g1.png", KEYS[:2])):
    fig = plt.figure(figsize=(ps.TW * 0.86, 0.34))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.legend(handles=[Patch(facecolor=c, edgecolor="0.15", label=t) for c, t in keys],
              loc="center", ncol=len(keys), frameon=True, edgecolor="black",
              handlelength=1.5, columnspacing=1.6, handletextpad=0.5,
              borderpad=0.45)
    fig.savefig(os.path.join(FIG, name), dpi=400, facecolor="white")
    plt.close(fig)
    print("  wrote " + name)
