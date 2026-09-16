"""Histogram of the crack-at-face share over every answerable throat.

For each throat the share is the larger of the two per-face values, the
fraction of the crack inside the throat that lies in the 1-3 voxel shell
against a grain surface.  Cohesive throats, the ones that keep ice on both
faces at C_HI = 0.80, are stacked at the bottom so the reader can see that they
occupy the low tail and that the rest of the set runs on continuously.

The values are read once from the census workbook on E: and cached under
data/, so the figure rebuilds without the drive.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import paper_style as ps

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "crack_at_face.csv")
XLSX = ("E:/RPTU-images/CT_images/paper_figures/bond_failure/"
        "all_specimens_bond_failure.xlsx")
OUT = os.path.join(os.path.dirname(HERE), "figures", "crack_at_face.png")

C_COH, C_ADH = "#4a7fb5", "#c8553d"

if not os.path.exists(CACHE):
    d = pd.read_excel(XLSX, "every_throat")
    d["share"] = d[["crack_at_A", "crack_at_B"]].max(axis=1)
    d["cohesive"] = (d.hi >= 0.80) & (d.lo >= 0.80)
    d[["specimen", "bead_a", "bead_b", "share", "cohesive"]].to_csv(
        CACHE, index=False)
d = pd.read_csv(CACHE)
coh, adh = d[d.cohesive].share.values, d[~d.cohesive].share.values
print("n = %d, cohesive %d, adhesive %d, median share %.2f, cohesive max %.2f"
      % (len(d), len(coh), len(adh), d.share.median(), coh.max()))

ps.apply(10.0)
fig, ax = plt.subplots(figsize=(ps.TW * 0.62, 2.05))
bins = np.arange(0, 1.0001, 0.05)
ax.hist([coh, adh], bins=bins, stacked=True, color=[C_COH, C_ADH],
        edgecolor="white", linewidth=0.5, label=["cohesive", "adhesive"])
ax.set_xlim(0, 1)
ax.set_xlabel("Share of throat crack at a grain face")
ax.set_ylabel("Throats")
ax.text(-0.13, 1.02, "(g)", transform=ax.transAxes, fontsize=10.5,
        ha="left", va="bottom")
ps.frame(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
          handlelength=1.3, columnspacing=1.6)
fig.subplots_adjust(left=0.14, right=0.98, top=0.93, bottom=0.36)
ps.save(fig, OUT)
