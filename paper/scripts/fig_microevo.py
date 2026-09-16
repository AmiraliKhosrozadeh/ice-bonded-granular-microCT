"""Fig. microevo: how void fraction, specific surface, Kozeny-Carman
permeability and ice tortuosity change with the shortening of the column,
each specimen referenced to its own unloaded scan, with one straight line
per material through the origin.

"Load step" is not a physical axis: step 2 of one specimen is a 5 %
shortening and of another 21 %.  Against the measured column shortening the
points of all specimens lie on one axis and the material lines say how much
each quantity changes per unit shortening.  Absolute values start from very
different places (an alumina packing is an order of magnitude more porous
than a glass one before any load), so every specimen is referenced to its
unloaded scan.  A specimen with a value at only one scan is not drawn in
that panel, which is why S1 and S2 are absent from the tortuosity panel: no
subvolume converged on their loaded scans.

Authored at the printed width so the labels are the size they will be on the
page.  Permeability is on a log axis, spanning three decades across the set.

The tortuosity panel carries a broken axis.  G1 reaches 5.74 at its third
load step while the other eleven specimens live between 1.45 and 2.67, and on
one continuous axis that single point flattens everything else into a band a
few millimetres tall.  The break keeps the outlier visible and gives the rest
of the set the whole lower axis.

Run:  python scripts/fig_microevo.py
"""
import csv
import io
import os

import matplotlib.pyplot as plt

import paper_style as ps

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "section36_core.csv")
DST = os.path.join(os.path.dirname(HERE), "figures", "microstructure_evolution.png")

COL = {"glass": ["#0d2f4f", "#1f4e79", "#3a72a4", "#5d95c6", "#8ab6de"],
       "alumina": ["#7f2b28", "#a83c39", "#c0504d", "#d47b78"],
       "sand": ["#1b5e20", "#2e7d32", "#66a06a"]}
MRK = {"G1": "o", "G2": "s", "G3": "^", "G4": "D", "G5": "v",
       "A1": "o", "A2": "s", "A3": "^", "A4": "D",
       "S1": "o", "S2": "s", "S3": "^"}
ORDER = ["G1", "G2", "G3", "G4", "G5", "A1", "A2", "A3", "A4", "S1", "S2", "S3"]

LO, HI = (1.35, 2.85), (5.4, 6.1)      # the two halves of the broken axis


def segments(xs, ys):
    """Split into runs of consecutive load steps.

    G1 has no tortuosity at step 2, and joining step 1 straight to step 3
    would draw a trend through a step that was never measured.
    """
    out, cur = [], []
    for x, y in zip(xs, ys):
        if cur and x != cur[-1][0] + 1:
            out.append(cur)
            cur = []
        cur.append((x, y))
    if cur:
        out.append(cur)
    return out


def load():
    d = {}
    for r in csv.DictReader(io.open(SRC, encoding="utf-8")):
        d.setdefault(r["id"], dict(material=r["material"], s=[], e=[],
                                   ts=[], t=[]))
        d[r["id"]]["s"].append(int(r["stage"]))
        d[r["id"]]["e"].append(float(r["eps"]))
        d[r["id"]].setdefault("sv", []).append(float(r["SV_per_mm"]))
        d[r["id"]].setdefault("k", []).append(float(r["k_KC_m2"]))
        d[r["id"]].setdefault("x", []).append(float(r["shortening"]))
        if r["tau_ice"]:
            d[r["id"]]["ts"].append(int(r["stage"]))
            d[r["id"]]["t"].append(float(r["tau_ice"]))
            d[r["id"]].setdefault("tok", []).append(int(r["n_cubes_ok"]) >= 2)
    return d


MAT_COL = {"glass": "#1f4e79", "alumina": "#a83c39", "sand": "#2e7d32"}
# points drawn hollow and left out of the material fit, (specimen, step, panel)
FLAG = set()
# specimens left out of the figure altogether
SKIP = {"S3"}
MAT_NAME = {"glass": "glass", "alumina": r"$\gamma$-alumina", "sand": "sand"}


def change(v, key, log=False):
    """Change from the unloaded scan, against the column shortening at each
    scan, for one specimen."""
    import math
    if key == "t":
        steps, vals = v["ts"], v["t"]
        if not vals or steps[0] != 1:
            return [], []
        xs = [v["x"][v["s"].index(st)] for st in steps]
    else:
        xs, vals = v["x"], v[key]
    ref = vals[0]
    ys = [math.log10(x / ref) for x in vals] if log else [x - ref for x in vals]
    return xs, ys


def main():
    ps.apply(base=10.5)
    d = load()
    fig = plt.figure(figsize=(ps.TW, 0.92 * ps.TW))
    plt.rcParams["axes.labelsize"] = 10.0
    gs = fig.add_gridspec(2, 2, wspace=0.40, hspace=0.30, bottom=0.13,
                          top=0.98, left=0.11, right=0.98)
    axes = {k: fig.add_subplot(gs[i // 2, i % 2])
            for i, k in enumerate(("e", "sv", "k", "t"))}
    label = {"e": r"change in void fraction $\Delta\varepsilon_{\mathrm{c}}$", "sv": r"change in specific surface $\Delta S_V$ (mm$^{-1}$)", "k": r"permeability ratio $k / k_{\mathrm{unloaded}}$", "t": r"change in ice tortuosity $\Delta\tau_{\mathrm{ice}}$"}

    for key, ax in axes.items():
        pts = {}
        for pid in ORDER:
            if pid in SKIP:
                continue
            v = d[pid]
            xs, ys = change(v, key, log=(key == "k"))
            if len(xs) < 2:
                continue
            c = MAT_COL[v["material"]]
            steps = v["ts"] if key == "t" else v["s"]
            ok = [(v["tok"][i] if key == "t" else True)
                  and (pid, st, key) not in FLAG
                  for i, st in enumerate(steps)]
            ax.plot(xs, ys, color=c, alpha=0.45, linewidth=1.0)
            for x, y, good in zip(xs, ys, ok):
                # a tortuosity from a single converged subvolume is drawn
                # hollow and left out of the material line
                ax.plot([x], [y], marker=MRK[pid], markersize=4, color=c,
                        alpha=0.45, mfc=c if good else "white", linestyle="none")
                if x > 0 and good:
                    pts.setdefault(v["material"], []).append((x, y))
        # one straight line per material through the origin, least squares,
        # so the slope is the change per unit shortening
        for mat, p in pts.items():
            import numpy as np
            x = np.array([a for a, _ in p]); y = np.array([b for _, b in p])
            m = float((x * y).sum() / (x * x).sum())
            xx = np.linspace(0, x.max(), 20)
            ax.plot(xx, m * xx, color=MAT_COL[mat], linewidth=2.4, zorder=5,
                    label=MAT_NAME[mat] if key == "e" else None)
        ax.axhline(0, color="0.5", linewidth=0.7, zorder=0)
        ax.set_ylabel(label[key])
        ax.set_xlim(-0.01, 0.39)
        ax.set_xticks([0, 0.1, 0.2, 0.3])
        ps.frame(ax)
    axes["k"].set_yticks([0, 1, 2, 3])
    axes["k"].set_yticklabels([r"$10^{0}$", r"$10^{1}$", r"$10^{2}$", r"$10^{3}$"])
    for k_ in ("k", "t"):
        axes[k_].set_xlabel("column shortening")
    for k_, t in zip(("e", "sv", "k", "t"), ("(a)", "(b)", "(c)", "(d)")):
        axes[k_].text(0.02, 0.96, t, transform=axes[k_].transAxes, ha="left",
                      va="top", fontsize=10.5)

    h, l = axes["e"].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.06),
               frameon=False, ncol=3, handlelength=2.0, columnspacing=2.0)
    ps.save(fig, DST)


if __name__ == "__main__":
    main()
