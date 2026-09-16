"""Regenerate the three paper particle-size-distribution figures.

Outputs (PNG + PDF) into MicroCT-paper/figures/:
    psd_glass         G1, G3, G4   vs Camsizer XT (x_area)
    psd_alumina       A1-A4        vs supplier nominal grades
    sand_psd_camsizer S1, S2, S3   vs Camsizer (x_Fe,min)

All three share one convention: volume density q3 (% um^-1) on the LEFT axis
(thick solid lines, Camsizer as grey bars) and cumulative undersize Q3 (%) on
the RIGHT axis (thin lines, Camsizer dashed).  Volume weighting is d^3 for the
bead pipelines and the segmented grain volume for the sand pipeline.

Run:  python scripts/psd_paper_figs.py
"""
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(HERE), "figures")
VOX_UM = 24.7660229

FS, LW = 26, 3
CAM_FACE = "0.80"
Y_Q3 = "Cumulative undersize $Q_3$ (%)"
Y_q3 = "Normalized volume density $q_3$ (% $\\mu$m$^{-1}$)"
X_LBL = "Equivalent diameter ($\\mu$m)"


def apply_style():
    plt.rcParams.update({
        "font.size": FS, "axes.labelsize": FS, "axes.labelweight": "bold",
        "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS - 4,
        "lines.linewidth": LW, "axes.linewidth": 2.2, "axes.edgecolor": "black",
        "xtick.major.width": 2.0, "ytick.major.width": 2.0,
        "xtick.major.size": 8, "ytick.major.size": 8,
        "xtick.major.pad": 8, "ytick.major.pad": 8,
        "grid.linewidth": 0.8, "grid.alpha": 0.35,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.dpi": 260, "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------- data readers
def glass_diam(spec):
    p = os.path.join(r"E:/RPTU-images/CT_images/Glass", spec,
                     "scan01_dragonfly", "results", "bead_measurements.csv")
    with open(p, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    d = np.array([float(r["eq_diameter_um"]) for r in rows])
    return d, d ** 3


def alumina_diam(spec):
    p = os.path.join(r"E:/RPTU-images/CT_images/Alumina/pyalumina/results",
                     spec, "scan01", "stage2", "meta.npz")
    d = np.load(p)["d_eq_um"].astype(float)
    return d, d ** 3


def sand_diam(resdir):
    base = os.path.join(r"E:/RPTU-images/CT_images/Sand/pipeline/dl", resdir)
    d = np.load(os.path.join(base, "feret_diam.npy")).astype(float)
    w = np.load(os.path.join(base, "feret_vols.npy")).astype(float) * VOX_UM ** 3
    return d, w


def read_xle(path):
    """Return (lo, hi, p3_pct) volume-frequency per size class."""
    with open(path, encoding="utf-16") as fh:
        lines = fh.readlines()
    head = next(i for i, l in enumerate(lines) if l.strip().startswith("Size class"))
    cols = [c.strip() for c in lines[head].rstrip("\n").split("\t")]
    lo, hi, val = [], [], []
    for ln in lines[head + 1:]:
        q = ln.rstrip("\n").split("\t")
        if len(q) < 3:
            continue
        try:
            a, b = float(q[0]), float(q[1])
            rest = [float(x) for x in q[2:]]
        except ValueError:
            continue
        lo.append(a); hi.append(b); val.append(rest)
    lo, hi = np.array(lo), np.array(hi)
    ncol = min(len(r) for r in val)
    val = np.array([r[:ncol] for r in val])
    if "p3 [%]" in cols:                      # volume basis exported directly
        p3 = val[:, cols.index("p3 [%]") - 2]
    else:                                     # number basis -> weight by x^3
        p0 = val[:, cols.index("p0") - 2]
        p3 = p0 * (0.5 * (lo + hi)) ** 3
    p3 = 100.0 * p3 / p3.sum()
    return lo, hi, p3


def camsizer(pattern):
    """Average replicate .xle runs onto the first run's size classes."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    lo, hi, p3 = read_xle(files[0])
    stack = [p3]
    for f in files[1:]:
        l2, h2, p2 = read_xle(f)
        stack.append(np.interp(0.5 * (lo + hi), 0.5 * (l2 + h2), p2))
    p3 = np.mean(stack, axis=0)
    p3 = 100.0 * p3 / p3.sum()
    width = hi - lo
    q3 = np.where(width > 0, p3 / np.maximum(width, 1e-12), 0.0)
    Q3 = np.cumsum(p3)
    return 0.5 * (lo + hi), width, q3, Q3, len(files)


# ---------------------------------------------------------------- curve makers
def q3_kde(d, w, grid, bw=1.0):
    kde = gaussian_kde(d, weights=w)
    kde.set_bandwidth(kde.factor * bw)
    return kde(grid) * 100.0


def Q3_exact(d, w):
    o = np.argsort(d)
    return d[o], 100.0 * np.cumsum(w[o]) / w[o].sum()


def d50(d, w):
    ds, Q = Q3_exact(d, w)
    return float(np.interp(50.0, Q, ds))


# ---------------------------------------------------------------- figure maker
def make_figure(series, xlim, out, cam=None, cam_label="Camsizer",
                bands=None, band_label="nominal size", ncol=3,
                xlabel=X_LBL, norm=False, q3_lw=LW + 0.5,
                cum_lw=LW - 1.2, cum_alpha=1.0, cum_zorder=3):
    apply_style()
    ylab = Y_q3
    scale = ((lambda v: 100.0 * v / max(v.max(), 1e-30)) if norm
             else (lambda v: v))
    grid = np.linspace(xlim[0], xlim[1], 800)
    fig, axL = plt.subplots(figsize=(15, 12))
    axR = axL.twinx()
    handles = []

    if bands:
        for b0, b1 in bands:
            axL.axvspan(b0, b1, color=CAM_FACE, alpha=0.75, zorder=0)
        handles.append(Patch(facecolor=CAM_FACE, alpha=0.75, label=band_label))

    if cam is not None:
        mid, width, q3c, Q3c, _ = cam
        keep = (mid >= xlim[0] - width.max()) & (mid <= xlim[1] + width.max())
        axL.bar(mid[keep], scale(q3c[keep]), width=width[keep] * 0.92, color=CAM_FACE,
                edgecolor="0.55", linewidth=0.8, zorder=0)
        axR.plot(mid, Q3c, color="0.35", linestyle="--", linewidth=LW - 0.5, zorder=2)
        handles.append(Patch(facecolor=CAM_FACE, edgecolor="0.55",
                             label=cam_label + " $q_3$"))

    for label, d, w, color in series:
        axL.plot(grid, scale(q3_kde(d, w, grid)), color=color,
                 linewidth=q3_lw, zorder=3)
        ds, Q = Q3_exact(d, w)
        axR.plot(ds, Q, color=color, linewidth=cum_lw,
                 alpha=cum_alpha, zorder=cum_zorder)
        handles.append(Line2D([0], [0], color=color, linewidth=LW, label=label))
        print("    %-12s n=%7d  D50=%7.1f um" % (label, d.size, d50(d, w)))

    if cam is not None:
        handles.append(Line2D([0], [0], color="0.35", linestyle="--",
                              linewidth=LW - 0.5, label=cam_label + " $Q_3$"))

    axL.set_xlabel(xlabel, labelpad=10)
    axL.set_ylabel(ylab, labelpad=10)
    axR.set_ylabel(Y_Q3, labelpad=10)
    axL.set_xlim(xlim[0], xlim[1])
    axL.set_ylim(bottom=0)
    axR.set_ylim(0, 100)
    for sp in list(axL.spines.values()) + list(axR.spines.values()):
        sp.set_visible(True)
        sp.set_linewidth(2.2)
        sp.set_edgecolor("black")
    axL.grid(True, which="major", linewidth=0.8, alpha=0.35)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=ncol, frameon=True, edgecolor="black", fontsize=FS - 4)
    fig.subplots_adjust(bottom=0.24)

    png = os.path.join(FIGDIR, out + ".png")
    pdf = os.path.join(FIGDIR, out + ".pdf")
    for p in (png, pdf):
        if os.path.exists(p):
            os.remove(p)
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("  wrote " + out + ".png + .pdf")


def main():
    print("glass:")
    g1 = glass_diam("Glass_75_1700_T5_HR")
    g3 = glass_diam("Glass_100_1700_T7")
    g4 = glass_diam("Glass_100_1700_T5_HR")
    cam_g = camsizer(r"D:/Rolling/Camsizer/Neuer Ordner/test[123]/*x_area_*.xle")
    print("    Camsizer x_area over %d runs, D50=%.1f um"
          % (cam_g[4], np.interp(50.0, cam_g[3], cam_g[0])))
    make_figure([("Glass G1", g1[0], g1[1], "#1f77b4"),
                 ("Glass G3", g3[0], g3[1], "#d95f4c"),
                 ("Glass G4", g4[0], g4[1], "#2ca02c")],
                (1450, 2000), "psd_glass", cam=cam_g, norm=True)

    print("alumina:")
    a1 = alumina_diam("100_1800_T5")
    a2 = alumina_diam("175_1800_T5")
    a3 = alumina_diam("75_1000_T5")
    a4 = alumina_diam("75_1800_T7")
    make_figure([("Alumina A1", a1[0], a1[1], "#1f77b4"),
                 ("Alumina A2", a2[0], a2[1], "#d95f4c"),
                 ("Alumina A3", a3[0], a3[1], "#2ca02c"),
                 ("Alumina A4", a4[0], a4[1], "#8b5fbf")],
                (700, 2300), "psd_alumina",
                bands=[(970, 1030), (1746, 1854)],
                q3_lw=LW + 1.5, cum_zorder=2)

    print("sand:")
    cam_s = camsizer(r"D:/Rolling/Sand/Camsizer/*xFemin_00*.xle")
    print("    Camsizer xFemin over %d runs, D50=%.1f um"
          % (cam_s[4], np.interp(50.0, cam_s[3], cam_s[0])))
    s1 = sand_diam("results_v2_100_500_T5_01")
    s2 = sand_diam("results_v2_25mm_T5_01")
    s3 = sand_diam("results_75_200_T5_01")
    make_figure([("Sand S1", s1[0], s1[1], "#1f77b4"),
                 ("Sand S2", s2[0], s2[1], "#d95f4c"),
                 ("Sand S3", s3[0], s3[1], "#2ca02c")],
                (0, 700), "sand_psd_camsizer", cam=cam_s,
                xlabel="Particle size ($\mu$m)")


if __name__ == "__main__":
    main()
