"""Collect the model-comparison runs and make publication plots:
  cmp_models/model_error_comparison.png   -- grouped bars: min-Feret RMS vs
                                             xFemin, per specimen, v1 vs v2
  cmp_models/psd_v1_vs_v2_<spec>.png      -- per specimen, CT Q3 (v1, v2) vs
                                             Camsizer xFemin (shows WHY v2 wins:
                                             v1's merged grains inflate the
                                             coarse tail; v2 splits them)
One plot per file, legend below, full box, large fonts, PDF beside every PNG.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import experimental_psd as exp
import plot_style as ps
import matplotlib.pyplot as plt

VOX = 24.7660229
OUT = "cmp_models"
SPECS = ["100_500_T5_01", "25mm_T5_01", "75_200_T5_01"]
SPEC_LABEL = {"100_500_T5_01": "Sand S1", "25mm_T5_01": "Sand S3",
              "75_200_T5_01": "Sand S2"}
MODELS = ["nodl", "v1", "v2", "ps3d", "cpsam"]
MODEL_LABEL = {"nodl": "no-DL (EDT watershed only)",
               "v1": "v1 U-Net (uniform-sand synth)",
               "v2": "v2 U-Net (ice-neck synth)",
               "ps3d": "ParticleSeg3D (external, pretrained)",
               "cpsam": "Cellpose-SAM (external, zero-shot)"}
MODEL_COLOR = {"nodl": "#9467bd", "v1": "#7f7f7f",
               "v2": "#d62728", "ps3d": "#2ca02c", "cpsam": "#ff7f0e"}
XMAX = 800

# Camsizer xFemin reference (load once)
mid_a, Q3_a, _, _ = exp.load_camsizer("xFemin")
p_a = exp.percentiles(mid_a, Q3_a)


def q3(d, w):
    o = np.argsort(d); ds = d[o]; Q = 100.0 * np.cumsum(w[o]) / w.sum()
    return ds, Q, {p: float(np.interp(p, Q, ds)) for p in (10, 50, 90)}


def load_run(spec, mdl):
    od = os.path.join(OUT, f"{spec}__{mdl}")
    fp = os.path.join(od, "feret_diam.npy")
    if not os.path.exists(fp):
        return None
    df = np.load(fp)                                          # um
    if df.size < 5:                                           # model failed on this scan
        return None
    vols = np.load(os.path.join(od, "feret_vols.npy"))        # voxel counts
    vol_um = vols.astype(float) * VOX ** 3
    ds, Q, p = q3(df, vol_um)
    pg = np.linspace(5, 95, 91)
    rms = float(np.sqrt(np.mean(((np.interp(pg, Q, ds) -
                np.interp(pg, Q3_a, mid_a)) /
                np.interp(pg, Q3_a, mid_a)) ** 2)) * 100.0)
    return ds, Q, p, rms, df.size


# ---- gather ----------------------------------------------------------------
res = {}
print(f"Camsizer xFemin D10/50/90 = {p_a[10]:.0f}/{p_a[50]:.0f}/{p_a[90]:.0f} um\n")
print(f"{'specimen':16s} {'model':4s} {'grains':>8s} {'D50':>5s} {'D90':>5s} {'RMS%':>6s}")
for spec in SPECS:
    for mdl in MODELS:
        r = load_run(spec, mdl)
        res[(spec, mdl)] = r
        if r is None:
            print(f"{spec:16s} {mdl:4s} {'--':>8s} {'--':>5s} {'--':>5s}   FAILED")
            continue
        ds, Q, p, rms, ng = r
        print(f"{spec:16s} {mdl:4s} {ng:8d} {p[50]:5.0f} {p[90]:5.0f} {rms:6.1f}")

# ---- plot 1: grouped RMS bars ---------------------------------------------
ps.apply_style()
fig, ax = plt.subplots(figsize=(13, 9))
x = np.arange(len(SPECS)); n = len(MODELS); w = 0.8 / n
for i, mdl in enumerate(MODELS):
    vals = [res[(s, mdl)][3] if res[(s, mdl)] is not None else np.nan for s in SPECS]
    ax.bar(x + (i - (n - 1) / 2.0) * w, vals, w, color=MODEL_COLOR[mdl],
           edgecolor="black", linewidth=1.5, label=MODEL_LABEL[mdl])
ax.set_xticks(x); ax.set_xticklabels([SPEC_LABEL[s] for s in SPECS])
ax.set_xlabel("Specimen", labelpad=10)
ax.set_ylabel("RMS error vs $x_{Fe,min}$ (%)", labelpad=10)
ps.style_axes(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
          frameon=True, edgecolor="gray")
ps.save_fig(fig, os.path.join(OUT, "model_error_comparison.png"))
print(f"\nwrote {OUT}/model_error_comparison.png")

# ---- plot 2: PSD overlay per specimen (why v2 wins) -----------------------
for spec in SPECS:
    fig, ax = plt.subplots(figsize=(14, 9))
    for mdl in MODELS:
        r = res[(spec, mdl)]
        if r is None:
            continue
        ds, Q, p, rms, ng = r
        ax.plot(ds, Q, color=MODEL_COLOR[mdl], linewidth=ps.LW + (mdl == "v2"),
                label=f"{MODEL_LABEL[mdl]}  ($D_{{50}}$={p[50]:.0f}, RMS {rms:.1f}%)")
    ax.plot(mid_a, Q3_a, color="#1f77b4", linewidth=ps.LW, linestyle="--",
            label=f"Camsizer $x_{{Fe,min}}$  ($D_{{50}}$={p_a[50]:.0f} $\\mu$m)")
    for pct in (10, 50, 90):
        ax.axhline(pct, color="dimgray", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Particle size ($\\mu$m)", labelpad=10)
    ax.set_ylabel("Cumulative undersize $Q_3$ (%)", labelpad=10)
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ps.style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1,
              frameon=True, edgecolor="gray")
    ps.save_fig(fig, os.path.join(OUT, f"psd_models_{spec}.png"))
    print(f"wrote {OUT}/psd_models_{spec}.png")

print("done")
