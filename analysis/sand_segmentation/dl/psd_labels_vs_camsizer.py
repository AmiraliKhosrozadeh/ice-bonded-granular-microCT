"""Complete PSD report: CT grain labels vs Camsizer (glass-pipeline convention).

Emits three figures, one plot per file (strict project style):
  *_Q3.png       : cumulative undersize Q3 (%)              -- CT line + Camsizer lines
  *_q3.png       : volume density q3 (% um^-1)              -- CT bars + Camsizer lines
  *_combined.png : both at once (twin axes: q3 left, Q3 right) -- the complete PSD
Both use the VOLUME-weighted equivalent-sphere diameter d=(6V/pi)^(1/3) (the
glass pipeline's eq_diameter measure).  Camsizer x_area (area-equivalent, the
CT analogue) + xFemin (sieve-equivalent) overlaid.  PNG + vector PDF each.

The Camsizer overlay is the definition that BEST MATCHES the CT curve (lowest
RMS of relative size over D5..D95), chosen automatically among x_area, xFemin,
xMamin, xFemax -- since a Camsizer reports the same powder at sizes differing
30-40 % by shape definition, the best-match definition is the fair comparison.

Run (WSL):
  SAND_CAMSIZER_DIR=/mnt/d/Rolling/Sand ~/ps3d/bin/python psd_labels_vs_camsizer.py \
      LABELS.tif|VOLS.npy [OUT_DIR] [TAG]
argv[1] may be a label .tif OR a grain-volume .npy (voxel counts, e.g.
resplit_vols.npy from the clump-split). Defaults: results/dl_grain_labels_granular.tif,
OUT_DIR=dirname(input), TAG="CT".
"""
import os, sys, numpy as np, tifffile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pipeline/
import experimental_psd as exp
import plot_style as ps
import matplotlib.pyplot as plt

VOX_UM = 24.7660229
HERE = os.path.dirname(os.path.abspath(__file__))
lbl_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "dl_grain_labels_granular.tif")
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(lbl_path)
tag = sys.argv[3] if len(sys.argv) > 3 else "CT"
os.makedirs(out_dir, exist_ok=True)
XMAX = 800

CT_C, A_C, F_C = "#d62728", "#1f77b4", "#2ca02c"
DEFS = ("x_area", "xFemin", "xMamin", "xFemax")
DEF_TEX = {"x_area": "x_{area}", "xFemin": "x_{Fe,min}",
           "xMamin": "x_{Ma,min}", "xFemax": "x_{Fe,max}"}


def sizes_to_dv(sizes, vox_um):
    sizes = np.asarray(sizes)
    sizes = sizes[sizes > 0]
    d = (6.0 * sizes / np.pi) ** (1.0 / 3.0) * vox_um          # equiv-sphere um
    vol = sizes.astype(float) * vox_um ** 3                    # grain volume um^3
    return d, vol


def grain_diam(path, vox_um):
    if path.lower().endswith(".npy"):
        sizes = np.load(path)                                  # voxel counts per grain
    else:
        sizes = np.bincount(tifffile.imread(path).ravel())[1:]
    return sizes_to_dv(sizes, vox_um)


def cum_Q3(d, vol):
    order = np.argsort(d)
    d_s = d[order]
    Q3 = 100.0 * np.cumsum(vol[order]) / vol.sum()
    pct = {p: float(np.interp(p, Q3, d_s)) for p in (10, 50, 90)}
    return d_s, Q3, pct


def density_q3(d, vol, nbins=40, dmax=XMAX):
    edges = np.linspace(0, dmax, nbins + 1)
    lo, hi = edges[:-1], edges[1:]
    mid = 0.5 * (lo + hi); dx = hi - lo
    p3 = np.zeros(nbins)
    for i in range(nbins):
        p3[i] = vol[(d >= lo[i]) & (d < hi[i])].sum()
    p3 = 100.0 * p3 / vol.sum()
    return mid, dx, p3 / dx


print(f"loading {lbl_path}")
d_ct, v_ct = grain_diam(lbl_path, VOX_UM)
d_s, Q3_ct, p_ct = cum_Q3(d_ct, v_ct)
mid_ct, dx_ct, q3_ct = density_q3(d_ct, v_ct)

# --- Camsizer overlay definition ------------------------------------------
# Fixed to xFemin (minimum Feret ~ sieve diameter) for ALL sand specimens so
# the whole study uses one consistent reference. Override via SAND_CAM_DEF.
# RMS of every definition is still printed for the record.
CAM_DEF = os.environ.get("SAND_CAM_DEF", "xFemin")
pgrid = np.linspace(5, 95, 91)
ct_p = np.interp(pgrid, Q3_ct, d_s)
cams, rms = {}, {}
for name in DEFS:
    mid, Q3, p3, n = exp.load_camsizer(name)
    cams[name] = (mid, Q3, p3)
    cam_p = np.interp(pgrid, Q3, mid)
    rms[name] = float(np.sqrt(np.mean(((ct_p - cam_p) / cam_p) ** 2)) * 100.0)
best = CAM_DEF
mid_a, Q3_a, p3_a = cams[best]
p_a = exp.percentiles(mid_a, Q3_a); q3_a = p3_a / np.gradient(mid_a)
A_TEX = DEF_TEX[best]

print(f"[{tag}] CT grains {len(d_ct)}: D10/50/90 = {p_ct[10]:.0f}/{p_ct[50]:.0f}/{p_ct[90]:.0f} um")
for name in DEFS:
    star = "  <-- USED (fixed)" if name == best else ""
    print(f"  Camsizer {name:8s} RMS={rms[name]:5.1f}%{star}")

ps.apply_style()

# ---- (1) cumulative Q3 -------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 9))
ax.plot(d_s, Q3_ct, color=CT_C, linewidth=ps.LW + 1,
        label=f"{tag} labels  ($D_{{50}}$={p_ct[50]:.0f} $\\mu$m)")
ax.plot(mid_a, Q3_a, color=A_C, linewidth=ps.LW, linestyle="--",
        label=f"Camsizer ${A_TEX}$  ($D_{{50}}$={p_a[50]:.0f} $\\mu$m)")
for pct in (10, 50, 90):
    ax.axhline(pct, color="dimgray", linewidth=1.0, linestyle=":")
ax.set_xlabel("Particle size ($\\mu$m)", labelpad=10)
ax.set_ylabel("Cumulative undersize $Q_3$ (%)", labelpad=10)
ax.set_xlim(0, XMAX); ax.set_ylim(0, 100)
ax.yaxis.set_major_locator(plt.MultipleLocator(10))
ps.style_axes(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=True, edgecolor="gray")
ps.save_fig(fig, os.path.join(out_dir, "psd_labels_vs_camsizer_Q3.png"))

# ---- (2) density q3 ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 9))
ax.bar(mid_ct, q3_ct, width=dx_ct * 0.9, color=CT_C, alpha=0.45,
       label=f"{tag} labels  ($D_{{50}}$={p_ct[50]:.0f} $\\mu$m)")
ax.plot(mid_a, q3_a, color=A_C, linewidth=ps.LW,
        label=f"Camsizer ${A_TEX}$  ($D_{{50}}$={p_a[50]:.0f} $\\mu$m)")
ax.set_xlabel("Particle size ($\\mu$m)", labelpad=10)
ax.set_ylabel("Volume density $q_3$ (% $\\mu$m$^{-1}$)", labelpad=10)
ax.set_xlim(0, XMAX); ax.set_ylim(bottom=0)
ps.style_axes(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=True, edgecolor="gray")
ps.save_fig(fig, os.path.join(out_dir, "psd_labels_vs_camsizer_q3.png"))

# ---- (3) combined: density (left axis) + cumulative (right axis) -------------
fig, axL = plt.subplots(figsize=(14, 9))
axR = axL.twinx()
# density on the LEFT axis
axL.bar(mid_ct, q3_ct, width=dx_ct * 0.9, color=CT_C, alpha=0.30,
        label=f"{tag} $q_3$")
axL.plot(mid_a, q3_a, color=A_C, linewidth=ps.LW, label=f"${A_TEX}$ $q_3$")
# cumulative on the RIGHT axis
axR.plot(d_s, Q3_ct, color=CT_C, linewidth=ps.LW + 1, linestyle="-",
         label=f"{tag} $Q_3$ ($D_{{50}}$={p_ct[50]:.0f})")
axR.plot(mid_a, Q3_a, color=A_C, linewidth=ps.LW, linestyle="--", label=f"${A_TEX}$ $Q_3$")
axL.set_xlabel("Particle size ($\\mu$m)", labelpad=10)
axL.set_ylabel("Volume density $q_3$ (% $\\mu$m$^{-1}$)", labelpad=10)
axR.set_ylabel("Cumulative undersize $Q_3$ (%)", labelpad=10)
axL.set_xlim(0, XMAX); axL.set_ylim(bottom=0); axR.set_ylim(0, 100)
axR.yaxis.set_major_locator(plt.MultipleLocator(10))
ps.style_axes(axL)
for sp in axR.spines.values():
    sp.set_visible(True); sp.set_linewidth(1.8)
hL, lL = axL.get_legend_handles_labels()
hR, lR = axR.get_legend_handles_labels()
axL.legend(hL + hR, lL + lR, loc="upper center", bbox_to_anchor=(0.5, -0.16),
           ncol=2, frameon=True, edgecolor="gray", fontsize=ps.FS - 10)
ps.save_fig(fig, os.path.join(out_dir, "psd_labels_vs_camsizer_combined.png"))
print("done")
