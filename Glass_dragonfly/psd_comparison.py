"""
CT vs Camsizer PSD comparison plot.
Run AFTER bead_analysis.py (needs results array in memory with eq_diameter_um).

Overlays:
  - CT-derived PSD (from watershed bead analysis)
  - Camsizer PSD (from .xle files)
on the same plot for direct comparison.
"""

import numpy as np
import os, sys
sys.path.insert(0, r'E:\RPTU-images\CT_images\Glass\Glass_dragonfly')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from plot_style import apply_style, FS, LW, style_axes, save_fig

OUT_DIR = r'E:\RPTU-images\CT_images\Glass\Glass_dragonfly\results'
os.makedirs(OUT_DIR, exist_ok=True)

# -- Camsizer data path (use Test 1 as reference, or average all 3) -----------
CAMSIZER_FILES = {
    "Camsizer Test 1": r"D:\Rolling\Camsizer\Neuer Ordner\test1\Glass_rolling_test5_xc_min_25-01-28.xle",
    "Camsizer Test 2": r"D:\Rolling\Camsizer\Neuer Ordner\test2\Glass_rolling_test5_xc_min_25-01-28.xle",
    "Camsizer Test 3": r"D:\Rolling\Camsizer\Neuer Ordner\test3\Glass_rolling_test3_xc_min_25-01-28.xle",
}

# -- Check results exist -------------------------------------------------------
if 'results' not in dir() or not results:
    raise RuntimeError("Run bead_analysis.py first! 'results' list must be in memory.")

print("=" * 60)
print("CT vs Camsizer PSD Comparison")
print("=" * 60)

# -- Parse Camsizer .xle files -------------------------------------------------
def parse_xle(path):
    with open(path, "r", encoding="utf-16") as fh:
        lines = fh.readlines()

    meta = {}
    for ln in lines:
        s = ln.strip()
        if s.startswith("x") and "Q3=0.100" in s:
            meta["D10"] = float(s.split("\t")[1])
        elif s.startswith("x") and "Q3=0.500" in s:
            meta["D50"] = float(s.split("\t")[1])
        elif s.startswith("x") and "Q3=0.900" in s:
            meta["D90"] = float(s.split("\t")[1])

    meta["Span"] = (meta["D90"] - meta["D10"]) / meta["D50"]

    start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Size class"):
            start = i + 1
            break
    if start is None:
        raise ValueError(f"No data table in {path}")

    x_lo, x_hi, p0 = [], [], []
    for ln in lines[start:]:
        parts = ln.strip().split("\t")
        if len(parts) < 3:
            continue
        try:
            lo, hi, p = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        if hi > 1e5:
            continue
        x_lo.append(lo); x_hi.append(hi); p0.append(p)

    x_lo, x_hi, p0 = np.array(x_lo), np.array(x_hi), np.array(p0)
    dx = x_hi - x_lo
    x_mid = 0.5 * (x_lo + x_hi)
    vol_weight = x_mid**3 * p0
    total = vol_weight.sum()
    if total == 0:
        raise ValueError("All p0 zero")
    p3 = 100.0 * vol_weight / total
    Q3 = np.cumsum(p3)
    q3 = p3 / dx
    return meta, x_mid, dx, q3, Q3

# -- Compute CT PSD ------------------------------------------------------------
print("\nComputing CT PSD...")
diams_ct = np.array([r['eq_diameter_um'] for r in results])
vols_ct = np.array([r['volume_mm3'] for r in results])

n_bins = 30
d_min = max(diams_ct.min() - 100, 0)
d_max = diams_ct.max() + 100
bin_edges = np.linspace(d_min, d_max, n_bins + 1)
bin_lo = bin_edges[:-1]
bin_hi = bin_edges[1:]
dx_ct = bin_hi - bin_lo
x_mid_ct = 0.5 * (bin_lo + bin_hi)

p3_ct = np.zeros(n_bins)
for i in range(n_bins):
    mask = (diams_ct >= bin_lo[i]) & (diams_ct < bin_hi[i])
    p3_ct[i] = vols_ct[mask].sum()

total_vol = p3_ct.sum()
if total_vol > 0:
    p3_ct = 100.0 * p3_ct / total_vol
Q3_ct = np.cumsum(p3_ct)
q3_ct = p3_ct / dx_ct

# CT D10, D50, D90
def interp_dx(Q3_arr, x_arr, pct):
    idx = np.searchsorted(Q3_arr, pct)
    if idx == 0: return x_arr[0]
    if idx >= len(Q3_arr): return x_arr[-1]
    frac = (pct - Q3_arr[idx-1]) / (Q3_arr[idx] - Q3_arr[idx-1])
    return x_arr[idx-1] + frac * (x_arr[idx] - x_arr[idx-1])

ct_D10 = interp_dx(Q3_ct, x_mid_ct, 10)
ct_D50 = interp_dx(Q3_ct, x_mid_ct, 50)
ct_D90 = interp_dx(Q3_ct, x_mid_ct, 90)
ct_Span = (ct_D90 - ct_D10) / ct_D50 if ct_D50 > 0 else 0

print(f"  CT:  D10={ct_D10:.0f}  D50={ct_D50:.0f}  D90={ct_D90:.0f}  Span={ct_Span:.3f}")

# -- Load Camsizer data --------------------------------------------------------
print("\nLoading Camsizer data...")
camsizer_data = []
for name, path in CAMSIZER_FILES.items():
    if os.path.exists(path):
        meta, x_mid_cam, dx_cam, q3_cam, Q3_cam = parse_xle(path)
        camsizer_data.append((name, meta, x_mid_cam, dx_cam, q3_cam, Q3_cam))
        print(f"  {name}: D50={meta['D50']:.0f} um")
    else:
        print(f"  {name}: file not found, skipping")

# -- Plot 1: Q3 cumulative comparison ------------------------------------------
print("\nPlotting Q3 comparison...")
apply_style()

CT_COLOR = "#d62728"       # red
CAM_COLORS = ["#1f77b4", "#2ca02c", "#9467bd"]   # blue, green, purple

fig, ax = plt.subplots(figsize=(14, 9))

# CT cumulative
ax.plot(x_mid_ct, Q3_ct, color=CT_COLOR, linewidth=LW + 1, linestyle='-',
        label=f'CT (D$_{{50}}$={ct_D50:.0f} $\mu$m)')

# Camsizer cumulatives
for (name, meta, x_mid_cam, dx_cam, q3_cam, Q3_cam), color in zip(camsizer_data, CAM_COLORS):
    ax.plot(x_mid_cam, Q3_cam, color=color, linewidth=LW, linestyle='--',
            label=f'{name} (D$_{{50}}$={meta["D50"]:.0f} $\mu$m)')

# Reference lines
for pct in [10, 50, 90]:
    ax.axhline(pct, color='dimgray', linewidth=1.0, linestyle=':')

ax.set_xlabel(r'Particle size ($\mu$m)', labelpad=10)
ax.set_ylabel(r'Cumulative undersize $Q_3$ (%)', labelpad=10)
ax.set_ylim(0, 100)
ax.yaxis.set_major_locator(plt.MultipleLocator(10))

# Focus on relevant range
all_d50 = [ct_D50] + [m['D50'] for _, m, *_ in camsizer_data]
center = np.mean(all_d50)
ax.set_xlim(center - 500, center + 500)

style_axes(ax)

fig.legend(loc='lower center', ncol=2,
           fontsize=FS - 6, frameon=True, framealpha=0.85, edgecolor='gray',
           bbox_to_anchor=(0.5, -0.12))

plt.title('PSD Comparison: CT vs Camsizer', fontsize=FS, fontweight='bold', pad=18)
plt.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, 'psd_ct_vs_camsizer_Q3.png'))

# -- Plot 2: q3 density comparison ---------------------------------------------
print("Plotting q3 comparison...")

fig, ax = plt.subplots(figsize=(14, 9))

# CT bars
ax.bar(x_mid_ct, q3_ct, width=dx_ct * 0.85, color=CT_COLOR, alpha=0.4,
       label=f'CT $q_3$ (D$_{{50}}$={ct_D50:.0f} $\mu$m)')

# Camsizer lines
for (name, meta, x_mid_cam, dx_cam, q3_cam, Q3_cam), color in zip(camsizer_data, CAM_COLORS):
    ax.plot(x_mid_cam, q3_cam, color=color, linewidth=LW,
            label=f'{name} $q_3$ (D$_{{50}}$={meta["D50"]:.0f} $\mu$m)')

ax.set_xlabel(r'Particle size ($\mu$m)', labelpad=10)
ax.set_ylabel(r'Volume density $q_3$ (% $\mu$m$^{-1}$)', labelpad=10)
ax.set_xlim(center - 500, center + 500)
ax.set_ylim(bottom=0)
style_axes(ax)

fig.legend(loc='lower center', ncol=2,
           fontsize=FS - 6, frameon=True, framealpha=0.85, edgecolor='gray',
           bbox_to_anchor=(0.5, -0.12))

plt.title('Volume Density Comparison: CT vs Camsizer', fontsize=FS, fontweight='bold', pad=18)
plt.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, 'psd_ct_vs_camsizer_q3.png'))

# -- Save comparison table to text file ----------------------------------------
table_path = os.path.join(OUT_DIR, 'psd_comparison_table.txt')
with open(table_path, 'w') as f:
    f.write("PSD Comparison: CT vs Camsizer\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"{'Method':<20} {'D10 (um)':>10} {'D50 (um)':>10} {'D90 (um)':>10} {'Span':>8}\n")
    f.write("-" * 62 + "\n")
    f.write(f"{'CT (micro-CT)':<20} {ct_D10:>10.1f} {ct_D50:>10.1f} {ct_D90:>10.1f} {ct_Span:>8.3f}\n")
    for name, meta, *_ in camsizer_data:
        f.write(f"{name:<20} {meta['D10']:>10.1f} {meta['D50']:>10.1f} {meta['D90']:>10.1f} {meta['Span']:>8.3f}\n")
    f.write("\n")
    f.write("Notes:\n")
    f.write("  CT: equivalent spherical diameter from 3D watershed bead analysis\n")
    f.write("  Camsizer: xc_min (minimum chord length) from optical analysis\n")
    f.write("  Size definitions differ: CT measures volume-equivalent diameter,\n")
    f.write("  Camsizer measures projected minimum chord. For spheres these are similar.\n")
print(f"  Saved: {table_path}")

plt.rcdefaults()

print("\n" + "=" * 60)
print("PSD COMPARISON DONE.")
print(f"  CT:       D50 = {ct_D50:.0f} um, Span = {ct_Span:.3f}")
for name, meta, *_ in camsizer_data:
    print(f"  {name}: D50 = {meta['D50']:.0f} um, Span = {meta['Span']:.3f}")
print(f"\nFiles saved to: {OUT_DIR}")
print("=" * 60)
