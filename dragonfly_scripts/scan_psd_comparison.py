"""CT vs Camsizer PSD comparison for scan 1 of the T7 1700-um project.
Run in Dragonfly's Python console AFTER scan01_bead_analysis.py
(uses the `results` list that bead_analysis leaves in memory).

Compares three CT size measures against Camsizer xc_min:
  * eq_diameter_um    -- equivalent-sphere diameter (Camsizer x_vol-like)
  * feret_min_um       -- min Feret / sieve-like (closest to xc_min)
  * feret_mean_um      -- mean Feret over 13 directions

Plots default to eq_diameter_um; change CT_DIAMETER_COL below to switch.
"""
import numpy as np
import os, sys
sys.path.insert(0, r'E:\RPTU-images\CT_images\Glass\<SPECIMEN>\scan02_dragonfly')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import apply_style, FS, LW, style_axes, save_fig

OUT_DIR = r'E:\RPTU-images\CT_images\Glass\<SPECIMEN>\scan{N}_dragonfly\results'
os.makedirs(OUT_DIR, exist_ok=True)

CT_DIAMETER_COL = 'eq_diameter_um'   # or 'feret_min_um' / 'feret_mean_um'

CAMSIZER_FILES = {
    "Camsizer Test 1": r"D:\Rolling\Camsizer\Neuer Ordner\test1\Glass_rolling_test5_xc_min_25-01-28.xle",
    "Camsizer Test 2": r"D:\Rolling\Camsizer\Neuer Ordner\test2\Glass_rolling_test5_xc_min_25-01-28.xle",
    "Camsizer Test 3": r"D:\Rolling\Camsizer\Neuer Ordner\test3\Glass_rolling_test3_xc_min_25-01-28.xle",
}

if 'results' not in dir() or not results:
    raise RuntimeError("Run scan01_bead_analysis.py first -- 'results' must be in memory.")

print("=" * 60)
print("Scan 1 CT vs Camsizer PSD comparison")
print(f"  Using CT diameter column: {CT_DIAMETER_COL}")
print("=" * 60)

def parse_xle(path):
    with open(path, "r", encoding="utf-16") as fh:
        lines = fh.readlines()
    meta = {}
    for ln in lines:
        s = ln.strip()
        if s.startswith("x") and "Q3=0.100" in s: meta["D10"] = float(s.split("\t")[1])
        elif s.startswith("x") and "Q3=0.500" in s: meta["D50"] = float(s.split("\t")[1])
        elif s.startswith("x") and "Q3=0.900" in s: meta["D90"] = float(s.split("\t")[1])
    meta["Span"] = (meta["D90"] - meta["D10"]) / meta["D50"]
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("Size class"):
            start = i + 1; break
    if start is None:
        raise ValueError(f"No data table in {path}")
    x_lo, x_hi, p0 = [], [], []
    for ln in lines[start:]:
        parts = ln.strip().split("\t")
        if len(parts) < 3: continue
        try:
            lo, hi, p = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        if hi > 1e5: continue
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

# -- CT PSD
diams_ct = np.array([r[CT_DIAMETER_COL] for r in results])
vols_ct  = np.array([r['volume_mm3']    for r in results])
n_bins = 30
d_min = max(diams_ct.min() - 50, 0)
d_max = diams_ct.max() + 50
bin_edges = np.linspace(d_min, d_max, n_bins + 1)
bin_lo, bin_hi = bin_edges[:-1], bin_edges[1:]
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

cam = []
for name, path in CAMSIZER_FILES.items():
    if os.path.exists(path):
        m, xm, dxm, q3m, Q3m = parse_xle(path)
        cam.append((name, m, xm, dxm, q3m, Q3m))
        print(f"  {name}: D50={m['D50']:.0f} um")

apply_style()
CT_COLOR = "#d62728"
CAM_COLORS = ["#1f77b4", "#2ca02c", "#9467bd"]

# --- Q3 comparison ---
fig, ax = plt.subplots(figsize=(14, 9))
ax.plot(x_mid_ct, Q3_ct, color=CT_COLOR, linewidth=LW + 1,
        label=f'CT scan01 ($D_{{50}}$={ct_D50:.0f} $\mu$m)')
for (name, m, xm, dxm, q3m, Q3m), c in zip(cam, CAM_COLORS):
    ax.plot(xm, Q3m, color=c, linewidth=LW, linestyle='--',
            label=f'{name} ($D_{{50}}$={m["D50"]:.0f} $\mu$m)')
for pct in [10, 50, 90]:
    ax.axhline(pct, color='dimgray', linewidth=1.0, linestyle=':')
ax.set_xlabel('Particle size ($\mu$m)', labelpad=10)
ax.set_ylabel('Cumulative undersize $Q_3$ (%)', labelpad=10)
ax.set_ylim(0, 100)
ax.yaxis.set_major_locator(plt.MultipleLocator(10))
all_d50 = [ct_D50] + [m['D50'] for _, m, *_ in cam]
center = np.mean(all_d50)
ax.set_xlim(center - 500, center + 500)
style_axes(ax)
fig.legend(loc='lower center', ncol=2, fontsize=FS - 6,
           frameon=True, framealpha=0.85, edgecolor='gray',
           bbox_to_anchor=(0.5, -0.14))
plt.title(f'PSD: scan01 CT vs Camsizer ({CT_DIAMETER_COL})',
          fontsize=FS, fontweight='bold', pad=18)
plt.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, 'psd_ct_vs_camsizer_Q3.png'))

# --- q3 comparison ---
fig, ax = plt.subplots(figsize=(14, 9))
ax.bar(x_mid_ct, q3_ct, width=dx_ct * 0.85, color=CT_COLOR, alpha=0.4,
       label=f'CT $q_3$ ($D_{{50}}$={ct_D50:.0f} $\mu$m)')
for (name, m, xm, dxm, q3m, Q3m), c in zip(cam, CAM_COLORS):
    ax.plot(xm, q3m, color=c, linewidth=LW,
            label=f'{name} ($D_{{50}}$={m["D50"]:.0f} $\mu$m)')
ax.set_xlabel('Particle size ($\mu$m)', labelpad=10)
ax.set_ylabel('Volume density $q_3$ (% $\mu$m$^{-1}$)', labelpad=10)
ax.set_xlim(center - 500, center + 500)
ax.set_ylim(bottom=0)
style_axes(ax)
fig.legend(loc='lower center', ncol=2, fontsize=FS - 6,
           frameon=True, framealpha=0.85, edgecolor='gray',
           bbox_to_anchor=(0.5, -0.14))
plt.title(f'PSD density: scan01 CT vs Camsizer',
          fontsize=FS, fontweight='bold', pad=18)
plt.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, 'psd_ct_vs_camsizer_q3.png'))

# --- table ---
with open(os.path.join(OUT_DIR, 'psd_comparison_table.txt'), 'w') as f:
    f.write("PSD Comparison: CT vs Camsizer - scan01 of T7 1700 um project\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"CT diameter column used: {CT_DIAMETER_COL}\n\n")
    f.write(f"{'Method':<20} {'D10 (um)':>10} {'D50 (um)':>10} {'D90 (um)':>10} {'Span':>8}\n")
    f.write("-" * 62 + "\n")
    f.write(f"{'CT scan01':<20} {ct_D10:>10.1f} {ct_D50:>10.1f} {ct_D90:>10.1f} {ct_Span:>8.3f}\n")
    for name, m, *_ in cam:
        f.write(f"{name:<20} {m['D10']:>10.1f} {m['D50']:>10.1f} {m['D90']:>10.1f} {m['Span']:>8.3f}\n")
    f.write("\nNotes:\n")
    f.write("  CT diameters available per bead (in bead_measurements.csv):\n")
    f.write("    eq_diameter_um  = 2*(3V/4pi)^(1/3)      (analogue of Camsizer x_vol)\n")
    f.write("    feret_min_um    = min caliper over 13 dirs (analogue of Camsizer x_Fe_min)\n")
    f.write("    feret_mean_um   = mean caliper over 13 dirs (paper Sec 3.5.4)\n")
    f.write("  Camsizer xc_min  = minimum chord length (sieve-equivalent size).\n")
    f.write("  For near-spherical glass beads all these values agree within ~5%.\n")

plt.rcdefaults()
print("Done.")
