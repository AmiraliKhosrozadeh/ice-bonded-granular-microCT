r"""
Scan 1 vs Scan 2 comparison.

Reads:
  Glass_dragonfly\results\bead_measurements.csv
  Glass_dragonfly\results\segmentation_statistics.txt
  Glass_dragonfly\results\porosity_radial_data.txt
  Glass_scan02_dragonfly\results\bead_measurements.csv
  Glass_scan02_dragonfly\results\segmentation_statistics.txt
  Glass_scan02_dragonfly\results\porosity_radial_data.txt

Writes three comparison PNGs to:
  Glass_scan02_dragonfly\results\comparison\

No Dragonfly required; runs from any Python with matplotlib + numpy.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCAN1_DIR = r'E:\RPTU-images\CT_images\Glass\Glass_dragonfly\results'
SCAN2_DIR = r'E:\RPTU-images\CT_images\Glass\Glass_scan02_dragonfly\results'
OUT_DIR   = os.path.join(SCAN2_DIR, 'comparison')
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, r'E:\RPTU-images\CT_images\Glass\Glass_scan02_dragonfly')
from plot_style import apply_style, FS, LW, style_axes, save_fig

# -- Helpers ------------------------------------------------------------------
def load_bead_csv(path):
    """Return dict of arrays from bead_measurements.csv (semicolon-separated)."""
    with open(path) as f:
        headers = f.readline().strip().split(';')
        rows = [line.strip().split(';') for line in f if line.strip()]
    cols = {h: [] for h in headers}
    for r in rows:
        for h, v in zip(headers, r):
            cols[h].append(v)
    # convert numeric columns
    out = {}
    for h, vs in cols.items():
        try:
            out[h] = np.array([float(v) for v in vs])
        except ValueError:
            out[h] = np.array(vs)
    return out


def load_radial_data(path):
    """Return (r, total_por, air_por, ice, packing, voxel_count)."""
    data = np.genfromtxt(path, delimiter=';', skip_header=2,
                         dtype=None, encoding='utf-8')
    # Fields: Norm_radius, Voxel_count, Total_porosity, Air_porosity,
    #         Ice_fraction, Packing_frac   (scan 2 has Voxel_count; scan 1
    #         original format may not — handle both)
    arr = np.atleast_2d(np.asarray([list(row) for row in data], dtype=float))
    if arr.shape[1] == 6:
        return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]
    else:
        # old format (no voxel count) — first col radius, then 4 fractions
        voxel_count = np.full(arr.shape[0], np.nan)
        return arr[:, 0], voxel_count, arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]


def load_stats_dict(path):
    """Parse segmentation_statistics.txt into a dict of fractions."""
    out = {}
    with open(path) as f:
        for line in f:
            for key, label in [
                ('total_porosity', 'Total porosity'),
                ('air_porosity',   'Air-filled porosity'),
                ('ice_fraction',   'Ice fraction'),
                ('packing',        'Packing fraction'),
            ]:
                if line.strip().startswith(label):
                    tail = line.split(':', 1)[1].strip().rstrip('%').strip()
                    try:
                        out[key] = float(tail.split()[0].rstrip('%'))
                    except ValueError:
                        pass
    return out


# -- Load ---------------------------------------------------------------------
print("Loading scan 1 data...")
b1 = load_bead_csv(os.path.join(SCAN1_DIR, 'bead_measurements.csv'))
s1 = load_stats_dict(os.path.join(SCAN1_DIR, 'segmentation_statistics.txt'))
r1 = load_radial_data(os.path.join(SCAN1_DIR, 'porosity_radial_data.txt'))

print("Loading scan 2 data...")
b2 = load_bead_csv(os.path.join(SCAN2_DIR, 'bead_measurements.csv'))
s2 = load_stats_dict(os.path.join(SCAN2_DIR, 'segmentation_statistics.txt'))
r2 = load_radial_data(os.path.join(SCAN2_DIR, 'porosity_radial_data.txt'))

# -- Plot style ---------------------------------------------------------------
apply_style()
plt.rcParams.update({
    "axes.labelsize":  FS + 2,
    "xtick.labelsize": FS - 2,
    "ytick.labelsize": FS - 2,
    "legend.fontsize": FS - 2,
})

# -- 1. Overlaid PSD (Feret) --------------------------------------------------
def feret_col(b):
    return b.get('feret_mean_um', b.get('eq_diameter_um'))

d1 = feret_col(b1)
d2 = feret_col(b2)

d_lo = min(d1.min(), d2.min()) - 50
d_hi = max(d1.max(), d2.max()) + 50
bin_edges = np.linspace(d_lo, d_hi, 31)
x_mid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
dx = np.diff(bin_edges)

def psd_curves(diams, vols):
    p3 = np.zeros(len(x_mid))
    for i in range(len(x_mid)):
        mask = (diams >= bin_edges[i]) & (diams < bin_edges[i+1])
        p3[i] = vols[mask].sum()
    if p3.sum() > 0:
        p3 = 100.0 * p3 / p3.sum()
    return np.cumsum(p3)

Q1 = psd_curves(d1, b1['volume_mm3'])
Q2 = psd_curves(d2, b2['volume_mm3'])

fig, ax = plt.subplots(figsize=(14, 9))
ax.plot(x_mid, Q1, '-',  color='#1f77b4', linewidth=LW + 1,
        label=f'Scan 1 (n={len(d1)})')
ax.plot(x_mid, Q2, '--', color='#d62728', linewidth=LW + 1,
        label=f'Scan 2 (n={len(d2)})')
ax.set_xlabel('Mean Feret diameter (µm)', labelpad=10)
ax.set_ylabel('Cumulative Q₃ (%)', labelpad=10)
ax.set_ylim(0, 100)
ax.yaxis.set_major_locator(plt.MultipleLocator(10))
style_axes(ax)
fig.legend(loc='lower center', ncol=2, fontsize=FS - 2,
           frameon=True, framealpha=0.85, edgecolor='gray',
           bbox_to_anchor=(0.5, -0.14))
plt.title('Particle Size Distribution — scan 1 vs scan 2',
          fontsize=FS, fontweight='bold', pad=18)
plt.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, 'compare_psd.png'))

# -- 2. Bar chart of phase fractions ------------------------------------------
labels_bar = ['Total porosity', 'Air porosity', 'Ice fraction', 'Packing']
keys_bar   = ['total_porosity', 'air_porosity', 'ice_fraction', 'packing']
vals_1 = [s1.get(k, 0.0) for k in keys_bar]
vals_2 = [s2.get(k, 0.0) for k in keys_bar]

fig, ax = plt.subplots(figsize=(14, 9))
x = np.arange(len(labels_bar))
w = 0.35
ax.bar(x - w/2, vals_1, w, color='#1f77b4', edgecolor='black', label='Scan 1')
ax.bar(x + w/2, vals_2, w, color='#d62728', edgecolor='black', label='Scan 2')
ax.set_xticks(x)
ax.set_xticklabels(labels_bar, rotation=0)
ax.set_ylabel('Fraction of interior (%)', labelpad=10)
style_axes(ax)
for i, (v1, v2) in enumerate(zip(vals_1, vals_2)):
    ax.text(i - w/2, v1 + 0.5, f'{v1:.1f}', ha='center', fontsize=FS - 4)
    ax.text(i + w/2, v2 + 0.5, f'{v2:.1f}', ha='center', fontsize=FS - 4)
fig.legend(loc='lower center', ncol=2, fontsize=FS - 2,
           frameon=True, framealpha=0.85, edgecolor='gray',
           bbox_to_anchor=(0.5, -0.10))
plt.title('Interior phase fractions — scan 1 vs scan 2',
          fontsize=FS, fontweight='bold', pad=18)
plt.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, 'compare_phase_fractions.png'))

# -- 3. Overlaid radial porosity ---------------------------------------------
rr1, _, t1, a1, i1, p1 = r1
rr2, _, t2, a2, i2, p2 = r2

fig, ax = plt.subplots(figsize=(14, 9))
ax.plot(rr1, t1 * 100, 'o-',  color='#1f77b4', linewidth=LW, markersize=8,
        markeredgecolor='black', label='Scan 1 total porosity')
ax.plot(rr2, t2 * 100, 's--', color='#d62728', linewidth=LW, markersize=8,
        markeredgecolor='black', label='Scan 2 total porosity')
ax.plot(rr1, p1 * 100, 'o-',  color='#2ca02c', linewidth=LW, markersize=8,
        markeredgecolor='black', label='Scan 1 packing', alpha=0.6)
ax.plot(rr2, p2 * 100, 's--', color='#ff7f0e', linewidth=LW, markersize=8,
        markeredgecolor='black', label='Scan 2 packing', alpha=0.6)
ax.set_xlabel('Normalized radius [-]', labelpad=10)
ax.set_ylabel('Fraction (%)', labelpad=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 100)
style_axes(ax)
fig.legend(loc='lower center', ncol=2, fontsize=FS - 4,
           frameon=True, framealpha=0.85, edgecolor='gray',
           bbox_to_anchor=(0.5, -0.20))
plt.title('Radial porosity profile — scan 1 vs scan 2',
          fontsize=FS, fontweight='bold', pad=18)
plt.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, 'compare_radial_porosity.png'))

plt.rcdefaults()
print(f"\nComparison plots saved to: {OUT_DIR}")
