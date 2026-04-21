"""
Porosity analysis for SCAN 1 (reference / pre-compression) CT specimen.
Run AFTER scan01_segmentation.py (uses seg + specimen_mask in memory).

Mirror of scan02_porosity_analysis.py, but with scan-1 cylinder caps and
scan-1 output folder. Lives in Glass_spam (NOT Glass_dragonfly — per user
constraint).

The specimen of interest is the ICE + GLASS + TRAPPED-AIR cylinder.
Aluminum is labeled in seg for presentation purposes only — it is NOT
part of any fraction in this analysis.

Three porosity definitions (all relative to interior voxels):
  1. Total porosity:      (ice + trapped air) / (ice + glass + trapped air)
  2. Air-filled porosity:        trapped air  / (ice + glass + trapped air)
  3. Packing fraction:                 glass  / (ice + glass + trapped air)
     (= 1 - total porosity)

Each computed as:
  - Overall value
  - Radial profile (center to specimen edge)
  - Height profile (top to bottom)
"""

import numpy as np
import os, time
import sys
sys.path.insert(0, r'E:\RPTU-images\CT_images\Glass\Glass_scan02_dragonfly')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import ndimage
from plot_style import apply_style, FS, LW, style_axes, save_fig

# -- Settings ------------------------------------------------------------------
VOXEL_SIZE_UM = 24.766
SCAN1_ROOT = r'E:\RPTU-images\CT_images\Glass\Glass_spam\scan01'
OUT_DIR    = os.path.join(SCAN1_ROOT, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

# Labels (must match scan01_segmentation.py)
LABEL_OUTSIDE_AIR = 0
LABEL_TRAPPED_AIR = 1
LABEL_ICE         = 2
LABEL_GLASS       = 3
LABEL_ALUMINUM    = 4

# Radial profile settings
N_RADIAL_BINS = 20
N_HEIGHT_BINS = 30

# Radial-coordinate method:
#   'distance_transform' (default, accurate for any specimen shape)
#       For each slice, compute the Euclidean distance of every specimen
#       voxel to the nearest specimen-mask boundary, normalize by the
#       max per-slice distance, and use (1 - norm_dist) as the radial
#       coordinate. r=0 at the deepest interior point, r=1 at the edge.
#       Robust when the specimen is not a perfect cylinder.
#   'cylinder_axis' (legacy)
#       distance-from-cylinder-axis / CYL_RADIUS_VOX. Only correct for
#       a specimen that matches the user-defined cylinder.
RADIAL_METHOD = 'distance_transform'

# Cylinder parameters — MUST match scan01_segmentation.py exactly.
CYL_CAP1_UM = (19501.05, 8206.22,   233.13)
CYL_CAP2_UM = (18737.14, 8178.46, 21259.05)
CYL_RADIUS_UM = 5256.61
CT_ORIGIN_UM = (11578.105, 1151.619, -12.383)

CYL_CAP1_VOX = tuple((c - o) / VOXEL_SIZE_UM for c, o in zip(CYL_CAP1_UM, CT_ORIGIN_UM))
CYL_CAP2_VOX = tuple((c - o) / VOXEL_SIZE_UM for c, o in zip(CYL_CAP2_UM, CT_ORIGIN_UM))
CYL_RADIUS_VOX = CYL_RADIUS_UM / VOXEL_SIZE_UM

# -- Check seg + specimen_mask exist in memory ---------------------------------
if 'seg' not in dir():
    raise RuntimeError("Run scan01_segmentation.py first!")

# specimen_mask is the strict interior (ice + glass + trapped air region).
# Fall back to building one from the seg labels if segmentation didn't
# leave it in memory (e.g. running this script in isolation on saved TIFFs).
if 'specimen_mask' in dir():
    print("Using specimen_mask from scan01_segmentation.py (contour or cylinder mode).")
else:
    print("specimen_mask not in memory — reconstructing from seg labels.")
    specimen_mask = np.isin(seg, [LABEL_TRAPPED_AIR, LABEL_ICE, LABEL_GLASS])

print("=" * 60)
print("SCAN 1 Porosity Analysis — interior (ice + glass + trapped air)")
print("=" * 60)

nz, ny, nx = seg.shape
yy, xx = np.mgrid[0:ny, 0:nx]
cap1_z = CYL_CAP1_VOX[2]
cap2_z = CYL_CAP2_VOX[2]

# -- 1. Overall statistics -----------------------------------------------------
print("\nStep 1: Computing overall statistics...")

n_ice         = int(np.sum(seg == LABEL_ICE))
n_air         = int(np.sum(seg == LABEL_TRAPPED_AIR))
n_glass       = int(np.sum(seg == LABEL_GLASS))
n_aluminum    = int(np.sum(seg == LABEL_ALUMINUM))   # reported only, not used
interior_voxels = n_ice + n_air + n_glass

# Fractions relative to interior (aluminum excluded from both numerator and denom)
total_porosity   = 100.0 * (n_ice + n_air) / interior_voxels if interior_voxels > 0 else 0
air_porosity     = 100.0 * n_air           / interior_voxels if interior_voxels > 0 else 0
ice_fraction     = 100.0 * n_ice           / interior_voxels if interior_voxels > 0 else 0
packing_fraction = 100.0 * n_glass         / interior_voxels if interior_voxels > 0 else 0

print(f"  Interior voxels            : {interior_voxels:>12,} "
      f"({interior_voxels*(VOXEL_SIZE_UM**3)/1e9:.2f} mm3)")
print(f"  Aluminum (not in analysis) : {n_aluminum:>12,} "
      f"({n_aluminum*(VOXEL_SIZE_UM**3)/1e9:.2f} mm3)")
print(f"  Total porosity (ice+air)   : {total_porosity:.2f}%")
print(f"  Air-filled porosity        : {air_porosity:.2f}%")
print(f"  Ice fraction               : {ice_fraction:.2f}%")
print(f"  Packing fraction (glass)   : {packing_fraction:.2f}%")

# -- 2. Compute radial and height profiles for all types -----------------------
print(f"\nStep 2: Computing radial/height profiles (method='{RADIAL_METHOD}')...")
t0 = time.time()

radial = {k: np.zeros(N_RADIAL_BINS) for k in
          ['specimen', 'total_pore', 'air', 'ice', 'glass']}
height = {k: np.zeros(N_HEIGHT_BINS) for k in
          ['specimen', 'total_pore', 'air', 'ice', 'glass']}

for z in range(nz):
    # Height normalization still comes from the cylinder caps (Z-axis is OK)
    t = (z - cap1_z) / (cap2_z - cap1_z)
    if t < 0 or t > 1:
        continue

    inside = specimen_mask[z]   # strict interior (aluminum already excluded)
    if not inside.any():
        continue
    seg_slice = seg[z]

    is_specimen = inside                                          # ice+glass+air
    is_ice      = inside & (seg_slice == LABEL_ICE)
    is_air      = inside & (seg_slice == LABEL_TRAPPED_AIR)
    is_glass    = inside & (seg_slice == LABEL_GLASS)             # no aluminum
    is_pore     = is_ice | is_air

    # ---- radial coordinate -------------------------------------------------
    if RADIAL_METHOD == 'distance_transform':
        # True distance from each specimen voxel to the nearest specimen edge
        # (Euclidean distance transform, per slice). Normalize per-slice so
        # r=0 is the deepest interior point, r=1 is the specimen boundary.
        d_edge = ndimage.distance_transform_edt(inside)
        d_max = d_edge.max()
        if d_max <= 0:
            continue
        norm_r_full = 1.0 - d_edge / d_max    # 0 center, 1 edge
    else:  # 'cylinder_axis'
        cx = CYL_CAP1_VOX[0] + t * (CYL_CAP2_VOX[0] - CYL_CAP1_VOX[0])
        cy = CYL_CAP1_VOX[1] + t * (CYL_CAP2_VOX[1] - CYL_CAP1_VOX[1])
        norm_r_full = np.sqrt((xx - cx)**2 + (yy - cy)**2) / CYL_RADIUS_VOX

    bin_idx = np.clip((norm_r_full * N_RADIAL_BINS).astype(int),
                      0, N_RADIAL_BINS - 1)

    for b in range(N_RADIAL_BINS):
        mb = inside & (bin_idx == b)
        radial['specimen'][b]   += np.sum(mb)
        radial['total_pore'][b] += np.sum(is_pore & mb)
        radial['air'][b]        += np.sum(is_air & mb)
        radial['ice'][b]        += np.sum(is_ice & mb)
        radial['glass'][b]      += np.sum(is_glass & mb)

    # Height bin
    h_bin = min(int(t * N_HEIGHT_BINS), N_HEIGHT_BINS - 1)
    height['specimen'][h_bin]   += np.sum(is_specimen)
    height['total_pore'][h_bin] += np.sum(is_pore)
    height['air'][h_bin]        += np.sum(is_air)
    height['ice'][h_bin]        += np.sum(is_ice)
    height['glass'][h_bin]      += np.sum(is_glass)

radial_r = (np.arange(N_RADIAL_BINS) + 0.5) / N_RADIAL_BINS
height_z = (np.arange(N_HEIGHT_BINS) + 0.5) / N_HEIGHT_BINS

def safe_div(a, b):
    return np.where(b > 0, a / b, 0)

r_total_por = safe_div(radial['total_pore'], radial['specimen'])
r_air_por   = safe_div(radial['air'],        radial['specimen'])
r_ice_frac  = safe_div(radial['ice'],        radial['specimen'])
r_packing   = safe_div(radial['glass'],      radial['specimen'])

h_total_por = safe_div(height['total_pore'], height['specimen'])
h_air_por   = safe_div(height['air'],        height['specimen'])
h_ice_frac  = safe_div(height['ice'],        height['specimen'])
h_packing   = safe_div(height['glass'],      height['specimen'])

print(f"  Profiles done in {time.time()-t0:.1f}s")

# -- 3. Save all data to text files --------------------------------------------
print("\nStep 3: Saving data files...")

# Overall statistics
stats_path = os.path.join(OUT_DIR, 'segmentation_statistics.txt')
with open(stats_path, 'w') as f:
    f.write("SCAN 1 Segmentation Statistics (aluminum excluded from analysis)\n")
    f.write("=" * 65 + "\n")
    f.write(f"Volume shape: {seg.shape} (Z, Y, X)\n")
    f.write(f"Voxel size: {VOXEL_SIZE_UM} um\n\n")
    f.write(f"{'Phase':<25} {'Voxels':>12} {'mm3':>12} {'% interior':>11}\n")
    f.write("-" * 65 + "\n")
    for lbl, name, n in [
        (LABEL_OUTSIDE_AIR, 'Outside air',          int(np.sum(seg == LABEL_OUTSIDE_AIR))),
        (LABEL_TRAPPED_AIR, 'Trapped air',          n_air),
        (LABEL_ICE,         'Ice',                  n_ice),
        (LABEL_GLASS,       'Glass beads',          n_glass),
        (LABEL_ALUMINUM,    'Aluminum (excluded)',  n_aluminum),
    ]:
        vol = n * (VOXEL_SIZE_UM ** 3) / 1e9
        if lbl in (LABEL_TRAPPED_AIR, LABEL_ICE, LABEL_GLASS) and interior_voxels > 0:
            pct_str = f"{100.0 * n / interior_voxels:>10.2f}%"
        else:
            pct_str = "        — "
        f.write(f"{name:<25} {n:>12,} {vol:>12.4f} {pct_str}\n")
    f.write("\n")
    f.write(f"Interior voxels (ice+glass+air): {interior_voxels:,}\n")
    f.write(f"Total porosity (ice+air):    {total_porosity:.2f}%\n")
    f.write(f"Air-filled porosity:         {air_porosity:.2f}%\n")
    f.write(f"Ice fraction:                {ice_fraction:.2f}%\n")
    f.write(f"Packing fraction (glass):    {packing_fraction:.2f}%\n")
print(f"  Saved: {stats_path}")

# Radial profiles (with voxel count per bin so you can see noise floor)
radial_path = os.path.join(OUT_DIR, 'porosity_radial_data.txt')
with open(radial_path, 'w') as f:
    f.write(f"SCAN 1 Radial Porosity Profiles (method='{RADIAL_METHOD}', fraction of interior per radial bin)\n")
    f.write(f"{'Norm_radius':>12};{'Voxel_count':>12};{'Total_porosity':>15};{'Air_porosity':>13};{'Ice_fraction':>13};{'Packing_frac':>13}\n")
    for i in range(N_RADIAL_BINS):
        f.write(f"{radial_r[i]:>12.4f};{int(radial['specimen'][i]):>12};"
                f"{r_total_por[i]:>15.4f};{r_air_por[i]:>13.4f};"
                f"{r_ice_frac[i]:>13.4f};{r_packing[i]:>13.4f}\n")
print(f"  Saved: {radial_path}")

# Height profiles
height_path_data = os.path.join(OUT_DIR, 'porosity_height_data.txt')
with open(height_path_data, 'w') as f:
    f.write("SCAN 1 Height Porosity Profiles (fraction of interior per height bin)\n")
    f.write(f"{'Norm_height':>12};{'Total_porosity':>15};{'Air_porosity':>13};{'Ice_fraction':>13};{'Packing_frac':>13}\n")
    for i in range(N_HEIGHT_BINS):
        f.write(f"{height_z[i]:>12.4f};{h_total_por[i]:>15.4f};{h_air_por[i]:>13.4f};{h_ice_frac[i]:>13.4f};{h_packing[i]:>13.4f}\n")
print(f"  Saved: {height_path_data}")

# Description file
desc_path = os.path.join(OUT_DIR, 'porosity_description.txt')
with open(desc_path, 'w') as f:
    f.write("SCAN 1 POROSITY ANALYSIS — DEFINITIONS AND FILES\n")
    f.write("=" * 60 + "\n\n")
    f.write("Glass bead / ice / trapped air CT specimen (pre-compression).\n")
    f.write("Aluminum is shown in seg for presentation but EXCLUDED from\n")
    f.write("all fractions — the analysis is about the ice+glass+air cylinder.\n\n")
    f.write("INTERIOR DEFINITION\n")
    f.write("-" * 40 + "\n")
    f.write("Interior = ice + glass + trapped air (aluminum excluded).\n")
    f.write("Given by specimen_mask from scan01_segmentation.py.\n\n")
    f.write("POROSITY DEFINITIONS (denominator = interior voxels)\n")
    f.write("-" * 40 + "\n\n")
    f.write("1. TOTAL POROSITY (ice + trapped air) / interior\n")
    f.write("   = all void space between glass beads, regardless of fill.\n")
    f.write(f"   Overall value: {total_porosity:.2f}%\n")
    f.write("   Files: porosity_total_radial.png, porosity_total_height.png\n\n")
    f.write("2. AIR-FILLED POROSITY (trapped air) / interior\n")
    f.write("   = fraction of pore space that remained unfrozen.\n")
    f.write(f"   Overall value: {air_porosity:.2f}%\n")
    f.write("   Files: porosity_air_radial.png, porosity_air_height.png\n\n")
    f.write("3. ICE FRACTION (ice) / interior\n")
    f.write("   = fraction of interior occupied by ice.\n")
    f.write(f"   Overall value: {ice_fraction:.2f}%\n")
    f.write("   Files: porosity_ice_radial.png, porosity_ice_height.png\n\n")
    f.write("4. PACKING FRACTION (glass) / interior\n")
    f.write("   = fraction of interior occupied by glass beads.\n")
    f.write("   packing + total porosity = 100%\n")
    f.write(f"   Overall value: {packing_fraction:.2f}%\n")
    f.write("   Files: porosity_packing_radial.png, porosity_packing_height.png\n\n")
    f.write("RADIAL PROFILES\n")
    f.write("-" * 40 + "\n")
    f.write("Porosity vs normalized radius (0 = cylinder axis, 1 = reference wall).\n")
    f.write("Shows the wall effect for packed beds.\n")
    f.write("Reference: Sadeq et al. (2024) Fuel Processing Technology 265, 108149.\n\n")
    f.write("HEIGHT PROFILES\n")
    f.write("-" * 40 + "\n")
    f.write("Porosity vs normalized height (0 = top, 1 = bottom).\n\n")
    f.write("DATA FILES\n")
    f.write("-" * 40 + "\n")
    f.write("segmentation_statistics.txt    Overall phase volumes and fractions\n")
    f.write("porosity_radial_data.txt       Radial profiles (semicolon-separated)\n")
    f.write("porosity_height_data.txt       Height profiles (semicolon-separated)\n")
    f.write("porosity_description.txt       This file\n")
print(f"  Saved: {desc_path}")

# -- 4. Plots ------------------------------------------------------------------
# Plot-style defaults (see memory/feedback_plot_style.md):
#   axis labels >= _FS, tick labels _FS - 2, legend bbox y <= -0.14.
print("\nStep 4: Creating plots...")
apply_style()
plt.rcParams.update({
    "axes.labelsize":  FS + 2,
    "xtick.labelsize": FS - 2,
    "ytick.labelsize": FS - 2,
    "legend.fontsize": FS - 2,
})

profiles = [
    ('total', 'Total Porosity (ice + air)',
     r_total_por, h_total_por, total_porosity, '#1f77b4'),
    ('air', 'Air-Filled Porosity',
     r_air_por, h_air_por, air_porosity, '#d62728'),
    ('ice', 'Ice Fraction',
     r_ice_frac, h_ice_frac, ice_fraction, '#2ca02c'),
    ('packing', 'Packing Fraction (glass)',
     r_packing, h_packing, packing_fraction, '#ff7f0e'),
]

for tag, title, r_data, h_data, overall, color in profiles:
    mean_val = overall

    # Radial plot
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.plot(radial_r, r_data * 100, 'o-', color=color, linewidth=LW,
            markersize=10, markeredgecolor='black', markeredgewidth=1.2)
    ax.axhline(mean_val, color='dimgray', linewidth=2, linestyle='--')
    ax.set_xlabel('Normalized radius [-]', labelpad=10)
    ax.set_ylabel('Fraction [%]', labelpad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    style_axes(ax)
    handles = [
        Line2D([0], [0], color=color, linewidth=LW, marker='o', markersize=10,
               markeredgecolor='black', label=f'Radial {title.lower()}'),
        Line2D([0], [0], color='dimgray', linewidth=2, linestyle='--',
               label=f'Mean = {mean_val:.1f}%'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               fontsize=FS - 2, frameon=True, framealpha=0.85, edgecolor='gray',
               bbox_to_anchor=(0.5, -0.14))
    plt.title(f'Radial {title}', fontsize=FS, fontweight='bold', pad=18)
    plt.tight_layout()
    save_fig(fig, os.path.join(OUT_DIR, f'porosity_{tag}_radial.png'))

    # Height plot
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.plot(height_z, h_data * 100, 's-', color=color, linewidth=LW,
            markersize=8, markeredgecolor='black', markeredgewidth=1.2)
    ax.axhline(mean_val, color='dimgray', linewidth=2, linestyle='--')
    ax.set_xlabel('Normalized height [-]', labelpad=10)
    ax.set_ylabel('Fraction [%]', labelpad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    style_axes(ax)
    handles = [
        Line2D([0], [0], color=color, linewidth=LW, marker='s', markersize=8,
               markeredgecolor='black', label=f'Height {title.lower()}'),
        Line2D([0], [0], color='dimgray', linewidth=2, linestyle='--',
               label=f'Mean = {mean_val:.1f}%'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               fontsize=FS - 2, frameon=True, framealpha=0.85, edgecolor='gray',
               bbox_to_anchor=(0.5, -0.14))
    plt.title(f'{title} vs Height', fontsize=FS, fontweight='bold', pad=18)
    plt.tight_layout()
    save_fig(fig, os.path.join(OUT_DIR, f'porosity_{tag}_height.png'))

# Combined overview: all 4 radial profiles on one plot
fig, ax = plt.subplots(figsize=(14, 9))
for tag, title, r_data, h_data, overall, color in profiles:
    ax.plot(radial_r, r_data * 100, 'o-', color=color, linewidth=LW,
            markersize=8, markeredgecolor='black', markeredgewidth=0.8,
            label=f'{title} ({overall:.1f}%)')
ax.set_xlabel('Normalized radius [-]', labelpad=10)
ax.set_ylabel('Fraction [%]', labelpad=10)
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)
style_axes(ax)
fig.legend(loc='lower center', ncol=2,
           fontsize=FS - 2, frameon=True, framealpha=0.85, edgecolor='gray',
           bbox_to_anchor=(0.5, -0.18))
plt.title('SCAN 1 Radial Profiles — All Fractions', fontsize=FS, fontweight='bold', pad=18)
plt.tight_layout()
save_fig(fig, os.path.join(OUT_DIR, 'porosity_all_radial.png'))

plt.rcdefaults()

print("\n" + "=" * 60)
print("SCAN 1 POROSITY ANALYSIS DONE.")
print(f"  Total porosity:   {total_porosity:.2f}%")
print(f"  Air porosity:     {air_porosity:.2f}%")
print(f"  Ice fraction:     {ice_fraction:.2f}%")
print(f"  Packing fraction: {packing_fraction:.2f}%")
print(f"\nAll files saved to: {OUT_DIR}")
print("=" * 60)
