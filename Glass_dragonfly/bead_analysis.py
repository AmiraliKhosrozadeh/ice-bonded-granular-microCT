"""
Individual glass bead labeling and analysis
Run AFTER segmentation.py (uses the seg array already in memory).

Pipeline:
  1. Extract glass-only mask from segmentation
  2. Morphological cleanup (fill holes, remove noise)
  3. 3D distance transform (distance from each glass voxel to nearest non-glass)
  4. Find markers (local maxima of distance map = bead centers)
  5. Watershed to split touching beads
  6. Connected component labeling -> each bead gets unique ID
  7. Measure each bead (volume, diameter, centroid, sphericity)
  8. Export results to CSV
"""

import numpy as np
import os, time
from scipy import ndimage

# -- Settings ------------------------------------------------------------------
# These can be adjusted to tune the separation quality

MIN_BEAD_VOXELS  = 200      # beads smaller than this are removed (noise)
MARKER_MIN_DIST  = 18        # min distance (voxels) between bead centers
                             # increase if beads are over-split
                             # decrease if touching beads are merged
MORPH_CLOSING_R  = 0        # 0 = no closing (closing glues beads together!)
DIST_SMOOTH_SIGMA = 0.5     # smooth distance map before peak detection

VOXEL_SIZE_UM    = 24.766   # um per voxel (must match segmentation.py)
LABEL_PARTICLE   = 3        # label of the particle phase to analyze in seg array
LABEL_WALL       = 4        # label of the wall/holder phase (set to None if no wall)
WALL_EXCLUSION_VOXELS = 0   # 0 = disabled (sphere-fit cleanup handles wall instead)

# Sphere-fit cleanup (set False for non-spherical particles like sand)
SPHERE_FIT_CLEANUP = False  # disabled — did not work well
SPHERICITY_THRESHOLD = 0.65 # only used if SPHERE_FIT_CLEANUP=True

PUBLISH_MULTI_ROI = True    # create MultiROI in Dragonfly for bead visualization
CT_CHANNEL_NAME   = 'Glass-1001800'  # search string to find CT channel in Dragonfly

# Feret's diameter settings -----------------------------------------------------
# The 3D mean Feret diameter is the mean caliper size averaged over a set of
# unit-sphere directions, following Ben Elhadj Hamida (2024, Section 3.5.4).
# 13 half-space directions = 26 antipodal axes covering the sphere uniformly.
FERET_N_DIRECTIONS = 13
USE_FERET_FOR_PSD  = True   # if True, PSD plot uses mean-Feret on the x-axis

OUT_DIR = r'E:\RPTU-images\CT_images\Glass\Glass_dragonfly\results'
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, 'bead_measurements.csv')

# -- Check that seg exists in memory -------------------------------------------
if 'seg' not in dir():
    raise RuntimeError(
        "Run segmentation.py first! The 'seg' array must be in memory.\n"
        "  exec(open(r'...\\segmentation.py', encoding='utf-8').read(), globals())"
    )

print("=" * 60)
print("Glass Bead Analysis: Individual bead labeling")
print("=" * 60)

# -- Step 1: Extract glass-only mask ------------------------------------------
print("\nStep 1: Extracting glass phase...")
glass_mask = (seg == LABEL_PARTICLE)
n_glass = int(np.sum(glass_mask))
print(f"  Glass voxels: {n_glass:,}")

# -- Step 1b: Remove particle-wall bridges -------------------------------------
# Wall can create false bridges to particles. Remove particle voxels
# within a few voxels of the wall before watershed.
wall_mask = (seg == LABEL_WALL) if LABEL_WALL is not None else None
if wall_mask is not None and np.any(wall_mask) and WALL_EXCLUSION_VOXELS > 0:
    print(f"\nStep 1b: Removing particle-wall bridges (exclusion zone = {WALL_EXCLUSION_VOXELS} vox)...")
    n_wall = int(np.sum(wall_mask))
    wall_zone = ndimage.binary_dilation(wall_mask, iterations=WALL_EXCLUSION_VOXELS)
    n_removed = int(np.sum(glass_mask & wall_zone))
    glass_mask = glass_mask & (~wall_zone)
    print(f"  Wall voxels: {n_wall:,}")
    print(f"  Glass voxels removed near wall: {n_removed:,}")
    print(f"  Glass voxels remaining: {int(np.sum(glass_mask)):,}")
else:
    print("\nStep 1b: No wall (aluminum) phase detected, skipping.")

# -- Step 2: Morphological cleanup --------------------------------------------
if MORPH_CLOSING_R > 0:
    print(f"\nStep 2: Morphological closing (radius={MORPH_CLOSING_R})...")
    t0 = time.time()
    struct = ndimage.generate_binary_structure(3, 1)
    struct = ndimage.iterate_structure(struct, MORPH_CLOSING_R)
    glass_mask = ndimage.binary_closing(glass_mask, structure=struct)
    glass_mask = ndimage.binary_fill_holes(glass_mask)
    n_after = int(np.sum(glass_mask))
    print(f"  Glass voxels after cleanup: {n_after:,} (was {n_glass:,})")
    print(f"  Closing done in {time.time()-t0:.1f}s")
else:
    print("\nStep 2: Skipping morphological closing (MORPH_CLOSING_R=0)")

# -- Step 2b: Erode to break bridges between touching beads --------------------
ERODE_ITERATIONS = 20   # voxels to erode (1-3). Higher = more aggressive splitting
print(f"\nStep 2b: Eroding glass mask by {ERODE_ITERATIONS} voxels to break bridges...")
glass_mask_original = glass_mask.copy()   # save original for grow-back
glass_mask_eroded = ndimage.binary_erosion(glass_mask, iterations=ERODE_ITERATIONS)
n_eroded = int(np.sum(glass_mask_eroded))
print(f"  Glass voxels after erosion: {n_eroded:,} (was {int(np.sum(glass_mask)):,})")

# -- Step 3: 3D distance transform (on eroded mask) ---------------------------
print("\nStep 3: Computing 3D distance transform...")
t0 = time.time()
dist = ndimage.distance_transform_edt(glass_mask_eroded)
print(f"  Max distance: {dist.max():.1f} voxels ({dist.max()*VOXEL_SIZE_UM:.0f} um)")
print(f"  Distance transform done in {time.time()-t0:.1f}s")

# Smooth distance map to stabilize peak detection
if DIST_SMOOTH_SIGMA > 0:
    dist_smooth = ndimage.gaussian_filter(dist, sigma=DIST_SMOOTH_SIGMA)
    print(f"  Smoothed distance map (sigma={DIST_SMOOTH_SIGMA})")
else:
    dist_smooth = dist

# -- Step 4: Find markers (bead centers) --------------------------------------
print(f"\nStep 4: Finding bead centers (min distance = {MARKER_MIN_DIST} vox)...")
t0 = time.time()

try:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    HAS_SKIMAGE = True
    print("  Using scikit-image (peak_local_max + watershed)")
except ImportError:
    HAS_SKIMAGE = False
    print("  scikit-image not available, using scipy fallback")

if HAS_SKIMAGE:
    # skimage approach: find local maxima of smoothed distance map
    coords = peak_local_max(
        dist_smooth,
        min_distance=MARKER_MIN_DIST,
        threshold_abs=4.0,        # ignore shallow peaks
        labels=glass_mask_eroded.astype(np.uint8),
        exclude_border=False
    )
    n_markers = len(coords)
    print(f"  Found {n_markers} bead markers in {time.time()-t0:.1f}s")

    # Create marker array
    markers = np.zeros(dist.shape, dtype=np.int32)
    for i, (z, y, x) in enumerate(coords):
        markers[z, y, x] = i + 1   # labels start at 1

    # -- Step 5: Watershed separation ------------------------------------------
    print("\nStep 5: Watershed separation...")
    t0 = time.time()
    # Watershed on eroded mask (bridges broken)
    bead_labels = watershed(-dist_smooth, markers, mask=glass_mask_eroded)
    print(f"  Watershed done in {time.time()-t0:.1f}s")

    # Grow labels back into original glass mask (recover eroded voxels)
    print("  Growing labels back into original glass mask...")
    t_grow = time.time()
    # Use watershed again: existing labels as seeds, grow into original mask
    bead_labels = watershed(-dist, bead_labels, mask=glass_mask_original)
    print(f"  Grow-back done in {time.time()-t_grow:.1f}s")

else:
    # scipy-only fallback: use local maximum filter to find markers
    from scipy.ndimage import maximum_filter, label as cc_label

    # Find local maxima by comparing dist to dilated dist
    footprint = np.ones((MARKER_MIN_DIST*2+1,)*3)
    local_max = (dist == maximum_filter(dist, footprint=footprint)) & (dist > 2.0)
    markers, n_markers = cc_label(local_max)
    print(f"  Found {n_markers} bead markers in {time.time()-t0:.1f}s")

    # Watershed using scipy
    print("\nStep 5: Watershed separation (scipy)...")
    t0 = time.time()
    # scipy.ndimage.watershed_ift needs integer input
    dist_int = (dist.max() - dist)   # invert: watersheds fill from minima
    dist_int = (dist_int * 1000).astype(np.int32)
    bead_labels = ndimage.watershed_ift(dist_int, markers)
    bead_labels[~glass_mask] = 0     # mask out non-glass
    print(f"  Watershed done in {time.time()-t0:.1f}s")

# -- Step 5b: Save marker QC image ---------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, r'E:\RPTU-images\CT_images\Glass\Glass_dragonfly')
from plot_style import apply_style, FS, LW, style_axes, save_fig

apply_style()
mid_z = dist.shape[0] // 2
marker_slice = markers[mid_z] > 0
yy_m, xx_m = np.where(marker_slice)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Left: markers on glass mask
axes[0].imshow(glass_mask[mid_z], cmap='gray', origin='upper', aspect='equal')
axes[0].plot(xx_m, yy_m, 'r.', markersize=12, markeredgecolor='yellow', markeredgewidth=1.0)
axes[0].set_title(f'Markers on glass mask (Z={mid_z})', fontsize=FS - 4, pad=10)
axes[0].set_xlabel('X (voxels)', fontsize=FS - 6)
axes[0].set_ylabel('Y (voxels)', fontsize=FS - 6)
style_axes(axes[0])

# Right: markers on distance map
im = axes[1].imshow(dist_smooth[mid_z], cmap='hot', origin='upper', aspect='equal')
axes[1].plot(xx_m, yy_m, 'c.', markersize=12, markeredgecolor='white', markeredgewidth=1.0)
axes[1].set_title(f'Markers on distance map (Z={mid_z})', fontsize=FS - 4, pad=10)
axes[1].set_xlabel('X (voxels)', fontsize=FS - 6)
axes[1].set_ylabel('Y (voxels)', fontsize=FS - 6)
style_axes(axes[1])
cbar = fig.colorbar(im, ax=axes[1], shrink=0.8, pad=0.02)
cbar.set_label('Distance (voxels)', fontsize=FS - 6)

fig.suptitle(f'Bead Marker Detection ({n_markers} markers)', fontsize=FS, fontweight='bold', y=1.02)
plt.tight_layout()
marker_path = os.path.join(OUT_DIR, 'bead_markers_check.png')
save_fig(fig, marker_path)
plt.rcdefaults()
print(f"  Saved: {marker_path}")

# -- Step 6: Remove small beads -----------------------------------------------
print(f"\nStep 6: Removing beads smaller than {MIN_BEAD_VOXELS} voxels...")
bead_sizes = np.bincount(bead_labels.ravel())
bead_sizes[0] = 0   # background

small_beads = np.where(bead_sizes < MIN_BEAD_VOXELS)[0]
if len(small_beads) > 0:
    # Build lookup table: small bead labels -> 0
    lut = np.arange(len(bead_sizes), dtype=np.int32)
    lut[small_beads] = 0
    bead_labels = lut[bead_labels]

# Relabel consecutively
unique_labels = np.unique(bead_labels)
unique_labels = unique_labels[unique_labels > 0]
n_beads = len(unique_labels)

relabel_lut = np.zeros(bead_labels.max() + 1, dtype=np.int32)
for new_id, old_id in enumerate(unique_labels, start=1):
    relabel_lut[old_id] = new_id
bead_labels = relabel_lut[bead_labels]

print(f"  Final bead count: {n_beads}")
print(f"  Removed {len(small_beads)} small objects")

# -- Step 6b: Detect wall-touching beads ---------------------------------------
wall_touching_ids = set()
if wall_mask is not None and np.any(wall_mask):
    wall_check = ndimage.binary_dilation(wall_mask, iterations=1)
    touch_ids = np.unique(bead_labels[wall_check])
    touch_ids = touch_ids[touch_ids > 0]
    wall_touching_ids = set(touch_ids.tolist())
    print(f"\n  Wall-touching beads: {len(wall_touching_ids)} (flagged, not removed)")
    if wall_touching_ids:
        print(f"  IDs: {sorted(wall_touching_ids)[:20]}{'...' if len(wall_touching_ids) > 20 else ''}")

# -- Step 6c: Sphere-fit cleanup (optional, for spherical particles only) ------
# For spherical particles (glass, alumina), trim non-spherical labels to best-fit
# sphere. Skip for irregular particles (sand).
if SPHERE_FIT_CLEANUP:
    print(f"\nStep 6c: Sphere-fit cleanup (sphericity threshold = {SPHERICITY_THRESHOLD})...")
else:
    print("\nStep 6c: Skipped (SPHERE_FIT_CLEANUP = False, for non-spherical particles)")

if SPHERE_FIT_CLEANUP:
    t0 = time.time()
    n_trimmed = 0

    for bead_id in range(1, n_beads + 1):
        mask_i = (bead_labels == bead_id)
        n_vox = int(np.sum(mask_i))
        if n_vox < MIN_BEAD_VOXELS:
            continue

        coords_i = np.argwhere(mask_i)
        cz = float(np.mean(coords_i[:, 0]))
        cy = float(np.mean(coords_i[:, 1]))
        cx = float(np.mean(coords_i[:, 2]))

        bb_z = coords_i[:, 0].max() - coords_i[:, 0].min() + 1
        bb_y = coords_i[:, 1].max() - coords_i[:, 1].min() + 1
        bb_x = coords_i[:, 2].max() - coords_i[:, 2].min() + 1

        vol_um3 = n_vox * (VOXEL_SIZE_UM ** 3)
        max_dim = max(bb_z, bb_y, bb_x) * VOXEL_SIZE_UM
        sphere_vol = (4.0/3.0) * np.pi * (max_dim/2.0)**3
        sph = vol_um3 / sphere_vol if sphere_vol > 0 else 0

        if sph >= SPHERICITY_THRESHOLD:
            continue

        cz_i, cy_i, cx_i = int(round(cz)), int(round(cy)), int(round(cx))
        cz_i = max(0, min(cz_i, dist.shape[0]-1))
        cy_i = max(0, min(cy_i, dist.shape[1]-1))
        cx_i = max(0, min(cx_i, dist.shape[2]-1))
        radius_vox = dist[cz_i, cy_i, cx_i]

        if radius_vox < 5:
            radius_vox = (3.0 * n_vox / (4.0 * np.pi)) ** (1.0/3.0)

        radius_vox = radius_vox * 1.05

        margin = int(radius_vox) + 2
        z_lo = max(0, int(cz) - margin)
        z_hi = min(dist.shape[0], int(cz) + margin + 1)
        y_lo = max(0, int(cy) - margin)
        y_hi = min(dist.shape[1], int(cy) + margin + 1)
        x_lo = max(0, int(cx) - margin)
        x_hi = min(dist.shape[2], int(cx) + margin + 1)

        zz_local, yy_local, xx_local = np.mgrid[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
        dist_from_center = np.sqrt((zz_local - cz)**2 + (yy_local - cy)**2 + (xx_local - cx)**2)
        sphere_mask_local = dist_from_center <= radius_vox

        sub_labels = bead_labels[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
        outside_sphere = (sub_labels == bead_id) & (~sphere_mask_local)
        n_removed_vox = int(np.sum(outside_sphere))

        if n_removed_vox > 0:
            sub_labels[outside_sphere] = 0
            bead_labels[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi] = sub_labels
            n_trimmed += 1
            print(f"    Bead {bead_id}: sphericity={sph:.2f}, trimmed {n_removed_vox:,} voxels "
                  f"(radius={radius_vox:.0f} vox)")

    print(f"  Trimmed {n_trimmed} non-spherical beads in {time.time()-t0:.1f}s")

# -- Feret's diameter helper ---------------------------------------------------
def _feret_directions(n):
    """Return n approximately uniform unit directions on the upper hemisphere
    (Fibonacci sphere). Feret is antipodal-symmetric so a half-sphere is enough.
    """
    if n <= 1:
        return np.array([[0.0, 0.0, 1.0]])
    phi = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n, dtype=np.float64)
    y = 1.0 - (i / (n - 1))                 # y in [0, 1]  (half-sphere)
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * i
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    d = np.stack([z, y, x], axis=1)         # order (Z, Y, X) to match coords
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d

_FERET_DIRS = _feret_directions(FERET_N_DIRECTIONS)
print(f"\nFeret directions: {_FERET_DIRS.shape[0]} unit vectors on upper hemisphere")

def mean_feret_diameter_um(coords, voxel_um=VOXEL_SIZE_UM, dirs=_FERET_DIRS):
    """Mean Feret diameter in um, averaged over all directions.
    coords: (N, 3) voxel indices (Z, Y, X) of the object.
    The +1 term accounts for finite voxel extent (range covers N voxels of
    size voxel_um each).
    """
    # (N,3) @ (3,K) -> (N,K)
    projs = coords @ dirs.T
    ranges_vox = projs.max(axis=0) - projs.min(axis=0) + 1.0
    return float(np.mean(ranges_vox)) * voxel_um

# -- Step 7: Measure each bead ------------------------------------------------
print("\nStep 7: Measuring each bead...")
t0 = time.time()

voxel_vol_um3 = VOXEL_SIZE_UM ** 3   # um3 per voxel

results = []
for bead_id in range(1, n_beads + 1):
    mask_i = (bead_labels == bead_id)
    n_vox = int(np.sum(mask_i))

    # Volume
    vol_um3 = n_vox * voxel_vol_um3
    vol_mm3 = vol_um3 / 1e9

    # Equivalent spherical diameter
    eq_diam_um = 2.0 * (3.0 * vol_um3 / (4.0 * np.pi)) ** (1.0/3.0)

    # Centroid (in voxel coordinates)
    coords_i = np.argwhere(mask_i)   # (N, 3) -> Z, Y, X
    centroid_z = float(np.mean(coords_i[:, 0]))
    centroid_y = float(np.mean(coords_i[:, 1]))
    centroid_x = float(np.mean(coords_i[:, 2]))

    # Bounding box dimensions
    bb_z = coords_i[:, 0].max() - coords_i[:, 0].min() + 1
    bb_y = coords_i[:, 1].max() - coords_i[:, 1].min() + 1
    bb_x = coords_i[:, 2].max() - coords_i[:, 2].min() + 1

    # Mean 3D Feret diameter (paper Section 3.5.4: mean caliper diameter)
    feret_mean_um = mean_feret_diameter_um(coords_i)

    # Sphericity estimate: compare volume to bounding-box sphere
    # True sphericity needs surface area (expensive), so use simplified version
    max_dim = max(bb_z, bb_y, bb_x) * VOXEL_SIZE_UM
    sphere_vol = (4.0/3.0) * np.pi * (max_dim/2.0)**3
    sphericity = vol_um3 / sphere_vol if sphere_vol > 0 else 0

    results.append({
        'bead_id': bead_id,
        'voxels': n_vox,
        'volume_mm3': vol_mm3,
        'eq_diameter_um': eq_diam_um,
        'feret_mean_um': feret_mean_um,
        'centroid_z_vox': centroid_z,
        'centroid_y_vox': centroid_y,
        'centroid_x_vox': centroid_x,
        'bbox_z': bb_z,
        'bbox_y': bb_y,
        'bbox_x': bb_x,
        'sphericity': sphericity,
        'wall_touching': 1 if bead_id in wall_touching_ids else 0,
    })

    if bead_id % 500 == 0:
        print(f"  Measured {bead_id}/{n_beads} beads...")

print(f"  All {n_beads} beads measured in {time.time()-t0:.1f}s")

# -- Step 8: Export CSV --------------------------------------------------------
print(f"\nStep 8: Exporting results to CSV...")

# Sort by volume (largest first)
results.sort(key=lambda r: r['voxels'], reverse=True)

with open(CSV_PATH, 'w') as f:
    headers = list(results[0].keys())
    f.write(';'.join(headers) + '\n')
    for r in results:
        vals = []
        for h in headers:
            v = r[h]
            if isinstance(v, int):
                vals.append(str(v))
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        f.write(';'.join(vals) + '\n')

print(f"  Saved: {CSV_PATH}")

# -- Step 9: Summary statistics ------------------------------------------------
print("\n" + "=" * 60)
print("BEAD ANALYSIS SUMMARY")
print("=" * 60)

voxels_all = [r['voxels'] for r in results]
diams_all  = [r['eq_diameter_um'] for r in results]
feret_all  = [r['feret_mean_um']  for r in results]
spher_all  = [r['sphericity'] for r in results]

print(f"  Total beads:        {n_beads}")
print(f"  Total glass voxels: {sum(voxels_all):,}")
print(f"")
print(f"  Equivalent diameter (um):")
print(f"    Min:    {min(diams_all):.1f}")
print(f"    Max:    {max(diams_all):.1f}")
print(f"    Mean:   {np.mean(diams_all):.1f}")
print(f"    Median: {np.median(diams_all):.1f}")
print(f"    Std:    {np.std(diams_all):.1f}")
print(f"")
print(f"  Mean Feret diameter (um):")
print(f"    Min:    {min(feret_all):.1f}")
print(f"    Max:    {max(feret_all):.1f}")
print(f"    Mean:   {np.mean(feret_all):.1f}")
print(f"    Median: {np.median(feret_all):.1f}")
print(f"    Std:    {np.std(feret_all):.1f}")
print(f"")
print(f"  Volume (mm3):")
print(f"    Total:  {sum(r['volume_mm3'] for r in results):.4f}")
print(f"    Mean:   {np.mean([r['volume_mm3'] for r in results]):.6f}")
print(f"")
print(f"  Sphericity:")
print(f"    Mean:   {np.mean(spher_all):.3f}")
print(f"    Median: {np.median(spher_all):.3f}")

# -- Step 10: PSD plot (same style as Camsizer psd_analysis.py) ----------------
print("\nStep 10: Saving particle size distribution plot...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# MATLAB-style settings (matching D:\Rolling\Camsizer\Neuer Ordner\psd_analysis.py)
FS = 30
LW = 3
PSD_COLOR = "#1f77b4"

plt.rcParams.update({
    "font.size"        : FS,
    "axes.titlesize"   : FS,
    "axes.labelsize"   : FS,
    "xtick.labelsize"  : FS,
    "ytick.labelsize"  : FS,
    "legend.fontsize"  : FS - 4,
    "lines.linewidth"  : LW,
    "axes.linewidth"   : 1.8,
    "xtick.major.width": 1.8,
    "ytick.major.width": 1.8,
    "xtick.major.size" : 8,
    "ytick.major.size" : 8,
    "xtick.minor.size" : 4,
    "ytick.minor.size" : 4,
    "grid.linewidth"   : 0.8,
    "grid.alpha"       : 0.4,
    "figure.facecolor" : "white",
    "axes.facecolor"   : "white",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

# --- Compute volume-weighted PSD from individual bead measurements ---
# Per paper Section 3.5.4, mean Feret diameter is preferred for irregular
# particles. Switch x-axis via USE_FERET_FOR_PSD.
if USE_FERET_FOR_PSD:
    diams = np.array(feret_all)
    diam_label = "Mean Feret diameter"
    psd_title_suffix = " (Feret's diameter)"
else:
    diams = np.array(diams_all)
    diam_label = "Equivalent diameter"
    psd_title_suffix = ""
vols  = np.array([r['volume_mm3'] for r in results])

# Create size bins
n_bins = 30
d_min = max(diams.min() - 50, 0)
d_max = diams.max() + 50
bin_edges = np.linspace(d_min, d_max, n_bins + 1)
bin_lo = bin_edges[:-1]
bin_hi = bin_edges[1:]
dx = bin_hi - bin_lo
x_mid = 0.5 * (bin_lo + bin_hi)

# Volume fraction per bin: sum of volumes of beads in each bin
p3 = np.zeros(n_bins)
for i in range(n_bins):
    mask_bin = (diams >= bin_lo[i]) & (diams < bin_hi[i])
    p3[i] = vols[mask_bin].sum()

# Normalize to 100%
total_vol = p3.sum()
if total_vol > 0:
    p3 = 100.0 * p3 / total_vol

Q3 = np.cumsum(p3)          # cumulative [%]
q3 = p3 / dx                # density [%/um]

# --- Compute D10, D50, D90 from cumulative ---
def interp_dx(Q3_arr, x_arr, target_pct):
    idx = np.searchsorted(Q3_arr, target_pct)
    if idx == 0:
        return x_arr[0]
    if idx >= len(Q3_arr):
        return x_arr[-1]
    # linear interpolation
    frac = (target_pct - Q3_arr[idx-1]) / (Q3_arr[idx] - Q3_arr[idx-1])
    return x_arr[idx-1] + frac * (x_arr[idx] - x_arr[idx-1])

D10 = interp_dx(Q3, x_mid, 10)
D50 = interp_dx(Q3, x_mid, 50)
D90 = interp_dx(Q3, x_mid, 90)
Span = (D90 - D10) / D50 if D50 > 0 else 0

print(f"  D10 = {D10:.1f} um")
print(f"  D50 = {D50:.1f} um")
print(f"  D90 = {D90:.1f} um")
print(f"  Span = {Span:.3f}")

# --- Export PSD statistics to text file ---
stats_path = os.path.join(OUT_DIR, 'bead_psd_statistics.txt')
with open(stats_path, 'w') as f:
    f.write("Glass Bead PSD Statistics (CT micro-CT analysis)\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total beads:          {n_beads}\n")
    f.write(f"Total glass volume:   {sum(r['volume_mm3'] for r in results):.4f} mm3\n\n")
    f.write(f"Size measure used for PSD: {diam_label}\n")
    f.write(f"D10:                  {D10:.1f} um\n")
    f.write(f"D50:                  {D50:.1f} um\n")
    f.write(f"D90:                  {D90:.1f} um\n")
    f.write(f"Span (D90-D10)/D50:   {Span:.3f}\n\n")
    f.write(f"Equivalent diameter:\n")
    f.write(f"  Min:                {min(diams_all):.1f} um\n")
    f.write(f"  Max:                {max(diams_all):.1f} um\n")
    f.write(f"  Mean:               {np.mean(diams_all):.1f} um\n")
    f.write(f"  Median:             {np.median(diams_all):.1f} um\n")
    f.write(f"  Std:                {np.std(diams_all):.1f} um\n\n")
    f.write(f"Mean Feret diameter ({FERET_N_DIRECTIONS} directions):\n")
    f.write(f"  Min:                {min(feret_all):.1f} um\n")
    f.write(f"  Max:                {max(feret_all):.1f} um\n")
    f.write(f"  Mean:               {np.mean(feret_all):.1f} um\n")
    f.write(f"  Median:             {np.median(feret_all):.1f} um\n")
    f.write(f"  Std:                {np.std(feret_all):.1f} um\n\n")
    f.write(f"Volume per bead:\n")
    f.write(f"  Mean:               {np.mean([r['volume_mm3'] for r in results]):.6f} mm3\n\n")
    f.write(f"Sphericity:\n")
    f.write(f"  Mean:               {np.mean(spher_all):.3f}\n")
    f.write(f"  Median:             {np.median(spher_all):.3f}\n")
print(f"  Saved: {stats_path}")

# --- PSD plot: q3 bars (left) + Q3 line (right) ---
fig, ax1 = plt.subplots(figsize=(14, 9))

# Bar chart for q3
ax1.bar(x_mid, q3, width=dx * 0.85, color=PSD_COLOR, alpha=0.55)
ax1.set_xlabel(f"{diam_label} (µm)", labelpad=10)
ax1.set_ylabel(r"Volume density  $q_3$  (% µm$^{-1}$)", color=PSD_COLOR, labelpad=10)
ax1.tick_params(axis="y", labelcolor=PSD_COLOR)
ax1.set_xlim(d_min, d_max)
ax1.set_ylim(bottom=0)

# Grid on both axes
ax1.grid(True, which="major", axis="both", linewidth=0.8, alpha=0.4)

# Box: show all 4 spines
ax1.spines["top"].set_visible(True)
ax1.spines["right"].set_visible(True)
ax1.spines["top"].set_linewidth(1.8)
ax1.spines["right"].set_linewidth(1.8)

# Cumulative Q3 on right axis
ax2 = ax1.twinx()
ax2.plot(x_mid, Q3, color="black", linewidth=LW + 1, linestyle="-")
ax2.set_ylabel(r"Cumulative undersize  $Q_3$  (%)", color="black", labelpad=10)
ax2.tick_params(axis="y", labelcolor="black")
ax2.set_ylim(0, 100)
ax2.yaxis.set_major_locator(plt.MultipleLocator(10))
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_linewidth(1.8)
ax2.spines["top"].set_visible(True)
ax2.spines["top"].set_linewidth(1.8)

# D10 / D50 / D90 markers with distinct colors
d_colors = {"D10": "#2ca02c", "D50": "#d62728", "D90": "#9467bd"}   # green, red, purple
for xd, key, ls in [
    (D10, "D10", "--"),
    (D50, "D50", "-"),
    (D90, "D90", ":"),
]:
    ax2.axvline(xd, color=d_colors[key], linewidth=2.2, linestyle=ls)

for pct in [10, 50, 90]:
    ax2.axhline(pct, color="dimgray", linewidth=1.0, linestyle=":")

# Legend: 2 rows below the plot
handles = [
    # Row 1
    mpatches.Patch(color=PSD_COLOR, alpha=0.6, label=r"$q_3(x)$ – volume density"),
    Line2D([0], [0], color="black", linewidth=LW + 1, label=r"$Q_3(x)$ – cumulative"),
    # Row 2
    Line2D([0], [0], color=d_colors["D10"], linewidth=2.2, linestyle="--",
           label=f"$D_{{10}}$ = {D10:.0f} µm"),
    Line2D([0], [0], color=d_colors["D50"], linewidth=2.2, linestyle="-",
           label=f"$D_{{50}}$ = {D50:.0f} µm"),
    Line2D([0], [0], color=d_colors["D90"], linewidth=2.2, linestyle=":",
           label=f"$D_{{90}}$ = {D90:.0f} µm"),
]
fig.legend(handles=handles, loc="lower center", ncol=3,
           fontsize=FS - 4, frameon=True, framealpha=0.85, edgecolor="gray",
           bbox_to_anchor=(0.5, -0.14))

plt.title(f"Particle Size Distribution (CT — Glass Beads){psd_title_suffix}", pad=18)
plt.tight_layout()
plot_path = os.path.join(OUT_DIR, 'bead_size_distribution.png')
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {plot_path}")

# Reset rcParams for subsequent plots
plt.rcdefaults()

# -- Step 11: Save labeled slice for visual check ------------------------------
apply_style()
mid_z = bead_labels.shape[0] // 2
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Left: original segmentation
axes[0].imshow(seg[mid_z], cmap='gray', origin='upper', aspect='equal')
axes[0].set_title(f'Segmentation (Z={mid_z})', fontsize=FS - 4, pad=10)
axes[0].set_xlabel('X (voxels)', fontsize=FS - 6)
axes[0].set_ylabel('Y (voxels)', fontsize=FS - 6)
style_axes(axes[0])

# Right: bead labels (random colormap with dark background)
labeled_slice = bead_labels[mid_z]
np.random.seed(42)
colors = np.random.rand(n_beads + 1, 3)
colors[0] = [0.05, 0.05, 0.05]   # near-black background
# Make colors more vivid (push toward saturation)
for i in range(1, len(colors)):
    colors[i] = np.clip(colors[i] * 1.4, 0, 1)
rgb = colors[labeled_slice]
axes[1].imshow(rgb, origin='upper', aspect='equal')
axes[1].set_title(f'Individual beads (Z={mid_z}, n={n_beads})', fontsize=FS - 4, pad=10)
axes[1].set_xlabel('X (voxels)', fontsize=FS - 6)
axes[1].set_ylabel('Y (voxels)', fontsize=FS - 6)
style_axes(axes[1])

fig.suptitle('Bead Labeling Check', fontsize=FS, fontweight='bold', y=1.02)
plt.tight_layout()
check_path = os.path.join(OUT_DIR, 'bead_labels_check.png')
save_fig(fig, check_path)
plt.rcdefaults()
print(f"  Saved: {check_path}")

# -- Step 12: Create MultiROI in Dragonfly for bead visualization ---------------
# MultiROI = one object containing all 282 labeled beads (not 282 separate ROIs).
# Workflow: publish temp labeled channel -> getLabelization -> MultiROI -> delete temp channel.
if PUBLISH_MULTI_ROI:
    try:
        from ORSModel import createChannelFromNumpyArray, ROI, orsColor, Channel

        print("\nStep 12: Creating MultiROI in Dragonfly...")

        # -- A: find CT channel ---------------------------------------------------
        ct_channel = None
        cls_name = Channel.getClassNameStatic()
        for ch in Channel.getAllObjectsOfClass(cls_name):
            title = ''
            for tmethod in ['getTitle', 'getPrivateTitle', 'getName', 'getLabel']:
                if hasattr(ch, tmethod):
                    try:
                        title = getattr(ch, tmethod)()
                        if title and 'orsObj' not in str(title):
                            break
                    except Exception:
                        pass
            if CT_CHANNEL_NAME in str(title):
                ct_channel = ch
                print(f"  Found CT channel: {title}")
                break

        if ct_channel is None:
            raise RuntimeError("CT channel not found.")

        # -- B: create temp labeled channel with CT coordinates -------------------
        label_ch = createChannelFromNumpyArray(bead_labels.astype(np.uint16))
        label_ch.setTitle('_bead_labels_temp')

        # Copy physical coordinates from CT channel so it aligns
        try:
            label_ch.setXSpacing(ct_channel.getXSpacing())
            label_ch.setYSpacing(ct_channel.getYSpacing())
            label_ch.setZSpacing(ct_channel.getZSpacing())
            label_ch.setOrigin(ct_channel.getOrigin())
            print("  Copied CT channel coordinates to label channel.")
        except Exception as e_coord:
            print(f"  WARNING: could not copy coordinates ({e_coord})")

        label_ch.publish()
        print("  Temporary label channel published.")

        # -- C: create MultiROI from labeled channel --------------------------------
        multi_roi = None
        nz_b, ny_b, nx_b = bead_labels.shape

        # Try 1: getAsMultiROI(inOutStructuredGrid, IProgress) — 2 args
        try:
            multi_roi = label_ch.getAsMultiROI(None, None)
            print("  MultiROI created via getAsMultiROI(None, None)")
        except Exception as e:
            print(f"  getAsMultiROI(None, None) failed: {e}")

        # Try 2: full getLabelization — 13 args
        if multi_roi is None:
            try:
                multi_roi = label_ch.getLabelization(
                    0, 0, 0,                              # minX, minY, minZ
                    nx_b - 1, ny_b - 1, nz_b - 1,        # maxX, maxY, maxZ
                    0,                                     # iTIndex (time index)
                    1, float(n_beads),                     # min, max label values
                    True,                                  # considerDiagonal
                    None,                                  # IProgress
                    None,                                  # pInVolumeROI
                    None                                   # pOutData
                )
                print("  MultiROI created via full getLabelization(13 args)")
            except Exception as e:
                print(f"  full getLabelization(13 args) failed: {e}")

        if multi_roi is not None:
            multi_roi.setTitle('Glass_Beads_MultiROI')
            # Assign distinct colors to each label (otherwise all black)
            try:
                multi_roi.assignDefaultColors()
                print("  Assigned default colors to labels.")
            except Exception as e_col:
                print(f"  assignDefaultColors() failed: {e_col}")
            multi_roi.publish()
            print(f"  Published: Glass_Beads_MultiROI ({n_beads} beads)")

            # Delete temp channel
            try:
                label_ch.deleteObject()
                print("  Temporary label channel deleted.")
            except Exception:
                try:
                    label_ch.remove()
                except Exception:
                    print("  WARNING: could not delete temp channel.")
        else:
            # MultiROI method not found - keep the labeled channel instead
            label_ch.setTitle('Glass_Beads_Labeled')
            print(f"  MultiROI method not available. Kept labeled channel: Glass_Beads_Labeled")
            print(f"  Use Dragonfly's Connected Components tool on this channel to create MultiROI.")

    except ImportError:
        print("\nStep 12: Skipped (not running inside Dragonfly)")
    except RuntimeError as e:
        print(f"\nStep 12 STOPPED: {e}")
    except Exception as e:
        import traceback
        print(f"\nStep 12 ERROR: {e}")
        traceback.print_exc()
else:
    print("\nStep 12: Skipped (PUBLISH_MULTI_ROI = False)")

print("\n" + "=" * 60)
print("DONE. Check bead_labels_check.png to verify bead separation.")
print(f"Adjust MARKER_MIN_DIST (currently {MARKER_MIN_DIST}) if beads are:")
print(f"  - over-split (too many small pieces) -> increase MARKER_MIN_DIST")
print(f"  - under-split (merged beads)         -> decrease MARKER_MIN_DIST")
print("=" * 60)
