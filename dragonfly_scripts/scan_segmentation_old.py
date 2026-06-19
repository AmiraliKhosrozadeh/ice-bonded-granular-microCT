"""
Segmentation of glass bead / ice / trapped air CT dataset
Phases: outside air | trapped air | ice | glass beads | aluminum

Two specimen-mask methods are implemented (choose via MASK_METHOD):
  'contour'  -> automatic per-slice OpenCV contour detection on an Otsu-
                binarized slice (Ben Elhadj Hamida 2024 workflow).
  'cylinder' -> user-defined tilted cylinder from Dragonfly caps.

Ice/air threshold (T_AIR_ICE) can be picked automatically with Otsu
(USE_OTSU_AIR_ICE = True) as recommended in the paper for the hard
saline/air (here: ice/air) differentiation, or kept as manual.

Thresholds from scan-2 histogram analysis (tuned against user ground truth):
  Air  (all):   gray < 6200
  Ice:          6200 <= gray < 16544
  Glass beads:  16544 <= gray < 37000
  Aluminum:     gray >= 37000
"""
# ===================================================================
# SCAN 1 / T7 1700 ┬Ám ÔÇö scaffolded by _scaffold.py.
# EDIT BEFORE RUNNING (measure in Dragonfly):
#   FIRST_SLICE / LAST_SLICE      ÔÇö particle Z range in THIS stack
#   X_MIN, X_MAX, Y_MIN, Y_MAX    ÔÇö XY crop (cover specimen in ALL scans)
#   CYL_CAP1_UM, CYL_CAP2_UM      ÔÇö cylinder caps (world coords, ┬Ám)
#   CYL_RADIUS_UM                  ÔÇö cylinder radius (┬Ám)
#   CT_ORIGIN_UM                   ÔÇö from ct_channel.getOrigin()
#   T_AIR_ICE, T_ICE_GLASS, T_GLASS_AL  ÔÇö thresholds from histogram_analysis
#   CT_CHANNEL_NAME                ÔÇö raw channel name in Dragonfly
# ===================================================================


import numpy as np
import os, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage


def label_counts(label_image):
    """Return (label_ids, counts) for nonzero labels without using bincount."""
    labels = label_image[label_image != 0]
    if labels.size == 0:
        return np.empty(0, dtype=labels.dtype), np.empty(0, dtype=np.int64)

    labels = labels.astype(np.int64, copy=True)
    labels.sort()
    changes = np.empty(labels.shape, dtype=bool)
    changes[0] = True
    changes[1:] = labels[1:] != labels[:-1]
    unique_labels = labels[changes]
    change_idx = np.flatnonzero(changes)
    counts = np.diff(np.concatenate((change_idx, [labels.size])))
    return unique_labels, counts

# -- Settings (SCAN 1 ÔÇö T5_HR 1700 ┬Ám high-roughness experiment) -------------
# Raw stack: 1432 slices. Dragonfly channel (from info.txt) shape XYZ =
# (612, 613, 1006), loaded starting at raw slice 0235.
# NOTE: the loader below uses INCLUSIVE end indices (X_MAX/Y_MAX/LAST_SLICE
# are the last voxel to include). So for channel X voxels wide we need
# X_MAX = X_MIN + X - 1. A 1-voxel miscount makes Python's volume a voxel
# larger than the channel and shifts every ROI imported back into Dragonfly.
FIRST_SLICE = 200      # first loaded slice (inclusive) ÔÇö channel title ends in _xy_0080
LAST_SLICE  = 1313    # last loaded slice (inclusive)  80..1026 = 947 slices

TIFF_DIR    = r'C:\Users\Lennard\Desktop\Data_Work_Part\alumina\Alumina_75_1000_T5\Alumina_75_1000_T5_01'
FILE_PREFIX = 'Alumina-75-1000-T5_100XXL_uc_xy_'

# X/Y crop ÔÇö matches the Dragonfly createDatasetFromFiles crop. Dragonfly
# reports origin at voxel CENTER, so voxel index = round(origin/voxel + 0.5).
# For scan 1 origin (10934.189, 1102.087) / 24.7660229 = (441.52, 44.50) ->
# Dragonfly minX = 442, minY = 45.
# Getting this off by one shifts every ROI by one voxel.
X_MIN, X_MAX = 419, 1030   # 608 voxels wide; round(14252.833 / 24.7660 + 0.5) = 576
Y_MIN, Y_MAX = 1, 587   # 608 voxels tall; round(4371.199  / 24.7660 + 0.5) = 177

VOXEL_SIZE_UM = 24.7660229   # um per voxel (isotropic)

# -- Specimen-mask method ------------------------------------------------------
# 'contour'  : HYBRID ÔÇö user-defined cylinder (rough bound) + automatic
#              per-slice contour detection on the ice+glass solid material
#              within the cylinder. The cylinder prevents the contour from
#              leaking outside the specimen; the contour tightens the mask
#              to the actual ice+glass boundary on each slice.
# 'cylinder' : user-defined tilted cylinder only (no contour refinement).
MASK_METHOD = 'cylinder'   # user draws cylinder inside specimen; no contour tightening needed

# Otsu-based ice/air threshold (inside specimen). The paper (Section 3.5.6.1)
# found that Otsu systematically overestimates the air content and preferred
# MANUAL thresholding, so this defaults to False. Flip to True only as a
# comparison run ÔÇö manual T_AIR_ICE from histogram_analysis.py is more reliable.
USE_OTSU_AIR_ICE = False

# Contour-detection tuning (used only when MASK_METHOD = 'cylinder'   # user draws cylinder inside specimen; no contour tightening needed).
# Target: the ice+glass cylinder (surrounded by an air gap, then the Al holder).
# Strategy: threshold ice+glass only, close gaps between beads to unite them,
# fill interior holes (trapped air pockets), keep largest CC, trace outline.
CONTOUR_CLOSING_R = 10     # closing radius (px). Must bridge bead-to-bead gaps
                           # but stay smaller than the specimen-to-Al-wall gap.
CONTOUR_MIN_AREA_FRAC = 0.02  # discard contours smaller than this fraction of
                              # the largest contour (suppresses specks)
# Cylinder-fallback rule: if the contour blob on a slice covers LESS than this
# fraction of the cylinder area on that same slice, fall back to the cylinder
# mask for that slice. Protects sparse top/bottom slices where beads are
# disconnected and the "largest CC" rule would throw away real glass.
CONTOUR_FALLBACK_FRAC = 0.30
# Safety margin (voxels) added around the contour blob to avoid trimming real
# peripheral beads. Dilates the blob, still clipped to the cylinder prior.
CONTOUR_DILATE_R = 2   # minimal safety margin
CONTOUR_SHRINK_VOX = 3  # extra 2D erosion after contour to pull mask inside specimen edge

# Cylinder parameters (world coordinates in um) ÔÇö MEASURE FRESH IN DRAGONFLY
# for scan 2 after compression, the specimen has moved. Defaults below mirror
# scan 1 as a starting point but WILL be wrong until remeasured.
# Steps in Dragonfly:
#   1) Load scan 2 TIFF stack.
#   2) Make a rough tilted-cylinder ROI covering the ice+glass specimen.
#   3) Read the two cap coordinates + radius from the ROI inspector.
#   4) Paste them here.
# TODO: measure these in Dragonfly (draw a tilted cylinder around the
# ice+glass specimen; read the two cap world-coords + radius from the ROI
# inspector). The values below are T7-scan-2 leftovers and MUST be replaced.
CYL_CAP1_UM   = (22711.14, 13516.13,   -255.44)   # <-- REPLACE with T5_HR measurement
CYL_CAP2_UM   = (21523.09, 13516.13, 23274.15)  # <-- REPLACE
CYL_RADIUS_UM = 5320.52                         # <-- REPLACE

# Safety margin on the cylinder radius ÔÇö disabled to honor the user-drawn
# cylinder exactly and prevent glass ROI from leaking onto the aluminum wall.
CYL_RADIUS_MARGIN_UM = 0.0

# CT channel origin in um ÔÇö from info.txt (scan{N}_dragonfly/info.txt).
# Dragonfly reports this for the loaded (pre-cropped) channel, so world
# coords measured in Dragonfly convert directly to Python voxel indices:
#       voxel = (world_um - CT_ORIGIN_UM) / VOXEL_SIZE_UM
# with NO additional crop offset, provided Python's X_MIN/Y_MIN/FIRST_SLICE
# above match the same channel.
CT_ORIGIN_UM = (14252.833, 4371.199, -12.383)

# Crop offset in cropped voxels ÔÇö 0 because CT_ORIGIN_UM is already relative
# to the cropped channel. (Set to (X_MIN, Y_MIN, FIRST_SLICE) only if your
# CT_ORIGIN_UM was reported for the UNCROPPED full dataset instead.)
CROP_OFFSET_VOX = (0, 0, 0)
_c1_full = tuple((c - o) / VOXEL_SIZE_UM for c, o in zip(CYL_CAP1_UM, CT_ORIGIN_UM))
_c2_full = tuple((c - o) / VOXEL_SIZE_UM for c, o in zip(CYL_CAP2_UM, CT_ORIGIN_UM))
CYL_CAP1_VOX = tuple(f - s for f, s in zip(_c1_full, CROP_OFFSET_VOX))
CYL_CAP2_VOX = tuple(f - s for f, s in zip(_c2_full, CROP_OFFSET_VOX))
CYL_RADIUS_VOX = (CYL_RADIUS_UM + CYL_RADIUS_MARGIN_UM) / VOXEL_SIZE_UM

OUT_DIR = os.path.join(os.path.dirname(r'C:\Users\Lennard\Micro-CT\dragonfly_scripts\scan_segmentation_old.py'),
                       'scan_segmentation_old_output', 'segmentation')
os.makedirs(OUT_DIR, exist_ok=True)

# Dragonfly CT channel search string (for Step 7 ROI creation)
CT_CHANNEL_NAME = 'Glass-75-1700-T5-HR_100XXL_uc_xy_'  # prefix-only; matches any loaded slice index

# Median filter size for noise reduction (0 = skip)
MEDIAN_SIZE = 3

# Thresholds (T5_HR scan 1) ÔÇö from scan01_histogram_analysis.py (info.txt).
# Peaks detected: Air=3279, Ice=11170, Glass=14851. No distinct Al peak;
# tail estimate Al~20806. T_GLASS_AL set at suggested midpoint (17828).
# If the segmentation later shows glass mis-classified as Al or vice versa,
# raise T_GLASS_AL toward the tail estimate.
T_AIR_ICE   = 7000    # midpoint Air  <-> Ice
T_ICE_GLASS = 18032.7   # midpoint Ice  <-> Glass
T_GLASS_AL  = 47275   # midpoint Glass <-> (Al tail)

# Noise removal: minimum connected-component size (voxels) to keep
# Clusters smaller than this are reclassified as their surrounding phase
MIN_ICE_VOXELS   = 2000     # remove ice specks smaller than this
MIN_GLASS_VOXELS = 10      # only remove tiny noise (real beads can be ~500 vox)
MIN_TRAPPED_AIR_VOXELS = 0 # 0 = no cleaning (set >0 to also clean trapped air)

# Label values
LABEL_OUTSIDE_AIR  = 0
LABEL_TRAPPED_AIR  = 1
LABEL_ICE          = 2
LABEL_GLASS        = 3
LABEL_ALUMINUM     = 4

LABEL_NAMES = {
    0: 'Outside air',
    1: 'Trapped air (inside)',
    2: 'Ice',
    3: 'Glass beads',
    4: 'Aluminum',
}

# -- Load TIFF reader -----------------------------------------------------------
try:
    import tifffile
    def read_tif(path):
        return tifffile.imread(path)
    def save_tif(path, arr):
        tifffile.imwrite(path, arr)
except ImportError:
    from PIL import Image
    def read_tif(path):
        return np.array(Image.open(path))
    def save_tif(path, arr):
        Image.fromarray(arr).save(path)

# -- 1. Load full cropped volume -----------------------------------------------
print("=" * 60)
print("Step 1: Loading full cropped volume...")
print(f"  Files: {FILE_PREFIX}{FIRST_SLICE:04d}.tif  ->  {FILE_PREFIX}{LAST_SLICE:04d}.tif")
print(f"  Crop:  X {X_MIN}-{X_MAX}  Y {Y_MIN}-{Y_MAX}  ({X_MAX-X_MIN+1} x {Y_MAX-Y_MIN+1} px)")
t0 = time.time()

slices = []
loaded = 0
for idx in range(FIRST_SLICE, LAST_SLICE + 1):
    fname = os.path.join(TIFF_DIR, f'{FILE_PREFIX}{idx:04d}.tif')
    if not os.path.exists(fname):
        print(f"  WARNING: missing {fname}")
        continue
    img = read_tif(fname)
    img = img[Y_MIN:Y_MAX+1, X_MIN:X_MAX+1]
    slices.append(img)
    loaded += 1
    if loaded % 100 == 0:
        print(f"  Loaded {loaded} slices...")

volume = np.stack(slices, axis=0)   # shape: (Z, Y, X)
print(f"Volume shape : {volume.shape}  (Z, Y, X)")
print(f"Dtype        : {volume.dtype}")
print(f"Loaded in {time.time()-t0:.1f}s")

nz, ny, nx = volume.shape

# Keep a copy of one raw mid-slice for the contour-detection QC figure
# (Fig 3.15 of Ben Elhadj Hamida 2024). Done before filtering.
_qc_mid_z = nz // 2
_qc_raw_slice = volume[_qc_mid_z].copy()

# -- 1b. 3D median filter (BEFORE contour detection) ---------------------------
# Paper workflow (Section 3.5.5): median filter -> binarize -> contour detect.
# Filtering first produces a cleaner per-slice Otsu threshold and stable
# contours. The filtered volume is also used directly for phase thresholding.
if MEDIAN_SIZE > 0:
    print(f"\nStep 1b: Applying 3D median filter (size={MEDIAN_SIZE})...")
    t_filt = time.time()
    volume = ndimage.median_filter(volume, size=MEDIAN_SIZE)
    print(f"  Filtered in {time.time()-t_filt:.1f}s")
else:
    print("\nStep 1b: Skipping noise filter (MEDIAN_SIZE=0)")

# -- 2. Build specimen mask ----------------------------------------------------
# Two methods are supported. The mask is a 3D boolean array that separates
# the specimen (True) from outside air (False).
yy, xx = np.mgrid[0:ny, 0:nx]

# Build the user-defined tilted cylinder. Used directly as the mask in
# 'cylinder' mode, and as a ROUGH BOUNDING PRIOR in 'contour' mode to keep
# the contour detection from leaking outside the specimen.
print("\nStep 2a: Building tilted cylinder (user-defined caps, rough bound)...")
print(f"  Cap1 (vox): X={CYL_CAP1_VOX[0]:.1f}  Y={CYL_CAP1_VOX[1]:.1f}  Z={CYL_CAP1_VOX[2]:.1f}")
print(f"  Cap2 (vox): X={CYL_CAP2_VOX[0]:.1f}  Y={CYL_CAP2_VOX[1]:.1f}  Z={CYL_CAP2_VOX[2]:.1f}")
print(f"  Radius: {CYL_RADIUS_VOX:.1f} vox")

cyl_mask_3d = np.zeros((nz, ny, nx), dtype=bool)
cap1_z = CYL_CAP1_VOX[2]
cap2_z = CYL_CAP2_VOX[2]
# Cylinder Z bound ALWAYS covers the full loaded stack, plus a small
# margin either side of the user-drawn caps. The X/Y centerline is
# extrapolated along the two caps for slices outside the drawn range.
# Punch / aluminium voxels in the extrapolated region are removed
# later by the AL_BUFFER_VOX patch, so a short user cylinder is OK.
Z_PAD_LOW  = 0      # cylinder extension at head / punch side (keep 0 to avoid punch)
Z_PAD_HIGH = 200    # manual lower bound for bottom extension; auto-raised to cover full stack
_z_hi_manual = max(cap1_z, cap2_z) + Z_PAD_HIGH
_z_hi = max(_z_hi_manual, nz - 1)   # always cover the full stack bottom
_z_lo = min(cap1_z, cap2_z) - Z_PAD_LOW
for z in range(nz):
    if z < _z_lo or z > _z_hi:
        continue
    t = (z - cap1_z) / (cap2_z - cap1_z)
    cx = CYL_CAP1_VOX[0] + t * (CYL_CAP2_VOX[0] - CYL_CAP1_VOX[0])
    cy = CYL_CAP1_VOX[1] + t * (CYL_CAP2_VOX[1] - CYL_CAP1_VOX[1])
    dist2 = (xx - cx)**2 + (yy - cy)**2
    cyl_mask_3d[z] = dist2 <= CYL_RADIUS_VOX**2
print(f"  X shift top->bottom: {CYL_CAP1_VOX[0]-CYL_CAP2_VOX[0]:.1f} vox")

# --- Aluminum-first subtraction -------------------------------------------
# Detect aluminium in the whole volume, dilate by AL_EXCLUDE_DILATE_VOX,
# and subtract from cyl_mask_3d. This is the USER-PROPOSED preprocessing:
# if the user-drawn cylinder encompasses the aluminium holder, this step
# removes the holder + a safety shell from the analysis region so the
# downstream contour detection sees only the glass / ice / air inside.
AL_EXCLUDE_DILATE_VOX = 10
_al_mask = (volume >= T_GLASS_AL)
if AL_EXCLUDE_DILATE_VOX > 0 and _al_mask.any():
    _al_grown = ndimage.binary_dilation(_al_mask, iterations=AL_EXCLUDE_DILATE_VOX)
    _before = int(cyl_mask_3d.sum())
    cyl_mask_3d = cyl_mask_3d & ~_al_grown
    _after = int(cyl_mask_3d.sum())
    print(f"  Al-first subtraction ({AL_EXCLUDE_DILATE_VOX} vox dilation): "
          f"removed {_before - _after:,} aluminum+shell voxels "
          f"({100*(_before - _after)/max(_before,1):.1f}% of cylinder)")

# --- Auto-drop non-specimen slices ------------------------------------------
# A slice belongs to the specimen only if its ice+glass fraction inside the
# cylinder is at least SPECIMEN_MIN_FRAC of the cylinder cross-section.
# Below that, the slice is almost certainly punch / pure aluminium / air and
# gets removed from cyl_mask_3d. Avoids the need to manually measure where
# the punch ends or the specimen bottom begins.
AIR_MIN_FRAC = 0.01          # 1 % of cylinder area must be air
SPECIMEN_MIN_FRAC = 0.10     # 10 % must also be ice+glass
_dropped = 0
for z in range(nz):
    cyl_area = int(cyl_mask_3d[z].sum())
    if cyl_area == 0:
        continue
    air = int(np.sum((volume[z] < T_AIR_ICE) & cyl_mask_3d[z]))
    ice_glass = int(np.sum(
        (volume[z] >= T_AIR_ICE) & (volume[z] < T_GLASS_AL) & cyl_mask_3d[z]
    ))
    air_frac       = air / cyl_area
    ice_glass_frac = ice_glass / cyl_area
    # Slice is NOT specimen if it lacks porosity (air gaps) even if
    # 'ice+glass' gray range is full ÔÇö this catches the compression punch
    # whose gray values happen to lie in the glass band.
    if air_frac < AIR_MIN_FRAC or ice_glass_frac < SPECIMEN_MIN_FRAC:
        cyl_mask_3d[z] = False
        _dropped += 1
if _dropped:
    print(f"  Auto-dropped {_dropped} non-specimen slices "
          f"(air < {int(AIR_MIN_FRAC*100)}% or ice+glass < "
          f"{int(SPECIMEN_MIN_FRAC*100)}% of cylinder area)")
print(f"  Cylinder voxels after auto-drop: {int(np.sum(cyl_mask_3d)):,}")
print(f"  Cylinder voxels: {int(np.sum(cyl_mask_3d)):,}")

if MASK_METHOD == 'cylinder':
    print("\nStep 2b: Using cylinder as specimen mask (no contour refinement).")
    specimen_mask = cyl_mask_3d & (volume < T_GLASS_AL)  # exclude aluminum wall voxels

elif MASK_METHOD == 'contour':
    # HYBRID: cylinder (rough prior) + per-slice contour detection that
    # tightens the mask to the actual ice+glass boundary.
    # Geometry (inside -> outside): ice+glass specimen -> air gap ->
    # aluminum wall -> outside air.
    #
    # Per slice, inside the cylinder:
    #   1) solid = voxels with T_AIR_ICE <= gray < T_GLASS_AL   (ice + glass)
    #   2) morphological close                                   (unite beads)
    #   3) fill interior holes                                   (capture
    #      trapped-air pockets INSIDE the specimen)
    #   4) keep the largest connected component                  (drop specks)
    #   5) the filled blob IS the specimen mask for that slice.
    import cv2

    print("\nStep 2b: Hybrid contour detection "
          "(cylinder prior + ice+glass boundary)...")
    specimen_mask = np.zeros((nz, ny, nx), dtype=bool)
    close_kernel_cv = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * CONTOUR_CLOSING_R + 1, 2 * CONTOUR_CLOSING_R + 1)
    ) if CONTOUR_CLOSING_R > 0 else None

    empty_slices = 0
    fallback_slices = 0
    t_ctr = time.time()
    _qc_contours = None
    _qc_filtered_slice = None
    dilate_kernel_cv = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * CONTOUR_DILATE_R + 1, 2 * CONTOUR_DILATE_R + 1)
    ) if CONTOUR_DILATE_R > 0 else None
    for z in range(nz):
        cyl_z = cyl_mask_3d[z]
        cyl_area = int(cyl_z.sum())
        if cyl_area == 0:
            empty_slices += 1
            continue

        # 1) Binarize the solid material (ice + glass) WITHIN the cylinder
        solid_bin = (
            (volume[z] >= T_AIR_ICE) & (volume[z] < T_GLASS_AL) & cyl_z
        ).astype(np.uint8) * 255

        # 2) Close so touching/near beads merge with ice into one blob
        if close_kernel_cv is not None:
            solid_bin = cv2.morphologyEx(solid_bin, cv2.MORPH_CLOSE,
                                         close_kernel_cv)

        # 3) Fill interior holes (trapped air pockets inside specimen) via
        #    flood-fill the outside, then invert + OR.
        h, w = solid_bin.shape
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        outside = solid_bin.copy()
        cv2.floodFill(outside, ff_mask, (0, 0), 255)
        holes = cv2.bitwise_not(outside)
        filled = cv2.bitwise_or(solid_bin, holes)

        # 4) Keep every connected component whose area is at least
        #    CONTOUR_MIN_AREA_FRAC of the largest CC. This preserves
        #    multiple specimen clusters (e.g. fracture, two bead piles).
        n_cc, cc_lbl, cc_stats, _ = cv2.connectedComponentsWithStats(filled)
        use_fallback = False
        if n_cc <= 1:
            use_fallback = True
        else:
            areas_cc = cc_stats[1:, cv2.CC_STAT_AREA]
            largest_area = int(areas_cc.max())
            keep_ids = 1 + np.where(areas_cc >= CONTOUR_MIN_AREA_FRAC * largest_area)[0]
            blob = np.isin(cc_lbl, keep_ids)
            # If the kept union is tiny vs the cylinder, beads are too sparse
            # for contour detection to be reliable. Fall back to the cylinder.
            if blob.sum() < CONTOUR_FALLBACK_FRAC * cyl_area:
                use_fallback = True

        if use_fallback:
            # Use the cylinder prior directly on this slice ÔÇö preserves real
            # glass voxels that contour detection would have discarded.
            _mask_z = cyl_z & (volume[z] < T_GLASS_AL)  # exclude aluminum wall voxels (fallback slices)
            if CONTOUR_SHRINK_VOX > 0 and _mask_z.any():
                _shrink_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * CONTOUR_SHRINK_VOX + 1, 2 * CONTOUR_SHRINK_VOX + 1))
                _mask_u8 = cv2.erode((_mask_z.astype(np.uint8) * 255), _shrink_kernel)
                _mask_z = _mask_u8 > 0
            specimen_mask[z] = _mask_z
            fallback_slices += 1
            continue

        # 5) Dilate the blob by CONTOUR_DILATE_R voxels to capture peripheral
        #    beads, then clip to the cylinder as a final safety bound.
        if dilate_kernel_cv is not None:
            blob_u8 = (blob.astype(np.uint8) * 255)
            blob_u8 = cv2.dilate(blob_u8, dilate_kernel_cv)
            blob = blob_u8 > 0
        _mask_z = blob & cyl_z & (volume[z] < T_GLASS_AL)  # exclude aluminum wall voxels
        if CONTOUR_SHRINK_VOX > 0 and _mask_z.any():
            _shrink_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * CONTOUR_SHRINK_VOX + 1, 2 * CONTOUR_SHRINK_VOX + 1))
            _mask_u8 = cv2.erode((_mask_z.astype(np.uint8) * 255), _shrink_kernel)
            _mask_z = _mask_u8 > 0
        specimen_mask[z] = _mask_z

        if z == _qc_mid_z:
            # Red QC contour = outer boundary of ice+glass specimen
            qc_contours, _ = cv2.findContours(
                blob.astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if qc_contours:
                areas_qc = [cv2.contourArea(c) for c in qc_contours]
                _qc_contours = [qc_contours[int(np.argmax(areas_qc))]]
            _qc_filtered_slice = volume[z].copy()

    if empty_slices:
        print(f"  WARNING: {empty_slices} slices outside cylinder (left empty).")
    if fallback_slices:
        print(f"  {fallback_slices} slices fell back to cylinder mask "
              f"(blob < {int(CONTOUR_FALLBACK_FRAC*100)}% of cylinder area).")
    print(f"  Contour mask built in {time.time()-t_ctr:.1f}s")
else:
    raise ValueError(f"MASK_METHOD must be 'contour' or 'cylinder' (got {MASK_METHOD!r})")

# Keep legacy name so downstream references continue to work
cylinder_mask = specimen_mask
# --- Aluminum-wall partial-volume buffer -----------------------------------
# Voxels immediately adjacent to aluminum have gray values on the ramp
# between Al-bright and glass/ice-dark, which can fall in the glass
# threshold band and get mis-labelled. Dilate the aluminum mask by
# AL_BUFFER_VOX voxels and EXCLUDE that shell from the specimen mask so
# the interior analysis starts inside the partial-volume ramp.
AL_BUFFER_VOX = 3
_al_mask_3d = (volume >= T_GLASS_AL)
if AL_BUFFER_VOX > 0 and _al_mask_3d.any():
    _al_buffer = ndimage.binary_dilation(_al_mask_3d, iterations=AL_BUFFER_VOX)
    _before_vox = int(specimen_mask.sum())
    specimen_mask = specimen_mask & ~_al_buffer
    cylinder_mask = specimen_mask
    _after_vox = int(specimen_mask.sum())
    print(f'  Al-wall buffer ({AL_BUFFER_VOX} vox): removed '
          f'{_before_vox - _after_vox:,} partial-volume voxels '
          f'({100*(_before_vox - _after_vox)/max(_before_vox,1):.2f}% of specimen mask)')

n_inside  = int(np.sum(specimen_mask))
n_outside = int(specimen_mask.size - n_inside)
print(f"  Method: {MASK_METHOD!r}")
print(f"  Inside specimen : {n_inside:>12,} voxels")
print(f"  Outside specimen: {n_outside:>12,} voxels")

# -- 3. Phase thresholding inside the specimen --------------------------------
print("\nStep 3: Thresholding phases inside specimen...")

# Otsu on air+ice voxels inside the specimen to pick T_AIR_ICE automatically.
# Ice/glass and glass/aluminum boundaries stay manual ÔÇö their peaks are well
# separated and Otsu would latch onto the dominant valley (air/ice) instead.
if USE_OTSU_AIR_ICE:
    from skimage.filters import threshold_otsu
    air_ice_samples = volume[specimen_mask & (volume < T_ICE_GLASS)]
    if air_ice_samples.size > 0:
        T_AIR_ICE_AUTO = int(threshold_otsu(air_ice_samples))
        print(f"  Otsu T_AIR_ICE = {T_AIR_ICE_AUTO} (manual was {T_AIR_ICE})")
        T_AIR_ICE = T_AIR_ICE_AUTO
    else:
        print("  WARNING: no air/ice voxels found for Otsu, keeping manual T_AIR_ICE")
else:
    print(f"  Using manual T_AIR_ICE = {T_AIR_ICE}")

seg = np.zeros(volume.shape, dtype=np.uint8)   # default = OUTSIDE_AIR

v = volume  # shorthand

# -- 3a. Topological outside/trapped-air classification (both modes) ----------
# A voxel with gray < T_AIR_ICE is air. An air voxel belongs to OUTSIDE air if
# and only if it is 3D-connected to the image boundary (outside the imaged
# volume is air by definition). All other air voxels are sealed inside the
# specimen -> TRAPPED air. This is robust against mask imperfections: even if
# morphological closing arced across beads at the specimen edge and pulled an
# outside-air pocket into specimen_mask, the pocket stays classified as
# outside because it is still connected through Z / neighboring slices to the
# true outside.
print("\nStep 3a: Classifying air by 3D connectivity to image boundary...")
all_air = (v < T_AIR_ICE)
t_cc = time.time()
labeled_air, n_air_comp = ndimage.label(all_air)
print(f"  {n_air_comp} air components labeled in {time.time()-t_cc:.1f}s")

edge_ids = set()
edge_ids.update(np.unique(labeled_air[0]).tolist())
edge_ids.update(np.unique(labeled_air[-1]).tolist())
edge_ids.update(np.unique(labeled_air[:, 0, :]).tolist())
edge_ids.update(np.unique(labeled_air[:, -1, :]).tolist())
edge_ids.update(np.unique(labeled_air[:, :, 0]).tolist())
edge_ids.update(np.unique(labeled_air[:, :, -1]).tolist())
edge_ids.discard(0)

if edge_ids:
    edge_arr = np.zeros(n_air_comp + 1, dtype=bool)
    for i in edge_ids:
        edge_arr[i] = True
    outside_air_3d = edge_arr[labeled_air]
else:
    outside_air_3d = np.zeros_like(all_air)

trapped_air_3d = all_air & ~outside_air_3d
print(f"  Outside-air voxels : {int(outside_air_3d.sum()):>12,}  "
      f"(components touching boundary: {len(edge_ids)})")
print(f"  Trapped-air voxels : {int(trapped_air_3d.sum()):>12,}  "
      f"(isolated components: {n_air_comp - len(edge_ids)})")

# Free label memory before Step 3 thresholding
del labeled_air

# -- 3b. Phase thresholding ---------------------------------------------------
# Ice/glass inside the specimen mask. Trapped air uses the topological rule
# AND lies inside specimen_mask (so air pockets in the Al gap don't leak in).
# Aluminum is labeled globally by gray value.
seg[specimen_mask & (v >= T_AIR_ICE) & (v < T_ICE_GLASS)]         = LABEL_ICE
seg[specimen_mask & (v >= T_ICE_GLASS) & (v < T_GLASS_AL)]        = LABEL_GLASS
seg[trapped_air_3d & specimen_mask]                               = LABEL_TRAPPED_AIR
# Aluminium label restricted to the cylinder. Flip
# RESTRICT_AL_TO_CYLINDER to False for a whole-volume Al label.
RESTRICT_AL_TO_CYLINDER = True
if RESTRICT_AL_TO_CYLINDER:
    seg[(v >= T_GLASS_AL) & cyl_mask_3d] = LABEL_ALUMINUM
else:
    seg[v >= T_GLASS_AL] = LABEL_ALUMINUM

# -- 3c. Remove salt-and-pepper noise from ice (and optionally trapped air) ----
# Tiny isolated voxel clusters (noise) are removed by connected component
# filtering. Removed ice voxels are reclassified based on their gray value
# neighbors, but in practice they become the surrounding dominant phase.
if MIN_ICE_VOXELS > 0:
    print(f"\nStep 3c: Removing ice clusters smaller than {MIN_ICE_VOXELS} voxels...")
    ice_mask = (seg == LABEL_ICE)
    n_ice_before = int(np.sum(ice_mask))
    ice_labeled, n_ice_comp = ndimage.label(ice_mask)
    try:
        ice_labels, ice_sizes = label_counts(ice_labeled)
    except MemoryError:
        print('  WARNING: MemoryError counting ice components; skipping ice small-object removal.')
        ice_labels = np.empty(0, dtype=np.int64)
        ice_sizes = np.empty(0, dtype=np.int64)

    small_ice_ids = ice_labels[ice_sizes < MIN_ICE_VOXELS]
    n_removed_comp = int(len(small_ice_ids))
    # Build mask of all small-component voxels using lookup array (memory-efficient)
    if len(small_ice_ids) > 0:
        small_ice_lookup = np.zeros(n_ice_comp + 1, dtype=bool)
        small_ice_lookup[small_ice_ids.astype(int)] = True
        remove_mask = small_ice_lookup[ice_labeled]
    else:
        remove_mask = np.zeros_like(ice_labeled, dtype=bool)
    n_removed_vox = int(np.sum(remove_mask))

    # Reclassify removed voxels: set to glass (most likely surrounding phase)
    # since noise ice voxels are typically at the boundary with glass beads
    seg[remove_mask] = LABEL_GLASS

    n_ice_after = int(np.sum(seg == LABEL_ICE))
    print(f"  Ice components: {n_ice_comp} total, {n_removed_comp} removed (< {MIN_ICE_VOXELS} vox)")
    print(f"  Ice voxels: {n_ice_before:,} -> {n_ice_after:,} (removed {n_removed_vox:,})")

if MIN_GLASS_VOXELS > 0:
    print(f"  Removing glass clusters smaller than {MIN_GLASS_VOXELS} voxels...")
    glass_mask = (seg == LABEL_GLASS)
    n_glass_before = int(np.sum(glass_mask))
    glass_labeled, n_glass_comp = ndimage.label(glass_mask)
    try:
        glass_labels, glass_sizes = label_counts(glass_labeled)
    except MemoryError:
        print('  WARNING: MemoryError counting glass components; skipping glass small-object removal.')
        glass_labels = np.empty(0, dtype=np.int64)
        glass_sizes = np.empty(0, dtype=np.int64)

    small_glass_ids = glass_labels[glass_sizes < MIN_GLASS_VOXELS]
    n_removed_glass_comp = int(len(small_glass_ids))
    # Build mask using lookup array (memory-efficient)
    if len(small_glass_ids) > 0:
        small_glass_lookup = np.zeros(n_glass_comp + 1, dtype=bool)
        small_glass_lookup[small_glass_ids.astype(int)] = True
        remove_glass_mask = small_glass_lookup[glass_labeled]
    else:
        remove_glass_mask = np.zeros_like(glass_labeled, dtype=bool)
    n_removed_glass_vox = int(np.sum(remove_glass_mask))

    # Reclassify removed glass voxels as ice (most likely surrounding phase)
    seg[remove_glass_mask] = LABEL_ICE

    n_glass_after = int(np.sum(seg == LABEL_GLASS))
    print(f"  Glass components: {n_glass_comp} total, {n_removed_glass_comp} removed (< {MIN_GLASS_VOXELS} vox)")
    print(f"  Glass voxels: {n_glass_before:,} -> {n_glass_after:,} (removed {n_removed_glass_vox:,})")

if MIN_TRAPPED_AIR_VOXELS > 0:
    print(f"\nStep 3c: Removing trapped air clusters smaller than {MIN_TRAPPED_AIR_VOXELS} voxels...")
    ta_mask = (seg == LABEL_TRAPPED_AIR)
    n_ta_before = int(np.sum(ta_mask))
    ta_labeled, n_ta_comp = ndimage.label(ta_mask)
    try:
        ta_labels, ta_sizes = label_counts(ta_labeled)
    except MemoryError:
        print('  WARNING: MemoryError counting trapped-air components; skipping trapped-air small-object removal.')
        ta_labels = np.empty(0, dtype=np.int64)
        ta_sizes = np.empty(0, dtype=np.int64)

    small_ta_ids = ta_labels[ta_sizes < MIN_TRAPPED_AIR_VOXELS]
    n_removed_ta_comp = int(len(small_ta_ids))
    # Build mask using lookup array (memory-efficient)
    if len(small_ta_ids) > 0:
        small_ta_lookup = np.zeros(n_ta_comp + 1, dtype=bool)
        small_ta_lookup[small_ta_ids.astype(int)] = True
        remove_ta_mask = small_ta_lookup[ta_labeled]
    else:
        remove_ta_mask = np.zeros_like(ta_labeled, dtype=bool)
    n_removed_ta_vox = int(np.sum(remove_ta_mask))
    seg[remove_ta_mask] = LABEL_ICE   # small air pockets inside ice -> reclassify as ice

    n_ta_after = int(np.sum(seg == LABEL_TRAPPED_AIR))
    print(f"  Trapped air components: {n_ta_comp} total, {n_removed_ta_comp} removed")
    print(f"  Trapped air voxels: {n_ta_before:,} -> {n_ta_after:,} (removed {n_removed_ta_vox:,})")

# -- 4. Statistics -------------------------------------------------------------
print("\nStep 4: Final segmentation statistics")
print("=" * 60)
total_voxels    = seg.size
# Interior specimen = ice + glass + trapped air (excludes aluminum holder).
# This is the volume of scientific interest.
n_trapped_air = int(np.sum(seg == LABEL_TRAPPED_AIR))
n_ice         = int(np.sum(seg == LABEL_ICE))
n_glass       = int(np.sum(seg == LABEL_GLASS))
n_aluminum    = int(np.sum(seg == LABEL_ALUMINUM))
interior_voxels = n_trapped_air + n_ice + n_glass

for lbl, name in LABEL_NAMES.items():
    n = int(np.sum(seg == lbl))
    vol_mm3 = n * (VOXEL_SIZE_UM ** 3) / 1e9
    pct_total    = 100.0 * n / total_voxels
    if lbl in (LABEL_TRAPPED_AIR, LABEL_ICE, LABEL_GLASS) and interior_voxels > 0:
        pct_interior = 100.0 * n / interior_voxels
        pct_str = f"{pct_interior:5.2f}% interior"
    else:
        pct_str = "       ÔÇö        "
    print(f"  {name:<25}: {n:>12,} voxels | {vol_mm3:>10.4f} mm3 | "
          f"{pct_total:5.2f}% total | {pct_str}")

print("=" * 60)

water_content_pct = 100.0 * n_ice / interior_voxels if interior_voxels > 0 else 0
pore_content_pct  = 100.0 * (n_ice + n_trapped_air) / interior_voxels if interior_voxels > 0 else 0
print(f"\n  Interior specimen (ice+glass+air): {interior_voxels:>12,} voxels "
      f"({interior_voxels*(VOXEL_SIZE_UM**3)/1e9:.2f} mm3)")
print(f"  Aluminum holder                  : {n_aluminum:>12,} voxels "
      f"({n_aluminum*(VOXEL_SIZE_UM**3)/1e9:.2f} mm3)")
print(f"  Ice fraction of interior         : {water_content_pct:.2f}%")
print(f"  Total pore fraction (ice+air)    : {pore_content_pct:.2f}%")

# Backward-compat alias so downstream scripts that reference `specimen_voxels`
# keep working. Now means interior (ice+glass+air).
specimen_voxels = interior_voxels

# -- 5. Save segmentation as TIFF stack ---------------------------------------
# Values scaled x50 so Dragonfly auto-display shows all phases without LUT adjustment:
#   0=outside air -> 0   1=trapped air -> 50   2=ice -> 100   3=glass -> 150   4=aluminum -> 200
print(f"\nStep 5: Saving segmentation TIFF stack to:\n  {OUT_DIR}")
seg_display = (seg * 50).astype(np.uint8)

# Pre-delete any stale files to fail fast if something has them locked
# (Dragonfly viewing the stack, Explorer preview, etc.)
locked_files = []
for old in os.listdir(OUT_DIR):
    if old.startswith('seg_') and old.endswith('.tif'):
        p = os.path.join(OUT_DIR, old)
        try:
            os.remove(p)
        except PermissionError:
            locked_files.append(old)
if locked_files:
    raise PermissionError(
        f"{len(locked_files)} file(s) in {OUT_DIR} are locked by another "
        f"process (first: {locked_files[0]}). Close Dragonfly's view of the "
        f"segmentation folder (or File Explorer preview pane) and re-run."
    )

import time as _t
for z in range(seg_display.shape[0]):
    fname = os.path.join(OUT_DIR, f'seg_{z:04d}.tif')
    # Retry once after a short pause in case of transient AV/indexing lock
    for attempt in range(2):
        try:
            save_tif(fname, seg_display[z])
            break
        except PermissionError:
            if attempt == 0:
                _t.sleep(0.5)
                continue
            raise PermissionError(
                f"Cannot write {fname}. Close any program holding this file "
                f"(Dragonfly, Explorer preview) and re-run."
            )
    if z % 100 == 0:
        print(f"  Saved slice {z}/{seg.shape[0]-1}")
print("  All slices saved.")
print("  Display values: 0=outside air  50=trapped air  100=ice  150=glass  200=aluminum")

# -- 6. Save overview plot -----------------------------------------------------
print("\nStep 6: Saving overview images...")

# Apply MATLAB-style
_FS = 24   # slightly smaller for 3-panel layout
plt.rcParams.update({
    "font.size": _FS, "axes.titlesize": _FS, "axes.labelsize": _FS + 2,
    "xtick.labelsize": _FS - 2, "ytick.labelsize": _FS - 2,
    "legend.fontsize": _FS - 2, "axes.linewidth": 1.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

CMAP = {
    LABEL_OUTSIDE_AIR: [0.05, 0.05, 0.05],   # near black
    LABEL_TRAPPED_AIR: [1.0,  0.2,  0.2 ],   # red
    LABEL_ICE:         [0.4,  0.7,  1.0 ],   # light blue
    LABEL_GLASS:       [0.9,  0.85, 0.6 ],   # tan/beige
    LABEL_ALUMINUM:    [0.8,  0.8,  0.8 ],   # silver
}

def seg_to_rgb(slice_2d):
    rgb = np.zeros((*slice_2d.shape, 3), dtype=np.float32)
    for lbl, color in CMAP.items():
        rgb[slice_2d == lbl] = color
    return rgb

mid_z = seg.shape[0] // 2
mid_y = seg.shape[1] // 2
mid_x = seg.shape[2] // 2

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

axes[0].imshow(seg_to_rgb(seg[mid_z, :, :]), origin='upper', aspect='equal')
axes[0].set_title(f'XY slice (Z={mid_z})', pad=10)
axes[0].set_xlabel('X (voxels)')
axes[0].set_ylabel('Y (voxels)')

axes[1].imshow(seg_to_rgb(seg[:, mid_y, :]), origin='upper', aspect='equal')
axes[1].set_title(f'XZ slice (Y={mid_y})', pad=10)
axes[1].set_xlabel('X (voxels)')
axes[1].set_ylabel('Z (voxels)')

axes[2].imshow(seg_to_rgb(seg[:, :, mid_x]), origin='upper', aspect='equal')
axes[2].set_title(f'YZ slice (X={mid_x})', pad=10)
axes[2].set_xlabel('Y (voxels)')
axes[2].set_ylabel('Z (voxels)')

for ax in axes:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.8)
    ax.grid(False)   # grid off for images

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=CMAP[LABEL_OUTSIDE_AIR], edgecolor='gray', label='Outside air'),
    Patch(facecolor=CMAP[LABEL_TRAPPED_AIR], edgecolor='gray', label='Trapped air'),
    Patch(facecolor=CMAP[LABEL_ICE],         edgecolor='gray', label='Ice'),
    Patch(facecolor=CMAP[LABEL_GLASS],       edgecolor='gray', label='Glass beads'),
    Patch(facecolor=CMAP[LABEL_ALUMINUM],    edgecolor='gray', label='Aluminum'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5,
           fontsize=_FS - 2, frameon=True, framealpha=0.85, edgecolor='gray',
           bbox_to_anchor=(0.5, -0.14))

fig.suptitle('Segmentation Overview', fontsize=_FS + 4, fontweight='bold', y=1.02)
plt.tight_layout()
results_dir = os.path.join(os.path.dirname(OUT_DIR), 'results')
os.makedirs(results_dir, exist_ok=True)
overview_path = os.path.join(results_dir, 'segmentation_overview.png')
import sys as _sys
_sys.path.insert(0, r'E:\RPTU-images\CT_images\Glass\<SPECIMEN>\\scan{N}_dragonfly')
from plot_style import save_fig as _save_fig
_save_fig(fig, overview_path)
plt.rcdefaults()

# -- 6b. Contour-detection QC figure (paper Fig 3.15 style) -------------------
# Three panels on the mid-Z slice:
#   (a) Raw data
#   (b) Median-filtered + red contour overlay (detected specimen boundary)
#   (c) Phase thresholding result (same slice segmented)
if MASK_METHOD == 'contour' and _qc_contours is not None:
    print("\nStep 6b: Saving contour-detection QC figure (Fig 3.15 style)...")

    _FS = 22
    plt.rcParams.update({
        "font.size": _FS, "axes.titlesize": _FS, "axes.labelsize": _FS + 2,
        "xtick.labelsize": _FS - 2, "ytick.labelsize": _FS - 2,
        "axes.linewidth": 1.8,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    # (a) Raw XY slice
    axes[0].imshow(_qc_raw_slice, cmap='gray', origin='upper', aspect='equal')
    axes[0].set_title(f'(a) Raw XY slice (Z={_qc_mid_z})', pad=10)
    axes[0].set_xlabel('X (voxels)')
    axes[0].set_ylabel('Y (voxels)')

    # (b) Median-filtered + red contour overlay
    axes[1].imshow(_qc_filtered_slice, cmap='gray', origin='upper', aspect='equal')
    for c in _qc_contours:
        pts = c.reshape(-1, 2)   # OpenCV returns (N, 1, 2) as (x, y)
        # Close the polygon
        pts_closed = np.vstack([pts, pts[0:1]])
        axes[1].plot(pts_closed[:, 0], pts_closed[:, 1],
                     color='red', linewidth=2.0)
    axes[1].set_title('(b) Median filter + contour detection', pad=10)
    axes[1].set_xlabel('X (voxels)')
    axes[1].set_ylabel('Y (voxels)')

    # (c) Phase segmentation on same slice
    axes[2].imshow(seg_to_rgb(seg[_qc_mid_z, :, :]), origin='upper', aspect='equal')
    axes[2].set_title('(c) Phase thresholding', pad=10)
    axes[2].set_xlabel('X (voxels)')
    axes[2].set_ylabel('Y (voxels)')

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.8)
        ax.grid(False)

    # Phase legend (panel c)
    legend_elements_qc = [
        Patch(facecolor=CMAP[LABEL_OUTSIDE_AIR], edgecolor='gray', label='Outside air'),
        Patch(facecolor=CMAP[LABEL_TRAPPED_AIR], edgecolor='gray', label='Trapped air'),
        Patch(facecolor=CMAP[LABEL_ICE],         edgecolor='gray', label='Ice'),
        Patch(facecolor=CMAP[LABEL_GLASS],       edgecolor='gray', label='Glass beads'),
        Patch(facecolor=CMAP[LABEL_ALUMINUM],    edgecolor='gray', label='Aluminum'),
    ]
    fig.legend(handles=legend_elements_qc, loc='lower center', ncol=5,
               fontsize=_FS - 2, frameon=True, framealpha=0.85, edgecolor='gray',
               bbox_to_anchor=(0.5, -0.14))

    fig.suptitle('Contour-Detection Workflow',
                 fontsize=_FS + 4, fontweight='bold', y=1.02)
    plt.tight_layout()
    contour_qc_path = os.path.join(results_dir, 'contour_detection_workflow.png')
    _save_fig(fig, contour_qc_path)
    plt.rcdefaults()
elif MASK_METHOD != 'contour':
    print("\nStep 6b: Skipped (MASK_METHOD != 'contour').")

print("\n" + "=" * 60)
print("DONE.")
print(f"Segmentation labels: 0=outside air  1=trapped air  2=ice  3=glass  4=aluminum")
print(f"Output folder: {OUT_DIR}")

# -- 7. Create ROIs in Dragonfly aligned to the existing CT channel -----------
# WorkingContext methods all require a plugin instance -- unusable from scripts.
# Instead use Channel.getAllObjectsOfClass() to find channels directly.
try:
    from ORSModel import createChannelFromNumpyArray, ROI, orsColor, Channel

    print("\nStep 7: Creating ROIs in Dragonfly...")

    # -- A: find the CT channel using Channel class methods -------------------
    ct_channel = None

    # Get the internal class name string that Dragonfly uses
    cls_name = None
    for attr in ['getClassNameStatic', 'getClassName', 'getClassDenomination']:
        if hasattr(Channel, attr):
            try:
                val = getattr(Channel, attr)
                cls_name = val() if callable(val) else val
                print(f"  Channel.{attr}() = {cls_name!r}")
                break
            except Exception as e:
                print(f"  Channel.{attr}() failed: {e}")

    # Method 1: getAllObjectsOfClass(cls_name) then match by title
    if cls_name:
        try:
            all_channels = Channel.getAllObjectsOfClass(cls_name)
            ch_list = list(all_channels) if all_channels is not None else []
            print(f"  getAllObjectsOfClass({cls_name!r}) -> {len(ch_list)} channels")
            for ch in ch_list:
                try:
                    # Try multiple name/title methods
                    title = ''
                    for tmethod in ['getTitle', 'getPrivateTitle', 'getName', 'getLabel']:
                        if hasattr(ch, tmethod):
                            try:
                                title = getattr(ch, tmethod)()
                                if title and 'orsObj' not in str(title):
                                    break
                            except Exception:
                                pass
                    if not title or 'orsObj' in str(title):
                        title = str(ch)
                    print(f"    Channel: {title}")
                    if CT_CHANNEL_NAME in str(title):
                        ct_channel = ch
                        print(f"  --> Using this as CT channel")
                except Exception:
                    pass
        except Exception as e1:
            print(f"  getAllObjectsOfClass({cls_name!r}) failed: {e1}")

    if ct_channel is None:
        raise RuntimeError(
            "CT channel not found -- stopping to avoid wrong ROIs.\n"
            "  Make sure the CT dataset is loaded in Dragonfly."
        )

    # -- B: create temporary segmentation channel --------------------------------
    seg_ch = createChannelFromNumpyArray(seg_display.astype(np.uint8))
    seg_ch.setTitle('_seg_temp_delete_me')
    # IMPORTANT: createChannelFromNumpyArray gives the new channel a default
    # origin (0, 0, 0) and unit spacing. If we publish as-is and then call
    # makeROIForChannel(ct_channel, 0, 0, 0) the ROIs are placed at world
    # (0, 0, 0) -- far from the specimen, with peripheral beads falling
    # outside the ROI. Copy the CT channel's origin + spacing onto seg_ch
    # BEFORE publishing so the two coordinate systems coincide.
    try:
        seg_ch.setXSpacing(ct_channel.getXSpacing())
        seg_ch.setYSpacing(ct_channel.getYSpacing())
        seg_ch.setZSpacing(ct_channel.getZSpacing())
        seg_ch.setOrigin(ct_channel.getOrigin())
        print(f"  Aligned seg channel to CT origin "
              f"(um): ({ct_channel.getOrigin().getX()*1e6:.3f}, "
              f"{ct_channel.getOrigin().getY()*1e6:.3f}, "
              f"{ct_channel.getOrigin().getZ()*1e6:.3f})")
    except Exception as e:
        print(f"  Warning: could not align seg channel to CT channel: {e}")
        try:
            for sm, gm in [('setXInMicrometers', 'getXInMicrometers'),
                           ('setYInMicrometers', 'getYInMicrometers'),
                           ('setZInMicrometers', 'getZInMicrometers')]:
                if hasattr(seg_ch, sm) and hasattr(ct_channel, gm):
                    getattr(seg_ch, sm)(getattr(ct_channel, gm)())
        except Exception as e2:
            print(f"  Alignment fallback also failed: {e2} -- ROIs will be off.")
    seg_ch.publish()
    print("  Temporary channel published.")

    # Raw integer thresholds (confirmed working in earlier run)
    phase_defs = [
        ("ROI_TrappedAir",  49,  51, orsColor(1.0, 0.2, 0.2, 1.0)),
        ("ROI_Ice",         99, 101, orsColor(0.4, 0.7, 1.0, 1.0)),
        ("ROI_GlassBeads", 149, 151, orsColor(0.9, 0.85, 0.6, 1.0)),
        ("ROI_Aluminum",   199, 201, orsColor(0.8, 0.8, 0.8, 1.0)),
    ]

    # -- C: extract ROIs from seg_ch, then remap to CT channel coordinates ----
    # makeROIForChannel(pChannel, x, y, z) creates a new ROI in CT coords.
    # Offsets (0,0,0) because seg and CT cover the same cropped volume.
    for title, vmin, vmax, color in phase_defs:
        # Step 1: extract ROI in seg_ch coordinates
        roi_tmp = ROI()
        roi_tmp.copyShapeFromStructuredGrid(seg_ch)
        seg_ch.getAsROIWithinRange(vmin, vmax, None, roi_tmp)

        # Step 2: remap ROI to CT channel coordinates
        try:
            roi_ct = roi_tmp.makeROIForChannel(ct_channel, 0, 0, 0)
            roi_ct.setTitle(title)
            roi_ct.setInitialColor(color)
            roi_ct.publish()
            print(f"  Published ROI: {title} (via makeROIForChannel)")
        except Exception as e1:
            print(f"  makeROIForChannel failed: {e1}")
            # Fallback: try adaptToChannel(pChannel, x, y, z, pTSourceOffset, pTRange)
            try:
                roi_tmp.adaptToChannel(ct_channel, 0, 0, 0, 0, 1)
                roi_tmp.setTitle(title)
                roi_tmp.setInitialColor(color)
                roi_tmp.publish()
                print(f"  Published ROI: {title} (via adaptToChannel)")
            except Exception as e2:
                print(f"  adaptToChannel failed: {e2}")
                # Last resort: publish as-is
                roi_tmp.setTitle(title)
                roi_tmp.setInitialColor(color)
                roi_tmp.publish()
                print(f"  Published ROI: {title} (not adapted, may be misaligned)")

        # Cleanup temp ROI
        try:
            roi_tmp.deleteObject()
        except Exception:
            pass

    # -- D: delete temporary channel ------------------------------------------
    try:
        seg_ch.deleteObject()
        print("  Temporary channel deleted.")
    except Exception:
        try:
            seg_ch.remove()
            print("  Temporary channel removed.")
        except Exception as e_del:
            print(f"  WARNING: could not delete temp channel: {e_del}")

    print("  Done -- 4 ROIs created.")

except ImportError as e:
    print(f"\nStep 7 ERROR: import failed -- {e}")
except RuntimeError as e:
    print(f"\nStep 7 STOPPED: {e}")
except Exception as e:
    import traceback
    print(f"\nStep 7 ERROR: {e}")
    traceback.print_exc()

# -- 8. Export volumes for SPAM (scan 2) --------------------------------------
# Write ct_scan01.tif and specimen_mask_scan01.tif into Glass_spam\data\ so the
# WSL-side SPAM pipeline can consume them without re-reading the raw stack.
SPAM_DATA_DIR = r'E:\RPTU-images\CT_images\Glass\<SPECIMEN>\<SPAM>\data'
os.makedirs(SPAM_DATA_DIR, exist_ok=True)
print(f"\nStep 8: Exporting TIFFs for SPAM to {SPAM_DATA_DIR}")
try:
    ct_tif   = os.path.join(SPAM_DATA_DIR, 'ct_scan01.tif')
    mask_tif = os.path.join(SPAM_DATA_DIR, 'specimen_mask_scan01.tif')
    for p in (ct_tif, mask_tif):
        if os.path.exists(p):
            try:
                os.remove(p)
            except PermissionError:
                raise PermissionError(f"Cannot overwrite {p}; close any viewer.")
    save_tif(ct_tif,   volume.astype(np.uint16))
    save_tif(mask_tif, (specimen_mask.astype(np.uint8) * 255))
    print(f"  Saved: {ct_tif}")
    print(f"  Saved: {mask_tif}")
except Exception as e:
    print(f"Step 8 ERROR: {e}")
