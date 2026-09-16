"""
Shared I/O and volume helpers for the Sand pipeline (Stages 0-3).

  - load_volume()     : read the TIFF slice stack (optionally a Z slab + XY crop)
  - save_tiff_stack() : write a 3D array as a numbered TIFF stack
  - save_label_image(): write a SPAM-compatible integer label volume
  - specimen_mask()   : automatic sand+ice plug mask (excludes outside air)
  - multi_otsu_phases(): suggest air/ice/sand thresholds inside the specimen

Designed so Stage 1 can later be swapped for a 3-class nnU-Net without
touching the rest of the pipeline: every stage consumes/produces plain
TIFF stacks + a label volume.
"""
import os
import glob
import time
import numpy as np
import tifffile
from scipy import ndimage


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def list_slices(tiff_dir, file_prefix):
    fs = sorted(glob.glob(os.path.join(tiff_dir, f"{file_prefix}*.tif")))
    if not fs:
        raise FileNotFoundError(f"No TIFFs matching {file_prefix}*.tif in {tiff_dir}")
    return fs


def load_volume(cfg, subsample=None, verbose=True):
    """Load the stack into a (Z, Y, X) uint16 array.

    subsample : int or None. If set, load only every Nth slice (Stage 0 audit
                uses this to histogram cheaply). Ignores z_range/xy_crop when
                subsampling for a global overview.
    Otherwise honours cfg.z_range (inclusive) and cfg.xy_crop.
    """
    fs = list_slices(cfg.tiff_dir, cfg.file_prefix)
    n = len(fs)

    if subsample:
        idx = list(range(0, n, subsample))
        if verbose:
            print(f"  [audit] subsampling {len(idx)}/{n} slices (every {subsample})")
        vol = np.stack([tifffile.imread(fs[i]) for i in idx])
        return vol, idx

    z0, z1 = (0, n - 1) if cfg.z_range is None else cfg.z_range
    z1 = min(z1, n - 1)
    sl = []
    t0 = time.time()
    for i in range(z0, z1 + 1):
        im = tifffile.imread(fs[i])
        if cfg.xy_crop is not None:
            x0, x1, y0, y1 = cfg.xy_crop
            im = im[y0:y1 + 1, x0:x1 + 1]
        sl.append(im)
        if verbose and (i - z0) % 50 == 0:
            print(f"  loaded slice {i - z0 + 1}/{z1 - z0 + 1}")
    vol = np.stack(sl, axis=0)
    if verbose:
        print(f"  volume {vol.shape} {vol.dtype} loaded in {time.time()-t0:.1f}s "
              f"(Z {z0}-{z1})")
    return vol, (z0, z1)


# ----------------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------------
def save_tiff_stack(out_dir, basename, arr):
    """Write arr (Z,Y,X) as numbered TIFFs basename_####.tif. Clears stale files."""
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, f"{basename}_*.tif")):
        try:
            os.remove(old)
        except PermissionError:
            raise PermissionError(f"{old} is locked (close Dragonfly/Explorer preview).")
    for z in range(arr.shape[0]):
        tifffile.imwrite(os.path.join(out_dir, f"{basename}_{z:04d}.tif"), arr[z])
    print(f"  wrote {arr.shape[0]} slices -> {out_dir}\\{basename}_*.tif")


def save_label_image(path, labels):
    """Save an integer grain-label volume as a single multipage TIFF.

    SPAM's discrete DVC / label toolkit reads integer voxel-patch labels, one
    unique id per grain, exactly this format. uint16 if it fits, else uint32.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mx = int(labels.max())
    dt = np.uint16 if mx < 65535 else np.uint32
    tifffile.imwrite(path, labels.astype(dt))
    print(f"  wrote label volume ({mx} grains, {dt.__name__}) -> {path}")


# ----------------------------------------------------------------------------
# Specimen mask + phase thresholds
# ----------------------------------------------------------------------------
def specimen_mask(volume, t_material, close_radius=6, min_area_frac=0.02,
                  keep_largest_only=True):
    """Automatic per-slice sand+ice plug mask (True inside specimen).

    Excludes the outside-air zeros AND the bright sample-holder wall + the air
    gap between wall and plug, so the phase thresholds are picked on interior
    (sand+ice) voxels only.

    Per slice: material = gray > t_material -> close -> KEEP component(s) ->
    fill holes. Component selection happens BEFORE hole-filling: otherwise the
    closed holder-wall ring is filled first and the mask swells out to the wall
    (capturing the dark air gap and inflating the apparent air fraction).
    keep_largest_only keeps just the plug (the wall ring is a separate, smaller
    component); set False to keep every component >= min_area_frac of the largest.
    """
    nz = volume.shape[0]
    mask = np.zeros(volume.shape, dtype=bool)
    if close_radius > 0:
        r = close_radius
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        se = (xx**2 + yy**2) <= r**2
    for z in range(nz):
        mat = volume[z] > t_material
        if close_radius > 0:
            mat = ndimage.binary_closing(mat, structure=se)
        lbl, n = ndimage.label(mat)          # select BEFORE filling
        if n == 0:
            continue
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        if keep_largest_only:
            keep = [int(sizes.argmax())]
        else:
            keep = np.where(sizes >= min_area_frac * sizes.max())[0].tolist()
        plug = np.isin(lbl, keep)
        mask[z] = ndimage.binary_fill_holes(plug)   # fill voids INSIDE the plug only
    return mask


def multi_otsu_phases(volume, mask, nbins=256):
    """3-class multi-Otsu INSIDE the mask -> (t_air_ice, t_ice_sand).

    Returns the two gray-value thresholds separating air | ice | sand.
    """
    from skimage.filters import threshold_multiotsu
    vals = volume[mask]
    # clip the bright mineral/holder tail so Otsu isn't dragged by rare outliers
    hi = np.percentile(vals, 99.9)
    vals = vals[vals <= hi]
    th = threshold_multiotsu(vals, classes=3, nbins=nbins)
    return int(th[0]), int(th[1])


def per_slice_thresholds(volume, mask, smooth=21):
    """Per-slice air|ice and ice|sand thresholds (counters the z-drift).

    A single global threshold fails across the full stack because the
    ice/sand gray boundary drifts by thousands of counts with height. This
    computes 3-class multi-Otsu inside the mask for every slice, then median-
    smooths each threshold along z (window `smooth`) for stability on sparse
    slices. Returns two (nz,) float arrays (NaN-free, gap-filled).
    """
    from skimage.filters import threshold_multiotsu
    nz = volume.shape[0]
    t_ai = np.full(nz, np.nan)
    t_is = np.full(nz, np.nan)
    for z in range(nz):
        m = mask[z]
        if m.sum() < 500:
            continue
        vals = volume[z][m]
        hi = np.percentile(vals, 99.9)
        vals = vals[vals <= hi]
        try:
            th = threshold_multiotsu(vals, classes=3)
            t_ai[z], t_is[z] = float(th[0]), float(th[1])
        except Exception:
            pass
    # fill gaps by interpolation, then median-smooth along z
    zs = np.arange(nz)
    for arr in (t_ai, t_is):
        good = ~np.isnan(arr)
        if good.any():
            arr[~good] = np.interp(zs[~good], zs[good], arr[good])
    if smooth > 1:
        t_ai = ndimage.median_filter(t_ai, size=smooth)
        t_is = ndimage.median_filter(t_is, size=smooth)
    return t_ai, t_is
