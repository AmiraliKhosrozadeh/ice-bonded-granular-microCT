"""
Stage 2 - Sand-only mask extraction & careful cleanup.

Takes the sand phase from Stage 1 and produces a clean binary mask for
instance labeling. Cleanup is deliberately GENTLE: remove only tiny specks
and fill 1-voxel holes. Opening is OFF by default (cfg.sand_open_radius=0)
because morphological opening erodes thin fines and breaks real grain-grain
contacts -- exactly the information Stage 3 / ParticleSeg3D must preserve.

Outputs (results/<sample>/stage2_sand/):
  sand_clean_####.tif   uint8 0/255 cleaned sand mask
Returns the cleaned boolean mask.
"""
import os
import numpy as np
from scipy import ndimage

from config import CFG
import common


def clean_sand_mask(sand_mask, cfg=CFG, verbose=True):
    m = sand_mask.copy()
    # remove specks below the noise floor
    if cfg.sand_min_voxels > 0:
        lbl, n = ndimage.label(m)
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        small = np.where((sizes > 0) & (sizes < cfg.sand_min_voxels))[0]
        removed = int(np.isin(lbl, small).sum())
        m[np.isin(lbl, small)] = False
        if verbose:
            print(f"  removed {len(small)} specks < {cfg.sand_min_voxels} vox "
                  f"({removed:,} voxels)")
    # optional opening (default off -> preserve fines/contacts)
    if cfg.sand_open_radius > 0:
        r = cfg.sand_open_radius
        zz, yy, xx = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
        se = (xx**2 + yy**2 + zz**2) <= r**2
        m = ndimage.binary_opening(m, structure=se)
        if verbose:
            print(f"  morphological opening r={r} (WARNING: may thin fines)")
    # fill 1-voxel interior holes (scanner noise inside grains)
    m = ndimage.binary_fill_holes(m)
    return m


def main():
    import stage1_phase_seg as s1
    _, _, sand_mask, _ = s1.main()
    out = CFG.d("stage2_sand")
    print("=" * 64)
    print("Stage 2 - sand-mask cleanup")
    print("=" * 64)
    clean = clean_sand_mask(sand_mask)
    print(f"  sand voxels: {int(sand_mask.sum()):,} -> {int(clean.sum()):,}")
    common.save_tiff_stack(out, "sand_clean", (clean.astype(np.uint8) * 255))
    print("Stage 2 done.")
    return clean


if __name__ == "__main__":
    main()
