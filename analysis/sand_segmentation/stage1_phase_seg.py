"""
Stage 1 - Phase segmentation: air / ice / sand.

Multi-threshold FIRST (per ChatGPT's guidance: do not assume clean peaks).
The code is structured so the threshold core can be swapped for a 3-class
nnU-Net (air/ice/sand) later if ice/sand or air/ice boundaries smear -- every
stage consumes/produces plain TIFF stacks, so only this function changes.

Air is split topologically into OUTSIDE air (connected to the volume boundary)
and interior VOID air (sealed pockets), matching the glass/ice pipeline.

Outputs (results/<sample>/stage1_phase/):
  phase_labels_####.tif   uint8 label stack (0 out-air,1 void,2 ice,3 sand) x50 for display
  sand_mask_####.tif      uint8 0/255 sand-only mask (input to Stage 2/3)
  phase_overview.png      3 orthogonal QC slices
Returns (volume, seg, sand_mask) in memory for the driver.
"""
import os
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

from config import (CFG, LABEL_OUTSIDE_AIR, LABEL_VOID_AIR, LABEL_ICE,
                    LABEL_SAND, LABEL_NAMES, PHASE_RGB)
import common
import plot_style as ps


def segment_phases(volume, cfg=CFG, verbose=True):
    """Return (seg uint8, sand_mask bool, info dict). THIS is the swap point
    for a future 3-class nnU-Net: keep the signature, replace the body."""
    t_material = int(threshold_otsu(volume))
    mask = common.specimen_mask(volume, t_material,
                                close_radius=cfg.specimen_close_radius,
                                min_area_frac=cfg.specimen_min_area_frac)

    v = volume
    seg = np.zeros(volume.shape, dtype=np.uint8)

    if getattr(cfg, "adaptive_thresholds", False):
        # per-slice z-smoothed thresholds (counter the drift)
        t_ai_z, t_is_z = common.per_slice_thresholds(volume, mask)
        t_ai = t_ai_z[:, None, None]            # broadcast over Y,X per slice
        t_is = t_is_z[:, None, None]
        if verbose:
            print(f"  ADAPTIVE thresholds: air|ice {t_ai_z.min():.0f}-{t_ai_z.max():.0f}  "
                  f"ice|sand {t_is_z.min():.0f}-{t_is_z.max():.0f} (per-slice)")
        info_t = dict(t_air_ice=f"{t_ai_z.min():.0f}-{t_ai_z.max():.0f}",
                      t_ice_sand=f"{t_is_z.min():.0f}-{t_is_z.max():.0f}")
    else:
        t_ai = cfg.t_air_ice
        t_is = cfg.t_ice_sand
        if t_ai is None or t_is is None:
            t_ai, t_is = common.multi_otsu_phases(volume, mask)
        if verbose:
            print(f"  global thresholds: t_air_ice={t_ai}  t_ice_sand={t_is}")
        info_t = dict(t_air_ice=t_ai, t_ice_sand=t_is)

    # Classify STRICTLY by specimen-mask membership. Inside the plug every
    # voxel is sand / ice / air-void; only voxels OUTSIDE the plug are
    # outside-air (label 0). The previous version split air topologically
    # (any air component touching the volume boundary = "outside"), which
    # leaked boundary-connected interior pores into label 0 -> "outside"
    # appeared INSIDE the sample. The specimen mask already defines inside vs
    # outside, so interior darkness is unambiguously interior void air.
    seg[mask & (v >= t_is)] = LABEL_SAND
    seg[mask & (v >= t_ai) & (v < t_is)] = LABEL_ICE
    seg[mask & (v < t_ai)] = LABEL_VOID_AIR
    # everything outside the specimen mask stays 0 = outside air

    # audit: no voxel inside the specimen mask may remain "outside" (label 0)
    inside_outside = int((mask & (seg == LABEL_OUTSIDE_AIR)).sum())
    if verbose and inside_outside:
        print(f"  WARNING: {inside_outside} voxels INSIDE the specimen mask are "
              f"labelled OUTSIDE (should be 0).")

    sand_mask = seg == LABEL_SAND
    info = dict(t_material=t_material, mask_voxels=int(mask.sum()),
                inside_outside_voxels=inside_outside, **info_t)
    return seg, sand_mask, mask, info


def _seg_to_rgb(s):
    rgb = np.zeros((*s.shape, 3), np.float32)
    for lab, c in PHASE_RGB.items():
        rgb[s == lab] = c
    return rgb


def main():
    out = CFG.d("stage1_phase")
    print("=" * 64)
    print(f"Stage 1 - phase segmentation : {CFG.sample_name}")
    print("=" * 64)

    vol, zr = common.load_volume(CFG)
    if CFG.median_size > 0:
        print(f"  3D median filter (size={CFG.median_size}) ...")
        vol = ndimage.median_filter(vol, size=CFG.median_size)

    seg, sand_mask, mask, info = segment_phases(vol)

    # stats
    tot = seg.size
    print("\n  phase           voxels         vol%")
    for lab, name in LABEL_NAMES.items():
        nvox = int((seg == lab).sum())
        print(f"  {name:<18} {nvox:>12,}  {100*nvox/tot:6.2f}")

    # audit: phase composition INSIDE the specimen mask (no 'outside' allowed)
    nin = int(mask.sum())
    print(f"\n  inside specimen mask ({nin:,} vox):")
    for lab, name in LABEL_NAMES.items():
        nvox = int((mask & (seg == lab)).sum())
        print(f"  {name:<18} {nvox:>12,}  {100*nvox/max(nin,1):6.2f}%")
    print(f"  inside-sample OUTSIDE voxels = {info['inside_outside_voxels']} "
          f"(must be 0)")

    # save label stack (x50 like glass pipeline so Dragonfly shows phases) + sand
    # mask + specimen mask (for QC / Dragonfly cylinder overlay)
    common.save_tiff_stack(out, "phase_labels", (seg * 50).astype(np.uint8))
    common.save_tiff_stack(out, "sand_mask", (sand_mask.astype(np.uint8) * 255))
    common.save_tiff_stack(out, "specimen_mask", (mask.astype(np.uint8) * 255))

    # QC overview: 3 orthogonal slices (one figure)
    ps.apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    zc, yc, xc = [s // 2 for s in seg.shape]
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    axes[0].imshow(_seg_to_rgb(seg[zc])); axes[0].set_title(f"XY  Z={zc}")
    axes[1].imshow(_seg_to_rgb(seg[:, yc])); axes[1].set_title(f"XZ  Y={yc}")
    axes[2].imshow(_seg_to_rgb(seg[:, :, xc])); axes[2].set_title(f"YZ  X={xc}")
    for ax in axes:
        ax.set_xlabel("voxels"); ax.set_ylabel("voxels")
        ps.style_axes(ax); ax.grid(False)
    handles = [Patch(facecolor=PHASE_RGB[l], edgecolor="gray", label=LABEL_NAMES[l])
               for l in LABEL_NAMES]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True,
               edgecolor="gray", bbox_to_anchor=(0.5, -0.05))
    ps.save_fig(fig, os.path.join(out, "phase_overview.png"))

    print(f"\nStage 1 done. thresholds used: {info}")
    return vol, seg, sand_mask, info


if __name__ == "__main__":
    main()
