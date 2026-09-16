"""
Stage 0b - Full-stack z-profile QC (memory-light, slice-by-slice).

Addresses ChatGPT's full-run requests #1 and #2, and finds the global plug
bounding box so the heavy Stage 1-3 baseline can crop to the specimen.

Per slice (whole stack, never holds the full volume in RAM):
  * specimen plug mask (largest material component, filled)
  * interior 3-class multi-Otsu  -> per-slice (t_air_ice, t_ice_sand)   [DRIFT]
  * phase volume fractions using the FIXED config thresholds            [FRACTIONS]
  * plug bounding box

Outputs (results/<sample>/stage0b_zprofile/):
  threshold_drift_vs_z.png    air|ice and ice|sand auto-thresholds per slice
  phase_fraction_vs_z.png     air / ice / sand interior fractions per slice
  zprofile_report.txt         drift stats + global plug bbox (paste as xy_crop)
  zprofile.npz                raw arrays for downstream use

Run:  python stage0b_zprofile.py
"""
import os
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu, threshold_multiotsu

from config import CFG
import common
import plot_style as ps


def main():
    out = CFG.d("stage0b_zprofile")
    print("=" * 64)
    print(f"Stage 0b - full-stack z-profile : {CFG.sample_name}")
    print("=" * 64)
    fs = common.list_slices(CFG.tiff_dir, CFG.file_prefix)
    nz = len(fs)
    vox = CFG.voxel_size_um
    t_ai_fix, t_is_fix = CFG.t_air_ice, CFG.t_ice_sand

    z = np.arange(nz)
    t_ai_z = np.full(nz, np.nan)
    t_is_z = np.full(nz, np.nan)
    f_air = np.full(nz, np.nan); f_ice = np.full(nz, np.nan); f_sand = np.full(nz, np.nan)
    bbox = []  # (y0,y1,x0,x1) per slice with a plug

    import tifffile
    for i in range(nz):
        im = tifffile.imread(fs[i]).astype(np.float32)
        # plug mask: material > otsu, largest CC, fill (same logic as common, 2D)
        try:
            tmat = threshold_otsu(im)
        except Exception:
            continue
        mat = im > tmat
        mat = ndimage.binary_closing(mat, iterations=CFG.specimen_close_radius)
        lbl, n = ndimage.label(mat)
        if n == 0:
            continue
        sizes = np.bincount(lbl.ravel()); sizes[0] = 0
        plug = ndimage.binary_fill_holes(lbl == sizes.argmax())
        npix = int(plug.sum())
        if npix < 500:
            continue
        ys, xs = np.where(plug)
        bbox.append((ys.min(), ys.max(), xs.min(), xs.max()))
        interior = im[plug]
        # per-slice drift thresholds
        try:
            hi = np.percentile(interior, 99.9)
            th = threshold_multiotsu(interior[interior <= hi], classes=3)
            t_ai_z[i], t_is_z[i] = float(th[0]), float(th[1])
        except Exception:
            pass
        # fixed-threshold fractions
        f_air[i]  = np.mean(interior < t_ai_fix)
        f_ice[i]  = np.mean((interior >= t_ai_fix) & (interior < t_is_fix))
        f_sand[i] = np.mean(interior >= t_is_fix)
        if i % 100 == 0:
            print(f"  slice {i}/{nz}")

    bbox = np.array(bbox)
    gy0, gx0 = bbox[:, 0].min(), bbox[:, 2].min()
    gy1, gx1 = bbox[:, 1].max(), bbox[:, 3].max()
    pad = 10
    gy0 = max(0, gy0 - pad); gx0 = max(0, gx0 - pad)

    valid = ~np.isnan(t_is_z)
    drift = np.nanmax(t_is_z) - np.nanmin(t_is_z)

    lines = []
    P = lines.append
    P(f"Sample           : {CFG.sample_name}")
    P(f"Slices with plug : {int(valid.sum())} / {nz}")
    P(f"Fixed thresholds : t_air_ice={t_ai_fix}  t_ice_sand={t_is_fix}")
    P("")
    P("Ice|sand auto-threshold drift across z:")
    P(f"  min {np.nanmin(t_is_z):.0f} | median {np.nanmedian(t_is_z):.0f} | "
      f"max {np.nanmax(t_is_z):.0f} | range {drift:.0f}")
    P(f"  -> {'LARGE drift: a single global threshold will mis-split ice/sand by z; favor nnU-Net' if drift > 1500 else 'moderate drift'}")
    P("")
    P("Mean interior phase fractions (fixed thresholds):")
    P(f"  air {np.nanmean(f_air)*100:.1f}%  ice {np.nanmean(f_ice)*100:.1f}%  "
      f"sand {np.nanmean(f_sand)*100:.1f}%")
    P("")
    P("Global plug bounding box (paste into config.xy_crop as (x0,x1,y0,y1)):")
    P(f"  xy_crop = ({gx0}, {gx1+pad}, {gy0}, {gy1+pad})")
    P(f"  -> crops the {nz}x... frame to {gx1+pad-gx0} x {gy1+pad-gy0} "
      f"(~{100*(gx1+pad-gx0)*(gy1+pad-gy0)/(1179*1179):.0f}% of full frame area)")
    report = "\n".join(lines)
    with open(os.path.join(out, "zprofile_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report + "\n")

    np.savez(os.path.join(out, "zprofile.npz"),
             z=z, t_ai_z=t_ai_z, t_is_z=t_is_z,
             f_air=f_air, f_ice=f_ice, f_sand=f_sand)

    # --- figures (one plot per file) -------------------------------------
    ps.apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.plot(z, t_ai_z, label="air | ice")
    ax.plot(z, t_is_z, label="ice | sand")
    ax.axhline(t_is_fix, color="black", linestyle="--", linewidth=2)
    ax.set_xlabel("Z slice"); ax.set_ylabel("Auto threshold (gray)")
    ps.style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
              frameon=True, edgecolor="gray")
    ps.save_fig(fig, os.path.join(out, "threshold_drift_vs_z.png"))

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.plot(z, f_air * 100, label="air")
    ax.plot(z, f_ice * 100, label="ice")
    ax.plot(z, f_sand * 100, label="sand")
    ax.set_xlabel("Z slice"); ax.set_ylabel("Interior fraction (%)")
    ps.style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
              frameon=True, edgecolor="gray")
    ps.save_fig(fig, os.path.join(out, "phase_fraction_vs_z.png"))

    print(f"Stage 0b done. Suggested xy_crop = ({gx0}, {gx1+pad}, {gy0}, {gy1+pad})")
    return (gx0, gx1 + pad, gy0, gy1 + pad)


if __name__ == "__main__":
    main()
