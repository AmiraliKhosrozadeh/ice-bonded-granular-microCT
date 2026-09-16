"""
Stage 0 - Scan audit (no GPU, no segmentation commitment).

Answers the questions ChatGPT flagged BEFORE any CNN work:
  * Are sand / ice / air separable by threshold? (interior multi-Otsu)
  * How bad is partial-volume mixing? (peak overlap, valley depth)
  * What is the SNR / contrast-to-noise between ice and sand?
  * How many voxels is a fine grain? (fines-floor risk)

Outputs (results/<sample>/stage0_audit/):
  audit_report.txt            numbers + suggested thresholds to paste in config
  audit_interior_hist.png     interior (in-specimen) gray histogram + Otsu lines
  audit_slice_raw.png         representative slice, grayscale
  audit_slice_mask.png        same slice with specimen-mask outline

Run:  python stage0_audit.py
"""
import os
import numpy as np
from skimage.filters import threshold_otsu

from config import CFG
import common
import plot_style as ps


def main():
    out = CFG.d("stage0_audit")
    print("=" * 64)
    print(f"Stage 0 - scan audit : {CFG.sample_name}")
    print("=" * 64)

    # Subsampled load for a cheap global view of the whole stack.
    vol, idx = common.load_volume(CFG, subsample=25)
    vox = CFG.voxel_size_um

    # Rough material threshold (separates near-zero outside air from specimen),
    # then the specimen plug mask, then the interior 3-phase Otsu.
    t_material = int(threshold_otsu(vol))
    mask = common.specimen_mask(vol, t_material,
                                close_radius=CFG.specimen_close_radius,
                                min_area_frac=CFG.specimen_min_area_frac)
    interior = vol[mask]
    t_air_ice, t_ice_sand = common.multi_otsu_phases(vol, mask)

    # Phase stats inside the specimen.
    air  = interior[interior < t_air_ice]
    ice  = interior[(interior >= t_air_ice) & (interior < t_ice_sand)]
    sand = interior[interior >= t_ice_sand]

    def stat(a):
        return (a.mean(), a.std(), 100.0 * a.size / interior.size) if a.size else (0, 0, 0)
    air_m, air_s, air_f = stat(air)
    ice_m, ice_s, ice_f = stat(ice)
    snd_m, snd_s, snd_f = stat(sand)

    # Contrast-to-noise ratio ice<->sand (the hard boundary for instances).
    cnr = abs(snd_m - ice_m) / np.sqrt(0.5 * (snd_s**2 + ice_s**2) + 1e-9)
    # In-phase SNR of sand (signal / noise within the sand population).
    snr_sand = snd_m / (snd_s + 1e-9)

    gmin_vox = CFG.min_grain_um / vox
    gmax_vox = CFG.max_grain_um / vox

    # ---- report ----------------------------------------------------------
    lines = []
    P = lines.append
    P(f"Sample              : {CFG.sample_name}")
    P(f"Slices (full stack) : {len(common.list_slices(CFG.tiff_dir, CFG.file_prefix))}")
    P(f"Slice shape         : {vol.shape[1]} x {vol.shape[2]} (Y x X)")
    P(f"Voxel size          : {vox:.4f} um")
    P(f"Grain size (nominal): {CFG.min_grain_um:.0f}-{CFG.max_grain_um:.0f} um "
      f"= {gmin_vox:.1f}-{gmax_vox:.1f} voxels")
    P("")
    P("Suggested phase thresholds (interior 3-class multi-Otsu):")
    P(f"  t_air_ice  = {t_air_ice}    (gray < this        -> air void)")
    P(f"  t_ice_sand = {t_ice_sand}    (this <= gray < ... -> ice; >= -> sand)")
    P(f"  (rough material threshold for specimen mask: {t_material})")
    P("")
    P(f"Interior voxels in mask: {interior.size:,}")
    P(f"  {'phase':<6} {'mean':>8} {'std':>8} {'vol%':>7}")
    P(f"  {'air':<6} {air_m:>8.0f} {air_s:>8.0f} {air_f:>7.2f}")
    P(f"  {'ice':<6} {ice_m:>8.0f} {ice_s:>8.0f} {ice_f:>7.2f}")
    P(f"  {'sand':<6} {snd_m:>8.0f} {snd_s:>8.0f} {snd_f:>7.2f}")
    P("")
    P(f"Ice<->Sand contrast-to-noise (CNR): {cnr:.2f}   "
      f"({'GOOD, thresholdable' if cnr > 2 else 'LOW - expect partial-volume smear; consider 3-class nnU-Net'})")
    P(f"Sand in-phase SNR                 : {snr_sand:.2f}")
    P("")
    floor_um = CFG.fines_floor_voxels * vox
    P(f"Fines-floor risk: instance methods miss grains < ~{CFG.fines_floor_voxels:.0f} voxels "
      f"(~{floor_um:.0f} um here).")
    if gmin_vox < 2 * CFG.fines_floor_voxels:
        P(f"  WARNING: smallest nominal grain ({CFG.min_grain_um:.0f} um = {gmin_vox:.1f} vox) is "
          f"near/below the floor. Fines will be UNRELIABLE at this voxel size.")
        P(f"  -> plan a high-res static rescan (~10-12 um) for quantitative fines PSD/shape.")
    report = "\n".join(lines)
    with open(os.path.join(out, "audit_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report + "\n")

    # ---- figures (one plot per file, strict style) -----------------------
    ps.apply_style()
    import matplotlib.pyplot as plt

    # interior histogram + threshold lines
    fig, ax = plt.subplots(figsize=(12, 8))
    hi = np.percentile(interior, 99.5)
    ax.hist(interior[interior <= hi], bins=200, color=[0.3, 0.45, 0.7])
    for t, lab in [(t_air_ice, "air|ice"), (t_ice_sand, "ice|sand")]:
        ax.axvline(t, color="black", linewidth=ps.LW, linestyle="--")
    ax.set_xlabel("Gray value (interior)")
    ax.set_ylabel("Voxel count")
    ps.style_axes(ax)
    ax.legend(["air | ice", "ice | sand"], frameon=True, edgecolor="gray")
    ps.save_fig(fig, os.path.join(out, "audit_interior_hist.png"))

    # representative slice raw + mask outline
    zmid = vol.shape[0] // 2
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(vol[zmid], cmap="gray", vmax=hi)
    ax.set_xlabel("X (voxels)"); ax.set_ylabel("Y (voxels)")
    ps.style_axes(ax); ax.grid(False)
    ps.save_fig(fig, os.path.join(out, "audit_slice_raw.png"))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(vol[zmid], cmap="gray", vmax=hi)
    ax.contour(mask[zmid], levels=[0.5], colors="red", linewidths=2)
    ax.set_xlabel("X (voxels)"); ax.set_ylabel("Y (voxels)")
    ps.style_axes(ax); ax.grid(False)
    ps.save_fig(fig, os.path.join(out, "audit_slice_mask.png"))

    print("Stage 0 done. Review audit_report.txt, then set thresholds in config.py "
          "(or leave None to auto-use these).")
    return dict(t_air_ice=t_air_ice, t_ice_sand=t_ice_sand, t_material=t_material)


if __name__ == "__main__":
    main()
