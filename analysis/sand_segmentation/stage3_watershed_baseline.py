"""
Stage 3 - Classical grain-instance baseline (EDT + H-maxima + marker watershed).

This is the CLASSICAL benchmark that ParticleSeg3D must beat, and the source
of pseudo-labels for fine-tuning. It WILL over/under-segment angular grains and
miss fines -- that failure is the point: it tells us exactly what the CNN must fix.

Method (standard touching-grain separation):
  1. Euclidean distance transform (EDT) of the clean sand mask.
  2. Suppress shallow EDT maxima with an h-maxima transform (merges spurious
     seeds inside one grain -> less over-segmentation).
  3. Seeds = local maxima of the suppressed EDT (>= seed_min_distance apart).
  4. Marker-controlled watershed on the inverted EDT, masked to sand.
  5. Size filter; relabel consecutively.

Outputs (results/<sample>/stage3_baseline/):
  grain_labels.tif           SPAM-compatible integer label volume (one id/grain)
  baseline_labels_####.tif   colored label stack for Dragonfly QC
  baseline_psd.png           grain-size distribution (equivalent-diameter)
  baseline_qc.png            mid-slice: sand mask | seeds | labels
  baseline_report.txt        grain count, PSD summary, merge/split & fines warnings
Run:  python stage3_watershed_baseline.py
"""
import os
import numpy as np
from scipy import ndimage
from skimage.morphology import h_maxima
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from config import CFG
import common
import plot_style as ps


def watershed_label(sand_mask, cfg=CFG, verbose=True):
    if verbose:
        print("  EDT ...")
    # float32 EDT (scipy returns float64 -> 2x memory; cast down for big volumes)
    edt = ndimage.distance_transform_edt(sand_mask).astype(np.float32)

    # Seeding. h_maxima (grayscale reconstruction) is robust but allocates
    # several full-size copies -> OOM on >~1 Gvoxel volumes. Above a size guard,
    # fall back to peak_local_max with an EDT-height threshold (threshold_abs=h),
    # which suppresses shallow noise maxima like h-maxima but is memory-light.
    BIG = 800_000_000
    if sand_mask.size <= BIG:
        if verbose:
            print(f"  h-maxima (h={cfg.hmaxima_h}) ...")
        hm = h_maxima(edt, cfg.hmaxima_h)
        coords = peak_local_max(edt, min_distance=cfg.seed_min_distance,
                                labels=(hm > 0).astype(np.uint8) * sand_mask)
        del hm
    else:
        if verbose:
            print(f"  peak_local_max (threshold_abs={cfg.hmaxima_h}) "
                  f"[memory-light, {sand_mask.size/1e9:.1f} Gvox] ...")
        coords = peak_local_max(edt, min_distance=cfg.seed_min_distance,
                                threshold_abs=cfg.hmaxima_h, labels=sand_mask)
    seeds = np.zeros(sand_mask.shape, dtype=np.int32)
    for i, (z, y, x) in enumerate(coords, start=1):
        seeds[z, y, x] = i
    if verbose:
        print(f"  {coords.shape[0]} seeds")

    if verbose:
        print("  watershed ...")
    labels = watershed(-edt, markers=seeds, mask=sand_mask)

    # size filter + consecutive relabel
    sizes = np.bincount(labels.ravel())
    keep = np.where(sizes >= cfg.watershed_min_grain_voxels)[0]
    keep = keep[keep != 0]
    remap = np.zeros(labels.max() + 1, dtype=np.int32)
    remap[keep] = np.arange(1, len(keep) + 1)
    labels = remap[labels]
    if verbose:
        print(f"  {len(keep)} grains after size filter (>= {cfg.watershed_min_grain_voxels} vox)")
    return labels, seeds, edt


def equiv_diameters_um(labels, vox):
    sizes = np.bincount(labels.ravel())
    sizes = sizes[1:]  # drop background
    sizes = sizes[sizes > 0]
    d_vox = (6.0 * sizes / np.pi) ** (1.0 / 3.0)   # equivalent-sphere diameter
    return d_vox * vox, sizes


def main():
    import stage2_sand_mask as s2
    clean = s2.main()
    out = CFG.d("stage3_baseline")
    print("=" * 64)
    print("Stage 3 - classical watershed baseline")
    print("=" * 64)

    labels, seeds, edt = watershed_label(clean)
    vox = CFG.voxel_size_um
    d_um, sizes = equiv_diameters_um(labels, vox)
    n_grains = len(sizes)

    # --- report ----------------------------------------------------------
    floor_um = CFG.fines_floor_voxels * vox
    n_below_floor = int((d_um < 2 * floor_um).sum())
    lines = []
    P = lines.append
    P(f"Sample            : {CFG.sample_name}")
    P(f"Grains labelled   : {n_grains}")
    if n_grains:
        P(f"Equiv-diameter um : min {d_um.min():.0f} | median {np.median(d_um):.0f} | "
          f"max {d_um.max():.0f} | mean {d_um.mean():.0f}")
        P(f"d10/d50/d90 (um)  : {np.percentile(d_um,10):.0f} / "
          f"{np.percentile(d_um,50):.0f} / {np.percentile(d_um,90):.0f}")
        P(f"Grain volume vox  : min {sizes.min()} | median {int(np.median(sizes))} | "
          f"max {sizes.max()}")
    P("")
    P(f"Nominal sieve range: {CFG.min_grain_um:.0f}-{CFG.max_grain_um:.0f} um")
    P(f"Fines floor        : ~{2*floor_um:.0f} um (2x the {CFG.fines_floor_voxels:.0f}-voxel limit)")
    P(f"Grains below floor : {n_below_floor} "
      f"({100*n_below_floor/max(n_grains,1):.1f}% of labelled grains)")
    P("")
    P("CAVEATS of this classical baseline (expected to be fixed by ParticleSeg3D):")
    P("  - MERGED grains: low/over-suppressed seeds fuse touching grains "
      "(raise hmaxima_h fewer seeds -> more merges).")
    P("  - SPLIT grains: angular/elongated grains get multiple EDT peaks -> "
      "one grain cut into pieces (lower hmaxima_h -> more splits).")
    P("  - MISSED fines: grains near the voxel floor have no stable EDT maximum "
      "and are dropped or absorbed by neighbours.")
    P("  Use these labels as pseudo-labels + benchmark, NOT as final results.")
    report = "\n".join(lines)
    with open(os.path.join(out, "baseline_report.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report + "\n")

    # --- save labels -----------------------------------------------------
    common.save_label_image(os.path.join(out, "grain_labels.tif"), labels)
    # colored stack for Dragonfly QC (random LUT via modulo)
    disp = (labels % 250 + (labels > 0)).astype(np.uint8)
    common.save_tiff_stack(out, "baseline_labels", disp)

    # --- figures ---------------------------------------------------------
    ps.apply_style()
    import matplotlib.pyplot as plt

    # PSD (one plot per file)
    if n_grains:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.hist(d_um, bins=40, color=[0.9, 0.8, 0.45], edgecolor="black")
        ax.axvspan(CFG.min_grain_um, CFG.max_grain_um, color="green", alpha=0.12)
        ax.axvline(2 * floor_um, color="red", linestyle="--", linewidth=ps.LW)
        ax.set_xlabel("Equivalent diameter (um)")
        ax.set_ylabel("Grain count")
        ps.style_axes(ax)
        ax.legend(["fines floor", "PSD", "nominal sieve band"],
                  frameon=True, edgecolor="gray")
        ps.save_fig(fig, os.path.join(out, "baseline_psd.png"))

    # QC triptych on the mid slice
    zc = clean.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    axes[0].imshow(clean[zc], cmap="gray"); axes[0].set_title("sand mask")
    axes[1].imshow(edt[zc], cmap="magma")
    sy, sx = np.where(seeds[zc] > 0)
    axes[1].scatter(sx, sy, s=18, c="cyan", marker="x"); axes[1].set_title("EDT + seeds")
    rng = np.random.default_rng(0)
    lut = np.vstack([[0, 0, 0], rng.random((labels.max() + 1, 3))])
    axes[2].imshow(lut[labels[zc]]); axes[2].set_title("watershed labels")
    for ax in axes:
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ps.style_axes(ax); ax.grid(False)
    ps.save_fig(fig, os.path.join(out, "baseline_qc.png"))

    print("Stage 3 done. grain_labels.tif is SPAM-ready; review baseline_qc.png "
          "for merge/split/fines failure modes before moving to ParticleSeg3D.")
    return labels


if __name__ == "__main__":
    main()
