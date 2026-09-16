"""
Build a Dragonfly-ready CORRECTION PACKAGE for the dense-contact ROI.

WHY: zero-shot ParticleSeg3D failed on this ice-bonded sand (0.50 mm -> 0.6%
coverage, 0.35 mm -> 0.7%; see sweep_summary.csv + memory). Next step is
human-in-the-loop: generate DRAFT instance labels here -> operator corrects them
in Dragonfly -> corrected TIFF -> border-core encoding -> fine-tune. This script
produces the draft + everything the operator needs to correct it.

ROI (dense-contact, where merging is the dominant classical error):
    (x0,x1,y0,y1,z0,z1) = (400, 600, 400, 600, 600, 800)  [pipeline CROPPED frame]
    half-open: x in [400,600), y in [400,600), z in [600,800)  -> 200 x 200 x 200

All four layers are recomputed FRESH on the ROI from the operator's
Dragonfly-validated gray bands (so phase / sand mask / watershed are mutually
consistent), NOT cropped from the full-volume config-threshold outputs:
    void  : gray <  2682.89
    ice   : 2682.89 <= gray < 3644.54
    sand  : gray >= 3644.54           (inside the specimen mask)

Outputs (results/Sand_100_500_T5_01/correction_roi/):
    raw_roi/raw_####.tif                 raw grayscale ROI (uint16)
    phase_roi/phase_####.tif             phase seg 0/1/2/3 (outside/void/ice/sand)
    sand_mask_roi/sand_####.tif          operator sand-band mask (0/255)
    watershed_labels_roi/labels_####.tif DRAFT instance labels (uint16, 1 id/grain)
    grain_labels_roi.tif                 same labels as one multipage TIFF
    boundary_overlay/boundary_####.tif   watershed cut lines (0/255) [optional layer]
    correction_guide.png/.pdf            one-page how-to-correct cheat sheet
    README.txt                           voxel size, ROI coords, label meaning, rules

Run on Windows (classical deps only, no GPU/WSL):
    & C:\\Python313\\python.exe build_correction_package.py
"""
import os
import sys
import glob
import numpy as np
import tifffile
from skimage.segmentation import find_boundaries

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)
sys.path.insert(0, PIPE)
from config import CFG                       # noqa: E402
from stage3_watershed_baseline import watershed_label  # noqa: E402

# ---------------------------------------------------------------- ROI + bands
X0, X1, Y0, Y1, Z0, Z1 = 400, 600, 400, 600, 600, 800   # cropped frame, half-open
T_VOID_ICE = 2682.89                          # operator Dragonfly band
T_ICE_SAND = 3644.54
VOX_UM = CFG.voxel_size_um                     # 24.766 um

OUT = os.path.join(CFG.out_root, "correction_roi")
MASK_DIR = os.path.join(CFG.out_root, "stage1_phase")


def _save_stack(subdir, base, arr):
    d = os.path.join(OUT, subdir)
    os.makedirs(d, exist_ok=True)
    for old in glob.glob(os.path.join(d, f"{base}_*.tif")):
        os.remove(old)
    for z in range(arr.shape[0]):
        tifffile.imwrite(os.path.join(d, f"{base}_{z:04d}.tif"), arr[z])
    print(f"  {arr.shape[0]} slices -> {subdir}\\{base}_####.tif")


def load_raw_roi():
    fs = sorted(glob.glob(os.path.join(CFG.tiff_dir, f"{CFG.file_prefix}*.tif")))
    x0c, x1c, y0c, y1c = CFG.xy_crop          # (255,1182,0,851)
    sl = []
    for z in range(Z0, Z1):
        im = tifffile.imread(fs[z])[y0c:y1c + 1, x0c:x1c + 1]   # specimen crop
        sl.append(im[Y0:Y1, X0:X1])                            # ROI crop
    return np.stack(sl).astype(np.uint16)


def load_mask_roi(shape):
    sl = []
    ok = True
    for z in range(Z0, Z1):
        p = os.path.join(MASK_DIR, f"specimen_mask_{z:04d}.tif")
        if not os.path.exists(p):
            ok = False
            break
        sl.append(tifffile.imread(p)[Y0:Y1, X0:X1] > 0)
    if ok:
        m = np.stack(sl)
        if m.shape == shape:
            print(f"  specimen mask: {100*m.mean():.1f}% of ROI in-specimen")
            return m
    print("  [mask] unavailable/mismatched -> all-True (dense-contact ROI is interior)")
    return np.ones(shape, bool)


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 64)
    print("Correction package - dense-contact ROI")
    print(f"  cropped frame  x[{X0},{X1}) y[{Y0},{Y1}) z[{Z0},{Z1})  -> "
          f"{X1-X0} x {Y1-Y0} x {Z1-Z0}")
    print("=" * 64)

    raw = load_raw_roi()
    print(f"  raw ROI {raw.shape} {raw.dtype}  "
          f"gray min/med/max {raw.min()}/{int(np.median(raw))}/{raw.max()}")
    mask = load_mask_roi(raw.shape)

    # ---- phase seg (operator bands, inside mask) ----
    phase = np.zeros(raw.shape, np.uint8)
    phase[mask & (raw < T_VOID_ICE)] = 1                                   # void
    phase[mask & (raw >= T_VOID_ICE) & (raw < T_ICE_SAND)] = 2            # ice
    phase[mask & (raw >= T_ICE_SAND)] = 3                                  # sand
    fr = {k: 100 * (phase == k).mean() for k in (0, 1, 2, 3)}
    print(f"  phase fractions  outside {fr[0]:.0f}% | void {fr[1]:.0f}% | "
          f"ice {fr[2]:.0f}% | sand {fr[3]:.0f}%")

    # ---- operator sand-band mask ----
    sand = (phase == 3)
    sand_u8 = (sand.astype(np.uint8) * 255)

    # ---- DRAFT watershed instance labels on the sand mask ----
    print("  watershed draft labels ...")
    labels, _seeds, _edt = watershed_label(sand, verbose=True)
    labels = labels.astype(np.uint16 if labels.max() < 65535 else np.uint32)
    n_grains = int(labels.max())

    # ---- boundary (cut-line) overlay ----
    bnd = (find_boundaries(labels, mode="outer") & sand).astype(np.uint8) * 255

    # ---- write Dragonfly stacks ----
    _save_stack("raw_roi", "raw", raw)
    _save_stack("phase_roi", "phase", phase)
    _save_stack("sand_mask_roi", "sand", sand_u8)
    _save_stack("watershed_labels_roi", "labels", labels)
    _save_stack("boundary_overlay", "boundary", bnd)
    tifffile.imwrite(os.path.join(OUT, "grain_labels_roi.tif"), labels)
    print(f"  grain_labels_roi.tif  ({n_grains} draft grains)")

    correction_guide(raw, sand, labels)
    draft_qc(raw, sand, labels)
    write_readme(raw.shape, n_grains, fr)
    print(f"\nDONE -> {OUT}")
    print("Open raw_roi + watershed_labels_roi in Dragonfly, convert labels to "
          "Multi-ROI, correct per README, export corrected label TIFF.")


# ---------------------------------------------------------------- draft QC
def draft_qc(raw, sand, labels):
    """Real ROI mid-slice: raw | operator sand band | draft watershed labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    z = raw.shape[0] // 2
    rng = np.random.default_rng(0)
    lut = np.vstack([[0, 0, 0], rng.random((int(labels.max()) + 1, 3))])
    fig, axs = plt.subplots(1, 3, figsize=(19, 7))
    axs[0].imshow(raw[z], cmap="gray"); axs[0].set_title("raw CT (ROI mid)")
    axs[1].imshow(raw[z], cmap="gray")
    axs[1].imshow(np.ma.masked_where(~sand[z], sand[z]),
                  cmap=ListedColormap(["#ffe000"]), alpha=0.55)
    axs[1].set_title(f"operator sand band (>={T_ICE_SAND:.0f})")
    axs[2].imshow(lut[labels[z]]); axs[2].set_title(f"draft watershed labels "
                                                    f"({int(labels.max())} grains)")
    for a in axs:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Correction-package draft QC  -  dense-contact ROI  z={z}",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "draft_qc.png"), dpi=140, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "draft_qc.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  draft_qc.png/.pdf")


# ---------------------------------------------------------------- guide image
def correction_guide(raw, sand, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    rng = np.random.default_rng(0)
    lut = np.vstack([[0, 0, 0], rng.random((int(labels.max()) + 1, 3))])

    # synthetic 1-page cheat sheet: each case is a 2-row (before/after) mini panel
    fig, axs = plt.subplots(2, 5, figsize=(22, 9))
    cases = ["GOOD\nsingle-grain label", "SPLIT\nmerged grains",
             "MERGE\nover-split pieces", "DELETE\nfalse speck in ice/void",
             "IGNORE / FLAG\nunder-resolved fine"]

    def blob(ax, cx, cy, r, val, cmap, title=None):
        yy, xx = np.ogrid[:80, :80]
        g = np.zeros((80, 80))
        for (ccx, ccy, rr) in ([(cx, cy, r)] if np.isscalar(cx) else zip(cx, cy, r)):
            g = np.maximum(g, np.exp(-(((xx-ccx)**2+(yy-ccy)**2)/(2*(rr*0.6)**2))))
        ax.imshow(g, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=12, fontweight="bold")
        return g

    # --- column 0: GOOD ---
    g = blob(axs[0, 0], 40, 40, 20, 1, "gray", "BEFORE")
    g2 = blob(axs[1, 0], 40, 40, 20, 1, "gray")
    axs[1, 0].imshow(np.ma.masked_where(g2 < 0.3, np.ones_like(g2)),
                     cmap=ListedColormap(["#33cc33"]), alpha=0.6)
    axs[1, 0].set_title("AFTER  (1 grain = 1 id)", fontsize=11)
    axs[1, 0].set_xticks([]); axs[1, 0].set_yticks([])

    # --- column 1: SPLIT (two touching grains under one label) ---
    g = blob(axs[0, 1], [28, 52], [40, 40], [16, 16], 1, "gray", "BEFORE\n(one label, two grains)")
    a = np.zeros_like(g); a[:, :40] = 1; a[:, 40:] = 2; a[g < 0.3] = 0
    axs[1, 1].imshow(np.ma.masked_where(a == 0, a),
                     cmap=ListedColormap(["#ff5050", "#3070ff"]))
    axs[1, 1].set_title("AFTER  (split -> 2 ids)", fontsize=11)
    axs[1, 1].set_xticks([]); axs[1, 1].set_yticks([])

    # --- column 2: MERGE (one grain cut into pieces) ---
    g = blob(axs[0, 2], 40, 40, 22, 1, "gray", "BEFORE\n(one grain, two labels)")
    a = np.zeros_like(g); a[:40, :] = 1; a[40:, :] = 2; a[g < 0.3] = 0
    axs[0, 2].imshow(np.ma.masked_where(a == 0, a),
                     cmap=ListedColormap(["#ff5050", "#3070ff"]), alpha=0.55)
    g2 = blob(axs[1, 2], 40, 40, 22, 1, "gray")
    axs[1, 2].imshow(np.ma.masked_where(g2 < 0.3, np.ones_like(g2)),
                     cmap=ListedColormap(["#33cc33"]), alpha=0.6)
    axs[1, 2].set_title("AFTER  (merge -> 1 id)", fontsize=11)

    # --- column 3: DELETE false speck (label sitting on dark ice/void) ---
    g = np.full((80, 80), 0.25)
    axs[0, 3].imshow(g, cmap="gray", vmin=0, vmax=1)
    axs[0, 3].add_patch(plt.Circle((40, 40), 6, color="#ff5050", alpha=0.8))
    axs[0, 3].set_title("BEFORE\n(label on ice/void)", fontsize=11)
    axs[0, 3].set_xticks([]); axs[0, 3].set_yticks([])
    axs[1, 3].imshow(g, cmap="gray", vmin=0, vmax=1)
    axs[1, 3].text(40, 40, "X", color="red", fontsize=40, ha="center", va="center",
                   fontweight="bold")
    axs[1, 3].set_title("AFTER  (deleted)", fontsize=11)
    axs[1, 3].set_xticks([]); axs[1, 3].set_yticks([])

    # --- column 4: IGNORE under-resolved fine (a few faint voxels) ---
    g = blob(axs[0, 4], 40, 40, 4, 1, "gray", "BEFORE\n(~3-voxel blur)")
    axs[1, 4].imshow(g, cmap="gray", vmin=0, vmax=1)
    axs[1, 4].text(40, 70, "leave unlabeled\n(do NOT invent)", color="orange",
                   fontsize=11, ha="center", fontweight="bold")
    axs[1, 4].set_title("AFTER  (ignore / flag)", fontsize=11)
    axs[1, 4].set_xticks([]); axs[1, 4].set_yticks([])

    for j, c in enumerate(cases):
        axs[0, j].set_ylabel("")
        axs[0, j].annotate(c, xy=(0.5, 1.28), xycoords="axes fraction",
                           ha="center", va="bottom", fontsize=13, fontweight="bold")
    fig.suptitle("Dragonfly grain-label correction guide  -  dense-contact ROI  "
                 f"(voxel {VOX_UM:.1f} um;  sand = gray >= {T_ICE_SAND:.0f})",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT, "correction_guide.png"), dpi=140, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "correction_guide.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("  correction_guide.png/.pdf")


# ---------------------------------------------------------------- readme
def write_readme(shape, n_grains, fr):
    nz, ny, nx = shape
    rx0, rx1 = X0 + CFG.xy_crop[0], X1 + CFG.xy_crop[0]   # raw-frame X (crop x0=255)
    txt = f"""DENSE-CONTACT ROI - DRAGONFLY CORRECTION PACKAGE
================================================================
Specimen : {CFG.sample_name}  (Sand_100_500_T5_01, scan 1)
Voxel    : {VOX_UM:.4f} um isotropic
ROI size : {nx} x {ny} x {nz}  (X x Y x Z)

COORDINATE FRAMES
  Pipeline CROPPED frame (half-open):  X[{X0},{X1})  Y[{Y0},{Y1})  Z[{Z0},{Z1})
  Raw-scan frame:                      X[{rx0},{rx1})  Y[{Y0},{Y1})  Z[{Z0},{Z1})
  (cropped X = raw X - {CFG.xy_crop[0]}; Y and Z share the raw indices.)

GRAY BANDS (operator Dragonfly-validated, used to build this package)
  void  : gray <  {T_VOID_ICE:.2f}
  ice   : {T_VOID_ICE:.2f} <= gray < {T_ICE_SAND:.2f}
  sand  : gray >= {T_ICE_SAND:.2f}
  ROI phase fractions: void {fr[1]:.0f}% | ice {fr[2]:.0f}% | sand {fr[3]:.0f}% | outside {fr[0]:.0f}%

FILES
  raw_roi/raw_####.tif                 raw grayscale (uint16) - the reference image
  phase_roi/phase_####.tif             phase seg: 0 outside, 1 void, 2 ice, 3 sand
  sand_mask_roi/sand_####.tif          sand-band mask (0 / 255) - label ONLY here
  watershed_labels_roi/labels_####.tif DRAFT instance labels ({n_grains} grains, uint16)
  grain_labels_roi.tif                 same draft labels as one multipage TIFF
  boundary_overlay/boundary_####.tif   current watershed cut-lines (0 / 255)
  correction_guide.png                 one-page visual cheat sheet of the 5 cases

DRAGONFLY WORKFLOW
  1. Import raw_roi/ as an image stack (the reference).
  2. Import watershed_labels_roi/ as an image stack, then
     Convert to Multi-ROI / Labelled volume (one ROI per integer id).
  3. (optional) Import sand_mask_roi/ and boundary_overlay/ as guides.
  4. Correct the Multi-ROI against the raw image (rules below).
  5. Export the corrected labels as a TIFF stack (one integer id per grain).

CORRECTION RULES (see correction_guide.png)
  * Every visible sand grain -> exactly one unique integer label.
  * SPLIT merged grains (one label spanning two touching grains -> two ids).
  * MERGE over-split pieces (one grain cut into several ids -> one id).
  * DELETE false labels sitting on ice/void (keep labels on sand only).
  * Keep labels ONLY on sand (gray >= {T_ICE_SAND:.0f}); trim into ice.
  * Do NOT invent invisible sub-voxel fines (~<=3 voxels) - leave them unlabeled.
  * Preserve realistic grain boundaries from the raw CT + sand-band mask.

AFTER EXPORT (Claude will run)
  1. QC the corrected labels vs raw / phase / sand mask.
  2. Border-core encode  (dl/encode_border_core.py).
  3. ParticleSeg3D fine-tune / custom border-core nnU-Net training.

This draft is the CLASSICAL watershed baseline (over-merges contacts, over-splits
angular grains, misses fines). It is a STARTING POINT to edit, NOT ground truth.
"""
    with open(os.path.join(OUT, "README.txt"), "w") as f:
        f.write(txt)
    print("  README.txt")


if __name__ == "__main__":
    main()
