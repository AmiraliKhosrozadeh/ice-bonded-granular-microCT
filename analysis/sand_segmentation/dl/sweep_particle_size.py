"""Stage 4 - ParticleSeg3D particle_size SWEEP harness (Sand_100_500_T5_01 subvol).

WHY: at particle_size=0.25 mm the instance seg coverage collapsed (genuine model
under-detection, confirmed by the triage). particle_size is the one knob that
controls how aggressively PS3D resamples our ~10-vox grains toward its trained
60-px target, so it is the first thing to sweep. The image zarr is already built
(ps3d_tiff2zarr was run once); each sweep value only rewrites metadata.json's
particle_size and re-runs inference on the SAME input zarr -> no re-tiff2zarr.

Coverage is scored against the user's Dragonfly-validated SAND reference for THIS
specimen: raw gray > 3644.54 inside the specimen mask (see memory
reference-dragonfly-phase-thresholds / the 100_500_01 bands). This replaces the
noisy watershed/multiOtsu reference used in the triage.

RUNTIME MODEL (per-axis upsample = 60*spacing/particle_size = 1.486/ps):
    ps=0.50 -> 2.97x (~6 min) | 0.40 -> ~11 min | 0.35 -> ~16 min
    ps=0.30 -> ~26 min | 0.25 -> ~45 min (baseline) | 0.20 -> ~88 min (VRAM risk)
    ps<0.18 -> hours + likely OOM on the 8 GB 3070. Smaller ps = more upsample.

RUN (WSL ps3d venv, do NOT run from Windows):
    ~/ps3d/bin/python sweep_particle_size.py
Edit PARTICLE_SIZES below first. Start with a cheap value (0.50) to validate the
harness end-to-end before committing to the slow ones.
"""
import os
import sys
import csv
import glob
import json
import time
import shutil
import subprocess
import numpy as np
import tifffile
import zarr

# ---------------------------------------------------------------- config
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "ps3d_data")
NAME = "Sand-100-500-T5_100XXL_uc_sub"
MODEL = os.path.join(HERE, "model", "Task310_particle_seg")

SPACING_MM = 0.0247660229          # voxel size; FIXED across the sweep
SAND_GRAY = 3644.54                # user's Dragonfly sand line for 100_500_01
T_AIR_ICE = 2682.89                # user's void|ice line (solid = > this)
ZMEAN, ZSTD = 3540.6, 638.5        # solid z-score of THIS dense subvol (prepare_ps3d_input.py)

# specimen mask (cropped-frame, global-z indexed) to confine the sand reference
MASK_DIR = "/mnt/e/RPTU-images/CT_images/Sand/pipeline/results/Sand_100_500_T5_01/stage1_phase"
SUBVOL = (450, 545, 400, 559, 360, 519)   # z0,z1,y0,y1,x0,x1 (cropped coords, inclusive) - dense sand

# values to test (mm). Edit me. Cheap-first is recommended.
PARTICLE_SIZES = [0.50, 0.35, 0.30, 0.25, 0.20]

SWEEP_OUT = os.path.join(ROOT, "predictions_sweep")
CSV_PATH = os.path.join(SWEEP_OUT, "sweep_summary.csv")


# ---------------------------------------------------------------- helpers
def upsample_factor(ps):
    return 60.0 * SPACING_MM / ps


def write_metadata(ps):
    meta = {NAME: {"spacing": SPACING_MM, "particle_size": float(ps)}}
    with open(os.path.join(ROOT, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


PS3D_INFERENCE = os.path.join(os.path.dirname(sys.executable), "ps3d_inference")


def run_inference(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    exe = PS3D_INFERENCE if os.path.exists(PS3D_INFERENCE) else "ps3d_inference"
    cmd = [exe, "-i", ROOT, "-o", out_dir,
           "-m", MODEL, "-n", NAME, "-f", "0", "-batch_size", "1",
           "-p", "0", "-z", str(ZMEAN), str(ZSTD)]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def find_pred_zarr(out_dir):
    hits = glob.glob(os.path.join(out_dir, "**", f"{NAME}.zarr"), recursive=True)
    if not hits:
        hits = glob.glob(os.path.join(out_dir, "**", "*.zarr"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"no prediction zarr under {out_dir}")
    return hits[0]


def load_raw_subvol():
    fs = sorted(glob.glob(os.path.join(ROOT, "tiff", NAME, "*.tif")))
    return np.stack([tifffile.imread(p) for p in fs]).astype(np.float32)


def load_specimen_mask_subvol(shape):
    """Crop the saved specimen mask to the subvol; fall back to all-True."""
    z0, z1, y0, y1, x0, x1 = SUBVOL
    fs = {int("".join(filter(str.isdigit, os.path.basename(p)[-9:]))): p
          for p in glob.glob(os.path.join(MASK_DIR, "specimen_mask_*.tif"))}
    try:
        m = np.stack([tifffile.imread(fs[z])[y0:y1 + 1, x0:x1 + 1] for z in range(z0, z1 + 1)]) > 0
        if m.shape == shape:
            return m
        print(f"  [mask] shape {m.shape} != raw {shape}; using all-True")
    except Exception as e:
        print(f"  [mask] unavailable ({e}); using all-True")
    return np.ones(shape, bool)


def _q3_percentiles(diam_um, vol, qs=(10, 50, 90)):
    """Volume-weighted (Q3) percentiles of equivalent diameter."""
    if len(diam_um) == 0:
        return {q: 0.0 for q in qs}
    order = np.argsort(diam_um)
    d = diam_um[order]
    cum = np.cumsum(vol[order]); cum = cum / cum[-1]
    return {q: float(np.interp(q / 100.0, cum, d)) for q in qs}


def psd(labels):
    """Per-instance volume -> volume-weighted equivalent-diameter percentiles (um)."""
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    vox = sizes[sizes > 0].astype(np.float64)
    if vox.size == 0:
        return {10: 0.0, 50: 0.0, 90: 0.0}, vox
    d_um = (6.0 * vox / np.pi) ** (1.0 / 3.0) * (SPACING_MM * 1000.0)
    return _q3_percentiles(d_um, vox), vox


def score(labels, raw, sand_ref, ice_ref, void_ref, purity_thr=0.70):
    """Full metric suite (ChatGPT spec): coverage + purity + false-labelling +
    volume ratio + gray + per-instance sand purity."""
    seg = labels > 0
    labeled = int(seg.sum())
    sand_vox = int(sand_ref.sum())
    g = raw[seg]

    in_sand = int((seg & sand_ref).sum())
    in_ice = int((seg & ice_ref).sum())
    in_void = int((seg & void_ref).sum())

    coverage = 100.0 * in_sand / max(sand_vox, 1)        # of operator sand, how much labelled
    purity = 100.0 * in_sand / max(labeled, 1)           # of labels, how much is real sand
    false_ice = 100.0 * in_ice / max(labeled, 1)         # labels that are actually ice
    false_void = 100.0 * in_void / max(labeled, 1)       # labels that are actually void
    vol_ratio = labeled / max(sand_vox, 1)               # labelled vol vs operator sand vol

    # per-instance sand purity (fraction of each instance inside operator sand band)
    sizes = np.bincount(labels.ravel())
    sand_per = np.bincount(labels.ravel(), weights=sand_ref.ravel().astype(np.float64))
    ids = np.where(sizes > 0)[0]; ids = ids[ids > 0]
    inst_purity = sand_per[ids] / sizes[ids]
    n_inst = len(ids)
    n_lowpur = int((inst_purity < purity_thr).sum())
    lowpur_volfrac = (100.0 * sizes[ids][inst_purity < purity_thr].sum() / max(sizes[ids].sum(), 1)
                      if n_inst else 0.0)

    pcts, _ = psd(labels)
    return dict(
        n_instances=n_inst, labeled_vox=labeled, sand_ref_vox=sand_vox,
        coverage_pct=coverage, purity_pct=purity,
        false_ice_pct=false_ice, false_void_pct=false_void,
        vol_ratio=vol_ratio,
        median_gray=float(np.median(g)) if g.size else 0.0,
        mean_gray=float(np.mean(g)) if g.size else 0.0,
        d10_um=pcts[10], d50_um=pcts[50], d90_um=pcts[90],
        n_lowpurity=n_lowpur, lowpurity_volfrac=lowpur_volfrac)


def overlay(labels, raw, sand_ref, out_png, ps):
    """3-panel mid-slice QC: raw | operator sand band | PS3D labels on raw."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    z = raw.shape[0] // 2
    seg = (labels[z] > 0)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.5))
    axs[0].imshow(raw[z], cmap="gray"); axs[0].set_title("raw CT (mid)")
    axs[1].imshow(raw[z], cmap="gray")
    axs[1].imshow(np.ma.masked_where(~sand_ref[z], sand_ref[z]),
                  cmap=ListedColormap(["#ffe000"]), alpha=0.55)
    axs[1].set_title(f"operator sand band (>{SAND_GRAY:.0f})")
    axs[2].imshow(raw[z], cmap="gray")
    axs[2].imshow(np.ma.masked_where(~seg, seg),
                  cmap=ListedColormap(["#ff2020"]), alpha=0.6)
    axs[2].set_title("PS3D labels")
    for a in axs: a.axis("off")
    fig.suptitle(f"PS3D QC overlay  particle_size={ps} mm  z={z}",
                 fontsize=16, fontweight="bold")
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    sizes_arg = [float(a) for a in sys.argv[1:]]
    sweep = sizes_arg if sizes_arg else PARTICLE_SIZES
    os.makedirs(SWEEP_OUT, exist_ok=True)
    raw = load_raw_subvol()
    mask = load_specimen_mask_subvol(raw.shape)
    sand_ref = (raw >= SAND_GRAY) & mask
    ice_ref = (raw >= T_AIR_ICE) & (raw < SAND_GRAY) & mask
    void_ref = (raw < T_AIR_ICE) & mask
    print(f"subvol {raw.shape}  operator sand-ref = {int(sand_ref.sum()):,} vox "
          f"({100.0*sand_ref.mean():.1f}% of box)   sweep={sweep}\n")

    rows = []
    for ps in sweep:
        f = upsample_factor(ps)
        print(f"=== particle_size={ps} mm  (upsample {f:.2f}x/axis, ~{f**3:.0f}x volume) ===")
        out_dir = os.path.join(SWEEP_OUT, f"ps_{ps:.3f}")
        write_metadata(ps)
        t0 = time.time()
        reuse = os.environ.get("PS3D_REUSE") == "1"
        if not (reuse and glob.glob(os.path.join(out_dir, "**", "*.zarr"), recursive=True)):
            run_inference(out_dir)
        dt = time.time() - t0
        zp = find_pred_zarr(out_dir)
        labels = np.asarray(zarr.open(zp, mode="r")[:])
        labels = np.rint(labels).astype(np.int64)   # instance IDs come back as float
        if labels.shape != raw.shape:
            print(f"  WARNING: labels {labels.shape} != raw {raw.shape}")
        tifffile.imwrite(os.path.join(out_dir, f"{NAME}_labels.tif"),
                         labels.astype(np.uint16 if labels.max() < 65535 else np.uint32))
        s = score(labels, raw, sand_ref, ice_ref, void_ref)
        s.update(particle_size=ps, upsample=round(f, 2), minutes=round(dt / 60.0, 1))
        overlay(labels, raw, sand_ref, os.path.join(out_dir, "qc_overlay.png"), ps)
        rows.append(s)
        print(f"  -> inst={s['n_instances']}  labeled={s['labeled_vox']:,}\n"
              f"     coverage={s['coverage_pct']:.1f}%  PURITY={s['purity_pct']:.1f}%  "
              f"false-ice={s['false_ice_pct']:.1f}%  false-void={s['false_void_pct']:.1f}%\n"
              f"     volRatio={s['vol_ratio']:.2f}  medGray={s['median_gray']:.0f}  "
              f"meanGray={s['mean_gray']:.0f}  d10/50/90={s['d10_um']:.0f}/{s['d50_um']:.0f}/{s['d90_um']:.0f}um\n"
              f"     low-purity(<70%) instances={s['n_lowpurity']} "
              f"({s['lowpurity_volfrac']:.0f}% of labelled vol)   ({s['minutes']} min)\n")

    cols = ["particle_size", "upsample", "n_instances", "labeled_vox", "sand_ref_vox",
            "coverage_pct", "purity_pct", "false_ice_pct", "false_void_pct", "vol_ratio",
            "median_gray", "mean_gray", "d10_um", "d50_um", "d90_um",
            "n_lowpurity", "lowpurity_volfrac", "minutes"]
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: round(r[k], 3) if isinstance(r[k], float) else r[k] for k in cols})
    print(f"summary appended -> {CSV_PATH}")
    print("\nps     up    inst   cov%   pur%  fIce%  fVoid  d50um  min")
    for r in rows:
        print(f"{r['particle_size']:.3f}  {r['upsample']:.2f}  {r['n_instances']:5d}  "
              f"{r['coverage_pct']:5.1f}  {r['purity_pct']:5.1f}  {r['false_ice_pct']:5.1f}  "
              f"{r['false_void_pct']:5.1f}  {r['d50_um']:5.0f}  {r['minutes']:5.1f}")


if __name__ == "__main__":
    main()
