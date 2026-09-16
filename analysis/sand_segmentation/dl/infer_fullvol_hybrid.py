"""
Full-volume HYBRID sand-grain instance segmentation.
====================================================
Combines (a) the operator's validated gray bands for the SEMANTIC sand mask
(robust, trusted) with (b) the trained 3D border-core U-Net only for the hard
part -- SPLITTING touching grains.  This sidesteps the synth->real domain gap
(synthetic background was ice ~3150; real non-sand space is mostly void ~1500),
which made pure-DL over-call grains in void and grab the bright container wall.

Pipeline
--------
1  load the full CT stack (slice dir or 3D tif)
2  specimen interior mask  : (gray > T_AIR_ICE) -> close -> largest CC -> fill
3  remove the container WALL: (gray > WALL_GRAY) dilated, subtracted
4  sand mask               : interior & (gray >= T_SAND), opened
5  restrict to the specimen bbox (skip the empty frame) for speed/memory
6  DL border-core prediction over the bbox (sliding window, GPU patches,
   CPU accumulators) -> CORE voxels
7  HYBRID decode: markers = connected-components(core & sand) ; watershed on
   the inverse EDT of the sand mask -> per-grain instances ; size filter
8  write grain labels (bbox frame) + bbox.json + PSD summary

Then validate the PSD against Camsizer:
    SAND_SPECIMEN=Sand_100_500_T5_01 python ../stage_validate_psd.py \
        results/dl_grain_labels_full.tif

Run (WSL, GPU):
    ~/ps3d/bin/python infer_fullvol_hybrid.py \
        --in /mnt/e/RPTU-images/CT_images/Sand/Sand_100_500_T5_01 \
        --out results [--zmax 200 for a dry run]
"""
import os
import glob
import json
import time
import argparse
import numpy as np
import tifffile
import torch
from scipy import ndimage
from skimage.segmentation import watershed

from unet3d import UNet3D, znorm, sliding_window_logits
from decode_instances import decode  # noqa: F401 (kept for pure-DL option)

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_CKPT = os.path.join(HERE, 'model', 'bordercore_unet_best.pt')

# --- operator gray bands for Sand_100_500_T5_01 (memory: project-sand-dl) ----
T_AIR_ICE = 2682.89     # void < this
T_SAND    = 3644.54     # sand >= this
WALL_GRAY = 9000.0      # bright alumina container wall
VOX_UM    = 24.7660229


def load_volume(path, zmin, zmax):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '*.tif')) +
                       glob.glob(os.path.join(path, '*.tiff')))
        files = files[zmin:(zmax if zmax is not None else len(files))]
        s0 = tifffile.imread(files[0])
        vol = np.empty((len(files),) + s0.shape, s0.dtype)
        vol[0] = s0
        for i, f in enumerate(files[1:], 1):
            vol[i] = tifffile.imread(f)
        print(f"  loaded {len(files)} slices {vol.shape} {vol.dtype}")
        return vol
    vol = tifffile.imread(path)
    return vol[zmin:(zmax if zmax is not None else len(vol))]


def predict_chunked(model, sub, patch, overlap, slab_z, device, halo=None):
    """Memory-bounded border-core argmax over `sub` (uint16), processing z in
    slabs with a `halo`-wide context margin so accumulators never span the whole
    bbox.

    Each slab is z-scored by its OWN statistics (matches the validated crop
    behavior).  Using one GLOBAL mean/std over the whole void-dominated bbox
    pushed sand to too-high z-scores -> the model under-predicted cores ->
    dense clumps merged into mega-grains.  Per-slab stats fix that.

    `halo` defaults to `patch` (full one-patch context for seamless slab joins).
    For a LARGE XY frame (e.g. the 1305^2 75_200 scans) the per-slab CPU
    accumulators (3 x haloed_z x Y x X float32) blow the WSL RAM cap, so the
    caller may pass a smaller halo (>= patch//2) + smaller slab_z to shrink the
    haloed z-extent.  The decode stage has its own halo, so a slightly softer
    prediction seam every slab_z slices is acceptable."""
    Z, Y, X = sub.shape
    pred = np.zeros((Z, Y, X), np.uint8)
    if halo is None:
        halo = patch
    z = 0
    while z < Z:
        ze = min(Z, z + slab_z)
        a0 = max(0, z - halo); a1 = min(Z, ze + halo)         # haloed read window
        slab = sub[a0:a1].astype(np.float32)
        slab = (slab - slab.mean()) / (slab.std() + 1e-6)     # per-slab z-score
        prob = sliding_window_logits(model, slab, patch=patch, overlap=overlap,
                                     device=device, acc_device='cpu')
        arg = prob.argmax(0).astype(np.uint8)
        pred[z:ze] = arg[z - a0:ze - a0]                      # keep central region
        del prob, arg, slab
        print(f"    slab z[{z}:{ze}] done", flush=True)
        z = ze
    return pred


def decode_slabbed(pred, sand, min_vox, slab_z=128, halo=48,
                   use_edt=False, edt_min_dist=3, edt_min_h=1.0):
    """Memory-bounded hybrid decode over z-slabs.

    Each slab is processed with a `halo` margin so any grain straddling the
    central boundary is fully present.  A grain is OWNED by the slab whose
    central z-band contains its centroid (grains << halo, so it is complete
    there) -> no seam splits, no double counting.  Returns (labels, n).

    Markers for the watershed are the DL CORE connected-components.  For very
    fine, densely packed / ice-bonded sand (the 75_200 specimens, grains only
    3-8 voxels) the border-core net under-produces cores, so adjacent grains
    flood together into non-physical mega-grains.  With `use_edt=True` we add
    the local maxima of the sand distance-transform (one seed per grain-centre)
    as SUPPLEMENTARY markers: a peak that falls inside an existing core blob
    just merges into it (no over-split), while a peak in a marker-less clump
    becomes its own seed -> the clump gets split.  `edt_min_dist` is the minimum
    voxel separation between peaks (~grain radius) and `edt_min_h` the minimum
    distance-to-boundary for a peak to count (rejects boundary noise)."""
    if use_edt:
        from skimage.feature import peak_local_max
    Z = pred.shape[0]
    out = np.zeros(pred.shape, np.int32)
    gid = 0
    z = 0
    while z < Z:
        ze = min(Z, z + slab_z)
        a0 = max(0, z - halo); a1 = min(Z, ze + halo)
        sand_s = sand[a0:a1]
        core = (pred[a0:a1] == 1) & sand_s
        edt = ndimage.distance_transform_edt(sand_s).astype(np.float32)
        if use_edt:
            peaks = peak_local_max(edt, min_distance=edt_min_dist,
                                   threshold_abs=edt_min_h, exclude_border=False)
            if len(peaks):
                core[tuple(peaks.T)] = True       # union: cores + EDT peaks
            del peaks
        markers, _ = ndimage.label(core)
        del core
        edt *= -1.0
        lab = watershed(edt, markers=markers, mask=sand_s)
        del edt, markers
        nlab = int(lab.max()) + 1
        sizes = np.bincount(lab.ravel(), minlength=nlab).astype(np.int64)
        sumz = np.zeros(nlab, np.float64)
        for zi in range(lab.shape[0]):                       # centroid-z per label
            sumz += np.bincount(lab[zi].ravel(), minlength=nlab) * zi
        czl = sumz / np.maximum(sizes, 1)
        cz_lo, cz_hi = z - a0, ze - a0
        keep = (sizes >= min_vox) & (czl >= cz_lo) & (czl < cz_hi)
        keep[0] = False
        nk = int(keep.sum())
        remap = np.zeros(nlab, np.int32)
        remap[keep] = gid + np.arange(1, nk + 1)
        gid += nk
        seg = remap[lab]
        w = seg > 0
        out[a0:a1][w] = seg[w]
        del lab, seg, w, sizes, sumz, czl, remap
        print(f"    decode slab z[{z}:{ze}] -> +{nk} grains (total {gid})", flush=True)
        z = ze
    return out, gid


def granular_zrange(sand, interior=None, sf_lo=0.12, sf_hi=0.85,
                    lgcc_max=0.40):
    """Find the granular sand COLUMN along z, excluding the solid end caps.

    The operator gray band also grabs the dense end caps / platens at the two
    axial extremes -- each is one 10-20M-voxel SOLID blob (separated from the
    grains by an air gap), not packed grains.  Watershed on those solid masses
    with sparse DL cores yields mega-grains that poison the volume-Q3 tail.

    A z-slice is 'granular' when its sand fraction is in a normal range AND its
    sand is NOT dominated by a single giant connected component:
        sf_lo <= sandfrac <= sf_hi   AND   largest_CC / sand_vox < lgcc_max
    Solid caps fail (lgcc ~ 1.0, high sandfrac); a SPARSE grain column passes
    (lgcc~0.03).  But a DENSE / ice-bonded column is itself one connected network
    (lgcc up to ~0.97), so the lgcc test would reject the whole column and fall
    back to the full z-range -> caps creep back in.  So we run BOTH tests and
    adapt: if the lgcc-restricted column is much shorter than the plain
    sandfrac-band column, the pack is dense and we drop lgcc, relying on the
    sandfrac upper bound + air-gap contiguity to exclude the caps instead.
    Returns (zlo, zhi) of the LONGEST contiguous granular run (zhi exclusive)."""
    # SAND FRACTION IS TAKEN INSIDE THE SPECIMEN, NOT OVER THE FRAME.
    # It used to be n / ss.size, the fraction of the WHOLE SLICE including the
    # empty frame outside the container, tested against a fixed band
    # [0.02, 0.15].  That quantity depends on how much of the frame the
    # specimen happens to occupy, so it is not comparable between scans: over
    # these seven it runs 0.022 to 0.061, a factor of 2.8, and the 25 mm series
    # exceeds 0.15 outright once compacted.  Slices then fall out of the band
    # for reasons that have nothing to do with whether they are granular, and
    # the detected column changes arbitrarily between load steps -- 1017 -> 497
    # -> 779 slices on the 25 mm series, and 431 -> 764 on the 75-200 series,
    # a column apparently LENGTHENING under compression.
    #
    # Dividing by the specimen cross-section instead gives the solid volume
    # fraction, which is physical and comparable: 0.21-0.53 across these scans,
    # against ~1.0 for a solid end cap.  The band is set on that scale.
    Z = sand.shape[0]
    in_band = np.zeros(Z, bool)     # solid VF inside the specimen, in band
    g_lgcc = np.zeros(Z, bool)      # in-band AND not single-blob-dominated
    for z in range(Z):
        ss = sand[z]; n = int(ss.sum())
        if n == 0:
            continue
        denom = int(interior[z].sum()) if interior is not None else ss.size
        if denom < 500:             # too little specimen in this slice to judge
            continue
        sf = n / denom
        if not (sf_lo <= sf <= sf_hi):
            continue
        in_band[z] = True
        lbl, k = ndimage.label(ss)
        if k and np.bincount(lbl.ravel())[1:].max() / n < lgcc_max:
            g_lgcc[z] = True

    def longest_run(mask):
        best_lo = best_hi = lo = 0
        in_run = False
        for z in range(Z + 1):
            on = mask[z] if z < Z else False
            if on and not in_run:
                lo = z; in_run = True
            elif not on and in_run:
                if z - lo > best_hi - best_lo:
                    best_lo, best_hi = lo, z
                in_run = False
        return best_lo, best_hi

    lo_l, hi_l = longest_run(g_lgcc)
    lo_b, hi_b = longest_run(in_band)
    # THE SOLID-FRACTION BAND NOW EXCLUDES THE CAPS BY ITSELF, so the
    # largest-connected-component test is no longer needed to do it, and it
    # actively harms these specimens.  An ice-bonded column IS one connected
    # network, so lgcc rejects the densest, most bonded part of the column --
    # exactly the part the analysis is about.
    #
    # The old rule fell back to the band only when the lgcc run was under HALF
    # the band run, which is a knife edge: the 25 mm stage-2 scan passes 966 of
    # 1174 slices on the band with a 910-slice contiguous run, but its lgcc run
    # of 483 is just over the 455 needed to win, so lgcc took it and the column
    # was truncated to the top 41%.  The measured caps sit at a solid fraction
    # of 0.99, well outside the 0.85 upper bound, so the band already removes
    # them: on this scan it rejects 208 slices whose median fraction is 0.99.
    #
    # lgcc is kept only as a tie-break for a LOOSE pack, where it agrees with
    # the band anyway (within 10%); it can no longer overrule it.
    if (hi_l - lo_l) >= 0.9 * (hi_b - lo_b):
        return lo_l, hi_l
    return lo_b, hi_b


def build_masks(vol, t_air_ice=T_AIR_ICE, t_sand=T_SAND, wall_gray=WALL_GRAY):
    """Per-slice specimen interior; wall removed; band sand mask.

    Gray bands are SPECIMEN-SPECIFIC (scan calibration differs) -- pass the
    operator's Dragonfly-validated values per scan via the CLI.

    Memory-frugal: everything except the final 3D opening is computed slice by
    slice, so at most `vol` + the two output bool arrays + tiny per-slice
    temporaries are resident.  The old version built the wall / sand masks with
    full-volume vectorized ops (vol>wall_gray, ~wall, vol>=t_sand) that held
    3-4 extra full bool arrays at once -- on the 1305^2 75_200 frame that
    overran the 16 GB WSL cap (OOM-killed, exit 15) before prediction even
    started.  Per-slice 2D wall dilation is adequate (the wall is a thick bright
    ring); the grain-splitting opening stays 3D."""
    Z = vol.shape[0]
    interior = np.zeros(vol.shape, bool)
    sand = np.zeros(vol.shape, bool)
    for z in range(Z):
        sl = vol[z]
        m = sl > t_air_ice                            # sand+ice+wall
        m = ndimage.binary_closing(m, iterations=3)
        lbl, n = ndimage.label(m)
        if n == 0:
            continue
        big = np.argmax(np.bincount(lbl.ravel())[1:]) + 1
        intz = ndimage.binary_fill_holes(lbl == big)
        wallz = ndimage.binary_dilation(sl > wall_gray, iterations=2)
        intz &= ~wallz
        interior[z] = intz
        # 2D per-slice opening (de-speckle) -- done INSIDE the loop so we never
        # allocate the transient full-volume copies that scipy's 3D
        # binary_opening makes (erosion+dilation each copy the 1305^2 x 1128
        # bool array -> ~+9 GB spike -> OOM-killed on the big 75_200 frame).
        # 2D is also gentler, preserving more of the 3-8 voxel fine grains.
        sand[z] = ndimage.binary_opening(intz & (sl >= t_sand), iterations=1)
    return interior, sand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', dest='out', required=True)
    ap.add_argument('--ckpt', default=DEF_CKPT)
    ap.add_argument('--patch', type=int, default=96)
    ap.add_argument('--overlap', type=float, default=0.5)
    ap.add_argument('--min', type=int, default=20)
    ap.add_argument('--slab_z', type=int, default=128, help='z-slab for chunked prediction')
    ap.add_argument('--halo', type=int, default=None,
                    help='prediction context margin (default=patch); lower it '
                         '(>=patch//2) to fit a large XY frame in RAM')
    ap.add_argument('--zmin', type=int, default=0)
    ap.add_argument('--zmax', type=int, default=None)
    ap.add_argument('--reuse', action='store_true', help='reuse cached prediction if present')
    ap.add_argument('--no-granular', dest='no_granular', action='store_true',
                    help='disable granular-column restriction (keep solid end caps)')
    ap.add_argument('--t_air_ice', type=float, default=T_AIR_ICE,
                    help='void|ice gray threshold (specimen interior)')
    ap.add_argument('--t_sand', type=float, default=T_SAND,
                    help='ice|sand gray threshold (sand mask)')
    ap.add_argument('--wall', type=float, default=WALL_GRAY,
                    help='sand|wall gray threshold (bright container wall)')
    ap.add_argument('--edt_markers', action='store_true',
                    help='add distance-transform peak seeds to the DL cores '
                         '(splits dense fine-grain clumps; use for 75_200)')
    ap.add_argument('--edt_min_dist', type=int, default=3,
                    help='min voxel separation between EDT peak seeds (~grain radius)')
    ap.add_argument('--edt_min_h', type=float, default=1.0,
                    help='min distance-to-boundary for an EDT peak seed')
    ap.add_argument('--no_dl', action='store_true',
                    help='ablation baseline: skip the U-Net entirely and split '
                         'grains with EDT-peak watershed markers ONLY (no learned '
                         'cores). Forces --edt_markers on.')
    a = ap.parse_args()
    if a.no_dl:
        a.edt_markers = True   # EDT peaks are the ONLY marker source without DL
    os.makedirs(a.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    T0 = time.time()

    print("loading volume ...")
    vol = load_volume(a.inp, a.zmin, a.zmax)

    print(f"building specimen + sand masks (bands: air|ice={a.t_air_ice:.0f} "
          f"ice|sand={a.t_sand:.0f} wall={a.wall:.0f}) ...")
    t = time.time()
    interior, sand = build_masks(vol, a.t_air_ice, a.t_sand, a.wall)
    print(f"  interior {interior.mean():.3f}  sand {sand.mean():.3f}  "
          f"({int(sand.sum()):,} sand vox)  {time.time()-t:.0f}s")
    if sand.sum() == 0:
        raise RuntimeError("empty sand mask -- check bands / input")

    # restrict to the granular sand COLUMN (drop the solid end caps/platens that
    # the gray band grabs at the axial extremes -> avoids mega-merge artifacts)
    if not a.no_granular and not a.reuse:
        zlo, zhi = granular_zrange(sand, interior)
        if zhi - zlo < 8:
            print(f"  WARNING granular column too short z[{zlo}:{zhi}] -- "
                  f"keeping full z-range")
        else:
            print(f"  granular column z[{zlo}:{zhi}] "
                  f"({zhi-zlo}/{sand.shape[0]} slices) -- caps excluded")
            interior[:zlo] = False; interior[zhi:] = False
            sand[:zlo] = False;     sand[zhi:] = False

    # bbox of the specimen interior (skip empty frame).  Use cheap per-axis
    # any()-reductions, NOT np.argwhere: argwhere materializes an (N,3) int64
    # array of EVERY foreground voxel -- on the dense 75_200 frame that is
    # ~874M voxels x 3 x 8B = ~21 GB -> instant OOM.  The reductions are O(vol)
    # in time but allocate only three 1D boolean axis-profiles.
    zany = interior.any(axis=(1, 2))
    yany = interior.any(axis=(0, 2))
    xany = interior.any(axis=(0, 1))
    zc, yc, xc = np.where(zany)[0], np.where(yany)[0], np.where(xany)[0]
    z0, z1 = int(zc[0]), int(zc[-1]) + 1
    y0, y1 = int(yc[0]), int(yc[-1]) + 1
    x0, x1 = int(xc[0]), int(xc[-1]) + 1
    pad = 8
    z0 = max(0, z0 - pad); y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
    z1 = min(vol.shape[0], z1 + pad); y1 = min(vol.shape[1], y1 + pad); x1 = min(vol.shape[2], x1 + pad)
    print(f"  specimen bbox z[{z0}:{z1}] y[{y0}:{y1}] x[{x0}:{x1}]")
    sub = np.ascontiguousarray(vol[z0:z1, y0:y1, x0:x1])
    sand_b = np.ascontiguousarray(sand[z0:z1, y0:y1, x0:x1])
    # free the full-frame arrays (WSL is memory-capped) -- keep only bbox crops
    full_shape = vol.shape
    del vol, interior, sand

    cache = os.path.join(a.out, 'fullvol_pred_cache.npz')
    if a.no_dl:
        print("NO-DL baseline: skipping U-Net; markers = EDT peaks only")
        pred = np.zeros(sub.shape, np.uint8)     # no learned cores anywhere
    elif a.reuse and os.path.exists(cache):
        print(f"reusing cached prediction {cache}")
        z = np.load(cache)
        pred = z['pred']; sand_b = z['sand']
    else:
        print("loading model ...")
        ck = torch.load(a.ckpt, map_location=device, weights_only=False)
        model = UNet3D(1, 3, base=ck.get('base', 24)).to(device)
        model.load_state_dict(ck['model'])
        print(f"  {a.ckpt} epoch {ck.get('epoch')} val dice {ck.get('val_dice_mean'):.3f}")
        print("DL border-core prediction (chunked sliding window) ...")
        t = time.time()
        pred = predict_chunked(model, sub, a.patch, a.overlap,
                               slab_z=a.slab_z, device=device, halo=a.halo)
        print(f"  fg {(pred>0).mean():.3f} core {(pred==1).mean():.3f} "
              f"border {(pred==2).mean():.3f}  {time.time()-t:.0f}s")
        np.savez_compressed(cache, pred=pred, sand=sand_b)
        print(f"  cached prediction -> {cache}")

    print("hybrid decode (chunked: DL core markers, watershed within band sand) ...")
    t = time.time()
    del sub                                   # not needed for decode; free RAM
    labels, n = decode_slabbed(pred, sand_b, a.min, slab_z=a.slab_z,
                               use_edt=a.edt_markers, edt_min_dist=a.edt_min_dist,
                               edt_min_h=a.edt_min_h)
    del pred, sand_b
    print(f"  {n} grains  {time.time()-t:.0f}s")

    # write labels (bbox frame) + bbox metadata
    dt = np.uint16 if labels.max() < 65535 else np.uint32
    lab_path = os.path.join(a.out, 'dl_grain_labels_full.tif')
    tifffile.imwrite(lab_path, labels.astype(dt))
    with open(os.path.join(a.out, 'dl_grain_labels_bbox.json'), 'w') as fh:
        json.dump({'bbox_zyx': [int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)],
                   'full_shape': [int(s) for s in full_shape],
                   'n_grains': int(n), 'voxel_um': VOX_UM,
                   'zmin_slice': a.zmin}, fh, indent=2)
    print(f"  wrote {lab_path} ({n} grains, {dt.__name__})")

    # PSD summary (volume-weighted, equiv-sphere)
    sizes = np.bincount(labels.ravel())[1:]; sizes = sizes[sizes > 0]
    d = (6.0 * sizes / np.pi) ** (1.0 / 3.0) * VOX_UM
    vol_w = sizes.astype(float)
    order = np.argsort(d); Q3 = 100 * np.cumsum(vol_w[order]) / vol_w.sum()
    q = {p: float(np.interp(p, Q3, d[order])) for p in (10, 50, 90)}
    cnt = {p: float(np.percentile(d, p)) for p in (10, 50, 90)}
    print(f"\nPSD volume-Q3  d10/d50/d90 = {q[10]:.0f}/{q[50]:.0f}/{q[90]:.0f} um")
    print(f"PSD count      d10/d50/d90 = {cnt[10]:.0f}/{cnt[50]:.0f}/{cnt[90]:.0f} um")
    print(f"\nTOTAL {time.time()-T0:.0f}s. Next: validate vs Camsizer ->")
    print(f"  SAND_SPECIMEN=Sand_100_500_T5_01 python ../stage_validate_psd.py {lab_path}")


if __name__ == '__main__':
    main()
