"""Re-run the remaining Alumina DIC with the settings that worked on Glass,
plus the grey-scale normalisation those three specimens need and T7 did not.

WHAT WENT WRONG ORIGINALLY
All four Alumina runs used MAX_ITERATIONS 50 (Glass: 100), UPDATE_GRADIENT
False (True), MULTISCALE_BINNING 1 (2), and a zero seed translation (Glass:
-58.6 voxels). Alumina beads move ~30 voxels with a radius of ~35, so with no
initial guess and no coarse pass every bead started a full radius from its
target -- the edge of the convergence basin. Re-running T7 1->2 with the Glass
settings took it from 47/472 converged to 288/474, which settles it.

WHY THESE THREE NEED ONE MORE THING
T7's two scans already matched in grey scale (grain peak 6013 -> 6452, 7%
apart), so the settings alone were enough there. These three do not match --
the reported mismatches are 2.0x, 3.9x and 5.4x. spam-ddic minimises a
sum-of-squared-grey-difference residual, so a scan that is uniformly 4x
brighter starts every correlation with a huge residual that no amount of
iteration can remove by moving the bead.

The fix is a two-landmark linear map, not a histogram equalisation. Two
physical levels are identifiable in both scans without any correlation:

    pore   the grey inside the specimen mask but OUTSIDE every bead label
    grain  the grey inside the bead labels

Mapping (pore_B, grain_B) -> (pore_A, grain_A) with I* = a*I + b puts the two
scans on one radiometric scale while leaving the geometry untouched, which is
what the correlation needs. A non-linear equalisation could deform the local
grey gradients that the Newton step differentiates, so it is deliberately not
used.

Landmarks are measured on every 10th slice and normalisation is applied slice
by slice, so peak memory stays near one slice rather than one volume.

The original tif is never modified: a normalised copy of scan B is written
beside it as ct_scanNN_aligned_norm.tif and that copy is what spam-ddic reads.

Usage:  python _alumina_ddic.py [specimen ...]      (default: all three)
"""
import os
import subprocess
import sys

import numpy as np
import tifffile
from scipy import ndimage as ndi

R = '/mnt/e/RPTU-images/CT_images/Alumina'
RES_ROOT = '/home/amirali_wsl/spam-results-alumina'
LO_PCT, HI_PCT = 5, 95
A, B = 1, 2
SAMPLE_EVERY = 10
# a normalised copy is only written when the map is materially non-trivial
GAIN_TOL = 0.10

# ITERATION CAP, measured rather than copied from T7.
#
# The first pass at -it 100 (the setting that fixed T7) gave Alumina_175 only
# 68/483. But the 369 near-misses were NOT diverging: their deltaPhiNorm median
# was 0.0159 against a 0.001 threshold, and 100% of them stopped at the cap.
# Half of all beads sat within ~14x of the threshold and still improving. That
# is a starved iteration budget, not a failed correlation.
#
# The reason this differs from T7 is the size of the motion: these beads move
# 77 voxels median = 2.2 bead radii, against roughly 0.9 for T7. The seed
# removes the bulk of it, but the per-bead residual still takes many more
# Newton steps to close.
ITERS = 300
MSB = 2
PHI_HEADER = ('NodeNumber\tZpos\tYpos\tXpos\tFzz\tFzy\tFzx\tZdisp\t'
              'Fyz\tFyy\tFyx\tYdisp\tFxz\tFxy\tFxx\tXdisp\t'
              'error\titerations\treturnStatus\tdeltaPhiNorm')

SPECIMENS = ['Alumina_175_1800_T5', 'Alumina_100_1800_T5', 'Alumina_75_1000_T5']
if len(sys.argv) > 1:
    SPECIMENS = sys.argv[1:]
os.makedirs(RES_ROOT, exist_ok=True)


def paths(name):
    spam = f'{R}/{name}/{name}_spam'
    return spam, f'{spam}/data', f'{spam}/results_voidcrack'


def shape_of(path):
    with tifffile.TiffFile(path) as tf:
        return (len(tf.pages),) + tuple(tf.pages[0].shape)


def mask_path(data, rd, s, ct_shape):
    """A mask in the SAME FRAME as the CT, or none at all.

    These three specimens were segmented with USE_RAW=1, so the masks under
    results_voidcrack are in the RAW frame while every DIC input is in the
    ALIGNED frame: 1315 slices against 1348 for Alumina_175_1800_T5, and one
    scan is even a column narrower (574x573 against 574x574). Mixing the two
    frames is not a broadcasting nuisance to be patched over -- the mask
    centroid would be measured in one coordinate system and the bead centroids
    in another, so the seed translation would be silently wrong. T7 worked only
    because it ran USE_RAW=0 and its frames happen to match.

    So the shape is CHECKED, never assumed, and a mask that does not match the
    CT exactly is refused. Returning none is safe: the caller then measures
    grey landmarks from filled bead labels and takes the lateral seed from bead
    centroids, both of which live in the CT frame by construction.
    """
    for p, tag in ((f'{data}/specimen_mask_scan{s:02d}_aligned.tif', 'aligned'),
                   (f'{rd}/sample_scan{s:02d}.tif', 'corrected')):
        if os.path.exists(p):
            sh = shape_of(p)
            if sh == ct_shape:
                return p, tag
            print(f'    mask {os.path.basename(p)} is {sh}, CT is {ct_shape}'
                  f' -- wrong frame, refused', flush=True)
    return None, 'none (bead-derived)'


def landmarks(ct_file, lab_file, mask_file):
    """Median grey of grain and of pore, sampled on every SAMPLE_EVERY slice."""
    grain, pore = [], []
    tm = tifffile.TiffFile(mask_file) if mask_file else None
    with tifffile.TiffFile(ct_file) as tc, tifffile.TiffFile(lab_file) as tl:
        n = min(len(tc.pages), len(tl.pages))
        for z in range(0, n, SAMPLE_EVERY):
            lab = tl.pages[z].asarray()
            if not lab.any():
                continue
            ct = tc.pages[z].asarray()
            bead = lab > 0
            inside = None
            if tm is not None and z < len(tm.pages):
                m = tm.pages[z].asarray() > 0
                # belt and braces: mask_path already refused a mismatched
                # frame, so a mismatch here would be a new bug, not a case to
                # paper over with a crop
                if m.shape == bead.shape:
                    inside = m
            if inside is None:
                inside = ndi.binary_fill_holes(bead)
            gap = inside & ~bead
            if bead.sum() > 200:
                grain.append(np.median(ct[bead]))
            if gap.sum() > 200:
                pore.append(np.median(ct[gap]))
    if tm is not None:
        tm.close()
    if not grain or not pore:
        return None, None
    return float(np.median(grain)), float(np.median(pore))


def normalise(src, dst, a, b):
    """Write a*I + b of src, slice by slice, clipped into uint16."""
    with tifffile.TiffFile(src) as tf, tifffile.TiffWriter(dst, bigtiff=True) as tw:
        for z in range(len(tf.pages)):
            sl = tf.pages[z].asarray().astype(np.float32) * a + b
            tw.write(np.clip(sl, 0, 65535).astype(np.uint16), contiguous=True)
    print(f'    wrote {dst}', flush=True)


def bead_centroids(lab_file):
    lab = tifffile.imread(lab_file)
    ids = np.unique(lab)
    ids = ids[ids > 0]
    cent = np.array(ndi.center_of_mass(lab > 0, lab, ids))
    shape = lab.shape
    del lab
    return cent, shape


def mask_radius_and_centre(mask_file):
    m = tifffile.imread(mask_file) > 0
    zs = np.where(m.any(axis=(1, 2)))[0]
    z0, z1 = zs[0], zs[-1]
    radii, ys, xs = [], [], []
    for z in range(z0 + (z1 - z0) // 3, z0 + 2 * (z1 - z0) // 3, 20):
        sl = m[z]
        if sl.sum() < 100:
            continue
        yy, xx = np.nonzero(sl)
        radii.append(np.sqrt(sl.sum() / np.pi))
        ys.append(yy.mean())
        xs.append(xx.mean())
    centre = np.array(np.nonzero(m)).mean(axis=1)
    del m
    return float(np.mean(radii)), float(np.mean(ys)), float(np.mean(xs)), centre


def convergence(tsv):
    if not os.path.exists(tsv):
        return None
    hdr = open(tsv).readline().strip().split('\t')
    arr = np.genfromtxt(tsv, delimiter='\t', skip_header=1)
    if arr.ndim == 1:
        arr = arr[None, :]
    rs = arr[:, hdr.index('returnStatus')].astype(int)
    dp = arr[:, hdr.index('deltaPhiNorm')]
    it = arr[:, hdr.index('iterations')]
    return (int((rs == 2).sum()), len(rs), float(np.nanmedian(dp)),
            float(np.mean(it >= ITERS)))


for name in SPECIMENS:
    print(f'\n{"=" * 70}\n{name}\n{"=" * 70}', flush=True)
    spam, data, rd = paths(name)
    res = f'{RES_ROOT}/{name}'
    os.makedirs(res, exist_ok=True)

    ct_a = f'{data}/ct_scan{A:02d}_aligned.tif'
    ct_b = f'{data}/ct_scan{B:02d}_aligned.tif'
    lab_a = f'{data}/bead_labels_scan{A:02d}_aligned.tif'
    lab_b = f'{data}/bead_labels_scan{B:02d}_aligned.tif'
    if not all(os.path.exists(p) for p in (ct_a, ct_b, lab_a, lab_b)):
        print('  missing input volumes, skipped', flush=True)
        continue
    ct_shape = shape_of(ct_a)
    if shape_of(ct_b) != ct_shape:
        print(f'  scan {A} is {ct_shape} but scan {B} is {shape_of(ct_b)} -- '
              f'the two CT volumes are not in the same frame, skipped',
              flush=True)
        continue
    mk_a, src_a = mask_path(data, rd, A, ct_shape)
    mk_b, src_b = mask_path(data, rd, B, ct_shape)
    print(f'  CT frame {ct_shape}   masks: scan{A} {src_a}   scan{B} {src_b}',
          flush=True)

    # ---- grey-scale landmarks and the linear map ----------------------------
    ga, pa = landmarks(ct_a, lab_a, mk_a)
    gb, pb = landmarks(ct_b, lab_b, mk_b)
    if ga is None or gb is None:
        print('  could not measure grey landmarks, skipped', flush=True)
        continue
    print(f'  grey landmarks   scan{A}: pore {pa:.0f}  grain {ga:.0f}  '
          f'(contrast {ga - pa:.0f})')
    print(f'                   scan{B}: pore {pb:.0f}  grain {gb:.0f}  '
          f'(contrast {gb - pb:.0f})', flush=True)
    if abs(gb - pb) < 1e-6:
        print('  scan B has no pore/grain contrast, skipped', flush=True)
        continue
    gain = (ga - pa) / (gb - pb)
    off = pa - gain * pb
    print(f'  map  I* = {gain:.4f} I {off:+.1f}   '
          f'(grain mismatch {gb / max(ga, 1e-9):.2f}x before)', flush=True)

    ct_b_use = ct_b
    if abs(gain - 1.0) > GAIN_TOL or abs(off) > 0.05 * max(ga - pa, 1.0):
        ct_b_use = f'{data}/ct_scan{B:02d}_aligned_norm.tif'
        if os.path.exists(ct_b_use):
            print('    normalised copy already present, reusing', flush=True)
        else:
            print(f'    normalising scan {B} ...', flush=True)
            normalise(ct_b, ct_b_use, gain, off)
    else:
        print('    map is close to identity, using the original', flush=True)

    # ---- geometric seed, same construction as the T7 run that worked --------
    ca, shape_a = bead_centroids(lab_a)
    cb, _ = bead_centroids(lab_b)
    print(f'  bead labels: {len(ca)} / {len(cb)}   frame {shape_a}', flush=True)
    F = np.eye(3)
    t = np.zeros(3)
    lo_a, hi_a = np.percentile(ca[:, 0], [LO_PCT, HI_PCT])
    lo_b, hi_b = np.percentile(cb[:, 0], [LO_PCT, HI_PCT])
    F[0, 0] = (hi_b - lo_b) / (hi_a - lo_a)
    t[0] = lo_b - F[0, 0] * lo_a
    # LATERAL STRETCH FROM BEAD CENTROIDS, NOT FROM A MASK.
    #
    # The only masks in the CT frame for these three specimens are the OLD
    # aligned ones -- the corrected masks were written with USE_RAW=1 and live
    # in the raw frame. Those old masks are exactly the ones whose specimen
    # finding was wrong: on this specimen they gave a lateral stretch of 1.136
    # (radius 199.4 -> 226.5) and an implied volume GAIN of +26.7% under
    # compression, which cannot be right. At radius ~200 voxels that is a
    # 27-voxel error at the edge, most of a bead radius, seeded into every
    # correlation.
    #
    # Bead centroids avoid the problem entirely: they are in the CT frame by
    # construction and they ARE the material. RMS radius rather than max, so a
    # single spalled bead or a differing label count (482 vs 465 here) cannot
    # set the scale.
    ctr_a, ctr_b = ca[:, 1:].mean(axis=0), cb[:, 1:].mean(axis=0)
    rms_a = float(np.sqrt((((ca[:, 1:] - ctr_a) ** 2).sum(axis=1)).mean()))
    rms_b = float(np.sqrt((((cb[:, 1:] - ctr_b) ** 2).sum(axis=1)).mean()))
    lat = rms_b / max(rms_a, 1e-9)
    centre = np.array([ca[:, 0].mean(), ctr_a[0], ctr_a[1]])
    t[1] = ctr_b[0] - lat * ctr_a[0]
    t[2] = ctr_b[1] - lat * ctr_a[1]
    note = f'bead RMS radius {rms_a:.1f} -> {rms_b:.1f}'
    F[1, 1] = F[2, 2] = lat
    vr = F[0, 0] * F[1, 1] * F[2, 2]
    print(f'  seed: Fzz {F[0, 0]:.4f} ({100 * (F[0, 0] - 1):+.1f}% axial)   '
          f'lateral {lat:.4f} ({note})   volume {100 * (vr - 1):+.1f}%',
          flush=True)
    if vr > 1.02:
        print('  WARNING: seed implies volume GAIN under compression',
              flush=True)

    disp = F @ centre + t - centre
    z, y, x = centre
    row = (f'1\t{z:.7f}\t{y:.7f}\t{x:.7f}\t'
           f'{F[0, 0]:.7f}\t{F[0, 1]:.7f}\t{F[0, 2]:.7f}\t{disp[0]:.7f}\t'
           f'{F[1, 0]:.7f}\t{F[1, 1]:.7f}\t{F[1, 2]:.7f}\t{disp[1]:.7f}\t'
           f'{F[2, 0]:.7f}\t{F[2, 1]:.7f}\t{F[2, 2]:.7f}\t{disp[2]:.7f}\t'
           f'0.0\t0\t2\t0.0')
    seed = f'{res}/phi_seed_{A}{B}.tsv'
    open(seed, 'w').write(PHI_HEADER + '\n' + row + '\n')
    print(f'  seed disp at node = ({disp[0]:+.1f}, {disp[1]:+.1f}, '
          f'{disp[2]:+.1f}) vox  -> {seed}', flush=True)

    # ---- correlate ---------------------------------------------------------
    prefix = f'{name}_ddic_{A}{B}'
    cmd = ['spam-ddic', ct_a, lab_a, ct_b_use,
           '-pf', seed, '-F', 'all', '-np', '8', '-ld', '2', '-m', '15',
           '-it', str(ITERS), '-msb', str(MSB), '-ug', '-od', res,
           '-pre', prefix]
    print('  running spam-ddic ...', flush=True)
    log = f'{res}/{prefix}.log'
    rc = subprocess.run(cmd, stdout=open(log, 'w'),
                        stderr=subprocess.STDOUT).returncode
    print(f'  spam-ddic exited {rc}', flush=True)

    c = convergence(f'{res}/{prefix}-ddic.tsv')
    if c is None:
        print('  NO OUTPUT TSV', flush=True)
        continue
    nc, nt, dp, cap = c
    print(f'  RESULT  {nc}/{nt} converged ({100 * nc / max(nt, 1):.1f}%)   '
          f'deltaPhiNorm median {dp:.4f}   at iteration cap {100 * cap:.0f}%',
          flush=True)

print('\nALUMINA DDIC SWEEP COMPLETE', flush=True)
