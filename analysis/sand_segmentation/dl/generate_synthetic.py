"""
Synthetic ice-bonded sand generator for DL grain-instance training.
=====================================================================
Sand Atlas does not expose downloadable labelled voxel volumes (the bulk
server 500s; only the per-grain shape CSVs download).  So we SYNTHESISE
training volumes whose grain SIZE + SHAPE statistics come from the real
Sand Atlas CSVs, packed into dense touching configurations with PERFECT
per-grain labels, then rendered as a CT-like grayscale matched to the
USER's frozen-sand scan (gray bands void<2682.89<ice<3644.54<sand,
voxel 24.766 um, plus PSF blur + noise + partial-volume).

Output (nnU-Net / ParticleSeg3D friendly):
    synth/imagesTr/sand_XXXX_0000.tif   uint16 grayscale
    synth/labelsTr/sand_XXXX.tif        int32  per-grain instance labels
    synth/manifest.csv

Why this is the right training target: the model must learn to (a) tell
sand from ice/air at the user's contrast, and (b) split TOUCHING grains
at the user's resolution where the contact is only partially resolved.
The synthetic packs contain exactly those flat partially-resolved
contacts, with ground-truth labels we control.

Run (Windows Python with numpy/scipy/tifffile):
    C:\\Python313\\python.exe generate_synthetic.py --n 24 --val 4
    C:\\Python313\\python.exe generate_synthetic.py --test   # 1 quick volume
"""
import os
import csv
import argparse
import numpy as np
import tifffile
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(HERE, 'sand_atlas', 'csv')
OUT_DIR = os.path.join(HERE, 'sand_atlas', 'synth')

# ---- USER scan parameters (match the target domain) -------------------------
VOXEL_UM   = 24.7660229
# Gray means (16-bit) chosen so noisy samples land in the operator's bands:
#   air  < 2682.89 <= ice < 3644.54 <= sand
GRAY_AIR   = 1900.0
GRAY_ICE   = 3150.0
GRAY_SAND  = 5200.0
NOISE_STD  = 430.0     # additive Gaussian (per voxel)
PSF_SIGMA  = 0.7       # Gaussian PSF in target voxels (partial-volume softener)

# ---- grain shape sources ----------------------------------------------------
CSV_FILES = {
    'hamburg':  'hamburg-0000004.csv',   # 8037 quartz grains, best size match
    'ottawa73': 'ottawa-0000009.csv',    # 3367 rounded quartz
    'hostun':   'hostun-0000005.csv',    # 54 angular quartz
}

# Rescale real equivalent diameters into the user's 100-500 um sieve band so the
# synthetic PSD matches the specimen (set to None to keep the CSV's native PSD).
TARGET_D_MIN_UM = 90.0
TARGET_D_MAX_UM = 520.0


def load_shapes(which=('hamburg', 'ottawa73', 'hostun')):
    """Return arrays: equiv_diam(um), elongation(mid/maj), flatness(min/mid)."""
    eds, els, fls = [], [], []
    for key in which:
        path = os.path.join(CSV_DIR, CSV_FILES[key])
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                ma = float(row['Major Axis Length (µm)'])
                mi = float(row['Middle Axis Length (µm)'])
                mn = float(row['Minor Axis Length (µm)'])
                ed = float(row['Equivalent Diameter (µm)'])
                if ma <= 0 or mi <= 0:
                    continue
                eds.append(ed)
                els.append(mi / ma)
                fls.append(mn / mi)
    eds = np.asarray(eds); els = np.asarray(els); fls = np.asarray(fls)
    if TARGET_D_MIN_UM is not None:
        lo, hi = np.percentile(eds, 2), np.percentile(eds, 98)
        eds = TARGET_D_MIN_UM + (eds - lo) * (TARGET_D_MAX_UM - TARGET_D_MIN_UM) / (hi - lo)
        eds = np.clip(eds, TARGET_D_MIN_UM * 0.8, TARGET_D_MAX_UM * 1.1)
    return eds, els, fls


def rand_rotation(rng):
    """Uniform random 3x3 rotation matrix (QR of a Gaussian)."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q *= np.sign(np.diag(r))            # fix signs -> proper-ish
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def grain_mask(shape, center, semi, R, expo):
    """Hard boolean mask of one superellipsoid (single voxel-centre eval).

    Returns (mask, (z0,z1,y0,y1,x0,x1)) or (None, None) if off-grid."""
    rad = semi.max() + 1.0
    lo = np.maximum(np.floor(center - rad).astype(int), 0)
    hi = np.minimum(np.ceil(center + rad).astype(int) + 1, np.array(shape))
    if np.any(hi <= lo):
        return None, None
    zz = np.arange(lo[0], hi[0]); yy = np.arange(lo[1], hi[1]); xx = np.arange(lo[2], hi[2])
    Z, Y, X = np.meshgrid(zz - center[0], yy - center[1], xx - center[2], indexing='ij')
    inv = R.T                       # world->grain
    gz = inv[0, 0] * Z + inv[0, 1] * Y + inv[0, 2] * X
    gy = inv[1, 0] * Z + inv[1, 1] * Y + inv[1, 2] * X
    gx = inv[2, 0] * Z + inv[2, 1] * Y + inv[2, 2] * X
    a, b, c = semi
    val = (np.abs(gz / a) ** expo + np.abs(gy / b) ** expo + np.abs(gx / c) ** expo)
    mask = val <= 1.0
    return mask, (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])


def make_volume(shape, eds, els, fls, rng, solid_frac=0.50,
                max_grains=3000, max_attempts=8000):
    """Pack superellipsoid grains (earlier grain wins contacts -> flat, fully
    touching boundaries) -> (labels int32, sand_frac float32, n_grains)."""
    labels = np.zeros(shape, np.int32)
    target_vox = solid_frac * np.prod(shape)
    sand_vox = 0
    nid = 0
    attempts = 0
    while sand_vox < target_vox and nid < max_grains and attempts < max_attempts:
        attempts += 1
        i = rng.integers(len(eds))
        ed_vox = eds[i] / VOXEL_UM
        el = np.clip(els[i] + rng.normal(0, 0.04), 0.4, 1.0)
        fl = np.clip(fls[i] + rng.normal(0, 0.04), 0.35, 1.0)
        a = 0.5 * ed_vox * (1.0 / (el * fl)) ** (1 / 3)   # a>=b>=c
        b = a * el
        c = b * fl
        semi = np.array([a, b, c])
        if semi.min() < 1.3:          # skip sub-resolution grains (<~2.6 vox)
            continue
        center = np.array([rng.uniform(0, shape[d]) for d in range(3)])
        ci = np.clip(center.astype(int), 0, np.array(shape) - 1)
        if labels[ci[0], ci[1], ci[2]] > 0:    # centre already buried -> skip cheap
            continue
        R = rand_rotation(rng)
        expo = rng.uniform(2.0, 3.2)          # 2=ellipsoid, >2 angular
        mask, bb = grain_mask(shape, center, semi, R, expo)
        if mask is None:
            continue
        z0, z1, y0, y1, x0, x1 = bb
        sub = labels[z0:z1, y0:y1, x0:x1]
        free = mask & (sub == 0)              # earlier grain keeps contested voxels
        if free.sum() < 8:                    # too occluded to be a real grain
            continue
        nid += 1
        sub[free] = nid
        sand_vox += int(free.sum())
    sand = (labels > 0).astype(np.float32)
    return labels, sand, nid


def render_ct(sand_frac, rng, p_air=0.5):
    """Render CT-like grayscale: ice background (+ optional air pockets),
    sand by partial-volume fraction, PSF blur, noise."""
    shape = sand_frac.shape
    bg = np.full(shape, GRAY_ICE, np.float32)
    if rng.random() < p_air:
        # low-frequency air pockets (unfrozen voids)
        noise = rng.standard_normal(shape).astype(np.float32)
        noise = ndimage.gaussian_filter(noise, sigma=max(shape) / 12.0)
        thr = np.percentile(noise, rng.uniform(60, 80))
        air = noise > thr
        bg[air] = GRAY_AIR
    gray = bg * (1.0 - sand_frac) + GRAY_SAND * sand_frac
    gray = ndimage.gaussian_filter(gray, sigma=PSF_SIGMA)
    gray = gray + rng.normal(0, NOISE_STD, shape).astype(np.float32)
    return np.clip(gray, 0, 65535).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=24, help='training volumes')
    ap.add_argument('--val', type=int, default=4, help='validation volumes')
    ap.add_argument('--size', type=int, default=160, help='cube edge (voxels)')
    ap.add_argument('--solid', type=float, default=0.55, help='target sand fraction')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--test', action='store_true', help='make 1 small volume to QC')
    a = ap.parse_args()

    eds, els, fls = load_shapes()
    print(f"shape pool: {len(eds)} grains | ED um p5/50/95 = "
          f"{np.percentile(eds,5):.0f}/{np.percentile(eds,50):.0f}/{np.percentile(eds,95):.0f}")

    if a.test:
        a.n, a.val, a.size = 1, 0, 128

    img_dir = os.path.join(OUT_DIR, 'imagesTr')
    lab_dir = os.path.join(OUT_DIR, 'labelsTr')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    shape = (a.size,) * 3
    rng = np.random.default_rng(a.seed)

    rows = []
    total = a.n + a.val
    for k in range(total):
        split = 'val' if k >= a.n else 'train'
        labels, sand, nid = make_volume(shape, eds, els, fls, rng, solid_frac=a.solid)
        gray = render_ct(sand, rng)
        name = f"sand_{k:04d}"
        tifffile.imwrite(os.path.join(img_dir, f"{name}_0000.tif"), gray)
        tifffile.imwrite(os.path.join(lab_dir, f"{name}.tif"), labels)
        sf = (labels > 0).mean()
        rows.append((name, split, nid, f"{sf:.3f}"))
        print(f"  {name} [{split}]: {nid} grains, sand frac {sf:.3f}")

    with open(os.path.join(OUT_DIR, 'manifest.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['name', 'split', 'n_grains', 'sand_fraction'])
        w.writerows(rows)
    print(f"\nDone. {total} volumes -> {OUT_DIR}")
    if a.test:
        print("QC: open imagesTr/sand_0000_0000.tif + labelsTr/sand_0000.tif in a viewer.")


if __name__ == '__main__':
    main()
