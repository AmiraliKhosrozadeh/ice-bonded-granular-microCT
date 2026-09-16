"""Ice-bonded synthetic v2: adds the grain-contact grayscale cue.

The v1 generator renders touching grains as UNIFORM sand, so a grain-grain
contact has no intensity cue -- the net must split purely on geometry, which
fails for low-concavity fused grains (the residual coarse-tail clumps).  Real
frozen sand has thin ICE FILMS / necks between many grains: a gray dip toward
the ice level right at the contact.  v2 injects that dip at a random fraction
of contact voxels, plus per-grain sand-gray variation, mild intra-grain
texture, and per-volume domain randomisation -- giving the net the cue it needs
to place a separating border at ice-bridged contacts.

Labels are UNCHANGED (full grains) so the border-core target still marks every
contact; only the INPUT grayscale gains the neck cue.

Writes new volumes into sand_atlas/synth/{imagesTr,labelsTr,bordercoreTr},
indexed after the existing set, and appends manifest.csv.

Run (Windows Python): python generate_synthetic_v2.py --n 24 --val 4
"""
import os, csv, argparse, numpy as np, tifffile
from scipy import ndimage
from generate_synthetic import load_shapes, make_volume, OUT_DIR
from encode_border_core import encode

VOXEL_UM = 24.7660229


def contact_mask(labels):
    """voxels on a grain-grain boundary (neighbour has a different positive label)."""
    m = np.zeros(labels.shape, bool)
    for ax in range(3):
        for sh in (1, -1):
            nb = np.roll(labels, sh, axis=ax)
            m |= (labels > 0) & (nb > 0) & (labels != nb)
    return m


def render_v2(labels, rng):
    shape = labels.shape
    fg = labels > 0
    nid = int(labels.max())

    # --- per-volume domain randomisation ---
    g_sand = rng.uniform(4600, 5800)
    g_ice = rng.uniform(2900, 3350)
    g_air = rng.uniform(1700, 2100)
    noise = rng.uniform(350, 520)
    psf = rng.uniform(0.6, 1.0)
    neck_frac = rng.uniform(0.30, 0.80)     # fraction of contact voxels with a dip
    gray_var = rng.uniform(0.06, 0.18)      # per-grain sand-gray spread
    tex_amp = rng.uniform(0.0, 0.08)        # intra-grain texture amplitude

    # --- per-grain sand gray + intra-grain texture ---
    gfac = rng.uniform(1 - gray_var, 1 + gray_var, nid + 1).astype(np.float32)
    gfac[0] = 1.0
    sand_gray = g_sand * gfac[labels]
    if tex_amp > 0:
        tex = ndimage.gaussian_filter(rng.standard_normal(shape).astype(np.float32), 2.0)
        tex /= (tex.std() + 1e-6)
        sand_gray *= (1.0 + tex_amp * tex)

    # --- sand partial-volume fraction, with ice necks at contacts ---
    sf = fg.astype(np.float32)
    cm = contact_mask(labels)
    pick = cm & (rng.random(shape) < neck_frac)
    dip = rng.uniform(0.10, 0.65, shape).astype(np.float32)   # how far toward ice
    sf[pick] *= dip[pick]

    # --- background: ice + optional low-frequency air pockets ---
    bg = np.full(shape, g_ice, np.float32)
    if rng.random() < 0.5:
        nz = ndimage.gaussian_filter(rng.standard_normal(shape).astype(np.float32),
                                     sigma=max(shape) / 12.0)
        bg[nz > np.percentile(nz, rng.uniform(60, 80))] = g_air

    gray = bg * (1.0 - sf) + sand_gray * sf
    gray = ndimage.gaussian_filter(gray, sigma=psf)
    gray = gray + rng.normal(0, noise, shape).astype(np.float32)
    return np.clip(gray, 0, 65535).astype(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--val', type=int, default=4)
    ap.add_argument('--size', type=int, default=160)
    ap.add_argument('--solid', type=float, default=0.55)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--r', type=int, default=1, help='border-core erosion radius')
    a = ap.parse_args()

    img_dir = os.path.join(OUT_DIR, 'imagesTr')
    lab_dir = os.path.join(OUT_DIR, 'labelsTr')
    bc_dir = os.path.join(OUT_DIR, 'bordercoreTr')
    for d in (img_dir, lab_dir, bc_dir):
        os.makedirs(d, exist_ok=True)

    # start index after the existing set
    existing = [f for f in os.listdir(img_dir) if f.endswith('_0000.tif')]
    start = len(existing)
    print(f"existing {start} volumes; appending {a.n + a.val} v2 volumes from idx {start}")

    eds, els, fls = load_shapes()
    shape = (a.size,) * 3
    rng = np.random.default_rng(a.seed)

    rows = []
    for k in range(a.n + a.val):
        idx = start + k
        split = 'val' if k >= a.n else 'train'
        labels, sand, nid = make_volume(shape, eds, els, fls, rng, solid_frac=a.solid)
        gray = render_v2(labels, rng)
        bc = encode(labels, a.r)
        name = f"sand_{idx:04d}"
        tifffile.imwrite(os.path.join(img_dir, f"{name}_0000.tif"), gray)
        tifffile.imwrite(os.path.join(lab_dir, f"{name}.tif"), labels)
        tifffile.imwrite(os.path.join(bc_dir, f"{name}.tif"), bc)
        sf = (labels > 0).mean()
        rows.append((name, split, nid, f"{sf:.3f}"))
        print(f"  {name} [{split}]: {nid} grains, sand {sf:.3f}", flush=True)

    man = os.path.join(OUT_DIR, 'manifest.csv')
    newfile = not os.path.exists(man)
    with open(man, 'a', newline='') as fh:
        w = csv.writer(fh)
        if newfile:
            w.writerow(['name', 'split', 'n_grains', 'sand_fraction'])
        w.writerows(rows)
    print(f"done -> appended {len(rows)} rows to {man}")


if __name__ == '__main__':
    main()
