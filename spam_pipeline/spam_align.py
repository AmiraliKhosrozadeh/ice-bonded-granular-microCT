"""Align T5_HR scan 1 and scan 2 to a common WORLD-FIXED voxel frame.

Unlike T7 (where Dragonfly had matching XY crops and only Z differed), scan 1
and scan 2 of T5_HR have different X, Y AND Z crops. This script:

  1. Reads each scan's origin from scan_meta.py (world coords of voxel 0).
  2. Computes a common world bounding box that covers BOTH scans.
  3. Pads each volume (CT + specimen mask + bead labels) with zeros into
     that common frame, so scan-1 voxel (i,j,k) and scan-2 voxel (i,j,k)
     correspond to the SAME world point.
  4. Saves <volume>_aligned.tif to the same data folder.

Use the _aligned.tif files as input for every downstream SPAM step
(ddic, ldic, bonds, contacts, failure modes). They all share one voxel grid.

Run in WSL spam-venv:
    python /mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/spam_align.py
"""
import os, sys
import numpy as np
import tifffile

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from scan_meta import SCANS, VOXEL_UM

VOLS_PER_SCAN = ['ct', 'specimen_mask', 'bead_labels']
DTYPES = {'ct': np.uint16, 'specimen_mask': np.uint8, 'bead_labels': np.uint16}

def vol_path(data_dir, kind, scan):
    return os.path.join(data_dir, f'{kind}_scan{scan:02d}.tif')

def aligned_path(data_dir, kind, scan):
    return os.path.join(data_dir, f'{kind}_scan{scan:02d}_aligned.tif')

# ---------- 1. discover shapes ------------------------------------------------
print('Reading source shapes:')
info = {}
for s, meta in SCANS.items():
    ct = vol_path(meta['data_dir'], 'ct', s)
    if not os.path.exists(ct):
        raise FileNotFoundError(ct)
    shape = tifffile.TiffFile(ct).pages[0].shape
    # tifffile reports a single page; for full volume we use imread lazily below
    full_shape = tifffile.imread(ct).shape   # (Z, Y, X)
    info[s] = dict(**meta, shape_zyx=full_shape)
    print(f'  scan {s}: shape ZYX = {full_shape}, origin um = {meta["origin_um"]}')

# ---------- 2. compute common world box --------------------------------------
# XY: use Dragonfly origin_um (it correctly reflects minX/minY crop).
# Z : use FIRST_SLICE * voxel_um to place each scan in the raw-TIFF frame.
#     Dragonfly's reported origin_Z is per-channel, not raw, so it's ignored
#     for Z alignment.
world_mins_xyz = []
world_maxs_xyz = []
for s, i in info.items():
    ox, oy, _ = i['origin_um']
    nz, ny, nx = i['shape_zyx']
    oz = i['first_slice'] * VOXEL_UM      # <-- true raw Z of voxel 0
    i['origin_um_corrected'] = (ox, oy, oz)
    world_mins_xyz.append((ox, oy, oz))
    world_maxs_xyz.append((ox + nx * VOXEL_UM,
                           oy + ny * VOXEL_UM,
                           oz + nz * VOXEL_UM))
wx_min = min(w[0] for w in world_mins_xyz)
wy_min = min(w[1] for w in world_mins_xyz)
wz_min = min(w[2] for w in world_mins_xyz)
wx_max = max(w[0] for w in world_maxs_xyz)
wy_max = max(w[1] for w in world_maxs_xyz)
wz_max = max(w[2] for w in world_maxs_xyz)

# Common frame voxel dimensions (rounded up)
nx_c = int(np.ceil((wx_max - wx_min) / VOXEL_UM))
ny_c = int(np.ceil((wy_max - wy_min) / VOXEL_UM))
nz_c = int(np.ceil((wz_max - wz_min) / VOXEL_UM))
common_origin_um = (wx_min, wy_min, wz_min)

print('\nCommon world-fixed frame:')
print(f'  origin um  = {common_origin_um}')
print(f'  shape ZYX  = ({nz_c}, {ny_c}, {nx_c})')
print(f'  extent um  = ({nx_c*VOXEL_UM:.1f} x {ny_c*VOXEL_UM:.1f} x {nz_c*VOXEL_UM:.1f})')

# ---------- 3. per-scan voxel offset into common frame -----------------------
offsets_zyx = {}
for s, i in info.items():
    ox, oy, oz = i['origin_um_corrected']
    dx = int(round((ox - wx_min) / VOXEL_UM))
    dy = int(round((oy - wy_min) / VOXEL_UM))
    dz = int(round((oz - wz_min) / VOXEL_UM))
    offsets_zyx[s] = (dz, dy, dx)
    print(f'  scan {s}: insert at common voxel (z,y,x) = ({dz}, {dy}, {dx})')

# ---------- 4. pad + save each volume ----------------------------------------
for s, i in info.items():
    data_dir = i['data_dir']
    dz, dy, dx = offsets_zyx[s]
    nz, ny, nx = i['shape_zyx']
    for kind in VOLS_PER_SCAN:
        src = vol_path(data_dir, kind, s)
        if not os.path.exists(src):
            print(f'  scan {s} {kind}: missing ({src}), skipping')
            continue
        print(f'  scan {s} {kind}: loading {src}')
        vol = tifffile.imread(src)
        assert vol.shape == (nz, ny, nx), f'shape mismatch for {src}: {vol.shape} vs {(nz, ny, nx)}'
        aligned = np.zeros((nz_c, ny_c, nx_c), dtype=DTYPES[kind])
        aligned[dz:dz+nz, dy:dy+ny, dx:dx+nx] = vol
        out = aligned_path(data_dir, kind, s)
        tifffile.imwrite(out, aligned, compression='zlib')
        print(f'    wrote {out}  ({aligned.shape}, {DTYPES[kind]})')

# ---------- 5. save a small metadata file ------------------------------------
meta_out = os.path.join(SCANS[1]['data_dir'], 'aligned_meta.py')
with open(meta_out, 'w') as f:
    f.write('"""Common aligned-frame metadata (written by spam_align.py)."""\n')
    f.write(f'VOXEL_UM = {VOXEL_UM}\n')
    f.write(f'COMMON_ORIGIN_UM = {common_origin_um}\n')
    f.write(f'COMMON_SHAPE_ZYX = ({nz_c}, {ny_c}, {nx_c})\n')
    f.write('OFFSETS_ZYX = {\n')
    for s, off in offsets_zyx.items():
        f.write(f'    {s}: {off},\n')
    f.write('}\n')
print(f'\nwrote metadata {meta_out}')
print('\nDone. Use the *_aligned.tif files as input for all downstream SPAM steps.')
