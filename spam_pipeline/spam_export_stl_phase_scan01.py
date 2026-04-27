"""Export a single STL of glass beads from scan 1 only.

Uses bead_labels_scan01.tif (the post-watershed labelled glass beads)
as the source — every voxel with label > 0 is glass.

Coordinates are in physical units (micrometres). Output as a binary STL,
written to <SPAM>/stl_scan01/glass_beads_scan01.stl.

Run in WSL spam-venv:
    python /mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/spam_export_stl_glass_scan01.py
"""
import os
import time
import numpy as np
import tifffile
from skimage import measure

DATA = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
OUT  = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/stl_scan01'
os.makedirs(OUT, exist_ok=True)

VOXEL_UM = 24.7660229
SMOOTH_SIGMA = 0.0    # raise to ~0.6 for slightly smoother bead surfaces
USE_ALIGNED  = False  # False = pre-align bead_labels (tighter); True = aligned

label_file = ('bead_labels_scan01_aligned.tif' if USE_ALIGNED
              else 'bead_labels_scan01.tif')

def write_stl_binary(verts_um, faces, path):
    n_tri = len(faces)
    with open(path, 'wb') as f:
        f.write(b'\x00' * 80)
        f.write(np.uint32(n_tri).tobytes())
        v = verts_um[faces]
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        nn = np.linalg.norm(n, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        n = n / nn
        rec = np.empty(n_tri, dtype=[('n', '<f4', 3), ('v0', '<f4', 3),
                                     ('v1', '<f4', 3), ('v2', '<f4', 3),
                                     ('attr', '<u2')])
        rec['n']  = n.astype(np.float32)
        rec['v0'] = v[:, 0].astype(np.float32)
        rec['v1'] = v[:, 1].astype(np.float32)
        rec['v2'] = v[:, 2].astype(np.float32)
        rec['attr'] = 0
        rec.tofile(f)

t0 = time.time()
print(f'Loading {label_file} ...')
lab_path = os.path.join(DATA, label_file)
lab = tifffile.imread(lab_path)
print(f'  shape {lab.shape}, dtype {lab.dtype}, max label {int(lab.max())}')

mask = (lab > 0).astype(np.float32)
del lab

if SMOOTH_SIGMA > 0:
    print(f'  smoothing (sigma={SMOOTH_SIGMA} vox) ...')
    from scipy.ndimage import gaussian_filter
    mask = gaussian_filter(mask, sigma=SMOOTH_SIGMA)

print('Running marching cubes ...')
verts, faces, _, _ = measure.marching_cubes(
    mask, level=0.5, spacing=(VOXEL_UM,) * 3)
del mask

# scikit-image returns (z, y, x); STL conventionally uses (x, y, z)
verts_xyz = verts[:, [2, 1, 0]]
print(f'  {len(verts_xyz):,} vertices, {len(faces):,} triangles')

stl_path = os.path.join(OUT, 'glass_beads_scan01.stl')
print(f'Writing {stl_path} ...')
write_stl_binary(verts_xyz, faces, stl_path)
size_mb = os.path.getsize(stl_path) / 1e6
print(f'\nDone in {time.time()-t0:.1f}s')
print(f'  {stl_path}  ({size_mb:.1f} MB)')
