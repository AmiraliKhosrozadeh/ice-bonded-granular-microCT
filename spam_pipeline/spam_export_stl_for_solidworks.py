"""SolidWorks-friendly STL exports of scan 1 glass beads.

SolidWorks chokes on huge meshes (cap ~20k triangles for solid import,
~500k for graphics body). The full-resolution glass_beads_scan01.stl is
16M triangles — way too big. This script produces two compact variants:

(A) glass_beads_scan01_decimated.stl
        Single merged mesh, smoothed + heavily decimated.
        ~100k triangles, opens as a SW graphics body / Mesh BREP.

(B) beads/bead_NNNN.stl  (one per bead, ~300 tri each)
        Per-bead meshes — assemble or import individually in SolidWorks.

Run in WSL spam-venv:
    python /mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/spam_export_stl_for_solidworks.py
"""
import os, time
import numpy as np
import tifffile
from skimage import measure
from scipy.ndimage import gaussian_filter, find_objects

DATA = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
OUT  = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/stl_scan01'
BEAD_DIR = os.path.join(OUT, 'beads')
os.makedirs(OUT, exist_ok=True)
os.makedirs(BEAD_DIR, exist_ok=True)

VOXEL_UM = 24.7660229
LABEL_FILE = 'bead_labels_scan01.tif'

# Variant (A) merged decimated mesh
EXPORT_MERGED   = True
MERGED_SIGMA    = 0.8       # Gaussian smoothing, voxels
MERGED_TARGET   = 100_000   # target triangle count
MERGED_PATH     = os.path.join(OUT, 'glass_beads_scan01_decimated.stl')

# Variant (B) per-bead meshes
EXPORT_PER_BEAD = True
BEAD_SIGMA      = 0.8
BEAD_TARGET     = 300        # target triangles per bead
BEAD_PAD_VOX    = 3          # padding around each bead's bbox

def write_stl_binary(verts_um, faces, path):
    n_tri = len(faces)
    if n_tri == 0:
        return
    v = verts_um[faces]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    nn = np.linalg.norm(n, axis=1, keepdims=True); nn[nn == 0] = 1.0
    n = n / nn
    rec = np.empty(n_tri, dtype=[('n', '<f4', 3), ('v0', '<f4', 3),
                                 ('v1', '<f4', 3), ('v2', '<f4', 3),
                                 ('attr', '<u2')])
    rec['n']  = n.astype(np.float32)
    rec['v0'] = v[:, 0].astype(np.float32)
    rec['v1'] = v[:, 1].astype(np.float32)
    rec['v2'] = v[:, 2].astype(np.float32)
    rec['attr'] = 0
    with open(path, 'wb') as f:
        f.write(b'\x00' * 80)
        f.write(np.uint32(n_tri).tobytes())
        rec.tofile(f)

def decimate_to(verts_xyz, faces, target_tri):
    """Reduce mesh to roughly target_tri triangles via PyVista decimation,
    then clean: merge duplicate vertices, drop degenerate triangles, fix
    normal orientation. Suppresses SolidWorks 'facet problem' warning."""
    import pyvista as pv
    tri = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces]).ravel()
    mesh = pv.PolyData(verts_xyz.astype(np.float32), tri)
    if len(faces) > target_tri:
        reduction = 1.0 - (target_tri / len(faces))
        mesh = mesh.decimate(reduction)
        mesh = mesh.triangulate()
    # Clean: merge coincident points, drop degenerate (zero-area) tris.
    mesh = mesh.clean(point_merging=True, merge_tol=1e-3,
                      lines_to_points=False, polys_to_lines=False,
                      strips_to_polys=False)
    # Re-orient face normals consistently outward to drop the
    # 'inconsistent normals' warning. auto_orient_normals walks the
    # mesh graph and flips inverted facets.
    mesh.compute_normals(auto_orient_normals=True, consistent_normals=True,
                         inplace=True)
    v = np.asarray(mesh.points, dtype=np.float64)
    f_arr = mesh.faces.reshape(-1, 4)
    f_arr = f_arr[f_arr[:, 0] == 3, 1:]   # keep only triangles
    return v, f_arr

def surface_um(mask, sigma, spacing=(VOXEL_UM,)*3):
    arr = mask.astype(np.float32)
    if sigma > 0:
        arr = gaussian_filter(arr, sigma=sigma)
    if arr.max() <= 0.5:
        return None, None
    verts, faces, _, _ = measure.marching_cubes(arr, level=0.5, spacing=spacing)
    return verts[:, [2, 1, 0]], faces   # -> (x, y, z)

# ---------------- load ------------------------------------------------------
t0 = time.time()
print(f'Loading {LABEL_FILE} ...')
lab = tifffile.imread(os.path.join(DATA, LABEL_FILE))
print(f'  shape {lab.shape}, dtype {lab.dtype}, max label {int(lab.max())}')

# (A) merged decimated mesh
if EXPORT_MERGED:
    print('\n=== (A) merged decimated mesh ===')
    print(f'  smoothing (sigma={MERGED_SIGMA} vox) + marching cubes ...')
    v, f = surface_um((lab > 0).astype(np.float32), MERGED_SIGMA)
    print(f'  full-res: {len(v):,} verts, {len(f):,} tris')
    print(f'  decimating to ~{MERGED_TARGET:,} triangles ...')
    v, f = decimate_to(v, f, MERGED_TARGET)
    print(f'  decimated: {len(v):,} verts, {len(f):,} tris')
    write_stl_binary(v, f, MERGED_PATH)
    print(f'  wrote {MERGED_PATH}  ({os.path.getsize(MERGED_PATH)/1e6:.1f} MB)')

# (B) per-bead meshes
if EXPORT_PER_BEAD:
    print('\n=== (B) per-bead meshes ===')
    slices = find_objects(lab)
    n_max = len(slices)
    print(f'  {n_max} labels, target {BEAD_TARGET} tri / bead')
    count, fail = 0, 0
    t1 = time.time()
    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        z0, z1 = max(0, sl[0].start - BEAD_PAD_VOX), min(lab.shape[0], sl[0].stop + BEAD_PAD_VOX)
        y0, y1 = max(0, sl[1].start - BEAD_PAD_VOX), min(lab.shape[1], sl[1].stop + BEAD_PAD_VOX)
        x0, x1 = max(0, sl[2].start - BEAD_PAD_VOX), min(lab.shape[2], sl[2].stop + BEAD_PAD_VOX)
        sub = (lab[z0:z1, y0:y1, x0:x1] == i)
        if not sub.any():
            continue
        v, f = surface_um(sub, BEAD_SIGMA)
        if v is None:
            fail += 1; continue
        # shift to global µm
        v = v + np.array([x0, y0, z0]) * VOXEL_UM
        if len(f) > BEAD_TARGET:
            v, f = decimate_to(v, f, BEAD_TARGET)
        write_stl_binary(v, f, os.path.join(BEAD_DIR, f'bead_{i:04d}.stl'))
        count += 1
        if count % 50 == 0:
            print(f'    {count}/{n_max} beads exported  ({time.time()-t1:.1f}s)')
    sizes = [os.path.getsize(os.path.join(BEAD_DIR, p))
             for p in os.listdir(BEAD_DIR) if p.endswith('.stl')]
    print(f'  exported {count} beads ({fail} failed) in {time.time()-t1:.1f}s')
    if sizes:
        print(f'  per-bead size: min {min(sizes)/1024:.1f} KB,  '
              f'mean {np.mean(sizes)/1024:.1f} KB,  '
              f'max {max(sizes)/1024:.1f} KB')
        print(f'  total bead-STL volume: {sum(sizes)/1e6:.1f} MB in {BEAD_DIR}')

print(f'\nAll done in {time.time()-t0:.1f}s.')
