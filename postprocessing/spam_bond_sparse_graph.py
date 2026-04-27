"""Per-scan ball-and-bar of the ICE-BOND network — tells the
initial-state -> final-state story.

For each scan N (1, 2, 3):
  ball = each bead's centroid (size scaled by bead volume, capped)
  bar  = an ice bond between two beads (from ice_bonds_scanNN.csv)
         coloured by remaining ice volume (thicker / brighter = stronger)

So scan 1 = INTACT initial network, scan 3 = FINAL deformed/broken network.

Outputs:
  results_<PRE>/bond_sparse_graph/bond_graph_scan{01,02,03}.png
"""
import os
import numpy as np
import tifffile
from scipy import ndimage as ndi
import pyvista as pv
pv.OFF_SCREEN = True

DATA = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
OUT  = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>'
DEST = os.path.join(OUT, 'bond_sparse_graph'); os.makedirs(DEST, exist_ok=True)

VOXEL_UM = 24.7660229
SCANS = [1, 2, 3]

def centroids(lab):
    n = int(lab.max())
    if n == 0:
        return np.zeros((1, 3))
    c = ndi.center_of_mass(lab > 0, lab, range(1, n + 1))
    arr = np.zeros((n + 1, 3))
    for i, v in enumerate(c, 1): arr[i] = v
    return arr

def read_bonds(csv_path):
    """Return list of (la, lb, ice_vox) for each bond."""
    bonds = []
    with open(csv_path) as f:
        next(f)
        for ln in f:
            p = ln.strip().split(';')
            try:
                la = int(p[0]); lb = int(p[1]); v = int(p[2])
            except (ValueError, IndexError):
                continue
            bonds.append((la, lb, v))
    return bonds

# Camera — common aligned frame for the 75 specimen.
NX_E, NY_E, NZ_E = 803, 707, 1241

LEGEND_ITEMS = [
    ('GLASS BEAD',     '#7e7e7e'),
    ('THICK ICE BOND', '#d62728'),
    ('THIN ICE BOND',  '#3b4cc0'),
]

for s in SCANS:
    lab_path = os.path.join(DATA, f'bead_labels_scan{s:02d}_aligned.tif')
    bond_csv = os.path.join(OUT, f'ice_bonds_scan{s:02d}.csv')
    if not os.path.exists(lab_path) or not os.path.exists(bond_csv):
        print(f'skip scan {s}: missing label TIF or bond CSV'); continue

    print(f'\n=== scan {s} ===')
    lab = tifffile.imread(lab_path)
    cent = centroids(lab)              # (N+1, 3) zyx
    sizes = np.bincount(lab.ravel())   # per-label voxel count
    bonds = read_bonds(bond_csv)
    print(f'  beads = {int(lab.max())}, bonds = {len(bonds)}')
    del lab

    # Ball cloud: one sphere per bead. Scale by cube-root volume,
    # capped so big beads don't blow up the picture.
    pts = []
    radii = []
    for L in range(1, len(cent)):
        c = cent[L]
        if c[0] == 0 and c[1] == 0 and c[2] == 0:    # missing centroid
            continue
        v = sizes[L] if L < len(sizes) else 0
        if v < 100:
            continue
        # Bead radius scaled by cube-root volume (proportional to actual
        # bead radius), with a wide visible range so size differences
        # between beads are obvious.
        r = float(np.clip(0.55 * (v ** (1/3)), 10.0, 45.0))
        pts.append(c[::-1])  # (z,y,x) -> (x,y,z)
        radii.append(r)
    if not pts:
        print(f'  no usable beads'); continue
    pts_arr = np.asarray(pts, dtype=np.float32)
    cloud = pv.PolyData(pts_arr); cloud['r'] = np.asarray(radii, np.float32)
    glyph = cloud.glyph(geom=pv.Sphere(theta_resolution=20, phi_resolution=20),
                        scale='r', orient=False, factor=1.0)

    p = pv.Plotter(off_screen=True, window_size=(2200, 2200))
    p.set_background('white')
    p.enable_anti_aliasing('msaa', multi_samples=8)
    p.add_mesh(glyph, color='#7e7e7e', smooth_shading=True, opacity=0.78)

    # Bond bars: tube between centroid of la and lb, colour by ice volume.
    if bonds:
        ice_max = max(b[2] for b in bonds)
        # pre-allocate poly-line points
        pts_b = []; cells = []; cmap_vals = []
        for la, lb, v in bonds:
            if la >= len(cent) or lb >= len(cent):
                continue
            ca = cent[la]; cb = cent[lb]
            if (ca[0] == 0 and ca[1] == 0 and ca[2] == 0) or \
               (cb[0] == 0 and cb[1] == 0 and cb[2] == 0):
                continue
            pts_b.append(ca[::-1]); pts_b.append(cb[::-1])
            idx = len(pts_b) - 2
            cells.append([2, idx, idx + 1])
            cmap_vals.append(v)
        if pts_b:
            line_pd = pv.PolyData()
            line_pd.points = np.asarray(pts_b, dtype=np.float32)
            line_pd.lines  = np.asarray(cells).ravel().astype(np.int64)
            line_pd['ice'] = np.repeat(cmap_vals, 2).astype(np.float32)
            tubes = line_pd.tube(radius=5.5, n_sides=12)
            # Red-vs-blue gradient (strong = red, weak = blue), softened
            # opacity for the "blurry" look the user asked for.
            p.add_mesh(tubes, scalars='ice', cmap='coolwarm',
                       clim=(0, ice_max),
                       smooth_shading=True, opacity=0.65,
                       show_scalar_bar=False)

    p.camera_position = [(NX_E * 1.9, -NY_E * 1.30, -NZ_E * 0.60),
                         (NX_E / 2, NY_E / 2, NZ_E / 2),
                         (0, 0, -1)]
    p.camera.zoom(0.95)
    raw_png = os.path.join(DEST, f'_bond_scene_raw_scan{s:02d}.png')
    p.screenshot(raw_png); p.close()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _sparse_graph_helpers import composite
    out_png = os.path.join(DEST, f'bond_graph_scan{s:02d}.png')
    composite(raw_png, LEGEND_ITEMS, out_png)
    try: os.remove(raw_png)
    except OSError: pass
    print(f'  wrote {out_png}')

print('\nDone.')
