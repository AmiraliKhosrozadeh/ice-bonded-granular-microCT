"""Build a sparse-graph view of the crack/damage network — balls & bars.

For each transition, take the existing damage_3d_*.tif (one per class) and:
  1) connected-component label every blob,
  2) compute centroid + voxel volume per blob,
  3) draw each blob as a sphere whose radius scales with its volume,
  4) connect blobs whose centroids are within EDGE_RADIUS_UM of each other
     (within the SAME class) with a tube — these are the "bars",
  5) colour by class (crack red, ice-fracture blue, cavity grey).

Output: results_<PRE>/transition_AtoB/crack_sparse_graph_3D.png
        results_<PRE>/transition_AtoB/crack_sparse_graph_nodes.csv
"""
import os
import numpy as np
import tifffile
from scipy import ndimage as ndi
import pyvista as pv

pv.OFF_SCREEN = True

OUT = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>'
VOXEL_UM = 24.7660229
TRANSITIONS = [(1, 2), (2, 3)]

# Edge if two blob centroids are within this distance, in voxels.
# Use ~1.5x bead diameter (2.5 mm) so the network only links genuinely
# nearby cracks; bigger values turn the graph into a hairball.
EDGE_RADIUS_VOX = 2500 / VOXEL_UM    # ~101 vox

# Sphere radius in voxels: wide log10 spread so small vs large damage
# blobs are visually distinguishable, with a hard cap so one giant
# component does not eclipse everything else.
#   100 vox  -> r ≈ 12     (smallest visible blob)
#   1k vox   -> r ≈ 22
#   10k vox  -> r ≈ 32
#  100k+ vox -> r = 50     (cap)
def vox_to_radius(vol_vox):
    r = 2.0 + 10.0 * np.log10(max(vol_vox, 1))
    return float(np.clip(r, 10.0, 50.0))

# Vibrant ball palette. The legend label is the human-readable name we
# want to print under the figure (matches the crack_summary.txt taxonomy
# but written in plain English so a reader without context understands
# what each colour represents).
#   columns: (display_label_for_legend, file_basename_in_transition_dir,
#             hex_color_for_balls, internal_class_name_for_csv)
CLASS_FILES = [
    ('PLANAR CRACK',     'damage_3d_crack.tif',         '#ff1744', 'crack'),
    ('ICE FRACTURE',     'damage_3d_ice_fracture.tif',  '#00b8d4', 'ice_fracture'),
    ('COMPACTION VOID',  'damage_3d_cavity.tif',        '#ffab00', 'cavity'),
]
# Bars: use a soft red-blue tone, translucent so they look "blurry"
# behind the brighter balls. red-tinted for crack-class adjacency,
# blue-tinted for everything else.
BAR_COLORS = {
    'crack':        '#c64a55',   # soft red
    'ice_fracture': '#3b6fc4',   # soft blue
    'cavity':       '#7e7393',   # muted lavender (for mixed groups)
}
BAR_OPACITY = 0.45

# Common-frame size for the camera (T5_HR-style frame extends per scan)
def grow_camera(p, vol_shape):
    nz, ny, nx = vol_shape
    p.camera_position = [
        (nx * 2.1, -ny * 1.5, -nz * 0.7),
        (nx / 2,    ny / 2,    nz / 2),
        (0, 0, -1),
    ]

for a, b in TRANSITIONS:
    tdir = os.path.join(OUT, 'crack_analysis', f'transition_{a}to{b}')
    if not os.path.isdir(tdir):
        print(f'skip {a}->{b}: no crack folder'); continue
    print(f'\n=== transition {a} -> {b} ===')

    p = pv.Plotter(off_screen=True, window_size=(2400, 1800))
    p.set_background('white')
    p.enable_anti_aliasing('msaa', multi_samples=8)

    nodes_csv = []
    vol_shape = None

    for legend_label, fname, hex_col, cls_name in CLASS_FILES:
        path = os.path.join(tdir, fname)
        if not os.path.exists(path):
            print(f'  no {fname}, skipping')
            continue
        mask = tifffile.imread(path) > 0
        if vol_shape is None:
            vol_shape = mask.shape

        # Connected components -> one node per component
        lab, n = ndi.label(mask)
        if n == 0:
            print(f'  {cls_name}: 0 components')
            continue
        sizes = ndi.sum(mask, lab, range(1, n + 1)).astype(np.int64)
        # drop tiny noise (< 100 vox)
        keep_idx = np.where(sizes >= 100)[0] + 1
        if keep_idx.size == 0:
            print(f'  {cls_name}: {n} components, none above noise floor'); continue
        cents = ndi.center_of_mass(mask, lab, list(keep_idx))
        cents_zyx = np.array(cents)                    # (n_keep, 3)  (z, y, x)
        cents_xyz = cents_zyx[:, [2, 1, 0]]            # (x, y, z) for pyvista
        sizes_keep = sizes[keep_idx - 1]
        radii = np.array([vox_to_radius(v) for v in sizes_keep])

        print(f'  {cls_name}: {len(keep_idx)} significant blobs')

        # Add a per-blob sphere glyph cloud
        cloud = pv.PolyData(cents_xyz.astype(np.float32))
        cloud['radius'] = radii.astype(np.float32)
        glyph = cloud.glyph(geom=pv.Sphere(theta_resolution=16, phi_resolution=16),
                            scale='radius', orient=False, factor=1.0)
        p.add_mesh(glyph, color=hex_col, smooth_shading=True,
                   label=f'{cls_name} ({len(keep_idx)})')

        # Edges: connect centroids within EDGE_RADIUS_VOX
        if len(cents_xyz) > 1:
            from scipy.spatial import cKDTree
            tree = cKDTree(cents_xyz)
            edges = []
            for i, c in enumerate(cents_xyz):
                neigh = tree.query_ball_point(c, EDGE_RADIUS_VOX)
                for j in neigh:
                    if j > i:
                        edges.append((i, j))
            if edges:
                pts = []; cells = []
                for (i, j) in edges:
                    pts.append(cents_xyz[i]); pts.append(cents_xyz[j])
                    idx = len(pts) - 2
                    cells.append([2, idx, idx + 1])
                line_pd = pv.PolyData()
                line_pd.points = np.asarray(pts, dtype=np.float32)
                line_pd.lines = np.asarray(cells).ravel().astype(np.int64)
                tubes = line_pd.tube(radius=6.0, n_sides=12)
                p.add_mesh(tubes,
                           color=BAR_COLORS.get(cls_name, '#3b6fc4'),
                           opacity=BAR_OPACITY,
                           show_scalar_bar=False)

        for i, (cz, cy, cx) in enumerate(cents_zyx):
            nodes_csv.append((cls_name, sizes_keep[i], cx, cy, cz,
                              radii[i]))

    if vol_shape is not None:
        grow_camera(p, vol_shape)

    # Render JUST the 3D scene (no legend, no axes widget). The
    # compositor then adds a horizontal legend row + a clean XYZ
    # indicator on top.
    raw_png = os.path.join(tdir, '_crack_scene_raw.png')
    p.screenshot(raw_png); p.close()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _sparse_graph_helpers import composite
    legend_items = [(c[0], c[2]) for c in CLASS_FILES]
    out_png = os.path.join(tdir, 'crack_sparse_graph_3D.png')
    composite(raw_png, legend_items, out_png)
    try: os.remove(raw_png)
    except OSError: pass
    print(f'  wrote {out_png}')

    out_csv = os.path.join(tdir, 'crack_sparse_graph_nodes.csv')
    with open(out_csv, 'w') as f:
        f.write('class;volume_vox;cx;cy;cz;render_radius_vox\n')
        for r in nodes_csv:
            f.write(f'{r[0]};{r[1]};{r[2]:.1f};{r[3]:.1f};{r[4]:.1f};{r[5]:.1f}\n')
    print(f'  wrote {out_csv}')

print('\nDone.')
