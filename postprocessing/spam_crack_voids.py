"""Find cracks as low-density VACANCIES in the ice-skeleton node cloud.

Idea: in healthy ice, every skeleton node has many neighbours within a
short radius. A crack is a place where that density drops -- the node
has few neighbours within the search radius -> the area is a vacancy.

Algorithm (per scan):
  1. Load nodes_scan{NN}.csv (skeleton junctions + endpoints).
  2. Subsample to MAX_NODES (uniform random) so cKDTree stays tractable.
  3. For each surviving node, count neighbours within SEARCH_RADIUS_VOX.
  4. Threshold: nodes whose neighbour count is below the LOWEST_PCT
     percentile -> "vacancy" nodes = crack candidates.
  5. Cluster the vacancy nodes (cKDTree + union-find) -> each cluster
     is one connected crack region.
  6. Save a CSV per scan and render a 3D figure with:
        - all nodes drawn as small, very transparent ("blurry") points,
        - vacancy clusters drawn on top as bright spheres in distinct
          colours so the cracks stand out clearly.

Inputs : results_<PRE>/sparse_graph/nodes_scan{01,02,03}.csv
Outputs: results_<PRE>/sparse_graph/voids/voids_scan{NN}.csv
         results_<PRE>/sparse_graph/voids/voids_3D_scan{NN}.png
"""
import os, sys, csv as _csv, time
import numpy as np
from scipy.spatial import cKDTree
import pyvista as pv
pv.OFF_SCREEN = True

OUT  = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>/sparse_graph'
DEST = os.path.join(OUT, 'voids'); os.makedirs(DEST, exist_ok=True)
VOXEL_UM = 24.7660229
SCANS = [1, 2, 3]

# Subsample so the cKDTree on ~1.5M nodes stays tractable.
MAX_NODES = 80000
# Radius (in voxels) used to count neighbours per node. ~1 bead diameter.
SEARCH_RADIUS_VOX = 70
# Nodes whose neighbour count is in the bottom LOWEST_PCT % become
# vacancy candidates. Tighter = fewer / sharper cracks.
LOWEST_PCT = 5.0
# Cluster vacancy nodes within this distance into one crack body.
CLUSTER_EPS_VOX = 30
MIN_CLUSTER_SIZE = 10   # at least this many vacancy nodes to count

NX_E, NY_E, NZ_E = 803, 707, 1241
PALETTE = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
           '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
           '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3']


def load_nodes(s):
    p = os.path.join(OUT, f'nodes_scan{s:02d}.csv')
    pts = []
    with open(p) as f:
        r = _csv.DictReader(f, delimiter=';')
        for x in r:
            pts.append((float(x['cx']), float(x['cy']), float(x['cz'])))
    return np.asarray(pts, dtype=np.float32)


def union_find(n, pairs):
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in pairs:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj: parent[ri] = rj
    return np.array([find(i) for i in range(n)])


for s in SCANS:
    print(f'\n=== scan {s} ===')
    t0 = time.time()
    pts_all = load_nodes(s)
    print(f'  total nodes: {len(pts_all):,}')
    if len(pts_all) > MAX_NODES:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pts_all), MAX_NODES, replace=False)
        pts = pts_all[idx]
        print(f'  subsampled to {len(pts):,}')
    else:
        pts = pts_all

    tree = cKDTree(pts)
    counts = tree.query_ball_point(pts, r=SEARCH_RADIUS_VOX,
                                   return_length=True)
    counts = np.asarray(counts) - 1   # exclude self
    thresh = np.percentile(counts, LOWEST_PCT)
    is_void = counts <= thresh
    print(f'  neighbour count: median={np.median(counts):.0f}  '
          f'p{LOWEST_PCT:g}={thresh:.0f}')
    print(f'  vacancy candidates: {int(is_void.sum()):,} '
          f'({100 * is_void.mean():.1f}%)')

    void_pts = pts[is_void]
    if len(void_pts) < MIN_CLUSTER_SIZE:
        print('  not enough vacancy nodes; skipping render'); continue

    # Cluster vacancies -> one crack per cluster
    vt = cKDTree(void_pts)
    pairs = vt.query_pairs(CLUSTER_EPS_VOX, output_type='ndarray')
    roots = union_find(len(void_pts), pairs)
    uniq, inv, sz = np.unique(roots, return_inverse=True, return_counts=True)
    cluster_id = np.where(sz[inv] >= MIN_CLUSTER_SIZE, inv, -1)
    n_cracks = int(cluster_id.max() + 1) if (cluster_id >= 0).any() else 0
    print(f'  clusters: {n_cracks} cracks (>={MIN_CLUSTER_SIZE} vacancies)')

    # CSV
    csv_p = os.path.join(DEST, f'voids_scan{s:02d}.csv')
    with open(csv_p, 'w') as f:
        f.write('crack_id;cx;cy;cz;neighbour_count\n')
        for k, (p, c, nb) in enumerate(zip(void_pts, cluster_id,
                                            counts[is_void])):
            if c < 0: continue
            f.write(f'{int(c)+1};{p[0]:.2f};{p[1]:.2f};{p[2]:.2f};{int(nb)}\n')
    print(f'  wrote {csv_p}')

    # 3D render
    pl = pv.Plotter(off_screen=True, window_size=(2200, 2200))
    pl.set_background('white')
    pl.enable_anti_aliasing('msaa', multi_samples=8)

    # Background: all (subsampled) nodes as faint, low-opacity blue dots
    bg_cloud = pv.PolyData(pts.astype(np.float32))
    bg_glyph = bg_cloud.glyph(geom=pv.Sphere(radius=2.5,
                                              theta_resolution=6,
                                              phi_resolution=6),
                              scale=False, orient=False)
    pl.add_mesh(bg_glyph, color='#7fb3ff', opacity=0.07,
                smooth_shading=True)

    # Cracks: each cluster a different colour, large opaque spheres
    for ci in range(n_cracks):
        m = (cluster_id == ci)
        if not m.any(): continue
        col = PALETTE[ci % len(PALETTE)]
        cl = pv.PolyData(void_pts[m].astype(np.float32))
        gly = cl.glyph(geom=pv.Sphere(radius=8, theta_resolution=12,
                                       phi_resolution=12),
                       scale=False, orient=False)
        pl.add_mesh(gly, color=col, smooth_shading=True)

    # Use parallel projection with an explicit view-volume size so we
    # have direct control over the framing — no auto-fit from add_mesh
    # that keeps cropping the top cluster.
    # Angled 3D perspective view (NOT a flat side projection).
    pl.camera_position = [(NX_E * 1.9, -NY_E * 1.30, -NZ_E * 0.60),
                          (NX_E / 2, NY_E / 2, NZ_E / 2),
                          (0, 0, -1)]
    pl.camera.zoom(0.80)
    # Native VTK 3D axes gizmo — proper arrows, never overlap.
    pl.add_axes(line_width=4, labels_off=False, color='#1a1a1a',
                x_color='#e6194B', y_color='#3cb44b', z_color='#4363d8',
                xlabel='X', ylabel='Y', zlabel='Z',
                viewport=(0.18, 0.30, 0.38, 0.50))

    raw_png = os.path.join(DEST, f'_voids_raw_scan{s:02d}.png')
    pl.screenshot(raw_png); pl.close()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _sparse_graph_helpers import composite
    # No legend -- the colored crack clusters speak for themselves
    out_png = os.path.join(DEST, f'voids_3D_scan{s:02d}.png')
    composite(raw_png, [], out_png)
    try: os.remove(raw_png)
    except OSError: pass
    print(f'  wrote {out_png}  ({time.time()-t0:.0f}s total)')

print('\nDone.')
