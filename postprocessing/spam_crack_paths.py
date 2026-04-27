"""Trace crack PATHS from the per-transition broken-edge CSV.

Inputs:
  results_<PRE>/sparse_graph/crack_candidates_AtoB.csv     (broken edges)

Algorithm:
  1) Drop edges that are too short or were already too thin before
     they broke (tunable filters).
  2) Take all unique endpoint coordinates and DBSCAN-cluster them
     spatially  -> each cluster = one connected fracture body.
  3) For each cluster:
        a) collect all broken edges whose endpoints fall in that cluster,
        b) compute PCA on the endpoint cloud -> principal axis,
        c) project each endpoint onto that axis,
        d) sort by projection -> ordered path through the crack.
  4) Save:
        crack_paths_AtoB.csv          one row per path-vertex
        crack_paths_summary_AtoB.txt  per-crack length, # vertices, axis
        crack_paths_3D_AtoB.png       3D render — coloured crack polylines

Run (WSL, spam-venv):
    python /mnt/e/.../<SPAM>/spam_crack_paths.py
"""
import os, sys
import numpy as np
import csv as _csv
from scipy.spatial import cKDTree
import pyvista as pv
pv.OFF_SCREEN = True

OUT = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>/sparse_graph'
DEST = os.path.join(OUT, 'crack_paths'); os.makedirs(DEST, exist_ok=True)

VOXEL_UM = 24.7660229
TRANSITIONS = [(1, 2), (2, 3)]

# Filters to reject skeleton noise: keep only edges that were
# - at least MIN_LENGTH_UM long before breaking, AND
# - at least MIN_THICK_UM thick before breaking
# These cuts kill most of the millions of micro-fragments the skeleton
# produces and keep mechanically meaningful cracks.
MIN_LENGTH_UM = 200.0
MIN_THICK_UM  = 5.0

# DBSCAN cluster tolerance: two endpoints within this distance are
# treated as belonging to the same fracture body.
DBSCAN_EPS_VOX = 25                # ~600 um at 24.77 um/vox
DBSCAN_MIN_PTS = 8                 # minimum endpoints for a real crack

# Camera (matches the rest of the 75 pipeline)
NX_E, NY_E, NZ_E = 803, 707, 1241

# Colour palette for clusters (cycled).
PALETTE = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
           '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
           '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3']


def load_broken_edges(a, b):
    p = os.path.join(OUT, f'crack_candidates_{a}to{b}.csv')
    rows = []
    with open(p) as f:
        r = _csv.DictReader(f, delimiter=';')
        for x in r:
            rows.append(dict(
                edge_id=int(x['edge_id']),
                p1=(float(x['n1_cx']), float(x['n1_cy']), float(x['n1_cz'])),
                p2=(float(x['n2_cx']), float(x['n2_cy']), float(x['n2_cz'])),
                length_um=float(x['length_um']),
                thick_um=float(x['thick_um']),
            ))
    return rows


for a, b in TRANSITIONS:
    print(f'\n=== transition {a} -> {b} ===')
    edges = load_broken_edges(a, b)
    if not edges:
        print('  no broken edges -- skipping'); continue
    n_raw = len(edges)
    # Filter
    edges = [e for e in edges
             if e['length_um'] >= MIN_LENGTH_UM and e['thick_um'] >= MIN_THICK_UM]
    print(f'  edges: {n_raw} raw -> {len(edges)} after length/thick filter')
    if len(edges) < 5:
        print('  too few edges left for path tracing; skipping'); continue

    # All endpoints stacked (each broken edge contributes 2 points)
    pts = np.array([e['p1'] for e in edges] + [e['p2'] for e in edges])
    edge_to_p1 = np.arange(len(edges))
    edge_to_p2 = np.arange(len(edges), 2 * len(edges))

    # Connected-component clustering: merge points within DBSCAN_EPS_VOX
    # using a union-find on cKDTree neighbours. Then drop tiny clusters.
    n = len(pts)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj: parent[ri] = rj
    tree = cKDTree(pts)
    pairs = tree.query_pairs(DBSCAN_EPS_VOX, output_type='ndarray')
    for i, j in pairs:
        union(int(i), int(j))
    roots = np.array([find(i) for i in range(n)])
    uniq, inv, counts = np.unique(roots, return_inverse=True,
                                  return_counts=True)
    labels = np.where(counts[inv] >= DBSCAN_MIN_PTS, inv, -1)
    n_clusters = int(labels.max() + 1) if (labels >= 0).any() else 0
    n_noise = int((labels == -1).sum())
    print(f'  clustered: {n_clusters} cracks, {n_noise} unclustered endpoints')

    # For each cluster: PCA -> bin endpoints along the principal axis to
    # produce a smooth "spine" polyline through the crack body. Length =
    # extent of points along that axis (NOT the summed zig-zag distance).
    SPINE_BINS = 16
    paths = []
    for cid in range(n_clusters):
        m = labels == cid
        cl_pts = pts[m]
        if len(cl_pts) < DBSCAN_MIN_PTS: continue
        c = cl_pts.mean(axis=0)
        _u, _s, vt = np.linalg.svd(cl_pts - c, full_matrices=False)
        axis = vt[0]                          # principal direction
        proj = (cl_pts - c) @ axis            # 1D projection on axis
        # Bin along the axis -> spine vertex = centroid of points in bin
        nbin = min(SPINE_BINS, max(2, len(cl_pts) // 4))
        bins = np.linspace(proj.min(), proj.max(), nbin + 1)
        spine = []
        for k in range(nbin):
            sel = (proj >= bins[k]) & (proj <= bins[k + 1])
            if sel.any():
                spine.append(cl_pts[sel].mean(axis=0))
        spine = np.array(spine)
        # Length along principal axis (extent), in um
        length_um = float((proj.max() - proj.min()) * VOXEL_UM)
        paths.append(dict(cluster_id=int(cid), axis=axis,
                          length_um=length_um, points=spine,
                          n_endpoints=int(len(cl_pts))))

    # Sort cracks by length descending
    paths.sort(key=lambda d: -d['length_um'])
    print(f'  kept {len(paths)} cracks; longest = {paths[0]["length_um"]:.0f} um '
          if paths else '  no cracks')

    # ---- CSVs ----
    csv_pts = os.path.join(DEST, f'crack_paths_{a}to{b}.csv')
    with open(csv_pts, 'w') as f:
        f.write('crack_id;vertex_index;cx;cy;cz\n')
        for ci, pth in enumerate(paths, start=1):
            for vi, (x, y, z) in enumerate(pth['points']):
                f.write(f'{ci};{vi};{x:.2f};{y:.2f};{z:.2f}\n')
    print(f'  wrote {csv_pts}')

    sum_path = os.path.join(DEST, f'crack_paths_summary_{a}to{b}.txt')
    with open(sum_path, 'w') as f:
        f.write(f'Crack-path summary  -  transition {a} -> {b}\n')
        f.write('=' * 56 + '\n')
        f.write(f'broken-edge filter: length >= {MIN_LENGTH_UM:.0f} um '
                f'AND thick >= {MIN_THICK_UM:.0f} um\n')
        f.write(f'DBSCAN: eps={DBSCAN_EPS_VOX} vox, min_samples={DBSCAN_MIN_PTS}\n\n')
        f.write(f"{'crack':>6} {'spine_v':>8} {'endpts':>7} "
                f"{'length_um':>11}    {'axis_x':>7} {'axis_y':>7} {'axis_z':>7}\n")
        for ci, pth in enumerate(paths, start=1):
            ax = pth['axis']
            f.write(f'{ci:>6} {len(pth["points"]):>8} '
                    f'{pth["n_endpoints"]:>7} '
                    f'{pth["length_um"]:>11.1f}    '
                    f'{ax[0]:>+7.3f} {ax[1]:>+7.3f} {ax[2]:>+7.3f}\n')
    print(f'  wrote {sum_path}')

    # ---- 3D render ----
    p = pv.Plotter(off_screen=True, window_size=(2200, 2200))
    p.set_background('white')
    p.enable_anti_aliasing('msaa', multi_samples=8)
    for ci, pth in enumerate(paths, start=1):
        col = PALETTE[(ci - 1) % len(PALETTE)]
        pl = pth['points'].astype(np.float32)
        # tube along the polyline
        pd = pv.PolyData()
        pd.points = pl
        cells = []
        for k in range(len(pl) - 1):
            cells.append([2, k, k + 1])
        pd.lines = np.asarray(cells).ravel().astype(np.int64)
        tube = pd.tube(radius=4.0, n_sides=12)
        p.add_mesh(tube, color=col, smooth_shading=True)
        # endpoints
        endpts = pv.PolyData(pl[[0, -1]])
        glyph = endpts.glyph(geom=pv.Sphere(radius=10, theta_resolution=14,
                                             phi_resolution=14),
                              scale=False, orient=False)
        p.add_mesh(glyph, color=col, smooth_shading=True)

    p.camera_position = [(NX_E * 1.9, -NY_E * 1.30, -NZ_E * 0.60),
                         (NX_E / 2, NY_E / 2, NZ_E / 2),
                         (0, 0, -1)]
    p.camera.zoom(0.95)
    raw_png = os.path.join(DEST, f'_paths_raw_{a}to{b}.png')
    p.screenshot(raw_png); p.close()

    # composite with the right-column legend
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _sparse_graph_helpers import composite
    legend = [(f'CRACK {ci+1}', PALETTE[ci % len(PALETTE)])
              for ci in range(min(len(paths), 8))]
    if len(paths) > 8:
        legend.append((f'+{len(paths) - 8} MORE', '#888888'))
    out_png = os.path.join(DEST, f'crack_paths_3D_{a}to{b}.png')
    composite(raw_png, legend, out_png)
    try: os.remove(raw_png)
    except OSError: pass
    print(f'  wrote {out_png}')

print('\nDone. crack paths in', DEST)
