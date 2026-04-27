"""Compare ice-phase sparse graphs across scans -> detect crack candidates.

Inputs:
  results_<PRE>/sparse_graph/nodes_scan{01,02,03}.csv
  results_<PRE>/sparse_graph/edges_scan{01,02,03}.csv

For each scan-A -> scan-B transition:
  1. Match scan-A nodes to scan-B nodes by spatial proximity
     (cKDTree, tolerance MATCH_TOL_VOX in the aligned frame).
  2. For each scan-A edge (n1, n2):
        a) if BOTH endpoints map to scan-B nodes AND a scan-B edge
           exists between the mapped pair  -> MATCHED.
           Compute thickness ratio  t_B / t_A.
        b) if neither matched edge exists  -> BROKEN  (crack candidate).
        c) thinned by >= THIN_RATIO       -> WEAKENED (crack precursor).
  3. New scan-B edges that have no scan-A counterpart -> NEW_PATH
     (created by ice rearrangement / crack opening).

Outputs:
  results_<PRE>/sparse_graph/crack_candidates_AtoB.csv
        (one row per BROKEN edge, with both endpoint coords)
  results_<PRE>/sparse_graph/edge_change_AtoB.csv
        (every scan-A edge tagged surviving / weakened / broken)
  results_<PRE>/sparse_graph/sparse_graph_change_3D_AtoB.png
        ball-and-bar render with bars coloured by status
        (green=surviving, orange=weakened, red=broken).

The BROKEN edges are the new method's crack candidates: the ice locally
disconnected between two junctions that were previously connected.
"""
import os, sys
import numpy as np
import csv as _csv
from scipy.spatial import cKDTree
import pyvista as pv
pv.OFF_SCREEN = True

OUT = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>/sparse_graph'
TRANSITIONS = [(1, 2), (2, 3)]

# Spatial matching tolerance (voxels) for node->node correspondence
# across scans. Big enough to absorb axial-compression bead motion.
MATCH_TOL_VOX = 80
# An edge is "weakened" if its thickness drops below this ratio.
THIN_RATIO = 0.50

# Scene size for camera (must match aligned-frame shape)
NX_E, NY_E, NZ_E = 803, 707, 1241

def load_nodes(s):
    rows = []
    with open(os.path.join(OUT, f'nodes_scan{s:02d}.csv')) as f:
        r = _csv.DictReader(f, delimiter=';')
        for x in r:
            rows.append(dict(node_id=int(x['node_id']), kind=x['kind'],
                             cz=float(x['cz']), cy=float(x['cy']),
                             cx=float(x['cx']),
                             thick_um=float(x['thick_um'])))
    return rows

def load_edges(s):
    rows = []
    with open(os.path.join(OUT, f'edges_scan{s:02d}.csv')) as f:
        r = _csv.DictReader(f, delimiter=';')
        for x in r:
            rows.append(dict(edge_id=int(x['edge_id']),
                             a=int(x['node_a']), b=int(x['node_b']),
                             length_um=float(x['length_um']),
                             thick_um=float(x['thick_mean_um'])))
    return rows

for a, b in TRANSITIONS:
    print(f'\n=== transition {a} -> {b} ===')
    nodes_a = load_nodes(a); nodes_b = load_nodes(b)
    edges_a = load_edges(a); edges_b = load_edges(b)
    print(f'  nodes  A={len(nodes_a)}  B={len(nodes_b)}')
    print(f'  edges  A={len(edges_a)}  B={len(edges_b)}')

    # Match A-nodes -> B-nodes by closest centroid (in voxels).
    pts_b = np.array([(n['cz'], n['cy'], n['cx']) for n in nodes_b])
    pts_a = np.array([(n['cz'], n['cy'], n['cx']) for n in nodes_a])
    tree_b = cKDTree(pts_b)
    dist, idx = tree_b.query(pts_a, k=1)
    a_to_b = {}
    for i, (d, j) in enumerate(zip(dist, idx)):
        if d <= MATCH_TOL_VOX:
            a_to_b[nodes_a[i]['node_id']] = nodes_b[j]['node_id']

    # Index B-edges by sorted endpoint pair for quick lookup.
    b_edge_by_pair = {}
    for e in edges_b:
        key = (min(e['a'], e['b']), max(e['a'], e['b']))
        b_edge_by_pair[key] = e

    # Walk through every scan-A edge, classify.
    surviving = weakened = broken = 0
    rows_change = []
    crack_candidates = []
    a_node_by_id = {n['node_id']: n for n in nodes_a}
    for ea in edges_a:
        ma = a_to_b.get(ea['a']); mb = a_to_b.get(ea['b'])
        a_n1 = a_node_by_id[ea['a']]; a_n2 = a_node_by_id[ea['b']]
        if ma is None or mb is None or ma == mb:
            status = 'broken'; broken += 1
            t_ratio = 0.0
            crack_candidates.append((ea['edge_id'],
                                     a_n1['cx'], a_n1['cy'], a_n1['cz'],
                                     a_n2['cx'], a_n2['cy'], a_n2['cz'],
                                     ea['length_um'], ea['thick_um']))
        else:
            key = (min(ma, mb), max(ma, mb))
            eb = b_edge_by_pair.get(key)
            if eb is None:
                status = 'broken'; broken += 1; t_ratio = 0.0
                crack_candidates.append((ea['edge_id'],
                                         a_n1['cx'], a_n1['cy'], a_n1['cz'],
                                         a_n2['cx'], a_n2['cy'], a_n2['cz'],
                                         ea['length_um'], ea['thick_um']))
            else:
                t_ratio = eb['thick_um'] / max(ea['thick_um'], 1e-6)
                if t_ratio < THIN_RATIO:
                    status = 'weakened'; weakened += 1
                else:
                    status = 'surviving'; surviving += 1
        rows_change.append((ea['edge_id'], status,
                            ea['length_um'], ea['thick_um'], t_ratio))

    # New B-edges (no matching A-edge)
    a_edge_pairs_in_b = set()
    for ea in edges_a:
        ma = a_to_b.get(ea['a']); mb = a_to_b.get(ea['b'])
        if ma and mb and ma != mb:
            a_edge_pairs_in_b.add((min(ma, mb), max(ma, mb)))
    new_paths = sum(1 for k in b_edge_by_pair if k not in a_edge_pairs_in_b)

    print(f'  surviving = {surviving}  weakened = {weakened}  '
          f'broken = {broken}  new_paths = {new_paths}')

    # Write change CSV
    out_change = os.path.join(OUT, f'edge_change_{a}to{b}.csv')
    with open(out_change, 'w') as f:
        f.write('edge_id;status;length_um;thick_um_A;thick_ratio_BoverA\n')
        for r in rows_change:
            f.write(f'{r[0]};{r[1]};{r[2]:.2f};{r[3]:.2f};{r[4]:.3f}\n')
    print(f'  wrote {out_change}')

    # Write crack candidates (broken edges)
    out_crack = os.path.join(OUT, f'crack_candidates_{a}to{b}.csv')
    with open(out_crack, 'w') as f:
        f.write('edge_id;n1_cx;n1_cy;n1_cz;n2_cx;n2_cy;n2_cz;length_um;thick_um\n')
        for r in crack_candidates:
            f.write(';'.join(f'{v:.2f}' if isinstance(v, float) else str(v)
                             for v in r) + '\n')
    print(f'  wrote {out_crack}')

    # 3D render. We have ~1.5M skeleton nodes per scan, far too many to
    # glyph as spheres. Render only the BROKEN-edge endpoints (the
    # crack candidates) and a small sample of surviving/weakened nodes
    # for context.
    p = pv.Plotter(off_screen=True, window_size=(2200, 2200))
    p.set_background('white')
    p.enable_anti_aliasing('msaa', multi_samples=8)

    # Endpoint coords for each broken edge -> sphere ball cloud
    crack_endpoints = []
    for ea in edges_a:
        rec = next((r for r in rows_change if r[0] == ea['edge_id']), None)
        if rec and rec[1] == 'broken':
            n1 = a_node_by_id[ea['a']]; n2 = a_node_by_id[ea['b']]
            crack_endpoints.append((n1['cx'], n1['cy'], n1['cz']))
            crack_endpoints.append((n2['cx'], n2['cy'], n2['cz']))
    if crack_endpoints:
        pts = np.asarray(crack_endpoints, dtype=np.float32)
        cloud = pv.PolyData(pts)
        glyph = cloud.glyph(geom=pv.Sphere(radius=10,
                                           theta_resolution=12,
                                           phi_resolution=12),
                            scale=False, orient=False)
        p.add_mesh(glyph, color='#7e7e7e', smooth_shading=True, opacity=0.85)

    # bars per status
    def add_status_bars(status_filter, color, opacity=0.7, radius=4.0):
        pts_l = []; cells = []
        for r in rows_change:
            if r[1] != status_filter: continue
            ea = next(e for e in edges_a if e['edge_id'] == r[0])
            n1 = a_node_by_id[ea['a']]; n2 = a_node_by_id[ea['b']]
            pts_l.append((n1['cx'], n1['cy'], n1['cz']))
            pts_l.append((n2['cx'], n2['cy'], n2['cz']))
            i = len(pts_l) - 2
            cells.append([2, i, i + 1])
        if not pts_l: return
        pd = pv.PolyData()
        pd.points = np.asarray(pts_l, dtype=np.float32)
        pd.lines = np.asarray(cells).ravel().astype(np.int64)
        tubes = pd.tube(radius=radius, n_sides=10)
        p.add_mesh(tubes, color=color, opacity=opacity, show_scalar_bar=False)

    # surviving/weakened set is huge (10s-100s of thousands) - skip for
    # the rendering, they'd OOM. Only render the few hundred BROKEN
    # edges that are the actual crack candidates.
    add_status_bars('broken', '#e6194B', opacity=1.0, radius=8.0)

    p.camera_position = [(NX_E * 2.4, -NY_E * 1.7, -NZ_E * 0.8),
                         (NX_E / 2, NY_E / 2, NZ_E / 2),
                         (0, 0, -1)]
    raw_png = os.path.join(OUT, f'_scene_raw_{a}to{b}.png')
    p.screenshot(raw_png); p.close()

    # Composite legend strip below
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _sparse_graph_helpers import composite
    legend = [('SKELETON NODE',           '#7e7e7e'),
              ('BROKEN EDGE (crack)',     '#e6194B')]
    out_png = os.path.join(OUT, f'sparse_graph_change_3D_{a}to{b}.png')
    composite(raw_png, legend, out_png)
    try: os.remove(raw_png)
    except OSError: pass
    print(f'  wrote {out_png}')

print('\nDone. crack_candidates_*.csv contains the new-method crack list.')
