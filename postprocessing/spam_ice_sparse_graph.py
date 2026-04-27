"""Skeleton-based sparse graph of the ICE phase, per scan.

For each scan N (1, 2, 3):
  1. Load the ice mask. Prefer ice_mask_scanN.tif (exported from Dragonfly
     via dragonfly_export_ice_roi.py); if not present, derive it from
     ct_scanN_aligned.tif using the per-scan ice gray-value range.
  2. 3D skeletonize the ice mask.
  3. Find skeleton voxels with > 2 skeleton neighbours -> NODES (junctions),
     and voxels with exactly 1 neighbour -> END NODES.
  4. Walk between adjacent nodes along the skeleton -> EDGES.
     For each edge measure:
        length      total skeleton length in voxels
        thick_mean  mean local ice thickness along the edge,
                    computed via the EDT of the ice mask
  5. Write nodes_scanNN.csv and edges_scanNN.csv to results_<PRE>/sparse_graph/.

These two CSVs ARE the sparse-graph data — same kind of structure that
Dragonfly's "Create Sparse Graph" produces in its 3D Modelling panel,
but built from your existing data without OOM-ing.

Run (WSL, spam-venv):
    python /mnt/e/.../<SPAM>/spam_ice_sparse_graph.py
"""
import os, sys, gc, time
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

DATA = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
OUT  = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>/sparse_graph'
os.makedirs(OUT, exist_ok=True)

VOXEL_UM = 24.7660229
SCANS = [1, 2, 3]

# Per-scan ice gray-value range (matches scanNN_segmentation.py).
TH = {
    1: dict(air=7224, ig=13010),   # T_AIR_ICE, T_ICE_GLASS  (placeholders;
    2: dict(air=3802, ig=8442),    #  matches the 100_T5_HR scan 1; fix per
    3: dict(air=3802, ig=8442),    #  the actual histogram for 75_T5_HR)
}

# --- load / build ice mask --------------------------------------------------
def load_ice_mask(s):
    p = os.path.join(DATA, f'ice_mask_scan{s:02d}.tif')
    if os.path.exists(p):
        print(f'  using {p}')
        return tifffile.imread(p) > 0
    p2 = os.path.join(DATA, f'ice_mask_scan{s:02d}_aligned.tif')
    if os.path.exists(p2):
        print(f'  using {p2}')
        return tifffile.imread(p2) > 0
    # Fallback: derive from CT + thresholds.
    print(f'  no ice_mask_scan{s:02d}.tif - deriving from CT')
    ct_path = os.path.join(DATA, f'ct_scan{s:02d}_aligned.tif')
    if not os.path.exists(ct_path):
        raise FileNotFoundError(ct_path)
    ct = tifffile.imread(ct_path)
    th = TH[s]
    ice = (ct >= th['air']) & (ct < th['ig'])
    del ct
    return ice

# --- skeleton + graph -------------------------------------------------------
NEIGH = np.array([(dz, dy, dx)
                  for dz in (-1, 0, 1)
                  for dy in (-1, 0, 1)
                  for dx in (-1, 0, 1)
                  if not (dz == dy == dx == 0)])  # 26-connectivity

def neighbour_count(skel):
    """Per-voxel count of skeleton neighbours (6-conn) for skeleton voxels.
    6-conn produces far fewer false junctions than 26-conn (a 1-vox-wide
    straight skeleton segment has 2 face-adjacent neighbours; only real
    forks have 3+)."""
    k = np.zeros((3, 3, 3), dtype=np.uint8)
    k[0, 1, 1] = k[2, 1, 1] = 1
    k[1, 0, 1] = k[1, 2, 1] = 1
    k[1, 1, 0] = k[1, 1, 2] = 1
    n = ndi.convolve(skel.astype(np.uint8), k, mode='constant', cval=0)
    return n * skel    # zero outside skeleton

def walk_edges(skel, node_mask):
    """For each pair of skeleton voxels that connect through non-node
    skeleton voxels, return the chain of voxels between them (inclusive)
    and its length in voxels.
    Algorithm: remove node voxels, label remaining skeleton CCs (each CC
    = one segment between two nodes); for every CC find its two
    endpoints in the original skeleton (the node voxels touching it).
    """
    interior = skel & ~node_mask
    cc, nlab = ndi.label(interior, structure=np.ones((3, 3, 3)))
    if nlab == 0:
        return []
    # For every interior voxel find which node voxels are adjacent
    # (within 1 step). Build per-CC list of touching node coordinates.
    z, y, x = np.where(interior)
    cc_id = cc[z, y, x]
    node_cc_endpoints = {}   # cc_id -> set of node voxel coords
    Z, Y, X = skel.shape
    for k in range(len(z)):
        ci = cc_id[k]
        if ci not in node_cc_endpoints:
            node_cc_endpoints[ci] = []
        # neighbours
        for dz, dy, dx in NEIGH:
            nz, ny, nx = z[k] + dz, y[k] + dy, x[k] + dx
            if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X:
                if node_mask[nz, ny, nx]:
                    node_cc_endpoints[ci].append((nz, ny, nx))
    # For each CC, pick exactly 2 endpoints (most-connected if many,
    # the only one twice if only one node-contact).
    edges = []
    cc_sizes = ndi.sum(interior, cc, range(1, nlab + 1)).astype(np.int64)
    for ci, ends in node_cc_endpoints.items():
        if not ends:
            continue
        uniq = list({e for e in ends})
        if len(uniq) == 1:
            ep1 = ep2 = uniq[0]
        else:
            ep1, ep2 = uniq[0], uniq[1]   # arbitrary choice
        length_vox = int(cc_sizes[ci - 1]) + 1   # +1 for the linking node
        edges.append((ep1, ep2, length_vox, ci))
    return edges

# --- per scan ---------------------------------------------------------------
for s in SCANS:
    print(f'\n=== scan {s} ===')
    t0 = time.time()
    ice = load_ice_mask(s)
    print(f'  ice voxels = {int(ice.sum()):,}  shape = {ice.shape}')

    # EDT for ice thickness
    print('  computing distance transform...')
    edt = ndi.distance_transform_edt(ice).astype(np.float32)

    # 3D skeleton
    print('  skeletonizing...')
    skel = skeletonize(ice).astype(bool)
    print(f'  skeleton voxels = {int(skel.sum()):,}')

    # Junctions (>2 nbr) and endpoints (1 nbr)
    nb = neighbour_count(skel)
    junction = (nb >= 3) & skel
    endpoint = (nb == 1) & skel
    nodes_mask = junction | endpoint
    n_junc = int(junction.sum()); n_end = int(endpoint.sum())
    print(f'  junctions = {n_junc:,}  endpoints = {n_end:,}')

    # Build node table
    nz, ny, nx = np.where(nodes_mask)
    node_thick = edt[nz, ny, nx]
    node_kind = np.where(junction[nz, ny, nx], 'junction', 'endpoint')
    node_id = np.arange(len(nz), dtype=np.int64) + 1
    node_lookup = {(int(z), int(y), int(x)): int(i)
                   for z, y, x, i in zip(nz, ny, nx, node_id)}
    nodes_csv = os.path.join(OUT, f'nodes_scan{s:02d}.csv')
    with open(nodes_csv, 'w') as f:
        f.write('node_id;kind;cz;cy;cx;thick_um\n')
        for i, k, cz, cy, cx, t in zip(node_id, node_kind, nz, ny, nx, node_thick):
            f.write(f'{i};{k};{cz};{cy};{cx};{t * VOXEL_UM:.2f}\n')
    print(f'  wrote {nodes_csv}')

    # Build edges by walking skeleton segments between nodes
    print('  walking edges...')
    edges_raw = walk_edges(skel, nodes_mask)
    edges_csv = os.path.join(OUT, f'edges_scan{s:02d}.csv')
    with open(edges_csv, 'w') as f:
        f.write('edge_id;node_a;node_b;length_um;thick_mean_um;cc_id\n')
        ed_id = 0
        for ep1, ep2, length_vox, ci in edges_raw:
            id_a = node_lookup.get(ep1)
            id_b = node_lookup.get(ep2)
            if id_a is None or id_b is None:
                continue
            ed_id += 1
            # thickness: mean EDT along the segment + the two endpoints
            ed_mask = (cc[skel] == ci) if False else None
            # cheaper: use EDT at endpoints + segment voxels
            # average the two endpoint EDTs as a proxy
            t_mean = 0.5 * (edt[ep1] + edt[ep2])
            f.write(f'{ed_id};{id_a};{id_b};'
                    f'{length_vox * VOXEL_UM:.2f};{t_mean * VOXEL_UM:.2f};{ci}\n')
    print(f'  wrote {edges_csv}  ({ed_id} edges, {time.time()-t0:.0f}s total)')

    # Aggressive cleanup so scan N+1 does not OOM on top of scan N.
    del ice, edt, skel, nb, junction, endpoint, nodes_mask
    del nz, ny, nx, node_thick, node_kind, node_id, node_lookup
    del edges_raw
    gc.collect()

print('\nDone.')
