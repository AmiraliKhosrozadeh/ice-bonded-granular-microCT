"""Ice-bond detection and breakage analysis for the T7 1700 project.

Per scan:
  * Reconstruct a 3-label phase array (air=0, ice=1, glass=2) from the
    aligned CT volume + this scan's manual thresholds.
  * For every pair of beads whose bounding boxes overlap after D voxels of
    dilation, count ICE voxels in the overlap shell -> this is the
    ice-bond strength for that pair.
  * Write bonds_scanNN.csv: (labelA, labelB, ice_vox_count, centres).

Per transition (A -> B):
  * Build a label-map scan-A -> scan-B from DDIC displacements (nearest
    centroid in aligned frame).
  * Compare scan-A bonds to scan-B bonds in the mapped frame:
      - surviving ice bonds (ice in both scans)
      - WEAKENED bonds   (ice voxel count dropped by >= 50 %)
      - BROKEN bonds     (ice gone in scan B)
      - NEW bonds        (ice bond formed in B)
  * Plot: histogram of ice-voxel count per bond (before/after), count of
    broken/weakened/surviving bonds, 3D render of the bond network with
    broken ones in red.

Run in WSL spam-venv:
    python /mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/spam_ice_bonds.py
"""
import os, time
import numpy as np
import tifffile
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyvista as pv

pv.OFF_SCREEN = True

DATA = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
RES  = '/home/cak7496/spam-results-75'
OUT  = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>'
os.makedirs(OUT, exist_ok=True)

# Per-scan phase thresholds (from each scanNN_segmentation.py). Scan 2 lacks
# a distinct aluminum peak so T_GLASS_AL is pushed above any observed gray.
TH = {
    1: dict(air=7000,    ig=18032.7, ga=47275),     # from scan01_segmentation.py
    2: dict(air=2139.4,  ig=6271.5,  ga=99999),     # from scan02_segmentation.py (no Al peak)
    3: dict(air=2015.4,  ig=6539.5,  ga=99999),     # from scan03_segmentation.py (no Al peak)
}

DILATE_VOX    = 3   # per-bead dilation to find inter-bead contact shells
MIN_ICE_VOX   = 10  # minimum ice voxels to call a bead pair "ice-bonded"
CENTROID_TOL  = 150 # voxels — loose tol, heavy axial compression

TRANSITIONS = [(1, 2), (2, 3)]
VOXEL_SIZE_UM = 24.7660229

# ---------------- helpers --------------------------------------------------
def reconstruct_seg(ct, th):
    """Return a uint8 phase array: 0=air, 1=ice, 2=glass, 3=aluminum."""
    seg = np.zeros(ct.shape, dtype=np.uint8)
    seg[(ct >= th['air']) & (ct < th['ig'])]  = 1
    seg[(ct >= th['ig'])  & (ct < th['ga'])] = 2
    seg[ct >= th['ga']]                      = 3
    return seg

def centroids_from_labels(lab):
    """Per-label centroid (z, y, x) in voxel coords. Returns (max+1, 3)."""
    max_lab = int(lab.max())
    if max_lab == 0: return np.zeros((1, 3))
    centroids = ndimage.center_of_mass(lab > 0, lab, range(1, max_lab + 1))
    out = np.zeros((max_lab + 1, 3))
    for i, c in enumerate(centroids, start=1):
        out[i] = c   # (z, y, x)
    return out

def detect_ice_bonds(lab, seg, dilate_vox=DILATE_VOX, min_ice=MIN_ICE_VOX):
    """Find all bead pairs bridged by ice.
    Strategy: dilate the whole labelled image; where two labels touch in
    the dilation ring AND the original gap is filled by ice -> bond."""
    print(f'    detecting ice bonds (dilate {dilate_vox} vox, min ice {min_ice})...')
    t0 = time.time()
    # Grey dilation: each background voxel takes the LABEL of the nearest
    # (within dilate_vox) foreground voxel. Fast.
    dilated = ndimage.grey_dilation(lab, size=(2*dilate_vox+1,)*3)
    # Find overlap shell: voxels where dilated label differs from original label
    # AND gray-dilated value > 0 (not outside).
    # We want CONTACT PAIRS: for each foreground voxel in the DILATED image
    # that's NOT in the ORIGINAL labeled set, check if it's near TWO different labels.
    # Easier approach: label the complement (non-bead voxels) and find which
    # original labels border each complement CC.
    # Even easier: iterate over label pairs using a dilation-by-one trick.

    # Fast pair-detection: for each voxel, if original = 0 and dilated = A,
    # record which label-A region it belongs to. Check if any neighbour voxel
    # (within 1 step) has dilated label B != A -> then (A, B) is a contact pair
    # and this voxel is in their shell. Ice voxels in the shell = bond strength.
    shell = (lab == 0) & (dilated > 0)
    # For each shell voxel, find its "dilated label" A and check neighbours
    # for other labels. Building the full pair-map vectorised is tricky; we
    # use scipy label + boundary lookup instead.

    # Alternative: for each pair (A, B), compute overlap of dilated_A ring
    # and dilated_B ring. Expensive if too many pairs. Use a sparse approach:
    # only pairs that have nonzero overlap in a small bounding box.
    # Trick: run ndimage.label on the binary foreground of (dilated > 0) -
    # no that doesn't help either.

    # Simplest working method: 6-connected neighbour check via np.roll.
    # For each direction, form (lab_at_voxel, lab_at_neighbour) pairs where
    # both are > 0 and different.
    pair_counts = {}
    ice_counts  = {}
    ice = seg == 1
    max_lab = int(dilated.max())
    base = np.int64(max_lab + 1)
    for axis in range(3):
        for shift in (-1, 1):
            rolled     = np.roll(dilated, shift, axis=axis)
            ice_rolled = np.roll(ice,     shift, axis=axis)
            mask = (dilated > 0) & (rolled > 0) & (dilated != rolled)
            if not mask.any():
                continue
            a  = dilated[mask].astype(np.int64)
            b  = rolled [mask].astype(np.int64)
            lo = np.minimum(a, b); hi = np.maximum(a, b)
            keys = lo * base + hi
            has_ice = (ice[mask] | ice_rolled[mask])
            # Tally all pair occurrences
            uniq, cnt = np.unique(keys, return_counts=True)
            for k, c in zip(uniq, cnt):
                la = int(k // base); lb = int(k % base)
                pair_counts[(la, lb)] = pair_counts.get((la, lb), 0) + int(c)
            # Tally pair occurrences where either endpoint is ice
            ice_keys = keys[has_ice]
            if ice_keys.size:
                uniq_i, cnt_i = np.unique(ice_keys, return_counts=True)
                for k, c in zip(uniq_i, cnt_i):
                    la = int(k // base); lb = int(k % base)
                    ice_counts[(la, lb)] = ice_counts.get((la, lb), 0) + int(c)

    # Filter to pairs with enough ice
    bonds = {k: v for k, v in ice_counts.items() if v >= min_ice}
    print(f'    found {len(pair_counts)} contact pairs, {len(bonds)} with >= {min_ice} ice vox   ({time.time()-t0:.1f}s)')
    return bonds, pair_counts

# ---------------- per-scan detection ---------------------------------------
bonds_by_scan = {}
cent_by_scan  = {}
ALL_SCANS = sorted({s for ab in TRANSITIONS for s in ab})
for s in ALL_SCANS:
    print(f'\n--- scan {s} ---')
    lab = tifffile.imread(os.path.join(DATA, f'bead_labels_scan{s:02d}_aligned.tif'))
    ct  = tifffile.imread(os.path.join(DATA, f'ct_scan{s:02d}_aligned.tif'))
    th  = TH[s]
    print(f'  lab shape {lab.shape}  max label {int(lab.max())}')
    seg = reconstruct_seg(ct, th)
    del ct
    bonds, all_pairs = detect_ice_bonds(lab, seg)
    cent = centroids_from_labels(lab)
    bonds_by_scan[s] = bonds
    cent_by_scan[s]  = cent
    del lab, seg

    # Write per-scan CSV
    csv = os.path.join(OUT, f'ice_bonds_scan{s:02d}.csv')
    with open(csv, 'w') as f:
        f.write('labelA;labelB;ice_voxels;cz_A;cy_A;cx_A;cz_B;cy_B;cx_B;dist_vox\n')
        for (la, lb), n in sorted(bonds.items(), key=lambda kv: -kv[1]):
            cA = cent[la]; cB = cent[lb]
            d = np.linalg.norm(cA - cB)
            f.write(f'{la};{lb};{n};{cA[0]:.2f};{cA[1]:.2f};{cA[2]:.2f};'
                    f'{cB[0]:.2f};{cB[1]:.2f};{cB[2]:.2f};{d:.2f}\n')
    print(f'  wrote {csv}')

# ---------------- pair DDIC label maps -------------------------------------
def build_label_map(tsv_path, cent_src, cent_dst):
    """Map scan-A label -> scan-B label using DDIC displacements."""
    with open(tsv_path) as f: header = f.readline().strip().split('\t')
    data = np.loadtxt(tsv_path, skiprows=1, delimiter='\t')
    col = {h: i for i, h in enumerate(header)}
    beads = data[data[:, col['Label']].astype(int) > 0]
    labA = beads[:, col['Label']].astype(int)
    zp   = beads[:, col['Zpos']]; yp = beads[:, col['Ypos']]; xp = beads[:, col['Xpos']]
    zd   = beads[:, col['Zdisp']]; yd = beads[:, col['Ydisp']]; xd = beads[:, col['Xdisp']]
    rs   = beads[:, col['returnStatus']].astype(int)
    err  = beads[:, col['error']]
    valid = (np.isfinite(zd) & np.isfinite(yd) & np.isfinite(xd) & (rs >= 1) & (err < 25000))
    # Predicted B position for each scan-A bead
    pred = np.column_stack([zp + zd, yp + yd, xp + xd])
    # Build kd-tree of scan-B centroids
    from scipy.spatial import cKDTree
    # cent_dst row = [z, y, x]
    mask_positive = cent_dst[:, 0] != 0
    valid_dst_labels = np.where(cent_dst[:, 0] != 0)[0]
    # Build tree only from valid destination centroids (skip label 0 background)
    dst_pts = cent_dst[1:]   # labels 1..max
    tree = cKDTree(dst_pts)
    dist, idx = tree.query(pred, k=1)
    lmap = {}
    for i in range(len(labA)):
        if valid[i] and dist[i] < CENTROID_TOL:
            lmap[int(labA[i])] = int(idx[i] + 1)  # +1 because idx is 0-based into cent_dst[1:]
    print(f'  built label map {os.path.basename(tsv_path)}: {len(lmap)} mapped, '
          f'{int(valid.sum()) - len(lmap)} unmapped valid')
    return lmap

# ---------------- per-transition comparison --------------------------------
for a, b in TRANSITIONS:
    print(f'\n=== transition {a} -> {b} ===')
    tsv = os.path.join(RES, f'75_ddic_{a}{b}-ddic.tsv')
    lmap = build_label_map(tsv, cent_by_scan[a], cent_by_scan[b])
    bA = bonds_by_scan[a]
    bB_set = set(bonds_by_scan[b].keys())
    bB     = bonds_by_scan[b]

    surviving, weakened, broken, new = 0, 0, 0, 0
    rows = []
    for (la, lb), ice_a in bA.items():
        mapA = lmap.get(la); mapB = lmap.get(lb)
        status = 'unmapped'
        ice_b = 0
        if mapA is not None and mapB is not None:
            key = (min(mapA, mapB), max(mapA, mapB))
            ice_b = bB.get(key, 0)
            if ice_b == 0:
                status = 'broken';  broken += 1
            elif ice_b < 0.5 * ice_a:
                status = 'weakened'; weakened += 1
            else:
                status = 'surviving'; surviving += 1
        rows.append((la, lb, ice_a, mapA, mapB, ice_b, status))

    # New bonds = in B but not pointed to by any mapping of an A-bond
    # (approximate: any bond in B whose (labelB_a, labelB_b) pair is not the
    #  image of a mapped A-bond)
    mapped_b_pairs = set()
    for (la, lb) in bA:
        mapA = lmap.get(la); mapB = lmap.get(lb)
        if mapA is not None and mapB is not None:
            mapped_b_pairs.add((min(mapA, mapB), max(mapA, mapB)))
    new = len(bB_set - mapped_b_pairs)

    print(f'  surviving: {surviving}   weakened: {weakened}   broken: {broken}   new: {new}')

    tdir = os.path.join(OUT, f'transition_{a}to{b}')
    os.makedirs(tdir, exist_ok=True)
    csv = os.path.join(tdir, f'ice_bonds_change.csv')
    with open(csv, 'w') as f:
        f.write('labelA_scanA;labelB_scanA;ice_vox_A;labelA_scanB;labelB_scanB;ice_vox_B;status\n')
        for (la, lb, ia, ma, mb, ib, st) in rows:
            f.write(f'{la};{lb};{ia};{ma};{mb};{ib};{st}\n')
    print(f'  wrote {csv}')

    # Plot: bar chart of statuses
    fig, ax = plt.subplots(figsize=(12, 8))
    cats = ['surviving', 'weakened', 'broken', 'new']
    vals = [surviving, weakened, broken, new]
    colors = ['#2ca02c', '#ff7f0e', '#d62728', '#1f77b4']
    ax.bar(cats, vals, color=colors, edgecolor='black')
    ax.set_ylabel('Bond count', fontsize=26, labelpad=10)
    ax.set_title(f'Ice-bond changes   scan {a} -> scan {b}',
                 fontsize=24, fontweight='bold', pad=15)
    ax.tick_params(labelsize=22)
    ax.grid(True, axis='y', alpha=0.3)
    for sp in ax.spines.values(): sp.set_linewidth(1.8)
    plt.tight_layout()
    fig.savefig(os.path.join(tdir, 'ice_bonds_status.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Plot: ice-vox count histogram before vs after
    fig, ax = plt.subplots(figsize=(14, 9))
    iceA = np.array([ia for (_, _, ia, *_) in rows])
    iceB = np.array([ib for (_, _, _, _, _, ib, _) in rows if ib > 0])
    bins = np.linspace(0, max(iceA.max(), iceB.max() if len(iceB) else iceA.max()) + 50, 40)
    ax.hist(iceA, bins=bins, alpha=0.55, color='#1f77b4', label=f'scan {a} (n={len(iceA)})',
            edgecolor='black')
    ax.hist(iceB, bins=bins, alpha=0.55, color='#d62728', label=f'scan {b} (n={len(iceB)})',
            edgecolor='black')
    ax.set_xlabel('Ice voxels per bond', fontsize=26, labelpad=10)
    ax.set_ylabel('Bond count',          fontsize=26, labelpad=10)
    ax.set_title(f'Ice-bond strength distribution  scan {a} -> scan {b}',
                 fontsize=24, fontweight='bold', pad=15)
    ax.tick_params(labelsize=22)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    ax.grid(True, alpha=0.3)
    for sp in ax.spines.values(): sp.set_linewidth(1.8)
    plt.tight_layout()
    fig.savefig(os.path.join(tdir, 'ice_bonds_strength_hist.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3D network render (scan-A network, green=surviving, red=broken/weakened)
    try:
        cent = cent_by_scan[a]   # (max+1, 3) in (z, y, x)
        # Points: use only beads that appear in any bond
        used = set(); [used.update(k) for k in bA]
        pts = np.array([cent[l][::-1] for l in sorted(used)])  # convert to (x,y,z)
        label_to_idx = {l: i for i, l in enumerate(sorted(used))}
        lines_pts, lines_cells, col_vals = [], [], []
        for row in rows:
            la, lb, ia, _, _, ib, st = row
            if la not in label_to_idx or lb not in label_to_idx:
                continue
            iA, iB = label_to_idx[la], label_to_idx[lb]
            code = {'surviving': 0, 'weakened': 1, 'broken': 2, 'unmapped': 3}.get(st, 3)
            lines_pts.append(cent[la][::-1])
            lines_pts.append(cent[lb][::-1])
            idx = len(lines_pts) - 2
            lines_cells.append([2, idx, idx + 1])
            col_vals.append(code)
        if lines_pts:
            lines_pts_arr = np.asarray(lines_pts, dtype=np.float32)
            lines_cells_arr = np.asarray(lines_cells).ravel().astype(np.int64)
            bond_pd = pv.PolyData()
            bond_pd.points = lines_pts_arr
            bond_pd.lines = lines_cells_arr
            bond_pd['status'] = np.repeat(col_vals, 2)
            tubes = bond_pd.tube(radius=2.0, n_sides=8)

            bead_pd = pv.PolyData(pts.astype(np.float32))
            bead_spheres = bead_pd.glyph(
                geom=pv.Sphere(radius=28, theta_resolution=18, phi_resolution=18),
                scale=False, orient=False)

            p = pv.Plotter(off_screen=True, window_size=(2400, 1800))
            p.set_background('white')
            p.enable_anti_aliasing('msaa', multi_samples=8)
            p.add_mesh(bead_spheres, color='#cdd3dd', opacity=0.25, smooth_shading=True)
            cmap_cols = ['#2ca02c', '#ff7f0e', '#d62728', '#888888']
            p.add_mesh(tubes, scalars='status', cmap=cmap_cols, clim=(0, 3),
                       show_scalar_bar=False)
            try:
                p.add_legend(labels=[('surviving','#2ca02c'), ('weakened','#ff7f0e'),
                                     ('broken','#d62728'), ('unmapped','#888888')],
                             size=(0.25, 0.16), loc='upper right', face='rectangle',
                             bcolor='#eeeeee', border=True)
            except Exception:
                pass
            nx_e, ny_e, nz_e = 803, 707, 1241   # T5_HR common aligned frame
            p.camera_position = [
                (nx_e * 2.1, -ny_e * 1.5, -nz_e * 0.7),
                (nx_e / 2, ny_e / 2, nz_e / 2),
                (0, 0, -1),
            ]
            fig_path = os.path.join(tdir, 'ice_bonds_network3D.png')
            p.screenshot(fig_path); p.close()
            print(f'  wrote {fig_path}')
    except Exception as e:
        print(f'  3D render failed: {e}')

print('\nDone. Outputs in', OUT)
