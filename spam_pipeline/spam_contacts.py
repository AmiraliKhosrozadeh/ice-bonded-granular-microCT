"""SPAM contact-network analysis for the T7 1700 project.

Per scan: uses spam.label.labelledContacts to detect bead-to-bead
CONTACTS (direct voxel adjacency, regardless of ice between them).

Per transition (1->2, 2->3, 3->4): compares contact sets using a
DDIC-based label map to classify each contact as
surviving / broken / new.

Outputs per scan (to results_<PRE>/):
  contacts_scanNN.csv                  (labelA, labelB, centres, distance)
  contacts_scanNN_coord_hist.png       coordination number histogram
  contacts_scanNN_orientation.png      orientation (angle vs horizontal) histogram

Outputs per transition (to results_<PRE>/transition_AtoB/):
  contacts_change.csv                  per-scan-A contact with broken/surviving tag
  contacts_status.png                  bar of surviving/broken/new counts
  contacts_rose.png                    orientation rose (before/after)
  contacts_network3D.png               3D render, surviving=green, broken=red

Run in WSL spam-venv:
    python /mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/spam_contacts.py
"""
import os, time
import numpy as np
import tifffile
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyvista as pv
import spam.label as slbl

pv.OFF_SCREEN = True

DATA = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
RES  = '/home/cak7496/spam-results-75'
OUT  = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>'
os.makedirs(OUT, exist_ok=True)

DILATE_VOX   = 3        # dilate labels before contact detection
CENTROID_TOL = 150      # heavy axial compression -> loose tol

TRANSITIONS = [(1, 2), (2, 3)]

def load_and_dilate(path, dvox=DILATE_VOX):
    print(f'  loading {path}')
    lab = tifffile.imread(path).astype(np.int32)
    if dvox > 0:
        lab = ndimage.grey_dilation(lab, size=(2 * dvox + 1,) * 3)
    return lab

def centroids_from_labels(lab):
    max_lab = int(lab.max())
    if max_lab == 0:
        return np.zeros((1, 3))
    c = ndimage.center_of_mass(lab > 0, lab, range(1, max_lab + 1))
    arr = np.zeros((max_lab + 1, 3))
    for i, v in enumerate(c, start=1):
        arr[i] = v   # (z, y, x)
    return arr

def inclinations_deg(pairs, cent):
    out = []
    for a, b in pairs:
        if a >= len(cent) or b >= len(cent):
            continue
        v = cent[b] - cent[a]
        n = np.linalg.norm(v)
        if n == 0:
            continue
        v = v / n
        theta = np.degrees(np.arccos(np.clip(abs(v[0]), 0, 1)))   # v[0] is Z comp
        out.append(90 - theta)
    return np.array(out)

# --------- plotting style -----------------------------------------------------
_FS = 24
plt.rcParams.update({
    "font.size": _FS, "axes.titlesize": _FS, "axes.labelsize": _FS + 2,
    "xtick.labelsize": _FS - 2, "ytick.labelsize": _FS - 2,
    "legend.fontsize": _FS - 2, "axes.linewidth": 1.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.spines.left": True, "axes.spines.bottom": True,
})

contacts_by_scan = {}   # scan -> dict(set_of_pairs, coord, centroids)

ALL_SCANS = sorted({s for ab in TRANSITIONS for s in ab})
for s in ALL_SCANS:
    print(f'\n--- scan {s} ---')
    lab = load_and_dilate(
        os.path.join(DATA, f'bead_labels_scan{s:02d}_aligned.tif'))
    print(f'  running labelledContacts...')
    t0 = time.time()
    _, Z, _, pairs = slbl.labelledContacts(lab, maximumCoordinationNumber=20)
    t1 = time.time() - t0
    pairs = np.sort(pairs, axis=1).astype(int)
    keep = (pairs[:, 0] > 0) & (pairs[:, 1] > 0)
    pairs = np.unique(pairs[keep], axis=0)
    cent = centroids_from_labels(lab)
    print(f'  {len(pairs)} contacts, coord Z median {np.median(Z[Z > 0]):.1f}   ({t1:.1f}s)')

    contacts_by_scan[s] = dict(
        pairs=set(map(tuple, pairs.tolist())),
        pair_list=pairs,
        Z=Z,
        cent=cent,
    )

    # Per-scan CSV
    csv = os.path.join(OUT, f'contacts_scan{s:02d}.csv')
    with open(csv, 'w') as f:
        f.write('labelA;labelB;cz_A;cy_A;cx_A;cz_B;cy_B;cx_B;dist_vox\n')
        for a, b in pairs:
            if a >= len(cent) or b >= len(cent):
                continue
            ca, cb = cent[a], cent[b]
            d = np.linalg.norm(ca - cb)
            f.write(f'{a};{b};{ca[0]:.2f};{ca[1]:.2f};{ca[2]:.2f};'
                    f'{cb[0]:.2f};{cb[1]:.2f};{cb[2]:.2f};{d:.2f}\n')
    print(f'  wrote {csv}')

    # Coordination-number histogram
    fig, ax = plt.subplots(figsize=(12, 8))
    Zv = Z[Z > 0]
    bins = np.arange(0, Zv.max() + 2) - 0.5
    ax.hist(Zv, bins=bins, color='#1f77b4', edgecolor='black', alpha=0.85)
    ax.axvline(np.median(Zv), color='#d62728', linewidth=3, linestyle='--',
               label=f'median = {np.median(Zv):.0f}')
    ax.set_xlabel('Coordination number Z', labelpad=10)
    ax.set_ylabel('Bead count', labelpad=10)
    ax.set_title(f'Coordination — scan {s}   (n={len(Zv)} beads)',
                 fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f'contacts_scan{s:02d}_coord_hist.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Orientation histogram
    incl = inclinations_deg(pairs, cent)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(incl, bins=np.linspace(0, 90, 19), color='#2ca02c',
            edgecolor='black', alpha=0.85)
    ax.set_xlabel('Contact inclination vs horizontal (°)', labelpad=10)
    ax.set_ylabel('Contact count', labelpad=10)
    ax.set_title(f'Contact orientation — scan {s}',
                 fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f'contacts_scan{s:02d}_orientation.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    del lab

# --------- per-transition comparison ----------------------------------------
from scipy.spatial import cKDTree

def build_label_map(tsv, cent_dst):
    with open(tsv) as f: header = f.readline().strip().split('\t')
    data = np.loadtxt(tsv, skiprows=1, delimiter='\t')
    col = {h: i for i, h in enumerate(header)}
    beads = data[data[:, col['Label']].astype(int) > 0]
    labA = beads[:, col['Label']].astype(int)
    pred = np.column_stack([beads[:, col['Zpos']] + beads[:, col['Zdisp']],
                            beads[:, col['Ypos']] + beads[:, col['Ydisp']],
                            beads[:, col['Xpos']] + beads[:, col['Xdisp']]])
    rs = beads[:, col['returnStatus']].astype(int)
    err = beads[:, col['error']]
    valid = (np.isfinite(pred).all(axis=1) & (rs >= 1) & (err < 25000))
    tree = cKDTree(cent_dst[1:])  # skip bg
    dist, idx = tree.query(pred, k=1)
    lmap = {}
    for i in range(len(labA)):
        if valid[i] and dist[i] < CENTROID_TOL:
            lmap[int(labA[i])] = int(idx[i] + 1)
    print(f'  label map {os.path.basename(tsv)}: {len(lmap)} mapped')
    return lmap

for a, b in TRANSITIONS:
    print(f'\n=== transition {a} -> {b} ===')
    tsv = os.path.join(RES, f'75_ddic_{a}{b}-ddic.tsv')
    lmap = build_label_map(tsv, contacts_by_scan[b]['cent'])
    set_a = contacts_by_scan[a]['pairs']
    set_b = contacts_by_scan[b]['pairs']

    surviving, broken, rows, mapped_b_pairs = 0, 0, [], set()
    for (la, lb) in set_a:
        mA = lmap.get(la); mB = lmap.get(lb)
        if mA is None or mB is None:
            status = 'unmapped'
            rows.append((la, lb, mA, mB, status))
            continue
        key = (min(mA, mB), max(mA, mB))
        mapped_b_pairs.add(key)
        if key in set_b:
            status = 'surviving'; surviving += 1
        else:
            status = 'broken'; broken += 1
        rows.append((la, lb, mA, mB, status))
    new = len(set_b - mapped_b_pairs)
    print(f'  surviving: {surviving}   broken: {broken}   new: {new}   '
          f'(scan {a}: {len(set_a)}, scan {b}: {len(set_b)})')

    tdir = os.path.join(OUT, f'transition_{a}to{b}')
    os.makedirs(tdir, exist_ok=True)
    csv = os.path.join(tdir, 'contacts_change.csv')
    with open(csv, 'w') as f:
        f.write('labelA_a;labelB_a;labelA_b;labelB_b;status\n')
        for la, lb, mA, mB, st in rows:
            f.write(f'{la};{lb};{mA if mA else -1};{mB if mB else -1};{st}\n')
    print(f'  wrote {csv}')

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    cats = ['surviving', 'broken', 'new']
    vals = [surviving, broken, new]
    colors = ['#2ca02c', '#d62728', '#1f77b4']
    ax.bar(cats, vals, color=colors, edgecolor='black')
    ax.set_ylabel('Contact count', labelpad=10)
    ax.set_title(f'Contact changes   scan {a} -> scan {b}',
                 fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(os.path.join(tdir, 'contacts_status.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Orientation rose before/after
    cent_a = contacts_by_scan[a]['cent']
    cent_b = contacts_by_scan[b]['cent']
    incl_a = inclinations_deg(list(set_a), cent_a)
    incl_b = inclinations_deg(list(set_b), cent_b)
    fig, ax = plt.subplots(figsize=(12, 8))
    bins = np.linspace(0, 90, 19)
    ax.hist(incl_a, bins=bins, color='#1f77b4', alpha=0.55,
            label=f'scan {a}  (mean {incl_a.mean():.1f}°)', edgecolor='black')
    ax.hist(incl_b, bins=bins, color='#d62728', alpha=0.55,
            label=f'scan {b}  (mean {incl_b.mean():.1f}°)', edgecolor='black')
    ax.set_xlabel('Contact inclination vs horizontal (°)', labelpad=10)
    ax.set_ylabel('Contact count', labelpad=10)
    ax.set_title(f'Orientation before/after   scan {a} -> {b}',
                 fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    plt.tight_layout()
    fig.savefig(os.path.join(tdir, 'contacts_rose.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3D network (flipped Z camera)
    try:
        cent = contacts_by_scan[a]['cent']
        used = set()
        for p in set_a: used.update(p)
        used = sorted(used)
        bead_pts = np.array([cent[l][::-1] for l in used])   # (x, y, z)
        label_to_idx = {l: i for i, l in enumerate(used)}
        lines_pts, lines_cells, col_vals = [], [], []
        for la, lb, mA, mB, st in rows:
            if la not in label_to_idx or lb not in label_to_idx:
                continue
            code = {'surviving': 0, 'broken': 1, 'unmapped': 2}.get(st, 2)
            lines_pts.append(cent[la][::-1]); lines_pts.append(cent[lb][::-1])
            idx = len(lines_pts) - 2
            lines_cells.append([2, idx, idx + 1])
            col_vals.append(code)
        if lines_pts:
            lines_pts_arr = np.asarray(lines_pts, dtype=np.float32)
            cells_arr = np.asarray(lines_cells).ravel().astype(np.int64)
            bond_pd = pv.PolyData()
            bond_pd.points = lines_pts_arr
            bond_pd.lines = cells_arr
            bond_pd['status'] = np.repeat(col_vals, 2)
            tubes = bond_pd.tube(radius=2.0, n_sides=8)

            bead_spheres = pv.PolyData(bead_pts.astype(np.float32)).glyph(
                geom=pv.Sphere(radius=28, theta_resolution=18, phi_resolution=18),
                scale=False, orient=False)
            p = pv.Plotter(off_screen=True, window_size=(2400, 1800))
            p.set_background('white')
            p.enable_anti_aliasing('msaa', multi_samples=8)
            p.add_mesh(bead_spheres, color='#cdd3dd', opacity=0.25,
                       smooth_shading=True)
            p.add_mesh(tubes, scalars='status',
                       cmap=['#2ca02c', '#d62728', '#888888'], clim=(0, 2),
                       show_scalar_bar=False)
            try:
                p.add_legend(labels=[('surviving', '#2ca02c'),
                                     ('broken', '#d62728'),
                                     ('unmapped', '#888888')],
                             size=(0.25, 0.12), loc='upper right',
                             face='rectangle', bcolor='#eeeeee', border=True)
            except Exception:
                pass
            nx_e, ny_e, nz_e = 803, 707, 1241   # T5_HR common aligned frame
            p.camera_position = [
                (nx_e * 2.1, -ny_e * 1.5, -nz_e * 0.7),
                (nx_e / 2, ny_e / 2, nz_e / 2),
                (0, 0, -1),
            ]
            fig3d = os.path.join(tdir, 'contacts_network3D.png')
            p.screenshot(fig3d); p.close()
            print(f'  wrote {fig3d}')
    except Exception as e:
        print(f'  3D render failed: {e}')

print('\nDone. Contacts outputs in', OUT)
