"""Classify each broken ice bond as DELAMINATION, COHESIVE, or MIXED mode.

Physical distinction (per broken bond, analyzed in the DEFORMED scan B):
  DELAMINATION -- ice peels off bead surface: the air voxels in the
                  contact zone sit directly next to a glass bead
                  (glass-air interface exposed, no ice buffer).
  COHESIVE     -- ice fractures through itself: air voxels are
                  surrounded by other air or residual ice, not by
                  fresh glass surfaces.
  MIXED        -- both happen in the same bond.

Algorithm per tracked bond (scan A -> scan B):
  1. Map bond (la, lb) -> (ma, mb) via (injective) DDIC label-map.
  2. Define the INTERFACE ZONE in scan B:
        zone = dilate(ma, R) AND dilate(mb, R)  -- overlapping reach
        inter = zone AND NOT(ma) AND NOT(mb)    -- pure between-bead volume
     (R = INTER_REACH voxels; a small gap around each bead surface where
     the bond material would live.)
  3. Measure composition of `inter`:
        V_air  = voxels of `inter` that are air  -- delaminated gap
        V_ice  = voxels of `inter` that are ice  -- bonded gap
        V_other = rest (other beads, aluminum)   -- ignored
  4. Measure DIRECT contact (crush): V_contact = | ma AND dilate(mb, 1) |
        -- glass-glass touching voxels.
  5. Fractions:
        f_crush = V_contact / (V_contact + V_air + V_ice)
        f_delam = V_air / (V_air + V_ice)   (undefined if V_air+V_ice == 0)
  6. Classification hierarchy:
        CRUSHED       -- V_air + V_ice very small AND V_contact > 0
                        (beads pressed directly, ice gone, no air gap either)
        DELAMINATION  -- f_delam >= 0.70  (air fills the gap)
        COHESIVE      -- f_delam <= 0.30  (ice still fills the gap)
        MIXED         -- between 0.30 and 0.70
        UNCLEAR       -- V_air + V_ice == 0 AND V_contact == 0

Outputs into results_<PRE>/transition_AtoB/:
  failure_modes.csv                 per broken bond: (la, lb, mode, scores,
                                                     centroid coords)
  failure_modes_stacked_bar.png     stacked bar of mode counts per transition
  failure_modes_network3D.png       3D render with bonds colored by mode
  failure_modes_spatial_map.png     XY scatter of failure locations, colored
                                    by mode (shows where each mode dominates)

Plus a cross-transition summary:
  failure_modes_summary.png         mode fractions across the 3 steps

Run in WSL spam-venv:
    python /mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/spam_failure_modes.py
"""
import os, time
import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyvista as pv
pv.OFF_SCREEN = True

DATA = '/home/cak7496/spam-data-t7'
RES  = '/home/cak7496/spam-results-75'
OUT  = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>'

# Per-scan phase thresholds (same as ice-bonds script — from scanNN_segmentation.py)
TH = {
    1: dict(air=7000,    ig=18032.7, ga=47275),
    2: dict(air=2139.4,  ig=6271.5,  ga=99999),
    3: dict(air=2015.4,  ig=6539.5,  ga=99999),
}

INTER_REACH  = 5       # vox, dilation used to define the inter-bead zone
CRUSH_FRAC   = 0.80    # f_crush above this -> CRUSHED
DELAM_RATIO  = 0.70    # f_delam above this -> DELAMINATION
COHES_RATIO  = 0.30    # f_delam below this -> COHESIVE
MIN_ZONE_VOX = 20      # below this, bond is too small to classify reliably

NEIGH_SHIFTS = [(0, 0, 1), (0, 0, -1),
                (0, 1, 0), (0, -1, 0),
                (1, 0, 0), (-1, 0, 0)]

CENTROID_TOL = 50     # voxels, for DDIC label map
VOXEL_UM = 24.7660229

TRANSITIONS = [(1, 2), (2, 3)]

# --------- plotting style ----------------------------------------------------
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
MODE_COLORS = {
    'DELAMINATION': '#d62728',   # red   — interface failure
    'COHESIVE':     '#1f77b4',   # blue  — ice fracture
    'CRUSHED':      '#2ca02c',   # green — hard glass-glass contact
    'MIXED':        '#ff7f0e',   # orange
}

# --------- helpers -----------------------------------------------------------
def centroids(lab):
    m = int(lab.max())
    c = ndimage.center_of_mass(lab > 0, lab, range(1, m + 1))
    arr = np.zeros((m + 1, 3))
    for i, v in enumerate(c, 1): arr[i] = v
    return arr

def load_bonds(csv):
    bonds = {}
    with open(csv) as f:
        next(f)
        for ln in f:
            parts = ln.strip().split(';')
            a, b = int(parts[0]), int(parts[1])
            bonds[(a, b)] = int(parts[2])
    return bonds

def load_labelmap(tsv, cent_b):
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
    tree = cKDTree(cent_b[1:])
    dist, idx = tree.query(pred, k=1)
    # Build injective map: each scan-B label is claimed by at most ONE scan-A
    # label (the closest match). Prevents ma==mb ambiguity downstream.
    best = {}   # lb -> (dist, la)
    for i in range(len(labA)):
        if not (valid[i] and dist[i] < CENTROID_TOL):
            continue
        lb_pick = int(idx[i] + 1)
        la_pick = int(labA[i])
        prev = best.get(lb_pick)
        if prev is None or dist[i] < prev[0]:
            best[lb_pick] = (dist[i], la_pick)
    lmap = {la: lb for lb, (_, la) in best.items()}
    return lmap

# --------- per-transition analysis ------------------------------------------
mode_counts_all = {}    # (a,b) -> dict(DELAM, COH, MIXED, broken_total)
for a, b in TRANSITIONS:
    print(f'\n=== transition {a} -> {b} ===')
    tdir = os.path.join(OUT, f'transition_{a}to{b}')
    bonds_a = load_bonds(os.path.join(OUT, f'ice_bonds_scan{a:02d}.csv'))
    bonds_b = load_bonds(os.path.join(OUT, f'ice_bonds_scan{b:02d}.csv'))

    # Load scan-B data (we analyze air/glass/ice in the deformed state)
    lab_b = tifffile.imread(os.path.join(DATA, f'bead_labels_scan{b:02d}_aligned.tif'))
    ct_b  = tifffile.imread(os.path.join(DATA, f'ct_scan{b:02d}_aligned.tif'))
    th_b  = TH[b]
    air_b   = ct_b < th_b['air']                        # air voxels in scan B
    ice_b   = (ct_b >= th_b['air']) & (ct_b < th_b['ig'])
    del ct_b
    cent_a = centroids(tifffile.imread(os.path.join(DATA, f'bead_labels_scan{a:02d}_aligned.tif')))
    cent_b = centroids(lab_b)
    lmap   = load_labelmap(os.path.join(RES, f'75_ddic_{a}{b}-ddic.tsv'), cent_b)
    print(f'  scan-A ice bonds: {len(bonds_a)}   scan-B ice bonds: {len(bonds_b)}')
    print(f'  label map:         {len(lmap)} mapped')

    # Precompute "all glass" mask once (dilated label > 0 AND not air AND not ice)
    # For speed we don't recompute per bond.

    rows = []
    t0 = time.time()
    n_broken = 0
    n_skip_same = 0
    for (la, lb), ice_a in bonds_a.items():
        ma = lmap.get(la); mb = lmap.get(lb)
        if ma is None or mb is None:
            continue
        if ma == mb:
            # DDIC ambiguity: both beads in scan A mapped to the same
            # scan-B label (likely crushed/merged). Skip -- can't analyse
            # a bond against itself.
            n_skip_same += 1
            continue
        key_b = (min(ma, mb), max(ma, mb))
        ice_b_count = bonds_b.get(key_b, 0)
        # Bond state by ice-volume change (not by mode; mode is computed below)
        if ice_b_count == 0:
            state = 'BROKEN'
            n_broken += 1
        elif ice_b_count < 0.50 * ice_a:
            state = 'WEAKENED'   # ice reduced by >50% but still present
        else:
            state = 'SURVIVING'  # intact bond; still classify mode for context
        # Build contact zone in scan B
        # Use a local bounding box around both centroids for speed
        c1 = cent_b[ma]; c2 = cent_b[mb]
        cz = int((c1[0] + c2[0]) / 2)
        cy = int((c1[1] + c2[1]) / 2)
        cx = int((c1[2] + c2[2]) / 2)
        HALF = 50   # vox
        z0, z1 = max(0, cz - HALF), min(lab_b.shape[0], cz + HALF)
        y0, y1 = max(0, cy - HALF), min(lab_b.shape[1], cy + HALF)
        x0, x1 = max(0, cx - HALF), min(lab_b.shape[2], cx + HALF)
        sub = lab_b[z0:z1, y0:y1, x0:x1]
        sub_air = air_b[z0:z1, y0:y1, x0:x1]
        sub_ice = ice_b[z0:z1, y0:y1, x0:x1]

        is_ma = sub == ma
        is_mb = sub == mb
        if not is_ma.any() or not is_mb.any():
            continue
        # Interface zone: intersection of ma's and mb's INTER_REACH dilations,
        # minus the beads themselves. This is the between-bead volume where
        # the ice bond would live.
        reach_ma = ndimage.binary_dilation(is_ma, iterations=INTER_REACH)
        reach_mb = ndimage.binary_dilation(is_mb, iterations=INTER_REACH)
        inter = reach_ma & reach_mb & ~is_ma & ~is_mb
        V_zone = int(inter.sum())
        V_air  = int((inter & sub_air).sum())
        V_ice  = int((inter & sub_ice).sum())
        V_contact = int((is_ma & ndimage.binary_dilation(
            is_mb, iterations=1)).sum())
        total = V_air + V_ice + V_contact
        if total < MIN_ZONE_VOX:
            mode = 'UNCLEAR'
            f_delam = f_crush = 0.0
        else:
            f_crush = V_contact / total
            denom_ai = V_air + V_ice
            if V_contact > 0 and denom_ai < MIN_ZONE_VOX:
                mode = 'CRUSHED'
                f_delam = 0.0
            elif f_crush >= CRUSH_FRAC:
                mode = 'CRUSHED'
                f_delam = V_air / denom_ai if denom_ai > 0 else 0.0
            else:
                f_delam = V_air / denom_ai if denom_ai > 0 else 0.0
                if f_delam >= DELAM_RATIO:
                    mode = 'DELAMINATION'
                elif f_delam <= COHES_RATIO:
                    mode = 'COHESIVE'
                else:
                    mode = 'MIXED'
        rows.append((la, lb, ma, mb, ice_a, ice_b_count, state,
                     V_zone, V_air, V_ice, V_contact,
                     f_delam, 1.0 - f_delam if mode != 'UNCLEAR' else 0.0,
                     f_crush, mode,
                     cent_a[la], cent_a[lb]))
    del lab_b, air_b, ice_b

    # Tally over FAILED bonds only (BROKEN + WEAKENED).
    # row layout now: la,lb,ma,mb,ia,ib,state,n_face,n_d,n_b,n_c,fd,fb,fc,mode,cA,cB
    # indices: state=6, mode=14
    mc = {'DELAMINATION': 0, 'COHESIVE': 0, 'CRUSHED': 0, 'MIXED': 0, 'UNCLEAR': 0}
    n_failed = 0
    state_tally = {'BROKEN': 0, 'WEAKENED': 0, 'SURVIVING': 0}
    for row in rows:
        state_tally[row[6]] = state_tally.get(row[6], 0) + 1
        if row[6] == 'SURVIVING':
            continue
        mc[row[14]] = mc.get(row[14], 0) + 1
        n_failed += 1
    mode_counts_all[(a, b)] = dict(mc)
    mode_counts_all[(a, b)]['broken_total'] = n_failed
    mode_counts_all[(a, b)]['state'] = dict(state_tally)
    print(f'  tracked bonds: {len(rows)}   '
          f'(surviving={state_tally["SURVIVING"]}, '
          f'weakened={state_tally["WEAKENED"]}, broken={state_tally["BROKEN"]})')
    print(f'  failed-bond modes: DELAM={mc["DELAMINATION"]} '
          f'COH={mc["COHESIVE"]} CRUSH={mc["CRUSHED"]} '
          f'MIXED={mc["MIXED"]} UNCLEAR={mc["UNCLEAR"]}  '
          f'({time.time()-t0:.1f}s)')
    print(f'  skipped (ma==mb ambiguity): {n_skip_same}')

    # --------- per-transition outputs ---
    # CSV
    csv = os.path.join(tdir, 'failure_modes.csv')
    with open(csv, 'w') as f:
        f.write('labelA;labelB;mappedA;mappedB;ice_vox_scanA;ice_vox_scanB;'
                'state;V_zone;V_air;V_ice;V_contact;'
                'f_delam;f_ice;f_crush;mode;'
                'cA_z;cA_y;cA_x;cB_z;cB_y;cB_x\n')
        for r in rows:
            (la, lb, ma, mb, ia, ib, state, V_zone,
             V_a, V_i, V_c, fd, fi, fc, mode, cA, cB) = r
            f.write(f'{la};{lb};{ma};{mb};{ia};{ib};{state};{V_zone};'
                    f'{V_a};{V_i};{V_c};'
                    f'{fd:.3f};{fi:.3f};{fc:.3f};{mode};'
                    f'{cA[0]:.1f};{cA[1]:.1f};{cA[2]:.1f};'
                    f'{cB[0]:.1f};{cB[1]:.1f};{cB[2]:.1f}\n')
    print(f'  wrote {csv}')

    # Stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    modes = ['DELAMINATION', 'COHESIVE', 'CRUSHED', 'MIXED']
    vals = [mc[m] for m in modes]
    colors = [MODE_COLORS[m] for m in modes]
    ax.bar(modes, vals, color=colors, edgecolor='black')
    ax.set_ylabel('Broken bond count', labelpad=10)
    ax.set_title(f'Failure mode classification   scan {a} -> {b}',
                 fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(os.path.join(tdir, 'failure_modes_stacked_bar.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # XY spatial map: FAILED bonds only (exclude SURVIVING), colored by mode
    # row layout now: la,lb,ma,mb,ia,ib,state,n_face,n_d,n_b,n_c,fd,fb,fc,mode,cA,cB
    # cA is r[15], cB is r[16]
    fig, ax = plt.subplots(figsize=(14, 12))
    failed_rows = [r for r in rows if r[6] != 'SURVIVING']
    for mode in modes:
        xs = [0.5 * (r[15][2] + r[16][2]) for r in failed_rows if r[14] == mode]
        ys = [0.5 * (r[15][1] + r[16][1]) for r in failed_rows if r[14] == mode]
        ax.scatter(xs, ys, s=100, color=MODE_COLORS[mode], edgecolor='black',
                   label=f'{mode} ({mc[mode]})', alpha=0.8)
    ax.set_xlabel('X (voxels)', labelpad=10)
    ax.set_ylabel('Y (voxels)', labelpad=10)
    ax.set_aspect('equal')
    ax.set_title(f'Failure spatial map — scan {a} -> {b}\n'
                 f'(XY projection of broken-bond midpoints)',
                 fontweight='bold', pad=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(tdir, 'failure_modes_spatial_map.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3D network render with mode colors
    try:
        lines_pts = []; lines_cells = []; col_vals = []
        for r in rows:
            (la, lb, ma, mb, ia, ib, state, V_zone,
             V_a, V_i, V_c, fd, fi, fc, mode, cA, cB) = r
            if state == 'SURVIVING':
                continue   # 3D network shows failed bonds only
            code = {'DELAMINATION': 0, 'COHESIVE': 1, 'CRUSHED': 2,
                    'MIXED': 3, 'UNCLEAR': 4}.get(mode, 4)
            lines_pts.append(cA[::-1]); lines_pts.append(cB[::-1])   # (x,y,z)
            idx = len(lines_pts) - 2
            lines_cells.append([2, idx, idx + 1])
            col_vals.append(code)
        if lines_pts:
            pts_arr = np.asarray(lines_pts, dtype=np.float32)
            cells_arr = np.asarray(lines_cells).ravel().astype(np.int64)
            bond_pd = pv.PolyData()
            bond_pd.points = pts_arr
            bond_pd.lines = cells_arr
            bond_pd['mode'] = np.repeat(col_vals, 2)
            tubes = bond_pd.tube(radius=2.5, n_sides=8)

            p = pv.Plotter(off_screen=True, window_size=(2400, 1800))
            p.set_background('white')
            p.enable_anti_aliasing('msaa', multi_samples=8)
            p.add_mesh(tubes, scalars='mode',
                       cmap=['#d62728', '#1f77b4', '#2ca02c',
                             '#ff7f0e', '#888888'],
                       clim=(0, 4), show_scalar_bar=False)
            try:
                p.add_legend(labels=[('DELAMINATION', '#d62728'),
                                     ('COHESIVE', '#1f77b4'),
                                     ('CRUSHED', '#2ca02c'),
                                     ('MIXED', '#ff7f0e'),
                                     ('UNCLEAR', '#888888')],
                             size=(0.27, 0.19), loc='upper right',
                             face='rectangle', bcolor='#eeeeee', border=True)
            except Exception:
                pass
            nx_e, ny_e, nz_e = 611, 614, 1106
            p.camera_position = [
                (nx_e * 2.1, -ny_e * 1.5, -nz_e * 0.7),
                (nx_e / 2, ny_e / 2, nz_e / 2),
                (0, 0, -1),
            ]
            fig3d = os.path.join(tdir, 'failure_modes_network3D.png')
            p.screenshot(fig3d); p.close()
            print(f'  wrote {fig3d}')
    except Exception as e:
        print(f'  3D render failed: {e}')

# --------- cross-transition summary bar chart -------------------------------
fig, ax = plt.subplots(figsize=(16, 9))
x = np.arange(len(TRANSITIONS))
modes4 = ['DELAMINATION', 'COHESIVE', 'CRUSHED', 'MIXED']
width = 0.20
for i, m in enumerate(modes4):
    vals = [mode_counts_all[t][m] for t in TRANSITIONS]
    ax.bar(x + (i - 1.5) * width, vals, width, color=MODE_COLORS[m],
           edgecolor='black', label=m)
ax.set_xticks(x); ax.set_xticklabels([f'{a}→{b}' for a, b in TRANSITIONS])
ax.set_ylabel('Broken bond count', labelpad=10)
ax.set_title('Bond failure modes across all compression steps',
             fontweight='bold', pad=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'failure_modes_summary.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# --------- summary text -----------------------------------------------------
with open(os.path.join(OUT, 'failure_modes_summary.txt'), 'w') as f:
    f.write('T7 1700 -- ice-bond failure mode classification\n')
    f.write('=' * 50 + '\n\n')
    f.write(f"{'trans':<8}{'total':>8}{'delam':>8}{'cohes':>8}{'crush':>8}"
            f"{'mixed':>8}{'unclr':>8}"
            f"{'delam%':>10}{'cohes%':>10}{'crush%':>10}{'mixed%':>10}\n")
    for a, b in TRANSITIONS:
        mc = mode_counts_all[(a, b)]
        total = max(mc['broken_total'], 1)
        f.write(f'{a}->{b:<5}{mc["broken_total"]:>8}'
                f'{mc["DELAMINATION"]:>8}{mc["COHESIVE"]:>8}'
                f'{mc["CRUSHED"]:>8}{mc["MIXED"]:>8}{mc["UNCLEAR"]:>8}'
                f'{100*mc["DELAMINATION"]/total:>10.1f}'
                f'{100*mc["COHESIVE"]/total:>10.1f}'
                f'{100*mc["CRUSHED"]/total:>10.1f}'
                f'{100*mc["MIXED"]/total:>10.1f}\n')
    f.write('\nInterpretation:\n')
    f.write('  DELAMINATION -- ice separated from the bead surface (interface failure).\n')
    f.write('  COHESIVE     -- ice stayed on the glass but fractured through itself.\n')
    f.write('  CRUSHED      -- ice squeezed out, beads now in direct glass-glass contact.\n')
    f.write('  MIXED        -- more than one of the above modes appears in the same bond.\n')
    f.write(f'\nThresholds (volumes in the inter-bead zone, scan B):\n')
    f.write(f'  V_air+V_ice very small with V_contact>0 -> CRUSHED\n')
    f.write(f'  f_crush >= {int(CRUSH_FRAC*100)}%                         -> CRUSHED\n')
    f.write(f'  f_delam = V_air/(V_air+V_ice) >= {int(DELAM_RATIO*100)}%  -> DELAMINATION\n')
    f.write(f'  f_delam <= {int(COHES_RATIO*100)}%                        -> COHESIVE\n')
    f.write(f'  between   -> MIXED\n')
print(f'\nwrote {os.path.join(OUT, "failure_modes_summary.txt")}')
print(f'wrote {os.path.join(OUT, "failure_modes_summary.png")}')


# ===========================================================================
# Per-BEAD delamination analysis
# For each bead that can be tracked scan A -> scan B, measure what fraction
# of its glass surface is adjacent to AIR (dryness). The CHANGE dryness_B -
# dryness_A captures "ice peeled off the bead's surface" -- direct visual
# delamination the user observed.
# ===========================================================================
print('\n\n===== Per-bead delamination analysis =====')

def bead_surface_counts(lab, air_mask, ice_mask):
    """Returns per-label counts of (air_neighbor_surf_vox, ice_neighbor_surf_vox).
    A surface voxel is a glass voxel with at least one non-same-label neighbour.
    For each surface voxel of label L, we count +1 air_adj[L] per shift whose
    neighbour is air, and +1 ice_adj[L] per shift whose neighbour is ice.
    """
    n_labels = int(lab.max())
    air_adj = np.zeros(n_labels + 1, dtype=np.int64)
    ice_adj = np.zeros(n_labels + 1, dtype=np.int64)
    glass_mask = lab > 0
    # any-different-neighbour mask: True where a glass voxel has a non-same-label
    # neighbour (i.e., surface of its own bead).
    any_diff = np.zeros_like(glass_mask)
    for dz, dy, dx in NEIGH_SHIFTS:
        rolled_lab = np.roll(lab, (dz, dy, dx), axis=(0, 1, 2))
        rolled_air = np.roll(air_mask, (dz, dy, dx), axis=(0, 1, 2))
        rolled_ice = np.roll(ice_mask, (dz, dy, dx), axis=(0, 1, 2))
        diff = glass_mask & (rolled_lab != lab)
        any_diff |= diff
        # surf voxels with air neighbour
        air_here = diff & rolled_air
        np.add.at(air_adj, lab[air_here], 1)
        ice_here = diff & rolled_ice
        np.add.at(ice_adj, lab[ice_here], 1)
    return air_adj, ice_adj, any_diff

bead_counts_all = {}   # transition -> list of (la, lb, dryness_A, dryness_B,
                       #                         delta, x, y, z)
for a, b in TRANSITIONS:
    print(f'\n--- {a} -> {b} per-bead dryness ---')
    tdir = os.path.join(OUT, f'transition_{a}to{b}')
    lab_a = tifffile.imread(os.path.join(DATA, f'bead_labels_scan{a:02d}_aligned.tif'))
    lab_b = tifffile.imread(os.path.join(DATA, f'bead_labels_scan{b:02d}_aligned.tif'))
    ct_a  = tifffile.imread(os.path.join(DATA, f'ct_scan{a:02d}_aligned.tif'))
    ct_b  = tifffile.imread(os.path.join(DATA, f'ct_scan{b:02d}_aligned.tif'))
    th_a = TH[a]; th_b = TH[b]
    air_a = ct_a < th_a['air']; ice_a = (ct_a >= th_a['air']) & (ct_a < th_a['ig'])
    air_b = ct_b < th_b['air']; ice_b = (ct_b >= th_b['air']) & (ct_b < th_b['ig'])
    del ct_a, ct_b

    t0 = time.time()
    airA, iceA, _ = bead_surface_counts(lab_a, air_a, ice_a)
    airB, iceB, _ = bead_surface_counts(lab_b, air_b, ice_b)
    print(f'  surface counts done ({time.time()-t0:.1f}s)')
    del air_a, ice_a, air_b, ice_b

    centA = centroids(lab_a)
    centB = centroids(lab_b)
    lmap = load_labelmap(os.path.join(RES, f'75_ddic_{a}{b}-ddic.tsv'), centB)
    del lab_a, lab_b

    rows_bead = []
    for la, lb in lmap.items():
        if la >= len(airA) or lb >= len(airB):
            continue
        dA = airA[la] + iceA[la]
        dB = airB[lb] + iceB[lb]
        if dA < 20 or dB < 20:
            continue
        dryness_A = airA[la] / dA
        dryness_B = airB[lb] / dB
        delta = dryness_B - dryness_A
        rows_bead.append((la, lb, dryness_A, dryness_B, delta,
                          centA[la][2], centA[la][1], centA[la][0]))

    print(f'  {len(rows_bead)} beads tracked with valid surface data')
    if not rows_bead:
        bead_counts_all[(a, b)] = []
        continue

    # CSV
    csvp = os.path.join(tdir, 'bead_delamination.csv')
    with open(csvp, 'w') as f:
        f.write('labelA;labelB;dryness_A;dryness_B;delta_dryness;cx;cy;cz\n')
        for r in rows_bead:
            la, lb, dA, dB, dd, cx, cy, cz = r
            f.write(f'{la};{lb};{dA:.4f};{dB:.4f};{dd:.4f};'
                    f'{cx:.1f};{cy:.1f};{cz:.1f}\n')
    print(f'  wrote {csvp}')
    bead_counts_all[(a, b)] = rows_bead

    # Histogram of Δdryness
    deltas = np.array([r[4] for r in rows_bead])
    n_delam = int((deltas > 0.10).sum())
    n_strong = int((deltas > 0.30).sum())
    n_stable = int((np.abs(deltas) <= 0.10).sum())
    n_rebond = int((deltas < -0.10).sum())
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.hist(deltas, bins=40, color='#d62728', edgecolor='black', alpha=0.7)
    ax.axvline(0.10, ls='--', color='k', alpha=0.5, label='Δ=0.10 delam onset')
    ax.axvline(0.30, ls='--', color='darkred', alpha=0.8, label='Δ=0.30 strong delam')
    ax.set_xlabel('Δ dryness  (air-fraction of surface, scan B − scan A)', labelpad=10)
    ax.set_ylabel('Number of beads', labelpad=10)
    ax.set_title(f'Per-bead delamination   scan {a} → {b}\n'
                 f'strong delam ({n_strong}) | mild ({n_delam - n_strong}) | '
                 f'stable ({n_stable}) | re-bonded ({n_rebond})',
                 fontweight='bold', pad=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(tdir, 'bead_delamination_hist.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3D render: beads as spheres colored by Δdryness (red = delaminated)
    try:
        pts = np.array([[r[5], r[6], r[7]] for r in rows_bead], dtype=np.float32)
        cloud = pv.PolyData(pts)
        cloud['delta_dry'] = deltas
        glyph = cloud.glyph(geom=pv.Sphere(radius=18, theta_resolution=14,
                                            phi_resolution=14),
                            scale=False, orient=False)
        p = pv.Plotter(off_screen=True, window_size=(2400, 1800))
        p.set_background('white'); p.enable_anti_aliasing('msaa', multi_samples=8)
        p.add_mesh(glyph, scalars='delta_dry', cmap='coolwarm',
                   clim=(-0.3, 0.3), smooth_shading=True,
                   scalar_bar_args=dict(title='Δ dryness',
                       title_font_size=60, label_font_size=60, color='#1a1a1a',
                       font_family='arial', position_x=0.86, position_y=0.16,
                       height=0.70, width=0.045, vertical=True, shadow=False,
                       n_labels=5))
        nx_e, ny_e, nz_e = 611, 614, 1106
        p.camera_position = [(nx_e * 2.1, -ny_e * 1.5, -nz_e * 0.7),
                             (nx_e / 2, ny_e / 2, nz_e / 2),
                             (0, 0, -1)]
        out3d = os.path.join(tdir, 'bead_delamination_3D.png')
        p.screenshot(out3d); p.close()
        print(f'  wrote {out3d}')
    except Exception as e:
        print(f'  3D render failed: {e}')

# Cross-transition summary
with open(os.path.join(OUT, 'bead_delamination_summary.txt'), 'w') as f:
    f.write('Per-bead delamination summary (air-exposure change scan A -> B)\n')
    f.write('=' * 60 + '\n\n')
    f.write(f"{'trans':<8}{'tracked':>10}{'strong>0.3':>12}"
            f"{'mild>0.1':>12}{'stable':>10}{'rebond':>10}"
            f"{'strong%':>10}{'mean_d':>10}\n")
    for a, b in TRANSITIONS:
        rows_bead = bead_counts_all.get((a, b), [])
        if not rows_bead:
            f.write(f'{a}->{b:<5}{0:>10}{0:>12}{0:>12}{0:>10}{0:>10}'
                    f'{0.0:>10.1f}{0.0:>10.3f}\n')
            continue
        deltas = np.array([r[4] for r in rows_bead])
        total = len(deltas)
        n_str = int((deltas > 0.30).sum())
        n_mid = int(((deltas > 0.10) & (deltas <= 0.30)).sum())
        n_stb = int((np.abs(deltas) <= 0.10).sum())
        n_reb = int((deltas < -0.10).sum())
        f.write(f'{a}->{b:<5}{total:>10}{n_str:>12}{n_mid:>12}'
                f'{n_stb:>10}{n_reb:>10}'
                f'{100*n_str/total:>10.1f}{float(deltas.mean()):>10.3f}\n')
    f.write('\nInterpretation:\n')
    f.write('  Δ dryness = (air-adjacent surface fraction in B) − (in A)\n')
    f.write('  > 0.30 -- strong delamination (much of bead surface newly exposed)\n')
    f.write('  0.10-0.30 -- mild delamination\n')
    f.write('  |Δ| <= 0.10 -- stable interface\n')
    f.write('  < -0.10 -- re-bonded / got buried in ice or other bead contact\n')
print(f'wrote {os.path.join(OUT, "bead_delamination_summary.txt")}')
print('Done.')
