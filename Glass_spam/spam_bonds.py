"""
Bond / contact analysis: scan 1 vs scan 2.

Three studies in one script:
  (1) CONTACT DETECTION  — spam.label.labelledContacts on each scan.
                           Produces coordination number, contact count,
                           broken/new bonds between scans.
  (2) BOND STRAIN        — for each scan-1 contact, compute change in
                           bead-center distance (elongation/shortening).
                           Bonds still present in scan 2 have a strain;
                           broken bonds have "d2 = NaN".
  (3) ORIENTATION STATS  — contact-normal = unit vector between bead
                           centroids. Rose histogram of inclination vs Z.

Outputs (results/bonds/):
  bonds_scan01.csv, bonds_scan02.csv   per-bond tables
  bonds_comparison.csv                 scan1 bonds with strain + broken flag
  bonds_summary.txt                    text summary + coordination stats
  bonds_coordination_hist.png          coordination number histogram
  bonds_rose_inclination.png           angle of contact normal vs Z axis
  bonds_strain_hist.png                distribution of bond strains
  bonds_network_iso.png                3D render of bond network (scan 1)

Run in WSL spam-venv:
    python /mnt/e/RPTU-images/CT_images/Glass/Glass_spam/spam_bonds.py
"""

import os
import numpy as np
import tifffile
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pyvista as pv

pv.OFF_SCREEN = True

SPAM_DATA    = '/home/cak7496/spam-data'
SPAM_RESULTS = '/home/cak7496/spam-results'
OUT_DIR      = '/mnt/e/RPTU-images/CT_images/Glass/Glass_spam/results/bonds'
os.makedirs(OUT_DIR, exist_ok=True)

LAB1_TIF = os.path.join(SPAM_DATA, 'bead_labels_scan01.tif')
LAB2_TIF = os.path.join(SPAM_DATA, 'bead_labels_scan02.tif')
DDIC_TSV = os.path.join(SPAM_RESULTS, 'glass_ddic-ddic.tsv')
VOXEL_SIZE_UM = 24.766

# Dilation BEFORE contact detection: beads were shrunk via erosion in the
# bead-labelling pipeline, so adjacent beads no longer share voxels. Dilate
# each label by DILATE_VOX to recover contact. Typical value 2-3.
DILATE_VOX = 3

# -- Import SPAM ---------------------------------------------------------------
import spam.label as slbl

# -- Helpers -------------------------------------------------------------------
def load_and_dilate(path):
    print(f'Loading {path}')
    lab = tifffile.imread(path).astype(np.int32)
    print(f'  shape={lab.shape}  max label={lab.max()}')
    # Dilate each label (binary dilation on whole, then re-assign label ids by
    # nearest label). Cheap approximation: do grayscale dilation which grows
    # each label into the background.
    if DILATE_VOX > 0:
        print(f'  dilating labels by {DILATE_VOX} voxels...')
        # For each foreground voxel, find the nearest foreground label within
        # DILATE_VOX voxels. `scipy.ndimage.grey_dilation` on int array does
        # exactly this with a cubic structuring element.
        lab = ndimage.grey_dilation(lab, size=(2 * DILATE_VOX + 1,) * 3)
    return lab

def run_contacts(lab, label_name):
    """Run labelledContacts; return (Z, pairs_sorted)."""
    print(f'Running labelledContacts on {label_name}...')
    contactVolume, Z, contactTable, pairs = slbl.labelledContacts(
        lab, maximumCoordinationNumber=20)
    # pairs is (n_contacts, 2). Sort each row so (a,b) canonical a<b.
    pairs_sorted = np.sort(pairs, axis=1).astype(int)
    # Drop pairs containing label 0 (background)
    keep = (pairs_sorted[:, 0] > 0) & (pairs_sorted[:, 1] > 0)
    pairs_sorted = pairs_sorted[keep]
    pairs_sorted = np.unique(pairs_sorted, axis=0)
    print(f'  contacts found: {len(pairs_sorted)}   max coord. number: {Z.max()}')
    return Z, pairs_sorted, contactVolume

def centroids_from_labels(lab):
    """Return (max_label+1, 3) array of centroids in (x, y, z)."""
    max_lab = int(lab.max())
    centroids = ndimage.center_of_mass(lab > 0, lab, range(1, max_lab + 1))
    # center_of_mass returns (z, y, x) — convert to (x, y, z)
    arr = np.zeros((max_lab + 1, 3))
    for i, c in enumerate(centroids, start=1):
        arr[i] = [c[2], c[1], c[0]]
    return arr

# -- (1) Contact detection on both scans ---------------------------------------
print('=' * 60)
print('(1) CONTACT DETECTION')
print('=' * 60)

lab1 = load_and_dilate(LAB1_TIF)
Z1, pairs1, _ = run_contacts(lab1, 'scan 1')
cent1 = centroids_from_labels(lab1)
del lab1

lab2 = load_and_dilate(LAB2_TIF)
Z2, pairs2, _ = run_contacts(lab2, 'scan 2')
cent2 = centroids_from_labels(lab2)
del lab2

print(f'\nScan 1: {len(pairs1)} bonds, median coord number = {np.median(Z1[Z1>0]):.1f}')
print(f'Scan 2: {len(pairs2)} bonds, median coord number = {np.median(Z2[Z2>0]):.1f}')

# Save per-scan bond lists
for name, pairs, cent in [('01', pairs1, cent1), ('02', pairs2, cent2)]:
    csv = os.path.join(OUT_DIR, f'bonds_scan{name}.csv')
    with open(csv, 'w') as f:
        f.write('bond_id;labelA;labelB;cx_A;cy_A;cz_A;cx_B;cy_B;cz_B;'
                'dist_vox;dist_um\n')
        for i, (a, b) in enumerate(pairs):
            if a >= len(cent) or b >= len(cent):
                continue
            cA = cent[a]; cB = cent[b]
            d = np.linalg.norm(cA - cB)
            f.write(f'{i};{a};{b};'
                    f'{cA[0]:.2f};{cA[1]:.2f};{cA[2]:.2f};'
                    f'{cB[0]:.2f};{cB[1]:.2f};{cB[2]:.2f};'
                    f'{d:.3f};{d*VOXEL_SIZE_UM:.1f}\n')
    print(f'Saved: {csv}')

# -- (2) Per-bond strain (using DDIC displacements) ----------------------------
print('\n' + '=' * 60)
print('(2) PER-BOND STRAIN')
print('=' * 60)

with open(DDIC_TSV) as f:
    header = f.readline().strip().split('\t')
data = np.loadtxt(DDIC_TSV, skiprows=1, delimiter='\t')
col = {h: i for i, h in enumerate(header)}
beads = data[data[:, col['Label']].astype(int) > 0]
# Per-label displacement from DDIC (relative to scan1 bead frame)
# Note: DDIC labels are the scan-1 aligned labels (0..864 slice frame), so
# the bead-center positions in DDIC file use aligned coords while our
# lab1 uses unaligned scan1 coords (864 slices). That's fine for matching
# label ID, but NOT for matching centroids — use DDIC positions for strain.
ddic_Zpos = beads[:, col['Zpos']]
ddic_Ypos = beads[:, col['Ypos']]
ddic_Xpos = beads[:, col['Xpos']]
ddic_u    = np.column_stack([beads[:, col['Xdisp']],
                             beads[:, col['Ydisp']],
                             beads[:, col['Zdisp']]])
ddic_lab  = beads[:, col['Label']].astype(int)

# Build per-label displacement lookup (label -> (u_x, u_y, u_z))
u_by_label = {int(l): u for l, u in zip(ddic_lab, ddic_u)}
pos_by_label_ddic = {int(l): (x, y, z) for l, x, y, z in
                     zip(ddic_lab, ddic_Xpos, ddic_Ypos, ddic_Zpos)}

# Note: DDIC positions are in the ALIGNED (874-slice) frame, where scan 1
# was placed at slices 10..873. Original cent1 is in the NATIVE (864-slice)
# frame. For bond strain, the offset cancels when we take distances.

# For each scan-1 bond, compute:
#   d1 = distance in reference config (scan 1)
#   d2 = distance in deformed config (scan 1 + u from DDIC)
#   strain = (d2 - d1) / d1
# If either bead has no DDIC entry, strain = NaN (bond untrackable).
print('Computing bond strain from DDIC displacements...')
bond_strain = np.full(len(pairs1), np.nan)
d1_arr = np.full(len(pairs1), np.nan)
d2_arr = np.full(len(pairs1), np.nan)
for i, (a, b) in enumerate(pairs1):
    if a not in u_by_label or b not in u_by_label:
        continue
    pA = np.array(pos_by_label_ddic[a])
    pB = np.array(pos_by_label_ddic[b])
    uA = u_by_label[a]
    uB = u_by_label[b]
    if np.any(~np.isfinite(uA)) or np.any(~np.isfinite(uB)):
        continue
    d1 = np.linalg.norm(pA - pB)
    d2 = np.linalg.norm((pA + uA) - (pB + uB))
    d1_arr[i] = d1
    d2_arr[i] = d2
    bond_strain[i] = (d2 - d1) / d1 if d1 > 0 else np.nan

# -- (2a) Broken / new bonds --------------------------------------------------
# A scan-1 pair (a, b) is BROKEN if (a, b) is NOT in scan-2 pairs.
# A scan-2 pair is NEW if not in scan-1.
set1 = set(map(tuple, pairs1.tolist()))
set2 = set(map(tuple, pairs2.tolist()))
broken = np.array([int(tuple(p) not in set2) for p in pairs1])
new_bonds = set2 - set1
surviving = set1 & set2
n_surv = len(surviving)
n_broken = len(set1 - set2)
n_new = len(new_bonds)
print(f'  Surviving bonds : {n_surv}')
print(f'  Broken bonds    : {n_broken}')
print(f'  New bonds       : {n_new}')

# -- Write comparison CSV -----------------------------------------------------
comp_csv = os.path.join(OUT_DIR, 'bonds_comparison.csv')
with open(comp_csv, 'w') as f:
    f.write('bond_id;labelA;labelB;d1_vox;d2_vox;strain;broken;'
            'nx_A;ny_A;nz_A;nx_B;ny_B;nz_B;normal_dz\n')
    for i, (a, b) in enumerate(pairs1):
        if a >= len(cent1) or b >= len(cent1):
            continue
        cA = cent1[a]; cB = cent1[b]
        normal = (cB - cA)
        nrm = np.linalg.norm(normal)
        if nrm > 0:
            normal = normal / nrm
        f.write(f'{i};{a};{b};'
                f'{d1_arr[i]:.3f};{d2_arr[i] if np.isfinite(d2_arr[i]) else np.nan:.3f};'
                f'{bond_strain[i] if np.isfinite(bond_strain[i]) else np.nan:.6f};'
                f'{broken[i]};'
                f'{cA[0]:.2f};{cA[1]:.2f};{cA[2]:.2f};'
                f'{cB[0]:.2f};{cB[1]:.2f};{cB[2]:.2f};'
                f'{normal[2]:.3f}\n')
print(f'Saved: {comp_csv}')

# -- (3) Orientation statistics ------------------------------------------------
print('\n' + '=' * 60)
print('(3) ORIENTATION STATISTICS')
print('=' * 60)

def contact_inclinations(pairs, cent):
    """Inclination in degrees: angle between contact normal and Z axis.
    Uses absolute value so 0 = horizontal, 90 = vertical."""
    incls = []
    for a, b in pairs:
        if a >= len(cent) or b >= len(cent):
            continue
        v = cent[b] - cent[a]
        n = np.linalg.norm(v)
        if n == 0:
            continue
        v = v / n
        # Angle to +Z axis: cos(theta) = |v_z|
        theta_deg = np.degrees(np.arccos(np.clip(abs(v[2]), 0, 1)))
        incls.append(90 - theta_deg)   # 0 = horizontal, 90 = along Z axis
    return np.array(incls)

incl1 = contact_inclinations(pairs1, cent1)
incl2 = contact_inclinations(pairs2, cent2)
print(f'Mean inclination vs horizontal -- scan 1: {incl1.mean():.1f}°  scan 2: {incl2.mean():.1f}°')
print(f'  (Compression should tilt contacts toward vertical -> higher values)')

# -- Summary text -------------------------------------------------------------
sum_path = os.path.join(OUT_DIR, 'bonds_summary.txt')
with open(sum_path, 'w') as f:
    f.write('BOND ANALYSIS SUMMARY (scan 1 vs scan 2)\n')
    f.write('=' * 60 + '\n\n')
    f.write(f'Dilation before contact detection: {DILATE_VOX} voxels\n\n')
    f.write(f'[Scan 1] bonds = {len(pairs1)},  mean Z = {Z1[Z1>0].mean():.2f},  median = {np.median(Z1[Z1>0]):.1f}\n')
    f.write(f'[Scan 2] bonds = {len(pairs2)},  mean Z = {Z2[Z2>0].mean():.2f},  median = {np.median(Z2[Z2>0]):.1f}\n\n')
    f.write(f'Surviving bonds   : {n_surv}  ({100*n_surv/max(len(pairs1),1):.1f}% of scan-1)\n')
    f.write(f'Broken bonds      : {n_broken}  ({100*n_broken/max(len(pairs1),1):.1f}% of scan-1)\n')
    f.write(f'New bonds         : {n_new}  ({100*n_new/max(len(pairs2),1):.1f}% of scan-2)\n\n')
    fin = np.isfinite(bond_strain)
    if fin.any():
        bs = bond_strain[fin]
        f.write(f'Per-bond strain (scan-1 bonds with DDIC data, n={fin.sum()}):\n')
        f.write(f'  min    : {bs.min():+.4f}\n')
        f.write(f'  median : {np.median(bs):+.4f}\n')
        f.write(f'  max    : {bs.max():+.4f}\n')
        f.write(f'  std    : {bs.std():.4f}\n\n')
    f.write(f'Contact inclination vs horizontal (higher = more vertical):\n')
    f.write(f'  scan 1: mean {incl1.mean():.1f}° median {np.median(incl1):.1f}°\n')
    f.write(f'  scan 2: mean {incl2.mean():.1f}° median {np.median(incl2):.1f}°\n')
    f.write(f'  delta : {incl2.mean() - incl1.mean():+.1f}° (positive = contacts rotated toward vertical under compression)\n')
print(f'Saved: {sum_path}')

# -- Plots ---------------------------------------------------------------------
# Project plot style
_FS = 24
plt.rcParams.update({
    "font.size": _FS, "axes.titlesize": _FS, "axes.labelsize": _FS + 2,
    "xtick.labelsize": _FS - 2, "ytick.labelsize": _FS - 2,
    "legend.fontsize": _FS - 2, "axes.linewidth": 1.8,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
LEG_BBOX = (0.5, -0.14)

# (P1) Coordination-number histogram
fig, ax = plt.subplots(figsize=(14, 9))
Z1_vals = Z1[Z1 > 0]
Z2_vals = Z2[Z2 > 0]
bins = np.arange(0, max(Z1_vals.max(), Z2_vals.max()) + 2) - 0.5
ax.hist(Z1_vals, bins=bins, color='#1f77b4', alpha=0.55, label='Scan 1', edgecolor='black')
ax.hist(Z2_vals, bins=bins, color='#d62728', alpha=0.55, label='Scan 2', edgecolor='black')
ax.axvline(Z1_vals.mean(), color='#1f77b4', linestyle='--', linewidth=2)
ax.axvline(Z2_vals.mean(), color='#d62728', linestyle='--', linewidth=2)
ax.set_xlabel('Coordination number Z', labelpad=10)
ax.set_ylabel('Bead count', labelpad=10)
ax.grid(True, alpha=0.3)
fig.legend(loc='lower center', ncol=2, bbox_to_anchor=LEG_BBOX,
           fontsize=_FS - 2, frameon=True, framealpha=0.85, edgecolor='gray')
plt.title('Coordination Number Distribution', pad=18, fontweight='bold')
plt.tight_layout()
out = os.path.join(OUT_DIR, 'bonds_coordination_hist.png')
fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'Saved: {out}')

# (P2) Contact orientation (rose: angle to horizontal)
fig, ax = plt.subplots(figsize=(14, 9))
bins_ang = np.linspace(0, 90, 19)
ax.hist(incl1, bins=bins_ang, color='#1f77b4', alpha=0.55,
        label=f'Scan 1  (mean {incl1.mean():.1f}°)', edgecolor='black')
ax.hist(incl2, bins=bins_ang, color='#d62728', alpha=0.55,
        label=f'Scan 2  (mean {incl2.mean():.1f}°)', edgecolor='black')
ax.set_xlabel('Contact inclination vs horizontal (°)', labelpad=10)
ax.set_ylabel('Bond count', labelpad=10)
ax.grid(True, alpha=0.3)
fig.legend(loc='lower center', ncol=2, bbox_to_anchor=LEG_BBOX,
           fontsize=_FS - 2, frameon=True, framealpha=0.85, edgecolor='gray')
plt.title('Contact Orientation — scan 1 vs scan 2', pad=18, fontweight='bold')
plt.tight_layout()
out = os.path.join(OUT_DIR, 'bonds_rose_inclination.png')
fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'Saved: {out}')

# (P3) Bond-strain histogram + broken fraction
fig, ax = plt.subplots(figsize=(14, 9))
if fin.any():
    bs = bond_strain[fin]
    bs_clip = np.clip(bs, -0.5, 0.5)
    ax.hist(bs_clip, bins=50, color='#2ca02c', edgecolor='black', alpha=0.75,
            label=f'Per-bond strain (n={fin.sum()})')
    ax.axvline(0, color='black', linestyle=':')
    ax.axvline(np.median(bs), color='red', linestyle='--', linewidth=2,
               label=f'median = {np.median(bs):+.3f}')
ax.set_xlabel('Bond strain (d2 − d1) / d1', labelpad=10)
ax.set_ylabel('Bond count', labelpad=10)
ax.grid(True, alpha=0.3)
fig.legend(loc='lower center', ncol=2, bbox_to_anchor=LEG_BBOX,
           fontsize=_FS - 2, frameon=True, framealpha=0.85, edgecolor='gray')
plt.title(f'Bond Strain Distribution  ({n_broken} broken, {n_new} new)',
          pad=18, fontweight='bold')
plt.tight_layout()
out = os.path.join(OUT_DIR, 'bonds_strain_hist.png')
fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f'Saved: {out}')

# (P4) 3D bond network (scan 1) — beads as faded spheres + bond cylinders
print('Rendering 3D bond network...')
# Bead centroids — use scan 1 native, not the DDIC-aligned coords, for
# self-consistency with pairs1 which are labelled from lab1.
valid_labs = np.unique(pairs1.ravel())
valid_labs = valid_labs[valid_labs > 0]
bead_pts = cent1[valid_labs]
bead_pd = pv.PolyData(bead_pts.astype(np.float32))

# Make bond line segments
lines_pts = []
lines_cells = []
for i, (a, b) in enumerate(pairs1):
    if a >= len(cent1) or b >= len(cent1):
        continue
    lines_pts.append(cent1[a])
    lines_pts.append(cent1[b])
    idx = len(lines_pts) - 2
    lines_cells.append([2, idx, idx + 1])
lines_pts = np.asarray(lines_pts, dtype=np.float32)
lines_cells = np.asarray(lines_cells).ravel().astype(np.int64)
bond_pd = pv.PolyData()
bond_pd.points = lines_pts
bond_pd.lines = lines_cells
# Per-bond scalars: broken flag, strain
bond_strain_full = np.repeat(bond_strain, 2)
broken_full      = np.repeat(broken, 2).astype(float)
bond_pd['broken'] = broken_full
bond_pd['strain'] = bond_strain_full

bond_tubes = bond_pd.tube(radius=2.0, n_sides=8)

nx_est, ny_est, nz_est = 578, 587, 864
p = pv.Plotter(off_screen=True, window_size=(2400, 1800))
p.set_background('white')
p.enable_anti_aliasing('msaa', multi_samples=8)
# Faded bead spheres
bead_spheres = bead_pd.glyph(
    geom=pv.Sphere(radius=32, theta_resolution=20, phi_resolution=20),
    scale=False, orient=False)
p.add_mesh(bead_spheres, color='#cdd3dd', opacity=0.22,
           smooth_shading=True, ambient=0.4, diffuse=0.6)
# Bond tubes, colored by broken vs kept
p.add_mesh(bond_tubes, scalars='broken', cmap=['#2ca02c', '#d62728'],
           clim=(0, 1), show_scalar_bar=False)
# Add manual legend
p.add_legend(labels=[('Surviving', '#2ca02c'), ('Broken', '#d62728')],
             size=(0.22, 0.12), loc='upper right', face='rectangle',
             bcolor='#eeeeee', border=True)
p.camera_position = [(nx_est * 2.1, -ny_est * 1.5, nz_est * 1.7),
                     (nx_est / 2, ny_est / 2, nz_est / 2),
                     (0, 0, 1)]
out = os.path.join(OUT_DIR, 'bonds_network_iso.png')
p.screenshot(out); p.close()
print(f'Saved: {out}')

print('\nDONE. Bond analysis outputs in:', OUT_DIR)
print('Open bonds_summary.txt for the headline numbers.')
