"""Aggregate T7 SPAM results into a single summary.

Produces in <SPAM>/results_<PRE>/:
  SUMMARY.txt                       headline numbers
  summary_bond_evolution.png        surviving/broken/new per transition
  summary_contact_evolution.png     contact count + coord number vs scan
  summary_displacement.png          median |u| vs transition
  cumulative_1to4.csv               cumulative displacement for beads that
                                    tracked through ALL three transitions

Run in WSL spam-venv:
    python /mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/spam_summary.py
"""
import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>'
RES = '/home/cak7496/spam-results-75'
VOXEL_UM = 24.7660229

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

# ---- 1) Ice-bond status per transition ------------------------------------
def read_ice_bonds_change(tdir):
    path = os.path.join(tdir, 'ice_bonds_change.csv')
    surviving = weakened = broken = unmapped = 0
    if not os.path.exists(path): return None
    with open(path) as f:
        next(f)
        for ln in f:
            parts = ln.strip().split(';')
            st = parts[-1]
            if st == 'surviving': surviving += 1
            elif st == 'weakened': weakened += 1
            elif st == 'broken': broken += 1
            elif st == 'unmapped': unmapped += 1
    return dict(surviving=surviving, weakened=weakened, broken=broken, unmapped=unmapped)

# new bonds = count in destination set not in mapped-a set (not in the CSV).
# approximated via per-scan counts.
def count_ice_bonds(s):
    path = os.path.join(OUT, f'ice_bonds_scan{s:02d}.csv')
    if not os.path.exists(path): return 0
    with open(path) as f:
        next(f)
        return sum(1 for _ in f)

TRANSITIONS = [(1, 2), (2, 3)]
ALL_SCANS = sorted({s for ab in TRANSITIONS for s in ab})
bond_change = {}
for a, b in TRANSITIONS:
    bond_change[(a, b)] = read_ice_bonds_change(os.path.join(OUT, f'transition_{a}to{b}'))

ice_totals = {s: count_ice_bonds(s) for s in ALL_SCANS}

# Bar plot: bond evolution across 3 transitions
fig, ax = plt.subplots(figsize=(14, 9))
x = np.arange(len(TRANSITIONS))
cats = ['surviving', 'weakened', 'broken']
colors = {'surviving': '#2ca02c', 'weakened': '#ff7f0e', 'broken': '#d62728'}
width = 0.25
for i, c in enumerate(cats):
    vals = [bond_change[t][c] if bond_change[t] else 0 for t in TRANSITIONS]
    ax.bar(x + (i - 1) * width, vals, width, color=colors[c],
           edgecolor='black', label=c)
ax.set_xticks(x)
ax.set_xticklabels([f'{a}->{b}' for a, b in TRANSITIONS])
ax.set_ylabel('Ice-bond count', labelpad=10)
ax.set_title('Ice bond evolution across 3 compression steps',
             fontweight='bold', pad=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'summary_bond_evolution.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# ---- 2) Contact counts + coordination per scan -----------------------------
import csv as _csv
contact_counts = {s: 0 for s in ALL_SCANS}
for s in ALL_SCANS:
    p = os.path.join(OUT, f'contacts_scan{s:02d}.csv')
    if os.path.exists(p):
        with open(p) as f: next(f); contact_counts[s] = sum(1 for _ in f)

fig, ax1 = plt.subplots(figsize=(14, 9))
sc = [1, 2]
cc = [contact_counts[s] for s in sc]
ax1.plot(sc, cc, 'o-', color='#1f77b4', linewidth=3, markersize=12,
         markeredgecolor='black', label='Total contacts')
ax1.set_xlabel('Scan number', labelpad=10)
ax1.set_ylabel('Contact count', color='#1f77b4', labelpad=10)
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax1.set_xticks(sc)

ax2 = ax1.twinx()
ic = [ice_totals[s] for s in sc]
ax2.plot(sc, ic, 's--', color='#d62728', linewidth=3, markersize=12,
         markeredgecolor='black', label='Ice bonds')
ax2.set_ylabel('Ice-bond count', color='#d62728', labelpad=10)
ax2.tick_params(axis='y', labelcolor='#d62728')
plt.title('Contact + ice-bond evolution across scans',
          fontweight='bold', pad=12)
ax1.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'summary_contact_evolution.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# ---- 3) Median displacement per transition ---------------------------------
def median_u_um(tag):
    tsv = os.path.join(RES, f'75_ddic_{tag}-ddic.tsv')
    if not os.path.exists(tsv): return None
    with open(tsv) as f: header = f.readline().strip().split('\t')
    data = np.loadtxt(tsv, skiprows=1, delimiter='\t')
    col = {h: i for i, h in enumerate(header)}
    beads = data[data[:, col['Label']].astype(int) > 0]
    zd, yd, xd = beads[:, col['Zdisp']], beads[:, col['Ydisp']], beads[:, col['Xdisp']]
    rs = beads[:, col['returnStatus']].astype(int)
    err = beads[:, col['error']]
    valid = (np.isfinite(zd) & np.isfinite(yd) & np.isfinite(xd) & (rs >= 1) & (err < 25000))
    if not valid.any(): return None
    u = np.sqrt(zd[valid]**2 + yd[valid]**2 + xd[valid]**2) * VOXEL_UM
    return dict(n=int(valid.sum()), median=float(np.median(u)),
                p90=float(np.percentile(u, 90)), max=float(u.max()))

disp_stats = {f'{a}{b}': median_u_um(f'{a}{b}') for a, b in TRANSITIONS}

fig, ax = plt.subplots(figsize=(14, 9))
labels = [f'{a}->{b}' for a, b in TRANSITIONS]
meds = [disp_stats[f'{a}{b}']['median'] if disp_stats[f'{a}{b}'] else 0 for a, b in TRANSITIONS]
p90s = [disp_stats[f'{a}{b}']['p90']    if disp_stats[f'{a}{b}'] else 0 for a, b in TRANSITIONS]
width = 0.35
x = np.arange(len(labels))
ax.bar(x - width/2, meds, width, color='#1f77b4', edgecolor='black',
       label='median |u|')
ax.bar(x + width/2, p90s, width, color='#ff7f0e', edgecolor='black',
       label='90th pct |u|')
for i, (m, p) in enumerate(zip(meds, p90s)):
    ax.text(i - width/2, m + max(p90s)*0.01, f'{m:.0f}', ha='center',
            fontsize=_FS - 4, fontweight='bold')
    ax.text(i + width/2, p + max(p90s)*0.01, f'{p:.0f}', ha='center',
            fontsize=_FS - 4, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('|u| (µm)', labelpad=10)
ax.set_title('Bead displacement magnitude per compression step',
             fontweight='bold', pad=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'summary_displacement.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# ---- 4) Cumulative 1→4 tracking --------------------------------------------
from scipy.spatial import cKDTree

def load_ddic(tag):
    tsv = os.path.join(RES, f'75_ddic_{tag}-ddic.tsv')
    with open(tsv) as f: header = f.readline().strip().split('\t')
    data = np.loadtxt(tsv, skiprows=1, delimiter='\t')
    col = {h: i for i, h in enumerate(header)}
    beads = data[data[:, col['Label']].astype(int) > 0]
    return beads, col

def build_lmap(beads, col, cent_b):
    lab = beads[:, col['Label']].astype(int)
    pred = np.column_stack([beads[:, col['Zpos']] + beads[:, col['Zdisp']],
                            beads[:, col['Ypos']] + beads[:, col['Ydisp']],
                            beads[:, col['Xpos']] + beads[:, col['Xdisp']]])
    rs = beads[:, col['returnStatus']].astype(int)
    err = beads[:, col['error']]
    valid = np.isfinite(pred).all(axis=1) & (rs >= 1) & (err < 25000)
    tree = cKDTree(cent_b[1:])
    dist, idx = tree.query(pred, k=1)
    lmap, pos = {}, {}
    for i in range(len(lab)):
        if valid[i] and dist[i] < 50:
            la = int(lab[i])
            lmap[la] = int(idx[i] + 1)
            pos[la] = tuple(pred[i])
    return lmap, pos

# Load bead_labels just for centroids
import tifffile
from scipy import ndimage
def cent_from_file(p):
    lab = tifffile.imread(p)
    m = int(lab.max())
    c = ndimage.center_of_mass(lab > 0, lab, range(1, m + 1))
    arr = np.zeros((m + 1, 3))
    for i, v in enumerate(c, 1): arr[i] = v
    return arr

DATA = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
print('loading centroids for scans 1 and 2 (from aligned label TIFFs)...')
cent_all = {s: cent_from_file(os.path.join(DATA, f'bead_labels_scan{s:02d}_aligned.tif'))
            for s in ALL_SCANS}

# T5_HR has only one transition so the "cumulative" track is just scan 1 -> 2.
b12, c12 = load_ddic('12')
lm12, pos12 = build_lmap(b12, c12, cent_all[2])
cent1 = cent_all[1]
cumulative = []
for l1, l2 in lm12.items():
    X1 = cent1[l1][::-1]
    X2 = cent_all[2][l2][::-1]
    u = np.linalg.norm(X2 - X1) * VOXEL_UM
    cumulative.append((l1, *X1, l2, *X2, u))
print(f'Cumulative 1->2 tracked beads: {len(cumulative)}')
csv_out = os.path.join(OUT, 'cumulative_1to2.csv')
with open(csv_out, 'w') as f:
    f.write('scan1_label;x1;y1;z1;scan2_label;x2;y2;z2;total_disp_um\n')
    for row in cumulative:
        f.write(';'.join(f'{v:g}' for v in row) + '\n')
print(f'wrote {csv_out}')

# ---- 5) Text summary -------------------------------------------------------
with open(os.path.join(OUT, 'SUMMARY.txt'), 'w') as f:
    f.write('T5_HR 1700 um SPAM analysis summary\n' + '=' * 50 + '\n\n')
    f.write('--- Per-scan counts ---\n')
    f.write(f"{'scan':<6}{'contacts':>12}{'ice_bonds':>14}\n")
    for s in ALL_SCANS:
        f.write(f'{s:<6}{contact_counts[s]:>12}{ice_totals[s]:>14}\n')
    f.write('\n--- Per-transition bead displacement ---\n')
    f.write(f"{'tran':<8}{'valid':>8}{'med_u_um':>10}{'p90_u_um':>10}{'max_u_um':>10}\n")
    for a, b in TRANSITIONS:
        k = f'{a}{b}'
        d = disp_stats.get(k)
        if d:
            f.write(f'{a}->{b:<5}{d["n"]:>8}{d["median"]:>10.0f}{d["p90"]:>10.0f}{d["max"]:>10.0f}\n')
    f.write('\n--- Ice-bond transitions (surviving / weakened / broken / unmapped) ---\n')
    for a, b in TRANSITIONS:
        c = bond_change[(a, b)]
        if c:
            f.write(f'{a}->{b}: S={c["surviving"]}  W={c["weakened"]}  B={c["broken"]}  U={c["unmapped"]}\n')
    f.write(f'\nCumulative 1->2 beads tracked: {len(cumulative)}\n')
print(f'wrote summary.txt')
print('\nDone. See', OUT)
