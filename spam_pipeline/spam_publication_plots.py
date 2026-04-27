"""Publication-quality plots for the per-specimen pipeline.

Every plot is a SEPARATE PNG (no multi-panel dashboards — multi-panel is not
acceptable for paper-grade output). Legends always BELOW axes. Large fonts.

Reads CSVs produced by:
  - spam_<tag>_summary.py            (overall counts, displacements)
  - spam_<tag>_failure_modes.py      (CRUSH / DELAM / COH counts)
  - spam_<tag>_microstructure.py     (per-scan microstructure)
  - spam_<tag>_crack_analysis.py     (per-transition damage classes)

Outputs to results_<tag>/publication/   (one PNG per quantity):
    phase_composition.png
    ssa_evolution.png
    fabric_anisotropy.png
    percolation.png
    k_eff.png
    mil_ice.png
    failure_modes_per_transition.png
    damage_classes_per_transition.png
    damage_volume_per_transition.png
    headline_summary.txt

Run after roughness_scan01 + microstructure + crack_analysis:
    python /mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/spam_publication_plots.py
"""
import os, sys, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

DATA = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
RES  = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>'
OUT  = os.path.join(RES, 'publication')
os.makedirs(OUT, exist_ok=True)

SPECIMEN_LABEL = '<SPECIMEN>'
TRANSITIONS = [(1, 2), (2, 3)]
ALL_SCANS = sorted({s for ab in TRANSITIONS for s in ab})

# ============================ STRICT plot defaults ============================
_FS = 24
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        _FS,
    "axes.titlesize":   _FS + 2,
    "axes.titleweight": "bold",
    "axes.labelsize":   _FS + 4,
    "axes.labelweight": "bold",
    "axes.linewidth":   2.2,
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.edgecolor":   "black",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.color":       "#d0d0d0",
    "grid.linewidth":   1.0,
    "xtick.labelsize":  _FS,
    "ytick.labelsize":  _FS,
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.major.pad":   8,
    "ytick.major.pad":   8,
    "xtick.direction":  "out",
    "ytick.direction":  "out",
    "legend.fontsize":  _FS,
    "legend.frameon":   False,
    "figure.facecolor": "white",
    "figure.dpi":       110,
    "savefig.dpi":      260,
    "savefig.bbox":     "tight",
    "savefig.facecolor": "white",
})

def save_with_legend_below(fig, ax, path, legend_ncol=None, anchor_y=-0.14):
    """Place legend BELOW the plot and save."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ncol = legend_ncol if legend_ncol is not None else min(len(handles), 4)
        fig.legend(handles, labels, loc='lower center',
                   bbox_to_anchor=(0.5, anchor_y), ncol=ncol,
                   fontsize=_FS - 2, frameon=False)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')

# Color palette
COL_AIR    = '#9aa0a6'
COL_ICE    = '#4878D0'
COL_GLASS  = '#D65F5F'
COL_GROW   = '#54A24B'
COL_HIGH   = '#F58518'

CLASS_COLOR = {
    'CRACK':        '#D62728',
    'DELAMINATION': '#FF7F0E',
    'ICE-FRACTURE': '#1F77B4',
    'CAVITY':       '#2CA02C',
}
MODE_COLOR = {
    'CRUSH':   '#2CA02C',
    'COH':     '#1F77B4',
    'DELAM':   '#D62728',
    'MIXED':   '#FF7F0E',
    'UNCLEAR': '#A0A0A0',
}

# ============================ CSV loaders ====================================
def safe_read_csv(path, sep=';'):
    if not os.path.exists(path):
        return None
    try: return pd.read_csv(path, sep=sep)
    except Exception: return None

def parse_failure_modes_summary(path):
    if not os.path.exists(path): return None
    rows = []
    for ln in open(path):
        s = ln.strip()
        if '->' in s and not s.startswith('#'):
            parts = s.split()
            try:
                a, b = parts[0].split('->')
                rows.append(dict(transition=f'{a}->{b}',
                                 total=int(parts[1]), delam=int(parts[2]),
                                 cohes=int(parts[3]), crush=int(parts[4]),
                                 mixed=int(parts[5]), unclr=int(parts[6])))
            except Exception: pass
    return pd.DataFrame(rows) if rows else None

print(f'Publication plots — {SPECIMEN_LABEL}')
ms_df  = safe_read_csv(os.path.join(RES, 'microstructure', 'summary.csv'))
fm_df  = parse_failure_modes_summary(os.path.join(RES, 'failure_modes_summary.txt'))
dmg_dfs = {}
for a, b in TRANSITIONS:
    df = safe_read_csv(os.path.join(RES, 'crack_analysis',
                                    f'transition_{a}to{b}', 'damage.csv'))
    if df is not None: dmg_dfs[(a, b)] = df
rough_df = safe_read_csv(os.path.join(RES, 'roughness_scan01', 'roughness_per_bead.csv'))

print(f'  microstructure: {"yes" if ms_df is not None else "MISSING"}')
print(f'  failure_modes : {"yes" if fm_df is not None else "MISSING"}')
print(f'  damage CSVs   : {len(dmg_dfs)} transitions')

# ============================ MICROSTRUCTURE PLOTS ===========================
if ms_df is not None and len(ms_df) >= 1:
    scans = ms_df['scan'].values

    # Phase composition (stacked area)
    fig, ax = plt.subplots(figsize=(13, 8))
    f_air, f_ice, f_glass = ms_df['f_air'].values, ms_df['f_ice'].values, ms_df['f_glass'].values
    ax.fill_between(scans, 0, f_glass, color=COL_GLASS, alpha=0.85, label='Glass beads')
    ax.fill_between(scans, f_glass, f_glass+f_ice, color=COL_ICE, alpha=0.85, label='Ice')
    ax.fill_between(scans, f_glass+f_ice, f_glass+f_ice+f_air, color=COL_AIR, alpha=0.85, label='Air')
    ax.set_xlabel('Scan #'); ax.set_ylabel('Phase volume fraction')
    ax.set_title(f'{SPECIMEN_LABEL} — phase composition')
    ax.set_xticks(scans); ax.set_ylim(0, 1.0)
    save_with_legend_below(fig, ax, os.path.join(OUT, 'phase_composition.png'),
                           legend_ncol=3)

    # SSA evolution (ice and glass)
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.plot(scans, ms_df['SSA_ice'] * 1e3, 'o-', color=COL_ICE, markersize=18,
            linewidth=3, markerfacecolor='white', markeredgewidth=3, label='Ice')
    ax.plot(scans, ms_df['SSA_glass'] * 1e3, 's-', color=COL_GLASS, markersize=18,
            linewidth=3, markerfacecolor='white', markeredgewidth=3, label='Glass')
    ax.set_xlabel('Scan #'); ax.set_ylabel('SSA  [10⁻³ µm⁻¹]')
    ax.set_title(f'{SPECIMEN_LABEL} — specific surface area')
    ax.set_xticks(scans)
    save_with_legend_below(fig, ax, os.path.join(OUT, 'ssa_evolution.png'),
                           legend_ncol=2)

    # Fabric anisotropy
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.bar(scans, ms_df['fabric_anisotropy'], color=COL_HIGH,
           edgecolor='black', linewidth=1.5, width=0.6)
    ax.set_xlabel('Scan #'); ax.set_ylabel('Anisotropy index a')
    ax.set_title(f'{SPECIMEN_LABEL} — contact-fabric anisotropy')
    ax.set_xticks(scans)
    ax.set_ylim(0, max(0.1, ms_df['fabric_anisotropy'].max() * 1.25))
    save_with_legend_below(fig, ax, os.path.join(OUT, 'fabric_anisotropy.png'))

    # Percolation per phase (grouped bars)
    fig, ax = plt.subplots(figsize=(13, 8))
    bw = 0.27
    for i, (col, lbl, c) in enumerate([
            ('perc_air',   'Air',   COL_AIR),
            ('perc_ice',   'Ice',   COL_ICE),
            ('perc_glass', 'Glass', COL_GLASS)]):
        vals = ms_df[col].astype(str).str.strip().str.lower().map(
            {'true': 1, 'false': 0}).fillna(0).values
        ax.bar(scans + (i - 1) * bw, vals, bw, color=c, edgecolor='black',
               linewidth=1.2, label=lbl)
    ax.set_xlabel('Scan #'); ax.set_ylabel('Percolates Z?')
    ax.set_yticks([0, 1]); ax.set_yticklabels(['No', 'Yes'])
    ax.set_title(f'{SPECIMEN_LABEL} — phase percolation top↔bottom')
    ax.set_xticks(scans)
    save_with_legend_below(fig, ax, os.path.join(OUT, 'percolation.png'),
                           legend_ncol=3)

    # Effective thermal conductivity
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.fill_between(scans, ms_df['k_eff_ser'], ms_df['k_eff_par'],
                    color=COL_GROW, alpha=0.20, label='Wiener bounds')
    ax.plot(scans, ms_df['k_eff_ME'], 'D-', color=COL_GROW, markersize=18,
            linewidth=3, markerfacecolor='white', markeredgewidth=3,
            label='Maxwell-Eucken')
    ax.set_xlabel('Scan #'); ax.set_ylabel('k_eff  [W m⁻¹ K⁻¹]')
    ax.set_title(f'{SPECIMEN_LABEL} — effective thermal conductivity')
    ax.set_xticks(scans)
    save_with_legend_below(fig, ax, os.path.join(OUT, 'k_eff.png'),
                           legend_ncol=2)

    # MIL ice (3 directions)
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.plot(scans, ms_df['MIL_ice_z'], 'o-', color='#1F77B4',
            linewidth=3, markersize=18, markerfacecolor='white',
            markeredgewidth=3, label='z direction')
    ax.plot(scans, ms_df['MIL_ice_y'], 's-', color='#7CB9E8',
            linewidth=3, markersize=18, markerfacecolor='white',
            markeredgewidth=3, label='y direction')
    ax.plot(scans, ms_df['MIL_ice_x'], '^-', color='#A0CFEC',
            linewidth=3, markersize=18, markerfacecolor='white',
            markeredgewidth=3, label='x direction')
    ax.set_xlabel('Scan #'); ax.set_ylabel('Ice MIL  [µm]')
    ax.set_title(f'{SPECIMEN_LABEL} — ice mean intercept length')
    ax.set_xticks(scans)
    save_with_legend_below(fig, ax, os.path.join(OUT, 'mil_ice.png'),
                           legend_ncol=3)

# ============================ FAILURE MODES PLOT =============================
if fm_df is not None:
    fig, ax = plt.subplots(figsize=(13, 8))
    transitions = fm_df['transition'].values
    bottom = np.zeros(len(fm_df))
    modes = [('crush', 'CRUSH'), ('cohes', 'COH'), ('delam', 'DELAM'),
             ('mixed', 'MIXED'), ('unclr', 'UNCLEAR')]
    for col, lbl in modes:
        vals = fm_df[col].values
        ax.bar(transitions, vals, bottom=bottom, label=lbl,
               color=MODE_COLOR[lbl], edgecolor='black', linewidth=1.2,
               width=0.55)
        bottom += vals
    ax.set_xlabel('Transition'); ax.set_ylabel('Bond count')
    ax.set_title(f'{SPECIMEN_LABEL} — bond failure modes')
    save_with_legend_below(fig, ax,
                           os.path.join(OUT, 'failure_modes_per_transition.png'),
                           legend_ncol=5, anchor_y=-0.18)

# ============================ DAMAGE CLASS PLOT ==============================
if dmg_dfs:
    dmg_classes = ['CRACK', 'DELAMINATION', 'ICE-FRACTURE', 'CAVITY']
    counts = {cls: [] for cls in dmg_classes}
    trans_lbls = []
    for (a, b), df in dmg_dfs.items():
        trans_lbls.append(f'{a}→{b}')
        for cls in dmg_classes:
            counts[cls].append(int((df['class'] == cls).sum()))

    fig, ax = plt.subplots(figsize=(13, 8))
    bottom = np.zeros(len(trans_lbls))
    for cls in dmg_classes:
        vals = np.array(counts[cls])
        ax.bar(trans_lbls, vals, bottom=bottom, label=cls,
               color=CLASS_COLOR[cls], edgecolor='black',
               linewidth=1.2, width=0.55)
        bottom += vals
    ax.set_xlabel('Transition'); ax.set_ylabel('Damage component count')
    ax.set_title(f'{SPECIMEN_LABEL} — damage components by class')
    save_with_legend_below(fig, ax,
                           os.path.join(OUT, 'damage_classes_per_transition.png'),
                           legend_ncol=4, anchor_y=-0.18)

# ============================ DAMAGE VOLUME PLOT =============================
if dmg_dfs:
    transitions = [f'{a}→{b}' for a, b in TRANSITIONS]
    total_dmg_mm3 = []
    full_crack_count = []
    for (a, b), df in dmg_dfs.items():
        total_dmg_mm3.append(df['V_um3'].sum() / 1e9)
        full_crack_count.append(
            int(df.get('fully_developed',
                       pd.Series([False] * len(df))).sum()))
    fig, ax = plt.subplots(figsize=(13, 8))
    # Color the bar red if any fully-developed crack exists; otherwise green.
    bar_colors = [('#D62728' if fc > 0 else COL_GROW) for fc in full_crack_count]
    ax.bar(transitions, total_dmg_mm3, color=bar_colors,
           edgecolor='black', linewidth=1.5, width=0.55, alpha=0.9)
    ax.set_xlabel('Transition'); ax.set_ylabel('Damage volume  [mm³]')
    ax.set_title(f'{SPECIMEN_LABEL} — damage volume per transition')
    save_with_legend_below(fig, ax,
                           os.path.join(OUT, 'damage_volume_per_transition.png'))

# ============================ headline_summary.txt ===========================
hl = os.path.join(OUT, 'headline_summary.txt')
with open(hl, 'w') as f:
    f.write(f'Headline summary — {SPECIMEN_LABEL}\n')
    f.write('=' * (24 + len(SPECIMEN_LABEL)) + '\n\n')
    if ms_df is not None and len(ms_df) >= 2:
        s0, sN = ms_df.iloc[0], ms_df.iloc[-1]
        f.write('Phase composition (scan 1 → final):\n')
        for ph in ('air', 'ice', 'glass'):
            v0, vN = s0[f'f_{ph}'] * 100, sN[f'f_{ph}'] * 100
            f.write(f'  {ph:<5}  {v0:6.2f}% → {vN:6.2f}%   '
                    f'(Δ {(vN - v0):+5.2f}%)\n')
        f.write(f'\nFabric anisotropy: {s0["fabric_anisotropy"]:.4f} → '
                f'{sN["fabric_anisotropy"]:.4f}\n')
        f.write(f'Ice SSA:           {s0["SSA_ice"]*1e3:.3f} → '
                f'{sN["SSA_ice"]*1e3:.3f}  ×10⁻³ µm⁻¹\n')
        f.write(f'k_eff (Maxwell-Eucken): {s0["k_eff_ME"]:.3f} → '
                f'{sN["k_eff_ME"]:.3f}  W m⁻¹ K⁻¹\n')
    if fm_df is not None:
        f.write('\nBond failure totals:\n')
        for _, row in fm_df.iterrows():
            f.write(f'  {row["transition"]:<6}  total={row["total"]:>4}   '
                    f'CRUSH={row["crush"]:>3}  COH={row["cohes"]:>3}  '
                    f'DELAM={row["delam"]:>3}\n')
    if dmg_dfs:
        f.write('\nDamage components per transition:\n')
        for (a, b), df in dmg_dfs.items():
            classes = ['CRACK', 'DELAMINATION', 'ICE-FRACTURE', 'CAVITY']
            counts_str = ', '.join(f"{c}={int((df['class']==c).sum())}"
                                    for c in classes)
            f.write(f'  {a}→{b}: V_total = {df["V_um3"].sum()/1e9:.4f} mm³,  '
                    f'{counts_str}\n')
            n_full = int(df.get('fully_developed',
                                pd.Series([False] * len(df))).sum())
            if n_full > 0:
                f.write(f'           ⚠ {n_full} fully-developed crack(s)\n')
    if rough_df is not None and 'eq_diam_um' in rough_df:
        f.write(f'\nBead D50 (scan 1):  {rough_df["eq_diam_um"].median():.0f} µm    '
                f'(N = {len(rough_df)})\n')
        if 'Ra_um' in rough_df:
            f.write(f'Bead Ra (median):   {rough_df["Ra_um"].median():.2f} µm\n')
print(f'wrote {hl}')
print('Done.')
