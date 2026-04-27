"""Fast replot — reads existing CSVs and regenerates all plots in the
corrected style (one plot per file, legend below, large fonts).

Use this AFTER spam_crack_analysis.py + spam_microstructure.py have
already produced their CSVs. Skips all heavy computation.

Run in WSL spam-venv:
    python /mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/spam_replot.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

DATA = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
RES  = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>'

SPECIMEN_LABEL = '<SPECIMEN>'
TRANSITIONS = [(1, 2), (2, 3)]

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

def save_with_legend_below(fig, ax, path, legend_ncol=None, anchor_y=-0.16):
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

CLASS_COLOR = {
    'CRACK':        '#D62728',
    'DELAMINATION': '#FF7F0E',
    'ICE-FRACTURE': '#1F77B4',
    'CAVITY':       '#2CA02C',
}
# Color per transition (cycles for >2 transitions)
TRANS_COLORS = ['#1F77B4', '#D62728', '#2CA02C', '#FF7F0E', '#9467BD']
PLANARITY_THRESH = 0.6

def safe_read_csv(path, sep=';'):
    if not os.path.exists(path):
        return None
    try: return pd.read_csv(path, sep=sep)
    except Exception: return None

# ============================================================================
# Crack plots: ONE merged plot per metric covering ALL transitions
# (per-transition CSVs and 3D renders stay in their transition_*/ folders;
#  per-transition PNGs of these metrics are removed)
# ============================================================================
print(f'Replotting crack-analysis figures (merged across transitions) for {SPECIMEN_LABEL}')
ca_dir = os.path.join(RES, 'crack_analysis')

# Load all transitions' CSVs once
df_per_trans = {}
for a, b in TRANSITIONS:
    csv_path = os.path.join(ca_dir, f'transition_{a}to{b}', 'damage.csv')
    df = safe_read_csv(csv_path)
    if df is None or 'class' not in df.columns:
        continue
    df = df[df['class'].isin(CLASS_COLOR.keys())].copy()
    df_per_trans[(a, b)] = df
    counts = {cls: int((df['class'] == cls).sum()) for cls in CLASS_COLOR}
    print(f'  {a}→{b}: counts {counts}')

if not df_per_trans:
    print('  no damage CSVs found, skipping merged crack plots')
else:
    # ----- merged class GROUPED bar (transitions side by side per class) ------
    fig, ax = plt.subplots(figsize=(13, 8))
    classes = list(CLASS_COLOR.keys())
    n_trans = len(df_per_trans)
    bw = 0.8 / max(n_trans, 1)
    x = np.arange(len(classes))
    for i, ((a, b), df) in enumerate(df_per_trans.items()):
        counts = [int((df['class'] == c).sum()) for c in classes]
        ax.bar(x + (i - (n_trans - 1) / 2) * bw, counts, bw,
               color=TRANS_COLORS[i % len(TRANS_COLORS)],
               edgecolor='black', linewidth=1.5,
               label=f'{a}→{b}')
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=15)
    ax.set_xlabel('Damage class')
    ax.set_ylabel('Component count')
    ax.set_title(f'{SPECIMEN_LABEL} — damage classes across transitions')
    ax.grid(True, alpha=0.3, axis='y')
    save_with_legend_below(fig, ax,
        os.path.join(ca_dir, 'damage_class_bar.png'),
        legend_ncol=n_trans)

    # ----- merged planarity histogram (overlaid) ------------------------------
    fig, ax = plt.subplots(figsize=(13, 8))
    bins = np.linspace(0, 1, 31)
    for i, ((a, b), df) in enumerate(df_per_trans.items()):
        if df.empty: continue
        ax.hist(df['planarity'].values, bins=bins, alpha=0.55,
                color=TRANS_COLORS[i % len(TRANS_COLORS)],
                edgecolor='black', linewidth=1.0,
                label=f'{a}→{b}')
    ax.axvline(PLANARITY_THRESH, color='black', ls='--', lw=3,
               label=f'CRACK threshold ({PLANARITY_THRESH})')
    ax.set_xlabel('Planarity index  (1 = plane, 0 = sphere)')
    ax.set_ylabel('Damage component count')
    ax.set_title(f'{SPECIMEN_LABEL} — damage planarity across transitions')
    ax.grid(True, alpha=0.3)
    save_with_legend_below(fig, ax,
        os.path.join(ca_dir, 'damage_planarity_hist.png'),
        legend_ncol=n_trans + 1)

    # ----- merged volume histogram (log-x, overlaid) --------------------------
    all_vols = np.concatenate([df.loc[df['V_um3'] > 0, 'V_um3'].values
                                for df in df_per_trans.values()])
    if len(all_vols) > 0:
        fig, ax = plt.subplots(figsize=(13, 8))
        bins = np.logspace(np.log10(max(all_vols.min(), 1.0)),
                           np.log10(max(all_vols.max(), 10.0)), 40)
        for i, ((a, b), df) in enumerate(df_per_trans.items()):
            sub = df.loc[df['V_um3'] > 0, 'V_um3'].values
            if len(sub) == 0: continue
            ax.hist(sub, bins=bins, alpha=0.55,
                    color=TRANS_COLORS[i % len(TRANS_COLORS)],
                    edgecolor='black', linewidth=1.0,
                    label=f'{a}→{b}')
        ax.set_xscale('log')
        ax.set_xlabel('Damage component volume [µm³]')
        ax.set_ylabel('Count')
        ax.set_title(f'{SPECIMEN_LABEL} — damage volume distribution across transitions')
        ax.grid(True, alpha=0.3, which='both')
        save_with_legend_below(fig, ax,
            os.path.join(ca_dir, 'damage_volume_hist.png'),
            legend_ncol=n_trans)

    # ----- cleanup: remove the per-transition versions of these metrics -------
    for a, b in df_per_trans:
        tdir = os.path.join(ca_dir, f'transition_{a}to{b}')
        for fn in ('damage_class_pie.png', 'damage_class_bar.png',
                   'damage_planarity_hist.png', 'damage_volume_hist.png'):
            p = os.path.join(tdir, fn)
            if os.path.exists(p):
                os.remove(p); print(f'  removed {p}')

# ============================================================================
# Microstructure S2 plots — regenerate from existing summary if needed
# (S2 plots are written inline inside microstructure.py and aren't in the CSV;
#  if the user wants them re-styled, re-run microstructure.py — quick because
#  it caches volumes from the aligned TIFFs.)
# ============================================================================

# ============================================================================
# Run publication_plots — fast (only reads CSVs)
# ============================================================================
print('\nRunning publication_plots (reads existing CSVs)…')
import importlib.util
pp = os.path.join(HERE, 'spam_publication_plots.py')
spec = importlib.util.spec_from_file_location('pp', pp)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

print('Replot done.')
