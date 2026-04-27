"""Directional tortuosity evolution under compression — crack-onset diagnostic.

Computes geodesic tortuosity τ along x, y, z for the air and ice phases,
per scan. The directional split is the key:

  - A horizontal crack opens a low-tortuosity path through the air phase
    along its plane normal direction. A sharp DROP in τ_x or τ_y for the air
    phase between scans flags a horizontal crack.
  - A vertical crack drops τ_z for the air phase.
  - Compaction (no crack) raises τ in all directions roughly together.
  - Loss of percolation gives τ = ∞ (no path); we record this as NaN with
    a `percolates_<dir>` flag.

Method:
  τ_dir = mean(geodesic_dist_through_phase_from_near_to_far_face) / L_dir

  Implementation uses skimage.graph.MCP_Geometric for true geodesic distance
  (handles arbitrary tortuous paths; chamfer-step BFS would underestimate).

Outputs to results_<PRE>/tortuosity_evolution/:
    summary.csv                per (scan, phase, direction): tau, percolates, vfrac
    summary.txt                human-readable
    tau_air_evolution.png      τ_x, τ_y, τ_z vs scan (air phase)
    tau_ice_evolution.png      τ_x, τ_y, τ_z vs scan (ice phase)
    tau_anisotropy.png         (τ_z - τ_x) and (τ_z - τ_y) — direction-asymmetry
    delta_tau_per_transition.png   bar chart of Δτ per transition × direction
    crack_indicator.png        flagged transitions with sharp Δτ drop

Run in WSL spam-venv (after the rest of the SPAM pipeline):
    python /mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/spam_tortuosity_evolution.py
"""
import os, sys, time, csv
import numpy as np
import tifffile
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.graph import MCP_Geometric

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from scan_meta import VOXEL_UM

DATA = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
OUT  = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>/tortuosity_evolution'
os.makedirs(OUT, exist_ok=True)

TRANSITIONS = [(1, 2), (2, 3)]
ALL_SCANS = sorted({s for ab in TRANSITIONS for s in ab})

# Per-scan phase thresholds (must match ice_bonds / failure_modes / microstructure)
TH = {
    1: dict(air=7000,    ig=18032.7, ga=47275),
    2: dict(air=2139.4,  ig=6271.5,  ga=99999),
    3: dict(air=2015.4,  ig=6539.5,  ga=99999),
}

# Downsample factor for tortuosity solve (3D MCP can be slow at full res).
# Tortuosity is dimensionless and largely insensitive to downsampling above
# ~5 voxels per phase feature.
DOWNSAMPLE = 3

# Sharp-Δτ threshold for the "crack indicator" plot.
CRACK_DELTA_TAU = 0.30   # |Δτ| > this between scans -> flagged

# Color palette
COL_X = '#E45756'
COL_Y = '#54A24B'
COL_Z = '#4878D0'
COL_AIR = '#9aa0a6'
COL_ICE = '#4878D0'
COL_FLAG = '#D62728'

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

def reconstruct_phases(ct, mask, th):
    p = np.zeros(ct.shape, dtype=np.uint8)
    inside = mask > 0
    p[inside & (ct < th['air'])] = 1
    p[inside & (ct >= th['air']) & (ct < th['ig'])] = 2
    p[inside & (ct >= th['ig'])  & (ct < th['ga'])] = 3
    return p

def directional_tortuosity(phase_mask, specimen_mask, axis):
    """Geodesic tortuosity τ along the given axis (0=z, 1=y, 2=x).

    The "near" and "far" faces are the FIRST and LAST slices along the axis
    that contain SPECIMEN voxels (not z=0 / z=Nz-1, which are typically
    outside the specimen). Phase voxels on those slices are the seeds /
    targets.

    τ = mean( geodesic distance through phase from near to far face ) / Δaxis

    Returns (tau, percolates_bool, n_paths_found).
    """
    if not phase_mask.any() or not specimen_mask.any():
        return float('nan'), False, 0

    # Identify axis-extent slices that contain ANY specimen voxel
    other_axes = [0, 1, 2]
    other_axes.remove(axis)
    axis_has_specimen = np.any(specimen_mask, axis=tuple(other_axes))
    if not axis_has_specimen.any():
        return float('nan'), False, 0
    a_near = int(np.argmax(axis_has_specimen))
    a_far  = int(len(axis_has_specimen) - 1 - np.argmax(axis_has_specimen[::-1]))
    L = a_far - a_near
    if L < 2:
        return float('nan'), False, 0

    # Build cost map: 1 inside phase, +inf outside
    cost = np.where(phase_mask, 1.0, np.inf)

    # Seed = phase voxels on the NEAR face (axis index a_near)
    sl_near = [slice(None)] * 3
    sl_near[axis] = a_near
    near_phase = phase_mask[tuple(sl_near)]
    if not near_phase.any():
        return float('inf'), False, 0
    starts = []
    near_idx = np.argwhere(near_phase)
    for k in range(len(near_idx)):
        s = [0, 0, 0]
        s[axis] = a_near
        s[other_axes[0]] = int(near_idx[k, 0])
        s[other_axes[1]] = int(near_idx[k, 1])
        starts.append(s)

    try:
        mcp = MCP_Geometric(cost, sampling=(1.0, 1.0, 1.0))
        cum, _ = mcp.find_costs(starts)
    except Exception:
        return float('nan'), False, 0

    sl_far = [slice(None)] * 3
    sl_far[axis] = a_far
    far_geo = cum[tuple(sl_far)]
    far_phase = phase_mask[tuple(sl_far)]
    finite = np.isfinite(far_geo) & far_phase
    if not finite.any():
        return float('inf'), False, 0
    geo = far_geo[finite]
    return float(np.mean(geo) / float(L)), True, int(finite.sum())

# ============================================================================
print(f'Tortuosity evolution — scans {ALL_SCANS}, downsample={DOWNSAMPLE}')
rows = []
for s in ALL_SCANS:
    print(f'\n=== scan {s} ===')
    t0 = time.time()
    ct   = tifffile.imread(os.path.join(DATA, f'ct_scan{s:02d}_aligned.tif'))
    mask = tifffile.imread(os.path.join(DATA, f'specimen_mask_scan{s:02d}_aligned.tif'))
    print(f'  loaded ({time.time()-t0:.1f}s)  shape={ct.shape}')

    phases = reconstruct_phases(ct, mask > 0, TH[s])
    del ct
    if DOWNSAMPLE > 1:
        ds = DOWNSAMPLE
        phases_ds = phases[::ds, ::ds, ::ds]
        mask_ds   = (mask[::ds, ::ds, ::ds] > 0)
    else:
        phases_ds = phases
        mask_ds = mask > 0
    n_inside = int(mask_ds.sum())
    if n_inside == 0:
        print('  empty downsampled mask, skipping'); continue

    for ph_name, ph_id in [('air', 1), ('ice', 2)]:
        ph = phases_ds == ph_id
        vfrac = float(ph.sum()) / max(n_inside, 1)
        print(f'  {ph_name:>3}  vfrac={vfrac:.3f}')
        for axis_name, axis in [('z', 0), ('y', 1), ('x', 2)]:
            t1 = time.time()
            tau, perc, n_paths = directional_tortuosity(ph, mask_ds, axis)
            print(f'    τ_{axis_name} = {tau:.3f}   percolates={perc}   '
                  f'paths={n_paths}   ({time.time()-t1:.1f}s)')
            rows.append(dict(
                scan=s, phase=ph_name, direction=axis_name,
                tau=tau, percolates=perc, n_paths=n_paths, vfrac=vfrac,
            ))
    del phases, mask

# Persist CSV
csv_path = os.path.join(OUT, 'summary.csv')
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter=';')
    w.writeheader(); [w.writerow(r) for r in rows]
print(f'\nwrote {csv_path}')

# Index helper
def find(scan, phase, direction):
    for r in rows:
        if r['scan']==scan and r['phase']==phase and r['direction']==direction:
            return r['tau'] if np.isfinite(r['tau']) else float('nan')
    return float('nan')

# ============================================================================
# PLOT: τ_x, τ_y, τ_z vs scan — one PNG per phase
# ============================================================================
for ph_name, color_phase in [('air', COL_AIR), ('ice', COL_ICE)]:
    fig, ax = plt.subplots(figsize=(13, 8))
    for axis_name, c, mk in [('z', COL_Z, 'o'), ('y', COL_Y, 's'), ('x', COL_X, '^')]:
        taus = [find(s, ph_name, axis_name) for s in ALL_SCANS]
        ax.plot(ALL_SCANS, taus, marker=mk, color=c, linewidth=3,
                markersize=18, markerfacecolor='white', markeredgewidth=3,
                label=f'τ_{axis_name}')
    ax.axhline(1.0, color='gray', ls=':', lw=1.5, alpha=0.6, label='τ = 1 (straight)')
    ax.set_xlabel('Scan #'); ax.set_ylabel('Tortuosity τ')
    ax.set_title(f'{ph_name.upper()} phase — directional tortuosity evolution')
    ax.set_xticks(ALL_SCANS)
    save_with_legend_below(fig, ax,
                           os.path.join(OUT, f'tau_{ph_name}_evolution.png'),
                           legend_ncol=4)

# ============================================================================
# PLOT: τ asymmetry — one PNG per phase
# ============================================================================
for ph_name in ('air', 'ice'):
    tz = np.array([find(s, ph_name, 'z') for s in ALL_SCANS])
    ty = np.array([find(s, ph_name, 'y') for s in ALL_SCANS])
    tx = np.array([find(s, ph_name, 'x') for s in ALL_SCANS])
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.plot(ALL_SCANS, tz - tx, marker='o', color=COL_X, linewidth=3,
            markersize=18, markerfacecolor='white', markeredgewidth=3,
            label='τ_z − τ_x')
    ax.plot(ALL_SCANS, tz - ty, marker='s', color=COL_Y, linewidth=3,
            markersize=18, markerfacecolor='white', markeredgewidth=3,
            label='τ_z − τ_y')
    ax.axhline(0.0, color='gray', ls='--', lw=1.5, alpha=0.7)
    ax.set_xlabel('Scan #'); ax.set_ylabel('Δτ (axial − lateral)')
    ax.set_title(f'{ph_name.upper()} phase — τ asymmetry (crack-orientation indicator)')
    ax.set_xticks(ALL_SCANS)
    save_with_legend_below(fig, ax,
                           os.path.join(OUT, f'tau_anisotropy_{ph_name}.png'),
                           legend_ncol=2)

# ============================================================================
# PLOT: Δτ per transition × direction — one PNG per phase
# ============================================================================
delta_rows = []
for a, b in TRANSITIONS:
    for ph in ('air', 'ice'):
        for axis_name in ('z', 'y', 'x'):
            ta = find(a, ph, axis_name); tb = find(b, ph, axis_name)
            d = tb - ta
            delta_rows.append(dict(transition=f'{a}→{b}', phase=ph,
                                   direction=axis_name, delta=d))

trans_lbls = [f'{a}→{b}' for a, b in TRANSITIONS]
x = np.arange(len(trans_lbls)); bw = 0.27
for ph in ('air', 'ice'):
    fig, ax = plt.subplots(figsize=(13, 8))
    for i, (axis_name, c) in enumerate(zip(['z', 'y', 'x'],
                                           [COL_Z, COL_Y, COL_X])):
        vals = [d['delta'] for d in delta_rows
                if d['phase'] == ph and d['direction'] == axis_name]
        ax.bar(x + (i - 1) * bw, vals, bw, color=c,
               edgecolor='black', linewidth=1.2,
               label=f'Δτ_{axis_name}')
    ax.axhline(0, color='gray', lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(trans_lbls)
    ax.set_xlabel('Transition'); ax.set_ylabel('Δτ (final − initial)')
    ax.set_title(f'{ph.upper()} — Δτ per transition × direction')
    save_with_legend_below(fig, ax,
                           os.path.join(OUT, f'delta_tau_{ph}_per_transition.png'),
                           legend_ncol=3)

# ============================================================================
# PLOT: crack indicator (single, air phase only, no panels)
# ============================================================================
fig, ax = plt.subplots(figsize=(13, 8))
labels_x, heights, colors = [], [], []
for d in delta_rows:
    if d['phase'] != 'air':
        continue
    flag = abs(d['delta']) > CRACK_DELTA_TAU
    labels_x.append(f"{d['transition']}\nair-{d['direction']}")
    heights.append(abs(d['delta']) if np.isfinite(d['delta']) else 0)
    colors.append(COL_FLAG if flag else '#d0d0d0')
ax.bar(labels_x, heights, color=colors, edgecolor='black', linewidth=1.5)
ax.axhline(CRACK_DELTA_TAU, color='black', ls='--', lw=2, alpha=0.8,
           label=f'|Δτ| > {CRACK_DELTA_TAU} threshold')
ax.set_ylabel('|Δτ| of air phase')
ax.set_title('Crack indicator — sharp Δτ in air-phase tortuosity')
save_with_legend_below(fig, ax, os.path.join(OUT, 'crack_indicator.png'),
                       legend_ncol=1)

# Text summary
txt = os.path.join(OUT, 'summary.txt')
with open(txt, 'w') as f:
    f.write('Directional tortuosity evolution\n')
    f.write('=' * 60 + '\n\n')
    f.write(f'{"scan":<5}{"phase":<6}{"τ_z":>9}{"τ_y":>9}{"τ_x":>9}'
            f'{"vfrac":>10}{"perc_z":>9}{"perc_y":>9}{"perc_x":>9}\n')
    for s in ALL_SCANS:
        for ph in ('air', 'ice'):
            row = {(r['direction'], 'tau'):    r['tau']    for r in rows
                   if r['scan']==s and r['phase']==ph}
            row.update({(r['direction'], 'perc'): r['percolates'] for r in rows
                        if r['scan']==s and r['phase']==ph})
            row.update({(r['direction'], 'vf'):  r['vfrac']     for r in rows
                        if r['scan']==s and r['phase']==ph})
            f.write(f"{s:<5}{ph:<6}"
                    f"{row.get(('z','tau'), float('nan')):>9.3f}"
                    f"{row.get(('y','tau'), float('nan')):>9.3f}"
                    f"{row.get(('x','tau'), float('nan')):>9.3f}"
                    f"{row.get(('z','vf'),  0):>10.3f}"
                    f"{str(row.get(('z','perc'), False)):>9}"
                    f"{str(row.get(('y','perc'), False)):>9}"
                    f"{str(row.get(('x','perc'), False)):>9}\n")
    f.write('\nΔτ per transition (final − initial):\n')
    for d in delta_rows:
        f.write(f"  {d['transition']}  {d['phase']:<5}  τ_{d['direction']}  "
                f"Δ = {d['delta']:+.3f}\n")
    f.write('\nInterpretation:\n')
    f.write('  τ = 1.0  -> straight pathway through phase\n')
    f.write('  τ > 1   -> tortuous pathway (the higher, the more winding)\n')
    f.write('  τ = ∞   -> phase does not percolate that direction\n')
    f.write(f'  |Δτ_dir| > {CRACK_DELTA_TAU} between scans -> potential crack-like event\n')
    f.write('  Sharp drop in air τ along ONE direction -> oriented crack along that axis\n')
print(f'wrote {txt}')
print('\nDone.')
