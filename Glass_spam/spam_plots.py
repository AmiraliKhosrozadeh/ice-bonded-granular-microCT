"""
Post-process spam-ddic output -> CSV + plots for quick QA.

Runs inside the WSL spam-venv (or any Python with numpy/matplotlib).
Reads: <SPAM_RESULTS>/glass_ddic-ddic.tsv
Writes to <RESULTS_DIR>/:
  bead_tracking_ddic.csv         full per-bead measurements (cleaned)
  zdisp_vs_height.png            Z-displacement vs Z-position + linear fit
  disp_magnitude_hist.png        |u| histogram
  rotation_hist.png              per-bead rotation (deg) histogram
  xy_quiver_zdisp.png            XY scatter of beads, color = Zdisp
  phi_qc.png                     3-panel iteration count / error / convergence
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SPAM_RESULTS = os.environ.get('SPAM_RESULTS', '/home/cak7496/spam-results')
RESULTS_DIR  = os.environ.get('RESULTS_DIR',
                              '/mnt/e/RPTU-images/CT_images/Glass/Glass_spam/results')
TSV_PATH = os.path.join(SPAM_RESULTS, 'glass_ddic-ddic.tsv')
os.makedirs(RESULTS_DIR, exist_ok=True)

VOXEL_SIZE_UM = 24.766

# -- Plot style (project defaults) --------------------------------------------
_FS = 24
plt.rcParams.update({
    "font.size":       _FS,
    "axes.titlesize":  _FS,
    "axes.labelsize":  _FS + 2,
    "xtick.labelsize": _FS - 2,
    "ytick.labelsize": _FS - 2,
    "legend.fontsize": _FS - 2,
    "axes.linewidth":  1.8,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
})
LEG_BBOX = (0.5, -0.14)

# -- Load TSV ------------------------------------------------------------------
with open(TSV_PATH) as f:
    header = f.readline().strip().split('\t')
data = np.loadtxt(TSV_PATH, skiprows=1, delimiter='\t')
print(f'Loaded {TSV_PATH}')
print(f'  shape={data.shape}  columns={header}')

col = {h: i for i, h in enumerate(header)}
beads = data[data[:, col['Label']].astype(int) > 0]
print(f'  Beads: {len(beads)}')

lab    = beads[:, col['Label']].astype(int)
zpos   = beads[:, col['Zpos']]
ypos   = beads[:, col['Ypos']]
xpos   = beads[:, col['Xpos']]
zdisp  = beads[:, col['Zdisp']]
ydisp  = beads[:, col['Ydisp']]
xdisp  = beads[:, col['Xdisp']]
rs     = beads[:, col['returnStatus']].astype(int)
err    = beads[:, col['error']]
niter  = beads[:, col['iterations']].astype(int)

# Per-bead rotation magnitude from F (proper rotation via polar decomposition)
def rotmag_deg(Fzz, Fzy, Fzx, Fyz, Fyy, Fyx, Fxz, Fxy, Fxx):
    F = np.array([[Fzz, Fzy, Fzx],
                  [Fyz, Fyy, Fyx],
                  [Fxz, Fxy, Fxx]])
    # Polar decomposition: F = R U, with R = U_f V_f^T (from SVD)
    try:
        U, _, Vt = np.linalg.svd(F)
    except np.linalg.LinAlgError:
        return np.nan
    R = U @ Vt
    # Ensure proper rotation (det=+1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    cos_theta = np.clip((np.trace(R) - 1) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

rot_deg = np.array([
    rotmag_deg(*beads[i, [col['Fzz'], col['Fzy'], col['Fzx'],
                          col['Fyz'], col['Fyy'], col['Fyx'],
                          col['Fxz'], col['Fxy'], col['Fxx']]])
    for i in range(len(beads))
])

disp_mag = np.sqrt(zdisp**2 + ydisp**2 + xdisp**2)

# Validity mask: finite displacements, reasonable error
valid = (
    np.isfinite(zdisp) & np.isfinite(ydisp) & np.isfinite(xdisp)
    & (rs >= 1) & (err < 10000)
)
n_valid = int(valid.sum())
print(f'  Valid beads (finite & rs>=1 & err<10000): {n_valid}/{len(beads)}')

# -- Write clean CSV -----------------------------------------------------------
csv_path = os.path.join(RESULTS_DIR, 'bead_tracking_ddic.csv')
with open(csv_path, 'w') as f:
    f.write(';'.join([
        'bead_id', 'zpos_vox', 'ypos_vox', 'xpos_vox',
        'zdisp_vox', 'ydisp_vox', 'xdisp_vox', 'disp_mag_vox',
        'zdisp_um', 'ydisp_um', 'xdisp_um', 'disp_mag_um',
        'rot_deg', 'error', 'iterations', 'returnStatus', 'valid'
    ]) + '\n')
    for i in range(len(beads)):
        f.write(';'.join([
            f'{lab[i]}',
            f'{zpos[i]:.2f}', f'{ypos[i]:.2f}', f'{xpos[i]:.2f}',
            f'{zdisp[i]:.3f}', f'{ydisp[i]:.3f}', f'{xdisp[i]:.3f}', f'{disp_mag[i]:.3f}',
            f'{zdisp[i]*VOXEL_SIZE_UM:.2f}', f'{ydisp[i]*VOXEL_SIZE_UM:.2f}',
            f'{xdisp[i]*VOXEL_SIZE_UM:.2f}', f'{disp_mag[i]*VOXEL_SIZE_UM:.2f}',
            f'{rot_deg[i]:.3f}' if np.isfinite(rot_deg[i]) else 'nan',
            f'{err[i]:.2f}', f'{niter[i]}', f'{rs[i]}', f'{int(valid[i])}',
        ]) + '\n')
print(f'Saved: {csv_path}')

# -- Plot 1: Zdisp vs Zpos with linear fit ------------------------------------
fig, ax = plt.subplots(figsize=(14, 9))
ax.scatter(zpos[valid], zdisp[valid], s=55, alpha=0.6,
           color='#1f77b4', edgecolor='black', linewidth=0.6)
# Fit only on valid beads
if n_valid >= 2:
    a, b = np.polyfit(zpos[valid], zdisp[valid], 1)
    xfit = np.array([zpos[valid].min(), zpos[valid].max()])
    ax.plot(xfit, a * xfit + b, '-', color='#d62728', linewidth=3,
            label=f'Fit: Zdisp = {a:+.4f}·Zpos {b:+.2f}')
# Theoretical line: -142/864 slope, through origin
xT = np.array([0, 874])
ax.plot(xT, (-142/864) * xT, '--', color='#2ca02c', linewidth=2.5,
        label=f'Theory: slope=-{142/864:.4f} (864→722 compression)')
ax.axhline(0, color='gray', linewidth=1, linestyle=':')
ax.set_xlabel('Bead Z-position (voxels, common frame)', labelpad=10)
ax.set_ylabel('Z-displacement (voxels)', labelpad=10)
ax.grid(True, alpha=0.3)
handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
           markeredgecolor='black', markersize=10, label=f'Beads (n={n_valid})'),
    Line2D([0], [0], color='#d62728', linewidth=3,
           label=f'Linear fit'),
    Line2D([0], [0], color='#2ca02c', linewidth=2.5, linestyle='--',
           label='Theoretical compression'),
]
fig.legend(handles=handles, loc='lower center', ncol=3,
           frameon=True, framealpha=0.85, edgecolor='gray',
           bbox_to_anchor=LEG_BBOX)
plt.title('Axial Compression Field (DDIC)', pad=18, fontweight='bold')
plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'zdisp_vs_height.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')

# -- Plot 2: displacement magnitude histogram ---------------------------------
fig, ax = plt.subplots(figsize=(14, 9))
ax.hist(disp_mag[valid] * VOXEL_SIZE_UM, bins=40,
        color='#1f77b4', edgecolor='black', linewidth=1.0, alpha=0.85)
ax.axvline(np.median(disp_mag[valid]) * VOXEL_SIZE_UM, color='#d62728',
           linewidth=3, linestyle='--',
           label=f'Median = {np.median(disp_mag[valid])*VOXEL_SIZE_UM:.0f} µm')
ax.set_xlabel('Displacement magnitude |u| (µm)', labelpad=10)
ax.set_ylabel('Bead count', labelpad=10)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')
plt.title('Bead Displacement Magnitude', pad=18, fontweight='bold')
plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'disp_magnitude_hist.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')

# -- Plot 3: rotation magnitude histogram -------------------------------------
fig, ax = plt.subplots(figsize=(14, 9))
rot_valid = rot_deg[valid & np.isfinite(rot_deg)]
ax.hist(rot_valid, bins=40, color='#9467bd', edgecolor='black',
        linewidth=1.0, alpha=0.85)
if len(rot_valid):
    ax.axvline(np.median(rot_valid), color='#d62728', linewidth=3, linestyle='--',
               label=f'Median = {np.median(rot_valid):.1f}°')
ax.set_xlabel('Rotation magnitude (degrees)', labelpad=10)
ax.set_ylabel('Bead count', labelpad=10)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')
plt.title('Per-Bead Rotation', pad=18, fontweight='bold')
plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'rotation_hist.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')

# -- Plot 4: XY scatter colored by Zdisp --------------------------------------
fig, ax = plt.subplots(figsize=(14, 11))
sc = ax.scatter(xpos[valid], ypos[valid], c=zdisp[valid],
                s=80, cmap='RdBu_r', edgecolor='black', linewidth=0.5,
                vmin=-np.max(np.abs(zdisp[valid])), vmax=np.max(np.abs(zdisp[valid])))
cb = plt.colorbar(sc, ax=ax, label='Zdisp (voxels)')
ax.set_xlabel('X (voxels)', labelpad=10)
ax.set_ylabel('Y (voxels)', labelpad=10)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.title('XY Map of Z-Displacement (all beads)', pad=18, fontweight='bold')
plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'xy_zdisp_map.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')

# -- Plot 5: convergence QC (3-panel) -----------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
axes[0].hist(niter, bins=np.arange(0, 52, 2), color='#ff7f0e',
             edgecolor='black', linewidth=1.0)
axes[0].set_xlabel('Iterations', labelpad=10)
axes[0].set_ylabel('Bead count', labelpad=10)
axes[0].set_title('Iterations used', pad=10)
axes[0].grid(True, alpha=0.3)

err_plot = err.copy()
err_plot = np.clip(err_plot, 1, 1e5)
axes[1].hist(np.log10(err_plot), bins=40, color='#d62728',
             edgecolor='black', linewidth=1.0)
axes[1].set_xlabel('log10(error)', labelpad=10)
axes[1].set_ylabel('Bead count', labelpad=10)
axes[1].set_title('Correlation error', pad=10)
axes[1].grid(True, alpha=0.3)

rs_vals, rs_counts = np.unique(rs, return_counts=True)
names = {0:'bg', 1:'max-iter', 2:'converged', -1:'diverged', -2:'NaN', -3:'degen'}
labels_plot = [f'{v}\n({names.get(int(v),str(v))})' for v in rs_vals]
axes[2].bar(range(len(rs_vals)), rs_counts, color='#2ca02c', edgecolor='black',
            linewidth=1.0)
axes[2].set_xticks(range(len(rs_vals)))
axes[2].set_xticklabels(labels_plot, fontsize=_FS - 4)
axes[2].set_ylabel('Bead count', labelpad=10)
axes[2].set_title('Return status', pad=10)
for i, c in enumerate(rs_counts):
    axes[2].text(i, c + max(rs_counts)*0.01, str(int(c)),
                 ha='center', fontsize=_FS - 4)
axes[2].grid(True, alpha=0.3, axis='y')

plt.suptitle('DDIC Convergence QC', fontsize=_FS + 4, fontweight='bold', y=1.02)
plt.tight_layout()
out = os.path.join(RESULTS_DIR, 'phi_qc.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')

# -- Summary -------------------------------------------------------------------
print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
print(f'Beads total       : {len(beads)}')
print(f'Valid beads       : {n_valid}')
if n_valid:
    dv = disp_mag[valid] * VOXEL_SIZE_UM
    zv = zdisp[valid] * VOXEL_SIZE_UM
    print(f'|u| median        : {np.median(dv):.0f} um')
    print(f'|u| 90th pctile   : {np.percentile(dv, 90):.0f} um')
    print(f'Zdisp range       : {zv.min():+.0f} to {zv.max():+.0f} um')
    a, b = np.polyfit(zpos[valid], zdisp[valid], 1)
    print(f'Compression slope : {a:+.4f}  (theory {-142/864:+.4f})')
print('Outputs in        :', RESULTS_DIR)
