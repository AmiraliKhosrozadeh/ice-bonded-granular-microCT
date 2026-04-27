"""Full post-processing for the 3 T7 DDIC transitions.

For each transition (1->2, 2->3, 3->4):
  * 2D QC plots (zdisp_vs_height, |u| hist, rotation hist, phi_qc)
  * 3D conference figures (displacement field + volumetric strain)
  * bead_tracking CSV

Output tree:  /mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>/<tag>/

Run in WSL spam-venv:
    python /mnt/e/.../<SPAM>/spam_all_plots.py
"""
import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pyvista as pv
from scipy.spatial import Delaunay

pv.OFF_SCREEN = True

DATA   = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
RES    = '/home/cak7496/spam-results-75'
OUT_BASE = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>'
os.makedirs(OUT_BASE, exist_ok=True)

VOXEL_SIZE_UM = 24.7660229
TRANSITIONS = [(1, 2), (2, 3)]

def polar_rot_volstrain(F):
    try:
        U, _, Vt = np.linalg.svd(F)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    cos_t = np.clip((np.trace(R) - 1) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_t)), np.linalg.det(F) - 1.0

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
LEG_BBOX = (0.5, -0.14)

for a, b in TRANSITIONS:
    tag = f'75_ddic_{a}{b}'
    out_dir = os.path.join(OUT_BASE, f'transition_{a}to{b}')
    os.makedirs(out_dir, exist_ok=True)
    tsv = os.path.join(RES, f'{tag}-ddic.tsv')
    print(f'\n=== Transition {a}->{b} ===')
    print(f'  reading {tsv}')
    with open(tsv) as f: header = f.readline().strip().split('\t')
    data = np.loadtxt(tsv, skiprows=1, delimiter='\t')
    col = {h:i for i,h in enumerate(header)}
    beads = data[data[:, col['Label']].astype(int) > 0]

    zpos, ypos, xpos = beads[:, col['Zpos']], beads[:, col['Ypos']], beads[:, col['Xpos']]
    zd, yd, xd = beads[:, col['Zdisp']], beads[:, col['Ydisp']], beads[:, col['Xdisp']]
    rs = beads[:, col['returnStatus']].astype(int)
    err = beads[:, col['error']]
    niter = beads[:, col['iterations']].astype(int)
    lab = beads[:, col['Label']].astype(int)

    valid = (np.isfinite(zd)&np.isfinite(yd)&np.isfinite(xd)&(rs>=1)&(err < 25000))
    n_valid = int(valid.sum())
    print(f'  valid beads: {n_valid}/{len(beads)}')
    if n_valid < 5:
        print(f'  SKIP transition {a}->{b}: not enough valid beads')
        continue

    # Rotation + volumetric strain per bead
    rot_deg = np.zeros(len(beads)); vol_strain = np.zeros(len(beads))
    for i in range(len(beads)):
        F = np.array([
            [beads[i,col['Fxx']], beads[i,col['Fxy']], beads[i,col['Fxz']]],
            [beads[i,col['Fyx']], beads[i,col['Fyy']], beads[i,col['Fyz']]],
            [beads[i,col['Fzx']], beads[i,col['Fzy']], beads[i,col['Fzz']]],
        ])
        rot_deg[i], vol_strain[i] = polar_rot_volstrain(F)
    disp_mag = np.sqrt(zd**2+yd**2+xd**2)

    # ---- CSV
    csv = os.path.join(out_dir, 'bead_tracking_ddic.csv')
    with open(csv, 'w') as f:
        f.write('bead_id;zpos;ypos;xpos;zdisp;ydisp;xdisp;disp_mag_um;'
                'rot_deg;vol_strain;error;iterations;returnStatus;valid\n')
        for i in range(len(beads)):
            f.write(f'{lab[i]};{zpos[i]:.2f};{ypos[i]:.2f};{xpos[i]:.2f};'
                    f'{zd[i]:.3f};{yd[i]:.3f};{xd[i]:.3f};'
                    f'{disp_mag[i]*VOXEL_SIZE_UM:.1f};'
                    f'{rot_deg[i] if np.isfinite(rot_deg[i]) else "nan":}{"" if not np.isfinite(rot_deg[i]) else ""};'
                    f'{vol_strain[i]:.4f};{err[i]:.2f};{niter[i]};{rs[i]};{int(valid[i])}\n')
    print(f'  wrote {csv}')

    # ---- zdisp vs height (compression field)
    # Z convention: low Zpos = physical TOP (Dragonfly Z-positive-down).
    # Invert X axis so physical top appears on the LEFT of the plot.
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.scatter(zpos[valid], zd[valid], s=55, alpha=0.6,
               color='#1f77b4', edgecolor='black', linewidth=0.6)
    slope, intercept = np.polyfit(zpos[valid], zd[valid], 1)
    xfit = np.array([zpos[valid].min(), zpos[valid].max()])
    ax.plot(xfit, slope*xfit + intercept, '-', color='#d62728', linewidth=3,
            label=f'Fit: Zdisp = {slope:+.4f}·Zpos {intercept:+.2f}')
    ax.axhline(0, color='gray', linewidth=1, linestyle=':')
    ax.invert_xaxis()
    ax.set_xlabel('Bead Z-position (voxels, top -> bottom)', labelpad=10)
    ax.set_ylabel('Z-displacement (voxels, + = downward)', labelpad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    plt.title(f'Compression Field — scan {a} → {b}', pad=18, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'zdisp_vs_height.png'),
                dpi=150, bbox_inches='tight'); plt.close(fig)

    # ---- |u| histogram
    fig, ax = plt.subplots(figsize=(14, 9))
    dv = disp_mag[valid]*VOXEL_SIZE_UM
    ax.hist(dv, bins=40, color='#1f77b4', edgecolor='black', alpha=0.85)
    ax.axvline(np.median(dv), color='#d62728', linewidth=3, linestyle='--',
               label=f'Median = {np.median(dv):.0f} µm')
    ax.set_xlabel('Displacement |u| (µm)', labelpad=10)
    ax.set_ylabel('Bead count', labelpad=10)
    ax.grid(True, alpha=0.3); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    plt.title(f'Bead Displacement Magnitude — scan {a} → {b}', pad=18, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'disp_magnitude_hist.png'),
                dpi=150, bbox_inches='tight'); plt.close(fig)

    # ---- rotation histogram
    fig, ax = plt.subplots(figsize=(14, 9))
    rv = rot_deg[valid & np.isfinite(rot_deg)]
    if len(rv):
        ax.hist(rv, bins=40, color='#9467bd', edgecolor='black', alpha=0.85)
        ax.axvline(np.median(rv), color='#d62728', linewidth=3, linestyle='--',
                   label=f'Median = {np.median(rv):.1f}°')
    ax.set_xlabel('Rotation (degrees)', labelpad=10)
    ax.set_ylabel('Bead count', labelpad=10)
    ax.grid(True, alpha=0.3); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    plt.title(f'Per-Bead Rotation — scan {a} → {b}', pad=18, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'rotation_hist.png'),
                dpi=150, bbox_inches='tight'); plt.close(fig)

    # ---- phi_qc (3-panel convergence)
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    axes[0].hist(niter, bins=np.arange(0, 52, 2), color='#ff7f0e',
                 edgecolor='black')
    axes[0].set_xlabel('Iterations'); axes[0].set_ylabel('Bead count')
    axes[0].set_title('Iterations used'); axes[0].grid(True, alpha=0.3)
    err_p = np.clip(err, 1, 1e5)
    axes[1].hist(np.log10(err_p), bins=40, color='#d62728', edgecolor='black')
    axes[1].set_xlabel('log10(error)'); axes[1].set_ylabel('Bead count')
    axes[1].set_title('Correlation error'); axes[1].grid(True, alpha=0.3)
    rs_v, rs_c = np.unique(rs, return_counts=True)
    axes[2].bar(range(len(rs_v)), rs_c, color='#2ca02c', edgecolor='black')
    axes[2].set_xticks(range(len(rs_v)))
    axes[2].set_xticklabels([f'{v}' for v in rs_v])
    axes[2].set_ylabel('Bead count'); axes[2].set_title('Return status')
    axes[2].grid(True, alpha=0.3, axis='y')
    plt.suptitle(f'DDIC QC — scan {a} → {b}', fontsize=_FS+4, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'phi_qc.png'),
                dpi=150, bbox_inches='tight'); plt.close(fig)

    # ---- 3D conference-style figures --------------------------------------
    X = np.column_stack([xpos[valid], ypos[valid], zpos[valid]]).astype(np.float32)
    U = np.column_stack([xd[valid], yd[valid], zd[valid]])

    # Delaunay tets for strain
    if len(X) < 5:
        print('  too few points for Delaunay; skipping 3D')
        continue
    try:
        tri = Delaunay(X)
        tets = tri.simplices
        tet_vs = np.full(len(tets), np.nan)
        x_def = X + U
        for k, (i0,i1,i2,i3) in enumerate(tets):
            dX = np.column_stack([X[i1]-X[i0], X[i2]-X[i0], X[i3]-X[i0]])
            dx = np.column_stack([x_def[i1]-x_def[i0], x_def[i2]-x_def[i0], x_def[i3]-x_def[i0]])
            try:
                cond = np.linalg.cond(dX)
                F = dx @ np.linalg.inv(dX)
                if cond < 500:
                    tet_vs[k] = np.linalg.det(F) - 1.0
            except np.linalg.LinAlgError:
                pass
    except Exception as e:
        print(f'  Delaunay failed: {e}')
        tet_vs = np.array([])

    # Bead-level volumetric strain from averaged F of surrounding tets
    bead_vs = np.full(len(X), np.nan)
    if len(tets):
        bead_to_tets = [[] for _ in range(len(X))]
        for k in range(len(tets)):
            if np.isfinite(tet_vs[k]):
                for bid in tets[k]:
                    bead_to_tets[bid].append(k)
        for bidx in range(len(X)):
            good = [tet_vs[k] for k in bead_to_tets[bidx] if np.isfinite(tet_vs[k])]
            if good:
                bead_vs[bidx] = float(np.mean(good))

    # Displacement field
    bead_pd = pv.PolyData(X)
    bead_pd['|u|_um'] = np.linalg.norm(U, axis=1) * VOXEL_SIZE_UM
    bead_pd['vol_strain'] = bead_vs
    bead_pd['u'] = U.astype(np.float32)
    BEAD_R = 32
    bead_spheres = bead_pd.glyph(geom=pv.Sphere(radius=BEAD_R, theta_resolution=20,
                                                phi_resolution=20),
                                 scale=False, orient=False)
    norms = np.linalg.norm(U, axis=1)
    norms_safe = np.where(norms > 0, norms, 1.0)
    bead_pd['unit_u'] = (U / norms_safe[:, None]).astype(np.float32)
    ARROW_LEN = 55
    arrows = bead_pd.glyph(
        geom=pv.Arrow(tip_length=0.32, tip_radius=0.10, shaft_radius=0.035),
        orient='unit_u', scale=False, factor=ARROW_LEN)

    nx_e, ny_e, nz_e = 803, 707, 1241   # T5_HR common aligned frame
    # Flipped Z camera: physical top (low slice index) appears at TOP of image.
    # Matches Dragonfly's Z-positive-down viewing convention.
    CAM = [(nx_e*2.1, -ny_e*1.5, -nz_e*0.7), (nx_e/2, ny_e/2, nz_e/2), (0,0,-1)]

    sbar = dict(title_font_size=60, label_font_size=60, color='#1a1a1a',
                font_family='arial', position_x=0.86, position_y=0.16,
                height=0.70, width=0.045, vertical=True, shadow=False, n_labels=5)

    # Displacement figure
    p = pv.Plotter(off_screen=True, window_size=(2400, 1800))
    p.set_background('white'); p.enable_anti_aliasing('msaa', multi_samples=8)
    p.add_mesh(bead_spheres, color='#cdd3dd', opacity=0.28, smooth_shading=True)
    p.add_mesh(arrows, scalars='|u|_um', cmap='turbo',
               scalar_bar_args={**sbar, 'title':'|u| (µm)'})
    p.camera_position = CAM
    fig_path = os.path.join(out_dir, 'FINAL_displacement_field.png')
    p.screenshot(fig_path); p.close()
    print(f'  wrote {fig_path}')

    # Strain figure
    finite_vs = bead_vs[np.isfinite(bead_vs)]
    vs_clim = (-0.30, 0.30)
    p = pv.Plotter(off_screen=True, window_size=(2400, 1800))
    p.set_background('white'); p.enable_anti_aliasing('msaa', multi_samples=8)
    nan_mask = ~np.isfinite(bead_vs)
    if nan_mask.any():
        ghost = pv.PolyData(X[nan_mask].astype(np.float32)).glyph(
            geom=pv.Sphere(radius=BEAD_R, theta_resolution=18, phi_resolution=18),
            scale=False, orient=False)
        p.add_mesh(ghost, color='#d8d8d8', opacity=0.4, smooth_shading=True)
    good_pd = pv.PolyData(X[~nan_mask].astype(np.float32))
    good_pd['vol_strain'] = bead_vs[~nan_mask]
    good_spheres = good_pd.glyph(
        geom=pv.Sphere(radius=BEAD_R, theta_resolution=24, phi_resolution=24),
        scale=False, orient=False)
    p.add_mesh(good_spheres, scalars='vol_strain', cmap='seismic_r',
               clim=vs_clim,
               scalar_bar_args={**sbar, 'title':'Volumetric strain'},
               smooth_shading=True, specular=0.2, ambient=0.4)
    p.camera_position = CAM
    fig_path = os.path.join(out_dir, 'FINAL_strain_field_volumetric.png')
    p.screenshot(fig_path); p.close()
    print(f'  wrote {fig_path}')

print('\nAll transitions done. Open:', OUT_BASE)
