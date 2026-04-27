"""3D visualization of LDIC strain fields for T7 transitions.

Reads  ~/spam-results-t7/t7_ldic_NN-ldic.tsv
Writes in results_<PRE>/transition_AtoB/:
  LDIC_volstrain_3D.png       volumetric strain, all valid nodes
  LDIC_deviatoric_3D.png      von Mises / deviatoric invariant
  LDIC_coverage_stats.txt     how many nodes actually converged
"""
import os
import numpy as np
import pyvista as pv
pv.OFF_SCREEN = True

RES = '/home/cak7496/spam-results-75'
OUT = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>'

TRANSITIONS = [(1, 2), (2, 3)]

def polar_vs_mises(F):
    try:
        U, _, Vt = np.linalg.svd(F)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    vs = np.linalg.det(F) - 1.0
    E = 0.5 * (F.T @ F - np.eye(3))
    Ed = E - np.trace(E) / 3 * np.eye(3)
    mi = np.sqrt(1.5 * np.sum(Ed * Ed))
    return vs, mi

nx_e, ny_e, nz_e = 803, 707, 1241   # T5_HR common aligned frame
CAM = [(nx_e * 2.1, -ny_e * 1.5, -nz_e * 0.7),
       (nx_e / 2, ny_e / 2, nz_e / 2),
       (0, 0, -1)]
SBAR = dict(title_font_size=60, label_font_size=60, color='#1a1a1a',
            font_family='arial', position_x=0.86, position_y=0.16,
            height=0.70, width=0.045, vertical=True, shadow=False, n_labels=5)

for a, b in TRANSITIONS:
    tag = f'{a}{b}'
    tsv = os.path.join(RES, f'75_ldic_{tag}-ldic.tsv')
    tdir = os.path.join(OUT, f'transition_{a}to{b}')
    os.makedirs(tdir, exist_ok=True)
    if not os.path.exists(tsv):
        print(f'skip {a}->{b}: no TSV'); continue
    with open(tsv) as f:
        header = f.readline().strip().split('\t')
    data = np.loadtxt(tsv, skiprows=1, delimiter='\t')
    col = {h: i for i, h in enumerate(header)}

    zp = data[:, col['Zpos']]; yp = data[:, col['Ypos']]; xp = data[:, col['Xpos']]
    zd = data[:, col['Zdisp']]; yd = data[:, col['Ydisp']]; xd = data[:, col['Xdisp']]
    rs = data[:, col['returnStatus']].astype(int)
    err = data[:, col['error']]

    vs  = np.full(len(data), np.nan)
    mi  = np.full(len(data), np.nan)
    for i in range(len(data)):
        F = np.array([
            [data[i, col['Fxx']], data[i, col['Fxy']], data[i, col['Fxz']]],
            [data[i, col['Fyx']], data[i, col['Fyy']], data[i, col['Fyz']]],
            [data[i, col['Fzx']], data[i, col['Fzy']], data[i, col['Fzz']]],
        ])
        if np.all(np.isfinite(F)):
            vs[i], mi[i] = polar_vs_mises(F)

    valid = (np.isfinite(zd) & np.isfinite(yd) & np.isfinite(xd)
             & (rs >= 1) & np.isfinite(err))
    n_valid = int(valid.sum())
    with open(os.path.join(tdir, 'LDIC_coverage_stats.txt'), 'w') as f:
        f.write(f'LDIC grid nodes total   : {len(data)}\n')
        f.write(f'LDIC nodes valid        : {n_valid}  ({100*n_valid/len(data):.1f}%)\n')
        f.write(f'returnStatus=2 (conv)   : {int((rs == 2).sum())}\n')
        f.write(f'returnStatus=1 (max-it) : {int((rs == 1).sum())}\n')
        f.write(f'returnStatus<0          : {int((rs < 0).sum())}\n')
    print(f'\n{a}->{b}: {n_valid}/{len(data)} LDIC nodes valid')
    if n_valid < 50:
        print('  too few nodes; skipping 3D render')
        continue

    pts = np.column_stack([xp[valid], yp[valid], zp[valid]]).astype(np.float32)

    cloud_vs = pv.PolyData(pts)
    cloud_vs['vol_strain'] = vs[valid]
    glyph_vs = cloud_vs.glyph(
        geom=pv.Sphere(radius=12, theta_resolution=14, phi_resolution=14),
        scale=False, orient=False)

    cloud_mi = pv.PolyData(pts)
    cloud_mi['mises'] = mi[valid]
    glyph_mi = cloud_mi.glyph(
        geom=pv.Sphere(radius=12, theta_resolution=14, phi_resolution=14),
        scale=False, orient=False)

    # Volumetric strain figure
    p = pv.Plotter(off_screen=True, window_size=(2400, 1800))
    p.set_background('white'); p.enable_anti_aliasing('msaa', multi_samples=8)
    p.add_mesh(glyph_vs, scalars='vol_strain', cmap='seismic_r',
               clim=(-0.30, 0.30), smooth_shading=True,
               scalar_bar_args={**SBAR, 'title': 'LDIC vol strain'})
    p.camera_position = CAM
    out1 = os.path.join(tdir, 'LDIC_volstrain_3D.png')
    p.screenshot(out1); p.close()
    print(f'  wrote {out1}')

    # Deviatoric (von Mises) strain figure
    mi_p = np.nanpercentile(mi[valid], 95) if np.isfinite(mi[valid]).any() else 0.3
    p = pv.Plotter(off_screen=True, window_size=(2400, 1800))
    p.set_background('white'); p.enable_anti_aliasing('msaa', multi_samples=8)
    p.add_mesh(glyph_mi, scalars='mises', cmap='plasma',
               clim=(0, mi_p), smooth_shading=True,
               scalar_bar_args={**SBAR, 'title': 'LDIC von Mises'})
    p.camera_position = CAM
    out2 = os.path.join(tdir, 'LDIC_deviatoric_3D.png')
    p.screenshot(out2); p.close()
    print(f'  wrote {out2}')

print('\nDone.')
