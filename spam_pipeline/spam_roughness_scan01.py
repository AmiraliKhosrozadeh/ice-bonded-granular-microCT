"""Per-bead surface roughness metrics for scan 1 (intact specimen).

Produces a roughness baseline to compare against a future high-roughness
material scan.

Metrics per bead (from the segmented label image):
  eq_diam_um      equivalent-sphere diameter   (from volume)
  radius_mean_um  mean distance center -> surface voxels
  Ra_um           arithmetic mean roughness   = <|r - R|>
  Rq_um           RMS roughness               = sqrt(<(r - R)^2>)
  Rt_um           peak-to-valley              = r_max - r_min
  Rsk             skewness of (r - R)
  Rku             kurtosis of (r - R)        (3 for a Gaussian surface)
  sphericity      Wadell sphericity          (pi^(1/3) (6V)^(2/3)) / A
                                              (1.0 = perfect sphere)
  ssa_um2_per_um3 specific surface area       = A / V

Aggregates across all beads into summary and plots.

Resolution caveat: voxel = 24.766 um, so roughness below ~12 um (half a voxel)
is invisible. Valid only for comparisons against roughness features larger
than one voxel.

Run in WSL spam-venv:
    python /mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/spam_roughness_scan01.py
"""
import os, time
import numpy as np
import tifffile
from scipy import ndimage
from skimage import measure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
OUT  = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>/roughness_scan01'
os.makedirs(OUT, exist_ok=True)

VOXEL_UM = 24.7660229
VOX_VOL_UM3 = VOXEL_UM ** 3
VOX_AREA_UM2 = VOXEL_UM ** 2

# Beads below this many voxels are too small for reliable roughness stats.
MIN_VOL_VOX = 500

# Plot style
_FS = 22
plt.rcParams.update({
    "font.size": _FS, "axes.titlesize": _FS, "axes.labelsize": _FS + 2,
    "xtick.labelsize": _FS - 2, "ytick.labelsize": _FS - 2,
    "legend.fontsize": _FS - 2, "axes.linewidth": 1.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

# --------- load --------------------------------------------------------------
print('loading scan01 label volume')
lab = tifffile.imread(os.path.join(DATA, 'bead_labels_scan01_aligned.tif'))
n_max = int(lab.max())
print(f'  shape {lab.shape}, {n_max} labels')
slices = ndimage.find_objects(lab)
print(f'  {sum(s is not None for s in slices)} non-empty labels')

# --------- per-bead analysis -------------------------------------------------
rows = []
t0 = time.time()
for i, sl in enumerate(slices, start=1):
    if sl is None:
        continue
    # pad bbox so marching cubes sees closed surface
    pad = 2
    z0, z1 = max(0, sl[0].start - pad), min(lab.shape[0], sl[0].stop + pad)
    y0, y1 = max(0, sl[1].start - pad), min(lab.shape[1], sl[1].stop + pad)
    x0, x1 = max(0, sl[2].start - pad), min(lab.shape[2], sl[2].stop + pad)
    sub = (lab[z0:z1, y0:y1, x0:x1] == i)
    V_vox = int(sub.sum())
    if V_vox < MIN_VOL_VOX:
        continue
    V_um3 = V_vox * VOX_VOL_UM3
    eq_diam_um = (6.0 * V_um3 / np.pi) ** (1.0 / 3.0)

    # Surface mesh via marching cubes -> exact area & vertex normals
    try:
        verts, faces, _, _ = measure.marching_cubes(
            sub.astype(np.uint8), level=0.5,
            spacing=(VOXEL_UM, VOXEL_UM, VOXEL_UM))
    except Exception:
        continue
    if len(verts) < 10:
        continue
    # area: sum of triangle areas
    tri = verts[faces]        # (n_tri, 3, 3)
    e1 = tri[:, 1] - tri[:, 0]; e2 = tri[:, 2] - tri[:, 0]
    tri_area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    A_um2 = float(tri_area.sum())
    # vertex centroid (weighted by incident triangle area)
    # simple volume-centroid works well for isolated beads
    # position of vertex averaged weighted by incident triangle area
    v_weights = np.zeros(len(verts))
    for k in range(3):
        np.add.at(v_weights, faces[:, k], tri_area / 3.0)
    C = (verts * v_weights[:, None]).sum(axis=0) / v_weights.sum()
    # radial distance from C to each vertex
    r = np.linalg.norm(verts - C, axis=1)       # um
    # weight by vertex area for unbiased surface stats
    w = v_weights / v_weights.sum()
    R_mean = float((r * w).sum())
    dev    = r - R_mean
    Ra     = float((np.abs(dev) * w).sum())
    Rq     = float(np.sqrt((dev ** 2 * w).sum()))
    Rt     = float(r.max() - r.min())
    # weighted central moments for skew / kurt
    m2     = float((dev ** 2 * w).sum())
    m3     = float((dev ** 3 * w).sum())
    m4     = float((dev ** 4 * w).sum())
    Rsk    = m3 / (m2 ** 1.5) if m2 > 0 else 0.0
    Rku    = m4 / (m2 ** 2)   if m2 > 0 else 0.0
    # sphericity (Wadell)
    sphericity = (np.pi ** (1.0 / 3.0)) * ((6.0 * V_um3) ** (2.0 / 3.0)) / A_um2
    ssa = A_um2 / V_um3

    rows.append(dict(
        label=i, V_vox=V_vox, V_um3=V_um3, A_um2=A_um2,
        eq_diam_um=eq_diam_um, R_mean_um=R_mean,
        Ra_um=Ra, Rq_um=Rq, Rt_um=Rt, Rsk=Rsk, Rku=Rku,
        sphericity=sphericity, ssa_um2_per_um3=ssa,
        cx=C[2], cy=C[1], cz=C[0],
    ))
    if len(rows) % 100 == 0:
        print(f'  {len(rows)} beads processed ({time.time()-t0:.1f}s)')

print(f'\n{len(rows)} beads analysed in {time.time()-t0:.1f}s')
del lab

# --------- CSV ---------------------------------------------------------------
csv_path = os.path.join(OUT, 'roughness_per_bead.csv')
with open(csv_path, 'w') as f:
    cols = ['label', 'V_vox', 'V_um3', 'A_um2', 'eq_diam_um', 'R_mean_um',
            'Ra_um', 'Rq_um', 'Rt_um', 'Rsk', 'Rku',
            'sphericity', 'ssa_um2_per_um3', 'cx', 'cy', 'cz']
    f.write(';'.join(cols) + '\n')
    for r in rows:
        f.write(';'.join(
            f'{r[c]:.4f}' if isinstance(r[c], float) else str(r[c])
            for c in cols) + '\n')
print(f'wrote {csv_path}')

# --------- summary text ------------------------------------------------------
arrs = {k: np.array([r[k] for r in rows]) for k in
        ['eq_diam_um', 'Ra_um', 'Rq_um', 'Rt_um',
         'Rsk', 'Rku', 'sphericity', 'ssa_um2_per_um3']}

def stat_line(name, a, unit=''):
    return (f'{name:<18}  mean={a.mean():8.3f}  median={np.median(a):8.3f}  '
            f'std={a.std():8.3f}  p5={np.percentile(a,5):8.3f}  '
            f'p95={np.percentile(a,95):8.3f} {unit}')

sumtxt = os.path.join(OUT, 'roughness_summary.txt')
with open(sumtxt, 'w') as f:
    f.write('Scan 1 glass-bead surface roughness baseline\n')
    f.write('=' * 55 + '\n\n')
    f.write(f'Voxel size: {VOXEL_UM:.3f} um  (half-voxel = {VOXEL_UM/2:.2f} um)\n')
    f.write(f'Beads analysed: {len(rows)}\n\n')
    f.write('Across-bead statistics:\n')
    f.write(stat_line('eq_diam_um',      arrs['eq_diam_um'],      '[um]') + '\n')
    f.write(stat_line('Ra_um',           arrs['Ra_um'],           '[um]') + '\n')
    f.write(stat_line('Rq_um',           arrs['Rq_um'],           '[um]') + '\n')
    f.write(stat_line('Rt_um',           arrs['Rt_um'],           '[um]') + '\n')
    f.write(stat_line('Rsk',             arrs['Rsk'])             + '\n')
    f.write(stat_line('Rku',             arrs['Rku'])             + '\n')
    f.write(stat_line('sphericity',      arrs['sphericity'])      + '\n')
    f.write(stat_line('ssa (A/V)',       arrs['ssa_um2_per_um3'], '[um^-1]') + '\n')
    f.write('\nInterpretation / comparison plan:\n')
    f.write('  Ra, Rq  -- magnitude of radial roughness (larger = rougher).\n')
    f.write('  Rt      -- worst-case peak-to-valley span.\n')
    f.write('  Rsk     -- >0: protruding peaks dominate; <0: pits dominate;\n')
    f.write('              ~0: symmetric (Gaussian-like).\n')
    f.write('  Rku     -- 3 = Gaussian; >3 spiky peaks/valleys; <3 flatter.\n')
    f.write('  sphericity -- 1.0 perfect sphere; drops with roughness & lobes.\n')
    f.write('  ssa     -- scales inversely with diameter; rougher beads\n')
    f.write('              have larger ssa at fixed diameter.\n')
    f.write('\nWhen the rough-material scan is ready, run the SAME script on\n')
    f.write('that scan (just change the DATA path / label file) and compare\n')
    f.write('histograms / means.\n')
print(f'wrote {sumtxt}')

# --------- plots -------------------------------------------------------------
# 1. distributions of Ra, Rq, Rt, sphericity  (one file per metric)
for key, unit, title, fname in [
    ('Ra_um',       'um', 'Ra — arithmetic mean roughness', 'hist_Ra.png'),
    ('Rq_um',       'um', 'Rq — RMS roughness',             'hist_Rq.png'),
    ('Rt_um',       'um', 'Rt — peak-to-valley',            'hist_Rt.png'),
    ('sphericity',  '',   'Wadell sphericity',              'hist_sphericity.png'),
    ('Rsk',         '',   'Rsk — surface height skewness',  'hist_Rsk.png'),
    ('Rku',         '',   'Rku — surface height kurtosis',  'hist_Rku.png'),
    ('eq_diam_um',  'um', 'Equivalent diameter',            'hist_eq_diam.png'),
    ('ssa_um2_per_um3', 'um^-1', 'Specific surface area (A/V)',
     'hist_ssa.png'),
]:
    a = arrs.get(key)
    if a is None:
        continue
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(a, bins=40, color='#1f77b4', edgecolor='black', alpha=0.75)
    ax.axvline(a.mean(), color='red', ls='--', lw=2,
               label=f'mean = {a.mean():.3f}' + (f' {unit}' if unit else ''))
    ax.axvline(np.median(a), color='green', ls='--', lw=2,
               label=f'median = {np.median(a):.3f}' + (f' {unit}' if unit else ''))
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel(f'{key} [{unit}]' if unit else key, labelpad=10)
    ax.set_ylabel('bead count', labelpad=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {os.path.join(OUT, fname)}')

# 2. Ra vs diameter (scale effect?)
fig, ax = plt.subplots(figsize=(13, 10))
ax.scatter(arrs['eq_diam_um'], arrs['Ra_um'], s=130, alpha=0.75,
           color='#1f77b4', edgecolor='black', linewidth=0.7)
ax.set_xlabel('Equivalent diameter [um]', labelpad=10)
ax.set_ylabel('Ra [um]', labelpad=10)
ax.set_title('Roughness vs bead size (scan 1)',
             fontweight='bold', pad=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'Ra_vs_diameter.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'wrote {os.path.join(OUT, "Ra_vs_diameter.png")}')

# 3. Rsk-Rku scatter (distribution shape)
fig, ax = plt.subplots(figsize=(13, 10))
ax.scatter(arrs['Rsk'], arrs['Rku'], s=130, alpha=0.75,
           color='#d62728', edgecolor='black', linewidth=0.7)
ax.axhline(3.0, color='k', ls='--', lw=2, alpha=0.7, label='Gaussian Rku=3')
ax.axvline(0.0, color='k', ls=':',  lw=2, alpha=0.7, label='Rsk=0 (symmetric)')
ax.set_xlabel('Rsk (skewness)', labelpad=10)
ax.set_ylabel('Rku (kurtosis)', labelpad=10)
ax.set_title('Surface height distribution shape per bead',
             fontweight='bold', pad=10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'Rsk_vs_Rku.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'wrote {os.path.join(OUT, "Rsk_vs_Rku.png")}')

print('\nDone. Use these outputs as the BASELINE for comparison against the')
print('upcoming high-roughness material scan. Re-run the same script on the')
print('new label TIFF and compare summary means and histograms.')
