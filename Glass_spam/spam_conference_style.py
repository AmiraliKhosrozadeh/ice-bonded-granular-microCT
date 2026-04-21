"""
Conference-style renders — TWO figures:

  1. FINAL_displacement_field.png     — faded real-size bead spheres with
                                        one small uniform arrow per bead,
                                        arrows colored by |u|.
  2. FINAL_strain_field_volumetric.png — faded real-size bead spheres,
                                        colored by per-bead volumetric strain
                                        (from Delaunay tet neighborhood).
  3. FINAL_strain_field_deviatoric.png — same but von Mises deviatoric strain.

Style mirrors the reference conference slide: light background, simple title,
thin vertical colorbar on the right, large readable labels.

Run in WSL spam-venv:
    python /mnt/e/RPTU-images/CT_images/Glass/Glass_spam/spam_conference_style.py
"""

import os
import numpy as np
import pyvista as pv
from scipy.spatial import Delaunay
from PIL import Image

pv.OFF_SCREEN = True

SPAM_RESULTS = '/home/cak7496/spam-results'
OUT_DIR      = '/mnt/e/RPTU-images/CT_images/Glass/Glass_spam/results/3d'
os.makedirs(OUT_DIR, exist_ok=True)

DDIC_TSV  = os.path.join(SPAM_RESULTS, 'glass_ddic-ddic.tsv')

VOXEL_SIZE_UM = 24.766

# Physical bead radius: D50 ≈ 1750 µm -> radius ≈ 875 µm -> 35.3 voxels.
# Use slightly smaller so beads don't fully overlap when packed.
BEAD_RADIUS_VOX = 32

# Arrow parameters — uniform size, compact so scene stays readable
ARROW_LENGTH = 55             # voxels (~1.5x bead radius, fits between beads)
ARROW_TIP_LEN = 0.32
ARROW_TIP_R   = 0.11
ARROW_SHAFT_R = 0.055

# Sphere transparency for the FADED effect
BEAD_OPACITY = 0.28

BG_COLOR  = 'white'
TXT_COLOR = '#1a1a1a'
WINDOW    = (2400, 1800)   # back to 4:3, with post-crop to trim left blank

# ============================================================================
# Load DDIC TSV
# ============================================================================
with open(DDIC_TSV) as f:
    header = f.readline().strip().split('\t')
data = np.loadtxt(DDIC_TSV, skiprows=1, delimiter='\t')
col = {h: i for i, h in enumerate(header)}
beads = data[data[:, col['Label']].astype(int) > 0]

xyz = np.column_stack([beads[:, col['Xpos']],
                       beads[:, col['Ypos']],
                       beads[:, col['Zpos']]])
uvw = np.column_stack([beads[:, col['Xdisp']],
                       beads[:, col['Ydisp']],
                       beads[:, col['Zdisp']]])
rs  = beads[:, col['returnStatus']].astype(int)
err = beads[:, col['error']]
valid = (np.all(np.isfinite(uvw), axis=1) & (rs >= 1) & (err < 10000))
print(f'Valid beads: {int(valid.sum())} / {len(beads)}')

X = xyz[valid]
U = uvw[valid]
x_def = X + U

# ============================================================================
# Per-bead strain from Delaunay tet NEIGHBOURHOOD:
# For each bead, find all tets that contain it, average F over those tets,
# then get volumetric & deviatoric strain from the averaged F.
# ============================================================================
print('Computing per-bead strain from Delaunay neighbourhood...')
tri = Delaunay(X)
tets = tri.simplices
n_beads = len(X)

# Per-tet F
tet_F = np.empty((len(tets), 3, 3))
tet_ok = np.zeros(len(tets), dtype=bool)
for k in range(len(tets)):
    i0, i1, i2, i3 = tets[k]
    dX = np.column_stack([X[i1] - X[i0], X[i2] - X[i0], X[i3] - X[i0]])
    dx = np.column_stack([x_def[i1] - x_def[i0],
                          x_def[i2] - x_def[i0],
                          x_def[i3] - x_def[i0]])
    try:
        cond = np.linalg.cond(dX)
        F = dx @ np.linalg.inv(dX)
        tet_F[k] = F
        # Skip degenerate or wildly nonphysical tets
        if np.isfinite(cond) and cond < 500 and np.all(np.isfinite(F)):
            tet_ok[k] = True
    except np.linalg.LinAlgError:
        pass

# Per-bead: list of neighbouring tet indices
bead_to_tets = [[] for _ in range(n_beads)]
for k, tet in enumerate(tets):
    if not tet_ok[k]:
        continue
    for bid in tet:
        bead_to_tets[bid].append(k)

bead_vol_strain = np.full(n_beads, np.nan)
bead_mises      = np.full(n_beads, np.nan)
for b in range(n_beads):
    idx = bead_to_tets[b]
    if not idx:
        continue
    F_avg = tet_F[idx].mean(axis=0)
    if not np.all(np.isfinite(F_avg)):
        continue
    bead_vol_strain[b] = np.linalg.det(F_avg) - 1.0
    E = 0.5 * (F_avg.T @ F_avg - np.eye(3))
    Ed = E - np.trace(E) / 3 * np.eye(3)
    bead_mises[b] = np.sqrt(1.5 * np.sum(Ed * Ed))

# Clip display colour range to physically plausible bounds.
# Axial compression Fzz = 722/864 = 0.836  =>  vol strain = -0.164 mean.
# Most beads should fall inside ±0.2. Cap harder to keep colours meaningful.
finite_vs = bead_vol_strain[np.isfinite(bead_vol_strain)]
finite_mi = bead_mises[np.isfinite(bead_mises)]
vs_clim = (-0.30, +0.30)   # ±30% volume change — hard physical limit
mi_clim = ( 0.00,  0.30)   # deviatoric capped at 30%
print(f'vol_strain range displayed: {vs_clim}  '
      f'(median {np.median(finite_vs):+.3f})')
print(f'deviatoric range displayed: {mi_clim}  '
      f'(median {np.median(finite_mi):+.3f})')

# ============================================================================
# PyVista data
# ============================================================================
# Bead sphere glyphs at real bead radius
bead_pd = pv.PolyData(X.astype(np.float32))
bead_pd['vol_strain'] = bead_vol_strain
bead_pd['mises']      = bead_mises
bead_pd['|u|_um']     = np.linalg.norm(U, axis=1) * VOXEL_SIZE_UM
bead_pd['u']          = U
bead_spheres = bead_pd.glyph(
    geom=pv.Sphere(radius=BEAD_RADIUS_VOX, theta_resolution=24,
                   phi_resolution=24),
    scale=False, orient=False)

# Arrows: UNIFORM length, uniform direction from u/|u|
norms = np.linalg.norm(U, axis=1)
norms_safe = np.where(norms > 0, norms, 1.0)
unit_u = U / norms_safe[:, None]
arrow_pd = pv.PolyData(X.astype(np.float32))
arrow_pd['unit_u'] = unit_u.astype(np.float32)
arrow_pd['|u|_um'] = norms * VOXEL_SIZE_UM
arrows = arrow_pd.glyph(
    geom=pv.Arrow(tip_length=ARROW_TIP_LEN, tip_radius=ARROW_TIP_R,
                  shaft_radius=ARROW_SHAFT_R),
    orient='unit_u', scale=False, factor=ARROW_LENGTH)

# Camera
nx_est, ny_est, nz_est = 578, 587, 874
CAM_ISO = [(nx_est * 2.1, -ny_est * 1.5, nz_est * 1.7),
           (nx_est / 2, ny_est / 2, nz_est / 2),
           (0, 0, 1)]

def scalar_bar_args(title, pos_x=0.86):
    # Big colorbar title AND big numbers, matched size.
    return dict(title=title, title_font_size=60, label_font_size=60,
                color=TXT_COLOR, font_family='arial',
                position_x=pos_x, position_y=0.16, height=0.70, width=0.045,
                vertical=True, shadow=False, n_labels=5)

def add_title(p, text):
    """No-op — scene title removed per user request."""
    return

def crop_left_blank(path, bg_rgb=(255, 255, 255), tol=5):
    """Open the saved PNG, trim columns on the left that are solid background
    until we hit the first non-background pixel. Leaves a small margin."""
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img)
    # Column is "blank" if all pixels are within tol of background
    diff = np.abs(arr.astype(int) - np.asarray(bg_rgb).reshape(1, 1, 3)).sum(axis=2)
    non_bg_cols = (diff > tol).any(axis=0)
    if not non_bg_cols.any():
        return
    first_col = int(np.argmax(non_bg_cols))
    margin = 40   # keep 40 px margin so content isn't flush with the edge
    x0 = max(0, first_col - margin)
    if x0 == 0:
        return
    cropped = img.crop((x0, 0, arr.shape[1], arr.shape[0]))
    cropped.save(path)

# ============================================================================
# FIGURE 1 — Displacement field
# ============================================================================
print('Rendering displacement field...')
p = pv.Plotter(off_screen=True, window_size=WINDOW)
p.set_background(BG_COLOR)
p.enable_anti_aliasing('msaa', multi_samples=8)

# Faded bead spheres as context (real size)
p.add_mesh(bead_spheres, color='#cdd3dd', opacity=BEAD_OPACITY,
           smooth_shading=True, specular=0.1, ambient=0.4, diffuse=0.6)

# Arrows on top, colored by |u|
p.add_mesh(arrows, scalars='|u|_um', cmap='turbo',
           scalar_bar_args=scalar_bar_args('|u| (µm)'),
           lighting=True, specular=0.3, ambient=0.3, diffuse=0.9,
           show_scalar_bar=True)

p.camera_position = CAM_ISO
add_title(p, 'Displacement field')
out = os.path.join(OUT_DIR, 'FINAL_displacement_field.png')
p.screenshot(out); p.close()
crop_left_blank(out)
print(f'Saved: {out}')

# ============================================================================
# FIGURE 2 — Volumetric strain (bead colored by trace strain)
# ============================================================================
print('Rendering volumetric strain field...')
p = pv.Plotter(off_screen=True, window_size=WINDOW)
p.set_background(BG_COLOR)
p.enable_anti_aliasing('msaa', multi_samples=8)

# Background ghost spheres (grey) for beads with nan strain
nan_mask = ~np.isfinite(bead_vol_strain)
if nan_mask.any():
    ghost_pd = pv.PolyData(X[nan_mask].astype(np.float32))
    ghost_spheres = ghost_pd.glyph(
        geom=pv.Sphere(radius=BEAD_RADIUS_VOX, theta_resolution=20,
                       phi_resolution=20),
        scale=False, orient=False)
    p.add_mesh(ghost_spheres, color='#d8d8d8', opacity=0.4,
               smooth_shading=True)

# Coloured spheres (real size, less faded so the color shows)
good_mask = np.isfinite(bead_vol_strain)
good_pd = pv.PolyData(X[good_mask].astype(np.float32))
good_pd['vol_strain'] = bead_vol_strain[good_mask]
good_spheres = good_pd.glyph(
    geom=pv.Sphere(radius=BEAD_RADIUS_VOX, theta_resolution=24,
                   phi_resolution=24),
    scale=False, orient=False)
p.add_mesh(good_spheres, scalars='vol_strain', cmap='seismic_r',
           clim=vs_clim,
           scalar_bar_args=scalar_bar_args('Volumetric strain'),
           smooth_shading=True, specular=0.2, ambient=0.4, diffuse=0.85,
           show_scalar_bar=True)

p.camera_position = CAM_ISO
add_title(p, 'Volumetric strain')
out = os.path.join(OUT_DIR, 'FINAL_strain_field_volumetric.png')
p.screenshot(out); p.close()
crop_left_blank(out)
print(f'Saved: {out}')

# ============================================================================
# FIGURE 3 — Deviatoric (von Mises) strain
# ============================================================================
print('Rendering deviatoric strain field...')
p = pv.Plotter(off_screen=True, window_size=WINDOW)
p.set_background(BG_COLOR)
p.enable_anti_aliasing('msaa', multi_samples=8)

if nan_mask.any():
    p.add_mesh(ghost_spheres, color='#d8d8d8', opacity=0.4,
               smooth_shading=True)

good_pd2 = pv.PolyData(X[good_mask].astype(np.float32))
good_pd2['mises'] = bead_mises[good_mask]
good_spheres2 = good_pd2.glyph(
    geom=pv.Sphere(radius=BEAD_RADIUS_VOX, theta_resolution=24,
                   phi_resolution=24),
    scale=False, orient=False)
p.add_mesh(good_spheres2, scalars='mises', cmap='plasma',
           clim=mi_clim,
           scalar_bar_args=scalar_bar_args('Deviatoric strain'),
           smooth_shading=True, specular=0.2, ambient=0.4, diffuse=0.85,
           show_scalar_bar=True)

p.camera_position = CAM_ISO
add_title(p, 'Deviatoric strain')
out = os.path.join(OUT_DIR, 'FINAL_strain_field_deviatoric.png')
p.screenshot(out); p.close()
crop_left_blank(out)
print(f'Saved: {out}')

# ============================================================================
# FIGURE 4 — Side-by-side slide layout (displacement + volumetric strain)
# ============================================================================
print('Compositing side-by-side...')
plot = pv.Plotter(off_screen=True, window_size=(3400, 1800),
                  shape=(1, 2), border=False)
plot.set_background(BG_COLOR, all_renderers=True)

# Left: displacement
plot.subplot(0, 0)
plot.set_background(BG_COLOR)
plot.enable_anti_aliasing('msaa', multi_samples=4)
plot.add_mesh(bead_spheres, color='#cdd3dd', opacity=BEAD_OPACITY,
              smooth_shading=True, ambient=0.4, diffuse=0.6)
plot.add_mesh(arrows, scalars='|u|_um', cmap='turbo',
              scalar_bar_args=scalar_bar_args('|u| (µm)'),
              lighting=True, specular=0.3, ambient=0.3, diffuse=0.9)
plot.camera_position = CAM_ISO

# Right: volumetric strain
plot.subplot(0, 1)
plot.set_background(BG_COLOR)
plot.enable_anti_aliasing('msaa', multi_samples=4)
if nan_mask.any():
    plot.add_mesh(ghost_spheres, color='#d8d8d8', opacity=0.4)
plot.add_mesh(good_spheres, scalars='vol_strain', cmap='seismic_r',
              clim=vs_clim,
              scalar_bar_args=scalar_bar_args('Volumetric strain'),
              smooth_shading=True, specular=0.2, ambient=0.4)
plot.camera_position = CAM_ISO

out = os.path.join(OUT_DIR, 'FINAL_conference_sidebyside.png')
plot.screenshot(out); plot.close()
crop_left_blank(out)
print(f'Saved: {out}')

print('\nDONE. Four conference-style figures in:', OUT_DIR)
