"""Combined 3D failure-mode visualization per transition.

Reads existing CSVs (no re-computation):
  transition_AtoB/failure_modes.csv      -- bond-level mode per pair
  transition_AtoB/bead_delamination.csv  -- per-bead Δdryness

Renders ONE 3D image per transition:
  - BEADS drawn as spheres colored by Δdryness (red = delaminated, blue = re-bonded)
  - BONDS drawn as tubes colored by failure mode:
       DELAMINATION = red, COHESIVE = blue, CRUSHED = green,
       MIXED = orange, UNCLEAR = gray (thin + transparent)

Output:
  transition_AtoB/failure_combined_3D.png
"""
import os
import numpy as np
import pyvista as pv
pv.OFF_SCREEN = True

OUT = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>'
TRANSITIONS = [(1, 2), (2, 3)]

MODE_RGB = {
    'DELAMINATION': '#d62728',
    'COHESIVE':     '#1f77b4',
    'CRUSHED':      '#2ca02c',
    'MIXED':        '#ff7f0e',
    'UNCLEAR':      '#bbbbbb',
}
MODE_CODE = {'DELAMINATION': 0, 'COHESIVE': 1, 'CRUSHED': 2,
             'MIXED': 3, 'UNCLEAR': 4}
CMAP_BONDS = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#bbbbbb']

# Camera (flipped Z, Dragonfly convention)
nx_e, ny_e, nz_e = 803, 707, 1241   # T5_HR common aligned frame
CAM = [(nx_e * 2.1, -ny_e * 1.5, -nz_e * 0.7),
       (nx_e / 2, ny_e / 2, nz_e / 2),
       (0, 0, -1)]

def read_failure_modes(path):
    """Yield (mode, cA_xyz, cB_xyz)."""
    rows = []
    with open(path) as f:
        header = next(f).strip().split(';')
        idx = {h: i for i, h in enumerate(header)}
        for line in f:
            parts = line.strip().split(';')
            if len(parts) < len(header):
                continue
            state = parts[idx['state']]
            if state == 'SURVIVING':
                continue
            mode = parts[idx['mode']]
            cA = (float(parts[idx['cA_x']]),
                  float(parts[idx['cA_y']]),
                  float(parts[idx['cA_z']]))
            cB = (float(parts[idx['cB_x']]),
                  float(parts[idx['cB_y']]),
                  float(parts[idx['cB_z']]))
            rows.append((mode, cA, cB))
    return rows

def read_bead_delam(path):
    """Yield (delta, (x, y, z))."""
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split(';')
            if len(parts) < 8:
                continue
            dd = float(parts[4])
            x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
            rows.append((dd, (x, y, z)))
    return rows

for a, b in TRANSITIONS:
    tdir = os.path.join(OUT, f'transition_{a}to{b}')
    fm_csv = os.path.join(tdir, 'failure_modes.csv')
    bd_csv = os.path.join(tdir, 'bead_delamination.csv')
    if not os.path.exists(fm_csv) or not os.path.exists(bd_csv):
        print(f'skip {a}->{b}: missing CSV'); continue

    fm = read_failure_modes(fm_csv)
    bd = read_bead_delam(bd_csv)
    print(f'{a}->{b}: {len(fm)} bonds, {len(bd)} beads')
    if not fm and not bd:
        continue

    p = pv.Plotter(off_screen=True, window_size=(2600, 1900))
    p.set_background('white')
    p.enable_anti_aliasing('msaa', multi_samples=8)

    # --- bead spheres colored by Δdryness ---------------------------------
    if bd:
        pts_b = np.array([list(r[1]) for r in bd], dtype=np.float32)
        deltas = np.array([r[0] for r in bd], dtype=np.float32)
        cloud = pv.PolyData(pts_b); cloud['delta'] = deltas
        glyph = cloud.glyph(geom=pv.Sphere(radius=22, theta_resolution=16,
                                            phi_resolution=16),
                            scale=False, orient=False)
        p.add_mesh(glyph, scalars='delta', cmap='coolwarm',
                   clim=(-0.3, 0.3), smooth_shading=True,
                   scalar_bar_args=dict(title='Δ dryness (bead)',
                       title_font_size=46, label_font_size=46,
                       color='#1a1a1a', font_family='arial',
                       position_x=0.86, position_y=0.52,
                       height=0.38, width=0.035, vertical=True,
                       shadow=False, n_labels=5))

    # --- bond tubes colored by bond mode ----------------------------------
    # Render non-UNCLEAR bonds as thick tubes on top, UNCLEAR as thin tubes
    # so the informative modes stand out.
    def bond_mesh(subset_modes, radius, opacity):
        pts = []; cells = []; codes = []
        for mode, cA, cB in fm:
            if mode not in subset_modes:
                continue
            pts.append(cA); pts.append(cB)
            idx = len(pts) - 2
            cells.append([2, idx, idx + 1])
            codes.append(MODE_CODE[mode])
        if not pts:
            return None, []
        pd = pv.PolyData()
        pd.points = np.asarray(pts, dtype=np.float32)
        pd.lines = np.asarray(cells).ravel().astype(np.int64)
        pd['mode'] = np.repeat(codes, 2)
        return pd.tube(radius=radius, n_sides=10), codes

    # Thin translucent: UNCLEAR (context)
    tubes_u, _ = bond_mesh({'UNCLEAR'}, 2.0, 0.35)
    if tubes_u is not None:
        p.add_mesh(tubes_u, scalars='mode', cmap=CMAP_BONDS, clim=(0, 4),
                   opacity=0.35, show_scalar_bar=False)
    # Thick: the informative failure modes
    tubes_f, _ = bond_mesh({'DELAMINATION', 'COHESIVE', 'CRUSHED', 'MIXED'},
                           4.5, 1.0)
    if tubes_f is not None:
        p.add_mesh(tubes_f, scalars='mode', cmap=CMAP_BONDS, clim=(0, 4),
                   show_scalar_bar=False)

    # Legend for bond modes
    try:
        p.add_legend(labels=[
            ('DELAMINATION', MODE_RGB['DELAMINATION']),
            ('COHESIVE',     MODE_RGB['COHESIVE']),
            ('CRUSHED',      MODE_RGB['CRUSHED']),
            ('MIXED',        MODE_RGB['MIXED']),
            ('UNCLEAR',      MODE_RGB['UNCLEAR']),
        ], size=(0.22, 0.22), loc='upper right', face='rectangle',
           bcolor='#f5f5f5', border=True)
    except Exception:
        pass

    p.camera_position = CAM
    out = os.path.join(tdir, 'failure_combined_3D.png')
    p.screenshot(out); p.close()
    print(f'  wrote {out}')

print('\nDone.')
