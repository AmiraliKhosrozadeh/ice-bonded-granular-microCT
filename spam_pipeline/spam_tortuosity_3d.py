"""3D view of geodesic tortuosity through the ice phase.

For each scan:
  1. Build the ice phase mask from CT + specimen mask + threshold.
  2. Downsample (memory).
  3. Compute the geodesic distance through the ice phase from the top z-slice
     using `skimage.graph.MCP_Geometric`.
  4. Compute the local tortuosity field   τ_local(x) = geodist(x) / |z(x) - z_src|
     so that τ → 1 for straight paths and τ ≫ 1 around obstacles / cracks.
  5. Render with PyVista — semi-transparent glass beads as context, ice phase
     volume coloured by τ.

Outputs (per scan):
  results_<PRE>/tortuosity_3d/tau3d_scan{NN}.png

Plus titles in tortuosity_3d/titles.txt (titles do NOT go on the figures).
"""
import os, sys, time
import numpy as np
import tifffile
from skimage.graph import MCP_Geometric
from scipy import ndimage as ndi

# Headless rendering — must come BEFORE `import pyvista`
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.image import imread

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from scan_meta import VOXEL_UM

DATA = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
OUT  = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>/tortuosity_3d'
os.makedirs(OUT, exist_ok=True)

TRANSITIONS = [(1, 2), (2, 3)]
ALL_SCANS   = sorted({s for ab in TRANSITIONS for s in ab})

TH = {
    1: dict(air=7000,    ig=18032.7, ga=47275),
    2: dict(air=2139.4,  ig=6271.5,  ga=99999),
    3: dict(air=2015.4,  ig=6539.5,  ga=99999),
}

DOWNSAMPLE = 4
TAU_VMIN, TAU_VMAX = 1.0, 2.5      # color range
RENDER_W, RENDER_H = 1600, 1200    # PNG size before legend overlay

# ---------------------------------------------------------------------------
def downsample_bool(mask, k):
    if k == 1: return mask
    Z, Y, X = mask.shape
    Z2, Y2, X2 = Z // k, Y // k, X // k
    cropped = mask[:Z2 * k, :Y2 * k, :X2 * k]
    blocks  = cropped.reshape(Z2, k, Y2, k, X2, k)
    # Majority vote → True if > half block voxels are True
    return blocks.sum(axis=(1, 3, 5)) > (k**3 // 2)

def compute_tau_field(ice_mask):
    """Geodesic-to-Euclidean ratio through the LARGEST connected ice component.

    The ice phase in early scans is highly fragmented; using `ice_mask` as a
    whole (where many islands are mutually disconnected) gives NaN almost
    everywhere.  We restrict the geodesic computation to the largest
    connected component (the dominant ice network) and seed from its
    topmost z-slice.

    Voxels outside that component are NaN in the returned field.
    """
    if not ice_mask.any():
        return np.full(ice_mask.shape, np.nan, dtype=np.float32), 0

    # 26-connectivity for thicker connections through diagonals
    cc, n_cc = ndi.label(ice_mask, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    largest_label = int(sizes.argmax())
    if sizes[largest_label] == 0:
        return np.full(ice_mask.shape, np.nan, dtype=np.float32), 0
    largest = cc == largest_label

    z_has = np.any(largest, axis=(1, 2))
    z_src = int(np.argmax(z_has))

    cost = np.where(largest, 1.0, np.inf).astype(np.float64)
    mcp = MCP_Geometric(cost, sampling=(1.0, 1.0, 1.0))
    starts = [(z_src, yi, xi) for (yi, xi) in zip(*np.where(largest[z_src]))]
    if len(starts) == 0:
        return np.full(ice_mask.shape, np.nan, dtype=np.float32), int(sizes[largest_label])
    geo, _ = mcp.find_costs(starts)

    Z, Y, X = ice_mask.shape
    eucl = np.abs(np.arange(Z, dtype=np.float64) - z_src).reshape(Z, 1, 1)
    eucl = np.broadcast_to(eucl, (Z, Y, X))

    tau = np.full(ice_mask.shape, np.nan, dtype=np.float32)
    valid = largest & np.isfinite(geo) & (eucl > 0)
    tau[valid] = (geo[valid] / eucl[valid]).astype(np.float32)
    return tau, int(sizes[largest_label])

def render_tau_3d(tau, glass_mask, voxel_um, out_png, title_for_legend):
    """Render τ volume — use a single-scalar grid masked to ice voxels only,
    no threshold/extract_surface (which drop the scalar coloring on
    UnstructuredGrid output in pyvista 0.47)."""
    Z, Y, X = tau.shape
    spacing = (voxel_um, voxel_um, voxel_um)

    # Build a value array where non-ice voxels have NaN (so they're ignored
    # by the renderer's clim) and ice voxels carry the real τ value.
    # Then convert to a PolyData of voxel cubes via marching_cubes-on-mask.
    ice_mask = np.isfinite(tau)

    p = pv.Plotter(off_screen=True, window_size=(RENDER_W, RENDER_H))
    p.set_background('white')

    if ice_mask.any():
        # marching cubes on the ice mask gives a clean ice surface.
        # IMPORTANT: skimage returns verts in NUMPY axis order (z, y, x);
        # PyVista expects Cartesian (x, y, z).  Reorder columns so the
        # cylinder's long axis (numpy z) ends up on PyVista's Z axis →
        # the cylinder will stand VERTICAL when we set up=(0,0,1).
        from skimage import measure
        verts_zyx, faces, _, _ = measure.marching_cubes(ice_mask.astype(np.uint8),
                                                        level=0.5,
                                                        spacing=spacing)
        # Vertex-to-voxel index lookup for τ (use numpy order)
        idx_z = np.clip(np.round(verts_zyx[:, 0] / spacing[0]).astype(int), 0, Z - 1)
        idx_y = np.clip(np.round(verts_zyx[:, 1] / spacing[1]).astype(int), 0, Y - 1)
        idx_x = np.clip(np.round(verts_zyx[:, 2] / spacing[2]).astype(int), 0, X - 1)
        tau_vert = tau[idx_z, idx_y, idx_x]
        if np.isnan(tau_vert).any():
            tau_vert = np.nan_to_num(tau_vert, nan=float(np.nanmin(tau)))

        # Reorder to PyVista Cartesian (x, y, z)
        verts_xyz = verts_zyx[:, [2, 1, 0]]

        n_faces = faces.shape[0]
        pv_faces = np.empty((n_faces, 4), dtype=np.int64)
        pv_faces[:, 0] = 3
        pv_faces[:, 1:] = faces
        surf = pv.PolyData(verts_xyz, pv_faces.ravel())
        surf['tau'] = tau_vert
        print(f'  ice surface: {surf.n_points} verts, {surf.n_cells} faces, '
              f'τ on verts: min={float(tau_vert.min()):.3f}  '
              f'max={float(tau_vert.max()):.3f}')

        p.add_mesh(surf, scalars='tau', cmap='turbo',
                   clim=(TAU_VMIN, TAU_VMAX), opacity=1.0,
                   smooth_shading=True, show_scalar_bar=False)

    # Compression axis is Z (long axis of the cylinder).  We want it VERTICAL
    # AND flipped (top of specimen at TOP of the screen) — Dragonfly convention
    # has Z+ pointing DOWN, so we use up=(0,0,-1) to put low-z at the top.
    p.view_xz()
    p.camera.up = (0, 0, -1)
    p.camera.azimuth   = -30
    p.camera.elevation =  15
    p.camera.zoom(1.10)

    # XYZ orientation triad — small marker in lower-left corner of viewport.
    # No PyVista triad — the BIG legible XYZ indicator is drawn by matplotlib
    # in the composition step below.

    tmp_png = out_png + '.raw.png'
    p.screenshot(tmp_png)
    p.close()

    # Compose with matplotlib so we get a proper colorbar legend BELOW the
    # image (per project plot-style rules: legend below, big fonts, framed).
    img = imread(tmp_png)
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_axes([0.0, 0.18, 1.0, 0.78])
    ax.imshow(img); ax.set_axis_off()

    # Single, BIG XYZ orientation indicator (lower-left), via 3D matplotlib.
    ax3d = fig.add_axes([0.01, 0.01, 0.32, 0.32], projection='3d')
    L = 1.0
    ax3d.quiver(0, 0, 0,  L, 0, 0, color='#d62728', linewidth=8,
                arrow_length_ratio=0.22)
    ax3d.quiver(0, 0, 0,  0, L, 0, color='#2ca02c', linewidth=8,
                arrow_length_ratio=0.22)
    ax3d.quiver(0, 0, 0,  0, 0, L, color='#1f77b4', linewidth=8,
                arrow_length_ratio=0.22)
    ax3d.text(L*1.30, 0, 0, 'X', color='#d62728', fontsize=36, fontweight='bold')
    ax3d.text(0, L*1.30, 0, 'Y', color='#2ca02c', fontsize=36, fontweight='bold')
    ax3d.text(0, 0, L*1.30, 'Z', color='#1f77b4', fontsize=36, fontweight='bold')
    ax3d.set_xlim(-0.3, 1.5); ax3d.set_ylim(-0.3, 1.5); ax3d.set_zlim(-0.3, 1.5)
    ax3d.set_axis_off()
    # Match the PyVista camera: up=(0,0,-1), azim -30, elev 15.
    ax3d.view_init(elev=15, azim=-30)
    ax3d.invert_zaxis()

    # Colorbar below the image
    cb_ax = fig.add_axes([0.30, 0.06, 0.55, 0.035])
    norm = plt.Normalize(vmin=TAU_VMIN, vmax=TAU_VMAX)
    sm = plt.cm.ScalarMappable(cmap='turbo', norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cb_ax, orientation='horizontal')
    cbar.set_label('Tortuosity  τ', fontsize=22, fontweight='bold', labelpad=10)
    cbar.ax.tick_params(labelsize=18, width=1.5)
    cbar.outline.set_linewidth(1.8)
    fig.savefig(out_png, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    if os.path.exists(tmp_png): os.remove(tmp_png)
    print(f'  wrote {out_png}')

def reconstruct_ice_glass(ct, mask, th):
    inside = mask > 0
    ice    = inside & (ct >= th['air']) & (ct < th['ig'])
    glass  = inside & (ct >= th['ig']) & (ct < th['ga'])
    return ice, glass

# ---------------------------------------------------------------------------
print(f'3D tortuosity — scans: {ALL_SCANS}')
print(f'  data dir: {DATA}')
print(f'  output  : {OUT}')
print(f'  downsample: {DOWNSAMPLE}, voxel: {VOXEL_UM*DOWNSAMPLE:.1f} µm')

titles = []
for s in ALL_SCANS:
    print(f'\n=== scan {s} ===')
    t0 = time.time()
    ct   = tifffile.imread(os.path.join(DATA, f'ct_scan{s:02d}_aligned.tif'))
    mask = tifffile.imread(os.path.join(DATA, f'specimen_mask_scan{s:02d}_aligned.tif'))
    print(f'  loaded shape={ct.shape}  ({time.time()-t0:.1f}s)')

    ice, glass = reconstruct_ice_glass(ct, mask, TH[s])
    del ct, mask

    if DOWNSAMPLE > 1:
        ice   = downsample_bool(ice,   DOWNSAMPLE)
        glass = downsample_bool(glass, DOWNSAMPLE)
    print(f'  ice voxels  : {ice.sum()}   shape={ice.shape}')
    print(f'  glass voxels: {glass.sum()}')

    t1 = time.time()
    tau, largest_cc_size = compute_tau_field(ice)
    n_valid = np.isfinite(tau).sum()
    if n_valid == 0:
        print(f'  WARNING: scan {s} has no connected ice path; skipping render')
        continue
    print(f'  tau field: largest CC = {largest_cc_size} vox  '
          f'({100 * largest_cc_size / max(ice.sum(),1):.1f}% of ice)  '
          f'mean(τ)={np.nanmean(tau):.3f}  max(τ)={np.nanmax(tau):.3f}'
          f'  ({time.time()-t1:.1f}s)')

    out_png = os.path.join(OUT, f'tau3d_scan{s:02d}.png')
    render_tau_3d(tau, glass, VOXEL_UM * DOWNSAMPLE, out_png,
                  title_for_legend=f'Scan {s}')
    titles.append(f'tau3d_scan{s:02d}.png\n  Geodesic tortuosity through ice phase, scan {s}')

# Titles file (per project rule: titles in text, not on plots)
with open(os.path.join(OUT, 'titles.txt'), 'w', encoding='utf-8') as f:
    f.write('3D tortuosity figures — titles\n')
    f.write('=' * 60 + '\n\n')
    for line in titles:
        f.write(line + '\n\n')
print(f'\nwrote {os.path.join(OUT, "titles.txt")}')
print('\nDone.')
