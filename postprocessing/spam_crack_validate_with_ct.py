"""Standalone CT-slice validation figure (does NOT include the sparse
graph). For each transition we render three large CT slices (top / mid
/ bottom of the damage Z-extent) of the post-damage scan, with the
damage masks overlaid in the same red / cyan / amber palette as the
sparse-graph view so the two figures can be compared side by side.

Output (NEW file, separate from the sparse graph and from any other
existing figure):
    results_<PRE>/transition_AtoB/crack_validation_CT_slices.png
"""
import os
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch

DATA = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
OUT  = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>'

# ((scan_a, scan_b), CT_scan_to_show)
# We show the post-damage CT (scan B) because that is when the damage
# is visible in the imagery.
PAIRS = [((1, 2), 2), ((2, 3), 3)]

CLASSES = [
    ('damage_3d_crack.tif',         'PLANAR CRACK',     '#ff1744'),
    ('damage_3d_ice_fracture.tif',  'ICE FRACTURE',     '#00b8d4'),
    ('damage_3d_cavity.tif',        'COMPACTION VOID',  '#ffab00'),
]

PCTL_LO, PCTL_HI = 1, 99    # CT contrast clip

for (a, b), ct_scan in PAIRS:
    tdir = os.path.join(OUT, 'crack_analysis', f'transition_{a}to{b}')
    if not os.path.isdir(tdir):
        print(f'skip {a}->{b}: no folder'); continue
    print(f'\n=== {a}->{b}: validating with CT scan {ct_scan} ===')

    ct_path = os.path.join(DATA, f'ct_scan{ct_scan:02d}_aligned.tif')
    if not os.path.exists(ct_path):
        print(f'  CT volume missing: {ct_path}'); continue
    print('  loading CT')
    ct = tifffile.imread(ct_path)
    nz = ct.shape[0]

    # Load each damage class as a binary mask
    masks = []
    for fname, label, hex_col in CLASSES:
        p = os.path.join(tdir, fname)
        if not os.path.exists(p):
            masks.append((label, hex_col, None))
            continue
        m = tifffile.imread(p) > 0
        if m.shape != ct.shape:
            print(f'  {fname}: shape mismatch ({m.shape} vs CT {ct.shape}); skipping')
            masks.append((label, hex_col, None))
        else:
            masks.append((label, hex_col, m))

    # Pick three Z slices that contain damage (top, mid, bottom of damage extent)
    any_damage = np.zeros(ct.shape, dtype=bool)
    for label, hex_col, m in masks:
        if m is not None:
            any_damage |= m
    z_with_damage = np.where(any_damage.any(axis=(1, 2)))[0]
    if len(z_with_damage) == 0:
        print('  no damage voxels in any class; skipping')
        continue
    z_lo = int(z_with_damage[0])
    z_hi = int(z_with_damage[-1])
    z_mid = int((z_lo + z_hi) / 2)
    # User asked to drop the mid slice; show only top + bottom.
    z_picks = [z_lo + (z_hi - z_lo) // 6,    # near-top
               z_hi - (z_hi - z_lo) // 6]    # near-bottom
    print(f'  damage Z range: {z_lo}..{z_hi}; showing {z_picks}')

    vmin = float(np.percentile(ct, PCTL_LO))
    vmax = float(np.percentile(ct, PCTL_HI))

    # Two big CT slices in their own figure (no sparse graph here).
    fig, axes = plt.subplots(1, len(z_picks),
                              figsize=(8 * len(z_picks), 9), dpi=120,
                              gridspec_kw=dict(wspace=0.04))
    if len(z_picks) == 1:
        axes = [axes]

    for col, z in enumerate(z_picks):
        ax = axes[col]
        ax.imshow(ct[z], cmap='gray', vmin=vmin, vmax=vmax,
                  interpolation='nearest')
        for label, hex_col, m in masks:
            if m is None:
                continue
            slc = m[z]
            if not slc.any():
                continue
            rgba = np.zeros(slc.shape + (4,), dtype=np.float32)
            rgb_t = matplotlib.colors.to_rgb(hex_col)
            rgba[..., 0] = rgb_t[0]; rgba[..., 1] = rgb_t[1]; rgba[..., 2] = rgb_t[2]
            rgba[..., 3] = slc.astype(np.float32) * 0.62
            ax.imshow(rgba, interpolation='nearest')
        ax.set_axis_off()
        # Z label on slice
        ax.text(0.02, 0.97, f'Z = {z}', transform=ax.transAxes,
                fontsize=24, fontweight='bold',
                color='white', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.35',
                          facecolor='black', alpha=0.55,
                          edgecolor='none'))
    fig.suptitle(f'CT slices with damage overlay  -  scan {ct_scan}'
                 f'  (transition {a} -> {b})',
                 fontsize=28, fontweight='bold', y=0.99)

    # Shared legend below
    handles = [Patch(facecolor=h, edgecolor='black', label=l)
               for (l, h, m) in [(c[1], c[2], m) for c, m in zip(CLASSES, masks)]
               if m is not None]
    if handles:
        fig.legend(handles=handles, loc='lower center', ncol=len(handles),
                   fontsize=26, frameon=True, framealpha=0.95,
                   edgecolor='gray', bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(top=0.92, bottom=0.10, left=0.01, right=0.99)

    out_png = os.path.join(tdir, 'crack_validation_CT_slices.png')
    fig.savefig(out_png, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote {out_png}')

print('\nDone.')
