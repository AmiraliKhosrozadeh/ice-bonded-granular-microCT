"""Export labeled damage volume per transition for 3D viewing in Dragonfly.

Re-derives the damage voxels (solid_A AND air_B, in the aligned frame),
labels connected components, then maps each component to its class
(read from results_<tag>/crack_analysis/transition_AtoB/damage.csv).

Output per transition:
    damage_3d_<phase_class>.tif    binary uint8 mask per class
    damage_3d_classes.tif          uint8 labelled volume:
                                     0 = none / outside specimen
                                     1 = CRACK
                                     2 = DELAMINATION
                                     3 = ICE-FRACTURE
                                     4 = CAVITY
                                     5 = FULLY-DEVELOPED CRACK
    damage_3d_README.txt           voxel-label legend

Drop these TIFFs into Dragonfly as a Channel, convert to MultiROI for
class-colored 3D visualization (see Dragonfly snippet below).

Run in WSL spam-venv:
    python /mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/spam_export_damage_3d.py
"""
import os, sys, time
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from scan_meta import VOXEL_UM

DATA = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
RES  = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>/crack_analysis'

TRANSITIONS = [(1, 2), (2, 3)]
TH = {
    1: dict(air=7000,    ig=18032.7, ga=47275),
    2: dict(air=2139.4,  ig=6271.5,  ga=99999),
    3: dict(air=2015.4,  ig=6539.5,  ga=99999),
}

# Class -> uint8 voxel label
CLASS_VAL = {
    'CRACK':        1,
    'DELAMINATION': 2,
    'ICE-FRACTURE': 3,
    'CAVITY':       4,
}
FULL_CRACK_VAL = 5

def reconstruct_phases(ct, mask, th):
    p = np.zeros(ct.shape, dtype=np.uint8)
    inside = mask > 0
    p[inside & (ct < th['air'])] = 1
    p[inside & (ct >= th['air']) & (ct < th['ig'])] = 2
    p[inside & (ct >= th['ig'])  & (ct < th['ga'])] = 3
    return p

# ============================================================================
for a, b in TRANSITIONS:
    print(f'\n=== transition {a} -> {b} ===')
    tdir = os.path.join(RES, f'transition_{a}to{b}')
    csv_path = os.path.join(tdir, 'damage.csv')
    if not os.path.exists(csv_path):
        print(f'  no damage.csv, skipping')
        continue
    df = pd.read_csv(csv_path, sep=';')
    print(f'  {len(df)} components in damage.csv')

    t0 = time.time()
    ct_a   = tifffile.imread(os.path.join(DATA, f'ct_scan{a:02d}_aligned.tif'))
    mask_a = tifffile.imread(os.path.join(DATA, f'specimen_mask_scan{a:02d}_aligned.tif'))
    ct_b   = tifffile.imread(os.path.join(DATA, f'ct_scan{b:02d}_aligned.tif'))
    mask_b = tifffile.imread(os.path.join(DATA, f'specimen_mask_scan{b:02d}_aligned.tif'))
    print(f'  loaded ({time.time()-t0:.1f}s)  shape={ct_a.shape}')

    phases_a = reconstruct_phases(ct_a, mask_a > 0, TH[a]); del ct_a
    phases_b = reconstruct_phases(ct_b, mask_b > 0, TH[b]); del ct_b
    inside_both = (mask_a > 0) & (mask_b > 0); del mask_a, mask_b
    solid_a = (phases_a == 2) | (phases_a == 3); del phases_a
    air_b   = (phases_b == 1);                   del phases_b
    damage = solid_a & air_b & inside_both; del solid_a, air_b, inside_both
    print(f'  damage voxels: {damage.sum():,}')

    # Re-label connected components (must match the IDs used in damage.csv —
    # ndi.label is deterministic so re-running gives the same numbering)
    lab, n_cc = ndi.label(damage, structure=np.ones((3,3,3), dtype=bool))
    print(f'  {n_cc} components')

    # Build cc -> class lookup from CSV
    id_to_class = dict(zip(df['id'].astype(int), df['class']))
    if 'fully_developed' in df.columns:
        id_to_full = dict(zip(df['id'].astype(int),
                              df['fully_developed'].astype(str).str.lower() == 'true'))
    else:
        id_to_full = {}
    classes_vol = np.zeros_like(lab, dtype=np.uint8)
    # Assign class label to every voxel of every CC
    # (vectorized via take on a LUT)
    max_id = int(lab.max())
    lut = np.zeros(max_id + 1, dtype=np.uint8)
    for cid, cls in id_to_class.items():
        if cid <= max_id and cls in CLASS_VAL:
            lut[cid] = CLASS_VAL[cls]
    # Promote fully-developed cracks to their special label
    for cid, full in id_to_full.items():
        if full and cid <= max_id:
            lut[cid] = FULL_CRACK_VAL
    classes_vol = lut[lab]

    # Write
    out_classes = os.path.join(tdir, 'damage_3d_classes.tif')
    tifffile.imwrite(out_classes, classes_vol, photometric='minisblack')
    print(f'  wrote {out_classes}')

    # Per-class binary masks (handy if user wants individual threshold ROIs)
    for cls_name, cls_val in CLASS_VAL.items():
        cls_bin = (classes_vol == cls_val).astype(np.uint8) * 255
        if not cls_bin.any():
            continue
        out = os.path.join(tdir, f'damage_3d_{cls_name.lower().replace("-", "_")}.tif')
        tifffile.imwrite(out, cls_bin, photometric='minisblack')
        print(f'  wrote {out}  ({int(cls_bin.sum()/255):,} voxels)')

    # Fully-developed crack (separate)
    fc = (classes_vol == FULL_CRACK_VAL).astype(np.uint8) * 255
    if fc.any():
        out = os.path.join(tdir, 'damage_3d_fully_developed_crack.tif')
        tifffile.imwrite(out, fc, photometric='minisblack')
        print(f'  wrote {out}  ({int(fc.sum()/255):,} voxels)')

    # 3D render of all classes — legend in bordered box BELOW the image
    try:
        import pyvista as pv
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        _FS = 24
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': _FS,
                             'savefig.dpi': 220, 'savefig.bbox': 'tight'})
        CLASS_COL = {'CRACK': '#D62728', 'DELAMINATION': '#FF7F0E',
                     'ICE-FRACTURE': '#1F77B4', 'CAVITY': '#2CA02C',
                     'FULL-CRACK': '#9467BD'}
        VAL_TO_NAME = {1: 'CRACK', 2: 'DELAMINATION', 3: 'ICE-FRACTURE',
                       4: 'CAVITY', 5: 'FULL-CRACK'}

        plotter = pv.Plotter(off_screen=True, window_size=(1600, 1200))
        plotter.set_background('white')
        plotter.camera.up = (0, 0, -1)
        legend_entries = []
        for val, name in VAL_TO_NAME.items():
            cls_mask = (classes_vol == val).astype(np.uint8)
            if not cls_mask.any():
                continue
            grid = pv.wrap(cls_mask)
            surf = grid.contour([0.5])
            plotter.add_mesh(surf, color=CLASS_COL[name], opacity=0.85)
            legend_entries.append((name, CLASS_COL[name]))
        plotter.view_isometric()

        # Save via screenshot then compose with matplotlib legend below
        out_3d = os.path.join(tdir, 'damage_3D.png')
        tmp = out_3d + '.__tmp__.png'
        plotter.screenshot(tmp, window_size=(1600, 1200))
        plotter.close()
        img = plt.imread(tmp)
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(2, 1, height_ratios=[10, 1.4], hspace=0.05)
        ax_img = fig.add_subplot(gs[0]); ax_img.imshow(img); ax_img.axis('off')
        ax_lg  = fig.add_subplot(gs[1]); ax_lg.axis('off')
        handles = [Patch(facecolor=c, edgecolor='black', linewidth=1.5,
                         label=l) for l, c in legend_entries]
        leg = ax_lg.legend(handles=handles, loc='center',
                           ncol=min(len(handles), 4),
                           fontsize=_FS,
                           frameon=True, fancybox=False,
                           edgecolor='black', framealpha=1.0,
                           borderpad=0.8, handlelength=2.2)
        leg.get_frame().set_linewidth(2.0)
        fig.savefig(out_3d, dpi=220, bbox_inches='tight', facecolor='white')
        plt.close(fig); os.remove(tmp)
        print(f'  wrote {out_3d}')
    except Exception as e:
        print(f'  3D render failed: {e}')

    # README in the transition dir
    with open(os.path.join(tdir, 'damage_3d_README.txt'), 'w') as f:
        f.write('damage_3d_classes.tif  uint8 voxel labels:\n')
        f.write('  0 = none / outside specimen\n')
        for cls, val in CLASS_VAL.items():
            f.write(f'  {val} = {cls}\n')
        f.write(f'  {FULL_CRACK_VAL} = FULLY-DEVELOPED CRACK\n')
        f.write('\nVoxel size: 24.7660229 um (isotropic).\n')
        f.write('Frame: ALIGNED — same coordinates as the aligned CT/mask/bead-labels TIFFs.\n')

print('\nDone.')
