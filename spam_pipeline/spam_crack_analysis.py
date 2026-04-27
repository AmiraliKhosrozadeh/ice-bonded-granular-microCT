"""Per-transition crack and damage analysis.

For each transition (A -> B), identify voxels that were SOLID (ice or glass)
in scan A and became AIR in scan B. That's the damage signal — voxels where
solid material was displaced or destroyed.

Then, connected-component label the damage volume, and for each component
compute geometric descriptors that distinguish:

  CRACK         — planar, thin (<2 vox local thickness)
  DELAMINATION  — most surface voxels touch glass surface
                  (interface failure between bead and ice)
  ICE-FRACTURE  — most surface voxels touch ice (cohesive failure in the ice)
  CAVITY        — roughly isotropic blob (compaction void, not a crack)

A "FULLY-DEVELOPED CRACK" is flagged separately:
  CRACK class AND percolates to specimen boundary (cylinder wall, top, bottom)
  AND length > 50% of specimen extent.

Why this is better than the 'trapped air' classifier:
  - 'trapped air' is a single-scan, topological label. It catches pre-existing
    porosity and partial cracks indistinguishably, and DOES NOT see fully
    developed cracks (they connect to outside air = no longer 'trapped').
  - The damage-delta approach uses scan-A-to-B transitions; it isolates new
    voids created by compression. Pre-existing porosity stays in scan A and
    is filtered out automatically.

Outputs to results_<PRE>/crack_analysis/transition_AtoB/:
    damage.csv               per-component descriptors
    damage_summary.txt
    damage_class_pie.png     fraction by class
    damage_planarity_hist.png
    damage_volume_hist.png
    damage_3D.png            3D render colored by class

Run in WSL spam-venv after spam_failure_modes.py:
    python /mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/spam_crack_analysis.py
"""
import os, sys, time, csv
import numpy as np
import tifffile
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyvista as pv
pv.OFF_SCREEN = True

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from scan_meta import VOXEL_UM

DATA = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
OUT  = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>/crack_analysis'
os.makedirs(OUT, exist_ok=True)

TRANSITIONS = [(1, 2), (2, 3)]

# Per-scan phase thresholds (same as ice-bonds / failure_modes)
TH = {
    1: dict(air=7000,    ig=18032.7, ga=47275),
    2: dict(air=2139.4,  ig=6271.5,  ga=99999),
    3: dict(air=2015.4,  ig=6539.5,  ga=99999),
}

# Detection / classification parameters
MIN_DAMAGE_VOX     = 50      # discard tiny damage clusters as noise
PLANARITY_THRESH   = 0.6     # >= -> planar feature (CRACK candidate)
LOCAL_THICK_CRACK  = 2.0     # vox; thinner than this in local thickness -> CRACK
DELAM_GLASS_FRAC   = 0.55    # surface fraction touching glass -> DELAMINATION
ICE_FRAC_THRESH    = 0.55    # surface fraction touching ice  -> ICE-FRACTURE
PERCOLATE_BOUNDARY = 5       # vox; if damage CC has any voxel within this many
                             # vox of the specimen-mask boundary -> "percolates"
FULL_CRACK_LEN_FRAC = 0.5    # CRACK length / specimen z-extent >= this -> FULL CRACK

CLASS_COLOR = {
    'CRACK':        '#d62728',
    'DELAMINATION': '#ff7f0e',
    'ICE-FRACTURE': '#1f77b4',
    'CAVITY':       '#2ca02c',
    'NOISE':        '#bbbbbb',
}

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

def save_with_legend_below(fig, ax, path, legend_ncol=None, anchor_y=-0.16):
    """Place legend BELOW axes; save tight."""
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

def save_3d_with_legend_below(plotter, path, legend_entries, figsize=(13, 11)):
    """3D PyVista render -> PNG with a matplotlib legend in a BORDERED BOX
    centered below the image. legend_entries: list of (label, color) tuples."""
    from matplotlib.patches import Patch
    tmp = path + '.__tmp__.png'
    plotter.screenshot(tmp, window_size=(1600, 1200))
    plotter.close()
    img = plt.imread(tmp)
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(2, 1, height_ratios=[10, 1.4], hspace=0.05)
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
    fig.savefig(path, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig); os.remove(tmp)

def reconstruct_phases(ct, mask, th):
    """uint8 phase array on the specimen mask:
       0 = outside, 1 = air, 2 = ice, 3 = glass."""
    phases = np.zeros(ct.shape, dtype=np.uint8)
    inside = mask > 0
    phases[inside & (ct < th['air'])]                              = 1
    phases[inside & (ct >= th['air']) & (ct < th['ig'])]           = 2
    phases[inside & (ct >= th['ig'])  & (ct < th['ga'])]           = 3
    return phases

def planarity_index(coords):
    """PCA-based planarity: 1.0 = perfect plane, 0 = sphere.
    Returns (planarity, length_along_principal, axis_unit_vector)."""
    if len(coords) < 3:
        return 0.0, 0.0, np.array([1.0, 0.0, 0.0])
    c = coords - coords.mean(axis=0)
    cov = (c.T @ c) / max(len(c) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvals sorted ascending: lambda_1 (smallest) <= lambda_2 <= lambda_3
    # Planarity index = (lambda_2 - lambda_1) / lambda_3.
    # 1 -> perfectly planar, 0 -> isotropic blob.
    if eigvals[2] <= 0:
        return 0.0, 0.0, eigvecs[:, 2]
    planar = (eigvals[1] - eigvals[0]) / eigvals[2]
    length = float(2.0 * np.sqrt(eigvals[2]))   # ~2 sigma along principal axis
    return float(planar), length, eigvecs[:, 2]

def classify(comp_props):
    """Return one of CRACK / DELAMINATION / ICE-FRACTURE / CAVITY."""
    if comp_props['n_vox'] < MIN_DAMAGE_VOX:
        return 'NOISE'
    is_planar = (comp_props['planarity'] >= PLANARITY_THRESH and
                 comp_props['local_thick'] <= LOCAL_THICK_CRACK)
    if is_planar:
        return 'CRACK'
    if comp_props['surf_frac_glass'] >= DELAM_GLASS_FRAC:
        return 'DELAMINATION'
    if comp_props['surf_frac_ice']   >= ICE_FRAC_THRESH:
        return 'ICE-FRACTURE'
    return 'CAVITY'

# ----------------------------------------------------------------------------
print(f'Crack / damage analysis — transitions {TRANSITIONS}')
all_summary = []

for a, b in TRANSITIONS:
    print(f'\n=== transition {a} -> {b} ===')
    tdir = os.path.join(OUT, f'transition_{a}to{b}')
    os.makedirs(tdir, exist_ok=True)

    t0 = time.time()
    ct_a   = tifffile.imread(os.path.join(DATA, f'ct_scan{a:02d}_aligned.tif'))
    mask_a = tifffile.imread(os.path.join(DATA, f'specimen_mask_scan{a:02d}_aligned.tif'))
    ct_b   = tifffile.imread(os.path.join(DATA, f'ct_scan{b:02d}_aligned.tif'))
    mask_b = tifffile.imread(os.path.join(DATA, f'specimen_mask_scan{b:02d}_aligned.tif'))
    print(f'  loaded volumes ({time.time()-t0:.1f}s)  shape={ct_a.shape}')

    phases_a = reconstruct_phases(ct_a, mask_a > 0, TH[a])
    phases_b = reconstruct_phases(ct_b, mask_b > 0, TH[b])
    del ct_a, ct_b

    # Damage = solid_in_A AND air_in_B (and inside specimen in BOTH scans)
    inside_both = (mask_a > 0) & (mask_b > 0)
    solid_a = (phases_a == 2) | (phases_a == 3)        # ice or glass
    air_b   = (phases_b == 1)
    damage = solid_a & air_b & inside_both
    n_dmg = int(damage.sum())
    n_solid_a = int(solid_a.sum())
    print(f'  damage voxels: {n_dmg:,}  ({100*n_dmg/max(n_solid_a,1):.2f}% of solid_A)')

    if n_dmg == 0:
        print('  no damage detected, skipping classification.')
        continue

    # Connected components of damage
    lab, n_cc = ndi.label(damage, structure=np.ones((3, 3, 3), dtype=bool))
    print(f'  {n_cc} damage components')

    # Per-component stats — slow, but bounded by n_cc
    comps = []
    glass_mask_b = phases_b == 3
    ice_mask_b   = phases_b == 2

    # Specimen Z-extent (used by the FULL-CRACK criterion)
    nz_inside = np.any(mask_b > 0, axis=(1, 2))
    z_top = int(np.argmax(nz_inside)) if nz_inside.any() else 0
    z_bot = int(len(nz_inside) - 1 - np.argmax(nz_inside[::-1])) if nz_inside.any() else 0
    spec_z_extent = max(z_bot - z_top + 1, 1)

    # Specimen-edge mask: voxels inside specimen but within PERCOLATE_BOUNDARY
    # of the boundary (cylinder wall or top/bottom).
    edge = inside_both & ~ndi.binary_erosion(inside_both, iterations=PERCOLATE_BOUNDARY)

    boundary_struct = ndi.generate_binary_structure(3, 1)
    sliced = ndi.find_objects(lab)
    print(f'  classifying {n_cc} components...')
    # Pre-compute volumes per CC in one shot — much faster than per-component .sum()
    cc_voxel_counts = np.bincount(lab.ravel())
    n_skipped = 0
    for cid, sl in enumerate(sliced, start=1):
        if sl is None:
            continue
        n_vox = int(cc_voxel_counts[cid]) if cid < len(cc_voxel_counts) else 0
        # FAST-SKIP: components below MIN_DAMAGE_VOX get a tiny noise record
        # (no expensive geometry / distance / surface-shell work).
        if n_vox < MIN_DAMAGE_VOX:
            n_skipped += 1
            continue
        sub_lab = lab[sl]
        cc_mask = sub_lab == cid
        # Voxel coords in global frame for PCA
        zz, yy, xx = np.where(cc_mask)
        coords = np.column_stack([zz + sl[0].start,
                                  yy + sl[1].start,
                                  xx + sl[2].start]).astype(np.float32)
        planar, length_vox, axis = planarity_index(coords)
        edt = ndi.distance_transform_edt(cc_mask)
        local_thick = float(edt.max())
        # Surface contact: dilate by 1 vox, intersect with adjacent phases (in scan B)
        cc_dil = ndi.binary_dilation(cc_mask, structure=boundary_struct)
        shell = cc_dil & ~cc_mask
        if shell.any():
            shell_glob_z = np.where(shell)[0] + sl[0].start
            shell_glob_y = np.where(shell)[1] + sl[1].start
            shell_glob_x = np.where(shell)[2] + sl[2].start
            inds = (shell_glob_z, shell_glob_y, shell_glob_x)
            n_shell = len(shell_glob_z)
            n_glass = int(glass_mask_b[inds].sum())
            n_ice   = int(ice_mask_b[inds].sum())
            sf_glass = n_glass / max(n_shell, 1)
            sf_ice   = n_ice / max(n_shell, 1)
        else:
            sf_glass = sf_ice = 0.0
        # Percolation to specimen boundary
        cc_global = np.zeros_like(damage, dtype=bool)
        cc_global[sl] = cc_mask
        percolates = bool((cc_global & edge).any())
        # Centroid
        centroid = coords.mean(axis=0)
        comp_props = dict(
            id=cid, n_vox=n_vox,
            V_um3=n_vox * VOXEL_UM**3,
            planarity=planar, local_thick=local_thick,
            length_um=length_vox * VOXEL_UM,
            cz=float(centroid[0]), cy=float(centroid[1]), cx=float(centroid[2]),
            ax_z=float(axis[0]), ax_y=float(axis[1]), ax_x=float(axis[2]),
            surf_frac_glass=sf_glass, surf_frac_ice=sf_ice,
            percolates=percolates,
        )
        cls = classify(comp_props)
        is_full_crack = (
            cls == 'CRACK'
            and percolates
            and (length_vox / spec_z_extent) >= FULL_CRACK_LEN_FRAC
        )
        comp_props['class'] = cls
        comp_props['fully_developed'] = is_full_crack
        comps.append(comp_props)

    # Drop noise components
    real = [c for c in comps if c['class'] != 'NOISE']
    print(f'  classified: {len(real)} non-noise components, {len(comps)-len(real)} noise')
    counts = {k: sum(1 for c in real if c['class'] == k)
              for k in ('CRACK', 'DELAMINATION', 'ICE-FRACTURE', 'CAVITY')}
    n_full = sum(1 for c in real if c['fully_developed'])
    vols   = {k: sum(c['V_um3'] for c in real if c['class'] == k) / 1e9    # mm^3
              for k in counts}
    print(f'  classes: {counts}    fully-developed cracks: {n_full}')
    print(f'  total damage volume: {sum(c["V_um3"] for c in real)/1e9:.4f} mm^3')

    # CSV
    csv_path = os.path.join(tdir, 'damage.csv')
    keys = list(real[0].keys()) if real else list(comps[0].keys()) if comps else []
    if keys:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, delimiter=';')
            w.writeheader()
            for c in real:
                w.writerow(c)
        print(f'  wrote {csv_path}')

    # Plots
    if real:
        # Class-count BAR (pie disallowed)
        fig, ax = plt.subplots(figsize=(13, 8))
        labels = [k for k, v in counts.items() if v > 0]
        sizes  = [counts[k] for k in labels]
        colors = [CLASS_COLOR[k] for k in labels]
        ax.bar(labels, sizes, color=colors, edgecolor='black',
               linewidth=1.5, width=0.6)
        ax.set_xlabel('Damage class')
        ax.set_ylabel('Component count')
        ax.set_title(f'Damage classes — transition {a}→{b}')
        ax.grid(True, alpha=0.3, axis='y')
        fig.savefig(os.path.join(tdir, 'damage_class_bar.png'),
                    dpi=220, bbox_inches='tight')
        plt.close(fig)
        # remove old pie if it exists
        old_pie = os.path.join(tdir, 'damage_class_pie.png')
        if os.path.exists(old_pie):
            os.remove(old_pie)

        # Planarity hist
        plan = np.array([c['planarity'] for c in real])
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.hist(plan, bins=30, color='#1f77b4', edgecolor='black', alpha=0.75)
        ax.axvline(PLANARITY_THRESH, color='#d62728', ls='--', lw=3,
                   label=f'CRACK threshold ({PLANARITY_THRESH})')
        ax.set_xlabel('Planarity index  (1 = plane, 0 = sphere)')
        ax.set_ylabel('Damage component count')
        ax.set_title(f'Damage planarity — transition {a}→{b}')
        ax.grid(True, alpha=0.3)
        save_with_legend_below(fig, ax,
                               os.path.join(tdir, 'damage_planarity_hist.png'),
                               legend_ncol=1)

        # Volume hist (log)
        vol_um3 = np.array([c['V_um3'] for c in real])
        fig, ax = plt.subplots(figsize=(13, 8))
        bins = np.logspace(np.log10(max(vol_um3.min(), 1.0)),
                           np.log10(max(vol_um3.max(), 10.0)), 40)
        for k in counts:
            sub = [c['V_um3'] for c in real if c['class'] == k]
            if sub:
                ax.hist(sub, bins=bins, alpha=0.7, color=CLASS_COLOR[k],
                        label=k, edgecolor='black')
        ax.set_xscale('log')
        ax.set_xlabel('Damage component volume [µm³]')
        ax.set_ylabel('Count')
        ax.set_title(f'Damage volume distribution — transition {a}→{b}')
        ax.grid(True, alpha=0.3, which='both')
        save_with_legend_below(fig, ax,
                               os.path.join(tdir, 'damage_volume_hist.png'),
                               legend_ncol=4)

        # 3D render: color by class — legend goes BELOW image in a bordered box
        plotter = pv.Plotter(off_screen=True, window_size=(1600, 1200))
        legend_entries = []
        for cls in counts:
            cls_lab = np.zeros_like(lab, dtype=np.uint8)
            for c in real:
                if c['class'] == cls:
                    cls_lab[lab == c['id']] = 1
            if cls_lab.any():
                grid = pv.wrap(cls_lab.astype(np.uint8))
                surf = grid.contour([0.5])
                plotter.add_mesh(surf, color=CLASS_COLOR[cls], opacity=0.85)
                legend_entries.append((cls, CLASS_COLOR[cls]))
        plotter.set_background('white')
        plotter.camera.up = (0, 0, -1)
        plotter.view_isometric()
        save_3d_with_legend_below(plotter,
                                  os.path.join(tdir, 'damage_3D.png'),
                                  legend_entries, figsize=(14, 12))

    # Per-transition summary line
    all_summary.append(dict(
        transition=f'{a}->{b}',
        total_damage_um3=sum(c['V_um3'] for c in real),
        n_crack=counts.get('CRACK', 0),
        n_delam=counts.get('DELAMINATION', 0),
        n_icefrac=counts.get('ICE-FRACTURE', 0),
        n_cavity=counts.get('CAVITY', 0),
        n_full_crack=n_full,
    ))

    # Per-transition text summary
    with open(os.path.join(tdir, 'damage_summary.txt'), 'w') as f:
        f.write(f'Damage / crack analysis — transition {a}→{b}\n')
        f.write('=' * 56 + '\n')
        f.write(f'damage voxels      : {n_dmg:,}\n')
        f.write(f'damage / solid_A   : {100*n_dmg/max(n_solid_a,1):.3f} %\n')
        f.write(f'damage volume      : {sum(c["V_um3"] for c in real)/1e9:.4f} mm³\n')
        f.write(f'\ncomponents (after dropping noise <{MIN_DAMAGE_VOX} vox):\n')
        for k, v in counts.items():
            f.write(f'  {k:<14} : {v:>4}    total volume {vols[k]:.4f} mm³\n')
        f.write(f'\nfully-developed cracks (planar + percolating + length >= '
                f'{FULL_CRACK_LEN_FRAC*100:.0f}% spec extent): {n_full}\n')
        if n_full > 0:
            f.write('\nfully-developed crack details:\n')
            for c in real:
                if c['fully_developed']:
                    f.write(f"  cc#{c['id']}: V={c['V_um3']/1e9:.4f} mm³, "
                            f"length={c['length_um']/1000:.2f} mm, "
                            f"planarity={c['planarity']:.2f}\n")
    print(f'  wrote {os.path.join(tdir, "damage_summary.txt")}')

# Cross-transition summary
summary_path = os.path.join(OUT, 'crack_summary.txt')
with open(summary_path, 'w') as f:
    f.write('Crack / damage analysis — cross-transition summary\n')
    f.write('=' * 60 + '\n\n')
    f.write(f'{"trans":<8}{"V_total [mm³]":>16}'
            f'{"crack":>8}{"delam":>8}{"ice-fr":>8}{"cavity":>8}{"FULL":>8}\n')
    for s in all_summary:
        f.write(f'{s["transition"]:<8}'
                f'{s["total_damage_um3"]/1e9:>16.4f}'
                f'{s["n_crack"]:>8}{s["n_delam"]:>8}'
                f'{s["n_icefrac"]:>8}{s["n_cavity"]:>8}{s["n_full_crack"]:>8}\n')
    f.write('\nClasses:\n')
    f.write('  CRACK         planar (planarity>=0.6) + thin (local_thick<=2 vox)\n')
    f.write('  DELAMINATION  >=55% of damage surface touches glass beads\n')
    f.write('  ICE-FRACTURE  >=55% of damage surface touches ice\n')
    f.write('  CAVITY        non-planar 3D blob (compaction void / rearrangement)\n')
    f.write('\nFully-developed crack:\n')
    f.write('  CRACK class AND damage extends to specimen boundary\n')
    f.write('  AND principal-axis length >= 50% of specimen Z-extent.\n')
print(f'\nwrote {summary_path}')
print('Done.')
