"""Failure geometry for EVERY specimen: figures plus one spreadsheet.

The numbers live in the workbook, not burned into the pictures. Each figure
says one thing with colour -- whether a crack is VERTICAL (running with the
loading axis, an axial split) or HORIZONTAL (running across it, a shear or
end-cap failure). Every angle, volume and quality measure is in the .xlsx.

AXIS CONVENTION
Low z index is the SUPPORT end, per the convention used in the existing plots.
Camera up is -z, so the support end is at the TOP of every render. Heights in
the workbook are quoted as a fraction from the support end: 0.00 = support,
1.00 = punch.

CLASSIFICATION
    dip = arccos(|n_z|)      0 deg = horizontal, 90 deg = along the loading axis
    dip >= 45  ->  vertical   (axial split)      red
    dip <  45  ->  horizontal (shear / end cap)  blue
The split is on GEOMETRY, never on size rank: on Glass_75 scan 2 the LARGEST
component sits at 51 deg, so ordering by size and applying a fixed name list
would have printed the wrong class into the figure.

WHY PER COMPONENT AND NOT ONE PLANE PER SCAN
Fitting one plane to a whole failure crack averages distinct mechanisms
together. On Glass_75 scan 3 that returns dip 74.8 deg with planarity 0.06 --
the two smallest eigenvalues nearly equal, so the cloud is not a sheet at all,
and the best RANSAC plane holds only 25% of voxels. Split by connected
component the same crack resolves into 86 deg and 35 deg pieces at planarity
0.14 and 0.57. PLANARITY IS REPORTED FOR EVERY ROW so a poorly constrained
angle cannot be quoted as if it were a sharp plane.

Usage: python _failure_all.py [specimen ...]
"""
import os
import sys

import numpy as np
import pandas as pd
import pyvista as pv
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import ndimage as ndi

R = '/mnt/e/RPTU-images/CT_images'
STY = f'{R}/Glass/Glass_75_1700_T5_HR/Glass_75_spam'
sys.path.insert(0, STY)
from spam_75_style import use_paper_style, _autocrop                # noqa: E402

pv.OFF_SCREEN = True
FONT = use_paper_style()
OUT = ('/mnt/c/Users/cak7496/AppData/Local/Temp/claude/D--wsl/'
       'a62987fb-a2df-41c2-a6a3-fd119ddec5c2/scratchpad/failure')
os.makedirs(OUT, exist_ok=True)
UM = 24.7660229                    # one value for every specimen, as the
MM = UM / 1000.0                   # pipeline itself uses
SKIN, STEP = 20, 2
MIN_FRAC = 0.02
# THREE CLASSES, NOT TWO.
#
# A single 45 deg split forces every crack to a pole, and almost none of them
# are at a pole: only 5 of 44 components sit above 75 deg and 7 below 15 deg.
# The other 32 are diagonal, which is what the renders plainly show, and a
# two-class figure contradicted them.
#
# An earlier version justified the 45 deg split by a "gap" between 41 and 51
# deg. That was read off COUNTS and does not survive volume weighting: the lone
# component in the 40-49 bin carries 115.2 mm3, 11.5% of all crack volume. The
# dip distribution is close to FLAT -- every 10 deg bin holds 5-14% of the
# volume -- so there is no natural threshold to find, and the honest scheme is
# three even bands rather than a boundary pretending to be physical.
#
# Bands set by the user: 0-20 horizontal, 20-70 diagonal, 70-90 vertical. This
# keeps horizontal and vertical for cracks that really are near a pole and lets
# the broad diagonal band carry the rest, which is where most of them sit.
DIP_HORZ, DIP_VERT = 20.0, 70.0
C_VERT, C_DIAG, C_HORZ = '#e00000', '#ef6c00', '#1565c0'


def classify(dip):
    if dip >= DIP_VERT:
        return 'vertical', C_VERT
    if dip >= DIP_HORZ:
        return 'diagonal', C_DIAG
    return 'horizontal', C_HORZ
EZ = np.array([1.0, 0.0, 0.0])
RNG = np.random.default_rng(7)

SPECS = [
    ('Glass_75_1700_T5_HR',  f'{R}/Glass/Glass_75_1700_T5_HR/Glass_75_spam',   [1, 2, 3]),
    ('Glass_100_1700_T5_HR', f'{R}/Glass/Glass_100_1700_T5_HR/Glass_T5_HR_spam', [1, 2]),
    ('Glass_75_1000_T6',     f'{R}/Glass/Glass_75_1000_T6/Glass_T6_spam',      [1, 2]),
    ('Glass_100_1700_T7',    f'{R}/Glass/Glass_100_1700_T7/Glass_1700_spam',   [1, 2, 3, 4]),
    ('Glass_100_1800_T5',    f'{R}/Glass/Glass_spam',                          [1, 2]),
    ('Alumina_100_1800_T5',  f'{R}/Alumina/Alumina_100_1800_T5/Alumina_100_1800_T5_spam', [1, 2, 3]),
    ('Alumina_175_1800_T5',  f'{R}/Alumina/Alumina_175_1800_T5/Alumina_175_1800_T5_spam', [1, 2]),
    ('Alumina_75_1000_T5',   f'{R}/Alumina/Alumina_75_1000_T5/Alumina_75_1000_T5_spam',   [1, 2]),
    ('Alumina_75_1800_T7',   f'{R}/Alumina/Alumina-75-1800-T7/Alumina_75_1800_T7_spam',   [1, 2, 3]),
]
if len(sys.argv) > 1:
    want = set(sys.argv[1:])
    SPECS = [s for s in SPECS if s[0] in want]


def surf_of(mask, n_iter=15):
    g = pv.ImageData(dimensions=np.array(mask.shape) + 1)
    g.cell_data['v'] = mask.flatten(order='F').astype(np.uint8)
    s = g.threshold(0.5).extract_surface()
    return s.smooth(n_iter=n_iter) if s.n_points else s


def read_summary(spam):
    """Per-scan volumes straight from the pipeline's own summary table."""
    p = f'{spam}/results_voidcrack/voidcrack_summary.txt'
    if not os.path.exists(p):
        return {}
    rows = {}
    for ln in open(p):
        f = ln.split()
        if len(f) >= 9 and f[0].isdigit():
            rows[int(f[0])] = dict(
                sample_mm3=float(f[1]), void_mm3=float(f[2]),
                surface_crack_mm3=float(f[3]), failure_crack_mm3=float(f[4]),
                separation_mm3=float(f[6]),
                crack_minus_baseline_mm3=float(f[7]))
    return rows


comp_rows, scan_rows = [], []
for name, spam, scans in SPECS:
    rd = f'{spam}/results_voidcrack'
    summary = read_summary(spam)
    for s in scans:
        cp, sp = f'{rd}/crack_scan{s:02d}.tif', f'{rd}/sample_scan{s:02d}.tif'
        if not (os.path.exists(cp) and os.path.exists(sp)):
            print(f'{name} scan {s}: missing volumes, skipped', flush=True)
            continue
        sample = tifffile.imread(sp) > 0
        if not sample.any():
            print(f'{name} scan {s}: empty mask, skipped', flush=True)
            del sample
            continue
        zocc = np.where(sample.any(axis=(1, 2)))[0]
        z0, z1 = int(zocc.min()), int(zocc.max())
        occ = np.array(np.nonzero(sample))
        ax_y, ax_x = float(occ[1].mean()), float(occ[2].mean())
        rad_spec = float(np.sqrt(((occ[1] - ax_y) ** 2
                                  + (occ[2] - ax_x) ** 2).max()))
        del occ
        crack = tifffile.imread(cp) > 0
        body = crack & ndi.binary_erosion(sample, iterations=SKIN)
        del crack

        row = dict(specimen=name, scan=s,
                   specimen_height_mm=(z1 - z0 + 1) * MM,
                   specimen_radius_mm=rad_spec * MM)
        row.update(summary.get(s, {}))
        tot = int(body.sum())
        row['failure_crack_voxels'] = tot
        row['n_components_ge_2pct'] = 0

        if tot < 500:
            print(f'{name} scan {s}: failure crack {tot*(MM**3):.2f} mm3 '
                  f'-- nothing to fit', flush=True)
            scan_rows.append(row)
            del sample, body
            continue

        lab, _ = ndi.label(body, structure=np.ones((3, 3, 3)))
        del body
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        order = [int(i) for i in np.argsort(sizes)[::-1]
                 if sizes[i] >= MIN_FRAC * sizes.sum()][:4]
        row['n_components_ge_2pct'] = len(order)
        scan_rows.append(row)
        print(f'{name} scan {s}: {sizes.sum()*(MM**3):7.2f} mm3, '
              f'{len(order)} components', flush=True)

        fits = []
        for i in order:
            P = np.argwhere(lab == i).astype(np.float64)
            Ps = (P if len(P) <= 60000
                  else P[RNG.choice(len(P), 60000, replace=False)])
            c = Ps.mean(axis=0)
            _, sv, Vt = np.linalg.svd(Ps - c, full_matrices=False)
            n = Vt[-1]
            lam = (sv ** 2) / len(Ps)
            dip = float(np.degrees(np.arccos(min(1.0, abs(n[0])))))
            azim = float(np.degrees(np.arctan2(n[2], n[1])) % 180.0)
            flat = float(1.0 - lam[2] / max(lam[1], 1e-12))
            rms = float(np.sqrt((((Ps - c) @ n) ** 2).mean()))
            cls, col = classify(dip)
            # 0 = support end (low z), 1 = punch end
            h0 = (P[:, 0].min() - z0) / max(z1 - z0, 1)
            h1 = (P[:, 0].max() - z0) / max(z1 - z0, 1)
            hc = (c[0] - z0) / max(z1 - z0, 1)
            d_ax = abs(float(np.dot(np.array([c[0], ax_y, ax_x]) - c, n)))
            comp_rows.append(dict(
                specimen=name, scan=s, component=len(fits) + 1,
                classification=cls,
                volume_mm3=round(sizes[i] * (MM ** 3), 3),
                percent_of_failure_crack=round(100 * sizes[i] / sizes.sum(), 1),
                dip_deg_from_horizontal=round(dip, 1),
                azimuth_deg=round(azim, 1),
                planarity=round(flat, 3),
                rms_thickness_mm=round(rms * MM, 3),
                height_from_support_lo=round(float(h0), 2),
                height_from_support_hi=round(float(h1), 2),
                height_from_support_centroid=round(float(hc), 2),
                distance_from_axis_mm=round(d_ax * MM, 2)))
            fits.append(dict(idx=i, c=c, n=n, cls=cls, col=col,
                             reach=float(np.percentile(np.abs((Ps - c) @ Vt[0]),
                                                       98))))

        # ---- figure: colour = class, nothing written on the image ----------
        p = pv.Plotter(off_screen=True, window_size=(1250, 2000))
        p.set_background('white')
        p.add_mesh(surf_of(sample[::STEP, ::STEP, ::STEP], 30),
                   color='#90a4ae', opacity=0.07, smooth_shading=True)
        for f in fits:
            sf = surf_of((lab == f['idx'])[::STEP, ::STEP, ::STEP])
            if sf.n_points:
                p.add_mesh(sf, color=f['col'], opacity=0.95,
                           smooth_shading=True)
            size = 2.0 * max(f['reach'], 0.75 * rad_spec) / STEP
            p.add_mesh(pv.Plane(center=f['c'] / STEP, direction=f['n'],
                                i_size=size, j_size=size,
                                i_resolution=1, j_resolution=1),
                       color=f['col'], opacity=0.20, show_edges=True,
                       edge_color=f['col'], line_width=3)
        shp = sample[::STEP, ::STEP, ::STEP].shape
        ctr3 = np.array(shp) / 2.0
        p.camera_position = [(ctr3[0], ctr3[1] - 2.7 * max(shp), ctr3[2]),
                             (ctr3[0], ctr3[1], ctr3[2]), (-1, 0, 0)]
        p.camera.azimuth = 25
        tmp = f'{OUT}/_t.png'
        p.screenshot(tmp)
        p.close()
        del lab, sample

        img = _autocrop(plt.imread(tmp))
        fig = plt.figure(figsize=(9, 9 * img.shape[0] / img.shape[1] + 1.3))
        gs = fig.add_gridspec(2, 1, height_ratios=[12, 1.2], hspace=0.02)
        ax = fig.add_subplot(gs[0])
        ax.imshow(img)
        ax.set_axis_off()
        axl = fig.add_subplot(gs[1])
        axl.axis('off')
        # ALL THREE CLASSES ALWAYS, even where a figure contains none of one.
        # A legend that varies figure to figure makes the set hard to compare:
        # a reader who sees only two entries cannot tell whether the third class
        # is absent from this specimen or simply was not part of the scheme.
        handles = [Line2D([0], [0], marker='s', color='none',
                          markerfacecolor=c, markeredgecolor='none',
                          markersize=18, label=t)
                   for c, t in ((C_VERT, 'vertical crack'),
                                (C_DIAG, 'diagonal crack'),
                                (C_HORZ, 'horizontal crack'))]
        # BOXED here, unlike the DIC figures. The crack renders are nearly all
        # white around the specimen, so an unboxed key floats with nothing to
        # anchor it; the strain and motion figures sit under a colourbar that
        # already provides that edge.
        leg = axl.legend(handles=handles, loc='center', ncol=len(handles),
                         frameon=True, fontsize=20, handletextpad=0.5,
                         columnspacing=2.0, borderpad=0.7)
        fr = leg.get_frame()
        fr.set_edgecolor('black')
        fr.set_linewidth(1.4)
        fr.set_facecolor('white')
        fr.set_alpha(1.0)
        out = f'{OUT}/{name}_scan{s:02d}_failure_class.png'
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        os.remove(tmp)
        print(f'   -> {os.path.basename(out)}', flush=True)

# ---- workbook --------------------------------------------------------------
xl = f'{OUT}/failure_and_crack_summary.xlsx'
dfc = pd.DataFrame(comp_rows)
dfs = pd.DataFrame(scan_rows)
notes = pd.DataFrame({'note': [
    'Low z index is the SUPPORT end. Heights are a fraction from the support: '
    '0.00 = support, 1.00 = punch.',
    'dip = angle of the fitted plane from HORIZONTAL. 0 deg lies across the '
    'specimen, 90 deg runs along the loading axis.',
    'classification: dip >= 70 deg vertical, 20-70 deg DIAGONAL, below 20 deg '
    'horizontal. Assigned from geometry, never from size rank.',
    'Most cracks are DIAGONAL. Only 5 of 44 components sit above 75 deg and 7 '
    'below 15 deg. An earlier two-class version split at 45 deg and forced '
    'every crack to one pole or the other, which contradicted the renders.',
    'The bands are a convention, not a feature of the data: the dip '
    'distribution is close to flat, with every 10 deg bin carrying 5-14% of '
    'the crack volume, so there is no natural threshold to find. Read the dip '
    'column itself for anything quantitative.',
    'planarity = 1 - lam3/lam2 of the component point cloud. Near 1 is a '
    'genuine sheet; near 0 is a diffuse damage zone whose fitted angle is only '
    'indicative and should not be quoted as a sharp plane.',
    'rms_thickness_mm is the RMS distance of crack voxels from the fitted '
    'plane -- read it together with planarity.',
    'Only components holding at least 2% of a scan failure crack are fitted.',
    'A single plane fitted to a whole failure crack averages separate '
    'mechanisms: on Glass_75_1700_T5_HR scan 3 that gives 74.8 deg at '
    'planarity 0.06, while the components resolve to 86 and 35 deg. '
    'Per-component fits are the ones to use.',
    'failure crack = crack deeper than a 20-voxel skin (the pipeline body '
    'crack class). Surface crack is excluded: it wraps the specimen as a '
    'shell and has no meaningful plane.',
    'Voxel size 24.7660229 um, the single value the pipeline uses for every '
    'specimen.',
    'Scan volumes are read from each voidcrack_summary.txt, not recomputed.',
]})
with pd.ExcelWriter(xl, engine='openpyxl') as w:
    dfc.to_excel(w, sheet_name='crack_components', index=False)
    dfs.to_excel(w, sheet_name='per_scan_volumes', index=False)
    notes.to_excel(w, sheet_name='notes', index=False)
    for sh, df in (('crack_components', dfc), ('per_scan_volumes', dfs),
                   ('notes', notes)):
        ws = w.sheets[sh]
        for j, col in enumerate(df.columns, 1):
            width = max(len(str(col)),
                        *(len(str(v)) for v in df[col].head(200))) + 2
            ws.column_dimensions[ws.cell(1, j).column_letter].width = \
                min(width, 95)
print(f'\n-> {xl}')
print(f'   {len(dfc)} components, {len(dfs)} scans')
