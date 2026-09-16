"""Every damaged bond in one specimen, one at a time, in its own file.

TWO IMAGES PER BOND

  ..._sections.png   four 2D sections: the BOND SECTION (the plane containing
                     both bead centres, resliced with trilinear interpolation
                     because contact axes are oblique) plus XY, XZ and YZ
                     through the contact.  The three orthogonal ones are there
                     to show what a single viewing plane gets wrong.

  ..._3d.png         the same two beads as surfaces, with the crack between
                     them as a solid body.  This is the section figure with the
                     guesswork removed: you see the shape of the crack instead
                     of one slice through it.

FOLDERS
A bond goes in exactly one folder.  DETACHED is assigned first, because "no
solid path from bead to bead" is a stronger statement than any position: that
bond no longer carries load.  The rest are placed by where the crack sits:

    detached/   no solid path A -> B through bead or binder
    cohesive/   median q >= 0.30, crack down the middle of the binder
    mixed/      0.15 < median q < 0.30
    adhesive/   median q <= 0.15, crack against a bead surface

FILE NAMES SORT BY SEVERITY
    r003_beads0033-0306_crack67_q034_sections.png
r is the rank within the class by crack share of the throat, so the worst bond
in a class is the first file in the folder.  crack is that share as a percent
and q is the median position, both also in index.csv beside the images.
"""
import os
import sys

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage as ndi
import pyvista as pv
from _spec import (B, OUT, FIG, UM, MM3, TAG, ALN, SCANS, FIRST, LAST, MIDDLE, CHAIN, PUB,

                   p_ct, p_lab, p_ph, p_crack, p_sample, p_core, have, openv)


def fs_for(fig, frac=0.030, floor=24):
    """Text size that stays readable after the figure is scaled to a column.

    A fixed point size is the wrong unit here.  These renders are tall -- a
    slender specimen gives a figure ~29 inches high -- so a 26pt legend is
    about 1% of the image height, and once the whole figure is scaled into a
    paper column or a slide it is unreadable.  Sizing text as a FRACTION of the
    figure height keeps it the same visual size whatever the aspect ratio.
    """
    return max(floor, frac * float(fig.get_figheight()) * 72.0)


# --- no text is burned into a figure ---------------------------------------
# Every title, suptitle and row label used to carry the numbers.  They are in
# figure_index.xlsx instead, keyed by file name, so a figure can be dropped
# into a paper and captioned there.  Axis labels stay: they name the axis, they
# are not data.  Legends and scale bars stay.
def _notitle(*a, **k):
    return None


pv.OFF_SCREEN = True

DEST = f'{OUT}/individual_bonds'
PH_BINDER = 2
PAD = 18
FOOT = 0.40
Q_ADH, Q_COH = 0.15, 0.30
MIN_CRACK = 200
_FS = 24
# --- nothing on a figure is allowed below FLOOR point ------------------------
# These are read in a paper column or from the back of a room.  The legend and
# the colorbar carry the units and the numbers, so they are the last things
# that should be shrunk to make a render fit -- which is what the old
# _FS - 6 / _FS - 8 reductions and the 15pt colorbar ticks were doing.
FLOOR = 22


C_A, C_B = '#1565c0', '#2e7d32'
C_CRACK, C_BINDER, C_LENS = '#e00000', '#26c6da', '#ab47bc'
CLASSES = ('detached', 'cohesive', 'mixed', 'adhesive')


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': _FS,
        'axes.titlesize': _FS - 2, 'axes.titleweight': 'bold',
        'axes.labelsize': _FS, 'axes.labelweight': 'bold',
        'axes.linewidth': 2.2, 'axes.edgecolor': 'black',
        'axes.spines.top': True, 'axes.spines.right': True,
        'axes.spines.left': True, 'axes.spines.bottom': True,
        'axes.facecolor': 'white', 'axes.grid': False,
        'figure.facecolor': 'white', 'savefig.dpi': 200,
        'savefig.bbox': 'tight'})


class Vol:
    def __init__(self, scan):
        self.ct = openv(f'{B}/data/ct_scan{scan:02d}{ALN}.tif',
                                  mode='r')
        self.lab = openv(
            f'{B}/data/bead_labels_scan{scan:02d}{ALN}.tif', mode='r')
        self.ph = openv(
            f'{B}/data/phases_scan{scan:02d}{ALN}.tif', mode='r')
        self.crk = openv(
            f'{B}/results_voidcrack/crack_scan{scan:02d}.tif', mode='r')
        self.core = openv(f'{OUT}/core_scan{scan:02d}.tif', mode='r')
        self.shape = self.lab.shape


def crop_for(vol, ia, ib, ca, cb, ra, rb):
    lo, hi = [], []
    for k in range(3):
        r = max(ra, rb) + PAD
        lo.append(int(max(0, min(ca[k], cb[k]) - r)))
        hi.append(int(min(vol.shape[k], max(ca[k], cb[k]) + r)))
    sl = tuple(slice(lo[k], hi[k]) for k in range(3))
    d = dict(
        ct=np.asarray(vol.ct[sl]).astype(np.float32),
        lab=np.asarray(vol.lab[sl]),
        ph=np.asarray(vol.ph[sl]),
        crk=(np.asarray(vol.crk[sl]) > 0) & (np.asarray(vol.core[sl]) > 0),
        lo=np.array(lo, dtype=np.float64))
    d['mA'] = d['lab'] == ia
    d['mB'] = d['lab'] == ib
    beads = d['lab'] > 0
    free = (d['ph'] > 0) & ~beads
    dA = ndi.distance_transform_edt(~d['mA'])
    dB = ndi.distance_transform_edt(~d['mB'])
    _, idx = ndi.distance_transform_edt(~beads, return_indices=True)
    near = d['lab'][idx[0], idx[1], idx[2]]
    ssum = dA + dB
    g = float(ssum[free].min()) if free.any() else 0.0
    rmean = 0.5 * (ra + rb)
    rth = FOOT * rmean
    slack = rth * rth / (rmean + 0.5 * g)
    d['lens'] = free & (ssum <= g + slack) & ((near == ia) | (near == ib))
    d['gap'] = max(g - 1.0, 0.0)
    return d


# --------------------------------------------------------------- sections ---
def _plane(d, kind, ca, cb):
    mid = 0.5 * (np.asarray(ca) + np.asarray(cb)) - d['lo']
    if kind in ('xy', 'xz', 'yz'):
        ax = {'xy': 0, 'xz': 1, 'yz': 2}[kind]
        i = int(round(np.clip(mid[ax], 0, d['ct'].shape[ax] - 1)))
        out = [np.take(d[k], i, axis=ax) for k in
               ('ct', 'mA', 'mB', 'ph', 'crk', 'lens')]
        out[3] = out[3] == PH_BINDER
        return out
    a = np.asarray(ca) - d['lo']
    b = np.asarray(cb) - d['lo']
    e1 = b - a
    n1 = float(np.linalg.norm(e1))
    e1 = e1 / max(n1, 1e-9)
    zhat = np.array([1.0, 0.0, 0.0])
    e2 = zhat - np.dot(zhat, e1) * e1
    if np.linalg.norm(e2) < 1e-6:
        e2 = np.array([0.0, 1.0, 0.0]) - np.dot(
            np.array([0.0, 1.0, 0.0]), e1) * e1
    e2 = e2 / max(float(np.linalg.norm(e2)), 1e-9)
    u = np.arange(-(0.5 * n1 + PAD), 0.5 * n1 + PAD + 1.0)
    v = np.arange(-(0.55 * n1 + PAD), 0.55 * n1 + PAD + 1.0)
    U, V = np.meshgrid(u, v, indexing='xy')
    mid2 = 0.5 * (a + b)
    pts = (mid2[:, None, None] + e1[:, None, None] * U[None]
           + e2[:, None, None] * V[None])
    out = []
    for k, order in (('ct', 1), ('mA', 0), ('mB', 0), ('ph', 0),
                     ('crk', 0), ('lens', 0)):
        out.append(ndi.map_coordinates(d[k].astype(np.float32), pts,
                                       order=order, mode='constant', cval=0))
    for i in (1, 2, 4, 5):
        out[i] = out[i] > 0.5
    out[3] = np.abs(out[3] - PH_BINDER) < 0.5
    return out


def draw(ax, d, kind, ca, cb):
    ct, mA, mB, binder, crack, lens = _plane(d, kind, ca, cb)
    lo, hi = np.percentile(ct[ct > 0], (1, 99)) if (ct > 0).any() else (0, 1)
    ax.imshow(ct, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
    ov = np.zeros(ct.shape + (4,))
    ov[binder] = matplotlib.colors.to_rgba(C_BINDER, 0.22)
    ov[crack] = matplotlib.colors.to_rgba(C_CRACK, 0.60)
    ax.imshow(ov, interpolation='nearest')
    for m, c in ((mA, C_A), (mB, C_B)):
        if m.any():
            ax.contour(m.astype(float), levels=[0.5], colors=[c],
                       linewidths=2.4)
    if lens.any():
        ax.contour(lens.astype(float), levels=[0.5], colors=[C_LENS],
                   linewidths=2.0, linestyles='--')
    ax.set_xticks([])
    ax.set_yticks([])
    n = 500.0 / UM
    x0, y0 = ct.shape[1] * 0.06, ct.shape[0] * 0.93
    ax.plot([x0, x0 + n], [y0, y0], color='white', lw=5, solid_capstyle='butt')
    ax.plot([x0, x0 + n], [y0, y0], color='black', lw=2, solid_capstyle='butt')


LEG2D = [('bead A', C_A), ('bead B', C_B), ('binder', C_BINDER),
         ('crack', C_CRACK), ('measured throat', C_LENS)]


def sections_png(d, ca, cb, path, title):
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.6))
    for ax, k in zip(axes, ('bond', 'xy', 'xz', 'yz')):
        draw(ax, d, k, ca, cb)
        _notitle('BOND SECTION' if k == 'bond' else k.upper(),
                     fontsize=max(_FS - 2, FLOOR))
    _notitle(title, fontsize=_FS - 2, fontweight='bold', y=1.03)
    h = [Patch(facecolor=c, edgecolor='black', linewidth=1.4, label=l)
         for l, c in LEG2D]
    leg = fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, -0.10),
                     ncol=5, fontsize=fs_for(fig), frameon=True, fancybox=False,
                     edgecolor='black', framealpha=1.0, borderpad=0.7)
    leg.get_frame().set_linewidth(2.0)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# --------------------------------------------------------------------- 3D ---
def _surf(mask, smooth=12):
    if not mask.any():
        return None
    g = pv.wrap(np.ascontiguousarray(mask.astype(np.float32)))
    s = g.contour([0.5])
    if s.n_points == 0:
        return None
    return s.smooth(n_iter=smooth, relaxation_factor=0.1)


LEG3D = [('bead A', C_A), ('bead B', C_B), ('binder in the throat', C_BINDER),
         ('crack in the throat', C_CRACK)]


def three_d_png(d, ca, cb, path, title):
    crack = d['lens'] & d['crk']
    binder = d['lens'] & (d['ph'] == PH_BINDER)
    pl = pv.Plotter(off_screen=True, window_size=(1500, 1250))
    pl.set_background('white')
    for m, c, op in ((d['mA'], C_A, 0.28), (d['mB'], C_B, 0.28),
                     (binder, C_BINDER, 0.35), (crack, C_CRACK, 1.0)):
        s = _surf(m)
        if s is not None:
            pl.add_mesh(s, color=c, opacity=op, smooth_shading=True,
                        specular=0.3)
    # look at the bond side-on: perpendicular to the contact axis, and as
    # close to level as that allows
    u = np.asarray(cb) - np.asarray(ca)
    u = u / max(float(np.linalg.norm(u)), 1e-9)
    up = np.array([1.0, 0.0, 0.0])          # z
    view = np.cross(u, up)
    if np.linalg.norm(view) < 1e-6:
        view = np.cross(u, np.array([0.0, 1.0, 0.0]))
    view = view / np.linalg.norm(view)
    ctr = np.array(d['mA'].shape) / 2.0
    pos = ctr + view[::-1] * max(d['mA'].shape) * 2.2
    pl.camera_position = [tuple(pos[::-1]), tuple(ctr[::-1]),
                          tuple(up[::-1])]
    pl.reset_camera()
    pl.camera.zoom(1.35)
    tmp = path + '.__tmp__.png'
    pl.screenshot(tmp)
    pl.close()
    img = plt.imread(tmp)
    fig = plt.figure(figsize=(11, 10.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[10, 1.5], hspace=0.03)
    ax = fig.add_subplot(gs[0])
    ax.imshow(img)
    ax.axis('off')
    _notitle(title, fontsize=max(_FS - 2, FLOOR), fontweight='bold')
    al = fig.add_subplot(gs[1])
    al.axis('off')
    h = [Patch(facecolor=c, edgecolor='black', linewidth=1.5, label=l)
         for l, c in LEG3D]
    leg = al.legend(handles=h, loc='center', ncol=2, fontsize=fs_for(fig),
                    frameon=True, fancybox=False, edgecolor='black',
                    framealpha=1.0, borderpad=0.8)
    leg.get_frame().set_linewidth(2.0)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    os.remove(tmp)


# ------------------------------------------------------------------- main ---
def classify(r):
    if not r.connected:
        return 'detached'
    if r.crack_vox < MIN_CRACK:
        return None
    if r.q_crack_med <= Q_ADH:
        return 'adhesive'
    if r.q_crack_med >= Q_COH:
        return 'cohesive'
    return 'mixed'


def main(scan=LAST):
    style()
    df = pd.read_csv(f'{OUT}/{TAG}_throats_scan{scan:02d}.csv')
    df['cls'] = df.apply(classify, axis=1)
    df = df[df.cls.notna()].copy()
    # A bond built on a label that is not a sphere is not a measurement, so it
    # does not belong in a class folder.  Those go to bead_shape_flagged/ with
    # the closed form drawn over them, and only there.
    qp = f'{OUT}/{TAG}_bead_quality.csv'
    if os.path.exists(qp):
        q = pd.read_csv(qp)
        ok = set(q[(q.scan == scan) & q.shape_ok].bead_id)
        sound = df.bead_a.isin(ok) & df.bead_b.isin(ok)
        print(f'{int((~sound).sum())} of {len(df)} bonds sit on a broken '
              f'label and are left to bead_shape_flagged/', flush=True)
        df = df[sound].copy()
    else:
        print('no bead-quality file: rendering every bond, INCLUDING any '
              'built on a broken label', flush=True)
    c = np.load(f'{OUT}/centroids_scan{scan:02d}.npz')
    pos = {int(i): (p, r) for i, p, r in zip(c['ids'], c['com'], c['req'])}
    vol = Vol(scan)

    for k in CLASSES:
        os.makedirs(f'{DEST}/{k}', exist_ok=True)
    print(f'{len(df)} bonds to render', flush=True)
    print(df.cls.value_counts().to_string(), flush=True)

    done = 0
    for k in CLASSES:
        sub = df[df.cls == k].sort_values('crack_frac', ascending=False)
        rows = []
        if not len(sub):
            with open(f'{DEST}/{k}/NONE_FOUND.txt', 'w') as f:
                f.write(
                    f'No bond in {TAG} scan {scan} falls in the '
                    f'{k.upper()} class.\n\n'
                    'The folder is kept so that an absent class is visible '
                    'rather than\nsilently missing. For ADHESIVE specifically, '
                    'read the limits section\nof the README: a crack lying ON a '
                    'bead surface that has not opened is\nthinner than the '
                    '4.57-voxel grey blur at that surface, so adhesive\n'
                    'failure is UNDER-DETECTED. "None found" is not "none '
                    'occurred".\n')
            print(f'  {k}: none', flush=True)
            continue
        for rank, r in enumerate(sub.itertuples(), 1):
            ia, ib = int(r.bead_a), int(r.bead_b)
            ca, ra = pos[ia]
            cb, rb = pos[ib]
            d = crop_for(vol, ia, ib, ca, cb, ra, rb)
            q = r.q_crack_med if r.q_crack_med == r.q_crack_med else float('nan')
            stem = (f'r{rank:03d}_beads{ia:04d}-{ib:04d}'
                    f'_crack{100*r.crack_frac:02.0f}'
                    f'_q{"nan" if q != q else f"{q:.2f}".replace(".", "")}')
            ttl = (f'beads {ia}-{ib}   {k.upper()}   '
                   f'gap {r.gap_vox:.1f} vox ({r.gap_vox*UM:.0f} um)   '
                   f'crack {100*r.crack_frac:.0f}% of throat'
                   + ('' if q != q else f'   q {q:.2f}'))
            sections_png(d, ca, cb, f'{DEST}/{k}/{stem}_sections.png', ttl)
            three_d_png(d, ca, cb, f'{DEST}/{k}/{stem}_3d.png', ttl)
            rows.append(dict(
                rank=rank, file_stem=stem, bead_a=ia, bead_b=ib,
                failure_location=k,
                gap_vox=round(float(r.gap_vox), 2),
                gap_um=round(float(r.gap_vox) * UM, 1),
                crack_pct_of_throat=round(100 * float(r.crack_frac), 1),
                q_crack_median=None if q != q else round(q, 3),
                q_binder_median=(None if r.q_binder_med != r.q_binder_med
                                 else round(float(r.q_binder_med), 3)),
                dq_paired=(None if r.dq_paired != r.dq_paired
                           else round(float(r.dq_paired), 3)),
                neck_severed=(None if r.neck_severed != r.neck_severed
                              else round(float(r.neck_severed), 3)),
                r_bond_over_R=(None if r.r_bond_over_R != r.r_bond_over_R
                               else round(float(r.r_bond_over_R), 3)),
                connected=bool(r.connected),
                axis_to_load_deg=round(float(r.axis_dip_deg), 1),
                z_mid_vox=round(float(r.z_mid), 0)))
            done += 1
            if done % 10 == 0:
                print(f'   {done} rendered', flush=True)
        pd.DataFrame(rows).to_csv(f'{DEST}/{k}/index.csv', index=False)
        print(f'  {k}: {len(rows)} bonds -> {k}/index.csv', flush=True)
    print(f'done, {done} bonds, {2*done} images -> {DEST}', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else LAST)
