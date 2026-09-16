"""Bead MOTION in 3D for every specimen that has DIC, not just one.

_render_move.py was written for Glass_75_1700_T5_HR alone: the specimen name,
the volume shape and the input file are all hard-coded, and it reads
cumulative_75.npz, which only exists because that specimen has TWO transitions
and a cumulative 1->3 could be built from them.  Everything else has 1->2 only,
so it got no motion figures at all.

This renders motion from the DIC file directly, so any specimen with any
transition gets its three figures.  Glass_75_1700_T5_HR is SKIPPED: its nine
existing figures cover three transitions on one shared scale, which is more
than this can produce, and overwriting them would be a downgrade.

    ghost sphere   where the bead started      (pale grey, transparent)
    solid sphere   where it ended, coloured    (relative displacement, rigid
                                                body removed)
    tube           the path between

THREE COLOUR MODES, as before
    strain   equivalent strain over each bead's k nearest neighbours (YlOrRd)
    disp     relative displacement magnitude (turbo)
    radial   outward component, diverging about zero -- the one that adds
             information, since it separates beads pushed OUT of the packing
             from beads settling inward, which tube length alone cannot.

No exaggeration factor: the relative motion is already ~1.6 bead radii and
visible at true scale.  No titles; captions live in figure_index.xlsx.
"""
import os
import sys

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import _plotrules as pr
import _legend_guard as lg
import matplotlib.pyplot as plt
import tifffile
from matplotlib import cm, colors
from scipy.spatial import cKDTree
from _strain_measure import eq_strain as _eq_strain

R = '/mnt/e/RPTU-images/CT_images'
sys.path.insert(0, f'{R}/Glass/Glass_75_1700_T5_HR/Glass_75_spam')
from spam_75_style import use_paper_style, _autocrop                # noqa: E402
from _spec import REG
from _bead_filter import keep_beads

def bead_radius_vox(X, fallback=32.0):
    """Drawing radius for THIS specimen, from its own correlation nodes.

    R_BEAD was a literal 32.  Eight of the nine specimens sit at 34.6-35.8 vox
    so 32 reads as a deliberate slight undersize and looks right -- but
    Alumina_75_1000_T5 has 19-vox beads, and drawing them at 32 fused the whole
    render into one opaque blob in which no bead or motion tube is resolvable.
    Touching beads sit ~2R apart, so R is half the median nearest-neighbour
    spacing.  Same rule _cumulative_all.py already uses.
    """
    from scipy.spatial import cKDTree as _KD
    if len(X) < 2:
        return fallback
    d, _ = _KD(X).query(X, k=2)
    r = 0.5 * float(np.median(d[:, 1]))
    return r if r > 1.0 else fallback



def fs_for(fig, frac=0.030, floor=24):
    """Text size that stays readable after the figure is scaled to a column.

    A fixed point size is the wrong unit here.  These renders are tall -- a
    slender specimen gives a figure ~29 inches high -- so a 26pt colorbar label
    is about 1% of the image height, and once the whole figure is scaled into a
    paper column or a slide it is unreadable.  Sizing the text as a FRACTION of
    the figure height keeps it the same visual size whatever the aspect ratio.
    """
    return max(floor, frac * float(fig.get_figheight()) * 72.0)

# --- nothing on a figure is allowed below FLOOR point ------------------------
# These are read in a paper column or from the back of a room.  The legend and
# the colorbar carry the units and the numbers, so they are the last things
# that should be shrunk to make a render fit -- which is what the old
# _FS - 6 / _FS - 8 reductions and the 15pt colorbar ticks were doing.
FLOOR = 22

pv.OFF_SCREEN = True
FONT = use_paper_style()
SC = ('/mnt/c/Users/cak7496/AppData/Local/Temp/claude/D--wsl/'
      'a62987fb-a2df-41c2-a6a3-fd119ddec5c2/scratchpad')
OUT = f'{SC}/dic'
os.makedirs(OUT, exist_ok=True)
UM = 24.7660229
R_BEAD, K, TUBE_R = 32.0, 12, 3.0   # R_BEAD is a FALLBACK only; see bead_radius_vox
# --- every specimen, every load step -----------------------------------------
# This list used to be five hard-coded rows pointing at three different results
# trees, each naming only the FIRST transition.  That silently made the motion
# figures a partial set: three Glass specimens had no row at all, and no
# specimen had its 2->3 or 3->4 step rendered even where the correlation
# existed.  Every DIC run now lands in one tree under one naming rule, so the
# jobs are DISCOVERED instead of listed -- a new correlation shows up here by
# existing, and nothing has to be remembered.
DDIC = '/home/amirali_wsl/spam-results-bond'
JOBS = []
for tag, e in REG.items():
    for a_, b_ in zip(e['scans'][:-1], e['scans'][1:]):
        p = f'{DDIC}/{tag}/{tag}_ddic_{a_}{b_}-ddic.tsv'
        if os.path.exists(p):
            JOBS.append((tag, p, f'{a_}to{b_}',
                         e['spam'],
                         f"bead_labels_scan{a_:02d}{e['aln']}.tif"))
SKIP = set()

# Optional specimen filter: any argv token that is a specimen tag restricts the
# run to it.  Job discovery happens at import, so a correlation that finishes
# while a render is already running is simply not seen -- that happened to
# Glass_100_1800_T5.  Re-rendering all fourteen transitions to pick up one is
# wasteful, so the missing specimen can be named instead.
_only = [a for a in sys.argv[1:] if a in REG]
if _only:
    JOBS = [j for j in JOBS if j[0] in _only]
    print(f'restricted to {_only}: {len(JOBS)} job(s)', flush=True)

# Labels and formats come from _plotrules.FIELDS -- see the note there on
# why these two scripts must not keep their own copies.
SPEC = pr.FIELDS
MODES = [m for m in sys.argv[1:] if m in SPEC] or list(SPEC)


def read_dic(path, tag=None, scan=None):
    if path.endswith('.csv'):
        raw = open(path).read().strip().split('\n')
        h = raw[0].split(';')
        d = np.array([[float(x) for x in r.split(';')] for r in raw[1:]])
        c = {n: i for i, n in enumerate(h)}
        ok = d[:, c['valid']] > 0
        return (d[ok][:, [c['zpos_vox'], c['ypos_vox'], c['xpos_vox']]],
                d[ok][:, [c['zdisp_vox'], c['ydisp_vox'], c['xdisp_vox']]],
                f'{int(ok.sum())} of {len(d)} valid')
    raw = open(path).read().strip().split('\n')
    h = raw[0].split()
    d = np.array([[float(x) for x in r.split()] for r in raw[1:]])
    c = {n: i for i, n in enumerate(h)}
    ok = d[:, c['returnStatus']] >= 1
    if tag is not None:
        lab = d[:, c.get('Label', c.get('NodeNumber', 0))].astype(int)[ok]
        keep = keep_beads(lab, f'{SC}/bond/{tag}', tag, scan, f'{tag} {scan}')
        ok = np.zeros(len(d), bool)
        ok[np.where(d[:, c['returnStatus']] >= 1)[0][keep]] = True
    n2 = int((d[ok, c['returnStatus']] >= 2).sum())
    return (d[ok][:, [c['Zpos'], c['Ypos'], c['Xpos']]],
            d[ok][:, [c['Zdisp'], c['Ydisp'], c['Xdisp']]],
            f'{int(ok.sum())} usable, {n2} converged')


def rigid_out(X, U):
    """relative displacement: least-squares translation + rotation removed"""
    Xc = X - X.mean(axis=0)
    Rr = U - U.mean(axis=0)
    A_ = np.zeros((3 * len(X), 3))
    for i, r in enumerate(Xc):
        A_[3*i:3*i+3] = [[0, r[2], -r[1]], [-r[2], 0, r[0]], [r[1], -r[0], 0]]
    om, *_ = np.linalg.lstsq(A_, Rr.reshape(-1), rcond=None)
    return Rr - np.cross(np.tile(om, (len(X), 1)), Xc)


def eq_strain(X, U, k=K):
    # Green-Lagrange, see _strain_measure.py -- the old 0.5*(G+G^T) here
    # reported spurious strain wherever a bead merely rotated.
    return _eq_strain(X, U, k)


def radial_component(X, Ur):
    c = X.mean(axis=0)
    v = X - c
    v[:, 0] = 0.0                      # radial in the cross-section only
    n = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / np.maximum(n, 1e-9)
    return np.sum(Ur * v, axis=1)


def glyph(pts, scal=None, rad=R_BEAD):
    pc = pv.PolyData(pts)
    if scal is not None:
        pc['s'] = scal
    return pc.glyph(geom=pv.Sphere(radius=rad, theta_resolution=16,
                                   phi_resolution=16), scale=False,
                    orient=False)


prepped = []
for name, path, tag, spam, labf in JOBS:
    if name in SKIP:
        continue
    if not os.path.exists(path):
        print(f'{name} {tag}: MISSING {os.path.basename(path)}', flush=True)
        continue
    X, U, note = read_dic(path, name, int(tag.split('to')[0]))
    if len(X) < 30:
        print(f'{name} {tag}: only {len(X)} beads, skipped', flush=True)
        continue
    Ur = rigid_out(X, U)
    fields = dict(strain=eq_strain(X, Ur),
                  disp=np.linalg.norm(Ur, axis=1) * UM,
                  radial=radial_component(X, Ur) * UM)
    try:
        with tifffile.TiffFile(f'{spam}/data/{labf}') as t:
            shape = tuple(t.series[0].shape)
    except Exception:
        shape = (int(X[:, 0].max() * 1.1), int(X[:, 1].max() * 2),
                 int(X[:, 2].max() * 2))
    prepped.append(dict(name=name, tag=tag, X=X, X_end=X + Ur,
                        f=fields, shape=shape, note=note))
    print(f'{name} {tag}: {note}   strain {fields["strain"].min():.3f}..'
          f'{fields["strain"].max():.3f}', flush=True)

# --- cumulative motion, referenced to scan 1 ---------------------------------
# The 1->3 motion figures existed for Glass_75_1700_T5_HR only, produced once by
# the old single-specimen script and never rebuilt -- they were still carrying
# the pre-correction strain eight hours after every other figure had been
# redone.  A folder-level coverage check cannot see that: the motion folder was
# not empty.  Building them here from the corrected cumulative field means they
# are rebuilt whenever anything upstream changes, and every multi-step specimen
# gets them instead of just the one.
for _tag, _e in REG.items():
    if _only and _tag not in _only:
        continue
    _npz = f'{SC}/cumulative_{_tag}.npz'
    if len(_e['scans']) < 3 or not os.path.exists(_npz):
        continue
    _d = np.load(_npz)
    _k = _d['keep_common'] if 'keep_common' in _d.files else None
    for _to in sorted(int(x.split('to')[1]) for x in _d.files
                      if x.startswith('U_1to')):
        if _to < _e['scans'][2]:
            continue                      # 1->2 is already a per-step figure
        _X = _d['X'][_k] if _k is not None else _d['X']
        _U = _d[f'U_1to{_to}'][_k] if _k is not None else _d[f'U_1to{_to}']
        if len(_X) < 30:
            continue
        _Ur = rigid_out(_X, _U)
        _f = dict(strain=eq_strain(_X, _Ur),
                  disp=np.linalg.norm(_Ur, axis=1) * UM,
                  radial=radial_component(_X, _Ur) * UM)
        _shape = (int(_X[:, 0].max() * 1.1), int(_X[:, 1].max() * 2),
                  int(_X[:, 2].max() * 2))
        try:
            with tifffile.TiffFile(
                    f"{_e['spam']}/data/bead_labels_scan"
                    f"{_e['scans'][0]:02d}{_e['aln']}.tif") as _t:
                _shape = tuple(_t.series[0].shape)
        except Exception:
            pass
        prepped.append(dict(name=_tag, tag=f'1to{_to}', X=_X, X_end=_X + _Ur,
                            f=_f, shape=_shape,
                            note=f'{len(_X)} beads, cumulative from scan 1'))
        print(f'{_tag} 1to{_to}: cumulative, {len(_X)} beads', flush=True)

for P in prepped:
    for mode in MODES:
        sp = SPEC[mode]
        s = P['f'][mode]
        if sp.get('diverging'):
            m = float(np.percentile(np.abs(s), 97))
            lo, hi = -m, m
        else:
            lo, hi = 0.0, float(np.percentile(s, 97))
        if hi <= lo:
            continue
        X, Xe, shape = P['X'], P['X_end'], P['shape']
        p = pv.Plotter(off_screen=True, window_size=(1800, 2400))
        p.set_background('white')
        _rad = bead_radius_vox(X, R_BEAD)
        # ghost = where the beads were at the start.  0.11 made the initial
        # scan all but invisible, so a loaded column read as if it were the
        # start state; GHOST_OPACITY raises it so the faded start is seen.
        p.add_mesh(glyph(X, rad=_rad), color='#90a4ae',
                   opacity=float(os.environ.get('GHOST_OPACITY', '0.11')),
                   smooth_shading=True)
        seg = np.empty((2 * len(X), 3))
        seg[0::2], seg[1::2] = X, Xe
        lines = np.hstack([np.full((len(X), 1), 2),
                           np.arange(0, 2 * len(X), 2)[:, None],
                           np.arange(1, 2 * len(X), 2)[:, None]]).ravel()
        pathm = pv.PolyData(seg, lines=lines)
        pathm['s'] = np.repeat(s, 2)
        p.add_mesh(pathm.tube(radius=TUBE_R), scalars='s', cmap=sp['cmap'],
                   clim=[lo, hi], show_scalar_bar=False)
        p.add_mesh(glyph(Xe, s, rad=_rad), scalars='s', cmap=sp['cmap'], clim=[lo, hi],
                   opacity=0.95, smooth_shading=True, show_scalar_bar=False)
        c = np.array(shape) / 2.0
        p.camera_position = [(c[0], c[1] - 2.7 * max(shape), c[2]),
                             (c[0], c[1], c[2]), (-1, 0, 0)]
        p.camera.azimuth = 25
        tmp = f'{OUT}/_m_{P["name"]}.png'
        p.screenshot(tmp)
        p.close()

        img = _autocrop(plt.imread(tmp))
        fig = plt.figure(figsize=(11, 11 * img.shape[0] / img.shape[1] + 3.2))
        gs = fig.add_gridspec(2, 1, height_ratios=[12, 3.2], hspace=0.02)
        ax = fig.add_subplot(gs[0])
        ax.imshow(img)
        ax.set_axis_off()
        axl = fig.add_subplot(gs[1])
        axl.set_axis_off()
        cax = axl.inset_axes([0.05, 0.52, 0.90, 0.26])
        cb = fig.colorbar(cm.ScalarMappable(colors.Normalize(lo, hi),
                                            cmap=sp['cmap']),
                          cax=cax, orientation='horizontal')
        pr.style_colorbar(fig, cb, sp['label'],
                          np.linspace(lo, hi, pr.N_TICKS),
                          sp['fmt'], bar_frac=0.90)
        lg.verify_or_fix(fig, ax, cb, gs, f'{P["name"]}_{P["tag"]}_{mode}')
        out = f'{OUT}/{P["name"]}_{P["tag"]}_motion_{mode}.png'
        fig.savefig(out, dpi=220, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        os.remove(tmp)
        print(f'    -> {os.path.basename(out)}', flush=True)

print(f'{len(prepped)} field set(s) rendered for '
      f'{len(set(P["name"] for P in prepped))} specimen(s).', flush=True)
