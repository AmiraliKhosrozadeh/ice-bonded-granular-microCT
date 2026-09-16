"""3D grain-motion and displacement figures for sand, from real per-grain
correspondence.

These are the same kind of figure as the bead motion renders: every tracked
grain drawn as an arrow from where it started along the way it went.  They are
possible because the DVC field predicts where a grain lands, which turns the
blind correlation the earlier probe attempted into a local match.  The probe's
verdict was right about blind search and wrong as a general statement.

Only grains that pass the strict acceptance are drawn -- unambiguous nearest
match with a 1.6x margin over the runner-up, and a conserved volume, since a
grain is rigid.

ORIENTATION, measured not assumed.  The array's z index runs from the LOADED
end to the SUPPORT: on S2 1->2 the axial displacement is flat at -22.58 bin2
voxels through the whole upper half of the index range and decays to zero at
low index, so the high-index end is the block that does not move and the
low-index end is the one the punch drives.  Here z is flipped so the loaded end
is at the TOP, which is how the specimen is drawn everywhere else.

REFERENCE FRAME.  Each scan is cropped to its own column, and the stage itself
can shift between scans -- on S1 both column ends move by exactly -64 voxels
with no length change, which is the stage, not the sand.  So a raw b - a
carries an arbitrary rigid offset.  Everything here is referenced to the
SUPPORT end, the one part of the specimen that is held fixed by the rig: the
median displacement of the grains in the support-side quarter is subtracted, so
zero means "moved with the support".

TWO FIGURES per pair, because they answer different questions.

  disp3d    support-referenced displacement.  Shows how much of the column
            travels with the punch and where that motion is given up, i.e.
            where the compaction band sits.

  motion    the same vectors after the per-height median is removed, so each
            grain is measured against its own layer.  What is left is grain
            rearrangement -- the part of the motion that is not the column
            simply shortening.
"""
import json
import os
import sys

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors

SC = ('/mnt/c/Users/cak7496/AppData/Local/Temp/claude/D--wsl/'
      'a62987fb-a2df-41c2-a6a3-fd119ddec5c2/scratchpad')
sys.path.insert(0, SC)
from _sand_rigid import remove_rig
from _plotrules import (record, vtk_times, write_index, style_colorbar,
                        N_TICKS, autocrop)
from _legend_guard import verify_or_fix
OUT = f'{SC}/sand_out'
FIG = f'{SC}/preview'
os.makedirs(FIG, exist_ok=True)
pv.OFF_SCREEN = True
MM = 24.7660229 / 1000.0
NICE = {'S1_12': 'S1  scan 1 to 2', 'S2_12': 'S2  scan 1 to 2',
        'S2_23': 'S2  scan 2 to 3', 'S3_12': 'S3  scan 1 to 2'}
GEO_A = {'S1_12': 'S1-1', 'S2_12': 'S2-1', 'S2_23': 'S2-2', 'S3_12': 'S3-1'}
MAX_GLYPH = 8000           # arrows actually drawn: density is what reads
MAX_CLOUD = 9000           # pale spheres for the grains that did not move
NOISE_P97 = 75.0           # um; tracking noise floor, p97 of the unloaded pair
MIN_MAG = 2 * NOISE_P97    # um; below this an arrow is drawing the error
SUPPORT_FRAC = 0.25        # bottom quarter of the column defines "not moving"
NL = chr(10)               # two-line titles; a literal escape gets mangled here


def _load(pair):
    """Grain positions in mm with +z up, and the support-referenced motion."""
    # The reciprocal set, not the original grain_track.npz.  That one was
    # produced by the DVC-seeded predictor, which lost the crop offset between
    # the two scans and matched every grain to a real grain about eight
    # spacings away in z.  The field it gave was still roughly right -- a
    # constant offset is what remove_rig removes -- but the grain identities
    # were wrong, and there is no reason to keep drawing them.
    # A pair whose specimen sits at a different lateral position in the two
    # reconstructions (S2 2->3 moved by about 70 x 90 voxels, measured by
    # cross-correlating the support-end grain masks) needs the seed to carry
    # that shift and the radial flow; _sand_track_flow.py does, and its
    # reciprocal set takes precedence where it exists.
    f = f'{OUT}/dvc_{pair}/grain_track_flow.npz'
    if not os.path.exists(f):
        f = f'{OUT}/dvc_{pair}/grain_track_recip.npz'
    if not os.path.exists(f):
        f = f'{OUT}/dvc_{pair}/grain_track.npz'
    if not os.path.exists(f):
        return None
    d = np.load(f)
    a, b = d['com_a'].astype(float), d['com_b'].astype(float)

    # +z UP: array z runs loaded-end -> support, so flip it for display
    zmax = a[:, 0].max()
    a[:, 0] = zmax - a[:, 0]
    b[:, 0] = zmax - b[:, 0]
    u = (b - a) * MM                              # mm, display orientation
    a = a * MM

    # Support end is the BOTTOM after the flip; hold it fixed.  A constant
    # offset is not enough -- the stage can rotate as well as translate between
    # scans -- so a full rigid transform is fitted there and removed.
    zcut = a[:, 0].min() + SUPPORT_FRAC * np.ptp(a[:, 0])
    u, info = remove_rig(a, u, a[:, 0] < zcut)
    if info.get('fitted'):
        print(f'   rig motion removed: tilt {info["tilt_deg"]:.2f} deg, '
              f'shift {np.linalg.norm(info["shift"]):.3f} mm '
              f'(fitted on {info["n_ref"]} support-side grains)', flush=True)

    # PHYSICAL BOUND, from a measurement that used no correlation at all.
    # Stage 1 measured both columns from the texture profile, so the shortening
    # dL of this load step is known independently.  Referenced to the support,
    # no grain can travel further along the axis than the punch did.  Anything
    # that claims to is a wrong match that happened to satisfy the residual,
    # margin and volume tests -- and on S2 1->2 that is 2.6% of the grains,
    # reaching 2.8 mm against a column that shortened by 1.19 mm.  Left in,
    # they set the colour scale and everything real became dark blue.
    # The tolerance is three times the tracking noise floor measured on the
    # unloaded pair (75 um at p97), not a number chosen to look right.
    dL = _shortening(pair)
    if dL is not None:
        cap = dL + 3 * NOISE_P97 / 1000.0
        ok = np.abs(u[:, 0]) <= cap
        n = int((~ok).sum())
        if n:
            print(f'   {n} of {len(ok)} grains ({100*n/len(ok):.1f}%) dropped: '
                  f'axial motion beyond the {dL:.2f} mm the column actually '
                  f'shortened', flush=True)
        a, u = a[ok], u[ok]
    return a, u


def _shortening(pair):
    """Column shortening for this load step, in mm, from stage-1 geometry."""
    try:
        from _sand_grain_track import PAIRS
        ta, _, tb, _ = PAIRS[pair]
        ga = json.load(open(f'{OUT}/{ta}_geom.json'))
        gb = json.load(open(f'{OUT}/{tb}_geom.json'))
        return abs((ga['z1'] - ga['z0']) - (gb['z1'] - gb['z0'])) * MM
    except Exception:
        return None


def _layer_relative(a, u, nband=24):
    """Motion measured against each grain's own height band."""
    z = a[:, 0]
    edges = np.linspace(z.min(), z.max() + 1e-6, nband + 1)
    idx = np.clip(np.digitize(z, edges) - 1, 0, nband - 1)
    out = u.copy()
    for k in range(nband):
        m = idx == k
        if m.sum() > 20:
            out[m] -= np.median(u[m], axis=0)
    return out


def _render(pts, vec, cb_title, path, r_spec=None, clim=None):
    """One arrow render: faint grain cloud, arrows scaled and coloured by |v|.

    Nothing is written on the image but the colour bar and the height scale.
    What the figure shows goes in the caption index, per _plotrules -- a title
    burned into a PNG cannot be edited by a journal and duplicates the caption,
    and an in-image note like "support (fixed)" tells the reader nothing they
    cannot get from the caption.
    """
    mag_all = np.linalg.norm(vec, axis=1) * 1000.0        # um
    rng = np.random.default_rng(0)

    # Only grains that actually moved get an arrow.  The unloaded pair S1 1->2,
    # which shortened by zero voxels, gives the noise floor of this tracking
    # directly: 56 um median and 75 um at the 97th percentile once the rig's
    # own tilt is removed.  An arrow below MIN_MAG is therefore drawing the
    # measurement error, not the sand, and thousands of them filled the quiet
    # part of the column with dark specks that hid the part that moved.  Those
    # grains are still shown -- as the pale cloud -- so the specimen keeps its
    # shape; they simply carry no direction.
    moving = mag_all >= MIN_MAG
    quiet = pts[~moving]
    if len(quiet) > MAX_CLOUD:
        quiet = quiet[rng.choice(len(quiet), MAX_CLOUD, replace=False)]
    pts, vec, mag = pts[moving], vec[moving], mag_all[moving]
    if len(pts) > MAX_GLYPH:
        sel = rng.choice(len(pts), MAX_GLYPH, replace=False)
        pts, vec, mag = pts[sel], vec[sel], mag[sel]
    if not len(pts):                                      # nothing above noise
        pts, vec, mag = pts[:0], vec[:0], mag_all[:0]

    xyz = pts[:, [2, 1, 0]] if len(pts) else np.zeros((0, 3))
    qxyz = quiet[:, [2, 1, 0]] if len(quiet) else np.zeros((0, 3))
    v = vec[:, [2, 1, 0]] if len(vec) else np.zeros((0, 3))
    # the whole specimen sets the scale and the camera, not just the arrows
    allp = np.vstack([p for p in (qxyz, xyz) if len(p)])
    span = float(np.ptp(allp[:, 2]))
    hi = float(clim if clim else (np.percentile(mag, 97) if len(mag)
                                  else np.percentile(mag_all, 99.5))) or 1.0
    # Clip the arrow LENGTH at the same p97 the colour saturates at.  Without
    # this the few grains right under the punch draw arrows several times
    # longer than everything else and merge into a solid block, which hides
    # the gradient that is the point of the figure.  Colour still separates
    # them: everything at the clip reads as the top of the scale.
    keep = np.minimum(1.0, hi / np.maximum(mag, 1e-9))
    v = v * keep[:, None]
    # ARROW LENGTH, relative to the specimen.  The longest arrow is 5.5% of the
    # column height, about 1.4 mm on a 10 mm wide specimen.  Two earlier values
    # were wrong in the same direction: at 13.5% the longest arrow was a third
    # of the specimen diameter and the punch zone rendered as one solid red
    # mass with no readable direction.  What makes the field carry is the
    # DENSITY and the colour, not the size of one glyph.
    factor = 0.055 * span / (hi / 1000.0)

    pl = pv.Plotter(off_screen=True, window_size=(1200, 1680))
    pl.set_background('white')
    if len(qxyz):
        cloud = pv.PolyData(qxyz)
        pl.add_mesh(cloud.glyph(geom=pv.Sphere(radius=0.06, theta_resolution=8,
                                               phi_resolution=8),
                                scale=False, orient=False),
                    color='#90a4ae', opacity=0.13, smooth_shading=True)

    # Fat arrows, fewer of them.  A thin arrow at this scale reads as a hair on
    # the page and the colour it carries is what the figure is for, so the
    # glyph is widened until the colour is legible at a glance.
    arr = pv.PolyData(xyz)
    arr['|u|'] = mag
    arr['v'] = v
    n_arrow = len(xyz)
    pl.add_mesh(arr.glyph(orient='v', scale='v', factor=factor,
                          geom=pv.Arrow(tip_length=0.34, tip_radius=0.17,
                                        shaft_radius=0.058)),
                scalars='|u|', cmap='turbo', clim=[0, hi],
                smooth_shading=True, specular=0.35, specular_power=18,
                show_scalar_bar=False)
    if r_spec:
        cyl = pv.Cylinder(center=(allp[:, 0].mean(), allp[:, 1].mean(),
                                  allp[:, 2].mean()),
                          direction=(0, 0, 1), radius=r_spec, height=span,
                          resolution=64, capping=False)
        pl.add_mesh(cyl, color='#607d8b', opacity=0.06, style='surface')

    # No axis box.  The bead renders for glass and alumina carry none either,
    # and the scale belongs in the caption, so the three materials can be laid
    # out side by side without one of them having furniture the others lack.

    c = allp.mean(axis=0)
    pl.camera_position = [(c[0], c[1] - 2.7 * span, c[2]),
                          (c[0], c[1], c[2]), (0, 0, 1)]
    pl.camera.azimuth = 22
    pl.camera.elevation = 6
    tmp = f'{FIG}/_scene.png'
    pl.screenshot(tmp)
    pl.close()

    # The colour bar is drawn by matplotlib, not by VTK, so it goes through the
    # same style_colorbar sizing and the same legend guard as every other
    # figure in the project.  VTK's own bar has no way to be checked for the
    # two things that must never happen -- overlapping the render, or falling
    # outside the frame -- and its first attempt did both.
    img = autocrop(plt.imread(tmp))
    fig = plt.figure(figsize=(11, 11 * img.shape[0] / img.shape[1] + 3.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[12, 3.2], hspace=0.02)
    ax = fig.add_subplot(gs[0]); ax.imshow(img); ax.set_axis_off()
    axl = fig.add_subplot(gs[1]); axl.axis('off')
    cax = axl.inset_axes([0.05, 0.52, 0.90, 0.26])
    sm = cm.ScalarMappable(norm=colors.Normalize(0, hi), cmap='turbo')
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    style_colorbar(fig, cb, cb_title, np.linspace(0, hi, N_TICKS), '{:.0f}',
                   bar_frac=0.90)
    verify_or_fix(fig, ax, cb, gs, os.path.basename(path))
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    os.remove(tmp)
    return mag, hi, n_arrow


def figures(pair):
    got = _load(pair)
    if got is None:
        print(f'{pair}: no grain track'); return
    a, u = got
    g = json.load(open(f'{OUT}/{GEO_A[pair]}_geom.json'))
    r_spec = 0.5 * float(g['diameter_mm'])

    # COLOUR SCALE = the column shortening this load step, measured in stage 1
    # without any correlation.  So the top of the bar always means "this grain
    # travelled as far as the punch did", which is comparable between figures
    # and is a number the reader can check, unlike a percentile of the data
    # being plotted.  The unloaded pair shortened by zero, so it falls back to
    # a few times the noise floor and its figure is near-empty by construction.
    dL = _shortening(pair)
    clim = max((dL or 0.0) * 1000.0, 4 * MIN_MAG)

    p = f'{FIG}/sand_disp3d_{pair}.png'
    mag, hi, na = _render(a, u, 'displacement  (µm)', p, r_spec, clim)
    scale = f'{dL:.2f} mm, the measured column shortening' if dL else \
            f'{clim:.0f} um, four times the tracking noise floor'
    record(p, pair, f'{NICE[pair]}: displacement of every tracked sand grain, '
           "referenced to the support end so the rig's own translation and "
           'tilt are removed. +z is up, the punch drives the top of the frame '
           'and the support holds the bottom. Grains below twice the tracking '
           'noise floor are drawn as pale spheres without an arrow; the colour '
           f'bar tops out at {scale}.',
           f'{na} arrows of {len(a)} tracked grains, median '
           f'{np.median(mag) if len(mag) else 0:.0f} um')
    print(f'wrote {p}   {na} arrows of {len(a)} grains   median '
          f'{np.median(mag) if len(mag) else 0:.0f} um   scale 0-{hi:.0f} um',
          flush=True)

    ur = _layer_relative(a, u)
    p = f'{FIG}/sand_motion_{pair}.png'
    mag, hi, na = _render(a, ur, 'rearrangement  (µm)', p, r_spec, clim)
    record(p, pair, f'{NICE[pair]}: the same grain motion with each grain '
           'measured against the median of its own height band, so what is '
           'left is rearrangement rather than the column shortening. Same '
           'colour scale as the displacement figure.',
           f'{na} arrows, median {np.median(mag) if len(mag) else 0:.0f} um')
    print(f'wrote {p}   {na} arrows   median '
          f'{np.median(mag) if len(mag) else 0:.0f} um', flush=True)


if __name__ == '__main__':
    for k in (sys.argv[1:] or list(NICE)):
        figures(k)
    write_index(f'{FIG}/sand_figure_captions.csv')
