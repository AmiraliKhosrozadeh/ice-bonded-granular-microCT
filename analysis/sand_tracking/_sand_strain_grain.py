"""Sand strain from the tracked grains, not from the continuum DVC.

WHY NOT THE DVC.  It converged on 10-11% of nodes for the two small-strain
pairs and on 1.6-1.7% for the two large-strain ones -- S2 2->3 returned status
2 at 228 of 13,122 nodes, the rest having run out of iterations.  The strain
built from those was -0.47 volumetric, i.e. the material losing 47% of its
volume, which the geometry contradicts: the specimen volume changes by 0.2%
over that step.  A field that fails exactly where the deformation is cannot
carry the strain figures.

The grain tracking does not have that failure mode.  It is reciprocally
confirmed -- matches found twice from independent starting points, with a
deliberate-shift null showing a 0% false-positive rate -- and it works from
the DL labels rather than from grey-value correlation, so decorrelation of the
speckle does not stop it.

STRAIN IS THEN COMPUTED THE SAME WAY AS FOR THE BEADS, through
_strain_measure: a least-squares displacement gradient over the k nearest
neighbours WITH an intercept, decomposed as Green-Lagrange.  Both details were
paid for once already on the bead specimens -- dropping the intercept created a
false wall-concentration that a null control exposed, and the infinitesimal
measure is not frame-indifferent at the 9-19 degree rotations those specimens
showed.  Sand rotates less, but there is no reason to use a weaker measure and
every reason to keep one definition across the three materials.

Three fields per pair, coloured on a diverging scale where the sign matters:

  volumetric    negative = the neighbourhood lost volume, i.e. compaction
  deviatoric    always positive, shape change irrespective of direction
  axial         the zz component, negative = shortening along the load
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
from _plotrules import (record, write_index, style_colorbar, N_TICKS, autocrop)
from _legend_guard import verify_or_fix
from _strain_measure import grad_u
from _sand_motion import _load, NICE

OUT = f'{SC}/sand_out'
FIG = f'{SC}/preview'
pv.OFF_SCREEN = True
K = 16                     # neighbours in the gradient fit
MAX_DOT = 30000
GEO_A = {'S1_12': 'S1-1', 'S2_12': 'S2-1', 'S2_23': 'S2-2', 'S3_12': 'S3-1'}


def strains(a, u):
    """Green-Lagrange volumetric, deviatoric and axial strain per grain."""
    G = grad_u(a, u, k=K, intercept=True)
    F = G + np.eye(3)[None, :, :]
    E = 0.5 * (np.einsum('nji,njk->nik', F, F) - np.eye(3)[None, :, :])
    vol = np.trace(E, axis1=1, axis2=2)
    dev = E - (vol / 3.0)[:, None, None] * np.eye(3)[None, :, :]
    devq = np.sqrt(2.0 / 3.0 * np.einsum('nij,nij->n', dev, dev))
    return vol, devq, E[:, 0, 0]


def scene(P, val, clim, cmap, span):
    pl = pv.Plotter(off_screen=True, window_size=(940, 1560))
    pl.set_background('white')
    c = pv.PolyData(P[:, [2, 1, 0]])
    c['v'] = val
    pl.add_mesh(c.glyph(geom=pv.Sphere(radius=0.075, theta_resolution=9,
                                       phi_resolution=9),
                        scale=False, orient=False),
                scalars='v', cmap=cmap, clim=clim, opacity=0.92,
                smooth_shading=True, show_scalar_bar=False)
    cen = np.array([P[:, 2].mean(), P[:, 1].mean(), P[:, 0].mean()])
    pl.camera_position = [(cen[0], cen[1] - 2.7 * span, cen[2]), cen, (0, 0, 1)]
    pl.camera.azimuth = 20
    return pl


def figure(pair):
    got = _load(pair)
    if got is None:
        print(f'{pair}: no track'); return
    a, u = got
    if len(a) < 500:
        print(f'{pair}: only {len(a)} grains, too few'); return
    vol, dev, axial = strains(a, u)

    rng = np.random.default_rng(0)
    k = np.arange(len(a))
    if len(k) > MAX_DOT:
        k = rng.choice(len(a), MAX_DOT, replace=False)
    P = a[k]
    span = float(np.ptp(a[:, 0]))

    print(f'\n{NICE[pair]}   {len(a)} grains, k={K} neighbours')
    for nm, v in (('volumetric', vol), ('deviatoric', dev), ('axial', axial)):
        print(f'   {nm:<11s} p5 {np.percentile(v,5):+.3f}   '
              f'median {np.median(v):+.3f}   p95 {np.percentile(v,95):+.3f}')

    lim_v = float(np.percentile(np.abs(vol), 97)) or 0.05
    lim_a = float(np.percentile(np.abs(axial), 97)) or 0.05
    hi_d = float(np.percentile(dev, 97)) or 0.05
    specs = [('volumetric  ($-$ compaction)', vol[k], 'RdBu_r',
              (-lim_v, lim_v), '{:+.2f}'),
             ('deviatoric', dev[k], 'inferno_r', (0.0, hi_d), '{:.2f}'),   # dark = high strain, as the bead figures
             ('axial  $E_{zz}$', axial[k], 'RdBu_r', (-lim_a, lim_a),
              '{:+.2f}')]

    tmps = []
    for nm, v, cmap, clim, fmt in specs:
        pl = scene(P, v, clim, cmap, span)
        t = f'{FIG}/_st_{pair}_{len(tmps)}.png'
        pl.screenshot(t); pl.close()
        tmps.append(t)

    # EACH PANEL IS BUILT AS ITS OWN 11-INCH FIGURE, then the three are
    # composed side by side.  style_colorbar sizes its lettering from the
    # FIGURE width, so three bars sharing one wide figure each got text scaled
    # for the whole width and the tick labels of neighbouring bars ran into one
    # another.  Giving each bar its own correctly sized figure keeps the shared
    # sizing rule and the legend guard, which is the point of having them.
    panels = []
    for t, (nm, _, cmap, clim, fmt) in zip(tmps, specs):
        im = autocrop(plt.imread(t))
        f1 = plt.figure(figsize=(11, 11 * im.shape[0] / im.shape[1] + 3.2))
        g1 = f1.add_gridspec(2, 1, height_ratios=[12, 3.2], hspace=0.02)
        ax1 = f1.add_subplot(g1[0]); ax1.imshow(im); ax1.set_axis_off()
        ax1.set_title(nm, fontsize=40, pad=18)
        axl = f1.add_subplot(g1[1]); axl.axis('off')
        cax = axl.inset_axes([0.05, 0.55, 0.90, 0.26])
        sm = cm.ScalarMappable(norm=colors.Normalize(*clim), cmap=cmap)
        cb = f1.colorbar(sm, cax=cax, orientation='horizontal')
        style_colorbar(f1, cb, 'strain', np.linspace(clim[0], clim[1], 3),
                       fmt, bar_frac=0.90)
        verify_or_fix(f1, ax1, cb, g1, f'sand_strain_{pair}_{nm}')
        tp = f'{FIG}/_sp_{pair}_{len(panels)}.png'
        f1.savefig(tp, dpi=170, facecolor='white', bbox_inches='tight')
        plt.close(f1)
        panels.append(tp)

    ims = [plt.imread(x) for x in panels]
    hmax = max(i.shape[0] for i in ims)
    ws = [i.shape[1] * hmax / i.shape[0] for i in ims]
    fig = plt.figure(figsize=(sum(ws) / 260.0, hmax / 260.0))
    gs = fig.add_gridspec(1, len(ims), width_ratios=ws, wspace=0.015)
    for j, im in enumerate(ims):
        ax = fig.add_subplot(gs[0, j]); ax.imshow(im); ax.set_axis_off()
    p = f'{FIG}/sand_strain_{pair}.png'
    fig.savefig(p, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    for t in tmps + panels:
        os.remove(t)
    record(p, pair, f'{NICE[pair]}: Green-Lagrange strain per tracked grain, '
           f'from a least-squares displacement gradient over {K} neighbours '
           'with an intercept -- the same measure as the bead specimens. '
           'Punch at the top.',
           f'{len(a)} reciprocally-confirmed grains; volumetric median '
           f'{np.median(vol):+.3f}, deviatoric median {np.median(dev):.3f}')
    print(f'   wrote {p}', flush=True)


if __name__ == '__main__':
    for pr in (sys.argv[1:] or ['S2_23', 'S2_12', 'S3_12', 'S1_12']):
        figure(pr)
    write_index(f'{FIG}/sand_figure_captions.csv')
