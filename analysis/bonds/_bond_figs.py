"""Figures for the bond / interface / bead failure analysis on Alumina_75_1800_T7.

One plot per file, full box, legend below the axes, no numbers written on the
data -- the project plot rules.  Scans are merged into one figure per metric
rather than split across files.

THE FIGURES AND WHAT EACH ONE IS FOR

  q_position          the answer.  Where the crack sits inside the bond,
                      against the two nulls (binder that was there to break,
                      pore that was already open).
  q_ratio             the same thing as a ratio, so "the crack prefers this
                      position" is read off a line at 1 instead of by eye.
  bond_radius         r_bond / R_bead, the quantity the snow literature
                      reports, before and after loading.
  connectivity        how many throats still have a solid path bead-to-bead.
  coordination        bonded neighbours per bead, before and after.
  crack_vs_gap        does a wide gap crack more than a tight contact?
  crack_vs_axis       does a contact aligned with the load crack more?
                      This is the mechanical test: if bonds fail because they
                      carry force, axis-aligned contacts must fail first.
  per_bead_broken     per bead, the fraction of its own throats holding crack.
  failure_class       adhesive / cohesive / mixed / intact counts.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _spec import (B, OUT, FIG, UM, MM3, TAG, ALN, SCANS, FIRST, LAST, PUB,
                   p_ct, p_lab, p_ph, p_crack, p_sample, p_core, have)

# --- no text is burned into a figure ---------------------------------------
# Every title, suptitle and row label used to carry the numbers.  They are in
# figure_index.xlsx instead, keyed by file name, so a figure can be dropped
# into a paper and captioned there.  Axis labels stay: they name the axis, they
# are not data.  Legends and scale bars stay.
def _notitle(*a, **k):
    return None


_FS = 24
# --- nothing on a figure is allowed below FLOOR point ------------------------
# These are read in a paper column or from the back of a room.  The legend and
# the colorbar carry the units and the numbers, so they are the last things
# that should be shrunk to make a render fit -- which is what the old
# _FS - 6 / _FS - 8 reductions and the 15pt colorbar ticks were doing.
FLOOR = 22

C1, C3 = '#1565c0', '#e00000'
C_BIND, C_PORE = '#2e7d32', '#8d6e63'
Q_ADH, Q_COH = 0.15, 0.30
# Measured, not assumed (_pv_band.py): the grey at a bead surface takes 4.57
# voxels to fall from the alumina plateau to the surrounding plateau, and the
# label boundary sits within about 0.4 voxels of the half-grey isosurface.  So
# q is NOT systematically displaced -- the label is in the right place -- but a
# crack thinner than about 2 voxels lying ON the surface is absorbed into that
# blur and is not detected.  The limitation is detection sensitivity at small
# q, and it applies to the crack and to the binder null alike, which is why the
# ratio of the two is the defensible statistic and the absolute q is not.
PV_VOX = 2.0


def style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': _FS,
        'axes.titlesize': _FS + 2, 'axes.titleweight': 'bold',
        'axes.labelsize': _FS + 4, 'axes.labelweight': 'bold',
        'axes.linewidth': 2.2, 'axes.edgecolor': 'black',
        'axes.spines.top': True, 'axes.spines.right': True,
        'axes.spines.left': True, 'axes.spines.bottom': True,
        'axes.facecolor': 'white', 'axes.grid': True,
        'grid.color': '#d0d0d0', 'grid.linewidth': 1.0,
        'xtick.labelsize': _FS, 'ytick.labelsize': _FS,
        'xtick.major.size': 8, 'ytick.major.size': 8,
        'xtick.major.width': 2.0, 'ytick.major.width': 2.0,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.pad': 8, 'ytick.major.pad': 8,
        'legend.fontsize': _FS, 'legend.frameon': False,
        'figure.facecolor': 'white', 'savefig.dpi': 260,
        'savefig.bbox': 'tight'})


def finish(fig, ax, path, ncol=2, y=-0.18):
    h, l = ax.get_legend_handles_labels()
    if h:
        ax.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, y),
                  ncol=ncol, frameon=False, fontsize=_FS - 2)
    fig.savefig(path, dpi=260, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'   -> {os.path.basename(path)}', flush=True)


def classify(r):
    if r.crack_vox < 200:
        return 'intact'
    if r.q_crack_med <= Q_ADH:
        return 'adhesive'
    if r.q_crack_med >= Q_COH:
        return 'cohesive'
    return 'mixed'


def main():
    style()
    d1 = pd.read_csv(f'{OUT}/{TAG}_throats_scan{FIRST:02d}.csv')
    d3 = pd.read_csv(f'{OUT}/{TAG}_throats_scan{LAST:02d}.csv')
    q1 = pd.read_csv(f'{OUT}/{TAG}_qhist_scan{FIRST:02d}.csv')
    q3 = pd.read_csv(f'{OUT}/{TAG}_qhist_scan{LAST:02d}.csv')

    # ---- 1. q position -------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 8))
    for h, c, lab in ((q3.crack, C3, f'crack, scan {LAST}'),
                      (q1.binder, C_BIND, f'binder, scan {FIRST}  (null: what could break)'),
                      (q1.pore, C_PORE, f'pore, scan {FIRST}  (null: already open)')):
        s = h.sum()
        if s:
            ax.plot(q1.q, 100 * h / s, color=c, lw=3.4, label=lab)
    ax.axvline(Q_ADH, color='0.4', lw=2, ls=':')
    ax.axvline(Q_COH, color='0.4', lw=2, ls=':')
    ax.set_xlabel('q   (0 = on a bead surface, 0.5 = mid-bond)')
    ax.set_ylabel('% of that phase')
    _notitle('Where inside the bond the crack sits')
    ax.set_xlim(0, 0.5)
    finish(fig, ax, f'{FIG}/q_position.png', ncol=1, y=-0.16)

    # ---- 2. q ratio ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 8))
    pc = q3.crack / max(q3.crack.sum(), 1)
    for base, c, lab in ((q1.binder, C_BIND, 'crack / binder'),
                         (q1.pore, C_PORE, 'crack / pore')):
        pb = base / max(base.sum(), 1)
        r = np.where(pb > 0, pc / np.maximum(pb, 1e-12), np.nan)
        ax.plot(q1.q, r, color=c, lw=3.4, label=lab)
    ax.axhline(1.0, color='black', lw=2.4, ls='--', label='no preference')
    ax.set_xlabel('q   (0 = on a bead surface, 0.5 = mid-bond)')
    ax.set_ylabel('crack density / null density')
    _notitle('Does the crack prefer a position in the bond?')
    ax.set_xlim(0, 0.5)
    finish(fig, ax, f'{FIG}/q_ratio.png', ncol=3, y=-0.16)

    # ---- 3. bond radius ------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 8))
    bins = np.linspace(0, 0.45, 31)
    for d, c, lab in ((d1, C1, f'scan {FIRST}'), (d3, C3, f'scan {LAST}')):
        v = d.loc[d.connected & d.r_bond_over_R.notna(), 'r_bond_over_R']
        if len(v):
            ax.hist(v, bins=bins, color=c, alpha=0.55, label=f'{lab}  (n={len(v)})')
    ax.set_xlabel(r'$r_{\mathrm{bond}}\,/\,R_{\mathrm{bead}}$')
    ax.set_ylabel('throats')
    _notitle('Bond size, before and after loading')
    finish(fig, ax, f'{FIG}/bond_radius.png')

    # ---- 4. connectivity -----------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 8))
    x = np.arange(2)
    w = 0.36
    for k, (d, c, lab) in enumerate(((d1, C1, f'scan {FIRST}'), (d3, C3, f'scan {LAST}'))):
        vals = [100 * d.connected.mean(), 100 * (~d.connected).mean()]
        ax.bar(x + (k - 0.5) * w, vals, w, color=c, edgecolor='black',
               linewidth=1.6, label=f'{lab}  (n={len(d)})')
    ax.set_xticks(x)
    ax.set_xticklabels(['connected', 'detached'])
    ax.set_ylabel('% of throats')
    _notitle('Is there still a solid path from bead to bead?')
    finish(fig, ax, f'{FIG}/connectivity.png')

    # ---- 5. coordination -----------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 8))
    bins = np.arange(-0.5, 13.5, 1)
    for d, c, lab in ((d1, C1, f'scan {FIRST}'), (d3, C3, f'scan {LAST}')):
        sub = d[d.connected]
        ids = pd.concat([sub.bead_a, sub.bead_b])
        cn = ids.value_counts().reindex(
            pd.concat([d.bead_a, d.bead_b]).unique(), fill_value=0)
        ax.hist(cn.values, bins=bins, color=c, alpha=0.55,
                label=f'{lab}  mean {cn.mean():.2f}')
    ax.set_xlabel('bonded neighbours per bead')
    ax.set_ylabel('beads')
    _notitle('Coordination number, bonds still connected')
    finish(fig, ax, f'{FIG}/coordination.png')

    # ---- 6. crack fraction vs gap --------------------------------------
    fig, ax = plt.subplots(figsize=(11, 8))
    e = np.array([0, 1, 2, 3, 4, 6, 8, 10, 12])
    ctr = 0.5 * (e[:-1] + e[1:])
    g = np.digitize(d3.gap_vox, e) - 1
    m = [100 * d3.crack_frac[g == i].mean() if (g == i).any() else np.nan
         for i in range(len(ctr))]
    n = [int((g == i).sum()) for i in range(len(ctr))]
    ax.plot(ctr * UM, m, color=C3, lw=3.4, marker='o', ms=11,
            label=f'scan {LAST}, mean over throats in the bin')
    ax.set_xlabel(r'surface-to-surface gap  ($\mu$m)')
    ax.set_ylabel('% of throat volume that is crack')
    _notitle('Do wide gaps crack more than tight contacts?')
    finish(fig, ax, f'{FIG}/crack_vs_gap.png', ncol=1)

    # ---- 7. crack fraction vs contact axis inclination ------------------
    fig, ax = plt.subplots(figsize=(11, 8))
    e = np.arange(0, 91, 10)
    ctr = 0.5 * (e[:-1] + e[1:])
    g = np.digitize(d3.axis_dip_deg, e) - 1
    m = [100 * d3.crack_frac[g == i].mean() if (g == i).any() else np.nan
         for i in range(len(ctr))]
    frac_det = [100 * (~d3.connected[g == i]).mean() if (g == i).any() else np.nan
                for i in range(len(ctr))]
    ax.plot(ctr, m, color=C3, lw=3.4, marker='o', ms=11,
            label='crack share of throat volume')
    ax.plot(ctr, frac_det, color='#ef6c00', lw=3.4, marker='s', ms=11,
            label='throats detached')
    ax.set_xlabel('contact axis to load axis  (deg)\n'
                  '0 = axis along the load,  90 = axis across it')
    ax.set_ylabel('%')
    _notitle('Do load-aligned contacts fail first?')
    finish(fig, ax, f'{FIG}/crack_vs_axis.png', ncol=1, y=-0.26)

    # ---- 8. per-bead broken fraction ------------------------------------
    fig, ax = plt.subplots(figsize=(11, 8))
    long = pd.concat([
        d3[['bead_a', 'crack_vox', 'connected']].rename(
            columns={'bead_a': 'bead'}),
        d3[['bead_b', 'crack_vox', 'connected']].rename(
            columns={'bead_b': 'bead'})])
    per = long.groupby('bead').agg(
        n=('crack_vox', 'size'),
        cracked=('crack_vox', lambda s: int((s > 200).sum())),
        detached=('connected', lambda s: int((~s.astype(bool)).sum())))
    per['frac_cracked'] = per.cracked / per.n
    per['frac_detached'] = per.detached / per.n
    per.to_csv(f'{OUT}/{TAG}_per_bead_bonds_scan{LAST:02d}.csv')
    bins = np.linspace(0, 1, 21)
    ax.hist(per.frac_cracked, bins=bins, color=C3, alpha=0.6,
            label=f'throats holding crack  (mean {per.frac_cracked.mean():.2f})')
    ax.hist(per.frac_detached, bins=bins, color='#ef6c00', alpha=0.6,
            label=f'throats detached  (mean {per.frac_detached.mean():.2f})')
    ax.set_xlabel('fraction of that bead\'s own throats')
    ax.set_ylabel('beads')
    _notitle('How much of each bead has let go')
    finish(fig, ax, f'{FIG}/per_bead_broken.png', ncol=1, y=-0.18)

    # ---- 9. failure class -----------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 8))
    order = ['intact', 'adhesive', 'mixed', 'cohesive']
    cols = {'intact': '#9e9e9e', 'adhesive': '#1565c0',
            'mixed': '#ef6c00', 'cohesive': '#e00000'}
    d3 = d3.copy()
    d3['cls'] = d3.apply(classify, axis=1)
    cnt = d3.cls.value_counts().reindex(order, fill_value=0)
    ax.bar(np.arange(len(order)), 100 * cnt.values / len(d3),
           color=[cols[o] for o in order], edgecolor='black', linewidth=1.8)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(['intact', 'adhesive\n(at the bead)', 'mixed',
                        'cohesive\n(mid-bond)'], fontsize=max(_FS - 2, FLOOR))
    ax.set_ylabel('% of throats')
    _notitle(f'Failure location, {len(d3)} throats at scan 3')
    fig.savefig(f'{FIG}/failure_class.png', dpi=260, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print('   -> failure_class.png', flush=True)
    d3.to_csv(f'{OUT}/{TAG}_throats_scan03_classified.csv', index=False)

    # ---- 10. one vote per throat, not one vote per voxel -----------------
    #
    # The pooled histogram is volume weighted, so a bond that opened wide
    # counts for more than one that barely parted.  That systematically
    # favours the cohesive reading, because an adhesive debond that has not
    # opened is one or two voxels thick.  Taking the MEDIAN q inside each
    # throat and then histogramming over throats gives every bond one vote.
    fig, ax = plt.subplots(figsize=(11, 8))
    bins = np.linspace(0, 0.5, 26)
    ax.hist(d3.loc[d3.crack_vox >= 200, 'q_crack_med'].dropna(), bins=bins,
            color=C3, alpha=0.6, label=f'crack, scan {LAST}')
    ax.hist(d1.q_binder_med.dropna(), bins=bins, color=C_BIND, alpha=0.55,
            label='binder, scan 1  (what could break)')
    ax.axvline(Q_ADH, color='0.4', lw=2, ls=':')
    ax.axvline(Q_COH, color='0.4', lw=2, ls=':')
    ax.set_xlabel('median q inside one throat')
    ax.set_ylabel('throats')
    _notitle('One vote per bond, not per voxel')
    ax.set_xlim(0, 0.5)
    finish(fig, ax, f'{FIG}/q_per_throat.png', ncol=1, y=-0.16)

    # ---- 11. paired within one throat ------------------------------------
    fig, ax = plt.subplots(figsize=(11, 8))
    v = d3.dq_paired.dropna()
    ax.hist(v, bins=np.linspace(-0.5, 0.5, 41), color=C3, alpha=0.7,
            label=f'{len(v)} throats,  median {v.median():+.3f}')
    ax.axvline(0.0, color='black', lw=2.6, ls='--',
               label='crack and binder in the same place')
    ax.set_xlabel('median q of the crack  minus  median q of the binder\n'
                  'both measured in the same throat, same scan')
    ax.set_ylabel('throats')
    _notitle('Where the crack sits relative to the binder still there')
    finish(fig, ax, f'{FIG}/q_paired.png', ncol=1, y=-0.26)

    # ---- 12. resolution: does gap width change the answer? ---------------
    fig, ax = plt.subplots(figsize=(11, 8))
    e = np.array([0, 1, 2, 3, 4, 6, 8, 12])
    ctr = 0.5 * (e[:-1] + e[1:])
    for d, c, lab, col in ((d3[d3.crack_vox >= 200], C3, f'crack, scan {LAST}',
                            'q_crack_med'),
                           (d1, C_BIND, f'binder, scan {FIRST}', 'q_binder_med')):
        g = np.digitize(d.gap_vox, e) - 1
        m = [d[col][g == i].median() if (g == i).sum() >= 5 else np.nan
             for i in range(len(ctr))]
        ax.plot(ctr * UM, m, color=c, lw=3.4, marker='o', ms=11, label=lab)
    ax.plot(ctr * UM, PV_VOX / (ctr + 1.0), color='0.35', lw=3.0, ls='--',
            label='below this line, partial volume')
    ax.set_ylim(0, 0.55)
    ax.set_xlabel(r'free gap between the bead surfaces  ($\mu$m)')
    ax.set_ylabel('median q')
    _notitle('Does the answer survive the resolution limit?')
    finish(fig, ax, f'{FIG}/q_by_gap.png', ncol=1, y=-0.20)

    # ---- printed summary -------------------------------------------------
    print(f'\n{"="*74}\nSUMMARY  Alumina_75_1800_T7\n{"="*74}')
    print(f'  throats measured        scan {FIRST} {len(d1)}   scan {LAST} {len(d3)}')
    print(f'  connected               scan {FIRST} {100*d1.connected.mean():5.1f}%'
          f'   scan {LAST} {100*d3.connected.mean():5.1f}%')
    for k in order:
        print(f'  {k:9s}             {int(cnt[k]):5d}  '
              f'{100*cnt[k]/len(d3):5.1f}%')
    crk = d3[d3.crack_vox >= 200]
    if len(crk):
        print(f'  of the {len(crk)} cracked throats, median q '
              f'{crk.q_crack_med.median():.3f}   '
              f'(binder at scan 1: {d1.q_binder_med.median():.3f})')
    v = d3.dq_paired.dropna()
    print(f'  paired, same throat and same scan: crack minus binder median '
          f'{v.median():+.3f};\n     crack further from the beads than the '
          f'surviving binder in {int((v>0).sum())} of {len(v)} throats '
          f'({100*(v>0).mean():.1f}%)')
    ok1 = d1[d1.connected & d1.r_bond_vox.notna()]
    area_mm2 = float(ok1.bond_area_vox.median()) * (UM / 1000.0) ** 2
    print(f'  bond cross-section at scan 1: median {area_mm2:.3f} mm2, '
          f'r_bond/R {ok1.r_bond_over_R.median():.3f}')
    print(f'  per-bead: mean {100*per.frac_cracked.mean():.1f}% of a bead\'s '
          f'throats hold crack, {100*per.frac_detached.mean():.1f}% detached')
    print(f'\n  -> {FIG}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f'  some figures skipped: {type(e).__name__}: {e}')
        traceback.print_exc()
