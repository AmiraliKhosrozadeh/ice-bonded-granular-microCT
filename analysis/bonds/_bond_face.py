"""Adhesive vs cohesive, redone without the two things that broke the first attempt.

WHAT WAS WRONG
1. The distance transform runs on the bead LABEL, so the smallest distance any
   voxel outside a bead can have is 1 voxel.  With q = min(dA,dB)/(dA+dB) that
   puts a floor q >= 1/(dA+dB): the adhesive class (q <= 0.15) is arithmetically
   unreachable unless the surfaces are more than 6.67 voxels apart, and at a
   contact with one free voxel q is IDENTICALLY 0.5.  In 88 of 132 cracked
   throats the verdict was forced by the grid, not measured.
2. Even where it was reachable, a crack lying on a bead surface sits inside the
   4.57-voxel grey blur at that surface and is not segmented as crack.

FIX 1 -- SUB-VOXEL SURFACE
The bead surface is taken as the half-grey isosurface, not the label boundary.
The throat box is trilinearly upsampled by UP and the distance transforms are
computed there, so the distance floor is 1/UP of a voxel instead of 1.

FIX 2 -- ONLY WHERE THE QUESTION IS ANSWERABLE
Classification is restricted to throats whose surfaces are at least MIN_DSUM
voxels apart, so both classes are reachable and both bead faces have room for a
shell that is clear of the other bead.

FIX 3, AND IT IS THE REAL ONE -- LOOK AT THE TWO BEAD FACES, NOT THE GAP
The user's observation: if binder is still stuck to one bead and the opposite
bead is bare, the bond came off that second bead -- that IS adhesive failure,
and it is what a fractographer looks at.  It is a better observable than q for
one specific reason: it is a COMPARISON BETWEEN THE TWO FACES OF THE SAME
THROAT, so the grey blur, the threshold, the partial volume and the opening
displacement are common to both terms and cancel.  q has no such protection.

    cA = binder fraction of a shell SHELL_LO..SHELL_HI voxels off bead A's
         surface, inside the throat
    cB = the same for bead B

    both high                      COHESIVE   binder held on both beads,
                                              the middle broke
    one high, one low              ADHESIVE   named by which face is bare
    both low                       STRIPPED   no binder left on either face
    asymmetry |cA-cB|/(cA+cB)      how one-sided the failure is

The scan-1 throats are measured the same way and are the reference for what
"nothing has failed yet" looks like.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from _spec import (B, OUT, FIG, UM, MM3, TAG, ALN, SCANS, FIRST, LAST, MIDDLE, CHAIN, PUB,
                   p_ct, p_lab, p_ph, p_crack, p_sample, p_core, have, openv)

UP = 3                 # upsampling factor: distance floor becomes 1/3 voxel
MARGIN = 10            # voxels of context around the throat box
MIN_DSUM = 7.0         # only throats this far apart can answer the question
FOOT = 0.40
SHELL_LO, SHELL_HI = 1.0, 3.0     # the face shell, in original voxels
PH_BINDER, PH_PORE = 2, 1
C_HIGH, C_LOW = 0.50, 0.20
QBINS = np.linspace(0.0, 0.5, 26)
NPROC = 5

_G = {}


def _init(scan, thalf):
    _G['lab'] = openv(
        f'{B}/data/bead_labels_scan{scan:02d}{ALN}.tif', mode='r')
    _G['ct'] = openv(f'{B}/data/ct_scan{scan:02d}{ALN}.tif',
                               mode='r')
    _G['ph'] = openv(
        f'{B}/data/phases_scan{scan:02d}{ALN}.tif', mode='r')
    _G['crk'] = openv(
        f'{B}/results_voidcrack/crack_scan{scan:02d}.tif', mode='r')
    _G['core'] = openv(f'{OUT}/core_scan{scan:02d}.tif', mode='r')
    _G['thalf'] = thalf
    # Not every volume of a scan has the same number of slices: on some
    # specimens the crack/sample stacks are a few slices shorter than the
    # labels.  Crop to the common depth, or the boolean combinations below
    # fail with a broadcast error.
    _G['sh'] = tuple(min(v.shape[k] for v in
                         (_G['lab'], _G['ct'], _G['ph'], _G['crk'], _G['core']))
                     for k in range(3))



def _one(job):
    ia, ib, ca, cb, ra, rb = job
    lab, ct, ph = _G['lab'], _G['ct'], _G['ph']
    crk, core, thalf = _G['crk'], _G['core'], _G['thalf']
    sh = _G['sh']
    # box just big enough to hold the gap and a slab of each bead
    rth = FOOT * 0.5 * (ra + rb)
    lo, hi = [], []
    mid = 0.5 * (np.asarray(ca) + np.asarray(cb))
    half = rth + MARGIN
    ext = 0.5 * float(np.linalg.norm(np.asarray(cb) - np.asarray(ca))) + MARGIN
    for k in range(3):
        lo.append(int(max(0, mid[k] - max(half, ext * 0.55))))
        hi.append(int(min(sh[k], mid[k] + max(half, ext * 0.55))))
    sl = tuple(slice(lo[k], hi[k]) for k in range(3))
    labc = np.asarray(lab[sl])
    if not (labc == ia).any() or not (labc == ib).any():
        return None
    ctc = np.asarray(ct[sl]).astype(np.float32)
    phc = np.asarray(ph[sl])
    ckc = (np.asarray(crk[sl]) > 0) & (np.asarray(core[sl]) > 0)

    # ---- fine grid ------------------------------------------------------
    z = [np.linspace(0, s - 1, (s - 1) * UP + 1) for s in labc.shape]
    G = np.meshgrid(*z, indexing='ij')
    C = np.stack(G, axis=0)
    ct_f = ndi.map_coordinates(ctc, C, order=1, mode='nearest')
    lab_f = ndi.map_coordinates(labc.astype(np.float32), C, order=0,
                                mode='nearest')
    ph_f = ndi.map_coordinates(phc.astype(np.float32), C, order=0,
                               mode='nearest')
    ck_f = ndi.map_coordinates(ckc.astype(np.float32), C, order=0,
                               mode='nearest') > 0.5
    del C, G

    # bead bodies from the GREY isosurface, attributed by the label
    solid = ct_f >= thalf
    mA = solid & (np.abs(lab_f - ia) < 0.5)
    mB = solid & (np.abs(lab_f - ib) < 0.5)
    if not mA.any() or not mB.any():
        return None
    beads_f = solid & (lab_f > 0.5)
    dA = ndi.distance_transform_edt(~mA) / UP
    dB = ndi.distance_transform_edt(~mB) / UP
    inside = ph_f > 0.5
    free = inside & ~beads_f
    if not free.any():
        return None
    ssum = dA + dB
    g = float(ssum[free].min())
    _, idx = ndi.distance_transform_edt(~beads_f, return_indices=True)
    near = lab_f[idx[0], idx[1], idx[2]]
    slack = rth * rth / (0.5 * (ra + rb) + 0.5 * g)
    lens = (free & (ssum <= g + slack)
            & ((np.abs(near - ia) < 0.5) | (np.abs(near - ib) < 0.5)))
    n_lens = int(lens.sum())
    if n_lens < 200:
        return None

    binder = lens & (np.abs(ph_f - PH_BINDER) < 0.5)
    crack = lens & ck_f
    q = np.where(lens, np.minimum(dA, dB) / np.maximum(ssum, 1e-6), np.nan)

    # ---- the two bead faces --------------------------------------------
    def cover(d):
        shell = lens & (d >= SHELL_LO) & (d <= SHELL_HI)
        n = int(shell.sum())
        if n < 50:
            return np.nan, 0, np.nan
        b = float((shell & binder).sum()) / n
        c = float((shell & crack).sum()) / n
        return b, n, c

    cA, nA, kA = cover(dA)
    cB, nB, kB = cover(dB)

    def hq(m):
        return (np.histogram(q[m], bins=QBINS)[0] if m.any()
                else np.zeros(len(QBINS) - 1, dtype=np.int64))

    med = lambda m: float(np.median(q[m])) if m.any() else np.nan
    lo_c, hi_c = (min(cA, cB), max(cA, cB)) if (cA == cA and cB == cB) \
        else (np.nan, np.nan)
    if not (cA == cA and cB == cB):
        cls = 'unmeasured'
    elif hi_c >= C_HIGH and lo_c >= C_HIGH:
        cls = 'cohesive'
    elif hi_c >= C_HIGH and lo_c < C_LOW:
        cls = 'adhesive'
    elif hi_c < C_LOW:
        cls = 'stripped'
    else:
        cls = 'mixed'
    bare = (int(ia) if cA == lo_c else int(ib)) if cls == 'adhesive' else 0

    return dict(
        row=dict(
            bead_a=int(ia), bead_b=int(ib),
            gap_sub_vox=round(g, 3), gap_sub_um=round(g * UM, 1),
            lens_fine_vox=n_lens,
            binder_frac=round(float(binder.sum()) / n_lens, 4),
            crack_frac=round(float(crack.sum()) / n_lens, 4),
            cover_A=round(cA, 4) if cA == cA else None,
            cover_B=round(cB, 4) if cB == cB else None,
            crack_at_A=round(kA, 4) if kA == kA else None,
            crack_at_B=round(kB, 4) if kB == kB else None,
            shell_A_vox=nA, shell_B_vox=nB,
            asymmetry=(round(abs(cA - cB) / max(cA + cB, 1e-9), 4)
                       if cA == cA and cB == cB else None),
            face_class=cls, bare_bead=bare,
            q_crack_med=round(med(crack), 4) if crack.any() else None,
            q_binder_med=round(med(binder), 4) if binder.any() else None,
            q_crack_min=(round(float(np.nanmin(q[crack])), 4)
                         if crack.any() else None)),
        h_crack=hq(crack), h_binder=hq(binder))


def landmark_half(scan):
    """Half-grey level between the pore and the grain plateau, PER SCAN.

    It has to be per scan.  The grey scale is not comparable between scans in
    this dataset -- Glass_75_1000_T6 has an air/ice threshold of 5055 at scan 1
    and 1184 at scan 2, a factor of four -- so a level taken from the first
    scan marks nothing as bead in the last one, every throat returns nothing,
    and the face CSV comes out empty.  That is exactly what happened.
    """
    lab = openv(p_lab(scan), mode='r')
    ct = openv(p_ct(scan), mode='r')
    ph = openv(p_ph(scan), mode='r')
    po, gr = [], []
    for z in range(0, lab.shape[0], 40):
        l = np.asarray(lab[z])
        c = np.asarray(ct[z]).astype(np.float32)
        p = np.asarray(ph[z])
        b = l > 0
        m = (p > 0) & ~b
        if b.sum() > 500:
            gr.append(np.median(c[b]))
        if m.sum() > 500:
            po.append(np.percentile(c[m], 5))
    return 0.5 * (float(np.median(po)) + float(np.median(gr)))


def run(scan, thalf, only_cracked):
    d = pd.read_csv(f'{OUT}/{TAG}_throats_scan{scan:02d}.csv')
    sub = d[d.dsum_min >= MIN_DSUM]
    if only_cracked:
        sub = sub[sub.crack_vox >= 200]
    c = np.load(f'{OUT}/centroids_scan{scan:02d}.npz')
    pos = {int(i): (p, r) for i, p, r in zip(c['ids'], c['com'], c['req'])}
    jobs = []
    for r in sub.itertuples():
        ia, ib = int(r.bead_a), int(r.bead_b)
        if ia not in pos or ib not in pos:
            continue
        ca, ra = pos[ia]
        cb, rb = pos[ib]
        jobs.append((ia, ib, tuple(ca), tuple(cb), float(ra), float(rb)))
    print(f'scan {scan}: {len(jobs)} throats with surfaces >= {MIN_DSUM} vox '
          f'apart{" and cracked" if only_cracked else ""}', flush=True)

    rows = []
    hc = np.zeros(len(QBINS) - 1, dtype=np.int64)
    hb = np.zeros(len(QBINS) - 1, dtype=np.int64)
    with Pool(NPROC, initializer=_init, initargs=(scan, thalf)) as pool:
        for k, r in enumerate(pool.imap_unordered(_one, jobs, chunksize=1), 1):
            if r is not None:
                rows.append(r['row'])
                hc += r['h_crack']
                hb += r['h_binder']
            if k % 20 == 0:
                print(f'   {k}/{len(jobs)}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/{TAG}_face_scan{scan:02d}.csv', index=False)
    qc = 0.5 * (QBINS[:-1] + QBINS[1:])
    pd.DataFrame(dict(q=qc, crack=hc, binder=hb)).to_csv(
        f'{OUT}/{TAG}_face_qhist_scan{scan:02d}.csv', index=False)
    return df, hc, hb, qc


def main():
    t_first, t_last = landmark_half(FIRST), landmark_half(LAST)
    print(f'half-grey isosurface level: scan {FIRST} {t_first:.0f}, '
          f'scan {LAST} {t_last:.0f}')
    print()

    d1, _, hb1, qc = run(FIRST, t_first, only_cracked=False)
    d3, hc3, hb3, _ = run(LAST, t_last, only_cracked=True)

    print(f'\n{"="*76}\nSUB-VOXEL q  (floor is now 1/{UP} voxel, was 1)\n{"="*76}')
    c = d3[d3.q_crack_med.notna()]
    print(f'  cracked throats classified: {len(c)}')
    print(f'  smallest median q seen     : {c.q_crack_med.min():.3f}'
          f'   (reachable floor here ~{1.0/UP/MIN_DSUM:.3f})')
    print(f'  smallest single crack voxel: {c.q_crack_min.min():.3f}')
    print(f'  median q, crack            : {c.q_crack_med.median():.3f}')
    print(f'  median q, binder scan 1    : {d1.q_binder_med.median():.3f}')

    print(f'\n{"="*76}\nFACE COVERAGE  -- binder still stuck to each bead\n{"="*76}')
    print(f'  shell {SHELL_LO}-{SHELL_HI} vox off each bead surface, '
          f'inside the throat')
    for nm, d in ((f'scan {FIRST}, nothing failed yet', d1),
                  (f'scan {LAST}, cracked throats', d3)):
        v = d[d.cover_A.notna() & d.cover_B.notna()]
        if not len(v):
            continue
        lo = v[['cover_A', 'cover_B']].min(axis=1)
        hi = v[['cover_A', 'cover_B']].max(axis=1)
        print(f'\n  {nm}   n = {len(v)}')
        print(f'    better face, median coverage  {hi.median():.3f}')
        print(f'    worse  face, median coverage  {lo.median():.3f}')
        print(f'    asymmetry |cA-cB|/(cA+cB)     median '
              f'{v.asymmetry.median():.3f}   90th {v.asymmetry.quantile(.9):.3f}')
        vc = v.face_class.value_counts()
        for k in ('cohesive', 'adhesive', 'mixed', 'stripped'):
            n = int(vc.get(k, 0))
            print(f'    {k:9s} {n:4d}   {100*n/len(v):5.1f}%')

    v3 = d3[d3.cover_A.notna()]
    adh = v3[v3.face_class == 'adhesive']
    if len(adh):
        print(f'\n  the {len(adh)} adhesive throats, bare face and its partner:')
        print(f'    {"beads":>13s} {"cover bare":>11s} {"cover held":>11s} '
              f'{"crack@bare":>11s} {"gap um":>8s}')
        for r in adh.itertuples():
            bare_c = r.cover_A if r.bare_bead == r.bead_a else r.cover_B
            held_c = r.cover_B if r.bare_bead == r.bead_a else r.cover_A
            kb = r.crack_at_A if r.bare_bead == r.bead_a else r.crack_at_B
            print(f'    {r.bead_a:5d}-{r.bead_b:<7d} {bare_c:11.3f} '
                  f'{held_c:11.3f} {kb if kb is not None else float("nan"):11.3f} '
                  f'{r.gap_sub_um:8.0f}')
    print(f'\n  -> {TAG}_face_scan{FIRST:02d}.csv, {TAG}_face_scan{LAST:02d}.csv', flush=True)


if __name__ == '__main__':
    main()
