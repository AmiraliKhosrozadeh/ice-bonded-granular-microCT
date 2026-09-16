"""Contact-throat analysis for Alumina_75_1800_T7: where does each bond fail?

WHY A THROAT AND NOT A GLOBAL HISTOGRAM
The earlier global test asked "how far is a crack voxel from the nearest bead
surface".  It answers only on average, and it has no notion of which two beads
a given piece of crack belongs to.  A bond is a local object between an
identified PAIR of beads, so the measurement is done per pair.

THE THROAT (the "small frame with two particles and their bond")
For a pair A,B let dA and dB be the distances to their surfaces.  Free space
between them satisfies dA + dB = g, the surface-to-surface gap, exactly on the
shortest path and more elsewhere.  The throat is

    lens = free space  AND  dA + dB <= g + slack  AND  nearest bead is A or B

a lens-shaped region straddling the contact.  slack is set per pair so the lens
has a fixed footprint radius, FOOT x mean bead radius, making every throat the
same size in bead radii and therefore comparable.  The nearest-bead condition
stops a third bead's territory being claimed.

THE POSITION PARAMETER (this is the whole answer)
For every voxel in the throat

    q = min(dA, dB) / (dA + dB)          in [0, 0.5]

q = 0 sits on a bead surface, q = 0.5 sits exactly mid-bond.  It is normalised
by the LOCAL gap, so a tight contact and a loose one are on the same scale and
no length threshold is needed anywhere.

    crack piled up at q ~ 0     ->  ADHESIVE, the binder peeled off the bead
    crack piled up at q ~ 0.5   ->  COHESIVE, the binder split down its middle

THE NULL, WHICH IS WHAT MAKES THIS AN ARGUMENT AND NOT A PICTURE
A crack at mid-bond proves nothing on its own: mid-bond may simply be where the
material was.  Two nulls are measured in the SAME throats:

    q of the binder at scan 1   the material that was available to break
    q of the pore   at scan 1   space that was already open before loading

The result is the ratio of crack density to those, bin by bin.  If crack sits
at the same q as the pre-existing pore, the "cohesive" reading is just the
shape of the pore space and means nothing.

CONNECTED OR DETACHED, WHICH IS THE USER'S OWN CRITERION
Within the throat, is there a path from A to B through solid material (bead or
binder)?  Connected = the bond still carries load.  Detached = it does not.
Counted at scan 1 and at scan 3, so the drop is the fraction of bonds lost.

BOND RADIUS, SO THE NUMBERS ARE COMPARABLE TO THE LITERATURE
The neck is the binder cross-section on the plane through the narrowest free
point, normal to the centre line, taking the connected patch that contains
that point.  r_bond = sqrt(A / pi), reported as r_bond / R_bead.
It is deliberately NOT capped at the lens footprint -- the bond is however wide
it is -- and it is deliberately not "the thinnest slab of the lens", because
the lens tapers to a rim and its thinnest slab is that rim, which would report
a one-voxel bond for every contact in the specimen.

GRADED DAMAGE PER BOND
neck_severed = crack / (crack + binder) on that same neck plane, in [0, 1].
It is a geometric surrogate for the continuous damage variable a DEM bond model
carries, not the same quantity, and is labelled as such.

CRACK USED HERE
The interior ("failure") crack: crack minus a 20-voxel skin, the same class the
failure-plane figures use.  Surface crack wraps the specimen and is spalling at
the boundary, not a bond failing.
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from _spec import (B, OUT, FIG, UM, MM3, TAG, ALN, SCANS, FIRST, LAST, PUB,
                   p_ct, p_lab, p_ph, p_crack, p_sample, p_core, have, openv)


GAP_MAX = 12.0      # surface-to-surface gap that still counts as a contact
FOOT = 0.40         # throat footprint radius, in bead radii
MARGIN = 6          # crop margin around the two bead boxes
SKIN = 20           # specimen skin excluded from the crack, as in _failure_all
QBINS = np.linspace(0.0, 0.5, 26)
PH_PORE, PH_BINDER, PH_BEAD = 1, 2, 3
NPROC = 6

_G = {}


def _init(scan):
    _G['lab'] = openv(
        f'{B}/data/bead_labels_scan{scan:02d}{ALN}.tif', mode='r')
    _G['ph'] = openv(
        f'{B}/data/phases_scan{scan:02d}{ALN}.tif', mode='r')
    _G['crk'] = openv(
        f'{B}/results_voidcrack/crack_scan{scan:02d}.tif', mode='r')
    _G['core'] = openv(f'{OUT}/core_scan{scan:02d}.tif', mode='r')
    # Not every volume of a scan has the same number of slices: on some
    # specimens the crack/sample stacks are a few slices shorter than the
    # labels.  Crop to the common depth, or the boolean combinations below
    # fail with a broadcast error.
    _G['sh'] = tuple(min(v.shape[k] for v in
                         (_G['lab'], _G['ph'], _G['crk'], _G['core']))
                     for k in range(3))


def _one(job):
    ia, ib, bba, bbb, ra, rb, ca, cb = job
    lab, ph, crk, core = _G['lab'], _G['ph'], _G['crk'], _G['core']
    sh = _G['sh']
    lo = [max(0, min(bba[k][0], bbb[k][0]) - MARGIN) for k in range(3)]
    hi = [min(sh[k], max(bba[k][1], bbb[k][1]) + MARGIN) for k in range(3)]
    sl = tuple(slice(lo[k], hi[k]) for k in range(3))

    labc = np.asarray(lab[sl])
    mA, mB = labc == ia, labc == ib
    if not mA.any() or not mB.any():
        return None
    beads = labc > 0
    phc = np.asarray(ph[sl])
    inside = phc > 0                      # phase 0 is outside the specimen
    free = inside & ~beads
    if not free.any():
        return None

    dA = ndi.distance_transform_edt(~mA)
    dB = ndi.distance_transform_edt(~mB)
    # nearest bead label, so a third bead cannot donate voxels to this throat
    _, idx = ndi.distance_transform_edt(~beads, return_indices=True)
    near = labc[idx[0], idx[1], idx[2]]

    ssum = dA + dB
    g = float(ssum[free].min())            # true surface-to-surface gap
    rmean = 0.5 * (ra + rb)
    rth = FOOT * rmean
    slack = rth * rth / (rmean + 0.5 * g)  # lens footprint radius = rth
    lens = free & (ssum <= g + slack) & ((near == ia) | (near == ib))
    n_lens = int(lens.sum())
    if n_lens < 50:
        return None

    q = np.where(lens, np.minimum(dA, dB) / np.maximum(ssum, 1e-6), np.nan)
    crack = lens & (np.asarray(crk[sl]) > 0) & (np.asarray(core[sl]) > 0)
    binder = lens & (phc == PH_BINDER)
    # 94% of the interior crack is labelled phase 1 by the phases map, so the
    # pore class has to have the crack removed from it or "crack sits where
    # pore sits" is true by construction rather than by measurement
    pore = lens & (phc == PH_PORE) & ~crack

    def hq(m):
        return (np.histogram(q[m], bins=QBINS)[0] if m.any()
                else np.zeros(len(QBINS) - 1, dtype=np.int64))

    def med(m):
        return float(np.median(q[m])) if m.any() else np.nan

    # ---- connected or detached: a path A -> B through bead or binder -------
    solid = mA | mB | binder
    cc, _ = ndi.label(solid, structure=np.ones((3, 3, 3)))
    la = np.bincount(cc[mA].ravel()).argmax()
    lbb = np.bincount(cc[mB].ravel()).argmax()
    connected = bool(la == lbb and la != 0)

    # ---- the neck: one voxel layer on the plane through the narrowest point
    #
    # NOT the minimum over axial slabs of the lens.  The lens tapers to a rim,
    # so its thinnest slab is that rim and the "bond radius" that comes out is
    # ~1 voxel for every contact, which is an artefact of the lens shape and
    # not a measurement of the bond.  The neck is the cross-section on the
    # plane through the narrowest free point, perpendicular to the centre line,
    # and it is deliberately NOT capped at the lens footprint: the bond is
    # however wide it is.
    u = np.asarray(cb) - np.asarray(ca)
    u = u / max(float(np.linalg.norm(u)), 1e-9)
    fz, fy, fx = np.nonzero(free)
    k0 = int(np.argmin(ssum[fz, fy, fx]))
    p0 = np.array([fz[k0], fy[k0], fx[k0]], dtype=np.float64)
    gz, gy, gx = np.ogrid[0:labc.shape[0], 0:labc.shape[1], 0:labc.shape[2]]
    off = ((gz - p0[0]) * u[0] + (gy - p0[1]) * u[1] + (gx - p0[2]) * u[2])
    plane = np.abs(off) <= 0.5
    terr = plane & ((near == ia) | (near == ib))
    neck_binder = terr & free & (phc == PH_BINDER)
    neck_crack = terr & free & (np.asarray(crk[sl]) > 0) & (
        np.asarray(core[sl]) > 0)
    r_bond = np.nan
    a_min = np.nan
    neck_sev = np.nan
    if neck_binder.any():
        nl, _ = ndi.label(neck_binder, structure=np.ones((3, 3, 3)))
        kz, ky, kx = np.nonzero(neck_binder)
        d0 = ((kz - p0[0]) ** 2 + (ky - p0[1]) ** 2 + (kx - p0[2]) ** 2)
        lab0 = nl[kz[d0.argmin()], ky[d0.argmin()], kx[d0.argmin()]]
        a_min = float((nl == lab0).sum())
        r_bond = float(np.sqrt(a_min / np.pi))
    n_nb = int(neck_binder.sum())
    n_nc = int(neck_crack.sum())
    if n_nb + n_nc > 0:
        neck_sev = n_nc / float(n_nb + n_nc)

    dip = float(np.degrees(np.arccos(abs(u[0]))))

    return dict(
        row=dict(
            bead_a=int(ia), bead_b=int(ib),
            r_a_vox=float(ra), r_b_vox=float(rb),
            # dA+dB is 2 when exactly one free voxel separates the surfaces,
            # so the free gap is g - 1
            gap_vox=max(g - 1.0, 0.0), gap_um=max(g - 1.0, 0.0) * UM,
            dsum_min=g,
            z_mid=float(0.5 * (ca[0] + cb[0])),
            axis_dip_deg=dip,
            lens_vox=n_lens, lens_mm3=n_lens * MM3,
            binder_vox=int(binder.sum()), pore_vox=int(pore.sum()),
            crack_vox=int(crack.sum()),
            binder_frac=float(binder.sum()) / n_lens,
            pore_frac=float(pore.sum()) / n_lens,
            crack_frac=float(crack.sum()) / n_lens,
            connected=connected,
            bond_area_vox=a_min, r_bond_vox=r_bond,
            r_bond_over_R=(r_bond / rmean) if r_bond == r_bond else np.nan,
            neck_binder_vox=n_nb, neck_crack_vox=n_nc,
            neck_severed=neck_sev,
            q_crack_med=med(crack), q_binder_med=med(binder),
            q_pore_med=med(pore),
            # paired, within one throat, both measured in the SAME deformed
            # frame: where the crack is relative to the binder still there.
            # Positive = crack is further from the beads than the surviving
            # binder, i.e. the binder stayed stuck to the beads and the
            # middle broke.  Negative = the crack is against a bead surface
            # and the binder came away whole.
            dq_paired=med(crack) - med(binder)),
        h_crack=hq(crack), h_binder=hq(binder), h_pore=hq(pore))


def pairs_for(scan):
    d = np.load(f'{OUT}/centroids_scan{scan:02d}.npz')
    com, req, ids = d['com'], d['req'], d['ids']
    tree = cKDTree(com)
    pr = np.array(sorted(tree.query_pairs(2 * float(req.max()) + GAP_MAX + 2)))
    gap = (np.linalg.norm(com[pr[:, 0]] - com[pr[:, 1]], axis=1)
           - req[pr[:, 0]] - req[pr[:, 1]])
    pr = pr[gap <= GAP_MAX]
    return ids, com, req, pr


def build_core(scan):
    """crack minus the 20-voxel specimen skin, written once and memmapped."""
    p = f'{OUT}/core_scan{scan:02d}.tif'
    if os.path.exists(p):
        return
    smp = tifffile.imread(
        f'{B}/results_voidcrack/sample_scan{scan:02d}.tif') > 0
    core = ndi.binary_erosion(smp, iterations=SKIN,
                              border_value=0).astype(np.uint8)
    tifffile.imwrite(p, core)
    del smp, core


def run(scan):
    build_core(scan)
    ids, com, req, pr = pairs_for(scan)
    lab = tifffile.imread(f'{B}/data/bead_labels_scan{scan:02d}{ALN}.tif')
    objs = ndi.find_objects(lab)
    del lab
    jobs = []
    for i, j in pr:
        ia, ib = int(ids[i]), int(ids[j])
        sa, sb = objs[ia - 1], objs[ib - 1]
        if sa is None or sb is None:
            continue
        jobs.append((ia, ib,
                     [(s.start, s.stop) for s in sa],
                     [(s.start, s.stop) for s in sb],
                     float(req[i]), float(req[j]),
                     tuple(com[i]), tuple(com[j])))
    print(f'scan {scan}: {len(jobs)} throats to measure', flush=True)

    rows = []
    hc = np.zeros(len(QBINS) - 1, dtype=np.int64)
    hb = np.zeros(len(QBINS) - 1, dtype=np.int64)
    hp = np.zeros(len(QBINS) - 1, dtype=np.int64)
    with Pool(NPROC, initializer=_init, initargs=(scan,)) as pool:
        for k, r in enumerate(pool.imap_unordered(_one, jobs, chunksize=4), 1):
            if r is not None:
                rows.append(r['row'])
                hc += r['h_crack']
                hb += r['h_binder']
                hp += r['h_pore']
            if k % 200 == 0:
                print(f'   {k}/{len(jobs)}', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/{TAG}_throats_scan{scan:02d}.csv', index=False)
    qc = 0.5 * (QBINS[:-1] + QBINS[1:])
    pd.DataFrame(dict(q=qc, crack=hc, binder=hb, pore=hp)).to_csv(
        f'{OUT}/{TAG}_qhist_scan{scan:02d}.csv', index=False)

    print(f'\n{"="*74}\nSCAN {scan}   {len(df)} throats measured\n{"="*74}')
    print(f'  free gap   median {df.gap_vox.median():5.2f} vox '
          f'({df.gap_vox.median()*UM:6.1f} um)')
    print(f'  connected  {int(df.connected.sum())}/{len(df)} '
          f'= {100*df.connected.mean():.1f}%   '
          f'(Roy 2023 calls this an ACTIVE bond)')
    ok = df[df.connected & df.r_bond_vox.notna()]
    if len(ok):
        print(f'  r_bond     median {ok.r_bond_vox.median():5.2f} vox '
              f'({ok.r_bond_vox.median()*UM:6.1f} um)   '
              f'r_bond/R median {ok.r_bond_over_R.median():.3f}')
    sev = df[df.neck_severed.notna()]
    if len(sev):
        print(f'  neck severed by crack: median {sev.neck_severed.median():.3f}'
              f'   mean {sev.neck_severed.mean():.3f}'
              f'   fully cut (>0.9): {int((sev.neck_severed>0.9).sum())}')
    print(f'  crack fills {100*df.crack_frac.mean():5.2f}% of throat volume '
          f'(mean over throats)')
    print(f'  throats with any crack: {int((df.crack_vox>0).sum())} '
          f'({100*(df.crack_vox>0).mean():.1f}%)')

    ct, bt, pt = hc.sum(), hb.sum(), hp.sum()
    print('\n  q = min(dA,dB)/(dA+dB):  0 = on a bead surface, 0.5 = mid-bond')
    print(f'  {"q":>5s} {"crack%":>7s} {"binder%":>8s} {"pore%":>7s} '
          f'{"crack/binder":>13s} {"crack/pore":>11s}')
    for i in range(len(qc)):
        c = 100 * hc[i] / max(ct, 1)
        b_ = 100 * hb[i] / max(bt, 1)
        p_ = 100 * hp[i] / max(pt, 1)
        rb_ = c / b_ if b_ > 0 else np.nan
        rp_ = c / p_ if p_ > 0 else np.nan
        print(f'  {qc[i]:5.3f} {c:7.2f} {b_:8.2f} {p_:7.2f} '
              f'{rb_:13.2f} {rp_:11.2f}')
    for nm, h in (('crack', hc), ('binder', hb), ('pore', hp)):
        if h.sum():
            print(f'  mean q {nm:7s} {float((h*qc).sum()/h.sum()):.4f}')
    print(f'\n  -> {TAG}_throats_scan{scan:02d}.csv, {TAG}_qhist_scan{scan:02d}.csv',
          flush=True)
    return df


if __name__ == '__main__':
    for s in ([int(a) for a in sys.argv[1:]] or [FIRST, LAST]):
        run(s)
