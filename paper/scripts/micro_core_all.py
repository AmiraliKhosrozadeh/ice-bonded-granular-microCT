"""Microstructure properties on a FIXED core region, at full resolution.

Why this exists.  The per-scan specimen mask is not a fixed region.  Its
98th-percentile radius sits at 5.07-6.03 mm on the first scan of each specimen
and runs out to about 6.40 mm on the later ones in almost every specimen,
which is the tube bore, while the correlation measures the outermost grains
moving only ~0.2 mm outward.  The mask is therefore picking up detached
material and the wall, and any fraction normalised by it mixes a change in the
packing with a change in what the mask enclosed.

The region here is fixed once per specimen, from its first scan:

    height    the middle 60 % of the common z range
    radius    0.85 x the 98th-percentile grain radius of the first scan,
              re-centred per scan on the grain centroid so that a lateral
              shift of the column does not move the region through the
              material

Definitions are otherwise those of _micro_props.py: fractions of the interior
(air + ice + grain), and S_V as the air|solid voxel-face area per interior
volume, counted only between voxel pairs that are both inside the region so
that the cut surface of the region is not counted as interface.

Output: micro/props_core.csv
"""
import csv
import os

import numpy as np
import tifffile

SC = ('C:/Users/cak7496/AppData/Local/Temp/claude/D--wsl/'
      'ce923714-58c7-48fb-8bce-70f1eccee47f/scratchpad/micro')
ALU = 'E:/RPTU-images/CT_images/Alumina/pyalumina/results'
GLA = 'E:/RPTU-images/CT_images/Glass/pyalumina_results'
VOX_MM = 24.7660229 / 1000.0
HBAND = (0.20, 0.80)
RFRAC = 0.85

SPECS = [
    ('A1', 'alumina', '100_1800_T5', 3), ('A2', 'alumina', '175_1800_T5', 2),
    ('A3', 'alumina', '75_1000_T5', 2), ('A4', 'alumina', '75_1800_T7', 3),
    ('G1', 'glass', '75_1700_T5_HR', 3), ('G2', 'glass', '75_1000_T6', 2),
    ('G3', 'glass', '100_1700_T7', 4), ('G4', 'glass', '100_1700_T5_HR', 2),
    ('G5', 'glass', '100_1800_T5', 2),
    ('S1', 'sand', '100_500_T5', 2), ('S2', 'sand', '25mm_100_500', 3),
    ('S3', 'sand', '75_200_T5', 2),
]


def candidates(mat, spec, n):
    tail = '%s/scan%02d/stage1/phase_labels.tif' % (spec, n)
    if mat == 'alumina':
        return ['%s/stage1/%s' % (SC, tail), '%s/%s' % (ALU, tail)]
    if mat == 'glass':
        return ['%s/stage1/glass/%s' % (SC, tail), '%s/%s' % (GLA, tail)]
    return ['%s/stage1/_xmat/sand/%s' % (SC, tail),
            '%s/_xmat/sand/%s' % (ALU, tail)]


def geom_from_bin2(pid, stage):
    """(z0, z1, cy, cx, R) in FULL-resolution voxels."""
    z = np.load('%s/bin2/%s_%d.npz' % (SC, pid, stage))
    ph = z['phase']
    inside = ph > 0
    occ = np.where(inside.any(axis=(1, 2)))[0]
    lo, hi = int(occ[0]), int(occ[-1])
    h = hi - lo
    band = slice(lo + int(HBAND[0] * h), lo + int(HBAND[1] * h) + 1)
    gr = ph[band] == 3
    if not gr.any():
        gr = ph[band] >= 2
    _, gy, gx = np.nonzero(gr)
    cy, cx = gy.mean(), gx.mean()
    R = float(np.quantile(np.hypot(gy - cy, gx - cx), 0.98))
    return (2 * band.start, 2 * band.stop - 1, 2 * cy, 2 * cx, 2 * R)


def props_core(path, z0, z1, cy, cx, r_vox):
    n_air = n_ice = n_gr = 0
    faces = 0
    prev = None
    disc = None
    with tifffile.TiffFile(path) as tf:
        pages = tf.pages
        for i in range(z0, min(z1 + 1, len(pages))):
            L = pages[i].asarray()
            if disc is None:
                Y, X = np.ogrid[:L.shape[0], :L.shape[1]]
                disc = np.hypot(Y - cy, X - cx) <= r_vox
            air = (L == 1) & disc
            ice = (L == 2) & disc
            gr = (L == 3) & disc
            sol = ice | gr
            n_air += int(air.sum())
            n_ice += int(ice.sum())
            n_gr += int(gr.sum())
            faces += int((air[:, :-1] & sol[:, 1:]).sum()
                         + (sol[:, :-1] & air[:, 1:]).sum()
                         + (air[:-1] & sol[1:]).sum()
                         + (sol[:-1] & air[1:]).sum())
            if prev is not None:
                faces += int((prev[0] & sol).sum() + (prev[1] & air).sum())
            prev = (air, sol)
    tot = n_air + n_ice + n_gr
    return dict(interior=tot, air=n_air, ice=n_ice, grain=n_gr,
                eps=n_air / tot, phi_ice=n_ice / tot, rho=n_gr / tot,
                SV_per_mm=faces * VOX_MM ** 2 / (tot * VOX_MM ** 3))


def kc(eps, sv_per_mm, c=5.0):
    """Carman form: S_V per unit total volume takes the Kozeny constant 5, not
    the 180 of the grain-diameter form.  Only the ratio k/k_unloaded is used
    in the paper, so the constant cancels there."""
    return eps ** 3 / (c * (sv_per_mm * 1000.0) ** 2)


def main():
    rows = []
    for pid, mat, spec, ns in SPECS:
        _, _, _, _, R0 = geom_from_bin2(pid, 1)
        for n in range(1, ns + 1):
            path = next((p for p in candidates(mat, spec, n)
                         if os.path.exists(p)), None)
            if path is None:
                print('%s stage %d: MISSING' % (pid, n), flush=True)
                continue
            z0, z1, cy, cx, _ = geom_from_bin2(pid, n)
            r = props_core(path, z0, z1, cy, cx, RFRAC * R0)
            two = mat == 'sand'
            rows.append(dict(
                id=pid, material=mat, stage=n,
                core_mm3=r['interior'] * VOX_MM ** 3,
                eps=r['eps'],
                phi_ice='' if two else r['phi_ice'],
                rho='' if two else r['rho'],
                SV_per_mm=r['SV_per_mm'],
                k_KC_m2=kc(r['eps'], r['SV_per_mm'])))
            print('%s %d: eps=%.4f  SV=%.3f/mm  core=%.0f mm3'
                  % (pid, n, r['eps'], r['SV_per_mm'],
                     r['interior'] * VOX_MM ** 3), flush=True)
    with open('%s/props_core.csv' % SC, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('->', '%s/props_core.csv' % SC, len(rows), 'rows')


if __name__ == '__main__':
    main()
