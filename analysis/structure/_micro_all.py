"""Phase fractions and specific surface for all twelve specimens, one table.

Reads whichever phase_labels.tif exists for each (specimen, stage): the ones
already on E: from the earlier pyalumina runs, and the eight this session
segmented into the scratchpad because the cross-material registry never listed
them.  Both come from the same stage1_phase_seg.py, which is the point.

Nothing on E: is written.  Output: micro/props.csv and micro/bin2/<ID>_<n>.npz
"""
import os
import csv
import numpy as np

import _micro_props as P

SC = ('C:/Users/cak7496/AppData/Local/Temp/claude/D--wsl/'
      'ce923714-58c7-48fb-8bce-70f1eccee47f/scratchpad/micro')
ALU = 'E:/RPTU-images/CT_images/Alumina/pyalumina/results'
GLA = 'E:/RPTU-images/CT_images/Glass/pyalumina_results'

# paper ID -> (material, pipeline spec tag, number of load stages)
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
    """New scratchpad run first, then the pre-existing run on E:."""
    tail = f'{spec}/scan{n:02d}/stage1/phase_labels.tif'
    if mat == 'alumina':
        return [f'{SC}/stage1/{tail}', f'{ALU}/{tail}']
    if mat == 'glass':
        return [f'{SC}/stage1/glass/{tail}', f'{GLA}/{tail}']
    return [f'{SC}/stage1/_xmat/sand/{tail}', f'{ALU}/_xmat/sand/{tail}']


def main():
    os.makedirs(f'{SC}/bin2', exist_ok=True)
    rows = []
    for pid, mat, spec, ns in SPECS:
        for n in range(1, ns + 1):
            path = next((p for p in candidates(mat, spec, n) if os.path.exists(p)),
                        None)
            if path is None:
                print(f'{pid} stage {n}: MISSING', flush=True)
                continue
            r = P.props(path, bin_out=f'{SC}/bin2/{pid}_{n}.npz')
            src = 'new' if path.startswith(SC) else 'existing'
            two = mat == 'sand'
            rows.append(dict(
                id=pid, material=mat, stage=n, source=src,
                interior_mm3=r['interior'] * P.VOX_MM ** 3,
                eps=r['eps'],
                phi_ice='' if two else r['phi_ice'],
                rho='' if two else r['rho'],
                SV_per_mm=r['SV_per_mm'],
                k_KC_m2=P.kozeny_carman(r['eps'], r['SV_per_mm'])))
            print(f'{pid} stage {n} [{src}]: eps={r["eps"]:.4f} '
                  f'SV={r["SV_per_mm"]:.3f}/mm', flush=True)
    with open(f'{SC}/props.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('->', f'{SC}/props.csv', len(rows), 'rows')


if __name__ == '__main__':
    main()
