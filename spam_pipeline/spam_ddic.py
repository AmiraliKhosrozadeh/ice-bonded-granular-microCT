"""Run spam-ddic for T5_HR transition 1 -> 2.

Uses the spam-ddic CLI (matches the T7 pipeline). A per-transition Phi seed
is written first (axial-compression Fzz from the specimen Z-extent ratio).

Output:
    ~/spam-results-75/75_ddic_12-ddic.tsv  (+ .vtk)
"""
import os, subprocess, numpy as np, tifffile

DATA_DIR = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
RES_DIR  = os.path.expanduser('~/spam-results-75')
os.makedirs(RES_DIR, exist_ok=True)

TRANSITIONS = [(1, 2), (2, 3)]

def z_extent_of_mask(path):
    m = tifffile.imread(path)
    nz_has = np.any(m > 0, axis=(1, 2))
    if not nz_has.any():
        return None, None
    idx = np.where(nz_has)[0]
    return int(idx[0]), int(idx[-1])

def write_phi_seed(path, Fzz, Zdisp, Zpos, Ypos, Xpos):
    header = ('NodeNumber\tZpos\tYpos\tXpos\tFzz\tFzy\tFzx\tZdisp\t'
              'Fyz\tFyy\tFyx\tYdisp\tFxz\tFxy\tFxx\tXdisp\t'
              'error\titerations\treturnStatus\tdeltaPhiNorm')
    row = (f'1\t{Zpos:.7f}\t{Ypos:.7f}\t{Xpos:.7f}\t{Fzz:.7f}\t0.0\t0.0\t{Zdisp:.7f}\t'
           f'0.0\t1.0\t0.0\t0.0\t0.0\t0.0\t1.0\t0.0\t0.0\t0.0\t2.0\t0.0')
    with open(path, 'w') as f:
        f.write(header + '\n' + row + '\n')

def run_ddic(im1, lab1, im2, phi_seed, prefix):
    cmd = [
        'spam-ddic',
        os.path.join(DATA_DIR, im1),
        os.path.join(DATA_DIR, lab1),
        os.path.join(DATA_DIR, im2),
        '-pf', phi_seed,
        '-F', 'all',
        '-np', '16',
        '-m', '15',
        '-it', '50',
        '-ld', '2',
        '-od', RES_DIR,
        '-pre', prefix,
    ]
    log = os.path.join(RES_DIR, f'{prefix}.log')
    print(f'=== {prefix} ===')
    print(' '.join(cmd))
    with open(log, 'w') as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=True)
    print(f'  log -> {log}')

for a, b in TRANSITIONS:
    # Phi seed from specimen Z-extent ratio
    z_a_lo, z_a_hi = z_extent_of_mask(os.path.join(DATA_DIR, f'specimen_mask_scan{a:02d}_aligned.tif'))
    z_b_lo, z_b_hi = z_extent_of_mask(os.path.join(DATA_DIR, f'specimen_mask_scan{b:02d}_aligned.tif'))
    h_a = z_a_hi - z_a_lo + 1
    h_b = z_b_hi - z_b_lo + 1
    Fzz = h_b / h_a
    # Node at volume centre (aligned frame shape = 1007 x 707 x 803)
    Zpos = 1240 / 2.0
    Ypos = 706 / 2.0
    Xpos = 802 / 2.0
    Zdisp = Zpos * (Fzz - 1.0)
    print(f'\nTransition {a}->{b}:  h_a={h_a}  h_b={h_b}  Fzz={Fzz:.4f}  Zdisp={Zdisp:.2f}')
    phi_seed = os.path.join(RES_DIR, f'phi_seed_{a}{b}.tsv')
    write_phi_seed(phi_seed, Fzz, Zdisp, Zpos, Ypos, Xpos)
    run_ddic(
        im1=f'ct_scan{a:02d}_aligned.tif',
        lab1=f'bead_labels_scan{a:02d}_aligned.tif',
        im2=f'ct_scan{b:02d}_aligned.tif',
        phi_seed=phi_seed,
        prefix=f'75_ddic_{a}{b}',
    )
    print(f'Transition {a}->{b} done. Output: {RES_DIR}/75_ddic_{a}{b}-ddic.tsv')

print('\nDDIC complete.')
