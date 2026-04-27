"""Grid LDIC for T5_HR transition 1 -> 2 (spam-ldic CLI).

Outputs:
    ~/spam-results-75/75_ldic_12-ldic.tsv  (+ .vtk)
"""
import os, subprocess, time

DATA = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
RES  = os.path.expanduser('~/spam-results-75')
os.makedirs(RES, exist_ok=True)

TRANSITIONS = [(1, 2), (2, 3)]
NS = 30; HWS = 20; NP_ = 16; IT = 50; MARGIN = 15

for a, b in TRANSITIONS:
    prefix = f'75_ldic_{a}{b}'
    phi_seed = os.path.join(RES, f'phi_seed_{a}{b}.tsv')
    if not os.path.exists(phi_seed):
        print(f'skip {a}->{b}: no phi_seed (run spam_ddic.py first)'); continue
    cmd = [
        'spam-ldic',
        os.path.join(DATA, f'ct_scan{a:02d}_aligned.tif'),
        os.path.join(DATA, f'ct_scan{b:02d}_aligned.tif'),
        '-mf1', os.path.join(DATA, f'specimen_mask_scan{a:02d}_aligned.tif'),
        '-pf', phi_seed,
        '-ns', str(NS), '-hws', str(HWS), '-np', str(NP_),
        '-it', str(IT), '-m', str(MARGIN),
        '-od', RES, '-pre', prefix,
    ]
    log = os.path.join(RES, f'{prefix}.log')
    print(f'\n=== LDIC {a}->{b} ===')
    print(' '.join(cmd))
    t0 = time.time()
    with open(log, 'w') as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=True)
    print(f'  done in {time.time()-t0:.0f}s  ->  {log}')

print('\nLDIC complete.')
