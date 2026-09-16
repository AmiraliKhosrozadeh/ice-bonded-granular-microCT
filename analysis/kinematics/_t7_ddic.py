"""Re-run the Alumina_75_1800_T7 1->2 DIC with the settings that worked on Glass.

WHY THIS SPECIMEN, AND WHY ONE TRANSITION
T7's two scans already match in grey scale (grain peak 6013 -> 6452, 7% apart),
so normalisation is nearly irrelevant here. That isolates the OTHER four
differences between the Alumina runs and the Glass run that converged:

                     Glass_75 2->3 (233/342)     Alumina T7 1->2 (47/472)
    MAX_ITERATIONS         100                          50
    UPDATE_GRADIENT        True                         False
    MULTISCALE_BINNING     2                            1
    seed translation       -58.6 voxels                 [0, 0, 0]

The zero seed with no multiscale is the suspect: T7 beads move ~30 voxels and
have a radius of ~35, so with no initial guess every bead starts a full radius
from its target -- the edge of the convergence basin. That is what "stuck at
deltaPhiNorm 0.4 after 50 iterations" looks like.

The seed is built from BEAD-LABEL CENTROIDS and the specimen mask, exactly as
spam_75_ddic.py does it, so it uses no grey values and cannot be thrown off by
calibration. The mask used is the CORRECTED one written today, not the old
specimen_mask files.

Writes to ~/spam-results-t7 so the existing a4_ddic_12 output is untouched and
the two can be compared.
"""
import os
import subprocess

import numpy as np
import tifffile
from scipy import ndimage as ndi

SPAM = ('/mnt/e/RPTU-images/CT_images/Alumina/Alumina-75-1800-T7/'
        'Alumina_75_1800_T7_spam')
DATA = f'{SPAM}/data'
RD = f'{SPAM}/results_voidcrack'
RES = '/home/amirali_wsl/spam-results-t7'
A, B = 1, 2
LO_PCT, HI_PCT = 5, 95
PHI_HEADER = ('NodeNumber\tZpos\tYpos\tXpos\tFzz\tFzy\tFzx\tZdisp\t'
              'Fyz\tFyy\tFyx\tYdisp\tFxz\tFxy\tFxx\tXdisp\t'
              'error\titerations\treturnStatus\tdeltaPhiNorm')
os.makedirs(RES, exist_ok=True)


def ct(s):
    return f'{DATA}/ct_scan{s:02d}_aligned.tif'


def labels(s):
    return f'{DATA}/bead_labels_scan{s:02d}_aligned.tif'


def bead_centroids(s):
    lab = tifffile.imread(labels(s))
    ids = np.unique(lab)
    ids = ids[ids > 0]
    cent = np.array(ndi.center_of_mass(lab > 0, lab, ids))
    print(f'  scan {s}: {len(cent)} bead labels, frame {lab.shape}', flush=True)
    del lab
    return cent


def mask_radius_and_centre(s):
    """From the CORRECTED mask, over the middle third only."""
    m = tifffile.imread(f'{RD}/sample_scan{s:02d}.tif') > 0
    zs = np.where(m.any(axis=(1, 2)))[0]
    z0, z1 = zs[0], zs[-1]
    radii, ys, xs = [], [], []
    for z in range(z0 + (z1 - z0) // 3, z0 + 2 * (z1 - z0) // 3, 20):
        sl = m[z]
        if sl.sum() < 100:
            continue
        yy, xx = np.nonzero(sl)
        radii.append(np.sqrt(sl.sum() / np.pi))
        ys.append(yy.mean()); xs.append(xx.mean())
    idx = np.array(np.nonzero(m))
    centre = idx.mean(axis=1)
    del m, idx
    return float(np.mean(radii)), float(np.mean(ys)), float(np.mean(xs)), centre


ca, cb = bead_centroids(A), bead_centroids(B)
F = np.eye(3)
t = np.zeros(3)
lo_a, hi_a = np.percentile(ca[:, 0], [LO_PCT, HI_PCT])
lo_b, hi_b = np.percentile(cb[:, 0], [LO_PCT, HI_PCT])
F[0, 0] = (hi_b - lo_b) / (hi_a - lo_a)
t[0] = lo_b - F[0, 0] * lo_a
r_a, y_a, x_a, centre = mask_radius_and_centre(A)
r_b, y_b, x_b, _ = mask_radius_and_centre(B)
lat = r_b / r_a
F[1, 1] = F[2, 2] = lat
t[1] = y_b - lat * y_a
t[2] = x_b - lat * x_a
vr = F[0, 0] * F[1, 1] * F[2, 2]
print(f'\nseed from geometry:')
print(f'  axial stretch Fzz  = {F[0,0]:.4f}   ({100*(F[0,0]-1):+.1f}% axial)')
print(f'  lateral stretch    = {lat:.4f}   (mask radius {r_a:.1f} -> {r_b:.1f})')
print(f'  implied volume     = {vr:.4f} ({100*(vr-1):+.1f}%)', flush=True)
if vr > 1.02:
    print('  WARNING: seed implies volume GAIN under compression', flush=True)

disp = F @ centre + t - centre
z, y, x = centre
row = (f'1\t{z:.7f}\t{y:.7f}\t{x:.7f}\t'
       f'{F[0,0]:.7f}\t{F[0,1]:.7f}\t{F[0,2]:.7f}\t{disp[0]:.7f}\t'
       f'{F[1,0]:.7f}\t{F[1,1]:.7f}\t{F[1,2]:.7f}\t{disp[1]:.7f}\t'
       f'{F[2,0]:.7f}\t{F[2,1]:.7f}\t{F[2,2]:.7f}\t{disp[2]:.7f}\t'
       f'0.0\t0\t2\t0.0')
seed = f'{RES}/phi_seed_{A}{B}.tsv'
open(seed, 'w').write(PHI_HEADER + '\n' + row + '\n')
print(f'  node ZYX = ({z:.0f}, {y:.0f}, {x:.0f})   '
      f'disp at node = ({disp[0]:+.1f}, {disp[1]:+.1f}, {disp[2]:+.1f}) vox')
print(f'  -> {seed}\n', flush=True)

prefix = f't7_ddic_{A}{B}'
cmd = ['spam-ddic', ct(A), labels(A), ct(B),
       '-pf', seed, '-F', 'all', '-np', '8', '-ld', '2', '-m', '15',
       '-it', '100', '-msb', '2', '-ug', '-od', RES, '-pre', prefix]
print('running:', ' '.join(cmd), flush=True)
log = f'{RES}/{prefix}.log'
rc = subprocess.run(cmd, stdout=open(log, 'w'), stderr=subprocess.STDOUT).returncode
print(f'spam-ddic exited {rc}', flush=True)

tsv = f'{RES}/{prefix}-ddic.tsv'
if os.path.exists(tsv):
    hdr = open(tsv).readline().strip().split('\t')
    arr = np.genfromtxt(tsv, delimiter='\t', skip_header=1)
    rs = arr[:, hdr.index('returnStatus')].astype(int)
    dp = arr[:, hdr.index('deltaPhiNorm')]
    it = arr[:, hdr.index('iterations')]
    print(f'\nRESULT  {int((rs==2).sum())}/{len(rs)} converged '
          f'({100*(rs==2).mean():.1f}%)')
    print(f'  deltaPhiNorm median {np.nanmedian(dp):.4f}   '
          f'at iteration cap {100*np.mean(it>=100):.0f}%')
    print(f'  BEFORE was 47/472 converged (10.0%), deltaPhiNorm median 0.40, '
          f'cap 91%', flush=True)
else:
    print('no output tsv produced', flush=True)
