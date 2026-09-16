"""Tortuosity of the ICE network, every scan, PuMA continuum diffusion.

WHY THE ICE AND NOT THE AIR.  Section 2.5.9 asks for the tortuosity of the void
network.  Measured, that quantity does not exist in most of these specimens.
The air phase is disconnected or barely connected: across 30 scans the solver
returned effective diffusivities of order 1e-5, and NEGATIVE values on four
specimens (G4_2 -9.7e-4, G5_2 -3.7e-6, G3_4 -4.1e-4, A3_2 -5.2e-5).  A negative
diffusivity is unphysical, so those are non-convergence, and where it is
positive but 1e-5 the resulting tau of 1e4-1e5 says that nothing flows rather
than describing a path.  Only the most damaged glass cube, G1 stage 3 at a
local porosity of 0.32, gave a defensible air tortuosity (2.2).

The ice phase is the opposite case: about 40% of the specimen, continuously
connected, and the phase whose connectivity governs the mechanics of a bonded
packing.  On G1 stage 1 -- where the air does not percolate at all -- the ice
solves cleanly at tau = 1.59 with Deff = 3.0e-3.  The earlier glass pipeline on
E: reported tau_ice beside tau_air for the same reason.

So this reports tau_ice for all thirty scans, and the air result is kept as the
negative it is: a void network that does not conduct.

Usage (in the WSL puma env):  python _micro_ice.py [n_cubes]
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pumapy as puma

ICE = (2, 2)
CUBE = 128
TOL = 1e-6
MAXITER = 200000


def sample(ph, n):
    has = (ph > 0).sum(axis=(1, 2))
    k = np.nonzero(has > 0.25 * has.max())[0]
    ph = ph[k[0]:k[-1] + 1]
    nz, ny, nx = ph.shape
    ys, xs = np.nonzero(ph[nz // 2] > 0)
    cy, cx = int(ys.mean()), int(xs.mean())
    c = min(CUBE, nz - 2)
    for z in np.linspace(0.28 * nz, 0.72 * nz, n).astype(int):
        sub = ph[z - c // 2:z + c // 2, cy - c // 2:cy + c // 2,
                 cx - c // 2:cx + c // 2]
        if sub.shape == (c, c, c) and (sub > 0).mean() > 0.98:
            yield sub


def main(n=3):
    os.makedirs('micro/ice', exist_ok=True)
    for src in sorted(glob.glob('micro/bin2/*.npz')):
        tag = os.path.basename(src)[:-4]
        dst = f'micro/ice/{tag}.json'
        if os.path.exists(dst):
            print(f'{tag}: done', flush=True)
            continue
        d = np.load(src)
        t0, rows = time.time(), []
        for sub in sample(d['phase'], n):
            ws = puma.Workspace.from_array(np.ascontiguousarray(
                sub.astype(np.uint16)))
            ws.voxel_length = float(d['vox_mm']) / 1000.0
            try:
                tau, deff, poro, _ = puma.compute_continuum_tortuosity(
                    ws, ICE, direction='z', side_bc='p', tolerance=TOL,
                    maxiter=MAXITER, display_iter=False)
                t = float(np.ravel(tau)[2])
                de = float(np.ravel(deff)[0])
                rows.append(dict(phi_ice=float(poro), tau=t, Deff=de,
                                 ok=bool(np.isfinite(t) and t >= 1.0
                                         and de > 0)))
            except Exception as e:
                rows.append(dict(error=str(e)[:120], ok=False))
        ok = [r for r in rows if r.get('ok')]
        res = dict(tag=tag, n_cubes=len(rows), n_ok=len(ok), cubes=rows)
        if ok:
            res['tau_ice'] = float(np.mean([r['tau'] for r in ok]))
            res['tau_ice_sd'] = float(np.std([r['tau'] for r in ok]))
            res['phi_ice_cube'] = float(np.mean([r['phi_ice'] for r in ok]))
        json.dump(res, open(dst, 'w'), indent=1)
        print(f'{tag}: tau_ice={res.get("tau_ice")} '
              f'({len(ok)}/{len(rows)} cubes, {time.time() - t0:.0f}s)',
              flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
