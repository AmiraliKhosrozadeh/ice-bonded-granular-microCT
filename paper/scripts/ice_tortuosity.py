"""Ice-network tortuosity of every scan by pumapy's continuum finite-volume
diffusion solver, with the per-cube record kept.

Re-run of the session script _micro_ice.py with one correction: that script
accepted a cube on ``Deff[0] > 0``, which is the x-component of the flux
returned for a z-direction solve, an off-diagonal term that may legitimately
be negative.  Acceptance is now on the axial component, ``Deff[2] > 0`` with
``tau_z`` finite and >= 1, and all three components are stored so the choice
can be audited.  Everything else is as before:

    phase maps      2x-binned (49.5 um voxel), ice = phase 2
    cubes           128^3 voxels (6.34 mm), centred on the column axis, at
                    three heights between 0.28 and 0.72 of the column,
                    attempted only if >= 98 % of the cube is inside the specimen
    solver          compute_continuum_tortuosity(ws, (2, 2), direction='z',
                    side_bc='p', tolerance=1e-6, maxiter=200000), i.e. the
                    pumapy default conjugate-gradient matrix-free solve with
                    c = 0 / 1 prescribed on the z faces and periodic sides
    tortuosity      tau_z = phi_ice(cube) / Deff_z, D0 = 1

Run in the WSL puma environment from the paper folder:

    ~/mamba/envs/puma/bin/python scripts/ice_tortuosity.py <bin2 dir> <out dir>

Writes one JSON per scan and ice_tortuosity.csv with the mean over accepted
cubes, its sd, and the counts.
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
        sub = ph[z - c // 2:z + c // 2, cy - c // 2:cy + c // 2, cx - c // 2:cx + c // 2]
        if sub.shape == (c, c, c) and (sub > 0).mean() > 0.98:
            yield float(z / nz), sub


def main(src_dir, out_dir, n=3):
    os.makedirs(out_dir, exist_ok=True)
    rows_csv = []
    for src in sorted(glob.glob(os.path.join(src_dir, "*.npz"))):
        tag = os.path.basename(src)[:-4]
        dst = os.path.join(out_dir, f"{tag}.json")
        if os.path.exists(dst):
            res = json.load(open(dst))
        else:
            d = np.load(src)
            t0, rows = time.time(), []
            for zfrac, sub in sample(d["phase"], n):
                ws = puma.Workspace.from_array(np.ascontiguousarray(sub.astype(np.uint16)))
                ws.voxel_length = float(d["vox_mm"]) / 1000.0
                try:
                    tau, deff, poro, _ = puma.compute_continuum_tortuosity(
                        ws, ICE, direction="z", side_bc="p", tolerance=TOL,
                        maxiter=MAXITER, display_iter=False)
                    tau = [float(v) for v in np.ravel(tau)]
                    deff = [float(v) for v in np.ravel(deff)]
                    ok = bool(np.isfinite(tau[2]) and tau[2] >= 1.0 and deff[2] > 0)
                    rows.append(dict(z_frac=zfrac, phi_ice=float(poro), tau_z=tau[2],
                                     Deff=deff, tau=tau, ok=ok))
                except Exception as e:
                    rows.append(dict(z_frac=zfrac, error=str(e)[:120], ok=False))
            ok = [r for r in rows if r.get("ok")]
            res = dict(tag=tag, cube_vox=CUBE, n_cubes=len(rows), n_ok=len(ok), cubes=rows,
                       seconds=round(time.time() - t0))
            if ok:
                res["tau_ice"] = float(np.mean([r["tau_z"] for r in ok]))
                res["tau_ice_sd"] = float(np.std([r["tau_z"] for r in ok]))
                res["phi_ice_cube"] = float(np.mean([r["phi_ice"] for r in ok]))
            json.dump(res, open(dst, "w"), indent=1)
        print(f'{tag}: tau_ice={res.get("tau_ice")} ({res["n_ok"]}/{res["n_cubes"]} cubes)', flush=True)
        pid, stage = tag.split("_")
        rows_csv.append(dict(id=pid, stage=int(stage), tau_ice=res.get("tau_ice", ""),
                             tau_ice_sd=res.get("tau_ice_sd", ""), phi_ice_cube=res.get("phi_ice_cube", ""),
                             n_cubes_ok=res["n_ok"], n_cubes=res["n_cubes"]))
    import csv
    with open(os.path.join(out_dir, "ice_tortuosity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
