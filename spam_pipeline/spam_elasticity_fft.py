"""FFT-based effective elastic moduli per scan (Moulinec-Suquet 1994).

For each scan, build a 3-phase voxel map (0=air, 1=ice, 2=glass) from the
aligned CT + specimen-mask + bead-labels TIFFs, downsample by DOWNSAMPLE,
and run a Moulinec-Suquet FFT iteration to recover the effective 6x6
stiffness tensor C_eff.  Extract:

    E_x, E_y, E_z   - direction-resolved Young's moduli  (GPa)
    K               - bulk modulus                       (GPa)
    G_yz, G_xz, G_xy- shear moduli                       (GPa)

Output: results_<PRE>/elasticity_fft/summary.csv

Storyline: as cracks form between scans, E_z (and to a lesser extent K, G)
drop monotonically — a complementary signal to the rising SSA_ice / cc_ice.

Phase moduli (isotropic, GPa, dimensionless ν):
    air  : E = 0.1   ν = 0.0   (placeholder; treats voids as ~void)
    ice  : E = 9.3   ν = 0.33
    glass: E = 70.0  ν = 0.22

Notes
-----
* Voigt notation (engineering shear): σ = (σ_xx σ_yy σ_zz σ_yz σ_xz σ_xy);
  ε = (ε_xx ε_yy ε_zz 2ε_yz 2ε_xz 2ε_xy).
* MS basic scheme uses an isotropic reference C0 = volume-fraction-weighted
  arithmetic mean of phase stiffnesses.
* Convergence is monitored by ‖ε_{n+1} - ε_n‖ / ‖ε_n‖ < TOL.
"""
import os, sys, time
import numpy as np
import tifffile
import pandas as pd
from scipy.fft import fftn as sfftn, ifftn as sifftn   # multithreaded

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from scan_meta import VOXEL_UM

DATA = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
OUT  = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>/elasticity_fft'
os.makedirs(OUT, exist_ok=True)

TRANSITIONS = [(1, 2), (2, 3)]
ALL_SCANS   = sorted({s for ab in TRANSITIONS for s in ab})

# Per-scan thresholds (must match microstructure/ice_bonds)
TH = {
    1: dict(air=7000,    ig=18032.7, ga=47275),
    2: dict(air=2139.4,  ig=6271.5,  ga=99999),
    3: dict(air=2015.4,  ig=6539.5,  ga=99999),
}

# Phase elastic constants (E [GPa], ν).
# Note: E_air is artificial — basic Moulinec-Suquet diverges at infinite contrast
# (air → void).  Using 1 GPa keeps glass:air contrast at 70:1 so MS converges
# in ~100 iters.  Air still acts as the softest phase, so qualitative trends
# (E drops as cracks form) are preserved.
PHASE_PROPS = [
    (1.0,  0.30),   # 0 = air (regularised)
    (9.3,  0.33),   # 1 = ice
    (70.0, 0.22),   # 2 = glass
]

DOWNSAMPLE = 8         # bin factor; full volume too big for FFT (~64x memory)
MAX_ITER   = 150
TOL        = 1e-3
DTYPE      = np.float64
DIV_FACTOR = 5.0       # abort case if residual >  best_so_far × DIV_FACTOR

# ---------------------------------------------------------------------------
def iso_C_voigt(E, nu):
    """Isotropic stiffness in Voigt notation (engineering shear)."""
    if E <= 0:
        return np.zeros((6, 6))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * mu
    C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C

def gamma0_apply(sigma_hat, kxg, kyg, kzg, mu0, lam0):
    """Apply -Γ̂⁰ : σ̂  (Khachaturyan/Mura form for isotropic reference).

    sigma_hat shape (Nz, Ny, Nx, 6) complex Voigt stress.
    Returns same-shape Voigt strain.  DC bin returned as zero (caller sets it).
    """
    knorm = np.sqrt(kxg**2 + kyg**2 + kzg**2)
    safe  = np.where(knorm > 0, knorm, 1.0)
    nx, ny, nz = kxg / safe, kyg / safe, kzg / safe

    sxx = sigma_hat[..., 0]; syy = sigma_hat[..., 1]; szz = sigma_hat[..., 2]
    syz = sigma_hat[..., 3]; sxz = sigma_hat[..., 4]; sxy = sigma_hat[..., 5]

    vx = sxx * nx + sxy * ny + sxz * nz
    vy = sxy * nx + syy * ny + syz * nz
    vz = sxz * nx + syz * ny + szz * nz
    nv = nx * vx + ny * vy + nz * vz

    inv2mu = 1.0 / (2.0 * mu0)
    f      = (lam0 + mu0) / (mu0 * (lam0 + 2.0 * mu0))

    e_xx = -inv2mu * (2 * nx * vx) + f * nx * nx * nv
    e_yy = -inv2mu * (2 * ny * vy) + f * ny * ny * nv
    e_zz = -inv2mu * (2 * nz * vz) + f * nz * nz * nv
    e_yz = -inv2mu * (ny * vz + nz * vy) + f * ny * nz * nv
    e_xz = -inv2mu * (nx * vz + nz * vx) + f * nx * nz * nv
    e_xy = -inv2mu * (nx * vy + ny * vx) + f * nx * ny * nv

    eps_hat = np.stack([e_xx, e_yy, e_zz, 2 * e_yz, 2 * e_xz, 2 * e_xy], axis=-1)
    eps_hat[0, 0, 0, :] = 0.0
    return eps_hat

def compute_sigma(eps, phase_map, C_phases):
    """σ = C(x) : ε per voxel, dispatched by phase."""
    sigma = np.zeros_like(eps)
    for p, Cp in enumerate(C_phases):
        m = phase_map == p
        if not m.any(): continue
        sigma[m] = eps[m] @ Cp.T.astype(eps.dtype)
    return sigma

def moulinec_suquet(phase_map, phase_props, max_iter=80, tol=1e-3, verbose=True):
    Nz, Ny, Nx = phase_map.shape
    C_phases = np.array([iso_C_voigt(E, nu) for E, nu in phase_props], dtype=np.float64)
    n = phase_map.size
    vfrac = np.array([(phase_map == p).sum() / n for p in range(len(phase_props))])

    # Reference C0 — for basic MS to converge we need
    #     μ0 ≥ (μ_max + μ_min)/2     (Moulinec-Suquet 1994)
    # Volume-weighted average is too soft when the stiff phase fraction is small.
    # Use the midpoint of phase shear and bulk moduli over phases that are
    # actually present (vfrac > 0).
    mus = []
    Ks  = []
    for p, (E, nu) in enumerate(phase_props):
        if vfrac[p] <= 0: continue
        mus.append(E / (2 * (1 + nu)))
        Ks.append(E / (3 * (1 - 2 * nu)))
    mu0 = (max(mus) + min(mus)) / 2.0
    K0  = (max(Ks)  + min(Ks))  / 2.0
    lam0 = K0 - (2.0 / 3.0) * mu0
    if mu0 < 1e-6:
        raise RuntimeError(f'Reference mu0 = {mu0:.2e} too small; check phase fractions')
    if verbose:
        print(f'    phase μ values (GPa):  '
              f'{[f"{m:.3f}" for m in mus]}')
        print(f'    reference (midpoint): λ0={lam0:.3f} GPa  μ0={mu0:.3f} GPa  '
              f'(vfrac air/ice/glass = {vfrac[0]:.3f}/{vfrac[1]:.3f}/{vfrac[2]:.3f})')

    kz = np.fft.fftfreq(Nz).astype(DTYPE).reshape(Nz, 1, 1)
    ky = np.fft.fftfreq(Ny).astype(DTYPE).reshape(1, Ny, 1)
    kx = np.fft.fftfreq(Nx).astype(DTYPE).reshape(1, 1, Nx)
    # broadcast to full (Nz, Ny, Nx)
    kxg = np.broadcast_to(kx, (Nz, Ny, Nx))
    kyg = np.broadcast_to(ky, (Nz, Ny, Nx))
    kzg = np.broadcast_to(kz, (Nz, Ny, Nx))

    C_eff = np.zeros((6, 6))
    iters_used = []

    for case in range(6):
        E_macro = np.zeros(6, dtype=DTYPE); E_macro[case] = 1.0
        eps = np.broadcast_to(E_macro, (Nz, Ny, Nx, 6)).astype(DTYPE).copy()
        prev_res  = np.inf
        best_res  = np.inf
        best_eps  = eps.copy()
        sigma = None
        t_case = time.time()
        for it in range(max_iter):
            sigma = compute_sigma(eps, phase_map, C_phases)

            # Batch FFT over the 3 spatial axes for all 6 Voigt components.
            sigma_hat = sfftn(sigma, axes=(0, 1, 2), workers=-1)

            # Equilibrium residual: ‖k_j σ̂_ij(k)‖ / ‖σ̂(0)‖ over k ≠ 0
            sxx, syy, szz = sigma_hat[..., 0], sigma_hat[..., 1], sigma_hat[..., 2]
            syz, sxz, sxy = sigma_hat[..., 3], sigma_hat[..., 4], sigma_hat[..., 5]
            div_x = kxg * sxx + kyg * sxy + kzg * sxz
            div_y = kxg * sxy + kyg * syy + kzg * syz
            div_z = kxg * sxz + kyg * syz + kzg * szz
            eq = np.sqrt((np.abs(div_x)**2 + np.abs(div_y)**2 + np.abs(div_z)**2).sum())
            ref = np.linalg.norm(sigma.mean(axis=(0, 1, 2))) + 1e-30
            res = eq / (ref * Nz * Ny * Nx)

            # Strain update: ε̂_new = ε̂_old + (−Γ̂⁰ σ̂)  for k ≠ 0,
            #                ε̂_new(0) = E_macro * N
            eps_hat = sfftn(eps, axes=(0, 1, 2), workers=-1)
            delta_eps_hat = gamma0_apply(sigma_hat, kxg, kyg, kzg, mu0, lam0)
            eps_hat = eps_hat + delta_eps_hat
            eps_hat[0, 0, 0, :] = E_macro * Nz * Ny * Nx

            eps = sifftn(eps_hat, axes=(0, 1, 2), workers=-1).real.astype(DTYPE)

            if res < best_res:
                best_res = res
                best_eps = eps.copy()
            if verbose and (it < 3 or (it + 1) % 10 == 0):
                print(f'    case {case} iter {it+1:3d}: equilib residual = {res:.3e}  (best = {best_res:.3e})')
            if res < tol:
                break
            if not np.isfinite(res) or res > DIV_FACTOR * best_res and it > 10:
                print(f'    case {case} DIVERGED at iter {it+1} (best res = {best_res:.3e}); rolling back')
                eps = best_eps
                sigma = compute_sigma(eps, phase_map, C_phases)
                break
            prev_res = res

        sigma_avg = sigma.mean(axis=(0, 1, 2))
        C_eff[:, case] = sigma_avg
        iters_used.append(it + 1)
        if verbose:
            print(f'    case {case} done in {it+1} iters '
                  f'({time.time()-t_case:.1f}s)  '
                  f'<σ> = [{", ".join(f"{v:.3f}" for v in sigma_avg)}]')

    return C_eff, iters_used

def moduli_from_C(C):
    """Extract direction-resolved Young's, bulk, and shear moduli from 6x6 C.

    Returns dict: E_x, E_y, E_z, G_yz, G_xz, G_xy, K, anisotropy
    """
    try:
        S = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return {k: float('nan') for k in
                ['E_x','E_y','E_z','G_yz','G_xz','G_xy','K','anisotropy']}
    E_x = 1.0 / S[0, 0] if S[0, 0] > 0 else float('nan')
    E_y = 1.0 / S[1, 1] if S[1, 1] > 0 else float('nan')
    E_z = 1.0 / S[2, 2] if S[2, 2] > 0 else float('nan')
    G_yz = 1.0 / S[3, 3] if S[3, 3] > 0 else float('nan')
    G_xz = 1.0 / S[4, 4] if S[4, 4] > 0 else float('nan')
    G_xy = 1.0 / S[5, 5] if S[5, 5] > 0 else float('nan')
    K = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9.0
    G_avg = (G_yz + G_xz + G_xy) / 3.0
    anisotropy = (max(E_x, E_y, E_z) - min(E_x, E_y, E_z)) / max(E_x, E_y, E_z)
    return dict(E_x=E_x, E_y=E_y, E_z=E_z,
                G_yz=G_yz, G_xz=G_xz, G_xy=G_xy, G_avg=G_avg,
                K=K, anisotropy=anisotropy)

def downsample_phase(phase_map, factor):
    """Block-mode downsampling for a small-int phase map."""
    if factor == 1: return phase_map
    Z, Y, X = phase_map.shape
    Z2, Y2, X2 = Z // factor, Y // factor, X // factor
    cropped = phase_map[:Z2 * factor, :Y2 * factor, :X2 * factor]
    blocks = cropped.reshape(Z2, factor, Y2, factor, X2, factor)
    # mode along blocked axes
    flat = blocks.reshape(Z2, Y2, X2, factor**3)
    # majority vote
    counts = np.zeros((Z2, Y2, X2, len(PHASE_PROPS)), dtype=np.int32)
    for p in range(len(PHASE_PROPS)):
        counts[..., p] = (flat == p).sum(axis=-1)
    out = counts.argmax(axis=-1).astype(np.uint8)
    return out

def reconstruct_phase_map(ct, mask, th):
    """0=air, 1=ice, 2=glass.

    Voxels OUTSIDE the specimen mask (corners of bbox, gripper region, etc.)
    are filled with ice — i.e. treated as the matrix phase.  This avoids the
    bbox padding being counted as void and dominating the FFT cell.
    """
    out = np.full(ct.shape, 1, dtype=np.uint8)   # default = ice (matrix)
    inside = mask > 0
    air    = inside & (ct <  th['air'])
    glass  = inside & (ct >= th['ig']) & (ct < th['ga'])
    out[air]   = 0
    out[glass] = 2
    return out

def tight_bbox(mask):
    z = np.where(mask.any(axis=(1, 2)))[0]
    y = np.where(mask.any(axis=(0, 2)))[0]
    x = np.where(mask.any(axis=(0, 1)))[0]
    return (slice(z[0], z[-1] + 1),
            slice(y[0], y[-1] + 1),
            slice(x[0], x[-1] + 1))

# ---------------------------------------------------------------------------
print(f'FFT elasticity — scans: {ALL_SCANS}')
print(f'  data dir   : {DATA}')
print(f'  output     : {OUT}')
print(f'  downsample : {DOWNSAMPLE}')
print(f'  phase props (E [GPa], ν): air {PHASE_PROPS[0]}  ice {PHASE_PROPS[1]}  '
      f'glass {PHASE_PROPS[2]}')

rows = []
for s in ALL_SCANS:
    print(f'\n=== scan {s} ===')
    t0 = time.time()
    ct   = tifffile.imread(os.path.join(DATA, f'ct_scan{s:02d}_aligned.tif'))
    mask = tifffile.imread(os.path.join(DATA, f'specimen_mask_scan{s:02d}_aligned.tif'))
    print(f'  loaded   shape={ct.shape}   ({time.time()-t0:.1f}s)')

    bbox = tight_bbox(mask > 0)
    ct   = ct[bbox]
    mask = mask[bbox]
    print(f'  cropped to bbox shape={ct.shape}')

    phase_map = reconstruct_phase_map(ct, mask, TH[s])
    del ct, mask
    print(f'  phase map: shape={phase_map.shape}  '
          f'air={ (phase_map==0).sum()/phase_map.size :.3f}  '
          f'ice={ (phase_map==1).sum()/phase_map.size :.3f}  '
          f'glass={ (phase_map==2).sum()/phase_map.size :.3f}')

    t1 = time.time()
    pm = downsample_phase(phase_map, DOWNSAMPLE)
    del phase_map
    print(f'  downsampled to shape={pm.shape}  ({time.time()-t1:.1f}s)')

    t2 = time.time()
    C_eff, iters_used = moulinec_suquet(pm, PHASE_PROPS,
                                         max_iter=MAX_ITER, tol=TOL, verbose=True)
    print(f'  MS done ({time.time()-t2:.1f}s).  iters per case: {iters_used}')

    np.save(os.path.join(OUT, f'C_eff_scan{s:02d}.npy'), C_eff)
    print(f'  C_eff[0..2,0..2] (GPa):\n    {C_eff[:3, :3]}')

    moduli = moduli_from_C(C_eff)
    moduli['scan'] = s
    moduli['n_iters_max'] = max(iters_used)
    rows.append(moduli)
    print(f'  E (x,y,z) = ({moduli["E_x"]:.3f}, {moduli["E_y"]:.3f}, '
          f'{moduli["E_z"]:.3f}) GPa  '
          f'K = {moduli["K"]:.3f}  G_avg = {moduli["G_avg"]:.3f}')

df = pd.DataFrame(rows, columns=['scan', 'E_x', 'E_y', 'E_z',
                                  'G_yz', 'G_xz', 'G_xy', 'G_avg',
                                  'K', 'anisotropy', 'n_iters_max'])
df.to_csv(os.path.join(OUT, 'summary.csv'), sep=';', index=False)
print(f'\nwrote {os.path.join(OUT, "summary.csv")}')
print('\nDone.')
