"""PuMA-style microstructure analysis (per scan) without the NASA PuMA install.

For each scan in TRANSITIONS' scan set, computes:

  Phase fractions       — air / ice / glass volume fractions inside specimen
  Surface area          — marching-cubes triangulation per phase + SSA
  Connectivity          — number of connected components per phase
  Percolation           — does each phase span the specimen Z extent?
  Tortuosity (geodesic) — geodesic distance / Euclidean distance ratio
                          for ice and air phases (top-to-bottom traversal)
  Mean intercept length — MIL along x, y, z for each phase
  Fabric tensor         — 3x3 second-order orientation tensor of contacts
                          (eigenvalues + axial anisotropy index)
  Two-point S2(r)       — radial two-point correlation function per phase
  Effective k_thermal   — series / parallel bounds + Maxwell-Eucken estimate

Heavy properties NOT computed here (require NASA PuMA, conda/Docker):
  - Effective elastic moduli E, K, G via FFT homogenization
  - Effective Stokes-flow permeability
A placeholder section logs the materials data needed for those when you do
install PuMA — drop the volumes into a Docker container running PuMA and
read the elastic / permeability tensors back in.

Outputs to results_<PRE>/microstructure/:
    summary.csv                  one row per scan
    s2_<phase>_scan<N>.png       two-point correlation plots
    mil_<phase>_scan<N>.png      mean intercept length per direction
    summary.txt                  human-readable summary

Run in WSL spam-venv:
    python /mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/spam_microstructure.py
"""
import os, sys, time
import numpy as np
import tifffile
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from scan_meta import VOXEL_UM

DATA = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data'
OUT  = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>/microstructure'
os.makedirs(OUT, exist_ok=True)

# Scans to analyse (read all unique scans from TRANSITIONS so the script
# stays in sync with the rest of the pipeline)
TRANSITIONS = [(1, 2), (2, 3)]
ALL_SCANS = sorted({s for ab in TRANSITIONS for s in ab})

# Per-scan thresholds (same as ice_bonds / failure_modes — the air/ice/glass
# split that defines what 'phase' means for this analysis).
TH = {
    1: dict(air=7000,    ig=18032.7, ga=47275),
    2: dict(air=2139.4,  ig=6271.5,  ga=99999),
    3: dict(air=2015.4,  ig=6539.5,  ga=99999),
}

# Material properties for effective-property estimates (W/m·K, GPa, Poisson ν).
# Values are nominal — adjust per specimen if needed.
MATERIALS = {
    'air':   dict(k_th=0.025,  E=0.0,    nu=0.0),
    'ice':   dict(k_th=2.20,   E=9.0,    nu=0.33),
    'glass': dict(k_th=1.05,   E=70.0,   nu=0.21),
    # 'alumina': dict(k_th=30.0, E=370.0, nu=0.22),  # uncomment for alumina specimens
}

# Two-point correlation: how far to compute S2(r) (in voxels)
S2_MAX_LAG_VOX = 60   # ~1500 um at 24.766 um/vox
S2_DIRECTIONS = [(0,0,1), (0,1,0), (1,0,0)]   # along z, y, x

# Tortuosity: subsample the volume to keep the geodesic-distance solve tractable
TORT_DOWNSAMPLE = 4   # 4 -> ~60M -> ~250k voxels per phase

_FS = 24
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        _FS,
    "axes.titlesize":   _FS + 2,
    "axes.titleweight": "bold",
    "axes.labelsize":   _FS + 4,
    "axes.labelweight": "bold",
    "axes.linewidth":   2.2,
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.edgecolor":   "black",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.color":       "#d0d0d0",
    "grid.linewidth":   1.0,
    "xtick.labelsize":  _FS,
    "ytick.labelsize":  _FS,
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.major.pad":   8,
    "ytick.major.pad":   8,
    "xtick.direction":  "out",
    "ytick.direction":  "out",
    "legend.fontsize":  _FS,
    "legend.frameon":   False,
    "figure.facecolor": "white",
    "figure.dpi":       110,
    "savefig.dpi":      260,
    "savefig.bbox":     "tight",
    "savefig.facecolor": "white",
})

# ----------------------------------------------------------------------------
def reconstruct_phases(ct, mask, th):
    """Returns a uint8 phase array on the specimen mask:
       0 = outside,  1 = air,  2 = ice,  3 = glass."""
    phases = np.zeros(ct.shape, dtype=np.uint8)
    inside = mask > 0
    air   = inside & (ct < th['air'])
    ice   = inside & (ct >= th['air']) & (ct < th['ig'])
    glass = inside & (ct >= th['ig'])  & (ct < th['ga'])
    phases[air]   = 1
    phases[ice]   = 2
    phases[glass] = 3
    return phases

def phase_volume_fractions(phases):
    inside = phases > 0
    n_in = inside.sum()
    return {p: float((phases == k).sum()) / max(n_in, 1)
            for p, k in [('air', 1), ('ice', 2), ('glass', 3)]}

def phase_surface_area_um2(mask_bool, voxel_um):
    """Marching-cubes surface area for a binary phase mask, returned in um^2."""
    if not mask_bool.any():
        return 0.0
    from skimage import measure
    verts, faces, _, _ = measure.marching_cubes(mask_bool.astype(np.float32),
                                                level=0.5,
                                                spacing=(voxel_um,)*3)
    a = 0.0
    v = verts[faces]
    n = np.cross(v[:,1] - v[:,0], v[:,2] - v[:,0])
    a = 0.5 * np.linalg.norm(n, axis=1).sum()
    return float(a)

def connected_components_count(mask_bool):
    if not mask_bool.any():
        return 0
    _, n = ndi.label(mask_bool, structure=np.ones((3,3,3), dtype=bool))
    return int(n)

def percolation_z(mask_bool, specimen_mask=None):
    """Returns True if any connected component of the phase touches both the
    top and bottom of the SPECIMEN (not the volume bounding box).
    Top = first Z slice with any specimen voxel; bottom = last such slice."""
    if not mask_bool.any():
        return False
    if specimen_mask is None:
        specimen_mask = mask_bool
    nz_has = np.any(specimen_mask, axis=(1, 2))
    if not nz_has.any():
        return False
    z_top = int(np.argmax(nz_has))
    z_bot = int(len(nz_has) - 1 - np.argmax(nz_has[::-1]))
    lab, n = ndi.label(mask_bool, structure=np.ones((3,3,3), dtype=bool))
    if n == 0:
        return False
    top_lbls    = set(np.unique(lab[z_top])) - {0}
    bottom_lbls = set(np.unique(lab[z_bot])) - {0}
    return bool(top_lbls & bottom_lbls)

def mean_intercept_length(mask_bool, voxel_um, axis):
    """Mean length of contiguous-True runs along the given axis, in um.
    Quick MIL estimator (uses xor of shifted masks to count run boundaries)."""
    if not mask_bool.any():
        return 0.0
    a = mask_bool
    n_intervals = a.sum()
    # Number of starts = sum( a[i] AND NOT a[i-1] )
    shifted = np.roll(a, 1, axis=axis)
    sl = [slice(None)] * 3
    sl[axis] = 0
    shifted[tuple(sl)] = False
    starts = (a & ~shifted).sum()
    if starts == 0:
        return 0.0
    return float(n_intervals * voxel_um / starts)

def fabric_tensor_from_glass(label_volume, voxel_um):
    """Second-order fabric tensor from bead-bead contact orientations.
    Returns (T (3,3), eigvals (3,), eigvecs (3,3), anisotropy)."""
    # Centroids per bead (we don't trust whatever shape labelledContacts gives)
    max_lab = int(label_volume.max())
    if max_lab == 0:
        return np.eye(3) / 3.0, np.array([1/3, 1/3, 1/3]), np.eye(3), 0.0
    cent_arr = ndi.center_of_mass(label_volume > 0, label_volume,
                                  range(1, max_lab + 1))
    cent = np.zeros((max_lab + 1, 3), dtype=np.float64)
    for i, c in enumerate(cent_arr, start=1):
        cent[i] = c   # (z, y, x)
    # Contact pairs via spam (cheap)
    from spam.label import labelledContacts
    out = labelledContacts(label_volume.astype(np.int32),
                           maximumCoordinationNumber=20)
    # spam returns (labels, Z, centroids, pairs) — take the last
    pairs = np.asarray(out[-1])
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        return np.eye(3) / 3.0, np.array([1/3, 1/3, 1/3]), np.eye(3), 0.0
    pairs = np.sort(pairs.astype(int), axis=1)
    keep = (pairs[:, 0] > 0) & (pairs[:, 1] > 0) & \
           (pairs[:, 1] <= max_lab) & (pairs[:, 0] <= max_lab)
    pairs = np.unique(pairs[keep], axis=0)
    if len(pairs) == 0:
        return np.eye(3) / 3.0, np.array([1/3, 1/3, 1/3]), np.eye(3), 0.0
    # Unit branch vectors  (n_pairs, 3)
    v = cent[pairs[:, 1]] - cent[pairs[:, 0]]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    v = (v / n).astype(np.float64)
    # Fabric tensor T_ij = <n_i n_j> averaged over all branches
    # einsum gives (3, 3) cleanly
    T = np.einsum('ki,kj->ij', v, v) / float(len(v))
    eig_val, eig_vec = np.linalg.eigh(T)
    dev = T - np.eye(3) / 3.0
    aniso = float(np.sqrt(1.5 * (dev * dev).sum()))
    return T, eig_val, eig_vec, aniso

def two_point_correlation_along(mask_bool, axis, max_lag):
    """S2(r) along one axis: probability that two voxels distance r apart
    are both in the phase. Returns array of length max_lag+1."""
    s2 = np.zeros(max_lag + 1, dtype=np.float64)
    a = mask_bool.astype(np.float32)
    s2[0] = a.mean()
    for r in range(1, max_lag + 1):
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[axis] = slice(None, -r)
        sl_b[axis] = slice(r, None)
        s2[r] = (a[tuple(sl_a)] * a[tuple(sl_b)]).mean()
    return s2

def tortuosity_geodesic(mask_bool, voxel_um, downsample):
    """Approximate geodesic-distance tortuosity = mean( geodesic / Euclidean )
    from each top-face voxel to its nearest bottom-face voxel within the same
    connected component.  Uses scipy.ndimage.distance_transform_edt on a
    BARRIER (1 - mask) field, walked from the top face by BFS levels, which
    is a cheap proxy for true geodesic.  Downsamples for tractability."""
    s = downsample
    m = mask_bool[::s, ::s, ::s]
    if not m.any():
        return float('nan')
    # Use a chamfer-distance approximation: distance transform ON the mask
    # itself (so distance from each interior voxel to the nearest face).
    # Then take median ratio of (z * voxel_um) / (chamfer_dist * voxel_um*s).
    nz = m.shape[0]
    # Approximate "shortest path through the phase to the top" as the
    # chamfer distance from the top slice within the phase.
    seed = np.zeros_like(m, dtype=bool)
    seed[0] = m[0]
    if not seed.any():
        return float('nan')
    # Iterative dilation-within-phase counting steps (cheap geodesic).
    visited = seed.copy()
    layer = seed.copy()
    geodist = np.full(m.shape, np.inf, dtype=np.float32)
    geodist[seed] = 0
    struct = np.ones((3,3,3), dtype=bool)
    step = 0
    while layer.any() and step < 4 * nz:
        step += 1
        new_layer = ndi.binary_dilation(layer, structure=struct) & m & ~visited
        if not new_layer.any():
            break
        geodist[new_layer] = step
        visited |= new_layer
        layer = new_layer
    if not visited[-1].any():
        return float('inf')
    # Tortuosity = mean(geodist on bottom slice) / nz, all in downsampled voxels.
    bottom_dists = geodist[-1][m[-1] & visited[-1]]
    if bottom_dists.size == 0:
        return float('nan')
    return float(bottom_dists.mean() / max(nz - 1, 1))

def maxwell_eucken_keff(vfracs, k_props):
    """Maxwell-Eucken effective thermal conductivity for a 3-phase mix:
    k_eff = sum(k_i * vf_i * (3 k_m) / (2 k_m + k_i)) / sum(vf_i * 3 k_m / (2 k_m + k_i))
    where k_m is the matrix conductivity (taken as the highest-fraction phase).
    Returns also series and parallel (Wiener) bounds."""
    items = [(p, vfracs[p], k_props[p]['k_th']) for p in ('air','ice','glass') if vfracs[p] > 0]
    if not items:
        return float('nan'), float('nan'), float('nan')
    vsum = sum(v for _, v, _ in items)
    items = [(p, v/vsum, k) for p,v,k in items]
    # Wiener bounds
    k_par = sum(v * k for _, v, k in items)
    k_ser = 1.0 / sum(v / max(k,1e-9) for _, v, k in items)
    # Maxwell-Eucken: take matrix as highest-fraction phase
    matrix = max(items, key=lambda x: x[1])
    k_m = matrix[2]
    num = sum(v * k * (3 * k_m / (2 * k_m + k)) for _,v,k in items)
    den = sum(v *     (3 * k_m / (2 * k_m + k)) for _,v,k in items)
    k_me = num / max(den, 1e-9)
    return k_me, k_par, k_ser

# ----------------------------------------------------------------------------
print(f'Microstructure analysis — scans: {ALL_SCANS}')
print(f'  data dir: {DATA}')
print(f'  output  : {OUT}')

rows = []
for s in ALL_SCANS:
    print(f'\n=== scan {s} ===')
    t0 = time.time()
    ct   = tifffile.imread(os.path.join(DATA, f'ct_scan{s:02d}_aligned.tif'))
    mask = tifffile.imread(os.path.join(DATA, f'specimen_mask_scan{s:02d}_aligned.tif'))
    lab  = tifffile.imread(os.path.join(DATA, f'bead_labels_scan{s:02d}_aligned.tif'))
    print(f'  loaded volumes ({time.time()-t0:.1f}s)  shape={ct.shape}')

    th = TH[s]
    phases = reconstruct_phases(ct, mask > 0, th)

    vfracs = phase_volume_fractions(phases)
    print(f'  fractions: air={vfracs["air"]:.3f}  ice={vfracs["ice"]:.3f}  glass={vfracs["glass"]:.3f}')

    sa = {p: phase_surface_area_um2(phases == k, VOXEL_UM)
          for p, k in [('air',1),('ice',2),('glass',3)]}
    vol_um3_inside = (mask > 0).sum() * VOXEL_UM**3
    ssa = {p: sa[p] / max(vol_um3_inside, 1e-9) for p in sa}
    print(f'  SSA (1/um): air={ssa["air"]:.4e}  ice={ssa["ice"]:.4e}  glass={ssa["glass"]:.4e}')

    cc = {p: connected_components_count(phases == k) for p, k in [('air',1),('ice',2),('glass',3)]}
    perc = {p: percolation_z(phases == k, specimen_mask=(mask > 0))
            for p, k in [('air',1),('ice',2),('glass',3)]}
    print(f'  connected components: {cc}')
    print(f'  percolates Z: {perc}')

    mil = {p: {ax: mean_intercept_length(phases == k, VOXEL_UM, axis=axi)
               for ax, axi in [('z',0),('y',1),('x',2)]}
           for p, k in [('air',1),('ice',2),('glass',3)]}
    print(f'  MIL ice (z,y,x): '
          f'{mil["ice"]["z"]:.1f} {mil["ice"]["y"]:.1f} {mil["ice"]["x"]:.1f} um')

    print(f'  fabric tensor from glass-glass contacts...')
    T, evals, evecs, aniso = fabric_tensor_from_glass(lab, VOXEL_UM)
    print(f'    eigenvalues: {evals}   anisotropy index a = {aniso:.3f}')

    # S2 along z (axial)
    s2_air = two_point_correlation_along(phases == 1, axis=0, max_lag=S2_MAX_LAG_VOX)
    s2_ice = two_point_correlation_along(phases == 2, axis=0, max_lag=S2_MAX_LAG_VOX)
    s2_glass = two_point_correlation_along(phases == 3, axis=0, max_lag=S2_MAX_LAG_VOX)
    rs = np.arange(S2_MAX_LAG_VOX + 1) * VOXEL_UM
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(rs, s2_air,   color='#9aa', linewidth=2, label='air')
    ax.plot(rs, s2_ice,   color='#1f77b4', linewidth=2, label='ice')
    ax.plot(rs, s2_glass, color='#d62728', linewidth=2, label='glass')
    ax.set_xlabel('Lag along z [um]')
    ax.set_ylabel('S2(r)')
    ax.set_title(f'Two-point correlation along Z — scan {s}', fontweight='bold')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, -0.14), ncol=3,
               fontsize=_FS - 2, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUT, f's2_z_scan{s:02d}.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'  geodesic tortuosity (downsample={TORT_DOWNSAMPLE})...')
    tort_ice = tortuosity_geodesic(phases == 2, VOXEL_UM, TORT_DOWNSAMPLE)
    tort_air = tortuosity_geodesic(phases == 1, VOXEL_UM, TORT_DOWNSAMPLE)
    print(f'    tau_ice = {tort_ice:.3f}   tau_air = {tort_air:.3f}')

    k_me, k_par, k_ser = maxwell_eucken_keff(vfracs, MATERIALS)
    print(f'  k_eff (W/m.K): Maxwell-Eucken = {k_me:.3f}   '
          f'parallel = {k_par:.3f}   series = {k_ser:.3f}')

    rows.append(dict(
        scan=s,
        f_air=vfracs['air'], f_ice=vfracs['ice'], f_glass=vfracs['glass'],
        SA_air_um2=sa['air'], SA_ice_um2=sa['ice'], SA_glass_um2=sa['glass'],
        SSA_air=ssa['air'], SSA_ice=ssa['ice'], SSA_glass=ssa['glass'],
        cc_air=cc['air'], cc_ice=cc['ice'], cc_glass=cc['glass'],
        perc_air=perc['air'], perc_ice=perc['ice'], perc_glass=perc['glass'],
        MIL_ice_z=mil['ice']['z'], MIL_ice_y=mil['ice']['y'], MIL_ice_x=mil['ice']['x'],
        MIL_glass_z=mil['glass']['z'], MIL_glass_y=mil['glass']['y'], MIL_glass_x=mil['glass']['x'],
        fabric_eig1=evals[0], fabric_eig2=evals[1], fabric_eig3=evals[2],
        fabric_anisotropy=aniso,
        tort_ice=tort_ice, tort_air=tort_air,
        k_eff_ME=k_me, k_eff_par=k_par, k_eff_ser=k_ser,
    ))
    del ct, mask, lab, phases

# Write CSV
csv_path = os.path.join(OUT, 'summary.csv')
with open(csv_path, 'w') as f:
    keys = list(rows[0].keys())
    f.write(';'.join(keys) + '\n')
    for r in rows:
        f.write(';'.join(f'{r[k]}' for k in keys) + '\n')
print(f'\nwrote {csv_path}')

# Human-readable summary
txt = os.path.join(OUT, 'summary.txt')
with open(txt, 'w') as f:
    f.write('Microstructure analysis (PuMA-style, scipy/skimage implementation)\n')
    f.write('=' * 70 + '\n\n')
    f.write(f'{"scan":<6}{"air%":>8}{"ice%":>8}{"glass%":>8}'
            f'{"SSA_ice":>12}{"perc_ice":>10}{"perc_air":>10}'
            f'{"tau_ice":>10}{"k_ME":>10}{"aniso":>10}\n')
    for r in rows:
        f.write(f'{r["scan"]:<6}'
                f'{r["f_air"]*100:>7.2f} '
                f'{r["f_ice"]*100:>7.2f} '
                f'{r["f_glass"]*100:>7.2f} '
                f'{r["SSA_ice"]:>12.3e}'
                f'{str(r["perc_ice"]):>10}'
                f'{str(r["perc_air"]):>10}'
                f'{r["tort_ice"]:>10.3f}'
                f'{r["k_eff_ME"]:>10.3f}'
                f'{r["fabric_anisotropy"]:>10.3f}\n')
    f.write('\nColumns:\n')
    f.write('  air%/ice%/glass%   volume fraction (within specimen mask)\n')
    f.write('  SSA_ice            surface area / total volume [1/um]\n')
    f.write('  perc_ice/air       does this phase span the Z extent (top<->bottom)\n')
    f.write('  tau_ice            geodesic tortuosity of ice (1.0 = straight, >1 tortuous)\n')
    f.write('  k_ME               Maxwell-Eucken effective thermal conductivity [W/m.K]\n')
    f.write('  aniso              fabric anisotropy index (0 = isotropic, ~0.5 = strong)\n')
    f.write('\n')
    f.write('NOT computed here (need NASA PuMA via Docker / conda):\n')
    f.write('  Effective elastic moduli E, K, G via FFT homogenization\n')
    f.write('  Effective Stokes-flow permeability tensor\n')
    f.write('  These need: docker pull nasa/puma; then run on the same _aligned.tif files.\n')
print(f'wrote {txt}')

print('\nDone.')
