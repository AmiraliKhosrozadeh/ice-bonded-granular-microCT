"""Estimate Kozeny-Carman air permeability per scan and per direction
for the <SPECIMEN> specimen, and correlate change with the
crack inventory already produced by spam_crack_analysis.py.

We do NOT solve Stokes flow (PuMA would, but it OOMs at this volume).
Instead we use the closed-form Kozeny-Carman estimate that is standard
for granular ice / packed glass beads:

    k_dir = (phi^3 / ((1 - phi)^2 * tau_dir^2 * c * S^2))

where:
    phi    = pore fraction (= air vfrac from segmentation)
    tau    = directional tortuosity (from spam_tortuosity_3d.py)
    S      = specific surface area of the SOLID (glass + ice) [1/length]
    c      = Kozeny constant (5 for spheres)

Also reports the much-simpler bead-pack form (Ergun packed-bed limit)
that depends only on porosity and bead diameter d:

    k_KC = (d^2 / 180) * (phi^3 / (1 - phi)^2)

Outputs into  results_<PRE>/permeability/
    permeability_summary.txt   per-scan + per-direction k, and Δk per transition
    permeability_summary.csv   machine-readable
    k_evolution.png            k_dir vs scan, one panel per direction
    delta_k_vs_crack.png       Δk per transition vs crack volume
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/results_<PRE>'
DEST = os.path.join(OUT, 'permeability'); os.makedirs(DEST, exist_ok=True)

VOXEL_UM = 24.7660229
BEAD_DIAM_UM = 1700.0    # nominal bead diameter
KOZENY_C = 5.0           # Kozeny constant for sphere packs

# 1) Read directional tortuosity + porosity ---------------------------------
TORT_CSV = os.path.join(OUT, 'tortuosity_evolution', 'summary.csv')
tort = []
with open(TORT_CSV) as f:
    r = csv.DictReader(f, delimiter=';')
    for row in r:
        tort.append(dict(scan=int(row['scan']), phase=row['phase'],
                         direction=row['direction'],
                         tau=float(row['tau']), percolates=row['percolates'] == 'True',
                         vfrac=float(row['vfrac'])))

# 2) Get per-scan SSA (m^-1).
# Strategy: total surface area of (glass + ice) divided by total specimen
# volume. We approximate this from the per-bead roughness CSV (scan 1) +
# the air vfrac evolution (assume the glass surface area is conserved; ice
# surface evolves with phase fractions). For an order-of-magnitude estimate
# we use the bead-only specific surface (6 / d for spheres) as the SSA of
# the SOLID, valid for the dilute-ice limit.
SSA_solid_per_um = 6.0 / BEAD_DIAM_UM    # 1/um; 6/d for spheres
SSA_solid_per_m  = SSA_solid_per_um * 1e6  # convert 1/um -> 1/m

# 3) Per-scan + per-direction permeability ---------------------------------
results = []
for entry in tort:
    if entry['phase'] != 'air':
        continue
    phi = entry['vfrac']
    tau = entry['tau']
    if not entry['percolates'] or not np.isfinite(tau) or tau == 0:
        k_full = float('nan')   # phase doesn't percolate -> no flow
    else:
        # full Kozeny-Carman with tortuosity, in m^2
        k_full = (phi**3) / ((1 - phi)**2 * (tau**2) * KOZENY_C * (SSA_solid_per_m**2))
    # Bead-pack closed form (no tortuosity, geometry-only)
    d_m = BEAD_DIAM_UM * 1e-6
    k_KC_bulk = (d_m**2 / 180.0) * (phi**3) / ((1 - phi)**2)
    results.append(dict(scan=entry['scan'], direction=entry['direction'],
                        phi=phi, tau=tau, percolates=entry['percolates'],
                        k_dir=k_full, k_KC_bulk=k_KC_bulk))

# 4) Cross-transition Δk + crack correlation -------------------------------
CRACK_FILE = os.path.join(OUT, 'crack_analysis', 'crack_summary.txt')
# Pull V_total per transition (in mm^3) from the existing summary text.
crack_V = {}
with open(CRACK_FILE) as f:
    for ln in f:
        ln = ln.strip()
        # rows like:  1->2          204.9532       9       0     391     101       0
        if ln.startswith('1->2') or ln.startswith('2->3'):
            tok = ln.split()
            crack_V[tok[0]] = float(tok[1])

# Δk per transition per direction (k_dir basis)
def get_k(scan, direction):
    for r in results:
        if r['scan'] == scan and r['direction'] == direction:
            return r['k_dir'], r['percolates']
    return float('nan'), False

deltas = []
for (a, b) in [(1, 2), (2, 3)]:
    for d in ('x', 'y', 'z'):
        ka, pa = get_k(a, d)
        kb, pb = get_k(b, d)
        if pa and pb and np.isfinite(ka) and np.isfinite(kb) and ka > 0:
            ratio = kb / ka
            dk = kb - ka
        else:
            ratio = float('nan'); dk = float('nan')
        deltas.append(dict(transition=f'{a}->{b}', direction=d,
                           k_a=ka, k_b=kb, dk=dk, ratio=ratio,
                           crack_vol_mm3=crack_V.get(f'{a}->{b}', float('nan'))))

# 5) Write summaries -------------------------------------------------------
csv_path = os.path.join(DEST, 'permeability_summary.csv')
with open(csv_path, 'w') as f:
    f.write('scan;direction;phi;tau;percolates;k_dir_m2;k_KC_bulk_m2\n')
    for r in results:
        f.write(f"{r['scan']};{r['direction']};{r['phi']:.4f};{r['tau']:.3f};"
                f"{r['percolates']};{r['k_dir']:.6e};{r['k_KC_bulk']:.6e}\n")
print(f'wrote {csv_path}')

txt_path = os.path.join(DEST, 'permeability_summary.txt')
with open(txt_path, 'w') as f:
    f.write('Kozeny-Carman air permeability — <SPECIMEN>\n')
    f.write('=' * 60 + '\n\n')
    f.write(f'Bead diameter             : {BEAD_DIAM_UM:.0f} um\n')
    f.write(f'Solid specific surface S  : {SSA_solid_per_m:.2e} 1/m '
            f'(6/d for spheres)\n')
    f.write(f'Kozeny constant           : {KOZENY_C}\n')
    f.write(f'Voxel size                : {VOXEL_UM:.4f} um\n\n')
    f.write('Per scan / direction (air phase):\n')
    f.write(f'{"scan":>4}  {"dir":>4}  {"phi":>6}  {"tau":>8}  '
            f'{"perc?":>6}  {"k_dir [m2]":>14}  {"k_KC_bulk [m2]":>16}\n')
    for r in results:
        ks = '   no-perc'   if not r['percolates'] else f'{r["k_dir"]:.3e}'
        f.write(f'{r["scan"]:>4}  {r["direction"]:>4}  {r["phi"]:>6.3f}  '
                f'{r["tau"]:>8.3f}  {str(r["percolates"]):>6}  '
                f'{ks:>14}  {r["k_KC_bulk"]:>16.3e}\n')
    f.write('\nΔ permeability per transition (per direction, air phase):\n')
    f.write(f'{"trans":>6}  {"dir":>4}  {"k_a [m2]":>12}  {"k_b [m2]":>12}  '
            f'{"k_b/k_a":>8}  {"crack_vol [mm3]":>16}\n')
    for d in deltas:
        ka = f'{d["k_a"]:.3e}' if np.isfinite(d['k_a']) else '   no-perc'
        kb = f'{d["k_b"]:.3e}' if np.isfinite(d['k_b']) else '   no-perc'
        rt = f'{d["ratio"]:.2f}' if np.isfinite(d['ratio']) else '    --'
        f.write(f'{d["transition"]:>6}  {d["direction"]:>4}  {ka:>12}  '
                f'{kb:>12}  {rt:>8}  {d["crack_vol_mm3"]:>16.2f}\n')
    f.write('\nInterpretation:\n')
    f.write('  k_dir uses the directional tortuosity tau computed earlier.\n')
    f.write('  k_KC_bulk is the geometry-only Kozeny-Carman estimate from\n')
    f.write('  porosity + bead diameter (no tortuosity, no crack info).\n\n')
    f.write('  k_b / k_a > 1.5 in a given direction = significant\n')
    f.write('  permeability INCREASE -> usually a NEW crack/pathway opened.\n')
    f.write('  Cross-checking: when crack_vol_mm3 jumps and a previously\n')
    f.write('  non-percolating direction starts percolating, that crack is\n')
    f.write('  the cause of the new flow path.\n')
print(f'wrote {txt_path}')

# 6) Plots ----------------------------------------------------------------
# (a) k_dir vs scan, 3 panels (x, y, z)
fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
for ax, d in zip(axes, ('x', 'y', 'z')):
    ks = [r['k_dir'] for r in results if r['direction'] == d]
    sc = [r['scan']  for r in results if r['direction'] == d]
    sk = [r['percolates'] for r in results if r['direction'] == d]
    # plot percolating ones; mark non-percolating with NaN at the bottom
    sc_p = [s for s, p in zip(sc, sk) if p]
    ks_p = [k for k, p in zip(ks, sk) if p]
    ax.semilogy(sc_p, ks_p, 'o-', linewidth=3, markersize=14,
                color='#1f77b4', markeredgecolor='black')
    sc_np = [s for s, p in zip(sc, sk) if not p]
    if sc_np:
        ax.scatter(sc_np, [1e-20] * len(sc_np), s=200, marker='x',
                   color='#d62728', label='no percolation')
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel('Scan')
    ax.set_title(f'k_{d} (m²)', fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    if sc_np:
        ax.legend()
axes[0].set_ylabel('Air permeability k (m²)')
plt.tight_layout()
fig.savefig(os.path.join(DEST, 'k_evolution.png'), dpi=150,
            bbox_inches='tight'); plt.close(fig)
print(f'wrote {os.path.join(DEST, "k_evolution.png")}')

# (b) Δk vs crack volume
fig, ax = plt.subplots(figsize=(13, 9))
xs = []; ys = []; lbls = []
for d in deltas:
    if not np.isfinite(d['ratio']):
        continue
    xs.append(d['crack_vol_mm3']); ys.append(d['ratio'])
    lbls.append(f'{d["transition"]} {d["direction"]}')
ax.scatter(xs, ys, s=200, alpha=0.85, color='#d62728',
           edgecolor='black', linewidth=0.8)
for x, y, lb in zip(xs, ys, lbls):
    ax.annotate(lb, (x, y), textcoords='offset points',
                xytext=(8, 6), fontsize=14)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5)
ax.set_xlabel('Crack volume in transition [mm³]', labelpad=10)
ax.set_ylabel('Permeability ratio k_after / k_before', labelpad=10)
ax.set_title('Permeability change vs crack volume',
             fontweight='bold', pad=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(DEST, 'delta_k_vs_crack.png'), dpi=150,
            bbox_inches='tight'); plt.close(fig)
print(f'wrote {os.path.join(DEST, "delta_k_vs_crack.png")}')

print('\nDone. permeability outputs at', DEST)
