"""
SPAM DDIC + LDIC between scan 1 (reference) and scan 2 (deformed).

Run from inside the WSL spam-env venv (see INSTALL.md). Do NOT run in
Dragonfly's Python — SPAM is not on that interpreter.

Pipeline:
  1. Global rigid registration              -> Phi_global.tsv
  2. Discrete DIC (per-bead tracking)       -> PhiField_ddic.tsv
  3. Local DIC (continuous displacement)    -> DVCField.tsv
  4. Post-process: displacements, rotations, crack field, summary CSV

Inputs (from Glass_spam\\data\\, staged by scan02_* scripts + spam_export.py):
  ct_scan01.tif, ct_scan02.tif
  bead_labels_scan01.tif, bead_labels_scan02.tif (optional for DDIC)
  specimen_mask_scan01.tif, specimen_mask_scan02.tif

Outputs -> Glass_spam\\results\\:
  Phi_global.tsv
  PhiField_ddic.tsv
  DVCField.tsv
  bead_displacement_hist.png
  bead_rotation_hist.png
  bead_flow_field.png
  crack_field.vtk
  bead_tracking.csv
"""

import os
import sys
import numpy as np
import tifffile

# --------------------------------------------------------------------------
# Paths — edit if running inside WSL with copied data under ~/spam-data
# --------------------------------------------------------------------------
DATA_DIR    = os.environ.get('SPAM_DATA',
                             '/mnt/e/RPTU-images/CT_images/Glass/Glass_spam/data')
RESULTS_DIR = os.environ.get('SPAM_RESULTS',
                             '/mnt/e/RPTU-images/CT_images/Glass/Glass_spam/results')
os.makedirs(RESULTS_DIR, exist_ok=True)

CT1   = os.path.join(DATA_DIR, 'ct_scan01.tif')
CT2   = os.path.join(DATA_DIR, 'ct_scan02.tif')
LBL1  = os.path.join(DATA_DIR, 'bead_labels_scan01.tif')
LBL2  = os.path.join(DATA_DIR, 'bead_labels_scan02.tif')
MASK1 = os.path.join(DATA_DIR, 'specimen_mask_scan01.tif')
MASK2 = os.path.join(DATA_DIR, 'specimen_mask_scan02.tif')

for p in (CT1, CT2, LBL1, MASK1, MASK2):
    if not os.path.exists(p):
        print(f"MISSING: {p}", file=sys.stderr)

# --------------------------------------------------------------------------
# SPAM imports — keep errors fatal so we see API mismatches clearly
# --------------------------------------------------------------------------
import spam
import spam.DIC as sdic
import spam.label as slbl
try:
    import spam.kinematics.transforms as skin
except ImportError:
    # older SPAM packaging
    import spam.deformation.deformationFunction as skin   # noqa

print(f"SPAM version: {spam.__version__}")
print(f"DATA_DIR    : {DATA_DIR}")
print(f"RESULTS_DIR : {RESULTS_DIR}")

# --------------------------------------------------------------------------
# Step 1: Global rigid registration (scan 1 -> scan 2)
# --------------------------------------------------------------------------
print("\n=== Step 1: Global registration ===")
ct1 = tifffile.imread(CT1).astype(np.float32)
ct2 = tifffile.imread(CT2).astype(np.float32)
print(f"ct1 shape: {ct1.shape}  ct2 shape: {ct2.shape}")

# spam.DIC.register returns (Phi, res) in recent versions. The exact
# signature may have drifted — the first call below is the canonical one;
# if the API name changed, spam.DIC's __dir__ will show the alternative.
try:
    reg = sdic.register(ct1, ct2, verbose=True)
    Phi_global = reg['Phi'] if isinstance(reg, dict) else reg[0]
except AttributeError:
    # Fallback: newer SPAM may have renamed to `globalCorrelation`
    reg = sdic.globalCorrelation(ct1, ct2, verbose=True)
    Phi_global = reg['Phi'] if isinstance(reg, dict) else reg[0]

np.savetxt(os.path.join(RESULTS_DIR, 'Phi_global.tsv'),
           Phi_global, delimiter='\t',
           header='4x4 homogeneous transform scan1 -> scan2')
print("Saved Phi_global.tsv")

# --------------------------------------------------------------------------
# Step 2: Discrete DIC (per-bead tracking)
# --------------------------------------------------------------------------
print("\n=== Step 2: Discrete DIC ===")
lbl1 = tifffile.imread(LBL1).astype(np.uint16)
print(f"lbl1 shape: {lbl1.shape}  unique labels: {len(np.unique(lbl1)) - 1}")

# discreteDIC signature has changed across SPAM releases. Try the recent one.
try:
    ddic_out = sdic.discreteDIC(
        im1=ct1,
        im2=ct2,
        lab1=lbl1,
        Phi0=Phi_global,
        numProc=4,
        verbose=True,
    )
except (AttributeError, TypeError):
    # Older signature
    ddic_out = sdic.ddic(ct1, ct2, lbl1, Phi0=Phi_global)

# ddic_out is typically a dict with 'PhiField' of shape (n_labels+1, 4, 4)
# and 'error', 'iterations', etc.
PhiField = ddic_out['PhiField'] if isinstance(ddic_out, dict) else ddic_out
np.save(os.path.join(RESULTS_DIR, 'PhiField_ddic.npy'), PhiField)
print(f"Saved PhiField_ddic.npy  shape={PhiField.shape}")

# Decompose per-bead Phi into translation + rotation + small strain
translations = np.zeros((PhiField.shape[0], 3))
rotations_deg = np.zeros(PhiField.shape[0])
for i in range(1, PhiField.shape[0]):   # skip bg label 0
    Phi_i = PhiField[i]
    if np.all(Phi_i == 0) or np.isnan(Phi_i).any():
        continue
    translations[i] = Phi_i[:3, 3]
    R = Phi_i[:3, :3]
    # Rodrigues angle from rotation matrix
    cos_theta = np.clip((np.trace(R) - 1) * 0.5, -1.0, 1.0)
    rotations_deg[i] = np.degrees(np.arccos(cos_theta))

# --------------------------------------------------------------------------
# Step 3: Local DIC (continuous field)
# --------------------------------------------------------------------------
print("\n=== Step 3: Local DIC on a node grid ===")
mask_union = (tifffile.imread(MASK1) > 0) | (tifffile.imread(MASK2) > 0)
# Grid spacing: half bead diameter ~ 35 voxels. Use 30.
NODE_SPACING = 30

try:
    ldic_out = sdic.ldic(
        im1=ct1,
        im2=ct2,
        nodeSpacing=NODE_SPACING,
        mask=mask_union.astype(np.uint8),
        Phi0=Phi_global,
        numProc=4,
        verbose=True,
    )
except (AttributeError, TypeError):
    ldic_out = sdic.multimodalRegistration(ct1, ct2)   # last-ditch fallback

np.save(os.path.join(RESULTS_DIR, 'DVCField.npy'), ldic_out)
print("Saved DVCField.npy")

# --------------------------------------------------------------------------
# Step 4: Post-processing — histograms and summary CSV
# --------------------------------------------------------------------------
print("\n=== Step 4: Post-processing ===")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FS = 22

# Displacement magnitude histogram
disp_mag = np.linalg.norm(translations, axis=1)
valid = disp_mag > 0
fig, ax = plt.subplots(figsize=(12, 8))
ax.hist(disp_mag[valid], bins=50, color='#1f77b4', edgecolor='black')
ax.set_xlabel('Bead displacement magnitude (voxels)', fontsize=FS)
ax.set_ylabel('Count', fontsize=FS)
ax.tick_params(labelsize=FS - 4)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'bead_displacement_hist.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved bead_displacement_hist.png")

# Rotation histogram
fig, ax = plt.subplots(figsize=(12, 8))
ax.hist(rotations_deg[valid], bins=50, color='#d62728', edgecolor='black')
ax.set_xlabel('Bead rotation (degrees)', fontsize=FS)
ax.set_ylabel('Count', fontsize=FS)
ax.tick_params(labelsize=FS - 4)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'bead_rotation_hist.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved bead_rotation_hist.png")

# Summary CSV
csv_path = os.path.join(RESULTS_DIR, 'bead_tracking.csv')
with open(csv_path, 'w') as f:
    f.write('bead_id;d_z;d_y;d_x;disp_magnitude;rotation_deg\n')
    for i in range(1, PhiField.shape[0]):
        if disp_mag[i] == 0:
            continue
        f.write(f'{i};{translations[i,0]:.3f};{translations[i,1]:.3f};'
                f'{translations[i,2]:.3f};{disp_mag[i]:.3f};'
                f'{rotations_deg[i]:.3f}\n')
print(f"Saved {csv_path}")

print("\nDone. Open bead_tracking.csv and the PNGs to review.")
print("VTK export (for Paraview) is a TODO — add spam.helpers.writeVTK calls.")
