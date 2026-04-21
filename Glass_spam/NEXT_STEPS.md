# SPAM — where we left off (2026-04-23 evening, done for the day)

Resume notes so tomorrow's Claude session can pick up without re-discovering.

## TL;DR — DDIC SUCCEEDED

- spam-ddic ran in 3:04 on 16 processes, 282 beads.
- Linear fit Zdisp vs Zpos: slope −0.1620 (theory −142/864 = −0.1644), 1.5% match.
- Compression captured: bottom fixed (−2.66 vox at z=0), top compressed (−144 vox at z=874).
- 279/282 beads valid. Median |u| = 2102 µm. Zdisp range −4460 to +479 µm.
- Only 2 beads fully converged (rs=2); 280 hit max-iter (rs=1) — USABLE but dPhiNorm didn't reach 0.001 in 50 iter.

## Output files (all in `results/`)

- `bead_tracking_ddic.csv` — per-bead displacements + rotation + QC flags (semicolon-separated)
- `zdisp_vs_height.png` — compression field scatter + fit + theory line
- `disp_magnitude_hist.png` — |u| distribution
- `rotation_hist.png` — per-bead rotation (deg)
- `xy_zdisp_map.png` — XY map, color = Zdisp
- `phi_qc.png` — iterations / error / returnStatus panels

TSVs + VTK live in WSL at `~/spam-results/`:
- `glass_ddic-ddic.tsv` (raw DDIC output, 283 rows × 22 cols)
- `glass_ddic-ddic.vtk` (ParaView-ready point cloud with displacements)
- `phi_seed.tsv` (hand-crafted axial-compression guess that made DDIC work)

## TOMORROW — recommended next steps

Pick one or more, in priority order:

1. **Open `results/` and review the 5 PNGs + CSV.**
   - Is the compression field clean? Any obvious outliers?
   - Are the rotations plausible (probably <10°)?
   - Does the XY Zdisp map show radial pattern or just noise?

2. **If rs=1 (max-iter) convergence bothers you:**
   - Re-run spam-ddic with `-it 200 -dp 0.0005 -ug`.
   - Expected runtime ~12 min (vs 3 min for 50 iter). Claude can do this.

3. **Open `glass_ddic-ddic.vtk` in ParaView for 3D visualization:**
   - Path in Windows: `\\wsl$\Ubuntu\home\cak7496\spam-results\glass_ddic-ddic.vtk`
   - Or copy to Glass_spam\results if you want it on Windows.
   - Each bead = one point with Zdisp/Ydisp/Xdisp vectors and F tensor.

4. **Run spam-ldic** (local continuous DIC on a grid) for strain/rotation fields
   BETWEEN beads, not just at bead centers. Takes ~20–60 min depending on grid
   spacing. Useful if you need voxel-wise strain for the paper.

5. **More separation on Z-stacked beads** (user flagged earlier):
   - Current ERODE_ITERATIONS=24 isotropic. If bead merges in Z bother the DDIC,
     bump to 28–30 OR add a Z-anisotropic erosion pass (see conversation notes).
   - Only worth doing if the DDIC plots show clustered outliers in Z.

## To resume in Claude

Say something like: "resume SPAM — read the results and tell me what's interesting"
or "open the vs_height plot and suggest filters"
or "rerun DDIC with -it 200".

Everything needed is in memory + this file.

---
(History below preserved for context.)

# SPAM — where we left off (2026-04-23)

Resume notes so tomorrow's Claude session can pick up without re-discovering.

## Completed today

- **WSL SPAM install is done.** Ubuntu 24.04.2 LTS (NOT 22.04), Python 3.12.3,
  venv at `~/spam-venv` (NOT `~/spam-env` as INSTALL.md says), SPAM 0.8.1.5 +
  spambind 0.8.3. Verified: `spam-ldic --help` works, `import spam.DIC, spam.label`
  works. INSTALL.md paths are stale.
- Activate with: `wsl -d Ubuntu` then `source ~/spam-venv/bin/activate`.
- All 6 TIFFs are staged in `data/` (2.6 GB total):
  `ct_scan01/02.tif`, `bead_labels_scan01/02.tif`, `specimen_mask_scan01/02.tif`.

## SPAM 0.8.1.5 API — confirmed names

Inspected `dir(spam.DIC)` on this install. The canonical calls are:

- `spam.DIC.register(im1, im2, ...)` — global rigid registration.
- `spam.DIC.ddic(...)` — per-label discrete DIC (**NOT** `discreteDIC`).
- `spam.DIC.ldic(...)` — local DIC on a grid.
- Also available: `globalCorrelation`, `globalDVC`, `registerMultiscale`,
  `pixelSearch`, `multimodalRegistration`, `kinematics`.

`spam_analysis.py`'s speculative `discreteDIC`-first call will fail with
AttributeError and fall through to the `ddic` branch — so the script works in
principle, but the `discreteDIC` branch is dead code.

## Progress 2026-04-23 evening — spam-reg convergence issues

Attempted spam-reg 4 times (rigid / affine / overlap mask / physical-Z alignment /
hand-crafted Phi seed). All hit max iterations at binning 4 or 8 without
converging. Root cause: the compression displacement (~71 voxel mean Z shift,
up to 142 at the top) is larger than the Newton-method search radius at
coarse binnings, and zero-padding artifacts dominate the residual. NOT a
label-quality problem.

**Workaround that should work:** skip spam-reg entirely and feed a
hand-crafted Phi seed (axial compression, Fzz=0.836, Zdisp=-71.7 at node
z=436.5) straight to spam-ddic with `-F all`. DDIC's per-bead correlation
is local and much more robust than global rigid. Seed file saved at
`~/spam-results/phi_seed.tsv`.

**Physical alignment step required before any SPAM run:**
Scan 1 (raw slices 98..961, 864 slices) and scan 2 (raw slices 88..809,
722 slices) have DIFFERENT physical Z ranges. Must pad both into a common
874-slice frame: scan 1 at common slices [10..873], scan 2 at [0..721].
Script sitting in `~/spam-data/` creates `*_aligned.tif` files.

## Bead labels staleness caught 2026-04-23

User noticed: current `bead_labels_scan01.tif` is from Apr 20 16:51, written
BEFORE the `MARKER_METHOD='cc'` connected-components approach was added to
`scan01_bead_analysis.py`. Scan 1 labels were produced with the older
peak-based method. Before retrying DDIC, user will re-run scan 1's
segmentation + bead_analysis + spam_export in Dragonfly console (scan 1 was
already in Dragonfly).

After that, Claude re-runs in WSL:
1. Regenerate aligned TIFFs (pad scan 1 to common 874-slice frame; scan 2
   aligned TIFF already exists).
2. Rerun spam-ddic with phi_seed.tsv.

## Chosen plan (agreed in outline, not yet executed)

User preference: pending between (A) CLI route and (B) patched Python script.
Discussion proposed **CLI route** as more stable:

1. **Copy TIFFs to WSL-native `~/spam-data`** — I/O over `/mnt/e/...` is much
   slower than `~/` for 500 MB files.
2. **Smoke-test `spam-reg` first** on scan1 vs scan2 — fast, produces
   `Phi_global.tsv`, confirms data/orientation before committing to DDIC.
3. **Then `spam-ddic`** using `bead_labels_scan01.tif` — per-bead displacement
   + rotation, scientifically the main result for this compression scan.
4. **Optional `spam-ldic`** — continuous field, only if DDIC isn't enough.

Tradeoff: CLI is slower to set up (config files / args) but its signatures are
frozen across SPAM versions. Python-API route (`spam_analysis.py`) is one
script but bound to 0.8.1.5 internals.

**Ask the user** which route to take before running anything.

## Commands to resume quickly (tomorrow)

```bash
# Open WSL
wsl -d Ubuntu
source ~/spam-venv/bin/activate

# If going CLI route — copy data first:
mkdir -p ~/spam-data
cp /mnt/e/RPTU-images/CT_images/Glass/Glass_spam/data/*.tif ~/spam-data/
cd ~/spam-data

# Smoke-test registration (CLI)
spam-reg --help          # review args
# then a real run, e.g.:
# spam-reg ct_scan01.tif ct_scan02.tif -od ~/spam-results -pre glass_

# Or: patch + run the Python script
python /mnt/e/RPTU-images/CT_images/Glass/Glass_spam/spam_analysis.py
```

## Known caveats

- scan 2 thresholds and cylinder are user-tuned in
  `scan02_segmentation.py` (T_AIR_ICE=6200, T_ICE_GLASS=16544, T_GLASS_AL=37000;
  cylinder cap1=(18723.10, 8408.06, 229.11), cap2=(18723.10, 8408.06, 17694.89),
  radius=6601.14 um). Already validated — don't re-tune unless asked.
- Scan 1 and scan 2 share the same X/Y crop (Dragonfly bounds minX=469,
  maxX=1045, minY=48, maxY=633) so the two volumes align voxel-for-voxel in
  (Y, X); Z differs (scan 1: slices 88..809; scan 2: same index range).
- `spam_analysis.py` references `spam.__version__` — that attribute does not
  exist on 0.8.1.5. Replace with `pip show spam` or skip the version print.
