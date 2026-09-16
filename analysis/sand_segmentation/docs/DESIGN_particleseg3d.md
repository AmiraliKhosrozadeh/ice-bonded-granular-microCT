# Design doc — Stage 4 deep learning: ParticleSeg3D (border–core nnU-Net)

Companion to the Stage 0–3 classical scaffold. **Do not train yet** — first
review the Stage 3 baseline (`baseline_qc.png`) and choose correction ROIs.
This doc specifies the data format, Sand-Atlas use, ROI correction, border–core
encoding, fine-tuning recipe, GPU needs, folder layout, and commands so the GPU
and Sand-Atlas download can be lined up in parallel.

## 0. Method recap (what it learns)
ParticleSeg3D = **nnU-Net** backbone + **particle-size normalization** +
**border–core** label encoding. Each grain is re-encoded as 3 semantic classes
— background / **core** (eroded interior) / **border** (shell). The network does
ordinary 3-class semantic segmentation; instances are recovered by connected
components on `core` (touching grains stay split by the border ring) then growing
cores back through the border. This is why it beats EDT-watershed on touching
grains. Hard limit: misses grains < ~3 voxels — our fines (~4 vox) are marginal.

## 1. Environment (WSL, GPU)
```
# WSL Ubuntu, conda env
conda create -n psed3d python=3.10 -y && conda activate psed3d
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match CUDA
pip install nnunetv2 particleseg3d            # MIC-DKFZ packages
```
**This workstation (confirmed): RTX 3070 8 GB (compute 8.6), Ryzen 9 3900X
12c/24t, 32 GB RAM, driver 595.79.** Use CUDA 12.1 (cu121) wheels.
- **Inference: comfortable** on the 3070 (tiled) — zero-shot the released
  ParticleSeg3D weights here, immediately, no training.
- **Training 3D nnU-Net: tight.** nnU-Net 3d_fullres defaults assume ~11 GB; on
  8 GB expect OOM unless the plan's `patch_size`/`batch_size` are reduced
  (edit `nnUNetPlans.json` → smaller patch e.g. 96³ or 112³, batch 2). Works but
  slower and slightly weaker.
- **Recommended:** (1) zero-shot ParticleSeg3D on the 3070; (2) if it merges
  ice-bonded grains, EITHER light fine-tune with reduced patches locally OR push
  from-scratch/full fine-tune to a cloud/cluster A100 and bring weights back to
  infer on the 3070. No from-scratch full training on 8 GB.

## 2. Folder layout
```
Sand/pipeline/dl/
  raw/                      # input grayscale volumes (sand-phase masked), .nii.gz or zarr
  sand_atlas/               # downloaded Sand Atlas labelled volumes
  nnUNet_raw/ nnUNet_preprocessed/ nnUNet_results/   # nnU-Net env dirs
  rois/                     # Dragonfly-corrected ground-truth ROIs (your scans)
  predictions/              # model output instance labels
  border_core/              # encoded training labels
```

## 3. Training data — two sources
**(a) Sand Atlas (bulk prior, ~free labels).** Vego et al. 2025,
`sand-atlas.scigem.com` — 12,628 labelled grains incl. Hostun/Ottawa/Hamburg as
labelled TIFF. Download, resample to a common particle-size-normalized spacing,
convert per-grain labels → border–core. This supplies most of the training
signal; little hand-annotation needed (ParticleSeg3D itself trained on 41 patches).

**(b) Your Dragonfly-corrected ROI (domain-gap closer).** Their sand is dry/dense;
ours is **ice-bonded** (ice background, ice bonds at contacts). Take ONE ~200³
ROI from `Sand_100_500_T5_01`, correct the Stage-3 watershed labels in Dragonfly
(merge over-splits, split merges, add missed grains), export as a label TIFF.
This teaches the model the ice background + ice-bond contacts.

## 4. Border–core encoding (label conversion)
For each labelled grain id:
1. `core`  = binary-erode the grain by `r_erode` (≈2 vox; ≈1 for fines).
2. `border`= grain minus core (the shell).
3. background = not-grain (ice + air).
Stack to a 3-class volume {0 bg, 1 core, 2 border}. Inference inverts this:
`label(core)` → markers → watershed within (core∪border). Implement as
`dl/encode_border_core.py` (works on both Sand-Atlas and ROI labels).

## 5. nnU-Net dataset + commands
```
# 1. build dataset (Dataset501_SandGrains): imagesTr/ + labelsTr/ (border-core), dataset.json (3 classes)
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
# 2. fine-tune (3D full-res). Either train from scratch on Sand Atlas...
nnUNetv2_train 501 3d_fullres 0
#    ...or warm-start from the released ParticleSeg3D weights, then fine-tune on ROI.
# 3. inference (tiled, overlap, modest GPU)
nnUNetv2_predict -i dl/raw -o dl/predictions -d 501 -c 3d_fullres
# 4. border-core -> instances
python dl/decode_instances.py dl/predictions dl/predictions_labels
```
`particleseg3d` ships its own ` ps3d_inference`/`ps3d_train` wrappers that handle
size normalization + border-core automatically — prefer those if using the
released model; the raw nnU-Net path above is the from-scratch fallback.

## 6. Inputs / outputs (pipeline contract)
- **Input** = grayscale sand-phase volume from Stage 1 (sand kept; ice/air as
  background), + voxel spacing 0.024766 mm + approx mean grain diameter (~250 µm).
- **Output** = integer instance-label volume, SAME format as Stage 3
  `grain_labels.tif` → flows straight into Stage 8 SPAM tracking + Stage 9
  PSD/contacts/permeability. No downstream change.

## 7. Fallback — contact-film nnU-Net (Stage 5)
If ParticleSeg3D still merges ice-bonded grains, train a sibling 3-class model
{background(ice/air) / grain-core / grain-boundary-contact} where the middle
class is the *learned* inter-grain contact film (from the corrected ROI), then
watershed seeded from cores and dammed by the contact. Same nnU-Net infra; only
the label encoding differs (learned contact vs geometric erosion shell).

## 8. Validation (QC beyond Dice)
Compare watershed / ParticleSeg3D / contact-nnU-Net on the corrected ROI with:
split error, merge error, missed-fines count, per-grain volume error, PSD error,
contact-network error, permeability sensitivity. **Dice alone hides merged-ID
and contact errors** — report the instance metrics.

## 9. Fines (decision already taken)
Fines must be quantitatively accurate, but ≈4 vox is below the floor → **do not
super-resolve**. Acquire a high-res static scan (~10–12 µm; cf. Sand Atlas
Hamburg 11 µm) of a representative subvolume for fines PSD/shape, and use it to
calibrate the coarse in-situ data.
```
```
## Build order
A (done) Stages 0–3 classical → choose ROI from baseline → B encode Sand Atlas +
ROI → run released ParticleSeg3D inference → fine-tune → fallback if needed.
