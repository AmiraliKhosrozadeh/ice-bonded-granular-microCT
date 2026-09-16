# Sand (sand / ice / air) micro-CT segmentation pipeline

Instance-segmentation pipeline for **frozen / ice-bonded sand** micro-CT, built
to give full downstream parity with the glass/ice `pipeline_github`
(PSD/shape, inter-grain contacts, pore/permeability, SPAM ddic/strain) while
replacing only the **grain-labeling core** with a method that works on angular,
polydisperse, touching sand grains.

## Why this is different from the glass/alumina pipeline
Glass beads are large, spherical, monodisperse → threshold + sphere-watershed
works. Sand here is **angular and fine**: voxel = 24.77 µm, grains 100–500 µm,
so grains are only **4–20 voxels** (fines ≈ 4 vox). Phase (solid/pore) is easy;
**separating touching grains is the hard part**, and fines sit near the
~3-voxel detection floor of every method.

## Pipeline stages
| Stage | Script | GPU | Output |
|-------|--------|-----|--------|
| 0 Scan audit | `stage0_audit.py` | no | histogram, phase peaks, CNR, fines-risk report |
| 1 Phase segmentation | `stage1_phase_seg.py` | no* | air/ice/sand label stack + sand mask |
| 2 Sand-mask cleanup | `stage2_sand_mask.py` | no | clean sand-only mask |
| 3 Watershed baseline | `stage3_watershed_baseline.py` | no | grain labels + PSD + QC (the classical benchmark + pseudo-labels) |
| 4 **ParticleSeg3D** | (see `DESIGN_particleseg3d.md`) | yes | DL grain instances (border–core nnU-Net) |
| 5 Contact-film nnU-Net | (fallback) | yes | grain instances where ParticleSeg3D merges |
| 6 Omnipose fines branch | (experiment) | yes | fines-rich ROI rescue |
| 7 Label fusion | (todo) | no | merged best-confidence labels |
| 8 Temporal tracking | SPAM discrete-DVC / ID-Track | no | stable grain IDs across load steps |
| 9 Downstream | reuse glass SPAM stack | no | PSD, contacts, permeability, strain |

\*Stage 1 is thresholding now; swap `segment_phases()` for a 3-class nnU-Net
(air/ice/sand) later — signature unchanged, rest of pipeline untouched.

## Run (Stages 0–3, no GPU)
```
cd E:\RPTU-images\CT_images\Sand\pipeline
python stage0_audit.py            # review results/<sample>/stage0_audit/
python stage3_watershed_baseline.py   # chains stage1 -> stage2 -> stage3
```
Everything is driven by `config.py` (paths, voxel size, thresholds, subvolume
crop). Set `z_range=None, xy_crop=None` there to process the full stack once a
subvolume looks right.

## Key findings on `Sand_100_500_T5_01` (subvolume, slices 440–560)
- Phases ARE separable in bulk: air 646 / ice 3531 / sand 6995 gray; ice↔sand
  **CNR ≈ 3.8** (thresholdable).
- BUT the ice/sand boundary **drifts** (audit 5253 vs mid-slab 3802) because
  partial-volume sand-grain edges overlap the ice band → ice over-counted,
  sand grains shrunk. **This is the case the 3-class nnU-Net must fix.**
- Classical watershed baseline: **2804 grains**, PSD d10/d50/d90 = 134/227/484 µm
  (inside the 100–500 µm sieve band — good sanity check), but **19 % of grains
  fall below the ~149 µm fines floor** and are unreliable.

## Honest limitations (carry forward)
- **Fines (≈100 µm = 4 vox)** are below the reliable instance floor. No
  super-resolution fix is trustworthy for quantitative PSD/contacts — plan a
  **high-res static rescan (~10–12 µm)** of a representative subvolume.
- Watershed over/under-segments angular grains (see `baseline_qc.png`); use its
  labels as pseudo-labels + benchmark, **not** as final results.

## Outputs are SPAM-ready
`stage3_baseline/grain_labels.tif` is an integer label volume (one id per grain)
in the format SPAM's discrete-DVC / label toolkit consumes directly.
