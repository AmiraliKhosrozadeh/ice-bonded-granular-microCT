# Viewing the Sand pipeline results in Dragonfly

How to load every stage output (classical **and** deep-learning) into Dragonfly
and inspect it. All paths are for `Sand_100_500_T5_01`; swap the specimen folder
for others. Voxel size is **24.766 µm isotropic** — set it on every import.

> PuMA WARNING: do **not** run Dragonfly's built-in PuMA on a full CT volume —
> it hangs. Use the WSL pipeline outputs instead (see project notes).

---

## 0. The files (what to load)

| Stage | What | Path | Type / values |
|---|---|---|---|
| Raw CT | grayscale scan, full | `Sand\Sand\Sand_100_500_T5_01\Sand-100-500-T5_100XXL_uc_xy_####.tif` | 996 × 1179² uint16 |
| Stage 1 | phase labels (air/ice/sand) | `pipeline\results\Sand_100_500_T5_01\stage1_phase\phase_labels_####.tif` | uint8: 0 outside-air, 1 void-air, 2 ice, 3 sand |
| Stage 2 | clean sand mask | `...\stage2_sand\sand_clean_####.tif` | binary (0/1) |
| Stage 3 | **watershed grain labels (full)** | `...\stage3_baseline\grain_labels.tif` | one 1.5 GB multipage, uint16/32, ~64,679 grains |
| Stage 4 | **ParticleSeg3D DL grain labels (subvol)** | `...\stage4_particleseg3d\grain_labels_particleseg3d_sub.tif` | one multipage, 96×160×160, 157 labels |
| Stage 4 | raw subvol the DL model saw | `pipeline\dl\ps3d_data\tiff\Sand-100-500-T5_100XXL_uc_sub\*.tif` | 96 slices, matches the DL labels grid |

### Coordinate frames (read this before overlaying)
- Stages 1–3 run on the **cropped frame** `x 255–1182, y 0–851` (852×928). They
  overlay **each other** perfectly, but are **shifted/smaller than the raw 1179²**.
- The Stage 4 subvolume is a small crop **inside** that frame. In **raw** voxel
  coordinates it is `x 415–574, y 420–579, z 620–715` (160×160×96). Use that box
  if you ever want to crop the raw stack to sit exactly on the DL labels.

---

## 1. Import a grayscale stack

1. `File ▸ Import Image Files…` (or *Import Image Stack*).
2. Select the **first** numbered TIFF; Dragonfly auto-detects the sequence.
3. Set **pixel size X=Y=Z = 24.766 µm**, data type **16-bit unsigned**.
4. Finish → a new **Dataset/Channel** appears in the Data Properties panel.

Do this for the raw CT (full) and, separately, for the **DL raw subvol** folder.

---

## 2. Import a label / segmentation stack as a Multi-ROI

Label stacks (phase, sand mask, watershed, DL) are integer images — to inspect
individual grains/phases you want them as a **Multi-ROI**, not a grayscale image.

1. Import the stack exactly like Section 1 (same 24.766 µm spacing). For the
   single-file multipage labels (`grain_labels.tif`, `..._sub.tif`) just pick
   that one file.
2. In the data tree, **right-click the imported label channel ▸ Convert to ▸
   Multi-ROI** (menu wording varies by version: may be *“Create Multi-ROI from
   labels”* or, in the Segmentation Wizard, *“Import label field”*).
   - Each integer value becomes one ROI/object.
   - For phase labels, the four values map to air/void/ice/sand.
3. Pick a **Glasbey / Label LUT** so touching grains get distinct colors.

---

## 3. Overlay labels on the grayscale (see what was segmented)

1. Load the matching grayscale (Section 1) and the Multi-ROI (Section 2).
   **Use a matched pair from the same frame:**
   - Classical: raw-cropped (or any stage 1–3 grayscale) + `grain_labels.tif`.
   - DL: the **DL raw subvol** + `grain_labels_particleseg3d_sub.tif` (these two
     are the same 96×160×160 grid — they line up with no work).
2. In the 2D view, keep the grayscale as the base layer and the Multi-ROI as an
   overlay; drop the Multi-ROI **opacity to ~40–60 %** so you see grain edges
   against the CT.
3. Scroll Z (or use Ortho view) to check boundaries slice by slice.

---

## 4. Inspect individual grains and get numbers

With the Multi-ROI selected:
- **Object count / per-grain volume:** Multi-ROI panel ▸ *Analyze* / *Compute
  statistics* (a.k.a. *Scalar & geometric analysis*). Export the per-object
  **Volume** to CSV.
- **Equivalent diameter** for PSD: `d = (6·V/π)^(1/3)`. Either let Dragonfly’s
  *Particle/Granulometry Analysis* compute it, or export volumes and run the
  pipeline’s `stage_validate_psd.py` to get the Q3 curve vs Camsizer.
- Click any object to isolate it; use *Hide others* to follow one grain in 3D.

---

## 5. The recommended viewing session (classical vs DL, side by side)

1. Import **raw CT (full)** → grayscale base.
2. Import **`grain_labels.tif`** → Multi-ROI → overlay (Section 3). This is the
   classical watershed result over the whole plug.
3. In a second view, import the **DL raw subvol** + **`..._sub.tif`** Multi-ROI.
4. Compare: the watershed labels fill the sand densely (but **over-merge** coarse
   grains); the DL subvol currently labels only a sparse scatter — that is the
   known **coverage-collapse** issue, not a Dragonfly display problem. Expect the
   DL Multi-ROI to look nearly empty until we fix Stage 4 coverage.

---

## 6. Dragonfly Python console (optional)

The glass pipeline drives Dragonfly from its Python console via
`exec(open(r'...py').read(), globals())`. The same diagnostic works here to print
loaded channel shapes/spacing (paste the output back if a stack looks mis-scaled):

```
exec(open(r'E:\RPTU-images\CT_images\pipeline_github\dragonfly_scripts\dragonfly_channel_info.py', encoding='utf-8').read(), globals())
```

---

## Quick reference — what each stage should look like

- **phase_labels**: 3 clean phases; ice/sand boundary is where the partial-volume
  smear lives (the thing the DL is meant to fix).
- **sand_clean**: sand only, contacts preserved.
- **grain_labels (watershed)**: dense, colorful; look for a few **giant merged**
  grains in dense-contact zones (the dominant PSD error).
- **grain_labels_particleseg3d_sub (DL)**: should separate touching grains
  cleanly — but right now under-covers; treat as work-in-progress.
