# Micro-CT pipeline for granular ice / glass-bead / alumina specimens

End-to-end image-processing and post-processing pipeline used to analyse
in-situ compression CT scans of ice-bonded glass-bead and alumina-bead
specimens. Same pipeline is applied to every specimen; only a small set
of per-specimen tunables (scan list, alignment shape, intensity
thresholds) changes from one specimen to the next.

## Folder structure

```
dragonfly_scripts/        Run inside Dragonfly (Python console).
                          Per-scan segmentation, bead labelling,
                          histogram / porosity / PSD analysis,
                          ROI -> TIFF export, optional PuMA driver.

spam_pipeline/            Run from WSL (SPAM virtualenv).
                          Alignment, ddic / ldic registration,
                          bead-bond detection, contact analysis,
                          failure-mode / crack / tortuosity / FFT
                          elasticity, plotting and STL export.

postprocessing/           Run from WSL.  Newer methods built on top of
                          the SPAM outputs:
                            permeability (Kozeny-Carman),
                            ball-and-bar sparse-graph rendering for
                            cracks / bonds / ice skeleton,
                            CT-vs-damage validation panels,
                            DBSCAN-style crack-path tracing,
                            density-vacancy crack detection.

templates/                Per-specimen CLAUDE.md template documenting
                          how to adapt scripts to a new specimen.

POSTPROCESSING_PIPELINE.md
                          Master pipeline document: prerequisites,
                          per-specimen tunables, ready-to-paste WSL
                          command list, outputs summary.
```

## Generic placeholders

Scripts use these placeholders instead of hard-coded specimen names so
they apply to any specimen:

| placeholder      | meaning                                         | example value              |
|------------------|-------------------------------------------------|----------------------------|
| `<SPECIMEN>`     | full specimen folder name                       | `Glass_75_1700_T5_HR`      |
| `<SPAM>`         | spam sub-folder under the specimen              | `Glass_75_spam`            |
| `<PRE>`          | short prefix used in script / output names      | `75`, `t5hr`, `100_1800_T5`|
| `scan{N}`        | per-scan placeholder                            | `scan1`, `scan2`, ...      |

When deploying these scripts to a new specimen, search & replace these
tokens (or use `_deploy_postprocessing_to_specimens.py` from the parent
project as a starting point). The substitutions are also documented in
the per-script header comments.

## Per-specimen tunables (set at the top of each script)

```python
SCANS        = [1, 2, 3]            # actual scan list for this specimen
TRANSITIONS  = [(1, 2), (2, 3)]     # consecutive scan pairs
NX_E, NY_E, NZ_E = 803, 707, 1241   # aligned-frame shape from <SPAM>/data/aligned_meta.py
TH = {                              # per-scan intensity thresholds
    1: dict(air=..., ig=..., ga=...),
    ...
}
```

`POSTPROCESSING_PIPELINE.md` lists the specimens this pipeline has been
applied to, with their prefixes, scan counts and spam-folder names.

## Running the pipeline

1. Stage 1 (Dragonfly, per scan) — open the scan session and run the
   per-scan segmentation script, then the ROI export.
2. Stage 2 (WSL) — run the SPAM pipeline scripts in the order listed in
   `POSTPROCESSING_PIPELINE.md`, then the post-processing scripts.

Run the WSL stages **sequentially** — each one loads several GB of
aligned TIFFs and parallel runs OOM-kill the WSL VM.

## What's NOT in this repo

- Raw CT data (`*.tif`, `*.tiff`, `*.rek`, scanner `*.PRM` files).
- Aligned and intermediate TIFF stacks.
- Per-specimen results folders (`results_*`, `*.png`, `*.csv` outputs).

The repo contains the pipeline only; the data lives outside it. See
`.gitignore` for the full exclusion list.
