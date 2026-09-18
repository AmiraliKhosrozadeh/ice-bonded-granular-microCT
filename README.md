# Grain-scale kinematics, ice-bond failure and damage evolution in ice-bonded granular media under in-situ uniaxial compression

Analysis code, per-specimen tables and supplementary material for the paper of
that title (Khosrozadeh et al., submitted to *Powder Technology*).  Twelve
ice-bonded columns, five of glass beads (G1–G5), four of γ-alumina beads
(A1–A4) and three of quartz sand (S1–S3), were compressed at −5 °C inside a
laboratory micro-CT and reconstructed at every load step at 24.77 µm per
voxel.  The reconstructions themselves are available from the corresponding
author on request; everything computed from them is here.

The repository is organised by what each analysis takes in, which is how
Fig. 1 of the paper draws it.

```
stage0_dragonfly/        per-scan segmentation run inside Dragonfly: phase classification,
                         watershed bead labels, histogram / porosity / PSD checks, ROI export
spam_pipeline/           first-generation SPAM pipeline (alignment, ddic, ldic, contacts,
                         bonds, crack analysis, tortuosity, plots) as first written; the
                         paper's numbers come from analysis/ and paper/, which supersede it
postprocessing/          first-generation post-processing (sparse graphs, permeability)

analysis/                the pipeline the paper reports, one folder per branch of Fig. 1
  kinematics/            branch A: bead DDIC (spam-ddic), composed 1->N fields, neighbourhood
                         gradient, Green-Lagrange strain, packing rotation, motion renders
  sand_tracking/         branch A for sand: predictor from column length, reciprocal
                         matching, grain strain, damage criteria and their nulls
  bonds/                 branch B: throat lens, neck, coverage pair, cohesive / adhesive
                         classification, survival census, breakage clustering null test;
                         _spec.py is the registry of every specimen's scans, frames and
                         thresholds that the other scripts import
  damage/                branch C: the pipeline's void / crack / separation classes and the
                         crack-orientation classification (the volumes the paper reports
                         come from paper/scripts/crack_split_tight.py, see below)
  structure/             branches C and D: fixed core, phase fractions, voxel-face specific
                         surface, 26-connected spanning air, Kozeny-Carman estimate, PuMA
                         ice-diffusion tortuosity
  sand_segmentation/     phase segmentation of the sand and, under dl/, the border-core 3D
                         U-Net (synthetic training data, training, inference, decoding) with
                         the five-way comparison against watershed, ParticleSeg3D and
                         Cellpose-SAM

paper/
  scripts/               the scripts that make every table and figure of the paper from the
                         outputs of analysis/; in particular
                           crack_split_tight.py   void / crack / separation inside the bead
                                                  envelope, the classes the paper reports
                           planar_test.py         breakage-clustering null test
                           ice_tortuosity.py      PuMA continuum diffusion on interior cubes
                           specimen_tables.py     Tables 2-4 of the paper
                           fig_*.py               the figures
                           acquisition_table.py   per-scan scanner record (Table S2)
                         review_loop_2026-09-18.md is the log of the ten-round
                         expert review of the manuscript and what it changed
  data/                  the tables behind the paper: bond geometry, failure mode at every
                         threshold, survival census, specimen summary, crack volumes, core
                         structure, ice tortuosity per cube, DIC summary, clustering sweeps,
                         loading rates, and under bonds/ the per-throat, per-pair and
                         per-bead tables of every bead specimen
supplementary/           supplementary.pdf and its figures: method settings and provenance,
                         crack classes, crack orientation and ice-path maps for every scan
data/                    the intermediate data the tables are built from: spam-ddic outputs,
                         composed kinematic fields, sand tracking outputs, crack-class masks,
                         Camsizer references (see data/README.md)
templates/               per-specimen template documenting how the scripts are adapted
```

## Results for every specimen

The paper shows one or two specimens per figure.  The same rendering for
every specimen and every scan is in [`supplementary/`](supplementary/README.md):
the void and crack classes (Fig. 11 of the paper), the crack orientation
classes (Fig. 12), the ice-path maps (Fig. 18) and the per-scan tables behind
Tables 2–4, each as one sheet per material, compiled with their text in
[`supplementary/supplementary.pdf`](supplementary/supplementary.pdf).

## Software

SPAM 0.9 (`spam-ddic`), pumapy 3.2.2, scikit-image, scikit-learn, scipy,
PyVista for the renders, PyTorch for the U-Net.  Paths at the top of each
script point at the reconstruction folders of the authors' machine.  Run the
WSL stages sequentially; each loads several GB of aligned TIFFs.

## Method settings

`supplementary/supplementary.pdf`, Table S1, lists for every analysis its
input, output, software and the settings that fix the result.

## What is not here

The reconstructions (`*.tif` stacks), the aligned and intermediate volumes and
the per-specimen result folders.  Only the code and the tables and figures
derived from them are versioned.

## Citation

Khosrozadeh A., Sinnwell Y., Pietsch-Braune S., Nikolaus K., Antonyuk S.,
Heinrich S.  Grain-scale kinematics, ice-bond failure and damage evolution in
ice-bonded granular media under in-situ uniaxial compression.  Submitted.
