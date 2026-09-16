# Intermediate data behind the paper

Everything derived from the reconstructions that the paper's tables and
figures are built from.  The reconstructions themselves (tif stacks, phase
maps, label images) are not versioned; they are available from the
corresponding author on request.

```
ddic/<specimen>/          spam-ddic output of every consecutive scan pair (per-grain Phi,
                          convergence status) as used by the paper: the input of
                          analysis/kinematics/_cumulative_all.py
kinematics/               cumulative_<specimen>.npz, the composed 1->N displacement fields at
                          scan-1 positions with the common grain set and per-grain strain,
                          the input of the kinematics tables and the motion renders
sand_tracking/            the sand tracking outputs (dvc_S*_NM/grain_track_*.npz), column
                          geometry and profiles, reciprocity and null tests, crack criteria
crack_classes/<specimen>/ classes_scanNN.npz: envelope, void, surface crack, body crack and
                          separation masks at 2x decimation, from
                          paper/scripts/crack_split_tight.py; input of the crack renders
camsizer/                 Camsizer XT (.xle) reference size distributions of the glass beads
                          (x_area) and the sand (xFe,min), the reference of Figs 6-8
```

The per-throat, per-pair and per-bead tables of the bond census are under
`paper/data/bonds/`, the per-cube tortuosity solves under
`paper/data/ice_tortuosity/`, the clustering sweeps under
`paper/data/planar_sweep/`, and the trained sand U-Net under
`analysis/sand_segmentation/dl/model/`.

```
phase_maps_bin2/          <specimen>_<scan>.npz: the three-phase map of every scan (0 outside,
                          1 air, 2 ice, 3 grain) binned 2x2x2 to 49.5 um, the input of the
                          PuMA tortuosity solve, the 26-connectivity test and the ice-path maps
```

## Full-resolution volumes

The reconstructions (24.77 um, uint16), the full-resolution phase maps and
the grain labels of all 30 scans total about 60 GB and are available from
the corresponding author on request.
