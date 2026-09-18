# Results for every specimen

The paper shows one or two specimens per figure; the same rendering for every
specimen and every scan is here.  Specimen labels (G1–G5 glass, A1–A4
γ-alumina, S1–S3 sand), load-step numbering and the axial convention (punch
at the top) are those of the paper.

| Paper figure | Every specimen |
|---|---|
| Fig. 11, void and crack classes inside the bead envelope | [crack3d_glass.png](crack3d_glass.png), [crack3d_alumina.png](crack3d_alumina.png); volumes in [tab_crack.tex](tab_crack.tex) |
| Fig. 12, crack orientation classes | [crackdir_glass.png](crackdir_glass.png), [crackdir_alumina.png](crackdir_alumina.png) |
| Fig. 18, ice-path detour map (extra ice path forced by the crack air of each scan) | [tau_geodesic_glass.png](tau_geodesic_glass.png), [tau_geodesic_alumina.png](tau_geodesic_alumina.png), [tau_geodesic_sand.png](tau_geodesic_sand.png); share of ice detoured by more than 0.3 mm per scan in [tab_taumap.tex](tab_taumap.tex) |
| Tables 3–5, per-scan values | ice tortuosity [tab_tau.tex](tab_tau.tex), kinematics [tab_kin.tex](tab_kin.tex), failure-mode census at every threshold [tab_fm.tex](tab_fm.tex), breakage-clustering sweep [tab_cluster.tex](tab_cluster.tex) |
| Table S1, method settings and provenance of every analysis | [settings_table.tex](settings_table.tex) |
| Table S2, acquisition record of every scan (projections, time) from the scanner | [tab_acq.tex](tab_acq.tex), made by `paper/scripts/acquisition_table.py` from the PRM files |

All of it, with the text that reads each sheet, is compiled in
[supplementary.pdf](supplementary.pdf) (source [supplementary.tex](supplementary.tex)).

The sheets are made by `paper/scripts/fig_crack3d_supp.py`,
`fig_crackdir_supp.py`, `fig_tau_geodesic.py` (with `TAU_SPECS` and
`TAU_TAG` set per material), `tau_geodesic_share.py` and `supp_tables.py`.
