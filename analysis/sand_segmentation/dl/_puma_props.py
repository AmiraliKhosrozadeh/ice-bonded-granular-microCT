"""PuMA microstructure properties per load stage (no registration needed).
Solid = v2 sand grains (labels>0); pore = ice+air. Central cube (REV) per stage.
Stage 1: porosity (volume fraction) + specific surface area.
"""
import os, sys, json, numpy as np, tifffile
import pumapy as puma
VOX_M = 24.7660229e-6
CUBE = int(sys.argv[1]) if len(sys.argv) > 1 else 150
STAGES = [("75_200_T5_01", "results_75_200_T5_01"),
          ("75_200_T5_02", "results_75_200_T5_02")]
OUT = "puma_out"; os.makedirs(OUT, exist_ok=True)
rows = []
for name, d in STAGES:
    lab = tifffile.imread(os.path.join(d, "dl_grain_labels_full.tif"))
    solid = lab > 0
    c = np.array(np.nonzero(solid)).mean(1).astype(int); h = CUBE // 2
    sl = tuple(slice(max(0, c[i]-h), min(solid.shape[i], max(0, c[i]-h)+CUBE)) for i in range(3))
    sub = np.ascontiguousarray(solid[sl]).astype(np.uint8)
    ws = puma.Workspace.from_array(sub); ws.voxel_length = VOX_M
    vf_solid = puma.compute_volume_fraction(ws, (1, 1), display=False)
    vf_pore = puma.compute_volume_fraction(ws, (0, 0), display=False)
    sa = puma.compute_surface_area(ws, (1, 1))     # (area m^2, specific area 1/m)
    area, ssa = (sa if isinstance(sa, (tuple, list)) else (sa, np.nan))
    # tortuosity of the pore phase (on-axis component of the eta vector per solve)
    tau = {}
    for dn, ax in (("x", 0), ("y", 1), ("z", 2)):
        r = puma.compute_continuum_tortuosity(ws, (0, 0), dn, side_bc="s",
                                              tolerance=1e-4, display_iter=False)
        tau[dn] = float(r[0][ax])
    rows.append({"stage": name, "cube": list(sub.shape), "solid_VF": float(vf_solid),
                 "porosity": float(vf_pore), "surface_area_m2": float(area),
                 "specific_surface_area_1_per_m": float(ssa), "tortuosity_xyz": tau})
    print(f"{name}: cube {sub.shape}  solidVF {vf_solid:.3f}  porosity {vf_pore:.3f}  "
          f"SSA {ssa:.3e} 1/m  tau {tau['x']:.2f}/{tau['y']:.2f}/{tau['z']:.2f}", flush=True)
json.dump(rows, open(os.path.join(OUT, "props_fast.json"), "w"), indent=2)
print("PUMA_PROPS_DONE", flush=True)
