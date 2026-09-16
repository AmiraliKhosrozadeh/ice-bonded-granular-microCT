"""PuMA SSA + tortuosity (150^3 REV) for ALL 7 load stages. Saves incrementally."""
import os, sys, json, time, numpy as np, tifffile
import pumapy as puma
VOX_M = 24.7660229e-6; CUBE = int(sys.argv[1]) if len(sys.argv) > 1 else 150
STAGES = [
    ("100_500_T5_01", "results_v2_100_500_T5_01"), ("100_500_T5_02", "results_v2_100_500_T5_02"),
    ("25mm_T5_01", "results_v2_25mm_T5_01"), ("25mm_T5_02", "results_v2_25mm_T5_02"),
    ("25mm_T5_03", "results_v2_25mm_T5_03"),
    ("75_200_T5_01", "results_75_200_T5_01"), ("75_200_T5_02", "results_75_200_T5_02"),
]
OUT = "puma_out"; os.makedirs(OUT, exist_ok=True)
fp = os.path.join(OUT, f"props_all_{CUBE}.json")
rows = json.load(open(fp)) if os.path.exists(fp) else []
done = {r["stage"] for r in rows}
for name, d in STAGES:
    if name in done:
        print(f"skip {name} (done)", flush=True); continue
    lab = tifffile.imread(os.path.join(d, "dl_grain_labels_full.tif")); solid = lab > 0
    c = np.array(np.nonzero(solid)).mean(1).astype(int); h = CUBE // 2
    sl = tuple(slice(max(0, c[i]-h), min(solid.shape[i], max(0, c[i]-h)+CUBE)) for i in range(3))
    sub = np.ascontiguousarray(solid[sl]).astype(np.uint8)
    ws = puma.Workspace.from_array(sub); ws.voxel_length = VOX_M
    vf_pore = puma.compute_volume_fraction(ws, (0, 0), display=False)
    sa = puma.compute_surface_area(ws, (1, 1)); area, ssa = (sa if isinstance(sa, (tuple, list)) else (sa, np.nan))
    tau = {}
    t = time.time()
    for dn, ax in (("x", 0), ("y", 1), ("z", 2)):
        r = puma.compute_continuum_tortuosity(ws, (0, 0), dn, side_bc="s", tolerance=1e-4, display_iter=False)
        tau[dn] = float(r[0][ax])
    rows.append({"stage": name, "cube": list(sub.shape), "porosity_REV": float(vf_pore),
                 "specific_surface_area_1_per_m": float(ssa), "tortuosity_xyz": tau})
    json.dump(rows, open(fp, "w"), indent=2)   # save after each stage
    print(f"{name}: poro {vf_pore:.3f} SSA {ssa:.3e} tau {tau['x']:.2f}/{tau['y']:.2f}/{tau['z']:.2f} ({time.time()-t:.0f}s)", flush=True)
print("PROPS_ALL_DONE", flush=True)
