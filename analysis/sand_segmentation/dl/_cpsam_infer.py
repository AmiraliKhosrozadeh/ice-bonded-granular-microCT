"""Cellpose-SAM (5th model) -- inference + post in one step.

Runs the Cellpose-SAM generalist model ZERO-SHOT on the SAME specimen crop the
other models use (gray_slices/ + sand_mask.npz produced by _ps3d_prep.py), so the
only thing that differs is the grain-splitting paradigm: Cellpose predicts flow
fields + gradient tracking (then links 2D masks across z by IoU stitching) rather
than the border-core representation used by our U-Nets and ParticleSeg3D.

Instances are restricted to the operator sand mask (np.where(sand, lab, 0)); no
connected-component relabel, so Cellpose's instance identity is preserved
(merges stay merged, over-splits stay split) -- faithful to its behavior.

Writes <out>/dl_grain_labels_full.tif (uint32) for _psd_feret.py.
"""
import os, sys, json, argparse, time
import numpy as np, tifffile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_fullvol_hybrid import load_volume

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True, help='cmp_models/<spec>__cpsam (prep already ran here)')
ap.add_argument('--diam', type=float, default=9.0, help='mean grain diameter in voxels (D50 minFeret ~218um / 24.766um ~= 8.8)')
ap.add_argument('--flow_threshold', type=float, default=0.4)
ap.add_argument('--cellprob_threshold', type=float, default=0.0)
ap.add_argument('--stitch_threshold', type=float, default=0.1, help='IoU to link 2D masks into 3D instances across z')
ap.add_argument('--batch_size', type=int, default=8)
a = ap.parse_args()

print(f"loading crop from {a.out}/gray_slices ...", flush=True)
vol = load_volume(os.path.join(a.out, 'gray_slices'), 0, None)
sand = np.load(os.path.join(a.out, 'sand_mask.npz'))['sand']
if vol.shape != sand.shape:
    raise SystemExit(f"shape mismatch vol{vol.shape} vs sand{sand.shape}")
print(f"  vol {vol.shape} dtype {vol.dtype}; sand_vox {int(sand.sum()):,}", flush=True)

from cellpose import models, core
import torch
gpu = torch.cuda.is_available()
print(f"GPU available: {gpu}; cellpose loading cpsam model ...", flush=True)
model = models.CellposeModel(gpu=gpu)

t0 = time.time()
# 3D via 2D+stitch: pass (Z,Y,X), z_axis=0, do_3D=False, stitch_threshold>0.
# Cellpose-SAM rescales each slice so grains ~= its native 30px scale, then tiles.
masks, flows, styles = model.eval(
    vol,
    z_axis=0,
    do_3D=False,
    stitch_threshold=a.stitch_threshold,
    diameter=a.diam,
    flow_threshold=a.flow_threshold,
    cellprob_threshold=a.cellprob_threshold,
    batch_size=a.batch_size,
    normalize=True,
)
lab = np.asarray(masks)
print(f"  cellpose done in {time.time()-t0:.0f}s; raw instances {int(lab.max())}", flush=True)

n_raw = int(lab.max())
lab = np.where(sand, lab, 0).astype(np.uint32)
n_kept = len(np.unique(lab)) - 1
out_tif = os.path.join(a.out, 'dl_grain_labels_full.tif')
tifffile.imwrite(out_tif, lab)
print(f"CPSAM_DONE raw_instances={n_raw} sand_masked_instances={n_kept} -> {out_tif}", flush=True)
