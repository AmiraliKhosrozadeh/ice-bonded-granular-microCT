"""
Stage 4 - prepare ParticleSeg3D input (TIFF slices + metadata.json).

Writes cropped TIFF slices for a chosen subvolume into the ParticleSeg3D
layout, ready for `ps3d_tiff2zarr` (which builds the zarr in the exact format
the model expects -- safer than hand-writing zarr v3):

    ps3d_data/
      metadata.json
      tiff/<sample>/<sample>_####.tif     (slices -> ps3d_tiff2zarr -> images/<sample>.zarr)

metadata.json carries spacing (mm/vox) and particle_size (mm mean diameter).

First run uses a SUBVOLUME to validate the pipeline + 8 GB VRAM/runtime before
the full stack. Set SUBVOL=None below (or env PS3D_FULL=1) for the full volume.

Run in the WSL venv:
    ~/ps3d/bin/python prepare_ps3d_input.py
"""
import os
import sys
import json
import glob
import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG   # noqa: E402

PARTICLE_SIZE_MM = 0.25   # rough mean grain diameter (Camsizer d50 ~245 um)

# Subvolume in CROPPED-volume voxel coords (z0,z1,y0,y1,x0,x1), inclusive.
# Small chunk in the dense-contact region: ParticleSeg3D upsamples our ~10-vox
# grains to its trained 60-px target (~6x), so volume balloons ~216x -> keep the
# source small so the resampled run is tractable on the 8 GB 3070. None = full.
SUBVOL = (450, 545, 400, 559, 360, 519)   # 96 x 160 x 160 dense-sand (40% sand, 93% in-mask)
if os.environ.get("PS3D_FULL") == "1":
    SUBVOL = None


def wsl_path(p):
    if sys.platform.startswith("win") or len(p) < 2 or p[1] != ":":
        return p
    return f"/mnt/{p[0].lower()}/" + p[2:].replace("\\", "/").lstrip("/")


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ps3d_data")
    name = CFG.sample_name + ("_sub" if SUBVOL else "_full")
    tiff_out = os.path.join(root, "tiff", name)
    os.makedirs(tiff_out, exist_ok=True)
    for f in glob.glob(os.path.join(tiff_out, "*.tif")):
        os.remove(f)

    fs = sorted(glob.glob(os.path.join(wsl_path(CFG.tiff_dir), f"{CFG.file_prefix}*.tif")))
    zc0, zc1 = (0, len(fs) - 1) if CFG.z_range is None else CFG.z_range
    zc1 = min(zc1, len(fs) - 1)
    x0c, x1c, y0c, y1c = CFG.xy_crop if CFG.xy_crop else (0, 1 << 30, 0, 1 << 30)

    if SUBVOL:
        sz0, sz1, sy0, sy1, sx0, sx1 = SUBVOL
    else:
        sz0, sz1, sy0, sy1, sx0, sx1 = 0, zc1 - zc0, 0, 1 << 30, 0, 1 << 30

    print(f"writing slices for {name}  subvol={SUBVOL} ...")
    written = 0
    for k in range(sz0, sz1 + 1):
        gi = zc0 + k
        if gi > zc1:
            break
        im = tifffile.imread(fs[gi])
        im = im[y0c:y1c + 1, x0c:x1c + 1]            # apply specimen crop
        im = im[sy0:sy1 + 1, sx0:sx1 + 1]            # apply subvolume crop
        tifffile.imwrite(os.path.join(tiff_out, f"{name}_{written:04d}.tif"),
                         im.astype(np.uint16))
        written += 1
    print(f"  wrote {written} slices  ({im.shape[0]}x{im.shape[1]} each) -> {tiff_out}")

    # z-score (mean,std) of this subvolume's solid region for -z normalization
    # (the model default 5850/7078 are its TRAINING stats; ours differ).
    allv = np.stack([tifffile.imread(p) for p in
                     sorted(glob.glob(os.path.join(tiff_out, "*.tif")))]).astype(np.float32)
    solid = allv[allv > 2095]   # exclude outside/void air (t_air_ice)
    print(f"  ZSCORE (solid mean,std): {solid.mean():.1f} {solid.std():.1f}")

    meta = {name: {"spacing": float(CFG.voxel_size_um) / 1000.0,
                   "particle_size": PARTICLE_SIZE_MM}}
    with open(os.path.join(root, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  metadata.json: {meta}")
    print(f"\nNEXT (WSL venv):")
    print(f"  ps3d_tiff2zarr -i {tiff_out} -o {os.path.join(root,'images')}")
    print(f"  ps3d_inference -i {root} -o {os.path.join(root,'predictions')} "
          f"-m <model>/Task310_particle_seg -n {name} -f 0 -batch_size 1")


if __name__ == "__main__":
    main()
