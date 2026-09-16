"""
Stage 4 prep - border-core ENCODER (no GPU, no training).

Converts an integer per-grain label volume (Sand Atlas grains, or a
Dragonfly-corrected ROI) into the 3-class semantic target ParticleSeg3D /
nnU-Net trains on:
    0 = background (ice / air / pore)
    1 = core      (grain interior, away from every boundary)
    2 = border    (shell around each grain)

Why it separates touching grains: a voxel is `core` only if its whole
neighbourhood shares its label. At a grain-grain contact the neighbourhood
spans two labels -> those voxels become `border`, so the two cores stay
disconnected and decode back to two instances. Implemented with min/max
filters (fast, exact, no per-label loop).

Usage:
    python encode_border_core.py labels.tif border_core.tif [--r 2]
"""
import sys
import argparse
import numpy as np
import tifffile
from scipy import ndimage


def encode(labels, r=2):
    size = 2 * r + 1
    mn = ndimage.minimum_filter(labels, size=size)
    mx = ndimage.maximum_filter(labels, size=size)
    fg = labels > 0
    core = fg & (mn == labels) & (mx == labels)   # neighbourhood all-same-label
    out = np.zeros(labels.shape, np.uint8)
    out[fg] = 2          # border (default for any foreground voxel)
    out[core] = 1        # core overrides
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("out")
    ap.add_argument("--r", type=int, default=2,
                    help="erosion radius (vox). Use 1 for fine grains (~4 vox).")
    a = ap.parse_args()
    labels = tifffile.imread(a.labels)
    bc = encode(labels, a.r)
    tifffile.imwrite(a.out, bc)
    n = labels.max()
    print(f"encoded {n} grains -> border-core (r={a.r}): "
          f"core {int((bc==1).sum()):,} | border {int((bc==2).sum()):,} -> {a.out}")


if __name__ == "__main__":
    main()
