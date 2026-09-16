"""
Stage 4 prep - border-core DECODER (no GPU).

Turns a 3-class border-core prediction (0 bg, 1 core, 2 border) from the
nnU-Net / ParticleSeg3D model back into an integer per-grain instance label
volume -- the SAME format as the Stage 3 watershed baseline `grain_labels.tif`,
so it drops straight into the existing SPAM / PSD / contact / validation stack.

Method: connected components on `core` give one marker per grain (touching
grains separated by the border gap); markers flood the full foreground
(core + border) via watershed on the inverse EDT.

Usage:
    python decode_instances.py border_core_pred.tif grain_labels.tif [--min 20]
"""
import argparse
import numpy as np
import tifffile
from scipy import ndimage
from skimage.segmentation import watershed


def decode(pred, min_voxels=20):
    fg = pred > 0
    markers, n = ndimage.label(pred == 1)         # cores -> seeds
    edt = ndimage.distance_transform_edt(fg).astype(np.float32)
    labels = watershed(-edt, markers=markers, mask=fg)
    # size filter + consecutive relabel
    sizes = np.bincount(labels.ravel())
    keep = np.where(sizes >= min_voxels)[0]
    keep = keep[keep != 0]
    remap = np.zeros(labels.max() + 1, np.int32)
    remap[keep] = np.arange(1, len(keep) + 1)
    return remap[labels], len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("out")
    ap.add_argument("--min", type=int, default=20, help="min grain voxels")
    a = ap.parse_args()
    pred = tifffile.imread(a.pred)
    labels, n = decode(pred, a.min)
    dt = np.uint16 if labels.max() < 65535 else np.uint32
    tifffile.imwrite(a.out, labels.astype(dt))
    print(f"decoded {n} grain instances -> {a.out}")


if __name__ == "__main__":
    main()
