"""Quartz fraction, ice fraction and saturation for all three sand specimens.

The grain labelling covers a different slice range in each specimen, but a
volume fraction does not care: the denominator is taken over exactly the
slices the labels cover, read from the labelling's own bbox json, which
indexes the same reconstruction as the phase segmentation (zmin_slice 0,
full_shape z equal to the segmentation's).  Laterally nothing is cut -- the
labelled cloud sits inside its window in every specimen.

    rho     labelled grain volume / interior volume, same slices
    eps     air / interior, same slices
    phi_ice 1 - rho - eps
    S       phi_ice / (1 - rho)

The one bias left is the fine fraction the labelling does not resolve
individually, which falls into the ice by difference.  Its size is bounded
from the Camsizer curve and printed alongside.
"""
import json
import os

import numpy as np
import tifffile

DL = r"E:/RPTU-images/CT_images/Sand/pipeline/dl"
SEG = r"E:/RPTU-images/CT_images/Alumina/pyalumina/results/_xmat/sand"
VOX = 24.7660229 / 1000.0

# paper id -> (labelling directory, segmentation spec)
SPECS = [("S1", "results_fix_100_500_T5_01", "100_500_T5"),
         ("S2", "results_25mm_T5_01", "25mm_100_500"),
         ("S3", "results_fix_75_200_T5_01", "75_200_T5")]


def grain_volume(d):
    p = os.path.join(DL, d, "dl_grain_labels_full.tif")
    n = 0
    with tifffile.TiffFile(p) as t:
        for pg in t.pages:
            n += int((pg.asarray() > 0).sum())
    return n


def interior(spec, z0, z1):
    p = os.path.join(SEG, spec, "scan01", "stage1", "phase_labels.tif")
    n_in = n_air = 0
    with tifffile.TiffFile(p) as t:
        for i in range(z0, min(z1, len(t.pages))):
            P = t.pages[i].asarray()
            n_in += int((P > 0).sum())
            n_air += int((P == 1).sum())
    return n_in, n_air


print("%-4s %6s %8s %9s %9s %7s %7s %8s %7s" %
      ("id", "slices", "grains", "V_grain", "V_int", "rho", "eps", "phi_ice", "S"))
out = {}
for pid, d, spec in SPECS:
    bb = json.load(open(os.path.join(DL, d, "dl_grain_labels_bbox.json")))
    z0, z1 = bb["bbox_zyx"][0], bb["bbox_zyx"][1]
    ng = bb["n_grains"]
    nv = grain_volume(d)
    n_in, n_air = interior(spec, z0, z1)
    Vg, Vi = nv * VOX ** 3, n_in * VOX ** 3
    rho = Vg / Vi
    eps = n_air / n_in
    phi = 1.0 - rho - eps
    S = phi / (1.0 - rho)
    out[pid] = dict(slices=z1 - z0, grains=ng, rho=rho, eps=eps, phi=phi, S=S)
    print("%-4s %6d %8d %9.1f %9.1f %7.3f %7.3f %8.3f %7.3f" %
          (pid, z1 - z0, ng, Vg, Vi, rho, eps, phi, S), flush=True)

json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "sand_rho_all.json"), "w"), indent=1)
print("\n-> sand_rho_all.json")
