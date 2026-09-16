"""Re-render G3's void/crack scans 3 and 4 inside a tighter specimen mask.

Under load the pipeline's specimen mask runs out to the tube bore, so at the
later scans it takes in the annulus and the platen-side slabs and draws crack
there as if it were specimen.  Here the specimen is taken as the beads
themselves.  Per slice the bead labels are closed with a disc a little larger
than a bead radius, which merges neighbouring beads into one solid section
bounded by the outermost beads, holes are filled, the result is dilated by a
few voxels to keep the ice skin on the outer beads, and the pipeline's mask is
cut down to it.  Everything else (crack classes, skin depth, camera, colours)
is the pipeline's own.

Writes voidcrack3d_scanNN.png into results_voidcrack/tight/ and prints the
class volumes the tighter mask gives, so they can be set against the
pipeline's summary.
"""
import os
import sys

import numpy as np
import tifffile
from scipy import ndimage as ndi
import pyvista as pv

pv.OFF_SCREEN = True

SPAM = "E:/RPTU-images/CT_images/Glass/Glass_100_1700_T7/Glass_1700_spam"
D = os.path.join(SPAM, "results_voidcrack")
OUT = os.path.join(D, "tight")
os.makedirs(OUT, exist_ok=True)

VOXEL_UM = 24.7660229
VOX_MM3 = (VOXEL_UM / 1000.0) ** 3
SKIN, STEP = 20, 2                    # as in render_voidcrack_3d.py
R_CLOSE, R_SKIN = 45, 24              # voxels; bead radius is about 36, the ice
                                      # around the outermost beads reaches ~0.6 mm
OPACITY_SURFACE = 0.18
C_VOID, C_SURF, C_BODY = "#4fc3f7", "#ff8f1f", "#e00000"

SCANS = [int(x) for x in sys.argv[1:]] or [1, 2, 3, 4]


def tight_mask(labels):
    """Solid per-slice envelope of the beads, plus a thin skin."""
    out = np.zeros(labels.shape, dtype=bool)
    for z in range(labels.shape[0]):
        beads = labels[z] > 0
        if beads.sum() < 50:
            continue
        dil = ndi.distance_transform_edt(~beads) <= R_CLOSE
        clo = ndi.distance_transform_edt(dil) > R_CLOSE
        clo = ndi.binary_fill_holes(clo)
        out[z] = ndi.distance_transform_edt(~clo) <= R_SKIN
    return out


def surf_of(mask, n_iter=15):
    g = pv.ImageData(dimensions=np.array(mask.shape) + 1)
    g.cell_data["v"] = mask.flatten(order="F").astype(np.uint8)
    s = g.threshold(0.5).extract_surface()
    return s.smooth(n_iter=n_iter) if s.n_points else s


shape = None
for s in SCANS:
    sample = tifffile.imread(os.path.join(D, f"sample_scan{s:02d}.tif")) > 0
    crack = tifffile.imread(os.path.join(D, f"crack_scan{s:02d}.tif")) > 0
    vp = os.path.join(D, f"void_scan{s:02d}.tif")
    void = tifffile.imread(vp) > 0 if os.path.exists(vp) else np.zeros_like(crack)
    labels = tifffile.imread(os.path.join(SPAM, "data",
                                          f"bead_labels_scan{s:02d}_aligned.tif"))
    tight = tight_mask(labels)
    del labels
    before = sample.sum() * VOX_MM3
    sample &= tight
    del tight
    core = ndi.binary_erosion(sample, iterations=SKIN)
    crack &= sample
    void &= sample
    cb, cs = crack & core, crack & ~core
    print(f"scan {s}: sample {before:7.1f} -> {sample.sum()*VOX_MM3:7.1f} mm3   "
          f"surface {cs.sum()*VOX_MM3:7.2f}   body {cb.sum()*VOX_MM3:7.2f} mm3",
          flush=True)

    smp = sample[::STEP, ::STEP, ::STEP]
    a = cs[::STEP, ::STEP, ::STEP]
    b = cb[::STEP, ::STEP, ::STEP]
    vd = void[::STEP, ::STEP, ::STEP]
    shape = smp.shape
    # the pipeline saved its frames at 2617 px wide; same camera, same size
    p = pv.Plotter(off_screen=True, window_size=(2617, 4187))
    p.set_background("white")
    p.add_mesh(surf_of(smp, 30), color="#90a4ae", opacity=0.07, smooth_shading=True)
    layers = [(a, C_SURF, OPACITY_SURFACE), (b, C_BODY, 1.0)]
    if s == 1:                      # void is pre-existing; drawn on the reference scan only
        layers.insert(0, (vd, C_VOID, 0.12))
    for m, col, op in layers:
        sf = surf_of(m)
        if sf.n_points:
            p.add_mesh(sf, color=col, opacity=op, smooth_shading=True)
    c = np.array(shape) / 2.0
    p.camera_position = [(c[0], c[1] - 2.7 * max(shape), c[2]),
                         (c[0], c[1], c[2]), (-1, 0, 0)]
    p.camera.azimuth = 25
    out = os.path.join(OUT, f"voidcrack3d_scan{s:02d}.png")
    p.screenshot(out)
    p.close()
    print("   wrote", out, flush=True)
    del sample, crack, void, core, cb, cs
print("done.")
