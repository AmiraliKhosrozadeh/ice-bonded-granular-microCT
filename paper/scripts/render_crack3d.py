"""3D renders of the void / surface crack / body crack classes written by
crack_split_tight.py (SAVE_CLASSES=1), same camera, colours and frame size as
the pipeline's renders and rerender_g3_tight.py, so fig_crack3d.py can read
them from results_voidcrack/tight2/.

    python scripts/render_crack3d.py G3
"""
import os
import sys

import numpy as np
import pyvista as pv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crack_split_tight import REG

pv.OFF_SCREEN = True
OPACITY_SURFACE = 0.18
C_VOID, C_SURF, C_BODY = "#4fc3f7", "#ff8f1f", "#e00000"


def surf_of(mask, n_iter=15):
    g = pv.ImageData(dimensions=np.array(mask.shape) + 1)
    g.cell_data["v"] = mask.flatten(order="F").astype(np.uint8)
    s = g.threshold(0.5).extract_surface()
    return s.smooth(n_iter=n_iter) if s.n_points else s


pid = sys.argv[1]
tag, spam, scans = REG[pid]
D = f"{spam}/results_voidcrack/tight2"
for s in scans:
    z = np.load(f"{D}/classes_scan{s:02d}.npz")
    smp, vd, a, b = z["sample"], z["void"], z["surface"], z["body"]
    shape = smp.shape
    p = pv.Plotter(off_screen=True, window_size=(2617, 4187))
    p.set_background("white")
    p.add_mesh(surf_of(smp, 30), color="#90a4ae", opacity=0.07, smooth_shading=True)
    layers = [(a, C_SURF, OPACITY_SURFACE), (b, C_BODY, 1.0)]
    if s == scans[0]:
        layers.insert(0, (vd, C_VOID, 0.12))
    for m, col, op in layers:
        sf = surf_of(m)
        if sf.n_points:
            p.add_mesh(sf, color=col, opacity=op, smooth_shading=True)
    c = np.array(shape) / 2.0
    p.camera_position = [(c[0], c[1] - 2.7 * max(shape), c[2]), (c[0], c[1], c[2]), (-1, 0, 0)]
    p.camera.azimuth = 25
    out = f"{D}/voidcrack3d_scan{s:02d}.png"
    p.screenshot(out)
    p.close()
    print("wrote", out, flush=True)
