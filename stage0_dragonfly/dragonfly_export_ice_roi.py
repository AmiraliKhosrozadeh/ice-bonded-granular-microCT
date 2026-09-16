"""Run inside Dragonfly's Python console to export the ROI_Ice mask
as a binary TIF for the WSL sparse-graph pipeline.

Looks for an ROI named "ROI_Ice" (created by the segmentation script)
and writes it next to the SPAM data folder.

Adapt OUT_TIF for each scan (scan01 / scan02 / scan03) before running.

Usage:
    exec(open(r"E:\\RPTU-images\\CT_images\\Glass\\<SPECIMEN>\\scan{N}_dragonfly\\dragonfly_export_ice_roi.py", encoding='utf-8').read(), globals())
"""
import numpy as np

# Edit per scan
OUT_TIF = r'E:\RPTU-images\CT_images\Glass\<SPECIMEN>\<SPAM>\data\ice_mask_scan01.tif'
TARGET_ROI_NAME = 'ROI_Ice'

# --- find the ROI ---
from ORSModel import ROI
cls = ROI.getClassNameStatic()
roi_obj = None
for r in ROI.getAllObjectsOfClass(cls) or []:
    title = ''
    for tm in ('getTitle', 'getPrivateTitle', 'getName'):
        if hasattr(r, tm):
            try:
                title = getattr(r, tm)() or ''
                if title and 'orsObj' not in str(title):
                    break
            except Exception:
                pass
    if TARGET_ROI_NAME in str(title):
        roi_obj = r
        print(f'  found ROI: {title}')
        break
if roi_obj is None:
    raise RuntimeError(f'ROI "{TARGET_ROI_NAME}" not found in current Dragonfly session')

# --- ROI -> NumPy ---
arr = roi_obj.getNDArray()  # (Z, Y, X), uint8 0/1
# Some Dragonfly versions return 0/1 and some 0/255 — normalise.
arr = (np.asarray(arr) > 0).astype(np.uint8)
print(f'  shape ZYX = {arr.shape}, ones = {int(arr.sum()):,}')

# --- save TIF ---
import tifffile, os
os.makedirs(os.path.dirname(OUT_TIF), exist_ok=True)
# Dragonfly bundles an older tifffile. Try the new compression= kwarg
# first; fall back to the old compress= kwarg; if neither works, write
# uncompressed (file is larger but always succeeds).
try:
    tifffile.imwrite(OUT_TIF, arr, photometric='minisblack',
                     compression='zlib')
except TypeError:
    try:
        tifffile.imwrite(OUT_TIF, arr, photometric='minisblack', compress=6)
    except TypeError:
        tifffile.imwrite(OUT_TIF, arr, photometric='minisblack')
print(f'  wrote {OUT_TIF}')
