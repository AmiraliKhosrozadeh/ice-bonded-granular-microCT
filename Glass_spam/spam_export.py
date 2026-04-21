"""
Stage CT volume, specimen_mask, and bead_labels as TIFFs for SPAM.

Scan 2's TIFFs are written directly by scan02_segmentation.py and
scan02_bead_analysis.py (into Glass_spam\\data\\). This script exists to do
the same for SCAN 1 without touching anything under Glass_dragonfly\\.

Usage (in Dragonfly Python console):

  1. Load scan 1 in Dragonfly.
  2. Run the scan-1 pipeline to populate `volume`, `specimen_mask`,
     `bead_labels` in the console namespace:
         exec(open(r'E:\\RPTU-images\\CT_images\\Glass\\Glass_dragonfly\\segmentation.py', encoding='utf-8').read(), globals())
         exec(open(r'E:\\RPTU-images\\CT_images\\Glass\\Glass_dragonfly\\bead_analysis.py', encoding='utf-8').read(), globals())
  3. Run this exporter:
         exec(open(r'E:\\RPTU-images\\CT_images\\Glass\\Glass_spam\\spam_export.py', encoding='utf-8').read(), globals())

Writes into Glass_spam\\data\\:
  ct_scan01.tif              uint16, filtered CT volume
  specimen_mask_scan01.tif   uint8 (0 / 255)
  bead_labels_scan01.tif     uint16 (0 = background, 1..N = bead ID)
"""

import os
import numpy as np
import tifffile

OUT_DIR = r'E:\RPTU-images\CT_images\Glass\Glass_spam\data'
os.makedirs(OUT_DIR, exist_ok=True)

needed = ['volume', 'specimen_mask', 'bead_labels']
# Check both globals() (the normal console namespace) and dir() (exec-local),
# so the script works whether it's invoked via exec(..., globals()) or imported.
_ns = dict(globals())
_ns.update({k: v for k, v in locals().items() if not k.startswith('_')})
missing = [n for n in needed if n not in _ns]
if missing:
    raise RuntimeError(
        f"Missing from console memory: {missing}. Run scan 1's "
        f"segmentation.py and bead_analysis.py first.\n"
        f"  Present names (sample): {sorted(k for k in _ns if not k.startswith('_'))[:25]}"
    )
# Rebind for the rest of the script regardless of where they came from
volume         = _ns['volume']
specimen_mask  = _ns['specimen_mask']
bead_labels    = _ns['bead_labels']

targets = [
    ('ct_scan01.tif',             volume.astype(np.uint16)),
    ('specimen_mask_scan01.tif', (specimen_mask.astype(np.uint8) * 255)),
    ('bead_labels_scan01.tif',    bead_labels.astype(np.uint16)),
]

for name, arr in targets:
    path = os.path.join(OUT_DIR, name)
    if os.path.exists(path):
        try:
            os.remove(path)
        except PermissionError:
            raise PermissionError(
                f"Cannot overwrite {path} — close any viewer holding it."
            )
    tifffile.imwrite(path, arr, photometric='minisblack')
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved: {path}  shape={arr.shape}  dtype={arr.dtype}  ({size_mb:.1f} MB)")

print("\nScan 1 TIFFs for SPAM are staged in:")
print(f"  {OUT_DIR}")
