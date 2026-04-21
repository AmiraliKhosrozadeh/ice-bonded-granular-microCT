"""
Pipeline visualization figure (like Sadeq et al. 2024, Fig. 2).
Shows: Raw CT -> Filtered -> Segmented -> Labeled beads
Run AFTER segmentation.py AND bead_analysis.py (needs seg + bead_labels in memory).
"""

import numpy as np
import os, sys
sys.path.insert(0, r'E:\RPTU-images\CT_images\Glass\Glass_dragonfly')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_style import apply_style, FS, style_axes, save_fig

OUT_DIR = r'E:\RPTU-images\CT_images\Glass\Glass_dragonfly\results'
os.makedirs(OUT_DIR, exist_ok=True)

# Check arrays exist
for name in ['volume', 'seg', 'bead_labels']:
    if name not in dir():
        raise RuntimeError(f"'{name}' not in memory. Run segmentation.py and bead_analysis.py first.")

print("=" * 60)
print("Pipeline Visualization Figure")
print("=" * 60)

# Labels
LABEL_OUTSIDE_AIR = 0
LABEL_TRAPPED_AIR = 1
LABEL_ICE         = 2
LABEL_GLASS       = 3
LABEL_ALUMINUM    = 4

CMAP = {
    LABEL_OUTSIDE_AIR: [0.05, 0.05, 0.05],
    LABEL_TRAPPED_AIR: [1.0,  0.2,  0.2 ],
    LABEL_ICE:         [0.4,  0.7,  1.0 ],
    LABEL_GLASS:       [0.9,  0.85, 0.6 ],
    LABEL_ALUMINUM:    [0.8,  0.8,  0.8 ],
}

def seg_to_rgb(slice_2d):
    rgb = np.zeros((*slice_2d.shape, 3), dtype=np.float32)
    for lbl, color in CMAP.items():
        rgb[slice_2d == lbl] = color
    return rgb

mid_z = seg.shape[0] // 2
n_beads = bead_labels.max()

# Bead label colors
np.random.seed(42)
bead_colors = np.random.rand(n_beads + 1, 3)
bead_colors[0] = [0.05, 0.05, 0.05]
for i in range(1, len(bead_colors)):
    bead_colors[i] = np.clip(bead_colors[i] * 1.4, 0, 1)

# -- Create figure: 2 rows x 3 columns ----------------------------------------
_FS = 22
plt.rcParams.update({
    "font.size": _FS, "axes.titlesize": _FS, "axes.labelsize": _FS - 4,
    "xtick.labelsize": _FS - 6, "ytick.labelsize": _FS - 6,
    "axes.linewidth": 1.8, "figure.facecolor": "white", "axes.facecolor": "white",
})

fig, axes = plt.subplots(2, 3, figsize=(22, 14))

# Row 1: XY slice at mid_z
# (a) Raw CT
raw_slice = volume[mid_z]
axes[0, 0].imshow(raw_slice, cmap='gray', origin='upper', aspect='equal')
axes[0, 0].set_title('(a) Raw CT', fontweight='bold', pad=10)

# (b) Segmented (phase colors)
axes[0, 1].imshow(seg_to_rgb(seg[mid_z]), origin='upper', aspect='equal')
axes[0, 1].set_title('(b) Segmented', fontweight='bold', pad=10)

# (c) Labeled beads
bead_rgb = bead_colors[bead_labels[mid_z]]
axes[0, 2].imshow(bead_rgb, origin='upper', aspect='equal')
axes[0, 2].set_title(f'(c) Labeled beads (n={n_beads})', fontweight='bold', pad=10)

# Row 2: XZ slice at mid_y
mid_y = seg.shape[1] // 2

# (d) Raw CT XZ
axes[1, 0].imshow(volume[:, mid_y, :], cmap='gray', origin='upper', aspect='equal')
axes[1, 0].set_title('(d) Raw CT — XZ', fontweight='bold', pad=10)
axes[1, 0].set_ylabel('Z (voxels)')

# (e) Segmented XZ
axes[1, 1].imshow(seg_to_rgb(seg[:, mid_y, :]), origin='upper', aspect='equal')
axes[1, 1].set_title('(e) Segmented — XZ', fontweight='bold', pad=10)

# (f) Labeled beads XZ
bead_rgb_xz = bead_colors[bead_labels[:, mid_y, :]]
axes[1, 2].imshow(bead_rgb_xz, origin='upper', aspect='equal')
axes[1, 2].set_title('(f) Labeled beads — XZ', fontweight='bold', pad=10)

# Style all axes
for ax in axes.flat:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.8)
    ax.set_xlabel('X (voxels)')

axes[0, 0].set_ylabel('Y (voxels)')
axes[1, 0].set_ylabel('Z (voxels)')

# Legend for segmentation colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=CMAP[LABEL_OUTSIDE_AIR], edgecolor='gray', label='Outside air'),
    Patch(facecolor=CMAP[LABEL_TRAPPED_AIR], edgecolor='gray', label='Trapped air'),
    Patch(facecolor=CMAP[LABEL_ICE],         edgecolor='gray', label='Ice'),
    Patch(facecolor=CMAP[LABEL_GLASS],       edgecolor='gray', label='Glass beads'),
    Patch(facecolor=CMAP[LABEL_ALUMINUM],    edgecolor='gray', label='Aluminum'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5,
           fontsize=_FS - 4, frameon=True, framealpha=0.85, edgecolor='gray',
           bbox_to_anchor=(0.5, -0.04))

fig.suptitle('CT Image Processing Pipeline', fontsize=_FS + 4, fontweight='bold', y=1.01)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'pipeline_overview.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
plt.rcdefaults()
print(f"  Saved: {out_path}")
print("Done.")
