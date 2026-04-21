"""
Histogram analysis for glass bead / ice / air CT dataset
Reads TIFF stack directly - no ORS API needed
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# ── 1. Settings (from report1.txt) ────────────────────────────────────────────
TIFF_DIR  = r'E:\RPTU-images\CT_images\Glass\Glass_100_1800_T5_01'
FILE_PREFIX = 'Glass-1001800-T5-normal_100XXL_uc_xy_'
FIRST_SLICE = 100
LAST_SLICE  = 962   # adjust if needed

# Crop applied in Dragonfly
X_MIN, X_MAX = 546, 1012
Y_MIN, Y_MAX = 96,  542

# Sample every Nth slice (faster - still accurate for histogram)
STEP = 5

RESULTS_DIR = r'E:\RPTU-images\CT_images\Glass\Glass_dragonfly\results'
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_PATH = os.path.join(RESULTS_DIR, 'histogram_analysis.png')

# ── 2. Load slices ────────────────────────────────────────────────────────────
try:
    import tifffile
    def read_tif(path):
        return tifffile.imread(path)
except ImportError:
    from PIL import Image
    def read_tif(path):
        return np.array(Image.open(path))

slices_loaded = 0
all_data = []

print("Loading slices (every {}th)...".format(STEP))
for idx in range(FIRST_SLICE, LAST_SLICE + 1, STEP):
    fname = os.path.join(TIFF_DIR, f'{FILE_PREFIX}{idx:04d}.tif')
    if not os.path.exists(fname):
        continue
    img = read_tif(fname)
    # Apply crop
    img = img[Y_MIN:Y_MAX, X_MIN:X_MAX]
    all_data.append(img.flatten())
    slices_loaded += 1

print(f"Loaded {slices_loaded} slices.")
data = np.concatenate(all_data).astype(np.float32)
print(f"Total voxels sampled : {len(data):,}")
print(f"Min gray value       : {data.min():.1f}")
print(f"Max gray value       : {data.max():.1f}")
print(f"Mean                 : {data.mean():.1f}")

# ── 3. Histogram ──────────────────────────────────────────────────────────────
N_BINS = 2000
hist, edges = np.histogram(data, bins=N_BINS,
                           range=(data.min(), data.max()))
centers = (edges[:-1] + edges[1:]) / 2.0

# ── 4. Peak detection ─────────────────────────────────────────────────────────
peaks, _ = find_peaks(
    hist,
    height=hist.max() * 0.005,
    distance=N_BINS // 40,
    prominence=hist.max() * 0.002
)

print(f"\nDetected {len(peaks)} peak(s):")
for i, p in enumerate(peaks):
    print(f"  Peak {i+1}: gray value = {centers[p]:.1f}   count = {hist[p]:,}")

thresholds = []
if len(peaks) >= 2:
    print("\nSuggested thresholds (midpoint between adjacent peaks):")
    for i in range(len(peaks) - 1):
        t = (centers[peaks[i]] + centers[peaks[i+1]]) / 2.0
        thresholds.append(t)
        print(f"  Threshold {i+1}: {t:.1f}   (between Peak {i+1} and Peak {i+2})")

# ── 5. Plot (MATLAB-style, matching PSD plot) ─────────────────────────────────
FS = 30
LW = 3
PEAK_COLORS = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]   # red, green, purple, orange
THRESH_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # blue, orange, green, red

plt.rcParams.update({
    "font.size"        : FS,
    "axes.titlesize"   : FS,
    "axes.labelsize"   : FS,
    "xtick.labelsize"  : FS,
    "ytick.labelsize"  : FS,
    "legend.fontsize"  : FS - 6,
    "lines.linewidth"  : LW,
    "axes.linewidth"   : 1.8,
    "xtick.major.width": 1.8,
    "ytick.major.width": 1.8,
    "xtick.major.size" : 8,
    "ytick.major.size" : 8,
    "grid.linewidth"   : 0.8,
    "grid.alpha"       : 0.4,
    "figure.facecolor" : "white",
    "axes.facecolor"   : "white",
})

# Phase names for legend (adapt per specimen)
PHASE_NAMES = ['Air', 'Ice', 'Glass beads', 'Aluminum']

fig, ax = plt.subplots(1, 1, figsize=(16, 9))

for ax, scale in zip([ax], ['log']):
    # Histogram fill + line
    ax.fill_between(centers, hist, color='steelblue', alpha=0.3)
    ax.plot(centers, hist, color='steelblue', linewidth=LW)

    # Peaks
    for i, p in enumerate(peaks):
        color = PEAK_COLORS[i % len(PEAK_COLORS)]
        label = PHASE_NAMES[i] if i < len(PHASE_NAMES) else f'Peak {i+1}'
        ax.plot(centers[p], hist[p], 'v', color=color, markersize=16,
                markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        ax.annotate(f"{label}\n{centers[p]:.0f}",
                    xy=(centers[p], hist[p]),
                    xytext=(0, 25), textcoords='offset points',
                    fontsize=FS - 8, color=color, ha='center', fontweight='bold')

    # Thresholds
    for j, t in enumerate(thresholds):
        color = THRESH_COLORS[j % len(THRESH_COLORS)]
        ax.axvline(t, color=color, linestyle='--', linewidth=2.2,
                   label=f'T{j+1} = {t:.0f}')

    ax.set_xlabel('Gray value', labelpad=10)
    ax.set_ylabel('Voxel count', labelpad=10)
    ax.set_yscale(scale)
    ax.set_title(f'Histogram ({scale} scale)', pad=12)
    ax.grid(True, which='major', axis='both')

    # Box: all 4 spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.8)

    ax.legend(loc='upper right', frameon=True, framealpha=0.85, edgecolor='gray')

plt.suptitle('CT Gray Value Histogram', fontsize=FS + 4, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
plt.close()
plt.rcdefaults()
print(f"\nHistogram saved to:\n  {OUT_PATH}")
print("\nDone. Share the peak gray values so I can write the segmentation script.")
