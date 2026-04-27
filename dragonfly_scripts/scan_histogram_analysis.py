"""
Histogram analysis for glass bead / ice / air CT dataset
Reads TIFF stack directly - no ORS API needed
"""
# ===================================================================
# SCAN 1 / T7 1700 µm — scaffolded by _scaffold.py.
# EDIT BEFORE RUNNING (measure in Dragonfly):
#   FIRST_SLICE / LAST_SLICE      — particle Z range in THIS stack
#   X_MIN, X_MAX, Y_MIN, Y_MAX    — XY crop (cover specimen in ALL scans)
#   CYL_CAP1_UM, CYL_CAP2_UM      — cylinder caps (world coords, µm)
#   CYL_RADIUS_UM                  — cylinder radius (µm)
#   CT_ORIGIN_UM                   — from ct_channel.getOrigin()
#   T_AIR_ICE, T_ICE_GLASS, T_GLASS_AL  — thresholds from histogram_analysis
#   CT_CHANNEL_NAME                — raw channel name in Dragonfly
# ===================================================================


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# ── 1. Settings (SCAN 2) ──────────────────────────────────────────────────────
TIFF_DIR    = r'E:\RPTU-images\CT_images\Glass\<SPECIMEN>\<SPECIMEN>_01'
FILE_PREFIX = 'Glass-75-1700-T5-HR_100XXL_uc_xy_'
FIRST_SLICE = 0
LAST_SLICE  = 1127

# Crop applied in Dragonfly (TODO scan 2: remeasure; defaults mirror scan 1)
X_MIN, X_MAX = 546, 1012
Y_MIN, Y_MAX = 96,  542

# Sample every Nth slice (faster - still accurate for histogram)
STEP = 5

RESULTS_DIR = r'E:\RPTU-images\CT_images\Glass\<SPECIMEN>\\scan{N}_dragonfly\results'
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

# ── 4. Robust peak detection ─────────────────────────────────────────────
# Smooth the histogram so close-by noise peaks merge into one. Require at
# least MIN_PEAK_SEP_GRAY gray units between peaks. Drop peaks below
# MIN_PHASE_GRAY (image-padding / background spike).
from scipy.ndimage import gaussian_filter1d

MIN_PHASE_GRAY     = 300     # peaks below this -> image-padding/background, ignored
MIN_PEAK_SEP_GRAY  = 800     # merge closer peaks
AL_MIN_GRAY        = 15000   # aluminium expected above this
SMOOTH_SIGMA_BINS  = 6       # Gaussian smoothing before peak-finding

gray_per_bin = (data.max() - data.min()) / N_BINS
hist_s = gaussian_filter1d(hist.astype(float), sigma=SMOOTH_SIGMA_BINS)

min_sep_bins = max(3, int(round(MIN_PEAK_SEP_GRAY / max(gray_per_bin, 1e-6))))
peaks_all, _ = find_peaks(
    hist_s,
    height     = hist_s.max() * 0.003,
    distance   = min_sep_bins,
    prominence = hist_s.max() * 0.01,
)

peak_grays = centers[peaks_all]
phase_mask = (peak_grays >= MIN_PHASE_GRAY) & (peak_grays <  AL_MIN_GRAY)
al_mask    = (peak_grays >= AL_MIN_GRAY)
phase_peaks_all = peaks_all[phase_mask]
phase_peaks     = phase_peaks_all[:3]        # first 3 labelled Air / Ice / Glass
phase_peaks_extra = phase_peaks_all[3:]     # any additional phase peaks (reported, not auto-labelled)
al_peaks    = peaks_all[al_mask]

# Report every detected peak so nothing is silently hidden:
print(f"\nAll detected peaks (smoothed sigma={SMOOTH_SIGMA_BINS} bins, "
      f"min separation {MIN_PEAK_SEP_GRAY} gray):")
_labels_3 = ['Air', 'Ice', 'Glass beads']
for p in peaks_all:
    g = centers[p]
    if g < MIN_PHASE_GRAY:
        tag = 'DROPPED (< MIN_PHASE_GRAY, treated as background)'
    elif g >= AL_MIN_GRAY:
        tag = 'Aluminum candidate'
    elif p in phase_peaks:
        i = list(phase_peaks).index(p)
        tag = _labels_3[i]
    else:
        tag = 'EXTRA phase peak (not auto-labelled -- consider manual threshold)'
    print(f"  gray = {g:>8.1f}   count = {hist[p]:>8,d}   -> {tag}")
if len(phase_peaks_extra) > 0:
    print(f"\nNOTE: {len(phase_peaks_extra)} extra peak(s) below AL_MIN_GRAY="
          f"{AL_MIN_GRAY} were not auto-labelled (double-peak / secondary mode). "
          f"If one of them is a real phase, edit T_* manually in segmentation.py.")

if len(al_peaks) > 0:
    al_idx = al_peaks[int(np.argmax(hist[al_peaks]))]
    al_gray = float(centers[al_idx])
else:
    # No aluminium peak above AL_MIN_GRAY -> estimate from high-gray tail
    if len(phase_peaks) > 0:
        tail = data[data > centers[phase_peaks[-1]] + 500]
        al_gray = float(np.percentile(tail, 90)) if tail.size > 0 else float(data.max())
    else:
        al_gray = float(data.max())
    al_idx = None

peaks = list(phase_peaks)
if al_idx is not None:
    peaks = peaks + [al_idx]
peaks = np.array(peaks, dtype=int) if peaks else np.array([], dtype=int)

PHASE_NAMES_ORDERED = ['Air', 'Ice', 'Glass beads', 'Aluminum']

print(f"\nDetected {len(peaks)} labelled peak(s) after filtering:")
for i, p in enumerate(peaks):
    name = PHASE_NAMES_ORDERED[i] if i < len(PHASE_NAMES_ORDERED) else f'Peak {i+1}'
    print(f"  {name:<14s} gray = {centers[p]:>8.1f}   count = {hist[p]:,}")
if al_idx is None:
    print(f"  (no Aluminum peak found above {AL_MIN_GRAY}; tail estimate Al ~ {al_gray:.1f})")

# Build a full list of gray positions including the estimated Al even if
# there was no detected Al peak, so we always print 3 thresholds.
all_gray = [float(centers[p]) for p in phase_peaks]
if al_idx is not None:
    all_gray.append(float(centers[al_idx]))
else:
    all_gray.append(al_gray)

thresholds = []
thresh_labels = []
thresh_names  = ['T_AIR_ICE', 'T_ICE_GLASS', 'T_GLASS_AL']
print("\nSuggested thresholds (copy into the segmentation script):")
for i in range(min(3, len(all_gray) - 1)):
    t = 0.5 * (all_gray[i] + all_gray[i+1])
    thresholds.append(t)
    thresh_labels.append(thresh_names[i])
    print(f"  {thresh_names[i]:<12s} = {t:>9.1f}   (midpoint "
          f"{PHASE_NAMES_ORDERED[i]:<12s} <-> {PHASE_NAMES_ORDERED[i+1]})")
if len(all_gray) < 4:
    print("  NOTE: fewer than 4 labelled peaks were found. The suggested "
          "T_GLASS_AL uses a tail estimate; verify it manually.")

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
    "axes.edgecolor"   : "black",
    "axes.spines.top"   : True,
    "axes.spines.right" : True,
    "axes.spines.left"  : True,
    "axes.spines.bottom": True,
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
        lbl = thresh_labels[j] if j < len(thresh_labels) else f'T{j+1}'
        ax.axvline(t, color=color, linestyle='--', linewidth=2.2,
                   label=f'{lbl} = {t:.0f}')

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
