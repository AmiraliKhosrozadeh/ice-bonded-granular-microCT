# Scan 2 pipeline — run order

Close any Dragonfly view / File Explorer preview of the scan-2 `segmentation\`
and `results\` folders before running. The safe-save guard will throw
PermissionError naming the locked file if anything has outputs open.

## One-time calibration (in Dragonfly)

Before the first run:

1. Open the scan 2 TIFF stack in Dragonfly (`Glass_100_1800_T5_02`).
2. Run the histogram check — tells you whether the gray thresholds still hold:

    ```python
    exec(open(r'E:\RPTU-images\CT_images\Glass\Glass_scan02_dragonfly\scan02_histogram_analysis.py', encoding='utf-8').read(), globals())
    ```

    Inspect `results\histogram_analysis.png`. If the three peaks shifted,
    update `T_AIR_ICE`, `T_ICE_GLASS`, `T_GLASS_AL` in
    `scan02_segmentation.py`.

3. Draw a rough tilted cylinder ROI covering the ice+glass specimen. Copy
   Cap1 (top), Cap2 (bottom) and Radius from the ROI inspector into the
   `CYL_CAP1_UM`, `CYL_CAP2_UM`, `CYL_RADIUS_UM` block near the top of
   `scan02_segmentation.py` AND `scan02_porosity_analysis.py`.

4. Verify `CT_CHANNEL_NAME` in both `scan02_segmentation.py` and
   `scan02_bead_analysis.py` matches the Dragonfly channel title for scan 2.

## Pipeline

Run these three lines in the Dragonfly Python console, in order. Each keeps
its variables (`seg`, `specimen_mask`, `bead_labels`, `results`) alive for the
next.

```python
exec(open(r'E:\RPTU-images\CT_images\Glass\Glass_scan02_dragonfly\scan02_segmentation.py', encoding='utf-8').read(), globals())
```

```python
exec(open(r'E:\RPTU-images\CT_images\Glass\Glass_scan02_dragonfly\scan02_bead_analysis.py', encoding='utf-8').read(), globals())
```

```python
exec(open(r'E:\RPTU-images\CT_images\Glass\Glass_scan02_dragonfly\scan02_porosity_analysis.py', encoding='utf-8').read(), globals())
```

## Outputs

All land in `Glass_scan02_dragonfly\`:
- `segmentation\seg_0000.tif..seg_NNNN.tif` — 5-phase segmentation stack
- `results\contour_detection_workflow.png` — Fig 3.15-style QC of the contour
- `results\segmentation_overview.png` — XY/XZ/YZ phase overview
- `results\bead_measurements.csv` — per-bead volume, Feret, eq. diameter, centroid
- `results\bead_psd_statistics.txt` — D10/D50/D90/Span
- `results\bead_size_distribution.png` — PSD plot (Feret x-axis)
- `results\porosity_*.png` and `.txt` — radial/height profiles
- `results\segmentation_statistics.txt` — overall phase fractions

Also written, for Part B (SPAM DVC/DDIC):
- `..\Glass_spam\data\ct_scan02.tif` — filtered CT volume (uint16)
- `..\Glass_spam\data\specimen_mask_scan02.tif` — interior mask (uint8 0/255)
- `..\Glass_spam\data\bead_labels_scan02.tif` — labelled beads (uint16)

## Scan 1 vs scan 2 comparison (after both pipelines done)

```python
exec(open(r'E:\RPTU-images\CT_images\Glass\Glass_scan02_dragonfly\scan_comparison.py', encoding='utf-8').read(), globals())
```

Reads both `bead_measurements.csv` files and both porosity statistics, writes
overlay plots into `Glass_scan02_dragonfly\results\comparison\`. Pure Python
matplotlib — runs from Dragonfly or standalone.
