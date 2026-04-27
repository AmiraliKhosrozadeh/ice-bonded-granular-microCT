"""Metadata for each <SPECIMEN> scan: world origin of voxel (0,0,0) and voxel size.

Read from the Dragonfly channel inspector (info.txt) at the time each scan's
TIFFs were exported in scanNN_segmentation.py's Step 8.

If you ever re-run segmentation with a different Dragonfly crop, update
ORIGIN_UM here to match the new channel origin so the SPAM alignment stays
consistent.

Three scans -> two compression transitions: 1->2 and 2->3.
"""
VOXEL_UM = 24.7660229

# IMPORTANT: Dragonfly's reported origin_um correctly reflects the X and Y
# crop offsets (origin_X = (minX - 0.5) * voxel_um), but Dragonfly RESETS
# its Z origin to the channel's own voxel 0 (origin_Z is always ~-0.5*voxel).
# So origin_um does NOT encode which raw slice the channel started from.
# We must use FIRST_SLICE (the raw-TIFF index of Dragonfly's loaded Z=0)
# as the true Z reference so scans can be stacked in the common raw frame
# without a spurious Z shift.
#
# ============================================================================
# TODO: fill in real values per scan from info.txt once the Dragonfly scripts
# have been run. Placeholders below MUST be replaced before running
# spam_align.py.
# ============================================================================
SCANS = {
    1: dict(
        origin_um=(14252.833, 4371.199, -12.383),
        first_slice=80,                        # channel title ends in _xy_0080
        data_dir='/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data',
    ),
    2: dict(
        origin_um=(14252.833, 4371.199, -12.383),
        first_slice=120,                       # channel title ends in _xy_0120
        data_dir='/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data',
    ),
    3: dict(
        origin_um=(14228.067, 4346.433, -12.383),
        first_slice=90,                        # channel title ends in _xy_0090
        data_dir='/mnt/e/RPTU-images/CT_images/Glass/<SPECIMEN>/<SPAM>/data',
    ),
}
