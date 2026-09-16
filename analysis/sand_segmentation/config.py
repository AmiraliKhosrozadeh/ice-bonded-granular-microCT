"""
Central configuration for the Sand (sand / ice / air) micro-CT pipeline.

This is the ONLY file you edit per specimen / per scan. Every stage script
imports `CFG` from here so the whole Stage 0-3 classical scaffold runs off one
set of tunables (mirrors the `pipeline_github` per-specimen-config convention).

Pipeline context (frozen / ice-bonded sand, 3 phases: air / ice / sand):
  Stage 0  scan audit            -> histogram, SNR, phase peaks, grain px-size
  Stage 1  phase segmentation    -> air / ice / sand masks (threshold now;
                                    swappable for a 3-class nnU-Net later)
  Stage 2  sand-mask extraction  -> clean sand-only mask (preserve contacts/fines)
  Stage 3  watershed baseline    -> EDT + H-maxima + marker watershed grain labels
                                    + QC report (the classical benchmark and the
                                    source of pseudo-labels for ParticleSeg3D)

Deep learning (ParticleSeg3D border-core nnU-Net) is Stage 4+, NOT here.
"""
import os
from dataclasses import dataclass, field


@dataclass
class SandConfig:
    # ---- identity -----------------------------------------------------------
    sample_name: str = "Sand-100-500-T5_100XXL_uc"
    file_prefix: str = "Sand-100-500-T5_100XXL_uc_xy_"

    # ---- input data ---------------------------------------------------------
    # Folder holding the reconstructed TIFF slice stack (uint16).
    # Complete data lives in the nested Sand\Sand\<specimen> tree. Available
    # specimens: Sand_100_500_T5_01/02, Sand_25mm_100_500_T5_01/02/03,
    # Sand_75_200_T5_01/02 (the 75-200 um ones are the finest -> hardest fines).
    tiff_dir: str = r"E:\RPTU-images\CT_images\Sand\Sand\Sand_100_500_T5_01"
    # Output root for all pipeline products (created if missing).
    out_root: str = r"E:\RPTU-images\CT_images\Sand\pipeline\results\Sand_100_500_T5_01"

    # ---- geometry -----------------------------------------------------------
    voxel_size_um: float = 24.7660229      # ORIVOXELSIZE from the .PRM (mm*1000)
    grain_size_um: tuple = (100.0, 500.0)  # nominal sieve range of this specimen

    # ---- subvolume (for fast iteration) -------------------------------------
    # Stage 1-3 are heavy on the full 996 x 1179 x 1179 volume. For the first
    # build/test run a Z slab and an optional XY crop. Set z_range=None and
    # xy_crop=None to process the full stack once validated.
    z_range: tuple | None = None           # (first, last) inclusive slice index; None = all
    xy_crop: tuple | None = (255, 1182, 0, 851)  # from Stage 0b plug bbox; None = full frame

    # ---- noise filter -------------------------------------------------------
    median_size: int = 3                   # 3D median filter window (0 = skip)

    # ---- specimen mask ------------------------------------------------------
    # Automatic specimen mask: per-slice, threshold material > air, close gaps,
    # fill holes, keep largest component -> the sand+ice plug (excludes the
    # outside-air zeros that dominate the global histogram). No Dragonfly
    # cylinder measurement needed for the scaffold; a manual cylinder can be
    # wired in later if the auto mask leaks.
    specimen_close_radius: int = 6         # px, bridge inter-grain gaps in the plug
    specimen_min_area_frac: float = 0.02   # drop slice components < this frac of largest

    # ---- phase thresholds (air / ice / sand) --------------------------------
    # None -> Stage 0/1 auto-pick with 3-class multi-Otsu INSIDE the specimen
    # mask and report them. Once you trust the audit, paste fixed integers here
    # for reproducibility across the scan series.
    # NOTE: Stage 0 audit (full-height sampling) gave 2095 / 5253. A mid-slab
    # auto-Otsu instead gave 3086 / 3802 -> the ice/sand boundary DRIFTS between
    # regions because partial-volume sand-grain edges overlap the ice gray band.
    # Fixed values below give reproducible phase seg across the load-step series;
    # the residual ice/sand ambiguity is what the 3-class nnU-Net (Stage 1 swap)
    # is meant to resolve.
    t_air_ice: int | None = 2095           # gray < t_air_ice            -> air (void)
    t_ice_sand: int | None = 5253          # t_air_ice <= gray < t_ice_sand -> ice
    #                                        gray >= t_ice_sand          -> sand
    # Z-drift is huge (range ~7105) so a single global threshold floods sand
    # together in bright z-bands (mega-merge) and shatters it in dark ones.
    # adaptive_thresholds picks PER-SLICE multi-Otsu thresholds (z-smoothed),
    # which is mandatory for the full stack. Threshold drift IS the classical
    # failure that the 3-class nnU-Net ultimately replaces.
    adaptive_thresholds: bool = True

    # ---- Stage 2 sand-mask cleanup ------------------------------------------
    sand_min_voxels: int = 8               # remove sand specks below this (noise)
    sand_open_radius: int = 0              # morphological opening radius (0 = off,
    #                                        keep 0 to preserve thin fines/contacts)

    # ---- Stage 3 watershed baseline -----------------------------------------
    # H-maxima suppression height on the EDT (in voxels): larger -> fewer seeds
    # -> less over-segmentation but more merged grains. Marker-controlled
    # watershed on the inverted EDT (classic touching-grain separation).
    # Best of the Stage 3b sweep vs Camsizer (h=1.0, md=2 -> d50 +21%, 12% merged;
    # the classical optimum -- still can't reach experimental PSD, hence the DL).
    hmaxima_h: float = 1.0                 # EDT h-maxima height (voxels)
    seed_min_distance: int = 2             # min voxels between seeds (peak_local_max)
    watershed_min_grain_voxels: int = 20   # discard labels below this after watershed

    # ---- reporting ----------------------------------------------------------
    # Smallest resolvable grain in voxels at this voxel size, used to flag the
    # missed-fines risk in the QC report (ParticleSeg3D ~3-voxel floor).
    fines_floor_voxels: float = 3.0

    def __post_init__(self):
        self.min_grain_um, self.max_grain_um = self.grain_size_um

    # convenience derived paths -------------------------------------------
    def d(self, *parts) -> str:
        p = os.path.join(self.out_root, *parts)
        os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
        return p


# ---- specimen registry -----------------------------------------------------
# Select with env var SAND_SPECIMEN (default 100_500_01). Lets the same scripts
# run any specimen without editing the dataclass:  SAND_SPECIMEN=75_200_01 python ...
_DATA = r"E:\RPTU-images\CT_images\Sand\Sand"
_OUT = r"E:\RPTU-images\CT_images\Sand\pipeline\results"
SPECIMENS = {
    "100_500_01": dict(
        sample_name="Sand-100-500-T5_100XXL_uc",
        file_prefix="Sand-100-500-T5_100XXL_uc_xy_",
        tiff_dir=os.path.join(_DATA, "Sand_100_500_T5_01"),
        out_root=os.path.join(_OUT, "Sand_100_500_T5_01"),
        grain_size_um=(100.0, 500.0),
        xy_crop=(255, 1182, 0, 851),     # from this specimen's Stage 0b
    ),
    "75_200_01": dict(
        sample_name="Sand-75-200-T5_100XXL_uc",
        file_prefix="Sand-75-200-T5_100XXL_uc_xy_",
        tiff_dir=os.path.join(_DATA, "Sand_75_200_T5_01"),
        out_root=os.path.join(_OUT, "Sand_75_200_T5_01"),
        grain_size_um=(75.0, 200.0),     # 75 um = 3 vox -> AT the floor (hardest)
        xy_crop=(264, 1308, 0, 964),     # from this specimen's Stage 0b (drift range 13170)
    ),
}

_sel = os.environ.get("SAND_SPECIMEN", "100_500_01")
if _sel not in SPECIMENS:
    raise ValueError(f"SAND_SPECIMEN={_sel!r} not in {list(SPECIMENS)}")
CFG = SandConfig(**SPECIMENS[_sel])
print(f"[config] specimen = {_sel}  ({CFG.sample_name})")

# Phase label values (uint8) shared across all stages and matched to the
# glass/ice pipeline convention so downstream/SPAM code reads them the same way.
LABEL_OUTSIDE_AIR = 0
LABEL_VOID_AIR    = 1   # air voids inside the specimen
LABEL_ICE         = 2
LABEL_SAND        = 3
LABEL_NAMES = {
    LABEL_OUTSIDE_AIR: "Outside air",
    LABEL_VOID_AIR:    "Air void (interior)",
    LABEL_ICE:         "Ice",
    LABEL_SAND:        "Sand",
}
# RGB for QC overlays (publication style: distinct, print-safe)
PHASE_RGB = {
    LABEL_OUTSIDE_AIR: [0.05, 0.05, 0.05],
    LABEL_VOID_AIR:    [1.00, 0.30, 0.30],
    LABEL_ICE:         [0.40, 0.70, 1.00],
    LABEL_SAND:        [0.90, 0.80, 0.45],
}
