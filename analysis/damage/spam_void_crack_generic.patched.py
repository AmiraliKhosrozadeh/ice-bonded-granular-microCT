"""Void/crack analysis for ANY specimen, ported from Glass_75_1700_T5_HR.

WHAT THIS DOES, AND WHY EACH STEP EXISTS
----------------------------------------
1. NORMALISE. Scans of the same specimen sit on different gray scales
   (Glass_75 scan 1 was ~2.9x scans 2/3; Glass_75_1000_T6 scan 1 peaks at
   51263 against scan 2's 9280). Any threshold shared across scans is
   meaningless until this is fixed. Landmarks are the two tallest phase peaks
   -- ice and the grain material -- which are physically invariant.

2. FIELD OF PLAY. If the tube is in frame, fit it as a CYLINDER (centre from
   the filled disc, radius as the median over 360 angular sectors) and search
   inside bore - MARGIN. The median ignores the inward spikes that a distance
   transform would chase. If the volume was already masked during alignment,
   the tube is gone and the nonzero region is used instead.

3. SAMPLE. solid = gray >= air/ice midpoint; sample = per-slice closing at
   R_TIGHT, 3D-labelled, keeping the main body and every detached part.

4. DROP THE PLATEN (solid metal end cap, collapses the bore) and the PUNCH
   BAND (over-bright run from the top; -z is up). Punch must be detected
   AFTER the platen, since platen metal is over-bright too.

4b. GRAY CALIBRATION ON INVARIANT PHASES ONLY. Each scan is mapped onto the
   UNLOADED scan using the glass beads and air -- the two phases that cannot
   change -- and the air/ice threshold is then frozen at the unloaded scan's
   value. Ice is never used as a landmark: ice turning into air is the
   measurement, so calibrating on it divides the signal out. The old map sent
   each scan's ice peak onto the reference ice peak and took the reference from
   the MIDDLE scan, which is a loaded one.

   Equally important, the threshold must not be recomputed per scan from the
   annulus between specimen and tube. That annulus is empty only while the
   specimen is unloaded; afterwards the specimen expands into it. On
   Glass_100_1700_T5_HR the annulus "air" median went 1073 -> 2308 and its p90
   2071 -> 6398, pushing the threshold 2527 -> 3600 and calling 18% of the ice
   matrix air. With the frozen threshold the same specimen reads 3.29% ->
   16.07%, so most of that was real damage and some of it was the artefact.

5. NARROW GAPS = closing(solid, R_TIGHT) AND NOT solid, i.e. gaps under
   2*R_TIGHT. The space outside the specimen is wide, so it cannot enter this
   class -- that is what makes crack and outside separable at all. In an
   intact specimen ice fills the space between grains, so a narrow air gap
   means a broken bond.

6. VOID vs CRACK by connected size, at a FIXED ceiling. Shape does not work:
   the crack network is one connected object, so PCA reads it as a blob. An
   isolated pore is a compact body of a couple of mm3; the crack network is one
   large connected object, so size separates them.

   The ceiling used to be "largest connected body in the unloaded scan". That
   rule is wrong and it silenced the crack class on EVERY specimen. In a dense
   bead pack the pore space between beads is narrow everywhere, so the closing
   bridges the whole pore network into one connected body of 40-60 mm3 already
   in scan 1. Nothing in a later scan can exceed that, so everything was
   labelled void: all four generic-path specimens reported crack 0.00 mm3 and
   0 crack bodies. Glass_75 escaped only because its own script uses a fixed
   2.0 mm3 -- which is where this value comes from, and those are the figures
   that read correctly.

7. SEPARATION -- wide openings enclosed by specimen material:

       separation = fill_holes(sample slice) AND NOT sample

   The narrow-gap class deliberately excludes wide gaps, because that exclusion
   is what keeps the outside air out. So a piece that parts and moves away is
   invisible to it. Enclosure is what makes wide gaps usable: the annulus is
   always OPEN to the outside, an internal opening is not. No radius and no
   threshold to tune.

   Two rejected alternatives, both measured on Glass_100_1700_T5_HR:
     closing(sample, 20 vox) marks only the crack mouth, a rim hugging the
       outer boundary, and misses the opening behind it.
     convex-hull deficit reported 78.0 mm3 (4.98% of the specimen) on the
       UNLOADED scan 1, as one annular ring spanning from the bead core out to
       stray bits near the wall. It fails the control.
   Enclosed openings give 1.72 mm3 (0.11%) on scan 1 against 28.54 mm3 (1.48%)
   on scan 2 -- a 16.6x rise off a floor that is already near zero.

Usage:
    python spam_void_crack_generic.py <spam_dir> <name> <scan1> [scan2 ...]
"""
import os
import sys

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

R_TIGHT = 5           # vox -> narrow gap is anything under 2*R
RAMP = 13             # vox (0.32 mm) clear of the TUBE ITSELF.
# The tube's partial-volume ramp stays above the air/ice threshold for 9-13
# voxels inward from the metal (measured: Glass_75 13/10/9, T5_HR 13, T6 12/10,
# Alumina_T7 11/10/9), so 13 is the largest of those. Applied as a distance
# transform from the actual tube mask, which follows the real bore wherever it
# runs. Nothing is trimmed where the specimen stands off the wall.
MARGIN = 18           # vox; retained ONLY for the wall-shell test below.
# Set from measurement, not inherited. The tube's partial-volume ramp stays
# ABOVE the air/ice threshold for 9-13 voxels inward from the bore on every
# specimen measured (Glass_75 13/10/9, T5_HR 13/-, T6 12/10, Alumina_T7
# 11/10/9). The earlier value of 8 came from asking a different question --
# where gray falls from METAL level, which happens in 2-4 vox -- and left a rim
# of "material" clinging to the wall. That rim became a sample part, and the
# annulus inside it was reported as void surrounding the specimen.
N_SECTORS = 360
PUNCH_PAD = 10
MIN_PART_MM3 = 1.0
MIN_SLICE_FRAC = 0.05   # of the median slice area
# A specimen slice cannot be much LARGER than the median section -- the
# same bore bounds it all the way up. Anything well above the median is an
# artefact spanning the whole disc, not material. Needed once
# BORE_FALLBACK is on: above the tube rim there is no tube, so a
# reconstruction streak in the empty band between punch and specimen is no
# longer excluded by the tube gate. Measured on Alumina_175_1800_T5
# scan 1: the two artefact slabs at z 151-159 and 178-183 read 190% and
# 133% of the median section; no real slice in 184..1140 exceeds 110%.
MAX_SLICE_FRAC = 1.25
CLIP_MARGIN = 40
# Carry the fitted bore past the ends of the TUBE. The tube bounds the
# specimen RADIALLY; it does not bound it AXIALLY -- the specimen may
# stand above the tube rim, and it does. On every specimen the tube leaves
# the field of view before the specimen does, and the old code dropped
# each such slice with a bare `continue` that printed nothing. On
# Alumina_175_1800_T5 scan 1 that discarded the top 257 slices -- 6.4 mm,
# a quarter of the specimen. BORE_FALLBACK=0 restores the old behaviour.
BORE_FALLBACK = os.environ.get('BORE_FALLBACK', '1') == '1'
SKIN = 20             # vox; surface layer for the surface/body split
TUBE_FACTOR = float(os.environ.get('TUBE_FACTOR', 1.5))
# Fixed void ceiling, in mm3. A connected narrow-gap body below this is an
# isolated pore; at or above it, part of the crack network. 2.0 is the value
# the Glass_75 script uses, and those figures were the ones that read
# correctly. NOT derived from scan 1's largest body -- see rule 6 for why that
# reported zero crack on every specimen.
VOID_CEILING_MM3 = float(os.environ.get('VOID_CEILING_MM3', 2.0))
MIN_SEP_MM2 = float(os.environ.get('MIN_SEP_MM2', 0.20))   # per-slice opening
VOXEL_UM = 24.7660229
VOX_MM3 = (VOXEL_UM / 1000.0) ** 3
PX_MM2 = (VOXEL_UM / 1000.0) ** 2


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import specimen_mask as SM

# Hand-entered T_AIR_ICE from each scan0N_segmentation.py. Set by eye against
# the images, so they are the closest thing to ground truth available and they
# vote as M1. Every one was checked against phase peaks measured in the
# matching volume and all 11 sit between that scan's air and ice.
ENTERED = {
    'Glass_75_1700_T5_HR':  {1: 7000.0, 2: 2139.4, 3: 2015.4},
    'Glass_100_1700_T5_HR': {1: 2500.0, 2: 3802.07},
    'Glass_100_1700_T7':    {1: 8750.0, 2: 2800.0, 3: 1600.0, 4: 1650.0},
    'Glass_75_1000_T6':     {1: 5055.5, 2: 1184.0},
}

# Half-width above which a gap is material that has PARTED rather than a broken
# bond. 10 vox = 0.25 mm, the same width the narrow-gap class always used.
NARROW_VOX = 10

# A cross-section covering more than this fraction of the search area is a
# solid disc -- platen, support or punch -- not a granular specimen, which
# always has air at its boundary. Catches the end slices that the bead test
# misses because a few stray bead voxels keep them above its 5 % floor.
FILL_FRAC = 0.90
# Contiguous end slices below this fraction of the median area are the taper
# into the platen, not specimen. Measured on Alumina_75_1800_T7: the last
# sections read 33.6 and 23.8 mm2 against a median of 83-88, i.e. 27-40%,
# while everything in the body stays within 10-16% of the median. Trimmed from
# each end only, and only while the condition holds, so a genuinely thin
# specimen is never eaten from the middle outward.
END_MIN_FRAC = 0.60
# A solid disc must ALSO be almost free of beads. 0.30 rather than the 0.05 the
# plain platen test uses, because the slices this is aimed at do contain a few
# stray bead voxels -- that is exactly why they slipped through.
DISC_BEAD_FRAC = 0.30
# Majority over this many neighbouring slices, applied to the ENVELOPE only.
# The specimen is continuous, so its cross-section cannot step and step back.
Z_MEDIAN = 5

USE_RAW = os.environ.get('USE_RAW', '0') == '1'


def vol_path(data, s):
    """CT volume for one scan.

    This analysis fits the tube bore per slice and works entirely inside a
    single scan -- there is no cross-scan registration anywhere in it. So the
    ALIGNED volumes are not required, and the raw per-scan volumes work just as
    well. That matters because:
      * Glass_100_1800_T5 and Glass_100_1700_T7 never had aligned volumes;
      * three alumina specimens have the OLD BROKEN MASK baked into their
        aligned volumes (nonzero region matches the stored mask at 99-100% IoU),
        so their aligned data is contaminated while their raw data is clean.
    Set USE_RAW=1 to force the raw volumes.
    """
    a = os.path.join(data, f'ct_scan{s:02d}_aligned.tif')
    r = os.path.join(data, f'ct_scan{s:02d}.tif')
    if USE_RAW and os.path.exists(r):
        return r
    return a if os.path.exists(a) else r


def labels_path(data, s):
    a = os.path.join(data, f'bead_labels_scan{s:02d}_aligned.tif')
    r = os.path.join(data, f'bead_labels_scan{s:02d}.tif')
    if USE_RAW and os.path.exists(r):
        return r
    return a if os.path.exists(a) else r


def closing2d(x, r):
    dil = ndi.distance_transform_edt(~x) <= r
    return ndi.distance_transform_edt(dil) > r


def phase_peaks(vol, labels_path):
    """(ice, grain) gray values, identified from the BEAD LABELS.

    Picking the two tallest histogram peaks does not work: whenever the air
    peak is prominent it wins, and the pair comes back as (air, ice) instead
    of (ice, grain). That happened on every specimen tried -- Glass_75_1000_T6
    gave (1600, 4928) when the real pair was (4928, 9280), and the sample
    collapsed to 14 mm3 of 2000.

    The bead labels remove the ambiguity: the grain value is simply the median
    gray inside labelled beads. Ice is then the tallest histogram peak below
    it, and air the tallest below that.
    """
    lab = tifffile.imread(labels_path)
    beads = lab > 0
    if beads.sum() < 10000:
        return None
    # sample sparsely; the full volume is not needed for a median
    idx = np.nonzero(beads[::4, ::4, ::4])
    grain = float(np.median(vol[::4, ::4, ::4][idx]))
    del lab, beads

    v = vol[::4, ::4, ::4]
    v = v[(v > 0) & (v < grain * 0.75)]
    if v.size < 1000:
        return None
    hist, edges = np.histogram(v, bins=256, range=(0, float(grain)))
    centres = (edges[:-1] + edges[1:]) / 2
    smooth = gaussian_filter1d(hist.astype(float), 3)
    idx2, _ = find_peaks(smooth, height=smooth.max() * 0.05, distance=4)
    if not len(idx2):
        return None
    ice = float(centres[max(idx2, key=lambda i: centres[i])])
    # AIR peak: the tallest peak below ice. Needed because guessing T_AIR as a
    # fraction of the ice peak put it below the background level outside the
    # specimen -- 0.6*ice = 977 on T6, while the annulus between specimen and
    # tube sits above that. The annulus then read as SOLID, the sample envelope
    # covered the whole bore, and the gaps inside it were reported as void
    # surrounding the specimen.
    below = [i for i in idx2 if centres[i] < ice * 0.85]
    air = float(centres[max(below, key=lambda i: smooth[i])]) if below else ice * 0.5
    return air, ice, grain


def analyse(spam_dir, name, scans):
    data = os.path.join(spam_dir, 'data')
    out = os.path.join(spam_dir, 'results_voidcrack')
    os.makedirs(out, exist_ok=True)
    print(f'\n########## {name} ##########', flush=True)

    # Clear previous volumes. Without this a scan that fails leaves its old tif
    # in place beside freshly written ones, and the folder silently mixes runs
    # -- which is how stale scan-1 volumes were read back as current.
    _stale = [f for f in os.listdir(out)
              if f.endswith('.tif') or f.endswith('.png') or f.endswith('.txt')]
    for f in _stale:
        os.remove(os.path.join(out, f))
    if _stale:
        print(f'  cleared {len(_stale)} files from a previous run', flush=True)

    # ---- 1. normalisation landmarks -------------------------------------
    peaks = {}
    for s in scans:
        p = vol_path(data, s)
        if not os.path.exists(p):
            print(f'  scan {s}: MISSING volume', flush=True)
            return None
        lp = labels_path(data, s)
        if not os.path.exists(lp):
            print(f'  scan {s}: no bead labels, cannot identify phases', flush=True)
            return None
        v = tifffile.imread(p).astype(np.float32)
        pk = phase_peaks(v, lp)
        del v
        if pk is None:
            print(f'  scan {s}: could not find two phase peaks', flush=True)
            return None
        peaks[s] = pk
        print(f'  scan {s}: peaks air {pk[0]:.0f}  ice {pk[1]:.0f}  '
              f'grain {pk[2]:.0f}', flush=True)
    # Reference is the FIRST scan, which carried no load. It used to be the
    # MIDDLE scan -- a loaded one -- so the landmarks were taken from a
    # specimen that had already changed.
    ref = scans[0]
    r_air, r_ice, r_grain = peaks[ref]
    # Air/ice midpoint of the UNLOADED scan, then FROZEN. Every scan of this
    # specimen is judged against this one number. It cannot drift with load,
    # which is what went wrong before: the threshold was recomputed per scan
    # from an annulus the specimen expands into, so it rose with deformation
    # and started calling intact ice "air".
    t_air = 0.5 * (r_air + r_ice)
    t_grain_mid = 0.5 * (r_ice + r_grain)
    # Tube threshold as a multiple of the grain value. 1.5 is right and was
    # verified per specimen: at 1.5 the tube ring is found (T5_HR 10.9M voxels,
    # T6 scan 1 10.5M, T6 scan 2 16.1M, each ~ a ring of the expected size),
    # while at 2.2 it is found NOWHERE (0, 0, 8780) and the field of play falls
    # back to the whole nonzero region, doubling the sample. It also matches
    # the reference specimen, where glass 9536 x 1.47 = the 14000 that works.
    # 2.2 came from a grain value produced by the broken histogram peak-picking.
    t_tube = r_grain * TUBE_FACTOR
    print(f'  reference scan {ref} (unloaded): air {r_air:.0f} ice {r_ice:.0f}  '
          f'T_AIR {t_air:.0f} (frozen)  T_ICE_GRAIN {t_grain_mid:.0f}  '
          f'T_TUBE {t_tube:.0f}', flush=True)

    results = []
    void_ceiling = None
    sep_base = 0.0
    for s in scans:
        v = tifffile.imread(vol_path(data, s)).astype(np.float32)
        _air, ice, grain = peaks[s]
        # Two-point map on the invariant phases only: glass beads and air.
        # Anchoring air at air_ref * f makes the offset cancel exactly, so the
        # map reduces to a single scale factor. ICE IS DELIBERATELY NOT USED --
        # ice becoming air is the quantity being measured, and the previous map
        # (ice_scan -> ice_ref) divided it straight out.
        f_scale = grain / max(r_grain, 1e-6)
        v = v / max(f_scale, 1e-6)
        print(f'  scan {s}: bead scale {f_scale:.4f}  -> ice peak lands at '
              f'{ice / max(f_scale, 1e-6):.0f} against the reference '
              f'{r_ice:.0f}', flush=True)
        nz = v.shape[0]

        # bead labels are needed inside the slice loop now (method M3)
        beads_v = None
        _lp = labels_path(data, s)
        if os.path.exists(_lp):
            # Cleaned, not just >0. Noise sprayed into the support is labelled
            # too on some specimens, which blinds every bead-based test.
            beads_v = SM.clean_beads(tifffile.imread(_lp))
        t_entered = ENTERED.get(name, {}).get(s)
        if t_entered is not None:
            t_entered = t_entered / max(f_scale, 1e-6)   # onto the mapped scale
        print(f'  scan {s}: entered threshold '
              f'{"none" if t_entered is None else f"{t_entered:.0f}"}  '
              f'auto {t_air:.0f}', flush=True)

        env = np.zeros(v.shape, dtype=bool)
        area = np.zeros(nz)
        bore_area = np.zeros(nz)
        bright = np.zeros(nz)
        yy_g, xx_g = np.mgrid[0:v.shape[1], 0:v.shape[2]]
        has_tube = (v > t_tube).sum() > 5000

        # Pre-fit the bore on every slice where the tube is visible, then fill
        # the gaps. Glass_75_1000_T6 has the tube in only PART of scan 2's
        # slices, and fitting per slice dropped the rest -- the sample
        # collapsed to zero. The tube is rigid, so its bore is constant: carry
        # the median radius across, and interpolate the centre.
        bore_r = np.full(nz, np.nan)
        bore_y = np.full(nz, np.nan)
        bore_x = np.full(nz, np.nan)
        if has_tube:
            for z in range(nz):
                tube = v[z] > t_tube
                if tube.sum() < 200:
                    continue
                filled = ndi.binary_fill_holes(tube)
                if filled.sum() < 1000:
                    continue
                cy, cx = ndi.center_of_mass(filled)
                ty, tx = np.nonzero(tube)
                rr = np.sqrt((ty - cy) ** 2 + (tx - cx) ** 2)
                aa = np.arctan2(ty - cy, tx - cx)
                sec = ((aa + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS
                r_min = np.full(N_SECTORS, np.inf)
                np.minimum.at(r_min, sec, rr)
                r_min[~np.isfinite(r_min)] = np.nan
                rb = np.nanmedian(r_min)
                # a partial ring gives a spuriously small radius; require the
                # tube to be seen over most directions
                if np.isfinite(rb) and np.mean(np.isfinite(r_min)) > 0.6:
                    bore_r[z], bore_y[z], bore_x[z] = rb, cy, cx
            good = np.isfinite(bore_r)
            if good.sum() >= 5:
                zz_all = np.arange(nz)
                bore_r[:] = np.nanmedian(bore_r[good])
                bore_y = np.interp(zz_all, zz_all[good], bore_y[good])
                bore_x = np.interp(zz_all, zz_all[good], bore_x[good])
                print(f'  scan {s}: bore fitted on {good.sum()}/{nz} slices, '
                      f'r = {np.nanmedian(bore_r)*VOXEL_UM/1000:.2f} mm', flush=True)
            else:
                has_tube = False
                print(f'  scan {s}: tube too sparse, using nonzero region', flush=True)

        # Every way a slice can be dropped, counted. These were four bare
        # `continue` statements with no print, which is why 257 missing
        # slices left no trace in any log.
        n_drop = dict(no_bore=0, small_bore=0, no_solid=0, no_masks=0,
                      fallback=0)
        for z in range(nz):
            sl = v[z]
            if has_tube:
                if not np.isfinite(bore_r[z]):
                    n_drop['no_bore'] += 1
                    continue
                # Distance to the REAL tube, not to a fitted circle. A circle
                # is concentric with the tube; the specimen is neither
                # concentric with it nor circular once deformed, so a radial
                # limit cuts the specimen wherever it has expanded to touch the
                # wall -- verified on T5_HR scan 2, where the sample radius came
                # out as exactly bore - MARGIN and the boundary was a smooth arc
                # running through continuous material.
                # ONE implementation, in specimen_mask.bore_bound. A second
                # copy lived here and bailed before the shared one ever ran,
                # so the open-ring fallback added there had no effect and
                # Glass_100_1800_T5 scan 1 still produced nothing.
                interior = SM.bore_bound(sl, t_tube)
                if interior is None or interior.sum() < 1000:
                    # No tube IN THIS SLICE. The tube is rigid and its bore
                    # was already fitted and interpolated to every z above;
                    # use that circle, less the ramp allowance, rather than
                    # discarding material that is simply above the rim.
                    if not BORE_FALLBACK:
                        n_drop['no_bore' if interior is None
                               else 'small_bore'] += 1
                        continue
                    rad = bore_r[z] - SM.RAMP
                    interior = ((yy_g - bore_y[z]) ** 2 +
                                (xx_g - bore_x[z]) ** 2) <= rad ** 2
                    n_drop['fallback'] += 1
                    if interior.sum() < 1000:
                        n_drop['small_bore'] += 1
                        continue
            else:
                # already masked at alignment: the nonzero region is the field
                interior = sl > 0
                if interior.sum() < 1000:
                    continue
            bore_area[z] = interior.sum() * PX_MM2
            solid = (sl >= t_air) & interior
            if solid.sum() < 200:
                n_drop['no_solid'] += 1
                continue
            # FOUR METHODS, then a vote. One threshold plus a closing was not
            # enough: it put a smooth arc through continuous material wherever
            # the specimen had expanded to touch the tube, and left a ring of
            # the tube's partial-volume ramp reading as ice. See specimen_mask.
            d = SM.masks_for_slice(sl, beads_v[z] if beads_v is not None
                                   else None, t_tube, t_entered, t_air)
            if d is None:
                n_drop['no_masks'] += 1
                continue
            e, _ = SM.consensus(d)
            env[z] = e
            area[z] = e.sum() * PX_MM2
            if e.any():
                bright[z] = 100.0 * (sl[e] > t_tube * 0.82).sum() / e.sum()
        del v

        # BEADS DEFINE THE SPECIMEN. Platen (top) and support (bottom) are
        # solid discs with no beads in them; a specimen slice always contains
        # beads. This replaces the bore-collapse test, which only caught a
        # platen that shrinks the bore and missed the support entirely --
        # T6 scan 1's bottom slices are a uniform bead-free disc that was
        # being kept as specimen, and the punch trim only ever looked at the
        # top, so nothing removed it.
        bead_area = (beads_v.sum(axis=(1, 2)) if beads_v is not None
                     else np.zeros(nz))
        med_bead = np.median(bead_area[bead_area > 0]) if (bead_area > 0).any() else 0.0
        no_bead = bead_area < 0.05 * med_bead
        n_platen = int((no_bead & (area > 0)).sum())
        env[no_bead] = False
        area[no_bead] = 0.0
        if n_platen:
            print(f'  scan {s}: dropped {n_platen} bead-free slices '
                  f'(platen / support)', flush=True)

        # SOLID DISC AND NO BEADS. Filling the search area is NOT on its own a
        # sign of platen: a compressed specimen expands to the wall and fills
        # it too, which is why an earlier version of this test that keyed on
        # fill alone removed 493 slices from Glass_75 scan 3 and cut the
        # specimen from 2122 to 638 mm3. Beads are the discriminator -- platen
        # is solid metal and has none. Both conditions together catch the end
        # slices that the 5% bead floor above lets through because a few stray
        # bead voxels sit in them.
        _fill = np.divide(area, np.maximum(bore_area, 1e-9))
        _few_beads = bead_area < DISC_BEAD_FRAC * max(med_bead, 1e-9)
        _solid_disc = (_fill > FILL_FRAC) & _few_beads & (area > 0)
        if _solid_disc.any():
            env[_solid_disc] = False
            area[_solid_disc] = 0.0
            print(f'    dropped {int(_solid_disc.sum())} slices that fill '
                  f'>{100*FILL_FRAC:.0f}% of the bore AND have '
                  f'<{100*DISC_BEAD_FRAC:.0f}% of the usual beads', flush=True)

        # END TAPER. The outermost sections press into the platen and thin out.
        # Runs LAST, so its median is taken over specimen sections only -- with
        # the platen still present the median sits at the full bore area and
        # the trim eats real specimen.
        _pos = np.where(area > 0)[0]
        if len(_pos) > 20:
            _medA = np.median(area[_pos])
            _n_end = 0
            for _z in _pos:                        # inward from the top
                if area[_z] < END_MIN_FRAC * _medA:
                    env[_z] = False; area[_z] = 0.0; _n_end += 1
                else:
                    break
            for _z in _pos[::-1]:                  # inward from the bottom
                if area[_z] < END_MIN_FRAC * _medA:
                    env[_z] = False; area[_z] = 0.0; _n_end += 1
                else:
                    break
            if _n_end:
                print(f'    trimmed {_n_end} tapered end slices '
                      f'(<{100*END_MIN_FRAC:.0f}% of the median section)',
                      flush=True)

        present = np.where(area > 0)[0]
        n_punch = 0
        if len(present):
            f = bright[present]
            core = np.median(f[len(f) // 4:3 * len(f) // 4])
            thr = max(core * 3, 0.5)
            run = 0
            for val in f:
                if val > thr:
                    run += 1
                else:
                    break
            if run:
                n_punch = run + PUNCH_PAD
                env[present[:n_punch]] = False
                area[present[:n_punch]] = 0.0

        print(f'  scan {s}: slices dropped -- '
              f"no bore {n_drop['no_bore']}  "
              f"small bore {n_drop['small_bore']}  "
              f"no solid {n_drop['no_solid']}  "
              f"no masks {n_drop['no_masks']}   "
              f"projected bore used on {n_drop['fallback']}", flush=True)
        med_area = np.median(area[area > 0]) if (area > 0).any() else 0.0
        # Discard whole-disc artefacts before the extent is measured, so
        # they cannot set z0/z1 or inflate the medians the trims key on.
        _fat = area > MAX_SLICE_FRAC * med_area
        if _fat.any():
            print(f'    dropped {int(_fat.sum())} slices above '
                  f'{100*MAX_SLICE_FRAC:.0f}% of the median section '
                  f'(whole-disc artefact)', flush=True)
            env[_fat] = False
            area[_fat] = 0.0
            med_area = (np.median(area[area > 0])
                        if (area > 0).any() else 0.0)
        zs = np.where(area >= MIN_SLICE_FRAC * med_area)[0]
        if not len(zs):
            print(f'  scan {s}: no sample found', flush=True)
            continue
        z0, z1 = int(zs[0]), int(zs[-1])
        zlo = z0 + CLIP_MARGIN if z0 == 0 else z0
        zhi = z1 - CLIP_MARGIN if z1 == nz - 1 else z1
        print(f'  scan {s}: z0 {z0} z1 {z1} -> zlo {zlo} zhi {zhi}  '
              f'({(zhi-zlo)*VOXEL_UM/1000:.2f} mm)', flush=True)
        keepz = np.zeros(nz, dtype=bool)
        keepz[zlo:zhi + 1] = True
        env &= keepz[:, None, None]

        lab, _ = ndi.label(env, structure=np.ones((3, 3, 3), bool))
        sizes = np.bincount(lab.ravel()) * VOX_MM3
        sizes[0] = 0
        keep = list(np.where(sizes >= MIN_PART_MM3)[0])

        # Reject WALL SHELLS. The tube's partial-volume ramp reads above T_AIR
        # and forms a thin ring just inside the bore. Because it spans the full
        # height it clears the 1 mm3 part threshold and was being kept as a
        # sample part -- visible as a second sample boundary at the bore, with
        # the annulus inside it reported as void. A real specimen part has
        # material well away from the wall; a ramp shell does not.
        if has_tube and keep:
            r_ref = np.nanmedian(bore_r)
            kept = []
            for i in keep:
                zz_i, yy_i, xx_i = np.nonzero(lab == i)
                rad = np.sqrt((yy_i - np.nanmedian(bore_y)) ** 2 +
                              (xx_i - np.nanmedian(bore_x)) ** 2)
                # fraction of the part sitting in the outermost 15 vox of the bore
                near_wall = float(np.mean(rad > (r_ref - MARGIN - 15)))
                if near_wall < 0.80:
                    kept.append(i)
                else:
                    print(f'    dropped wall shell: {sizes[i]:.1f} mm3, '
                          f'{100*near_wall:.0f}% of it within 15 vox of the bore',
                          flush=True)
            keep = kept
        sample = np.isin(lab, keep) if keep else np.zeros(env.shape, bool)
        del lab, env

        # Continuity in z. The consensus is decided independently on each
        # slice, so a method flipping between slices steps the area; the
        # specimen itself cannot. Majority over Z_MEDIAN neighbours, on the
        # ENVELOPE only -- cracks are found inside it afterwards and may
        # legitimately be thin in z.
        if Z_MEDIAN > 1 and sample.any():
            _before = sample.sum()
            sample = ndi.median_filter(sample, size=(Z_MEDIAN, 1, 1),
                                       mode='nearest')
            print(f'    z-median {Z_MEDIAN}: envelope '
                  f'{100.0*(sample.sum()-_before)/max(_before,1):+.2f}%',
                  flush=True)

        # SIDE FINS. Thin sheets reaching toward the tube at scattered heights.
        # They carry almost no area, so the slice-to-slice area statistic stays
        # clean, and they span several slices, so the z-median majority keeps
        # them -- but in 3D they are plainly not specimen. The signal is radial
        # REACH against the local trend, measured per angular sector so a
        # barrelled or oval section is not clipped on its long axis.
        sample, _nfin = SM.trim_fins(sample)

        # DROP EDGE FRAGMENTS. Small pieces wedged between the tube wall and the
        # support are not specimen. They survive the skirt rule because that keys
        # on radial REACH, and a fragment sitting inside the specimen radius does
        # not reach outward. Three conditions together, so a genuine detached piece
        # of specimen is not lost: the part must be SMALL, sit NEAR THE TUBE, and
        # sit NEAR AN END of the specimen.
        _lab_p, _np = ndi.label(sample, structure=np.ones((3, 3, 3), bool))
        if _np > 1:
            _sz = np.bincount(_lab_p.ravel()) * VOX_MM3
            _sz[0] = 0
            _main = int(np.argmax(_sz))
            _zs = np.where(sample.any(axis=(1, 2)))[0]
            _z0, _z1 = _zs[0], _zs[-1]
            _span = max(_z1 - _z0, 1)
            _idx = np.array(np.nonzero(sample))
            _cy, _cx = _idx[1].mean(), _idx[2].mean()
            _rmain = np.sqrt((_idx[1] - _cy) ** 2 + (_idx[2] - _cx) ** 2).max()
            _dropped = 0
            _nobead = 0
            for _i in range(1, _np + 1):
                if _i == _main or _sz[_i] == 0:
                    continue
                _part = _lab_p == _i
                _pz, _py, _px = np.nonzero(_part)

                # NO BEADS, NOT SPECIMEN. A detached piece of specimen carries
                # beads with it; frost and ice wisps in the annulus do not.
                # This is what the three-condition edge test below misses,
                # because it requires the part to sit near an END -- these sit
                # at mid-height, and showed in the 3D renders as blobs floating
                # beside the specimen (Glass_75_1000_T6 scan 1, z=485 and 712).
                if beads_v is not None and not beads_v[_part].any():
                    sample[_part] = False
                    _nobead += 1
                    continue

                _small = _sz[_i] < 0.05 * _sz[_main]
                _r = np.sqrt((_py.mean() - _cy) ** 2 + (_px.mean() - _cx) ** 2)
                _near_wall = _r > 0.75 * _rmain
                _h = (_pz.mean() - _z0) / _span
                _near_end = (_h < 0.12) or (_h > 0.88)
                if _small and _near_wall and _near_end:
                    sample[_part] = False
                    _dropped += 1
            if _nobead:
                print(f'    dropped {_nobead} detached parts containing no '
                      f'beads (frost / ice in the annulus, not specimen)',
                      flush=True)
            if _dropped:
                print(f'    dropped {_dropped} edge fragments '
                      f'(small, near tube, near end)', flush=True)
        del _lab_p


        # TRIM THE SUPPORT SKIRT. The support and the ice wedged between it and the
        # tube form thin sheets that reach radially outward: measured max-radius
        # rises to 140% of the specimen median at the bottom of Glass_75 scan 1
        # (7.60 mm against 5.42) and 131% on T6. They carry little AREA, so an
        # equivalent-radius test misses them entirely -- the reach is the signal.
        # Trimmed as a contiguous run inward from each end.
        _idx = np.array(np.nonzero(sample))
        if not _idx.shape[1]:
            # Everything below is skipped when the mask is empty. Say so:
            # silence here is what let Glass_100_1800_T5 scan 1 report OK
            # while writing no files at all.
            print(f'  scan {s}: SPECIMEN MASK IS EMPTY -- no output written. '
                  f'The bore was not found, or no method agreed anywhere.',
                  flush=True)
        if _idx.shape[1]:
            _cy, _cx = _idx[1].mean(), _idx[2].mean()
            _Y, _X = np.ogrid[:sample.shape[1], :sample.shape[2]]
            _rad = np.sqrt((_Y - _cy) ** 2 + (_X - _cx) ** 2)
            _zs = np.where(sample.any(axis=(1, 2)))[0]
            _rmax = np.array([_rad[sample[z]].max() if sample[z].any() else 0.0
                              for z in range((sample).shape[0])])
            _med = np.median(_rmax[_zs])
            _bad = _rmax > 1.12 * _med
            _n = 0
            for z in _zs[::-1]:                      # inward from the bottom
                if _bad[z]:
                    sample[z] = False; _n += 1
                else:
                    break
            for z in _zs:                            # inward from the top
                if _bad[z] and sample[z].any():
                    sample[z] = False; _n += 1
                elif sample[z].any():
                    break
            if _n:
                print(f'    trimmed {_n} support-skirt slices '
                      f'(max-radius > 112% of median)', flush=True)


            # narrow gaps inside the sample
            gaps = np.zeros(sample.shape, dtype=bool)
            vv = tifffile.imread(vol_path(data, s)).astype(np.float32)
            vv = vv / max(f_scale, 1e-6)      # same invariant map as above
            for z in range(zlo, zhi + 1):
                if not sample[z].any():
                    continue
                solid = (vv[z] >= t_air) & sample[z]
                gaps[z] = sample[z] & ~solid
            del vv

            # Isolated pores first, by connected size. The consensus envelope
            # fills internal holes, so every air voxel inside the specimen is
            # in `gaps` and nothing has to be recovered separately afterwards.
            glab, _ = ndi.label(gaps, structure=np.ones((3, 3, 3), bool))
            gsz = np.bincount(glab.ravel()) * VOX_MM3
            gsz[0] = 0
            small = np.where((gsz > 0) & (gsz < VOID_CEILING_MM3))[0]
            void = np.isin(glab, small) if len(small) else np.zeros_like(gaps)
            rest = gaps & ~void
            del glab, gaps

            # Then split what is left by WIDTH, which is the physical
            # difference between a broken bond and a piece that has moved: a
            # bond leaves a gap under 0.25 mm, parted material leaves more.
            # Measured per voxel, not inferred from the size of the connected
            # body -- the crack network is one object, so body size says
            # nothing about how wide any part of it is.
            width = 2.0 * ndi.distance_transform_edt(rest)
            crack = rest & (width <= NARROW_VOX)
            sep = rest & (width > NARROW_VOX)
            del rest, width

            core_m = ndi.binary_erosion(sample, iterations=SKIN)
            v_sample = sample.sum() * VOX_MM3
            v_void = void.sum() * VOX_MM3
            v_cs = (crack & ~core_m).sum() * VOX_MM3
            v_cb = (crack & core_m).sum() * VOX_MM3
            v_sep = sep.sum() * VOX_MM3
            if void_ceiling is None:        # first scan = unloaded control
                void_ceiling = v_cs + v_cb  # crack the method reports with no load
                sep_base = v_sep
            print(f'  scan {s}: platen {n_platen}  punch {n_punch}  '
                  f'sample {v_sample:7.1f}   void {v_void:6.2f}   '
                  f'surface crack {v_cs:7.2f}   body crack {v_cb:6.2f}   '
                  f'separation {v_sep:7.2f} mm3', flush=True)
            _cl, _cn = ndi.label(crack, structure=np.ones((3, 3, 3), bool))
            _csz = np.bincount(_cl.ravel()) * VOX_MM3
            _csz[0] = 0
            n_bodies = int((_csz >= 0.5).sum())   # crack bodies above 0.5 mm3
            del _cl
            results.append((s, v_sample, v_void, v_cs, v_cb, n_bodies, v_sep))
            tifffile.imwrite(os.path.join(out, f'sample_scan{s:02d}.tif'),
                             sample.astype(np.uint8))
            tifffile.imwrite(os.path.join(out, f'crack_scan{s:02d}.tif'),
                             crack.astype(np.uint8))
            tifffile.imwrite(os.path.join(out, f'void_scan{s:02d}.tif'),
                             void.astype(np.uint8))
            tifffile.imwrite(os.path.join(out, f'sep_scan{s:02d}.tif'),
                             sep.astype(np.uint8))
            del sample, void, crack, sep, core_m
            beads_v = None

    with open(os.path.join(out, 'voidcrack_summary.txt'), 'w') as f:
        f.write(f'{name}\n' + '=' * 78 + '\n\n')
        f.write(f'void ceiling {VOID_CEILING_MM3:.2f} mm3, fixed. A connected\n'
                'narrow-gap body below it is an isolated pore; at or above it,\n'
                'part of the crack network.\n\n')
        f.write('SEPARATION = wide opening enclosed by specimen material, taken\n'
                f'per slice, bodies >= {MIN_SEP_MM2:.2f} mm2. Catches material that has\n'
                'parted and moved, which the narrow-gap class excludes by\n'
                'construction.\n\n')
        f.write('Scan 1 carried no load, so its crack and separation are the\n'
                'floor of the method. The baseline-subtracted columns are the\n'
                'ones to quote.\n\n')
        f.write('scan     sample     void   surf crack   body crack  bodies'
                '   separation    crack-base     sep-base\n')
        for r in results:
            f.write(f'{r[0]:>4} {r[1]:>10.1f} {r[2]:>8.2f} {r[3]:>12.2f} '
                    f'{r[4]:>12.2f} {r[5]:>7} {r[6]:>12.2f} '
                    f'{r[3] + r[4] - void_ceiling:>13.2f} '
                    f'{r[6] - sep_base:>12.2f}\n')
    print(f'  wrote {os.path.join(out, "voidcrack_summary.txt")}', flush=True)
    return results


if __name__ == '__main__':
    spam_dir, name = sys.argv[1], sys.argv[2]
    scans = [int(x) for x in sys.argv[3:]]
    analyse(spam_dir, name, scans)
