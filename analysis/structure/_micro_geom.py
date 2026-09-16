"""Column geometry and three-phase classification, recomputed per scan.

WHY THIS EXISTS.  Nothing already on disk can support a consistent section 3.6.
Three separate pipelines produced microstructure numbers and no two of them
share a definition:

  * the four ALUMINA Dragonfly scripts still carry T_AIR_ICE = 0.0 and
    T_ICE_GLASS = 0.0 with their scaffold TODO comments -- the thresholds were
    never filled in, so those scripts cannot be the source of anything;
  * the geometry constants in those scripts are template copies.  CYL_RADIUS_UM
    = 5320.52 with FIRST_SLICE 80 and LAST_SLICE 1026 appears verbatim under
    Alumina_100_1800_T5, Alumina-75-1800-T7 AND Glass_75_1700_T5_HR, three
    different specimens, each line still marked "<-- REPLACE";
  * the sand PuMA run reports the column growing from 0.402 to 0.669 porosity
    while it is being compressed, which is not physical.

So every quantity here is measured from the scan itself, with no constant
carried in from another specimen and none frozen across load steps.

PER-SCAN, NOT FROZEN.  The reconstructed grey scale drifts between load steps
by up to a factor of three to six (section 2.5.1 says so for the correlation,
and the hand-tuned glass thresholds show it directly: G1 goes 7000 -> 2139 ->
2015 for air/ice across its three stages).  A threshold frozen across scans is
therefore wrong, and freezing one is exactly what produced the impossible sand
porosity trend.  Every scan is classified on its OWN pooled in-column
histogram, which is also what section 2.4.2 describes.

COLUMN.  Granular material has grain-scale texture; the platens, the tube and
the empty gap do not.  The local standard deviation over a 7-voxel window
separates them, and the column is the longest axial run whose textured area
fraction stays above half its own plateau.  This is the construction already
validated for the sand specimens in _sand_geom.py, retuned here for the beads.
"""
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu

VOX_UM = 24.7660229
TEX_BOX = 7


def texture(b):
    """Local standard deviation -- high in a granular packing, low in metal."""
    m = ndi.uniform_filter(b, TEX_BOX)
    v = ndi.uniform_filter(b * b, TEX_BOX) - m * m
    return np.sqrt(np.maximum(v, 0.0))


def specimen_mask(a):
    """In-specimen mask for one slice: section 2.4.1, from the slice itself.

    Threshold solid against the surrounding air, keep the largest connected
    component, fill the interior holes.  The holes ARE the trapped air, so
    filling them is what makes the denominator the specimen interior rather
    than its solid part.  A grey threshold is used and not the texture, because
    the texture level is set by the grain size and would have to be retuned per
    material; the solid/air step is the same measurement in all three.
    """
    b = a[::2, ::2]
    t = float(threshold_multiotsu(b.ravel()[::7], classes=2)[0])
    m = b > t
    m = ndi.binary_closing(m, np.ones((7, 7)), border_value=0)
    lab, n = ndi.label(m)
    if n == 0:
        return np.zeros(a.shape, bool)
    sz = np.bincount(lab.ravel())
    sz[0] = 0
    m = ndi.binary_fill_holes(lab == int(np.argmax(sz)))
    m = ndi.binary_erosion(m, np.ones((5, 5)), border_value=0)
    return np.repeat(np.repeat(m, 2, axis=0), 2, axis=1)[:a.shape[0], :a.shape[1]]


def _tex_frac(a):
    """Fraction of the specimen cross-section carrying grain-scale texture."""
    m = specimen_mask(a)[::2, ::2]
    if m.sum() < 500:
        return 0.0, m
    b = a[::2, ::2]
    t = texture(b)
    lvl = float(np.percentile(t[m], 50))
    return lvl, m


def column_range(files, step=8):
    """Axial extent of the granular column, in full-resolution slice indices.

    The platens and the specimen both fill the bore, so a mask area profile
    cannot separate them.  The grain-scale texture can: inside the packing the
    median in-mask local standard deviation is several times its value across a
    solid platen face.
    """
    zs = list(range(0, len(files), step))
    lvl, area = [], []
    for z in zs:
        a = tifffile.imread(files[z]).astype(np.float32)
        l, m = _tex_frac(a)
        lvl.append(l)
        area.append(float(m.sum()))
    lvl = np.asarray(lvl)
    plateau = float(np.percentile(lvl, 90))
    ok = (lvl > 0.5 * plateau) & (np.asarray(area) > 0.2 * max(area))
    best, i = (0, 0, 0), 0
    while i < len(ok):
        if ok[i]:
            j = i
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            if j - i > best[0]:
                best = (j - i, i, j)
            i = j + 1
        else:
            i += 1
    _, i0, i1 = best
    return zs[i0], min(zs[i1] + step, len(files) - 1), plateau


def phase_thresholds(files, z0, z1, n=24):
    """Two grey levels (air|ice, ice|grain) from the pooled in-column histogram."""
    rng = np.random.default_rng(1)
    vals = []
    for z in np.linspace(z0, z1, n).astype(int):
        a = tifffile.imread(files[z]).astype(np.float32)
        m = specimen_mask(a)
        if m.sum() < 1000:
            continue
        s = a[m]
        vals.append(s[rng.choice(s.size, min(200000, s.size), replace=False)])
    v = np.concatenate(vals)
    t = threshold_multiotsu(v, classes=3)
    return float(t[0]), float(t[1]), v
