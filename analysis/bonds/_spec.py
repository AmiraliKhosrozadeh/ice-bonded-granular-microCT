"""One place that says which specimen the pipeline is running on.

Every analysis script imports B, OUT, UM, TAG, ALN and SCANS from here instead
of hard-coding Alumina_75_1800_T7.  Pick the specimen with the SPEC environment
variable:

    SPEC=Glass_75_1000_T6 python _bond_throat.py 1 2

ALN is '_aligned' for the specimens whose data folder carries aligned volumes
and '' for Glass_100_1800_T5, which only has the raw per-scan stacks.  That
specimen is still self-consistent within a scan, which is all the throat
statistics need; only the cross-scan tracking is affected, and that is reported
per specimen rather than assumed.

VOXEL is 24.7660229 um for every specimen -- the same scanner setting, confirmed
in each spam folder's own scripts, except Glass_100_1800_T5 whose folder records
none and which is assumed the same.
"""
import os

R = '/mnt/e/RPTU-images/CT_images'
UM_DEFAULT = 24.7660229

REG = {
    # ---- Alumina -------------------------------------------------------
    # Three of these carry their results_voidcrack stacks in the RAW frame,
    # so aln='' for them: pairing an aligned label stack with a raw crack
    # stack offsets the two in z and is silently wrong.  check_frame() below
    # verifies the choice per scan instead of trusting it.
    'Alumina_75_1800_T7': dict(
        spam=f'{R}/Alumina/Alumina-75-1800-T7/Alumina_75_1800_T7_spam',
        scans=(1, 2, 3), aln='_aligned', phases='pipeline',
        beads_break=True),
    'Alumina_100_1800_T5': dict(
        spam=f'{R}/Alumina/Alumina_100_1800_T5/Alumina_100_1800_T5_spam',
        scans=(1, 2, 3), aln='', phases='pipeline',
        beads_break=True),
    'Alumina_175_1800_T5': dict(
        spam=f'{R}/Alumina/Alumina_175_1800_T5/Alumina_175_1800_T5_spam',
        scans=(1, 2), aln='', phases='build', beads_break=True),
    'Alumina_75_1000_T5': dict(
        spam=f'{R}/Alumina/Alumina_75_1000_T5/Alumina_75_1000_T5_spam',
        scans=(1, 2), aln='', phases='build', beads_break=True),
    # ---- Glass ---------------------------------------------------------
    'Glass_100_1700_T5_HR': dict(
        spam=f'{R}/Glass/Glass_100_1700_T5_HR/Glass_T5_HR_spam',
        scans=(1, 2), aln='_aligned', phases='build', beads_break=False,
        t_air={1: 2500.0, 2: 3802.07}),
    'Glass_100_1700_T7': dict(
        spam=f'{R}/Glass/Glass_100_1700_T7/Glass_1700_spam',
        scans=(1, 2, 3, 4), aln='_aligned', phases='build', beads_break=False,
        # the four copies inside the spam folder all read 6200, which is a
        # copy-paste; the per-scan dragonfly scripts differ and are used
        t_air={1: 8750.0, 2: 2800.0, 3: 1600.0, 4: 1650.0}),
    'Glass_75_1000_T6': dict(
        spam=f'{R}/Glass/Glass_75_1000_T6/Glass_T6_spam',
        scans=(1, 2), aln='_aligned', phases='build', beads_break=False,
        t_air={1: 5055.5, 2: 1184.0}),
    'Glass_75_1700_T5_HR': dict(
        spam=f'{R}/Glass/Glass_75_1700_T5_HR/Glass_75_spam',
        scans=(1, 2, 3), aln='_aligned', phases='build', beads_break=False,
        t_air={1: 7000.0, 2: 2139.4, 3: 2015.4}),
    'Glass_100_1800_T5': dict(
        spam=f'{R}/Glass/Glass_spam',
        scans=(1, 2), aln='', phases='build', beads_break=False,
        # scan 2 has no recorded value; it falls back to the computed midpoint
        t_air={1: 6800.0}),
}

TAG = os.environ.get('SPEC', 'Alumina_75_1800_T7')
if TAG not in REG:
    raise SystemExit(f'unknown SPEC {TAG!r}; known: {sorted(REG)}')
_S = REG[TAG]
B = _S['spam']
ALN = _S['aln']
SCANS = _S['scans']
UM = _S.get('um', UM_DEFAULT)
MM3 = (UM / 1000.0) ** 3
FIRST, LAST = SCANS[0], SCANS[-1]
# the tracking chain: one load step at a time, composed
CHAIN = list(SCANS)
MIDDLE = SCANS[1] if len(SCANS) > 2 else SCANS[-1]
# Glass beads do not fracture -- checked directly on the images --
# so the internal-grey bead-fracture test is skipped there.  The
# bead SHAPE gate still runs everywhere: it is about segmentation
# quality, not physics.
BEADS_BREAK = _S.get('beads_break', True)
T_AIR = _S.get('t_air', {})

OUT = ('/mnt/c/Users/cak7496/AppData/Local/Temp/claude/D--wsl/'
       f'a62987fb-a2df-41c2-a6a3-fd119ddec5c2/scratchpad/bond/{TAG}')
FIG = f'{OUT}/figs'
os.makedirs(FIG, exist_ok=True)

PUB = f'{R}/paper_figures/bond_failure/{TAG}'


def p_ct(s):
    return f'{B}/data/ct_scan{s:02d}{ALN}.tif'


def p_lab(s):
    return f'{B}/data/bead_labels_scan{s:02d}{ALN}.tif'


def p_ph(s):
    return f'{B}/data/phases_scan{s:02d}{ALN}.tif'


def p_crack(s):
    return f'{B}/results_voidcrack/crack_scan{s:02d}.tif'


def p_sample(s):
    return f'{B}/results_voidcrack/sample_scan{s:02d}.tif'


def p_core(s):
    return f'{OUT}/core_scan{s:02d}.tif'


def have(s):
    return all(os.path.exists(p) for p in
               (p_ct(s), p_lab(s), p_crack(s), p_sample(s)))


# --- reading volumes that are not memory-mappable ---------------------------
# The Alumina stacks are plain uncompressed TIFF and memmap directly.  Several
# Glass stacks are compressed, where memmap raises "image data are not
# memory-mappable".  openv() returns a memmap when it can and a lazy per-page
# reader when it cannot, supporting v[z], v[a:b] and v[(zslice, yslice, xslice)]
# the same way, so nothing downstream has to know which it got.
import numpy as _np
import tifffile as _tf


class _Pages:
    def __init__(self, path):
        self._f = _tf.TiffFile(path)
        self._s = self._f.series[0]
        self.shape = tuple(self._s.shape)
        self.dtype = self._s.dtype

    def _page(self, i):
        return self._s.pages[int(i)].asarray()

    def __getitem__(self, k):
        if isinstance(k, (int, _np.integer)):
            return self._page(k)
        if isinstance(k, slice):
            return _np.stack([self._page(i)
                              for i in range(*k.indices(self.shape[0]))])
        if isinstance(k, tuple):
            z, rest = k[0], k[1:]
            if isinstance(z, slice):
                idx = range(*z.indices(self.shape[0]))
                return _np.stack([self._page(i)[rest] for i in idx])
            return self._page(z)[rest]
        raise TypeError(f'unsupported index {k!r}')


_CACHE = os.path.join(OUT, 'uncompressed')


def openv(path, mode='r'):
    """memmap the volume, decompressing once into a cache if it will not map.

    The per-throat analysis indexes a small box out of four volumes ~1000 times
    per scan.  Against a compressed TIFF that means decompressing whole pages
    on every access, which is unusable.  So a stack that will not memmap is
    written out once, uncompressed, beside the results and memmapped from
    there.  It costs disk and one pass; it turns hours into minutes.
    """
    try:
        return _tf.memmap(path, mode=mode)
    except (ValueError, MemoryError):
        pass
    os.makedirs(_CACHE, exist_ok=True)
    dst = os.path.join(_CACHE, os.path.basename(path))
    if not os.path.exists(dst):
        src = _Pages(path)
        out = _tf.memmap(dst, shape=src.shape, dtype=src.dtype, mode='w+')
        for z in range(src.shape[0]):
            out[z] = src[z]
        out.flush()
        del out, src
    return _tf.memmap(dst, mode='r')


# --- the frame the voidcrack outputs actually live in -----------------------
# "aligned" in this pipeline means cropped to a common height, not registered,
# and on three Alumina specimens the crack/sample stacks were written in the
# RAW frame instead.  Pairing an aligned label stack with a raw crack stack
# offsets them in z and pairs a bond with crack from a different height, which
# is silent and wrong.  This checks the choice rather than trusting it.
def check_frame(scan, loud=True):
    import tifffile as _t
    def _sh(p):
        try:
            with _t.TiffFile(p) as f:
                return tuple(f.series[0].shape)
        except Exception:
            return None
    smp, lab = _sh(p_sample(scan)), _sh(p_lab(scan))
    if smp is None or lab is None:
        return None
    if smp == lab:
        return True
    if loud:
        print(f'FRAME MISMATCH {TAG} scan {scan}: sample {smp} against '
              f'labels {lab} -- ALN={ALN!r} is the wrong frame', flush=True)
    return False
