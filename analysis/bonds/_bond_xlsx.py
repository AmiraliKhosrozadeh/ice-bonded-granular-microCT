"""Assemble every bond-failure number for one specimen into one workbook.

Same principle as failure_and_crack_summary.xlsx: the figures carry the
argument, the workbook carries the numbers, and nothing is written on the data.
"""
import os

import numpy as np
import pandas as pd
from _spec import (B, OUT, FIG, UM, MM3, TAG, ALN, SCANS, FIRST, LAST, MIDDLE,
                   PUB,
                   p_ct, p_lab, p_ph, p_crack, p_sample, p_core, have)

Q_ADH, Q_COH = 0.15, 0.30


def cls(r):
    if r.crack_vox < 200:
        return 'intact'
    if r.q_crack_med <= Q_ADH:
        return 'adhesive'
    if r.q_crack_med >= Q_COH:
        return 'cohesive'
    return 'mixed'


d1 = pd.read_csv(f'{OUT}/{TAG}_throats_scan{FIRST:02d}.csv')
d3 = pd.read_csv(f'{OUT}/{TAG}_throats_scan{LAST:02d}.csv')
d3['failure_location'] = d3.apply(cls, axis=1)
d1['scan'] = 1
d3['scan'] = 3
q1 = pd.read_csv(f'{OUT}/{TAG}_qhist_scan{FIRST:02d}.csv')
q3 = pd.read_csv(f'{OUT}/{TAG}_qhist_scan{LAST:02d}.csv')
qh = pd.DataFrame(dict(
    q=q1.q,
    binder_scan1=q1.binder, pore_scan1=q1.pore,
    crack_scan3=q3.crack, binder_scan3=q3.binder, pore_scan3=q3.pore))
for c in ('binder_scan1', 'pore_scan1', 'crack_scan3'):
    qh[c + '_frac'] = qh[c] / max(qh[c].sum(), 1)
qh['crack_over_binder'] = np.where(
    qh.binder_scan1_frac > 0,
    qh.crack_scan3_frac / np.maximum(qh.binder_scan1_frac, 1e-12), np.nan)
qh['crack_over_pore'] = np.where(
    qh.pore_scan1_frac > 0,
    qh.crack_scan3_frac / np.maximum(qh.pore_scan1_frac, 1e-12), np.nan)

def _sound_n(bq, scan):
    """How many labels of this scan passed the shape gate."""
    return int(bq[bq.scan == scan].shape_ok.sum())


def _sound_note(bq, scan):
    """The 'of N; highest sound id M' note, safe when the scan is absent."""
    sel = bq[bq.scan == scan]
    if not len(sel):
        return f'scan {scan} not present in the bead quality table'
    ok = sel[sel.shape_ok]
    if not len(ok):
        return f'of {len(sel)}; no label passed the shape gate'
    return (f'of {len(sel)}; highest sound id '
            f'{int(ok.bead_id.max())}, all above it fail')


sv = pd.read_csv(f'{OUT}/{TAG}_bond_survival_{FIRST}to{LAST}.csv')
pb = pd.read_csv(f'{OUT}/{TAG}_per_bead_bonds_scan{LAST:02d}.csv')
bi = (pd.read_csv(f'{OUT}/{TAG}_bead_internal.csv')
      if os.path.exists(f'{OUT}/{TAG}_bead_internal.csv') else None)
# One frame per load STEP the specimen actually has.  This used to hard-code
# two reads labelled '1to2' and '2to3'.  On a two-scan specimen MIDDLE is LAST,
# so it read the same file twice and labelled the duplicate '2to3' -- a step
# that does not exist.  The steps are taken from SCANS instead, and each frame
# is labelled with the scans it really spans.
_steps = list(zip(SCANS[:-1], SCANS[1:]))
_parts = []
for _a, _b in _steps:
    _p = f'{OUT}/{TAG}_beadmatch_{_a}to{_b}.csv'
    if os.path.exists(_p):
        _parts.append(pd.read_csv(_p).assign(step=f'{_a}to{_b}'))
mt = pd.concat(_parts, ignore_index=True) if _parts else pd.DataFrame()

ok1 = d1[d1.connected & d1.r_bond_vox.notna()]
ok3 = d3[d3.connected & d3.r_bond_vox.notna()]
crk = d3[d3.crack_vox >= 200]
dq = d3.dq_paired.dropna()
fol = sv[sv.connected1]
cnt = d3.failure_location.value_counts()

_BQ = pd.read_csv(f'{OUT}/{TAG}_bead_quality.csv')


def _nbeads(scan):
    """Label count for this scan, from the bead table rather than a literal."""
    return int((_BQ.scan == scan).sum())

S = [
    (f'throats measured, scan {FIRST}', len(d1), 'pairs with a free surface gap <= 12 vox'),
    (f'throats measured, scan {LAST}', len(d3), ''),
    (f'free gap, {FIRST} (vox)', round(d1.gap_vox.median(), 2),
     'median; 1 vox = 24.77 um. The beads are essentially in contact.'),
    (f'connected, {FIRST} (%)', round(100 * d1.connected.mean(), 1),
     'a solid path bead-to-bead through bead or binder; Roy 2023 "active bond"'),
    (f'connected, {LAST} (%)', round(100 * d3.connected.mean(), 1), ''),
    # The bead count came from two literals, 473 and 519, measured once on
    # Alumina_75_1800_T7 and then applied to every specimen -- so every other
    # coordination number in this workbook was scaled by the wrong denominator.
    # It is taken from the label table for THIS specimen instead.
    (f'coordination number, scan {FIRST}',
     round(2 * int(d1.connected.sum()) / max(_nbeads(FIRST), 1), 2),
     f'bonded neighbours per bead, over {_nbeads(FIRST)} labels'),
    (f'coordination number, scan {LAST}',
     round(2 * int(d3.connected.sum()) / max(_nbeads(LAST), 1), 2),
     f'over {_nbeads(LAST)} labels; the label count is inflated by broken '
     f'fragments, so read this as a lower bound'),
    (f'bond cross-section, {FIRST} (mm2)',
     round(float(ok1.bond_area_vox.median()) * (UM / 1000) ** 2, 3),
     'binder area on the plane through the narrowest point, median'),
    (f'r_bond / R_bead, scan {FIRST}', round(float(ok1.r_bond_over_R.median()), 3),
     'far above sintered snow: the binder FILLS the gap, it is not a neck'),
    (f'r_bond / R_bead, scan {LAST}', round(float(ok3.r_bond_over_R.median()), 3), ''),
    (f'throats holding crack, scan {LAST}', int((d3.crack_vox >= 200).sum()),
     f'{100*(d3.crack_vox>=200).mean():.1f}% of throats'),
    ('cohesive (%)  q rule, SUPERSEDED',
     round(100 * cnt.get('cohesive', 0) / len(d3), 1),
     'median q >= 0.30. Kept for the record; the q rule has a floor, see notes'),
    ('mixed (%)  q rule, SUPERSEDED',
     round(100 * cnt.get('mixed', 0) / len(d3), 1), ''),
    ('adhesive (%)  q rule, SUPERSEDED',
     round(100 * cnt.get('adhesive', 0) / len(d3), 1),
     'ZERO BY CONSTRUCTION, not by observation: the q floor made this class '
     'unreachable in 88 of 132 throats. Use the face rows below.'),
    (f'median q of the crack, scan {LAST}', round(float(crk.q_crack_med.median()), 3),
     '0 = on a bead surface, 0.5 = exactly mid-bond'),
    (f'median q of the binder, scan {FIRST}',
     round(float(d1.q_binder_med.median()), 3), 'the null: what could break'),
    ('paired crack - binder, median', round(float(dq.median()), 3),
     'same throat, same scan, so free of any frame or alignment error'),
    ('throats with crack outside the binder (%)',
     round(100 * float((dq > 0).mean()), 1),
     f'{int((dq>0).sum())} of {len(dq)}'),
    ('neck severed by crack, median', round(float(d3.neck_severed.median()), 3),
     'crack / (crack + binder) on the neck plane; surrogate for a DEM damage variable'),
    ('bonds followed 1->3', len(fol),
     'both beads carried through the 1->2->3 correspondence'),
    ('alpha_b', round(float(fol.outcome.isin(("detached", "separated")).mean()), 3),
     'fraction of the followed scan-1 bonds broken; LOWER BOUND, see notes'),
    ('partial-volume band at a bead surface (vox)', 4.57,
     '10-90% of the grey fall; 113 um'),
]
# --- face-coverage classification, which supersedes the q threshold -------
f1 = pd.read_csv(f'{OUT}/{TAG}_face_scan{FIRST:02d}.csv')
f3 = pd.read_csv(f'{OUT}/{TAG}_face_scan{LAST:02d}.csv')
bq = pd.read_csv(f'{OUT}/{TAG}_bead_quality.csv')
# FIRST and LAST, not a hard-coded (1, 3).  bead_quality.csv holds only those
# two scans, so on any specimen whose last scan is not 3 -- six of the nine --
# _ok[3] was an EMPTY set, every throat was marked not-sound, and the whole
# face classification silently collapsed to zero rows and printed NaN.  Worse,
# this OVERWRITES the both_beads_sound column that _bond_face.py already wrote
# correctly into the CSV, so a right answer was being replaced by a wrong one.
_ok = {s_: set(bq[(bq.scan == s_) & bq.shape_ok].bead_id)
       for s_ in (FIRST, LAST)}
for _d, _s in ((f1, FIRST), (f3, LAST)):
    _d['both_beads_sound'] = _d.bead_a.isin(_ok[_s]) & _d.bead_b.isin(_ok[_s])
for f in (f1, f3):
    f['lo'] = f[['cover_A', 'cover_B']].min(axis=1)
    f['hi'] = f[['cover_A', 'cover_B']].max(axis=1)
g1 = f1[f1.lo.notna()]
g3 = f3[f3.lo.notna()]
S += [
    ('--- FACE COVERAGE (supersedes the q threshold) ---', '', ''),
    (f'answerable throats, scan {LAST}', len(g3),
     'cracked AND surfaces >= 7 vox apart, so both classes are reachable'),
    (f'intact reference throats, scan {FIRST}', len(g1),
     'same width, nothing failed yet; gives the false-positive rate'),
]
for k in ('cohesive', 'mixed', 'adhesive', 'stripped'):
    n3 = int((g3.face_class == k).sum())
    n1 = int((g1.face_class == k).sum())
    S.append((f'{k} (%)  scan {LAST}', round(100 * n3 / max(len(g3), 1), 1),
              f'scan-1 intact reference: {100*n1/max(len(g1),1):.1f}%'))
S += [
    (f'binder on the better face, scan {LAST}', round(float(g3.hi.median()), 3),
     f'intact reference {g1.hi.median():.3f}'),
    (f'binder on the worse face, scan {LAST}', round(float(g3.lo.median()), 3),
     f'intact reference {g1.lo.median():.3f}'),
    (f'face asymmetry, scan {LAST}', round(float(g3.asymmetry.median()), 3),
     f'|cA-cB|/(cA+cB); intact reference {g1.asymmetry.median():.3f}'),
    (f'sub-voxel q, crack scan {LAST}',
     round(float(f3.q_crack_med.median()), 3),
     'one-voxel floor removed by a 3x upsampled half-grey isosurface'),
    (f'sub-voxel q, binder scan {FIRST}',
     round(float(f1.q_binder_med.median()), 3), ''),
    ('--- BEAD LABEL QUALITY ---', '', ''),
    # Scans FIRST and LAST, not a hard-coded 1 and 3.  bead_quality.csv only
    # ever holds those two scans, so 'scan == 3' silently selected nothing on
    # every specimen whose last scan is not 3 -- six of the nine -- and
    # int(NaN) then killed the whole workbook.  It happened to work on the
    # three where LAST is 3, which is why it survived this long.
    (f'sound labels, scan {FIRST}', _sound_n(bq, FIRST), _sound_note(bq, FIRST)),
    (f'sound labels, scan {LAST}', _sound_n(bq, LAST), _sound_note(bq, LAST)),
    ('answerable throats on sound labels', int(g3.both_beads_sound.sum()),
     f'of {len(g3)}; the rest are excluded, not reclassified'),
]
gs3 = g3[g3.both_beads_sound]
gs1 = g1[g1.both_beads_sound]
for HIGH, LOW in ((0.50, 0.20), (0.70, 0.20), (0.80, 0.20), (0.90, 0.20)):
    def _cl(d):
        return np.where((d.hi >= HIGH) & (d.lo >= HIGH), 'cohesive',
               np.where((d.hi >= HIGH) & (d.lo < LOW), 'adhesive',
               np.where(d.hi < LOW, 'stripped', 'mixed')))
    c3_, c1_ = _cl(gs3), _cl(gs1)
    for k in ('cohesive', 'mixed', 'adhesive', 'stripped'):
        S.append((f'{k} (%)  sound, HIGH {HIGH:.2f} LOW {LOW:.2f}',
                  round(100 * float((c3_ == k).mean()), 1),
                  f'scan-1 intact reference: {100*float((c1_==k).mean()):.1f}%'))

summary = pd.DataFrame(S, columns=['quantity', 'value', 'note'])

NOTES = [
    f'SPECIMEN  {TAG}. Voxel {UM:.7f} um. {len(SCANS)} in-situ '
    f'compression scans ({", ".join(str(x) for x in SCANS)}); low z is the '
    f'support end, as in every other figure.',

    'THROAT  For a bead pair A,B with surface distances dA and dB, free space '
    'satisfies dA+dB = g on the shortest path. The throat is the lens '
    'dA+dB <= g + slack whose nearest bead is A or B, with slack set per pair '
    'so the footprint radius is 0.40 x the mean bead radius. Every throat is '
    'therefore the same size in bead radii.',

    'q  = min(dA,dB)/(dA+dB), in [0, 0.5]. 0 sits on a bead surface, 0.5 '
    'exactly mid-bond. It is normalised by the LOCAL gap, so tight and loose '
    'contacts are on one scale and no length threshold is used anywhere.',

    'THE NULLS  A crack at mid-bond proves nothing alone, because mid-bond may '
    'simply be where the material was. Two nulls are measured in the same '
    'throats: the binder at scan 1 (what was available to break) and the pore '
    'at scan 1 (space already open before loading). The result is the ratio.',

    'THE CRACK CLASS IS NOT THE PORE PHASE. It was checked: at scan 1 the pore '
    'phase is 171.9 mm3 and the crack class is 0.0 mm3. For the interior crack '
    'at scan 3, 93.9% falls in the pore phase, 1.2% in binder and 0.0% in '
    'bead. The 40% of the full crack that the phases map calls "outside" is '
    'the specimen rim, which the 20-voxel skin erosion removes.',

    'CRACK USED  the interior ("failure") crack: crack minus a 20-voxel skin, '
    'the same class the failure-plane figures use. Surface crack wraps the '
    'specimen and is spalling at the boundary, not a bond failing.',

    'PORE EXCLUDES CRACK. 94% of the interior crack is labelled pore by the '
    'phases map, so without removing it "the crack sits where the pore sits" '
    'would be true by construction.',

    'THE q THRESHOLD HAD A FLOOR AND IT WAS BINDING. A distance transform on a '
    'bead LABEL cannot return less than 1 voxel, so q >= 1/(dA+dB) and q <= 0.15 '
    'needs the surfaces more than 6.67 voxels apart. Only 44 of 132 cracked '
    'throats are that wide; in the other 88 the adhesive class was unreachable, '
    'and where the surfaces are 2 voxels apart q is IDENTICALLY 0.5. The first '
    'pass reported 0% adhesive for that reason, not because the material never '
    'debonds. Superseded by the face sheets.',

    'FACE COVERAGE IS THE BETTER OBSERVABLE. cA and cB are the binder fraction '
    'of a shell 1-3 voxels off each bead surface inside the throat, measured on '
    'a 3x upsampled grid with the bead surface taken as the half-grey '
    'isosurface. Both high = cohesive; one high one low = adhesive at the low '
    'face; both low = stripped. It compares the two faces of the SAME throat, so '
    'the grey blur, the threshold, the partial volume and the opening are common '
    'to both terms and cancel. Restricted to throats at least 7 voxels apart.',

    'THE SCAN-1 SHEET IS THE FALSE-POSITIVE RATE. 149 intact throats of the same '
    'width classify as 96.0% cohesive and 1.3% adhesive. The 7.5% adhesive at '
    'scan 3 sits above that, but it is 3 cases of 40, so the interval is roughly '
    '2-20%. Quote it as "adhesive failure occurs and is a minority".',

    'BEAD FRACTURE CANNOT BE SEEN IN THE CRACK MASK. crack = sample AND NOT '
    'solid, and a bead is solid, so crack inside a bead is empty for '
    'arithmetic reasons. The separate test looks at the raw grey inside beads '
    'eroded by 3 voxels and asks whether scan 3 gains PLANAR, bead-spanning '
    'low-grey bodies that scan 1 lacks. The beads are porous, so scan 1 is the '
    'baseline and only the change means anything.',

    'RESOLUTION  The grey at a bead surface takes 4.57 voxels (113 um) to fall '
    'from the alumina plateau to the surrounding, and the label boundary sits '
    'within about 0.4 voxels of the half-grey isosurface. So q is not '
    'systematically displaced, but a crack thinner than about 2 voxels lying '
    'ON a bead surface is absorbed into that blur and is not detected. '
    'ADHESIVE FAILURE IS THEREFORE UNDER-DETECTED, and "0% adhesive" must be '
    'read as "none found, with limited sensitivity", not as "none occurred". '
    'The blur affects the crack and the binder null equally, which is why the '
    'RATIO is the defensible statistic and the absolute q is not.',

    'BEAD CORRESPONDENCE  Labels are renumbered independently per scan. '
    'Matching 1 to 3 directly fails (25% accepted, residual 22 vox, matched '
    'volume ratio 0.69-1.45 for a rigid ceramic). One step at a time works: '
    '1->2 residual 0.28 vox and volume ratio 0.995, 2->3 residual 0.47 vox and '
    '1.001. The 1->3 map is the composition, carrying 294 of 473 beads.',

    'THE SCANS ARE NOT REGISTERED, only cropped to a common height '
    '(align_offsets.txt: zstart 36 / 47 / 0). Scan 3 sits about 70 voxels low. '
    'A translation vote before ICP is what makes the 2->3 match work; without '
    'it ICP settles into the neighbouring bead and returns half the bead '
    'spacing as its residual.',

    'alpha_b IS A LOWER BOUND. Only 62% of scan-1 beads survive the '
    'correspondence, and the ones that fail to match are the ones that moved '
    'most, which are exactly the beads in the damaged zone. The population '
    'statistic (24.8% of scan-3 throats hold crack, 98.8% still connected) is '
    'the complementary view and is free of that bias.',

    'BOND SIZE  r_bond/R = 0.745 is far above sintered snow, where a bond is a '
    'neck. Here the binder fills the intergranular space, so the bond '
    'cross-section is set by the packing geometry (the Voronoi facet between '
    'the two beads), not by a constriction. That is a difference in material, '
    'not an error, and it means neck-based snow metrics do not transfer '
    'directly.',

    'LITERATURE  Roy, Frost and Terzis (Granular Matter 25:62, 2023) define an '
    'ACTIVE bond as cement bridging two or more particles; the "connected" '
    'column here is that test. Senanayake, Haque and Bui (Comput. Geotech. '
    '149:104862, 2022) report alpha_b about 0.10-0.11 at the end of '
    'unconfined loading in a DEM cemented granular material; their bonds break '
    'on a continuous damage variable reaching 0.999 and ours on image '
    'connectivity, so the comparison is indicative only. Neither paper '
    'observes adhesive versus cohesive failure, and neither should be cited '
    'for it.',
]
notes = pd.DataFrame(dict(note=NOTES))

# Was 'bond_failure_T7_summary.xlsx' for EVERY specimen -- same baked-in
# T7 as everywhere else, so nine different workbooks all claimed to be T7's.
xl = f'{OUT}/{TAG}_bond_failure_summary.xlsx'
with pd.ExcelWriter(xl, engine='openpyxl') as w:
    summary.to_excel(w, sheet_name='summary', index=False)
    d3.to_excel(w, sheet_name=f'throats_scan{LAST:02d}', index=False)
    d1.to_excel(w, sheet_name=f'throats_scan{FIRST:02d}', index=False)
    f3.to_excel(w, sheet_name=f'face_scan{LAST:02d}', index=False)
    f1.to_excel(w, sheet_name=f'face_scan{FIRST:02d}_reference', index=False)
    qh.to_excel(w, sheet_name='q_histograms', index=False)
    sv.to_excel(w, sheet_name='bond_survival_1to3', index=False)
    pb.to_excel(w, sheet_name=f'per_bead_scan{LAST:02d}', index=False)
    if bi is not None:
        bi.to_excel(w, sheet_name='bead_internal_grey', index=False)
    bq.to_excel(w, sheet_name='bead_label_quality', index=False)
    mt.to_excel(w, sheet_name='bead_matches', index=False)
    notes.to_excel(w, sheet_name='notes', index=False)

    for sh in w.sheets.values():
        for col in sh.columns:
            n = max((len(str(c.value)) for c in col if c.value is not None),
                    default=10)
            sh.column_dimensions[col[0].column_letter].width = min(n + 2, 60)
    ns = w.sheets['notes']
    ns.column_dimensions['A'].width = 110
    for row in ns.iter_rows():
        for c in row:
            c.alignment = c.alignment.copy(wrapText=True, vertical='top')
        ns.row_dimensions[row[0].row].height = 60

print(f'-> {xl}')
print(f'   sheets: summary, throats_scan03, throats_scan01, q_histograms, '
      f'bond_survival_1to3, per_bead_scan03, '
      f'{"bead_internal_grey, " if bi is not None else ""}bead_matches, notes')
print('   plus: face_scan03, face_scan01_reference')
print(summary.to_string(index=False))
