"""Does the planar-cracking claim survive a correction for multiple testing?

The sweep in _planar_cracks.py tries six neighbourhood sizes and two statistics,
so twelve tests per specimen.  At a 5% threshold that is about 0.6 false
positives per specimen by chance, and across nine specimens roughly five
specimens showing "a hit somewhere" is the NULL expectation, not evidence.
Six do.  Reporting that as planar cracking would be a multiple-comparisons
artefact.

The test is therefore restated with one PRE-SPECIFIED neighbourhood, chosen on
physical grounds rather than from the results: eps = one grain diameter, which
links break events that share a grain or sit one grain apart, the smallest
neighbourhood at which a connected crack path could be traced.  Benjamini-
Hochberg is then applied across the nine specimens.  The sweep is retained as a
sensitivity check and reported in full.
"""
import glob
import os

import numpy as np
import pandas as pd

SC = ('/mnt/c/Users/cak7496/AppData/Local/Temp/claude/D--wsl/'
      'a62987fb-a2df-41c2-a6a3-fd119ddec5c2/scratchpad')
PRIMARY = 1.0        # pre-specified: one grain diameter
ID = {'Alumina_100_1800_T5': 'A1', 'Alumina_175_1800_T5': 'A2',
      'Alumina_75_1000_T5': 'A3', 'Alumina_75_1800_T7': 'A4',
      'Glass_75_1700_T5_HR': 'G1', 'Glass_75_1000_T6': 'G2',
      'Glass_100_1700_T7': 'G3', 'Glass_100_1700_T5_HR': 'G4',
      'Glass_100_1800_T5': 'G5'}
SCOPE = {'A1', 'A2', 'A3', 'G1', 'G4', 'G5'}      # -5 C, main text

rows = []
for f in sorted(glob.glob(f'{SC}/bond/*/*_planar_sweep.csv')):
    d = pd.read_csv(f)
    tag = d.specimen.iloc[0]
    pid = ID.get(tag, tag)
    pr = d[np.isclose(d.eps_d, PRIMARY)]
    if not len(pr):
        continue
    pr = pr.iloc[0]
    n_hit = int(((d.p_biggest <= 0.05) | (d.p_rms <= 0.05)).sum())
    rows.append(dict(id=pid, tag=tag, scope='-5C' if pid in SCOPE else 'other',
                     clusters=int(pr.clusters), biggest=int(pr.biggest),
                     biggest_null=pr.biggest_null, p_biggest=pr.p_biggest,
                     rms_R=pr.rms_R, rms_null=pr.rms_null, p_rms=pr.p_rms,
                     sweep_hits=n_hit, sweep_tests=2 * len(d)))

r = pd.DataFrame(rows).sort_values(['scope', 'id'])


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.full(len(p), np.nan)
    q = p[ok]
    n = len(q)
    order = np.argsort(q)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, q[i] * n / (rank + 1))
        adj[i] = prev
    out[ok] = adj
    return out


for stat in ('p_biggest', 'p_rms'):
    r[stat + '_bh'] = bh(r[stat].values)

print(f'PRIMARY TEST, pre-specified eps = {PRIMARY:.1f} grain diameters\n')
print(f'{"ID":<4s}{"scope":>7s}{"breaks in":>11s}{"largest cluster":>17s}'
      f'{"p":>7s}{"BH":>7s}{"  RMS/R obs":>12s}{"null":>7s}{"p":>7s}{"BH":>7s}')
print('-' * 88)
for x in r.itertuples():
    print(f'{x.id:<4s}{x.scope:>7s}{x.clusters:11d}'
          f'{x.biggest:9d} vs {x.biggest_null:<4.0f}{x.p_biggest:7.3f}'
          f'{x.p_biggest_bh:7.3f}{x.rms_R:12.2f}{x.rms_null:7.2f}'
          f'{x.p_rms:7.3f}{x.p_rms_bh:7.3f}')

sig = r[(r.p_biggest_bh <= 0.05) | (r.p_rms_bh <= 0.05)]
print(f'\nSpecimens significant after Benjamini-Hochberg: '
      f'{len(sig)} of {len(r)}'
      + (f'  ({", ".join(sig.id)})' if len(sig) else ''))

print(f'\nSWEEP as a sensitivity check (uncorrected, 12 tests per specimen):')
print(f'{"ID":<4s}{"hits":>6s}{"of":>5s}   expected by chance at 5%: '
      f'{0.05 * r.sweep_tests.iloc[0]:.1f} per specimen')
for x in r.itertuples():
    print(f'{x.id:<4s}{x.sweep_hits:6d}{x.sweep_tests:5d}')
tot_hits = int(r.sweep_hits.sum())
exp = 0.05 * int(r.sweep_tests.sum())
print(f'\nTotal sweep hits {tot_hits} against {exp:.1f} expected by chance '
      f'-- the sweep is consistent with no effect.')
r.to_csv(f'{SC}/bond/planar_crack_test.csv', index=False)
print(f'-> {SC}/bond/planar_crack_test.csv')
