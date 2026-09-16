"""Bead tracking and bond survival across every specimen, in one table.

The correspondence is built one load step at a time and composed, because a
direct first-to-last match fails.  Its quality is not assumed: the beads are
rigid, so a matched pair must conserve volume, and a ratio away from 1.00 means
the match is wrong however plausible the residual looks.

alpha_b is a LOWER BOUND everywhere.  Only the beads that survive the
correspondence can be followed, and the ones that fail to match are the ones
that moved most -- which are the beads in the damaged zone.
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = ('/mnt/c/Users/cak7496/AppData/Local/Temp/claude/D--wsl/'
        'a62987fb-a2df-41c2-a6a3-fd119ddec5c2/scratchpad/bond')
ORDER = ['Alumina_75_1800_T7', 'Alumina_100_1800_T5', 'Alumina_175_1800_T5',
         'Alumina_75_1000_T5', 'Glass_100_1700_T5_HR', 'Glass_100_1700_T7',
         'Glass_75_1000_T6', 'Glass_75_1700_T5_HR', 'Glass_100_1800_T5']

rows = []
for tag in ORDER:
    d = f'{ROOT}/{tag}'
    if not os.path.isdir(d):
        continue
    ms = sorted(glob.glob(f'{d}/{tag}_beadmatch_*.csv'))
    if not ms:
        continue
    steps = []
    for p in ms:
        m = pd.read_csv(p)
        ok = m[m.accepted]
        if not len(ok):
            continue
        vr = ok.vol_b / ok.vol_a
        steps.append(dict(
            step=os.path.basename(p).split('beadmatch_')[1].replace('.csv', ''),
            accepted=len(ok), total=len(m),
            residual=float(ok.dist.median()),
            vol_ratio=float(vr.median()),
            vol_lo=float(vr.quantile(.1)), vol_hi=float(vr.quantile(.9))))
    sv = glob.glob(f'{d}/{tag}_bond_survival_*.csv')
    surv = None
    if sv:
        s = pd.read_csv(sv[0])
        f = s[s.connected1] if 'connected1' in s.columns else s
        if len(f):
            broke = f.outcome.isin(('detached', 'separated'))
            surv = dict(followed=len(f), alpha_b=float(broke.mean()),
                        survived=float((f.outcome == 'survived').mean()),
                        separated=float((f.outcome == 'separated').mean()),
                        detached=float((f.outcome == 'detached').mean()))
    nb = glob.glob(f'{d}/{tag}_neighbour_2d_vs_3d.csv')
    nbr = None
    if nb:
        n = pd.read_csv(nb[0])
        nbr = dict(precision=float(n.precision.median()),
                   recall=float(n.recall.median()),
                   share=float(n.share_of_all_bonds.median()))
    rows.append(dict(specimen=tag, steps=steps, surv=surv, nbr=nbr))

print(f'{"specimen":<22s} {"step":>6s} {"accepted":>10s} {"resid":>7s} '
      f'{"vol ratio (10-90th)":>24s}')
print('-' * 78)
for r in rows:
    for i, s in enumerate(r['steps']):
        nm = r['specimen'] if i == 0 else ''
        print(f'{nm:<22s} {s["step"]:>6s} {s["accepted"]:5d}/{s["total"]:<4d} '
              f'{s["residual"]:7.2f} {s["vol_ratio"]:8.3f} '
              f'({s["vol_lo"]:.3f}-{s["vol_hi"]:.3f})')

print(f'\n{"specimen":<22s} {"followed":>9s} {"alpha_b":>8s} {"survived":>9s} '
      f'{"separated":>10s} {"detached":>9s}')
print('-' * 74)
for r in rows:
    s = r['surv']
    if not s:
        print(f'{r["specimen"]:<22s} {"-":>9s}   no survival file')
        continue
    print(f'{r["specimen"]:<22s} {s["followed"]:9d} {s["alpha_b"]:8.3f} '
          f'{100*s["survived"]:8.1f}% {100*s["separated"]:9.1f}% '
          f'{100*s["detached"]:8.1f}%')

print(f'\n{"specimen":<22s} {"2D precision":>13s} {"2D recall":>11s} '
      f'{"share of bonds one slice recovers":>35s}')
print('-' * 84)
for r in rows:
    n = r['nbr']
    if not n:
        continue
    print(f'{r["specimen"]:<22s} {100*n["precision"]:12.1f}% '
          f'{100*n["recall"]:10.1f}% {100*n["share"]:34.2f}%')

out = []
for r in rows:
    base = dict(specimen=r['specimen'])
    if r['surv']:
        base.update({f'surv_{k}': v for k, v in r['surv'].items()})
    if r['nbr']:
        base.update({f'nbr_{k}': v for k, v in r['nbr'].items()})
    for s in r['steps']:
        out.append({**base, **{f'match_{k}': v for k, v in s.items()}})
pd.DataFrame(out).to_csv(f'{ROOT}/tracking_and_survival.csv', index=False)
print(f'\n-> {ROOT}/tracking_and_survival.csv')
