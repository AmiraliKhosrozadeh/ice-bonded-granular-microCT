"""Specimen-level tables for the revision (reviewer items 1, 2, 3, 7, 15).

Reads the bond pipeline output on E:, the DIC kinematics summary, the planar
null-test sweeps and the core transport table, and writes three CSVs to
scripts/data plus the LaTeX rows for the manuscript:

    bond_geometry.csv          per specimen, first and last scan: grains passing
                               the gate, throats, ice-bonded throats, bonded
                               coordination, gap, neck radius, bridge volume
    failure_mode_thresholds.csv  cohesive / adhesive per specimen at every C_HI
    specimen_summary.csv       one row per specimen: scans, shortening,
                               survival census, cracked and answerable throats,
                               local rotation, clustering test, spanning air

Coordination here is ICE-BONDED THROATS PER GRAIN over the grains that pass the
shape gate, which is what the pipeline's contact network is (a throat exists
for every pair whose lens is ice-connected; gap up to ~1 mm).  It is not a
touching-contact count.
"""
import glob
import os

import numpy as np
import pandas as pd

BF = "E:/RPTU-images/CT_images/paper_figures/bond_failure"
FC = "E:/RPTU-images/CT_images/paper_figures/failure_classification"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
VOX = 24.7660229  # um

ID = {"Alumina_100_1800_T5": "A1", "Alumina_175_1800_T5": "A2",
      "Alumina_75_1000_T5": "A3", "Alumina_75_1800_T7": "A4",
      "Glass_75_1700_T5_HR": "G1", "Glass_75_1000_T6": "G2",
      "Glass_100_1700_T7": "G3", "Glass_100_1700_T5_HR": "G4",
      "Glass_100_1800_T5": "G5"}
ORDER = ["G1", "G2", "G3", "G4", "G5", "A1", "A2", "A3", "A4"]
C_HI = [0.5, 0.7, 0.8, 0.9]


def throat_stats(tag, scan):
    f = f"{BF}/{tag}/data/{tag}_throats_scan{scan:02d}.csv"
    t = pd.read_csv(f)
    q = pd.read_csv(f"{BF}/{tag}/data/{tag}_bead_quality.csv")
    q = q[q.scan == scan]
    gated = q[q.radius_ok & q.shape_ok].bead_id
    ok = t.bead_a.isin(gated) & t.bead_b.isin(gated)
    t = t[ok]
    b = t[t.connected.astype(bool)]
    n_gr = len(gated)
    R = 0.5 * (b.r_a_vox + b.r_b_vox)
    return dict(grains=n_gr, throats=len(t), bonded=len(b),
                z_bar=round(2 * len(b) / n_gr, 2),
                gap_med_um=round(float(b.gap_um.median()), 0),
                gap_p90_um=round(float(b.gap_um.quantile(0.9)), 0),
                neck_over_R_med=round(float(b.r_bond_over_R.median()), 2),
                neck_over_R_p10=round(float(b.r_bond_over_R.quantile(0.1)), 2),
                neck_um_med=round(float((b.r_bond_vox * VOX).median()), 0),
                bridge_mm3_med=round(float((b.binder_vox * (VOX / 1000) ** 3).median()), 4),
                cracked=int((t.crack_frac > 0).sum()))


def survival(tag):
    f = glob.glob(f"{BF}/{tag}/data/{tag}_bond_survival_*.csv")[0]
    s = pd.read_csv(f)
    s = s[s.outcome != "lost"]
    n = len(s)
    return dict(followed=n,
                separated_pct=round(100 * (s.outcome == "separated").mean(), 1),
                detached_pct=round(100 * (s.outcome == "detached").mean(), 1),
                survived_cracked=int(((s.outcome == "survived") & (s.crack_frac3 > 0)).sum()))


def main():
    every = pd.read_excel(f"{BF}/all_specimens_bond_failure.xlsx", "every_throat")
    every["id"] = every.specimen.map(ID)
    funnel = pd.read_csv(f"{BF}/bond_funnel.csv").set_index("specimen")
    vols = pd.read_excel(f"{FC}/failure_and_crack_summary.xlsx", "per_scan_volumes")
    core = pd.read_csv(os.path.join(DATA, "section36_core.csv"))
    kin = pd.read_csv(os.path.join(DATA, "kinematics_summary.csv"))
    sweeps = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(DATA, "planar_sweep", "*.csv"))])
    tight = pd.read_csv(os.path.join(DATA, "crack_volumes_tight.csv"))

    # ---- bond geometry, first and last scan --------------------------------
    geo = []
    for tag, pid in ID.items():
        scans = sorted(int(os.path.basename(f)[-6:-4]) for f in glob.glob(f"{BF}/{tag}/data/{tag}_throats_scan??.csv"))
        for lab, sc in (("first", scans[0]), ("last", scans[-1])):
            geo.append(dict(id=pid, specimen=tag, which=lab, scan=sc, **throat_stats(tag, sc)))
    geo = pd.DataFrame(geo)
    geo["id"] = pd.Categorical(geo.id, ORDER)
    geo = geo.sort_values(["id", "which"], ascending=[True, False])
    geo.to_csv(os.path.join(DATA, "bond_geometry.csv"), index=False)

    # ---- failure mode at every threshold ------------------------------------
    fm = []
    for pid, d in every.groupby("id"):
        row = dict(id=pid, answerable=len(d), gap_med_um=round(float(d.gap_sub_um.median()), 0),
                   face_share_med=round(float(np.maximum(d.crack_at_A, d.crack_at_B).median()), 2))
        for C in C_HI:
            coh = int((d.lo >= C).sum())
            row[f"coh_{C}"] = coh
            row[f"adh_{C}"] = len(d) - coh
        fm.append(row)
    fm = pd.DataFrame(fm)
    fm["id"] = pd.Categorical(fm.id, ORDER)
    fm = fm.sort_values("id")
    tot = {"id": "all", "answerable": len(every), "gap_med_um": round(float(every.gap_sub_um.median()), 0),
           "face_share_med": round(float(np.maximum(every.crack_at_A, every.crack_at_B).median()), 2)}
    for C in C_HI:
        tot[f"coh_{C}"] = int((every.lo >= C).sum())
        tot[f"adh_{C}"] = len(every) - tot[f"coh_{C}"]
    fm = pd.concat([fm, pd.DataFrame([tot])])
    fm.to_csv(os.path.join(DATA, "failure_mode_thresholds.csv"), index=False)

    # ---- specimen summary ----------------------------------------------------
    rows = []
    for tag, pid in ID.items():
        v = vols[vols.specimen == tag].sort_values("scan")
        h0 = v.specimen_height_mm.iloc[0]
        short = [round(1 - h / h0, 3) for h in v.specimen_height_mm]
        g1 = geo[(geo.id == pid) & (geo.which == "first")].iloc[0]
        gN = geo[(geo.id == pid) & (geo.which == "last")].iloc[0]
        su = survival(tag)
        fu = funnel.loc[tag]
        k = kin[kin.id == pid].sort_values("pair").iloc[-1]
        sw = sweeps[(sweeps.specimen == tag) & np.isclose(sweeps.eps_d, 1.0)]
        c = core[core.id == pid].sort_values("stage")
        last = v.iloc[-1]
        tv = tight[tight.id == pid].sort_values("scan")
        rows.append(dict(
            id=pid, n_scan=len(v), shortening=" / ".join(f"{s:.2f}" for s in short[1:]),
            z_first=g1.z_bar, z_last=gN.z_bar,
            followed=su["followed"], separated_pct=su["separated_pct"], detached_pct=su["detached_pct"],
            cracked_last=int(fu.cracked), wide_enough=int(fu.wide_enough), answerable=int(fu.both_labels_sound),
            rot_med_deg=k.rot_med_deg, rot_p95_deg=k.rot_p95_deg, eq_med=k.eq_med, dic_grains=f"{k.n_common}/{k.n_total}",
            # crack volumes from the bead-envelope classes (crack_split_tight.py),
            # as the change from the unloaded scan
            d_surface_mm3=round(float(tv.surface_mm3.iloc[-1] - tv.surface_mm3.iloc[0]), 0),
            d_body_mm3=round(float(tv.body_mm3.iloc[-1] - tv.body_mm3.iloc[0]), 0),
            body_components=int(last.n_components_ge_2pct),
            cluster_biggest=int(sw.biggest.iloc[0]) if len(sw) else np.nan,
            cluster_null=float(sw.biggest_null.iloc[0]) if len(sw) else np.nan,
            p_biggest=float(sw.p_biggest.iloc[0]) if len(sw) else np.nan,
            p_rms=float(sw.p_rms.iloc[0]) if len(sw) else np.nan,
            air_spans_last=bool(c.air_percolates.iloc[-1]),
            tau_first=c.tau_ice.iloc[0], tau_last=c.tau_ice.dropna().iloc[-1] if c.tau_ice.notna().sum() > 1 else np.nan))
    S = pd.DataFrame(rows)
    # Benjamini-Hochberg over the specimens that have a test
    p = S.p_biggest.values.astype(float)
    ok = np.isfinite(p)
    adj = np.full(len(p), np.nan)
    q = p[ok]; n = len(q); o = np.argsort(q); r = np.empty(n)
    r[o] = np.minimum.accumulate((q[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    adj[ok] = np.minimum(r, 1)
    S["p_biggest_bh"] = adj.round(3)
    S["id"] = pd.Categorical(S.id, ORDER)
    S = S.sort_values("id")
    S.to_csv(os.path.join(DATA, "specimen_summary.csv"), index=False)

    pd.set_option("display.width", 250)
    print(geo.to_string(index=False)); print()
    print(fm.to_string(index=False)); print()
    print(S.to_string(index=False))


if __name__ == "__main__":
    main()
