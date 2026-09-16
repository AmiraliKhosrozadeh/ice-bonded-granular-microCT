"""LaTeX bodies of the supplementary tables from scripts/data.

    python scripts/supp_tables.py   -> supplementary/tab_*.tex
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(os.path.dirname(HERE), "supplementary")
ORDER = ["G1", "G2", "G3", "G4", "G5", "A1", "A2", "A3", "A4", "S1", "S2", "S3"]


def order(df, col="id"):
    df = df.copy()
    df[col] = pd.Categorical(df[col], ORDER)
    return df.sort_values([col] + [c for c in ("scan", "stage", "pair") if c in df.columns])


def write(name, header, rows, spec):
    body = ["\\begin{tabular}{" + spec + "}", "\\toprule", header + " \\\\", "\\midrule"]
    body += [r + " \\\\" for r in rows]
    body += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, f"tab_{name}.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(body) + "\n")
    print("->", name, len(rows), "rows")


# ---- crack volumes per scan ---------------------------------------------------
d = order(pd.read_csv(os.path.join(DATA, "crack_volumes_tight.csv")))
rows = [f"{r.id} & {r.scan} & {r.envelope_mm3:.0f} & {r.void_mm3:.1f} & {r.surface_mm3:.1f} & {r.body_mm3:.1f} & {r.separation_mm3:.1f}"
        for r in d.itertuples()]
write("crack", "Specimen & scan & envelope (mm$^3$) & void & surface crack & body crack & separation (mm$^3$)", rows, "lcccccc")

# ---- ice tortuosity per scan ----------------------------------------------------
t = order(pd.read_csv(os.path.join(DATA, "ice_tortuosity", "ice_tortuosity.csv")), "id")
core = pd.read_csv(os.path.join(DATA, "section36_core.csv"))
t = t.merge(core[["id", "stage", "shortening"]], on=["id", "stage"])
rows = [f"{r.id} & {r.stage} & {r.shortening:.3f} & {r.phi_ice_cube:.3f} & {r.tau_ice:.3f} & {r.tau_ice_sd:.3f} & {r.n_cubes_ok}/{r.n_cubes}"
        for r in t.itertuples()]
write("tau", "Specimen & scan & shortening & $\\phi_\\mathrm{ice}$ (cubes) & $\\tau_\\mathrm{ice}$ & sd & cubes", rows, "lcccccc")

# ---- kinematics per composed pair ------------------------------------------------
k = order(pd.read_csv(os.path.join(DATA, "kinematics_summary.csv")))
rows = [f"{r.id} & {r.pair.replace('to', '$\\to$')} & {r.n_common}/{r.n_total} & {r.u_rel_med_mm:.2f} & {r.global_rot_deg:.1f} & {r.rot_med_deg:.1f} & {r.rot_p95_deg:.1f} & {r.eq_med:.3f} & {r.eq_p95:.3f}"
        for r in k.itertuples()]
write("kin", "Specimen & scans & grains & $|\\mathbf{u}-\\mathbf{u}_\\mathrm{rigid}|$ (mm) & column rotation ($^\\circ$) & $\\theta$ median & $\\theta$ p95 & $E_\\mathrm{eq}$ median & p95", rows, "lcccccccc")

# ---- clustering sweep per specimen ----------------------------------------------
import glob
ID = {"Alumina_100_1800_T5": "A1", "Alumina_175_1800_T5": "A2", "Alumina_75_1000_T5": "A3", "Alumina_75_1800_T7": "A4",
      "Glass_75_1700_T5_HR": "G1", "Glass_75_1000_T6": "G2", "Glass_100_1700_T7": "G3", "Glass_100_1700_T5_HR": "G4",
      "Glass_100_1800_T5": "G5"}
sw = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(DATA, "planar_sweep", "*.csv"))])
sw["id"] = sw.specimen.map(ID)
sw = order(sw)
rows = []
for r in sw.itertuples():
    rms = "--" if pd.isna(r.rms_R) else f"{r.rms_R:.2f} / {r.rms_null:.2f} / {r.p_rms:.3f}"
    rows.append(f"{r.id} & {r.eps_d:.2f} & {r.clusters} / {r.clusters_null:.0f} / {r.p_clusters:.3f} & {r.biggest} / {r.biggest_null:.0f} / {r.p_biggest:.3f} & {rms}")
write("cluster", "Specimen & $R_\\mathrm{nb}/d$ & clusters / null / $p$ & largest / null / $p$ & RMS/$R$ / null / $p$", rows, "lcccc")

# ---- failure mode at every threshold ------------------------------------------
fm = pd.read_csv(os.path.join(DATA, "failure_mode_thresholds.csv"))
rows = [f"{r.id} & {r.answerable} & {r.gap_med_um:.0f} & {r.face_share_med:.2f} & {getattr(r, 'coh_0_5')}/{getattr(r, 'adh_0_5')} & {getattr(r, 'coh_0_7')}/{getattr(r, 'adh_0_7')} & {getattr(r, 'coh_0_8')}/{getattr(r, 'adh_0_8')} & {getattr(r, 'coh_0_9')}/{getattr(r, 'adh_0_9')}"
        for r in fm.rename(columns=lambda c: c.replace(".", "_")).itertuples()]
write("fm", "Specimen & answerable & gap median (\\si{\\micro\\meter}) & face share median & $C_\\mathrm{HI}=0.5$ & 0.7 & 0.8 & 0.9", rows, "lccccccc")
