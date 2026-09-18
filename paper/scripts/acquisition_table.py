"""Per-scan acquisition record from the scanner's PRM files (projections,
measurement time, geometry), for the supplement.

    python scripts/acquisition_table.py   -> supplementary/tab_acq.tex,
                                             scripts/data/acquisition.csv
"""
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CT = "E:/RPTU-images/CT_images"
# specimen -> (PRM stem, scan-number -> file suffix)
PRM = {
    "G3": ("Glass/Glass_100_1700_T7/Glass-100-1700-T7-tryagain", {1: "", 2: "-scan2", 3: "-scan3", 4: "-scan4"}),
    "G5": ("Glass/Glass-1001800-T5-normal", {1: "", 2: "_scan2_4"}),
    "A1": ("Alumina/Alumina/Alumina-100-1800-T5", {1: "", 2: "-scan2", 3: "-scan3"}),
    "A2": ("Alumina/Alumina/Alumina-175-1800-T5", {1: "", 2: "scan2"}),
    "A3": ("Alumina/Alumina/Alumina-75-1000-T5", {1: "", 2: "-scan2"}),
    "A4": ("Alumina/Alumina/Alumina-75-1800-T7", {1: "", 2: "-scan2-tryagain1", 3: "-scan3-tryagain1"}),
    "S1": ("Sand/Sand-100-500-T5", {1: "", 2: "-scan2"}),
    "S2": ("Sand/Sand25mm-100-500-T5", {1: "", 2: "-scan2", 3: "-scan3"}),
    "S3": ("Sand/Sand/Sand-75-200-T5", {1: "", 2: "-scan2"}),
}
SCANS = {"G1": 3, "G2": 2, "G3": 4, "G4": 2, "G5": 2, "A1": 3, "A2": 2, "A3": 2, "A4": 3, "S1": 2, "S2": 3, "S3": 2}
ORDER = ["G1", "G2", "G3", "G4", "G5", "A1", "A2", "A3", "A4", "S1", "S2", "S3"]


def read(path):
    txt = open(path, encoding="latin-1").read()
    g = lambda k: re.search(r"^%s=([^\r\n]*)" % k, txt, re.M).group(1).strip()
    return dict(proj=int(g("STEPS360")), t_s=int(g("MEASTIMESEC")), kv=int(g("VOLTAGE")),
                ua=int(g("CURRENT")), ms=int(g("INTTIME")), fod=float(g("CTFODIST")),
                fdd=float(g("CTFDDIST")), vox=float(g("ORIVOXELSIZE")) * 1000)


def main():
    rows = []
    for pid in ORDER:
        for s in range(1, SCANS[pid] + 1):
            rec = None
            if pid in PRM:
                stem, suf = PRM[pid]
                p = f"{CT}/{stem}{suf[s]}_100XXL_uc.PRM"
                if os.path.exists(p):
                    rec = read(p)
            rows.append((pid, s, rec))
    os.makedirs(os.path.join(ROOT, "scripts", "data"), exist_ok=True)
    with open(os.path.join(ROOT, "scripts", "data", "acquisition.csv"), "w") as f:
        f.write("id,scan,projections,time_s,kV,uA,int_ms,FOD_mm,FDD_mm,voxel_um\n")
        for pid, s, r in rows:
            if r:
                f.write(f"{pid},{s},{r['proj']},{r['t_s']},{r['kv']},{r['ua']},{r['ms']},{r['fod']:.1f},{r['fdd']:.1f},{r['vox']:.3f}\n")
            else:
                f.write(f"{pid},{s},,,,,,,,\n")
    L = [r"\begin{tabular}{llrr}", r"\toprule",
         r"Specimen & scan & projections & time (min) \\", r"\midrule"]
    for pid, s, r in rows:
        if r:
            L.append(f"{pid} & {s} & {r['proj']} & {r['t_s'] / 60:.0f} \\\\")
        else:
            L.append(f"{pid} & {s} & \\multicolumn{{2}}{{c}}{{record not archived}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(ROOT, "supplementary", "tab_acq.tex"), "w") as f:
        f.write("\n".join(L) + "\n")
    have = [r for _, _, r in rows if r]
    print(len(have), "of", len(rows), "records;",
          "kV", {r["kv"] for r in have}, "uA", {r["ua"] for r in have}, "ms", {r["ms"] for r in have},
          "vox", {round(r["vox"], 3) for r in have}, "FOD", {round(r["fod"], 1) for r in have})


if __name__ == "__main__":
    main()
