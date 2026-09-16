"""
Experimental PSD reference (Retsch Camsizer dynamic image analysis).

Loads the ground-truth sieve/Camsizer PSD measured on the same F36 sand
("jesssand") so the CT-segmented grain PSD can be validated against a real
measurement -- the decisive QC for the whole pipeline.

Source data: D:\\Rolling\\Sand (Camsizer .xle files, triplicate runs).
Diameter definitions available: x_area (area-equivalent, the closest analog to
a CT equivalent-sphere diameter), xFemin (Feret-min ~ sieve), xMamin, xFemax.
"""
import glob
import os
import numpy as np
import pandas as pd

EXP_DIR = os.environ.get("SAND_CAMSIZER_DIR", r"D:\Rolling\Sand")
# x_area = area-equivalent diameter: the most defensible comparison to a CT
# equivalent-sphere(-volume) diameter. xFemin is the closest to a sieve size.
DEFAULT_DEFINITION = "x_area"


def _load_xle(fp):
    with open(fp, encoding="utf-16") as f:
        lines = f.readlines()
    s = next(i for i, l in enumerate(lines) if l.startswith("Size class"))
    return pd.read_csv(fp, delimiter="\t", skiprows=s + 1,
                       names=["mn", "mx", "p3", "Q3", "iQ3", "q3", "p0", "Q0", "q0"],
                       engine="python", encoding="utf-16")


def load_camsizer(definition=DEFAULT_DEFINITION, exp_dir=EXP_DIR):
    """Return (size_mid_um, Q3_pct, p3_pct) averaged over replicate runs.

    Q3 = cumulative volume %; p3 = volume-frequency %. Replicates are
    interpolated onto the first run's size grid before averaging.
    """
    fs = sorted(glob.glob(os.path.join(exp_dir, "Camsizer", f"*{definition}_00*.xle")))
    fs += sorted(glob.glob(os.path.join(exp_dir, f"*{definition}_00*.xle")))
    if not fs:
        raise FileNotFoundError(f"No Camsizer {definition} .xle files under {exp_dir}")
    base = _load_xle(fs[0])
    mid = ((base["mn"] + base["mx"]) / 2).to_numpy()
    Q3s, p3s = [], []
    for fp in fs:
        d = _load_xle(fp)
        m = ((d["mn"] + d["mx"]) / 2).to_numpy()
        Q3s.append(np.interp(mid, m, d["Q3"].to_numpy()))
        p3s.append(np.interp(mid, m, d["p3"].to_numpy()))
    return mid, np.mean(Q3s, axis=0), np.mean(p3s, axis=0), len(fs)


def percentiles(mid, Q3, ps=(10, 50, 90)):
    return {p: float(np.interp(p, Q3, mid)) for p in ps}


if __name__ == "__main__":
    for defn in ["x_area", "xFemin", "xMamin", "xFemax"]:
        try:
            mid, Q3, p3, n = load_camsizer(defn)
            d = percentiles(mid, Q3)
            print(f"{defn:8s} n={n}  d10={d[10]:.0f}  d50={d[50]:.0f}  d90={d[90]:.0f} um")
        except FileNotFoundError as e:
            print(e)
