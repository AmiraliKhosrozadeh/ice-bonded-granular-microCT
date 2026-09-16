"""
Stage 3b - Watershed parameter sweep (on the saved subvolume sand mask).

Sweeps the two parameters that control over/under-segmentation -- the EDT
h-maxima height and the seed minimum-distance -- and scores each setting
against the Camsizer PSD. EDT is computed ONCE (depends only on the mask),
so the sweep is cheap.

Reports per setting: grain count, volume-weighted Q3 d10/d50/d90, merged-tail
fraction (volume in grains coarser than the sieve upper limit), and the d50
error vs Camsizer x_area.

Outputs (results/<sample>/stage3b_sweep/):
  sweep_table.csv          full grid
  sweep_grain_count.png    grain count vs h-maxima (one line per seed spacing)
  sweep_d50_error.png      |d50 - Camsizer| vs h-maxima
  sweep_merged_tail.png    merged-tail volume fraction vs h-maxima

Run:  python stage3b_param_sweep.py
"""
import os
import glob
import numpy as np
import tifffile
from scipy import ndimage
from skimage.morphology import h_maxima
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from config import CFG
import experimental_psd as exp
import plot_style as ps

H_VALUES = [1.0, 1.5, 2.0, 3.0, 4.0]
MINDIST_VALUES = [2, 3, 5]
SIZE_FILTER = CFG.watershed_min_grain_voxels


def vol_psd(labels, vox):
    sizes = np.bincount(labels.ravel())[1:]
    sizes = sizes[sizes > 0]
    if sizes.size == 0:
        return dict(n=0, d10=0, d50=0, d90=0, merged=0)
    d = (6.0 * sizes / np.pi) ** (1 / 3) * vox
    order = np.argsort(d)
    d_s = d[order]
    vol = sizes[order].astype(float)
    Q3 = 100 * np.cumsum(vol) / vol.sum()
    p = {q: float(np.interp(q, Q3, d_s)) for q in (10, 50, 90)}
    merged = float(vol[d_s > CFG.max_grain_um].sum() / vol.sum())  # vol frac > sieve upper
    return dict(n=int(sizes.size), d10=p[10], d50=p[50], d90=p[90], merged=merged)


def main():
    out = CFG.d("stage3b_sweep")
    sand_dir = os.path.join(CFG.out_root, "stage2_sand")
    fs = sorted(glob.glob(os.path.join(sand_dir, "sand_clean_*.tif")))
    if not fs:
        raise FileNotFoundError("run Stage 2 first (no sand_clean_*.tif found)")
    print(f"Stage 3b - param sweep on {len(fs)} saved subvolume slices")
    mask = np.stack([tifffile.imread(f) for f in fs]) > 0
    vox = CFG.voxel_size_um

    print("  EDT (once) ...")
    edt = ndimage.distance_transform_edt(mask).astype(np.float32)

    mid_a, Q3_a, _, _ = exp.load_camsizer("x_area")
    d50_exp = exp.percentiles(mid_a, Q3_a)[50]

    rows = []
    for h in H_VALUES:
        hm = h_maxima(edt, h)
        for md in MINDIST_VALUES:
            coords = peak_local_max(edt, min_distance=md,
                                    labels=(hm > 0).astype(np.uint8) * mask)
            seeds = np.zeros(mask.shape, np.int32)
            for i, (z, y, x) in enumerate(coords, 1):
                seeds[z, y, x] = i
            lab = watershed(-edt, markers=seeds, mask=mask)
            # size filter
            sizes = np.bincount(lab.ravel())
            keep = np.where(sizes >= SIZE_FILTER)[0]
            keep = keep[keep != 0]
            remap = np.zeros(lab.max() + 1, np.int32)
            remap[keep] = np.arange(1, len(keep) + 1)
            lab = remap[lab]
            m = vol_psd(lab, vox)
            m["h"] = h; m["min_dist"] = md
            m["d50_err_pct"] = 100 * (m["d50"] - d50_exp) / d50_exp
            rows.append(m)
            print(f"  h={h} md={md}: n={m['n']} d50={m['d50']:.0f} "
                  f"(err {m['d50_err_pct']:+.0f}%) merged={m['merged']*100:.0f}%")

    # write csv
    cols = ["h", "min_dist", "n", "d10", "d50", "d90", "merged", "d50_err_pct"]
    with open(os.path.join(out, "sweep_table.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])
                             for c in cols) + "\n")

    # plots (one per file): metric vs h, one line per min_dist
    ps.apply_style()
    import matplotlib.pyplot as plt

    def plot_metric(key, ylabel, fname, expline=None):
        fig, ax = plt.subplots(figsize=(12, 8))
        for md in MINDIST_VALUES:
            xs = [r["h"] for r in rows if r["min_dist"] == md]
            ys = [r[key] for r in rows if r["min_dist"] == md]
            ax.plot(xs, ys, marker="o", label=f"min_dist={md}")
        if expline is not None:
            ax.axhline(expline, color="black", linestyle="--", linewidth=2)
        ax.set_xlabel("EDT h-maxima height (voxels)")
        ax.set_ylabel(ylabel)
        ps.style_axes(ax)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
                  frameon=True, edgecolor="gray")
        ps.save_fig(fig, os.path.join(out, fname))

    plot_metric("n", "Grain count", "sweep_grain_count.png")
    plot_metric("d50", "Volume d50 (um)", "sweep_d50.png", expline=d50_exp)
    plot_metric("merged", "Merged-tail volume fraction", "sweep_merged_tail.png")
    print(f"\nStage 3b done. Camsizer x_area d50 = {d50_exp:.0f} um (target).")
    print("Lower h + smaller min_dist -> more seeds -> less merging but more splits.")


if __name__ == "__main__":
    main()
