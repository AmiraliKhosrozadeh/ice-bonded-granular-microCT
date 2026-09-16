"""
3-way PSD comparison: adaptive watershed baseline vs ParticleSeg3D zero-shot
vs experimental Camsizer.

The headline question for Stage 4: does ParticleSeg3D reduce the coarse merged
tail (the watershed baseline's +265% d90)? This builds the comparison table and
an overlay plot of the three volume-weighted Q3 curves, plus the merged-tail
fraction (grain volume in grains coarser than the sieve upper limit).

Usage:
    python compare_methods.py \
        results/<sample>/stage3_baseline/grain_labels.tif \
        results/<sample>/stage4_particleseg3d/grain_labels_particleseg3d.tif
"""
import os
import sys
import numpy as np
import tifffile

from config import CFG
import experimental_psd as exp
import plot_style as ps


def vol_psd(labels, vox):
    sizes = np.bincount(labels.ravel())[1:]
    sizes = sizes[sizes > 0]
    d = (6.0 * sizes / np.pi) ** (1 / 3) * vox
    order = np.argsort(d)
    d_s, vol = d[order], sizes[order].astype(float)
    Q3 = 100 * np.cumsum(vol) / vol.sum()
    pct = {q: float(np.interp(q, Q3, d_s)) for q in (10, 50, 90)}
    merged = float(vol[d_s > CFG.max_grain_um].sum() / vol.sum())
    return d_s, Q3, pct, int(sizes.size), merged


def main():
    ws_path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(CFG.out_root, "stage3_baseline", "grain_labels.tif")
    ps3_path = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(CFG.out_root, "stage4_particleseg3d", "grain_labels_particleseg3d.tif")
    out = CFG.d("validation")
    vox = CFG.voxel_size_um

    mid_a, Q3_a, _, na = exp.load_camsizer("x_area")
    p_a = exp.percentiles(mid_a, Q3_a)

    rows = []
    curves = [("Camsizer x_area", mid_a, Q3_a)]
    for name, path in [("watershed", ws_path), ("ParticleSeg3D", ps3_path)]:
        if not os.path.exists(path):
            print(f"  (skip {name}: {path} not found)")
            continue
        lab = tifffile.imread(path)
        d_s, Q3, pct, n, merged = vol_psd(lab, vox)
        rows.append((name, n, pct, merged))
        curves.append((name, d_s, Q3))

    # table
    L = []; P = L.append
    P(f"{'method':<16}{'grains':>9}{'d10':>7}{'d50':>7}{'d90':>7}{'merged%':>9}{'d50_err':>9}")
    P(f"{'Camsizer x_area':<16}{na:>9}{p_a[10]:>7.0f}{p_a[50]:>7.0f}{p_a[90]:>7.0f}{'-':>9}{'-':>9}")
    for name, n, pct, merged in rows:
        err = 100 * (pct[50] - p_a[50]) / p_a[50]
        P(f"{name:<16}{n:>9}{pct[10]:>7.0f}{pct[50]:>7.0f}{pct[90]:>7.0f}"
          f"{merged*100:>8.0f}%{err:>+8.0f}%")
    rep = "\n".join(L)
    with open(os.path.join(out, "method_comparison.txt"), "w") as f:
        f.write(rep + "\n")
    print("\n" + rep + "\n")

    # overlay (one plot per file)
    ps.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 8))
    for name, x, Q in curves:
        ax.plot(x, Q, label=name)
    ax.axvspan(CFG.min_grain_um, CFG.max_grain_um, color="green", alpha=0.10)
    ax.set_xlim(0, 800)
    ax.set_xlabel("Equivalent diameter (um)")
    ax.set_ylabel("Cumulative passing Q3 (%)")
    ps.style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
              frameon=True, edgecolor="gray")
    ps.save_fig(fig, os.path.join(out, "psd_3way_comparison.png"))
    print("compare_methods done.")


if __name__ == "__main__":
    main()
