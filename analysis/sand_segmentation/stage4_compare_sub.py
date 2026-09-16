"""
Stage 4 - subvolume comparison: ParticleSeg3D zero-shot vs watershed vs Camsizer.

Compares both methods on the SAME 96x160x160 dense-contact subvolume and the
experimental Camsizer PSD. Headline question: does ParticleSeg3D cut the merged
coarse tail (oversized merged grains) that inflates the watershed PSD?
"""
import os
import numpy as np
import tifffile
from config import CFG
import experimental_psd as exp
import plot_style as ps

# subvolume in cropped-volume coords (must match dl/prepare_ps3d_input.py SUBVOL)
SUBVOL = (620, 715, 420, 579, 160, 319)
PS3D = os.path.join(CFG.out_root, "stage4_particleseg3d", "grain_labels_particleseg3d_sub.tif")
WS_FULL = os.path.join(CFG.out_root, "stage3_baseline", "grain_labels.tif")


def vol_psd(labels, vox):
    sizes = np.bincount(labels.ravel())[1:]
    sizes = sizes[sizes > 0]
    if sizes.size == 0:
        return None
    d = (6.0 * sizes / np.pi) ** (1 / 3) * vox
    o = np.argsort(d); d_s = d[o]; v = sizes[o].astype(float)
    Q3 = 100 * np.cumsum(v) / v.sum()
    pct = {q: float(np.interp(q, Q3, d_s)) for q in (10, 50, 90)}
    merged = float(v[d_s > CFG.max_grain_um].sum() / v.sum())
    return dict(n=int(sizes.size), d10=pct[10], d50=pct[50], d90=pct[90],
                merged=merged, dmax=float(d_s.max()), d_s=d_s, Q3=Q3)


def main():
    out = CFG.d("validation")
    vox = CFG.voxel_size_um
    z0, z1, y0, y1, x0, x1 = SUBVOL

    ps3d = vol_psd(tifffile.imread(PS3D), vox)
    ws_full = tifffile.imread(WS_FULL)
    ws_sub = ws_full[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1].copy()
    ws = vol_psd(ws_sub, vox)

    mid_a, Q3_a, _, na = exp.load_camsizer("x_area")
    p_a = exp.percentiles(mid_a, Q3_a)

    L = []; P = L.append
    P("Same 96x160x160 dense-contact subvolume; volume-weighted Q3.")
    P(f"{'method':<16}{'grains':>8}{'d10':>6}{'d50':>6}{'d90':>6}{'dmax':>7}{'merged%':>9}")
    P(f"{'Camsizer x_area':<16}{na:>8}{p_a[10]:>6.0f}{p_a[50]:>6.0f}{p_a[90]:>6.0f}{'-':>7}{'-':>9}")
    for nm, r in [("watershed", ws), ("ParticleSeg3D", ps3d)]:
        P(f"{nm:<16}{r['n']:>8}{r['d10']:>6.0f}{r['d50']:>6.0f}{r['d90']:>6.0f}"
          f"{r['dmax']:>7.0f}{r['merged']*100:>8.0f}%")
    P("")
    dd90 = ws['d90'] - ps3d['d90']
    P(f"d90: watershed {ws['d90']:.0f} -> ParticleSeg3D {ps3d['d90']:.0f} um "
      f"({'REDUCED' if dd90>0 else 'increased'} by {abs(dd90):.0f})")
    P(f"merged tail: watershed {ws['merged']*100:.0f}% -> ParticleSeg3D {ps3d['merged']*100:.0f}%")
    P(f"max grain:   watershed {ws['dmax']:.0f} -> ParticleSeg3D {ps3d['dmax']:.0f} um "
      f"(sieve upper {CFG.max_grain_um:.0f})")
    rep = "\n".join(L)
    with open(os.path.join(out, "stage4_subvolume_comparison.txt"), "w") as f:
        f.write(rep + "\n")
    print("\n" + rep + "\n")

    ps.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(ws['d_s'], ws['Q3'], label="watershed")
    ax.plot(ps3d['d_s'], ps3d['Q3'], label="ParticleSeg3D")
    ax.plot(mid_a, Q3_a, label="Camsizer x_area")
    ax.axvspan(CFG.min_grain_um, CFG.max_grain_um, color="green", alpha=0.10)
    ax.set_xlim(0, 800)
    ax.set_xlabel("Equivalent diameter (um)")
    ax.set_ylabel("Cumulative passing Q3 (%)")
    ps.style_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
              frameon=True, edgecolor="gray")
    ps.save_fig(fig, os.path.join(out, "stage4_subvolume_psd.png"))


if __name__ == "__main__":
    main()
