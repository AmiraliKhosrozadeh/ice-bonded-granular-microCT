"""Geodesic path length through the ice network as a map, Fig. 18 of the
paper (G1, G3, A2, S3) and the supplementary sheets.

For every ice voxel of the largest 26-connected ice body inside the grain
envelope, the cheapest path through the ice from the support face is found
(scikit-image MCP_Geometric) with a cost inversely proportional to the local
ice fraction in a 0.5 mm window, so a crack that cuts the ice forces a detour
and a zone of raised porosity lengthens the path in proportion.  The support
face is an equal-potential boundary as in the diffusion solve: every ice
voxel that is the first of its axial line from the support, within 3 mm of
it, seeds the path, and the axial distance is measured from that seed
surface.  The ratio of path to axial distance is divided by its per-height
median in the unloaded scan of the same specimen (relratio), and ice above
the unloaded scan's 99.5th percentile at the same height is drawn solid and
coloured, the rest as a faint shell, so an unloaded column is blank by
construction.  The 6 mm next to the support are cut, bodies below 0.5 mm3
dropped.  Beads on the 2x-binned phase maps, sand at full resolution with
the envelope pulled in by 1 mm (the ice-rich skin against the tube wall).

    python scripts/fig_tau_geodesic.py                       -> figures/tau_geodesic.png
    TAU_SPECS=G1,G2,G3,G4,G5 TAU_TAG=glass python scripts/fig_tau_geodesic.py
    TAU_SPECS=A1,A2,A3,A4 TAU_TAG=alumina ...;  TAU_SPECS=S1,S2,S3 TAU_TAG=sand ...

Every setting is an environment variable (TAU_MODE, TAU_WEIGHT_MM, TAU_SEED,
TAU_SEED_FACE, TAU_SEED_MM, TAU_CORE_MM, TAU_HALO_MM, TAU_FADE, TAU_FADE_PCT,
TAU_MIN_MM3, TAU_FULL, TAU_DS, TAU_CLIP); fields are cached in
scripts/data/tau_geodesic/.  Camera as the crack renders, punch at the top.
"""
import os
import sys

import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from skimage.graph import MCP_Geometric
import pyvista as pv
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps
import matplotlib.pyplot as plt

pv.OFF_SCREEN = True
BIN2 = ("C:/Users/cak7496/AppData/Local/Temp/claude/D--wsl/"
        "ce923714-58c7-48fb-8bce-70f1eccee47f/scratchpad/micro/bin2")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "tau_geodesic")
OUT = os.path.join(os.path.dirname(HERE), "figures", "tau_geodesic.png")
os.makedirs(CACHE, exist_ok=True)

ALL = {"G1": (1, 2, 3), "G2": (1, 2), "G3": (1, 2, 3, 4), "G4": (1, 2), "G5": (1, 2),
       "A1": (1, 2, 3), "A2": (1, 2), "A3": (1, 2), "A4": (1, 2, 3),
       "S1": (1, 2), "S2": (1, 2, 3), "S3": (1, 2)}
RVOX = {"G": 36, "A": 36, "S": 20}              # bead radius in full-res voxels (sand: 1 mm clearance)
RVOX_A3 = 19
MAIN = ["G1", "G3", "A2", "S3"]
SPECS = [(k, k[0], v, RVOX_A3 if k == "A3" else RVOX[k[0]]) for k, v in ALL.items()]
DS = int(os.environ.get("TAU_DS", "1"))         # extra downsampling on top of bin2
FULL = os.environ.get("TAU_FULL", "1") == "1"    # full-resolution phase maps (sand only)
WEIGHT_MM = float(os.environ.get("TAU_WEIGHT_MM", "0.5"))   # if > 0, the path cost is 1 / local ice
                                                 # fraction over a window of this size, so a porous
                                                 # zone lengthens the path even if the ice still bridges it
VOX_MM = 0.0495320458 * DS


def full_for(pid):
    """full resolution is used for the sand only, where the ice is fine"""
    return FULL and pid.startswith("S")


def vox_of(pid):
    return (0.0247660229 if full_for(pid) else 0.0495320458) * DS
E = "E:/RPTU-images/CT_images/Alumina/pyalumina/results/_xmat/sand"
SC = ("C:/Users/cak7496/AppData/Local/Temp/claude/D--wsl/"
      "ce923714-58c7-48fb-8bce-70f1eccee47f/scratchpad/micro/stage1/_xmat/sand")
FULLPATH = {("S1", 1): f"{E}/100_500_T5/scan01", ("S1", 2): f"{SC}/100_500_T5/scan02",
            ("S2", 1): f"{E}/25mm_100_500/scan01", ("S2", 2): f"{SC}/25mm_100_500/scan02",
            ("S2", 3): f"{SC}/25mm_100_500/scan03",
            ("S3", 1): f"{E}/75_200_T5/scan01", ("S3", 2): f"{SC}/75_200_T5/scan02"}
MODE = os.environ.get("TAU_MODE", "relratio")
CLIP = os.environ.get("TAU_CLIP", "0") == "1"   # draw the half column
SEED = os.environ.get("TAU_SEED", "support")     # or "punch"
# the support platen is an equal-potential face, as in the diffusion solve: with
# TAU_SEED_FACE=1 every ice voxel that is the first of its (y, x) column from
# the support, within TAU_SEED_MM of the support, seeds the path, so a support
# face that is mostly grain does not start the corners with a lateral run
SEED_FACE = os.environ.get("TAU_SEED_FACE", "1") == "1"
# the drawing threshold is the unloaded column's own range at the same height,
# this percentile of its field per 1 mm of height (empty: the fixed TAU_FADE)
FADE_PCT = os.environ.get("TAU_FADE_PCT", "99.5")
HALO_MM = float(os.environ.get("TAU_HALO_MM", "6"))   # next to the support, cut
SEED_MM = float(os.environ.get("TAU_SEED_MM", "3"))
# TAU_CORE_MM > 0 pulls the sand envelope in by that margin, so the ice-rich
# skin the sand packs against the tube wall (about 1 mm wide) is neither a
# cheap channel for the path nor drawn; the bead envelope is the bead closing
# and needs no margin
CORE_MM = float(os.environ.get("TAU_CORE_MM", "1"))


def core_of(pid):
    return CORE_MM if pid.startswith("S") else 0.0
ONLY = os.environ.get("TAU_SPECS")               # e.g. "G1,A2"
FADE = float(os.environ.get("TAU_FADE", "1.15"))    # if > 0, only ice above this value is drawn opaque,
                                                 # the rest of the column as a faint shell
RATIO_LIM = (1.1, 1.6)
REL_LIM = (1.0, 1.5)
EXCESS_LIM = (0.0, 1.5)
MIN_BODY_MM3 = float(os.environ.get("TAU_MIN_MM3", "0.5"))
CMAP = "inferno_r"                               # pale where nothing changed, dark where the path is longest


def majority(mask, k):
    if k == 1:
        return mask
    Z, Y, X = mask.shape
    m = mask[:Z // k * k, :Y // k * k, :X // k * k]
    return m.reshape(Z // k, k, Y // k, k, X // k, k).sum(axis=(1, 3, 5)) > (k ** 3 // 2)


def field(pid, stage, r_vox):
    FULLP = full_for(pid)
    VOX_MM = vox_of(pid)
    f = os.path.join(CACHE, f"{pid}_{stage}_{MODE}_{SEED}{'face' if SEED_FACE else ''}{'_core%g' % core_of(pid) if core_of(pid) else ''}_ds{DS}{'_full' if FULLP else ''}{'_w%g' % WEIGHT_MM if WEIGHT_MM else ''}.npz")
    if os.path.exists(f):
        z = np.load(f)
        return z["val"], z["big"], z["ice"], int(z["halo"])
    if FULLP:
        import tifffile
        ph = tifffile.imread(f"{FULLPATH[(pid, stage)]}/stage1/phase_labels.tif")
        r_vox = r_vox * 2                       # the clearance was given at bin2
    else:
        d = np.load(os.path.join(BIN2, f"{pid}_{stage}.npz"))
        ph = d["phase"]
    # the platens and the punch are classed as phase mixtures of a composition
    # no packing has; the column is the longest run of slices of packing composition
    ins = (ph > 0).sum(axis=(1, 2)).astype(float)
    gr = (ph == 3).sum(axis=(1, 2)) / np.maximum(ins, 1)
    ic = (ph == 2).sum(axis=(1, 2)) / np.maximum(ins, 1)
    ok = (gr > 0.30) & (gr < 0.63) & (ic < 0.65) & (ins > 0.3 * np.median(ins))
    ok = ndi.binary_closing(ok, structure=np.ones(15))
    lab, _ = ndi.label(ok)
    sz = np.bincount(lab.ravel()); sz[0] = 0
    zc = np.nonzero(lab == sz.argmax())[0]
    clear = int(r_vox)                          # one bead diameter at bin2
    ph = ph[zc[0] + clear:zc[-1] + 1 - clear]
    # only ice inside the grain packing: the per-slice closing of the grain
    # phase (1.25 grain radii, holes filled, two-voxel skin), as for the crack
    # classes, so ice in the annulus outside the outermost grains is not drawn
    r_close = max(int(round(1.25 * r_vox / 2)), 3)
    env = np.zeros(ph.shape, bool)
    for z in range(ph.shape[0]):
        g = ph[z] == 3
        if g.sum() < 50:
            continue
        dil = ndi.distance_transform_edt(~g) <= r_close
        clo = ndi.binary_fill_holes(ndi.distance_transform_edt(dil) > r_close)
        env[z] = ndi.distance_transform_edt(~clo) <= 2
    if core_of(pid) > 0:
        m = int(round(core_of(pid) / VOX_MM))
        for z in range(env.shape[0]):
            if env[z].any():
                env[z] = ndi.distance_transform_edt(env[z]) > m
    ice = majority((ph == 2) & env, DS)
    halo = int(HALO_MM / VOX_MM)                # slices next to the seed, cut
    cc, _ = ndi.label(ice, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(cc.ravel()); sizes[0] = 0
    big = cc == sizes.argmax()
    zz = np.nonzero(big.any(axis=(1, 2)))[0]
    z_src = int(zz[-1] if SEED == "support" else zz[0])        # support is high z, punch low z
    if WEIGHT_MM > 0:
        w = max(3, int(round(WEIGHT_MM / VOX_MM)) | 1)
        # ice fraction of the packing within the window, normalised by how much
        # of the window lies inside the envelope, so the lateral surface and
        # the ends do not read as porous
        envf = majority(env, DS) if DS > 1 else env
        frac = ndi.uniform_filter(big.astype(np.float32), size=w, mode="constant")
        inside = ndi.uniform_filter(envf.astype(np.float32), size=w, mode="constant")
        frac = frac / np.maximum(inside, 0.05)
        del inside, envf
        cost = np.where(big, 1.0 / np.clip(frac, 0.1, 1.0), np.inf)
        del frac
    else:
        cost = np.where(big, 1.0, np.inf)
    mcp = MCP_Geometric(cost, sampling=(1.0, 1.0, 1.0))
    del cost
    # seed from a 1 mm slab, not one slice, so that the seed face covers the
    # whole cross-section and no side of the column starts with a lateral run
    n_seed = max(1, int(1.0 / VOX_MM))
    zs = range(z_src - n_seed + 1, z_src + 1) if SEED == "support" else range(z_src, z_src + n_seed)
    starts = [(z, y, x) for z in zs for y, x in zip(*np.nonzero(big[z]))]
    if SEED_FACE:
        depth = int(SEED_MM / VOX_MM)
        sl = big[z_src - depth + 1:z_src + 1] if SEED == "support" else big[z_src:z_src + depth]
        anyc = sl.any(axis=0)
        first = sl.shape[0] - 1 - np.argmax(sl[::-1], axis=0) if SEED == "support" else np.argmax(sl, axis=0)
        yy, xx = np.nonzero(anyc)
        z0 = (z_src - depth + 1) if SEED == "support" else z_src
        starts = [(int(z0 + first[y, x]), int(y), int(x)) for y, x in zip(yy, xx)]
        # the axial distance is then measured from the seed surface under the
        # voxel, not from one plane, so a support face that ice reaches only in
        # places does not read as a lengthened path just above it
        zseed = np.full(anyc.shape, float(z_src))
        zseed[anyc] = z0 + first[anyc]
    geo, _ = mcp.find_costs(starts)
    if SEED_FACE:
        eucl = np.abs(np.arange(big.shape[0], dtype=np.float32)[:, None, None] - zseed[None].astype(np.float32))
    else:
        eucl = np.broadcast_to(np.abs(np.arange(big.shape[0], dtype=float) - z_src)[:, None, None], big.shape)
    val = np.full(big.shape, np.nan, np.float32)
    okv = big & np.isfinite(geo) & (eucl > 0)
    if MODE in ("ratio", "relratio"):
        val[okv] = (geo[okv] / eucl[okv]).astype(np.float32)
    else:
        val[okv] = ((geo[okv] - eucl[okv]) * VOX_MM).astype(np.float32)
    if MODE in ("excess", "relratio"):
        # an intact packing accumulates excess path steadily with distance from
        # the support as the ice winds round the grains; that profile, taken
        # from the unloaded scan of the same specimen, is subtracted so that
        # what remains is the excess a crack adds
        prof = np.array([np.nanmedian(val[z]) if np.isfinite(val[z]).any() else np.nan
                         for z in range(val.shape[0])])
        dist = np.abs(np.arange(val.shape[0]) - z_src)
        pf = os.path.join(CACHE, f"{pid}_baseline{'_face' if SEED_FACE else ''}{'_core%g' % core_of(pid) if core_of(pid) else ''}{'_full' if FULLP else ''}{'_w%g' % WEIGHT_MM if WEIGHT_MM else ''}.npz")
        if stage == 1 or not os.path.exists(pf):
            np.savez(pf, dist=dist, prof=prof)
        b = np.load(pf)
        okb = np.isfinite(b["prof"])
        order = np.argsort(b["dist"][okb])          # np.interp needs increasing abscissae
        base = np.interp(dist, b["dist"][okb][order], b["prof"][okb][order])
        if MODE == "excess":
            val = val - base[:, None, None].astype(np.float32)
        else:                                   # tortuosity relative to the unloaded column
            val = val / base[:, None, None].astype(np.float32)
    print(f"{pid} {stage}: ice {ice.sum()} vox, largest body {100*big.sum()/ice.sum():.0f} %, "
          f"{MODE} median {np.nanmedian(val):.3f} p95 {np.nanpercentile(val, 95):.3f}", flush=True)
    np.savez_compressed(f, val=val, big=big, ice=ice, halo=halo)
    return val, big, ice, halo


def mesh(mask):
    verts, faces, _, _ = measure.marching_cubes(mask.astype(np.uint8), level=0.5)
    pf = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    return pv.PolyData(verts, pf).smooth(n_iter=40), verts


def render(val, big, ice, halo, out, shell=0.06, fade=None, vox=VOX_MM):
    fade = FADE if fade is None else fade
    mask = big.copy()
    rest = ice & ~mask
    if SEED == "support":
        mask[-halo:] = False; rest[-halo:] = False
    else:
        mask[:halo] = False; rest[:halo] = False
    if CLIP:                                    # keep the half of the column facing the camera
        cy = mask.shape[1] // 2
        mask[:, :cy] = False
        rest[:, :cy] = False
    p = pv.Plotter(off_screen=True, window_size=(1400, 2200))
    p.set_background("white")
    if np.ndim(fade) or fade > 0:
        # the whole column as a faint shell, the high-value ice solid inside it
        p.add_mesh(mesh(mask)[0], color="#9aa4ad", opacity=shell, smooth_shading=True)
        thr = np.asarray(fade, np.float32)
        if thr.ndim:                            # one threshold per slice, by distance from the seed
            thr = thr[:, None, None]
        high = mask & (np.nan_to_num(val, nan=-1e9) > thr)
        high = ndi.binary_opening(high, iterations=1)
        # bodies below MIN_BODY_MM3 are specks of the threshold, not damage
        lab, n = ndi.label(high)
        if n:
            sz = np.bincount(lab.ravel()); sz[0] = 0
            keep = sz * vox ** 3 >= MIN_BODY_MM3
            high = keep[lab]
        mask = high                             # nothing above the threshold: shell only
    colour = None
    if mask.any():
        colour, verts = mesh(mask)
        idx = np.clip(np.round(verts).astype(int), 0, np.array(mask.shape) - 1)
        tv = val[idx[:, 0], idx[:, 1], idx[:, 2]]
        colour["v"] = np.where(np.isfinite(tv), tv, np.nanmin(val))
    if rest.any():
        p.add_mesh(mesh(rest)[0], color="#c8ccd0", smooth_shading=True)
    lim = {"ratio": RATIO_LIM, "relratio": REL_LIM, "excess": EXCESS_LIM}[MODE]
    if colour is not None:
        p.add_mesh(colour, scalars="v", cmap="turbo" if MODE != "excess" else CMAP, clim=lim,
                   smooth_shading=True, show_scalar_bar=False)
    c = np.array(mask.shape) / 2.0
    # camera on the -y side looking at the cut face, punch (low z) at the top
    p.camera_position = [(c[0], c[1] - 2.7 * max(mask.shape), c[2]), (c[0], c[1], c[2]), (-1, 0, 0)]
    p.camera.azimuth = 25 if not CLIP else 0
    p.screenshot(out)
    p.close()


def crop(path):
    im = np.asarray(Image.open(path).convert("RGB"))
    ink = (im < 250).any(axis=2)
    ys, xs = np.nonzero(ink)
    return im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def main():
    panels = []
    want = ONLY.split(",") if ONLY else MAIN
    for pid, mat, stages, r in SPECS:
        if want != ["all"] and pid not in want:
            continue
        row = []
        fade = FADE
        for st in stages:
            val, big, ice, halo = field(pid, st, r)
            halo = int(HALO_MM / vox_of(pid))        # the cached value may be older
            if st == stages[0] and FADE_PCT:
                # the threshold is what the unloaded column reaches at the same
                # distance from the support (a high percentile of its own field
                # per 1 mm of height), so that only ice above the unloaded
                # column's own range there is drawn; the range near the support,
                # where the seed face is seen at short range, is absorbed
                pct = float(FADE_PCT)
                Z = val.shape[0]
                dist0 = np.abs(np.arange(Z) - (Z - 1 if SEED == "support" else 0)) * vox_of(pid)
                nb = max(1, int(round(1.0 / vox_of(pid))))
                prof = np.full(Z, np.nan)
                for k in range(0, Z, nb):
                    v = val[k:k + nb]
                    if np.isfinite(v).any():
                        prof[k:k + nb] = np.nanpercentile(v, pct)
                okp = np.isfinite(prof)
                o = np.argsort(dist0[okp])          # np.interp needs increasing abscissae
                prof = np.interp(dist0, dist0[okp][o], prof[okp][o]) if okp.any() else np.full(Z, FADE)
                prof = ndi.uniform_filter1d(prof, 2 * nb + 1, mode="nearest")
                fade = (dist0, np.maximum(prof, FADE))
                print(f"{pid}: threshold {np.nanmin(fade[1]):.3f}..{np.nanmax(fade[1]):.3f}", flush=True)
            if isinstance(fade, tuple):
                Z = val.shape[0]
                dz = np.abs(np.arange(Z) - (Z - 1 if SEED == "support" else 0)) * vox_of(pid)
                o = np.argsort(fade[0])
                fade_z = np.interp(dz, fade[0][o], fade[1][o])
                ftag = f"fadez{FADE_PCT}"
            else:
                fade_z = fade
                ftag = f"fade{fade:.3f}"
            png = os.path.join(CACHE, f"{pid}_{st}_{MODE}_{SEED}{'face' if SEED_FACE else ''}{'_core%g' % core_of(pid) if core_of(pid) else ''}_ds{DS}_clip{int(CLIP)}_{ftag}{'_full' if full_for(pid) else ''}{'_w%g' % WEIGHT_MM if WEIGHT_MM else ''}.png")
            if not os.path.exists(png):
                render(val, big, ice, halo, png, shell=0.015 if pid.startswith("S") else 0.06, fade=fade_z, vox=vox_of(pid))
            row.append((st, crop(png)))
        panels.append((pid, mat, row))
    # pad every panel of a row to the row's height so the titles sit level;
    # the punch end stays at the top
    for _, _, r in panels:
        H = max(im.shape[0] for _, im in r)
        W = max(im.shape[1] for _, im in r)
        for k, (st, im) in enumerate(r):
            can = np.full((H, W, 3), 255, np.uint8)
            x0 = (W - im.shape[1]) // 2
            can[:im.shape[0], x0:x0 + im.shape[1]] = im
            r[k] = (st, can)

    ps.apply(9.0)
    ncol = max(len(r) for _, _, r in panels)
    if len(panels) > 4:                           # the supplementary sheet, all specimens
        ncol = 4
    ncol = max(ncol, 4) if any(len(r) == 4 for _, _, r in panels) else ncol
    h_in = (0.255 if len(panels) <= 4 else 0.25) * ps.TW * len(panels) + 0.95
    fig = plt.figure(figsize=(ps.TW, h_in))
    gs = fig.add_gridspec(len(panels) + 1, ncol, height_ratios=[1] * len(panels) + [0.10],
                          hspace=0.28, wspace=0.04, left=0.02, right=0.98, top=1 - 0.22 / h_in,
                          bottom=0.92 / h_in)
    for i, (pid, mat, row) in enumerate(panels):
        for j in range(ncol):
            ax = fig.add_subplot(gs[i, j])
            ax.set_axis_off()
            if j < len(row):
                st, im = row[j]
                ax.imshow(im)
                ax.set_title(f"{pid}, {'unloaded' if st == 1 else f'load step {st}'}", fontsize=9, pad=3)
            if j == 0:
                # a light rule between the unloaded column and the loaded ones
                bb = ax.get_position()
                xr = bb.x1 + 0.5 * (fig.add_subplot(gs[i, 1]).get_position().x0 - bb.x1)
                fig.axes[-1].remove()
                fig.add_artist(plt.Line2D([xr, xr], [bb.y0 - 0.005, bb.y1 + 0.03], transform=fig.transFigure,
                                          color="0.6", linewidth=0.8, alpha=0.8))
    cb_ax = fig.add_axes([0.28, 0.075, 0.44, 0.014])
    if MODE == "ratio":
        sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(*RATIO_LIM))
        label = "geodesic tortuosity of the ice path from the support, " + r"$\tau_\mathrm{g}$"
    elif MODE == "relratio":
        sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(*REL_LIM))
        label = "geodesic tortuosity of the ice path relative to the unloaded column, " + r"$\tau_\mathrm{g}/\tau_\mathrm{g,0}$"
    else:
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(*EXCESS_LIM))
        label = "extra length of the ice path from the support, relative to the unloaded column (mm)"
    cb = fig.colorbar(sm, cax=cb_ax, orientation="horizontal")
    cb.set_label(label)
    out = OUT if not os.environ.get("TAU_TAG") else OUT.replace(".png", f"_{os.environ['TAU_TAG']}.png")
    fig.savefig(out, dpi=300)
    print("->", out)


if __name__ == "__main__":
    main()
