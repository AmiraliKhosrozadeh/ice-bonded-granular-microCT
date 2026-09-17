"""Geodesic path length through the ice network as a map, one specimen per
material.

For every ice voxel of the largest connected ice body the geodesic distance
through the ice from the support face is compared with the straight axial
distance.  Two colourings are available:

    excess   d_geo - |z - z_support|, in mm (default).  A crack that cuts the
             ice adds the detour round it to every path beyond, so the region
             behind a crack steps up by a fixed length and stays there; an
             intact column stays near zero.
    ratio    d_geo / |z - z_support|, the geodesic tortuosity, which carries
             the same step but divided by distance, so it fades away from the
             crack.

The map says where the ice network is severed; it is not the diffusion
tortuosity of the Methods.  Computed on the 2x-binned phase maps (49.5 um)
that the diffusion solve uses, restricted to the column (platens and punch
found by their phase composition, one bead diameter cleared at each end),
largest 26-connected ice body; ice cut off from it is drawn grey; the slices
next to the seed are cut.  The column is shown cut through its axis so the
interior is visible.  Camera as the crack renders, punch at the top.

    python scripts/fig_tau_geodesic.py                 # G1, A2, S2 -> figures/tau_geodesic.png
    TAU_MODE=ratio TAU_DS=2 TAU_CLIP=0 python ...      # the variants
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
FULL = os.environ.get("TAU_FULL", "0") == "1"    # full-resolution phase maps (sand only)
WEIGHT_MM = float(os.environ.get("TAU_WEIGHT_MM", "0"))   # if > 0, the path cost is 1 / local ice
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
MODE = os.environ.get("TAU_MODE", "excess")
CLIP = os.environ.get("TAU_CLIP", "1") == "1"   # draw the half column
SEED = os.environ.get("TAU_SEED", "support")     # or "punch"
ONLY = os.environ.get("TAU_SPECS")               # e.g. "G1,A2"
FADE = float(os.environ.get("TAU_FADE", "0"))    # if > 0, only ice above this value is drawn opaque,
                                                 # the rest of the column as a faint shell
RATIO_LIM = (1.1, 1.6)
REL_LIM = (1.0, 1.5)
EXCESS_LIM = (0.0, 1.5)
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
    f = os.path.join(CACHE, f"{pid}_{stage}_{MODE}_{SEED}_ds{DS}{'_full' if FULLP else ''}{'_w%g' % WEIGHT_MM if WEIGHT_MM else ''}.npz")
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
    ice = majority((ph == 2) & env, DS)
    halo = int(float(os.environ.get("TAU_HALO_MM", "4")) / VOX_MM)   # slices next to the seed, cut
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
    geo, _ = mcp.find_costs(starts)
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
        pf = os.path.join(CACHE, f"{pid}_baseline{'_full' if FULLP else ''}{'_w%g' % WEIGHT_MM if WEIGHT_MM else ''}.npz")
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


def render(val, big, ice, halo, out, shell=0.06):
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
    if FADE > 0:
        # the whole column as a faint shell, the high-value ice solid inside it
        p.add_mesh(mesh(mask)[0], color="#9aa4ad", opacity=shell, smooth_shading=True)
        high = mask & np.nan_to_num(val, nan=-np.inf) > FADE if False else (mask & (np.nan_to_num(val, nan=-1e9) > FADE))
        high = ndi.binary_opening(high, iterations=1)
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
        for st in stages:
            val, big, ice, halo = field(pid, st, r)
            if os.environ.get("TAU_HALO_MM"):
                halo = int(float(os.environ["TAU_HALO_MM"]) / vox_of(pid))
            png = os.path.join(CACHE, f"{pid}_{st}_{MODE}_{SEED}_ds{DS}_clip{int(CLIP)}_fade{FADE}{'_full' if full_for(pid) else ''}{'_w%g' % WEIGHT_MM if WEIGHT_MM else ''}.png")
            if not os.path.exists(png):
                render(val, big, ice, halo, png, shell=0.015 if pid.startswith("S") else 0.06)
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
    fig = plt.figure(figsize=(ps.TW, (0.285 if len(panels) <= 4 else 0.52) * ps.TW * len(panels) + 0.6))
    gs = fig.add_gridspec(len(panels) + 1, ncol, height_ratios=[1] * len(panels) + [0.10],
                          hspace=0.28, wspace=0.04, left=0.02, right=0.98, top=0.96, bottom=0.11)
    for i, (pid, mat, row) in enumerate(panels):
        for j in range(ncol):
            ax = fig.add_subplot(gs[i, j])
            ax.set_axis_off()
            if j < len(row):
                st, im = row[j]
                ax.imshow(im)
                ax.set_title(f"{pid}, {'unloaded' if st == 1 else f'load step {st}'}", fontsize=9, pad=3)
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
