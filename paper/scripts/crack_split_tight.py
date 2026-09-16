"""Void / crack / separation classes for every bead specimen, recomputed inside
one bead-bounded envelope so that every scan of a specimen is measured by the
same rule.

WHY.  The pipeline's per-scan sample mask is a vote of four boundary methods.
On the unloaded scan it hugs the ice skin; on the loaded scans it steps out
and takes in the scalloped air between the outer beads, and that air, lying
in concavities narrower than the crack width, is classified as narrow-gap
crack.  On A4 scan 2 this produced 300 mm3 of "crack" in a scan where DIC
measures 0.04 mm of motion and the whole-volume ice is unchanged (checked
2026-09-16).  The zero of the unloaded scan and the 300 mm3 of the loaded one
are therefore two different masks, not damage.

WHAT.  The specimen is taken as the beads themselves.  Per slice the gated
bead labels are closed with a disc of 1.25 bead radii, holes are filled and
the result is dilated by two thirds of a radius to keep the ice skin on the
outer beads; that envelope is the same construction on every scan.  Inside it
the pipeline's own classes are rebuilt from the per-scan phase map (2 = ice,
3 = grain):

    gaps        envelope AND NOT (ice or grain)
    void        connected gap bodies below 2.0 mm3 (pipeline ceiling) and of
                at least 100 voxels, smaller bodies being phase-map specks
    crack       remaining gap voxels within 5 voxels of solid, i.e. an opening
                up to 0.25 mm wide (pipeline width rule, here by erosion)
    separation  remaining gap voxels deeper than that
    surface / body crack   outside / inside a 20-voxel skin of the envelope

Because the envelope admits the same boundary concavities on every scan, the
unloaded scan carries a baseline; the change from it is the damage.  Labels
are gated by volume alone (a quarter to 1.6 bead volumes), which removes
the wall label, and the envelope is kept to the slices the pipeline's sample
mask occupies less one bead diameter at each end, which removes the platens
and the platen-contact air.

Writes scripts/data/crack_volumes_tight.csv, one row per scan.

    python scripts/crack_split_tight.py            # all nine
    python scripts/crack_split_tight.py A4 G3      # a subset, merged into the csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

CT = "E:/RPTU-images/CT_images"
BF = f"{CT}/paper_figures/bond_failure"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "crack_volumes_tight.csv")
VOX_MM3 = (24.7660229 / 1000.0) ** 3
SKIN = 20                       # voxels, pipeline surface/body split
VOID_CEIL = 2.0                 # mm3, pipeline void ceiling
HALF_WIDTH = 5                  # voxels, 2 x 5 x 24.77 um = 0.25 mm crack width
MIN_BODY = 100                  # voxels (0.0015 mm3, ~5 voxels across); smaller bodies are specks

REG = {
    "A1": ("Alumina_100_1800_T5", f"{CT}/Alumina/Alumina_100_1800_T5/Alumina_100_1800_T5_spam", (1, 2, 3)),
    "A2": ("Alumina_175_1800_T5", f"{CT}/Alumina/Alumina_175_1800_T5/Alumina_175_1800_T5_spam", (1, 2)),
    "A3": ("Alumina_75_1000_T5", f"{CT}/Alumina/Alumina_75_1000_T5/Alumina_75_1000_T5_spam", (1, 2)),
    "A4": ("Alumina_75_1800_T7", f"{CT}/Alumina/Alumina-75-1800-T7/Alumina_75_1800_T7_spam", (1, 2, 3)),
    "G1": ("Glass_75_1700_T5_HR", f"{CT}/Glass/Glass_75_1700_T5_HR/Glass_75_spam", (1, 2, 3)),
    "G2": ("Glass_75_1000_T6", f"{CT}/Glass/Glass_75_1000_T6/Glass_T6_spam", (1, 2)),
    "G3": ("Glass_100_1700_T7", f"{CT}/Glass/Glass_100_1700_T7/Glass_1700_spam", (1, 2, 3, 4)),
    "G4": ("Glass_100_1700_T5_HR", f"{CT}/Glass/Glass_100_1700_T5_HR/Glass_T5_HR_spam", (1, 2)),
    "G5": ("Glass_100_1800_T5", f"{CT}/Glass/Glass_spam", (1, 2)),
}


def n_pages(p):
    with tifffile.TiffFile(p) as t:
        return len(t.pages), t.pages[0].shape


def framed(spam, stem, s, shape):
    """The stack <stem>_scanNN[_aligned].tif in the frame of the crack stacks."""
    for suf in ("_aligned", ""):
        p = f"{spam}/data/{stem}_scan{s:02d}{suf}.tif"
        if os.path.exists(p) and n_pages(p) == shape:
            return p
    raise FileNotFoundError(f"no {stem} stack of shape {shape} for scan {s} in {spam}")


def gated_labels(labels, q, s):
    """Every label of bead size.  The bond pipeline's shape gate is too strict
    for an envelope, since a bead it rejects for internal porosity is still a
    bead at the boundary; only the wall and platen labels (far too large) and
    crumbs below a quarter of a bead are dropped."""
    g1 = q[(q.scan == q.scan.min()) & q.radius_ok & q.shape_ok]
    v_med = float(np.median(g1.voxels))
    cnt = np.bincount(labels.ravel())
    keep = (cnt > 0.25 * v_med) & (cnt < 1.6 * v_med)
    keep[0] = False
    R = float(np.median(g1.r_eq_vox))
    return keep[labels], R, int(keep.sum())


def envelope(beads, r_close, r_skin):
    out = np.zeros(beads.shape, dtype=bool)
    for z in range(beads.shape[0]):
        b = beads[z]
        if b.sum() < 50:
            continue
        dil = ndi.distance_transform_edt(~b) <= r_close
        clo = ndi.distance_transform_edt(dil) > r_close
        clo = ndi.binary_fill_holes(clo)
        out[z] = ndi.distance_transform_edt(~clo) <= r_skin
    return out


def run(pid):
    tag, spam, scans = REG[pid]
    q = pd.read_csv(f"{BF}/{tag}/data/{tag}_bead_quality.csv")
    rows = []
    for s in scans:
        t0 = time.time()
        shape = n_pages(f"{spam}/results_voidcrack/crack_scan{s:02d}.tif")
        labels = tifffile.imread(framed(spam, "bead_labels", s, shape))
        beads, R, n_gated = gated_labels(labels, q, s)
        del labels
        r_close, r_skin = int(round(1.25 * R)), 4
        env = envelope(beads, r_close, r_skin)
        del beads
        # the label stacks run into the platens, whose bead-sized pieces pass
        # a volume gate; keep the envelope to the slices the pipeline's own
        # sample mask occupies, which is where it dropped platen and punch
        with tifffile.TiffFile(f"{spam}/results_voidcrack/sample_scan{s:02d}.tif") as t:
            occ = [z for z, pg in enumerate(t.pages) if pg.asarray().any()]
        # and one bead diameter clear of each platen, where the air between
        # the platen face and the first layer of beads is not a crack
        env[:occ[0] + int(2 * R)] = False
        env[occ[-1] + 1 - int(2 * R):] = False
        ph = tifffile.imread(framed(spam, "phases", s, shape))
        solid = ph >= 2
        n_ice = int((ph[env] == 2).sum())
        n_grain = int((ph[env] == 3).sum())
        n_grain_all = int((ph == 3).sum())
        del ph
        gaps = env & ~solid
        del solid
        # void: small isolated bodies
        lab, n = ndi.label(gaps)
        sizes = np.bincount(lab.ravel())
        small = (sizes * VOX_MM3 < VOID_CEIL) & (sizes >= MIN_BODY)
        small[0] = False
        void = small[lab]
        # bodies below MIN_BODY are partial-volume specks of the phase map,
        # neither void nor crack
        gaps &= (sizes >= MIN_BODY)[lab]
        del lab
        rest = gaps & ~void
        # width rule by erosion: a voxel within HALF_WIDTH of solid is crack
        deep = ndi.binary_erosion(rest, iterations=HALF_WIDTH)
        crack = rest & ~deep
        sep = deep
        core = ndi.binary_erosion(env, iterations=SKIN)
        body = crack & core
        surface = crack & ~core
        row = dict(id=pid, specimen=tag, scan=s, R_vox=round(R, 1), r_close=r_close, r_skin=r_skin,
                   n_gated=n_gated,
                   envelope_mm3=round(env.sum() * VOX_MM3, 1),
                   ice_mm3=round(n_ice * VOX_MM3, 1), grain_mm3=round(n_grain * VOX_MM3, 1),
                   grain_in_env=round(n_grain / max(n_grain_all, 1), 3),
                   gaps_mm3=round(gaps.sum() * VOX_MM3, 2),
                   void_mm3=round(void.sum() * VOX_MM3, 2),
                   surface_mm3=round(surface.sum() * VOX_MM3, 2),
                   body_mm3=round(body.sum() * VOX_MM3, 2),
                   separation_mm3=round(sep.sum() * VOX_MM3, 2),
                   n_gap_bodies=int(n), seconds=round(time.time() - t0))
        print(row, flush=True)
        rows.append(row)
        if os.environ.get("SAVE_CLASSES"):
            # class stacks at 2x decimation for the 3D renders (render_crack3d.py)
            d = f"{spam}/results_voidcrack/tight2"
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(f"{d}/classes_scan{s:02d}.npz", sample=env[::2, ::2, ::2],
                                void=void[::2, ::2, ::2], surface=surface[::2, ::2, ::2],
                                body=body[::2, ::2, ::2], sep=sep[::2, ::2, ::2])
        del env, gaps, void, rest, deep, crack, sep, core, body, surface
    return rows


if __name__ == "__main__":
    ids = sys.argv[1:] or list(REG)
    rows = []
    for pid in ids:
        rows += run(pid)
    df = pd.DataFrame(rows)
    if os.path.exists(OUT) and len(ids) < len(REG):
        old = pd.read_csv(OUT)
        df = pd.concat([old[~old.id.isin(ids)], df]).sort_values(["id", "scan"])
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
