"""Axis triad on the render figures that are not rebuilt by a script here
(the DIC renders of G1 come from render_move_all.py under WSL).  The
untouched render is kept in figures/_orig/ and the triad is pasted from it,
so the script can be run again.

    python scripts/add_triads.py
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
ORIG = os.path.join(FIG, "_orig")
# the three DIC panels sit side by side at 0.31 of the text width; one triad,
# on the first, serves the row
FILES = {"dic_G1_1to3_motion.png": 0.30, "dic_G1_1to3_radial.png": None, "dic_G1_1to3_strain.png": None}


def main():
    os.makedirs(ORIG, exist_ok=True)
    for name, frac in FILES.items():
        src = os.path.join(ORIG, name)
        dst = os.path.join(FIG, name)
        if not os.path.exists(src):
            shutil.copy2(dst, src)
        shutil.copy2(src, dst)
        if frac:
            ps.paste_triad(dst, frac=frac, margin=0.02, band=True)
        else:
            # the same white band, so the three panels keep their bottoms level
            from PIL import Image
            im = Image.open(dst).convert("RGB")
            W, H = im.size
            tall = Image.new("RGB", (W, H + int(round(0.30 * W))), (255, 255, 255))
            tall.paste(im, (0, 0))
            tall.save(dst)


if __name__ == "__main__":
    main()
