"""Sand at the two displacement rates, one figure per specimen, same fields.

    sand_S3_fields.png   S3, higher rate, its one load step
    sand_S2_fields.png   S2, lower rate, its last load step (scan 2 -> 3)

    column 1   local void concentration per grain at the loaded scan
    column 2   rearrangement field, the motion of each grain relative to the
               median of its own layer, so the bulk shortening is removed
    column 3   deviatoric strain field per grain

S3 shows two linear traces in the void field that reappear as bands in the
rearrangement field; S2 shows neither, its rearrangement field being the
outward flow of the head over a quiet lower column.  The two specimens are
kept in separate figures so a reader never takes a panel of one for the
other.

Every render is scaled to a common specimen height and its colour bar to a
common width, cut from the source figures.  The source bar labels are erased
and rewritten in the paper's font so the wording is the paper's.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SAND = 'E:/RPTU-images/CT_images/paper_figures/Sand'
FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'figures')

H_REND = 1750           # common specimen height
W_BAR = 1000            # common colour-bar width
GAP = 60                # render -> bar
COL = 110               # between panels
LAB = 160               # strip under the row, holds the panel letters
FONT = 'C:/Windows/Fonts/times.ttf'
FS_LABEL = 78           # colour-bar label
FS_LETTER = 96

LABELS = {'void': 'local void concentration',
          'rearr': 'rearrangement field  (\u00b5m)',
          'strain': 'deviatoric strain field'}


def read(p):
    im = Image.open(p)
    if im.mode in ('RGBA', 'LA', 'P'):
        im = im.convert('RGBA')
        bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert('RGB')).astype(np.float32) / 255.0


def runs(idx, gap):
    out, s0, p0 = [], idx[0], idx[0]
    for r in idx[1:]:
        if r - p0 > gap:
            out.append((s0, p0))
            s0 = r
        p0 = r
    out.append((s0, p0))
    return out


def crop(a, x0=0.0, x1=1.0, y0=0.0, y1=1.0, pad=8, tol=0.985):
    """Tight crop of non-white content inside a fractional window."""
    H, W = a.shape[:2]
    sub = a[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
    m = sub.min(axis=2) < tol
    r = np.where(m.any(axis=1))[0]
    c = np.where(m.any(axis=0))[0]
    return sub[max(r[0] - pad, 0):r[-1] + pad + 1,
               max(c[0] - pad, 0):c[-1] + pad + 1]


def main_body(a, gap=25, tol=0.985):
    """The tallest run of ink rows, so a detached lip or title is dropped."""
    m = a.min(axis=2) < tol
    r0, r1 = max(runs(np.where(m.any(axis=1))[0], gap), key=lambda t: t[1] - t[0])
    sub = a[r0:r1 + 1]
    c = np.where((sub.min(axis=2) < tol).any(axis=0))[0]
    return sub[:, c[0]:c[-1] + 1]


def bar_only(a, ticks=1, tol=0.985):
    """The colour bar with its tick labels, without the source's own label.

    The bar is the first ink run with saturated colour.  `ticks` says how
    many black runs after it belong to the tick labels (0 where the source
    drew the digits touching the bar, so they are inside the bar run
    already); anything after that is the source label and is dropped.
    """
    m = a.min(axis=2) < tol
    sat = a.max(axis=2) - a.min(axis=2)
    rr = runs(np.where(m.any(axis=1))[0], 5)
    k = next(i for i, (r0, r1) in enumerate(rr)
             if sat[r0:r1 + 1][m[r0:r1 + 1]].mean() > 0.3)
    end = rr[min(k + ticks, len(rr) - 1)][1]
    sub = a[rr[k][0]:end + 1]
    c = np.where((sub.min(axis=2) < tol).any(axis=0))[0]
    return sub[:, c[0]:c[-1] + 1]


def scale(a, h=None, w=None):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    if h is not None:
        w = max(int(round(im.width * h / im.height)), 1)
    else:
        h = max(int(round(im.height * w / im.width)), 1)
    return np.asarray(im.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255


def column_panel(voids, k, n, y0, y1):
    """Panel k of the n specimen renders that sit side by side in a voids strip."""
    H = voids.shape[0]
    band = voids[int(y0 * H):int(y1 * H)]
    cols = np.where((band.min(axis=2) < 0.985).any(axis=0))[0]
    cr = sorted(sorted(runs(cols, 30), key=lambda t: t[1] - t[0])[-n:])
    c0, c1 = cr[k]
    return main_body(band[:, c0:c1 + 1])


def panel(render, bar, label, ticks=1):
    """One render, its colour bar, and the paper's label beneath."""
    r = scale(render, h=H_REND)
    b = scale(bar_only(bar, ticks), w=W_BAR)
    lab_h = int(FS_LABEL * 1.5)
    w = max(r.shape[1], b.shape[1])
    out = np.ones((H_REND + GAP + b.shape[0] + lab_h, w, 3), np.float32)
    for im, y in ((r, 0), (b, H_REND + GAP)):
        x = (w - im.shape[1]) // 2
        out[y:y + im.shape[0], x:x + im.shape[1]] = im
    img = Image.fromarray((out * 255).astype(np.uint8))
    ImageDraw.Draw(img).text((w // 2, H_REND + GAP + b.shape[0] + lab_h // 2),
                             label, fill='black',
                             font=ImageFont.truetype(FONT, FS_LABEL), anchor='mm')
    return np.asarray(img).astype(np.float32) / 255


def erase_text(a, y0, y1, max_h=140, tol=0.985):
    """Whiten baked titles in a margin band, by component height."""
    from scipy import ndimage as ndi
    a = a.copy()
    H = a.shape[0]
    sl = slice(int(y0 * H), int(y1 * H))
    band = a[sl]
    lab, n = ndi.label(band.min(axis=2) < tol)
    if n:
        drop = np.zeros(n + 1, bool)
        for i, sl2 in enumerate(ndi.find_objects(lab), start=1):
            drop[i] = (sl2[0].stop - sl2[0].start) <= max_h
        band[drop[lab]] = 1.0
    a[sl] = band
    return a


def plain_panel(render, label):
    """A render with a text label beneath and no colour bar, at the same
    render height as the others so the row lines up."""
    r = scale(render, h=H_REND)
    lab_h = int(FS_LABEL * 1.5)
    out = np.ones((H_REND + GAP + lab_h, r.shape[1], 3), np.float32)
    out[:H_REND] = r
    img = Image.fromarray((out * 255).astype(np.uint8))
    ImageDraw.Draw(img).text((r.shape[1] // 2, H_REND + GAP + lab_h // 2),
                             label, fill='black',
                             font=ImageFont.truetype(FONT, FS_LABEL), anchor='mm')
    return np.asarray(img).astype(np.float32) / 255


def compose(panels, out):
    hmax = max(p.shape[0] for p in panels)
    panels = [np.concatenate([p, np.ones((hmax - p.shape[0], p.shape[1], 3),
                                         np.float32)]) for p in panels]
    W = sum(p.shape[1] for p in panels) + COL * (len(panels) - 1)
    canvas = np.ones((hmax + LAB, W, 3), np.float32)
    x, letters = 0, []
    for j, p in enumerate(panels):
        canvas[:hmax, x:x + p.shape[1]] = p
        letters.append((x + p.shape[1] // 2, '(%s)' % 'abc'[j]))
        x += p.shape[1] + COL
    img = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    dr = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, FS_LETTER)
    for cx, t in letters:
        dr.text((cx, hmax + LAB // 2), t, fill='black', font=f, anchor='mm')
    # axis triad at the lower left, on a band added below the panel letters
    import paper_style as ps
    px = int(round(0.11 * W))
    tri = Image.fromarray(ps.triad_rgba(px))
    tall = Image.new('RGBA', (img.size[0], img.size[1] + px - LAB // 2), (255, 255, 255, 255))
    tall.paste(img.convert('RGBA'), (0, 0))
    tall.alpha_composite(tri, (int(0.01 * W), tall.size[1] - px))
    img = tall.convert('RGB')
    img.save(out)
    print('->', out, os.path.getsize(out) // 1024, 'kB', img.size)


def s3():
    v = read(f'{SAND}/S3/S3_voids.png')
    a = panel(column_panel(v, 1, 2, 0.105, 0.70), crop(v, 0.05, 0.95, 0.72, 1.0),
              LABELS['void'])
    r = read(f'{SAND}/S3/S3_1to2_rearrangement.png')
    b = panel(crop(r, 0.0, 1.0, 0.0, 0.76), crop(r, 0.0, 1.0, 0.77, 1.0),
              LABELS['rearr'])
    s = read(f'{SAND}/S3/S3_1to2_strain_grain.png')
    c = panel(main_body(crop(s, 0.345, 0.655, 0.055, 0.80)),
              crop(s, 0.345, 0.655, 0.82, 1.0), LABELS['strain'], ticks=0)
    compose([a, b, c], os.path.join(FIG, 'sand_S3_fields.png'))


def s2():
    v = read(f'{SAND}/S2/S2_voids.png')
    a = panel(column_panel(v, 2, 3, 0.09, 0.72), crop(v, 0.05, 0.95, 0.74, 1.0),
              LABELS['void'])
    # S2 2->3 re-tracked with a seed that carries the 70 x 90 voxel lateral
    # shift of the specimen between the two reconstructions and the radial
    # spread of the head (scripts/sand_track_flow.py); the earlier render
    # assumed no lateral shift.
    r = read(f'{SAND}/S2/S2_2to3_rearrangement.png')
    b = panel(crop(r, 0.0, 1.0, 0.0, 0.78), crop(r, 0.0, 1.0, 0.79, 1.0),
              LABELS['rearr'])
    s = read(f'{SAND}/S2/S2_2to3_strain_grain.png')
    c = panel(main_body(crop(s, 0.345, 0.655, 0.055, 0.80)),
              crop(s, 0.345, 0.655, 0.82, 1.0), LABELS['strain'])
    compose([a, b, c], os.path.join(FIG, 'sand_S2_fields.png'))


if __name__ == '__main__':
    s3()
    s2()
