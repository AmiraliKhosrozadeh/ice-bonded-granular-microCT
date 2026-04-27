"""Shared helpers for sparse-graph renders.

PyVista renders the 3D scene; we then:
  1) auto-crop the white border around the specimen so no wasted space,
  2) place the cropped scene at the top of a tight composite canvas,
  3) draw a horizontal legend ROW directly underneath (matplotlib native),
  4) overlay a clean XYZ orientation indicator,
  5) save at 300 dpi.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.image as mpimg

# Padding around the auto-cropped specimen so it does not touch the
# canvas edges or the legend strip.
SCENE_MARGIN_PX  = 0      # no extra canvas margin around scene
LEGEND_WIDTH_PX  = 900    # right strip width for the vertical legend
SAVE_DPI         = 300
LEGEND_FONT_SIZE = 56
AXES_FONT_SIZE   = 24

def _autocrop_white(img, white_thresh=0.998, pad=40):
    """Return img cropped to its non-white bounding box plus a small
    safety margin so faint / low-opacity pixels are not chopped off.
    A pixel counts as background only if ALL 3 RGB channels >= 0.998.
    Works for 3- or 4-channel uint8 arrays as returned by mpimg."""
    if img.dtype != np.float32 and img.dtype != np.float64:
        a = img.astype(np.float32) / 255.0
    else:
        a = img
    rgb = a[..., :3]
    nonwhite = (rgb < white_thresh).any(axis=-1)
    if not nonwhite.any():
        return img
    rows = np.any(nonwhite, axis=1)
    cols = np.any(nonwhite, axis=0)
    r0 = max(0, np.argmax(rows) - pad)
    r1 = min(len(rows), len(rows) - np.argmax(rows[::-1]) + pad)
    c0 = max(0, np.argmax(cols) - pad)
    c1 = min(len(cols), len(cols) - np.argmax(cols[::-1]) + pad)
    return img[r0:r1, c0:c1]

def composite(scene_png_path, legend_items, out_path,
              cam_to_world_basis=None):
    """Compose final figure with the 3D PNG on top + a horizontal legend
    below + a clean XYZ orientation marker in the bottom-left corner.

    legend_items : list of (label, hex_color) tuples
    cam_to_world_basis : optional 3x3 matrix where columns are the world
        +X, +Y, +Z directions in image-pixel space (so we can draw the
        XYZ widget consistent with the PyVista camera). If None, we use a
        sensible default that matches the camera in the spam_75 scripts
        (Z down, Y back-right, X right).
    """
    img = mpimg.imread(scene_png_path)
    pad = SCENE_MARGIN_PX
    img_h, img_w = img.shape[:2]
    # Layout: scene on the left, optional legend column on the right.
    # If legend_items is empty / None, skip the legend column entirely.
    has_legend = bool(legend_items)
    legend_w = LEGEND_WIDTH_PX if has_legend else 0
    canvas_w = img_w + 2 * pad + legend_w
    canvas_h = img_h + 2 * pad
    # figsize in inches so that saving at SAVE_DPI yields exactly
    # canvas_w x canvas_h pixels (avoids the 3x explosion that caused
    # PIL save errors at 300 dpi).
    fig = plt.figure(figsize=(canvas_w / SAVE_DPI, canvas_h / SAVE_DPI),
                     dpi=SAVE_DPI)

    # Scene panel
    ax_img = fig.add_axes((pad / canvas_w,            pad / canvas_h,
                           img_w / canvas_w,          img_h / canvas_h))
    ax_img.imshow(img)
    ax_img.set_axis_off()

    if has_legend:
        legend_left = (img_w + 2 * pad) / canvas_w
        ax_leg = fig.add_axes((legend_left, 0.0,
                               legend_w / canvas_w, 1.0))
        ax_leg.set_axis_off()
        handles = [Line2D([0], [0], marker='o', linestyle='',
                          markerfacecolor=hex_col,
                          markeredgecolor=hex_col,
                          markersize=30, label=label.upper())
                   for label, hex_col in legend_items]
        leg = ax_leg.legend(handles=handles, loc='center left', ncol=1,
                            frameon=False,
                            fontsize=LEGEND_FONT_SIZE,
                            handletextpad=0.5,
                            labelspacing=1.4,
                            borderaxespad=0.5)
        for txt in leg.get_texts():
            txt.set_color('#1a1a1a')

    # XYZ orientation now drawn by PyVista's native add_axes() widget
    # in the rendered PNG, so no matplotlib overlay here.

    # No bbox_inches='tight' -- it would expand the canvas around the
    # legend and break the layout. Save to a tmp filename first then
    # rename, so an open-in-Windows lock on the existing PNG doesn't
    # cause an Errno 22 ("Invalid argument") write error.
    tmp_path = out_path + '.new.png'
    fig.savefig(tmp_path, dpi=SAVE_DPI, facecolor='white', edgecolor='none')
    plt.close(fig)
    try:
        os.replace(tmp_path, out_path)
    except OSError:
        # Target is locked by another process; leave the new file in
        # place and emit a clear message.
        print(f'  NOTE: could not overwrite {out_path} '
              f'(probably open in a viewer); new image at {tmp_path}')
