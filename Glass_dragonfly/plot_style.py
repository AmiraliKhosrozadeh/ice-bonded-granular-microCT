"""
Shared plot style for all CT analysis scripts.
Import this at the top of any script that creates plots.

Usage:
    from plot_style import apply_style, FS, LW, save_fig
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Global constants
FS = 30          # font size
LW = 3           # line width

def apply_style():
    """Apply MATLAB-style plot settings to all subsequent plots."""
    plt.rcParams.update({
        "font.size"        : FS,
        "axes.titlesize"   : FS,
        "axes.labelsize"   : FS,
        "xtick.labelsize"  : FS,
        "ytick.labelsize"  : FS,
        "legend.fontsize"  : FS - 6,
        "lines.linewidth"  : LW,
        "axes.linewidth"   : 1.8,
        "xtick.major.width": 1.8,
        "ytick.major.width": 1.8,
        "xtick.major.size" : 8,
        "ytick.major.size" : 8,
        "xtick.minor.size" : 4,
        "ytick.minor.size" : 4,
        "grid.linewidth"   : 0.8,
        "grid.alpha"       : 0.4,
        "figure.facecolor" : "white",
        "axes.facecolor"   : "white",
    })


def style_axes(ax):
    """Apply box + grid to a single axes."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.8)
    ax.grid(True, which='major', axis='both', linewidth=0.8, alpha=0.4)


def save_fig(fig, path, dpi=150):
    """Save figure, overwriting any existing file.

    Pre-deletes the target so a locked file (image viewer, PowerPoint,
    OneDrive sync) raises immediately instead of silently leaving the old
    image in place. Retries once after 0.5s to shake off transient AV locks.
    """
    import os, time
    if os.path.exists(path):
        for attempt in range(2):
            try:
                os.remove(path)
                break
            except PermissionError:
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                raise PermissionError(
                    f"Cannot overwrite {path} — it is locked by another "
                    f"program (image viewer, PowerPoint, OneDrive). Close it "
                    f"and re-run."
                )
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(path)))
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved: {path}  ({size_kb:.0f} KB, {mtime})")
