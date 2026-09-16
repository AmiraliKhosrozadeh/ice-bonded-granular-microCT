"""
Shared publication plot style for the Sand pipeline.
Copied from the glass `pipeline_github` style so all figures match the
project's STRICT publication rules (large fonts, full box, no on-data
annotations, one plot per file). Import: from plot_style import apply_style, FS, save_fig
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FS = 30          # font size
LW = 3           # line width


def apply_style():
    plt.rcParams.update({
        "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
        "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS - 6,
        "lines.linewidth": LW, "axes.linewidth": 1.8,
        "xtick.major.width": 1.8, "ytick.major.width": 1.8,
        "xtick.major.size": 8, "ytick.major.size": 8,
        "xtick.minor.size": 4, "ytick.minor.size": 4,
        "grid.linewidth": 0.8, "grid.alpha": 0.4,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def style_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.8)
    ax.grid(True, which='major', axis='both', linewidth=0.8, alpha=0.4)


def save_fig(fig, path, dpi=300, pdf=True):
    """Save a publication figure. Emits a 300-dpi PNG and (for journal
    submission, 'first paper' / Powder Technology convention) a vector PDF
    alongside it. Set pdf=False for raster overlays/micrographs."""
    import os, time

    def _rm(p):
        if not os.path.exists(p):
            return
        for attempt in range(2):
            try:
                os.remove(p)
                return
            except PermissionError:
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                raise PermissionError(f"Cannot overwrite {p} — close any viewer and re-run.")

    _rm(path)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    if pdf:
        pdf_path = os.path.splitext(path)[0] + ".pdf"
        _rm(pdf_path)
        fig.savefig(pdf_path, bbox_inches='tight')   # vector
    plt.close(fig)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved: {path}  ({size_kb:.0f} KB)" + (" + .pdf" if pdf else ""))
