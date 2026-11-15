"""Single source of truth for figure sizing and typography in the paper.

The rule this module enforces: a figure is generated at exactly the width it
will occupy on the page, and LaTeX includes it without rescaling.  Font sizes
set here are therefore the font sizes that appear in the PDF.

Usage:

    from experiments.paper_style import apply_paper_style, figure_size, save_figure

    apply_paper_style()
    fig, ax = plt.subplots(figsize=figure_size(0.5, 0.72))   # half column
    ...
    save_figure(fig, out_path)

and on the LaTeX side, with no width= key at all:

    \\includegraphics{plots/tpc_ntasks_mean_latency_rps5.pdf}
"""

from pathlib import Path

import matplotlib.pyplot as plt

# ACM sigconf geometry, measured from the compiled paper (page 11, US Letter).
COLUMN_WIDTH_IN = 241.14 / 72.27   # 3.336 in, \columnwidth
TEXT_WIDTH_IN = 506.61 / 72.27     # 7.010 in, \textwidth

# Body text is 9 pt.  Figure text sits one to two points below it.
BASE_FONT_PT = 7.0
TICK_FONT_PT = 6.5
ANNOTATION_FONT_PT = 6.0


def figure_size(fraction=1.0, aspect=0.62, span="column"):
    """Exact figsize for a figure printed at `fraction` of a column or the page.

    fraction: share of the target width, e.g. 0.5 for two subfigures side by side.
              Leave ~0.02 of slack when subfigures sit next to each other.
    aspect:   height / width.
    span:     "column" for a one-column figure, "text" for a figure* spanning both.
    """
    width = (TEXT_WIDTH_IN if span == "text" else COLUMN_WIDTH_IN) * fraction
    return (width, width * aspect)


def apply_paper_style():
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     "black",
        "axes.labelcolor":    "black",
        "axes.linewidth":     0.6,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "grid.color":         "#cccccc",
        "grid.linestyle":     ":",
        "grid.linewidth":     0.4,
        "grid.alpha":         1.0,
        "xtick.color":        "black",
        "ytick.color":        "black",
        "xtick.major.width":  0.5,
        "ytick.major.width":  0.5,
        "xtick.major.size":   2.5,
        "ytick.major.size":   2.5,
        "xtick.major.pad":    1.5,
        "ytick.major.pad":    1.5,
        "text.color":         "black",
        "font.family":        "sans-serif",

        # Printed point sizes.  Do not override these per-plot.
        "font.size":          BASE_FONT_PT,
        "axes.titlesize":     BASE_FONT_PT,
        "axes.labelsize":     BASE_FONT_PT,
        "xtick.labelsize":    TICK_FONT_PT,
        "ytick.labelsize":    TICK_FONT_PT,
        "legend.fontsize":    TICK_FONT_PT,
        "legend.title_fontsize": TICK_FONT_PT,

        "axes.labelpad":      1.5,
        "legend.borderpad":   0.3,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.2,
        "legend.frameon":     False,

        "lines.linewidth":    1.0,
        "lines.markersize":   3.0,

        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.facecolor":  "white",

        # Fit the axes inside the requested canvas instead of growing the canvas
        # around the axes.  This is what keeps the saved width equal to figsize.
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.01,
        "figure.constrained_layout.w_pad": 0.01,
        "figure.constrained_layout.hspace": 0.02,
        "figure.constrained_layout.wspace": 0.02,
    })


def save_figure(fig, out_path):
    """Save at exactly `fig`'s figsize.

    Deliberately no bbox_inches="tight": a tight bbox crops to the drawn content,
    so the output width depends on how long the tick labels happen to be, and a
    later \\includegraphics[width=...] rescales the fonts by an unpredictable factor.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    if out_path.suffix != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"))
    print(f"[Plot] Saved: {out_path} ({fig.get_size_inches()[0]:.2f} in wide)")
