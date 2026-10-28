"""Shared manuscript figure style, matching the landing-site Figure 1."""

BLUE = "#1f5fa8"
ORANGE = "#c8541b"
GREEN = "#2e7d32"
RED = "#b23a3a"
PURPLE = "#7562a3"
INK = "#1a1a19"
QUIET = "#666666"
GRID = "#dddddd"
BLUE_FILL = "#f3f6fb"
ORANGE_FILL = "#fbf3ee"
NEUTRAL_FILL = "#f4f4f4"


def apply_style():
    """Use the reference figure's serif typography and understated strokes."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 6.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": QUIET,
        "axes.linewidth": 0.65,
        "xtick.color": QUIET,
        "ytick.color": QUIET,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.4,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    })


def style_axes(ax):
    """Consistent unobtrusive axes for the three numerical plots."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.tick_params(width=0.65, length=3)

