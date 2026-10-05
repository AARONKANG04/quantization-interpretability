"""Shared chart style: the validated reference palette (dataviz skill), thin marks, recessive grid, one accent."""
import matplotlib as mpl

ACCENT = "#2a78d6"      # categorical slot 1 (blue): the series the title is about
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order, never cycled
MUTED = "#c3c2b7"       # de-emphasis gray for context series
INK = "#0b0b0b"; INK2 = "#52514e"; AXIS = "#c3c2b7"; GRID = "#e1e0d9"; SURFACE = "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS_GOOD, STATUS_BAD = "#0ca30c", "#d03b3b"


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2, "lines.markersize": 6,
        "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def finish(ax, title, note=None):
    ax.set_title(title, loc="left", color=INK)
    if note:
        ax.text(0, -0.18, note, transform=ax.transAxes, fontsize=8, color=INK2, va="top")
