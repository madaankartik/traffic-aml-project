"""Figure helpers with one consistent, colour-blind-safe style.

Categorical colours are assigned in a *fixed* order (never cycled), sequential
scales use a single blue hue, and diverging scales use blue <-> red around a
neutral grey, following standard data-visualisation accessibility practice.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.colors import LinearSegmentedColormap   # noqa: E402
import numpy as np                                  # noqa: E402

from . import config as C                           # noqa: E402

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODEL_COLOURS = {                       # fixed assignment: colour follows the model, never its rank
    "HA": "#898781", "Persistence": "#c3c2b7", "Ridge-local": "#eda100",
    "XGB-local": "#2a78d6", "XGB-spatial": "#eb6834", "RF-local": "#1baf7a", "RF-spatial": "#6250d6",
}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
CMAP_SEQ_R = CMAP_SEQ.reversed()
CMAP_DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#7a1f1f"])


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK2, "text.color": INK,
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.6, "figure.dpi": 120,
    })


def save(fig, name: str, directory: Path = C.FIGURES_DIR) -> Path:
    """Save one PNG per figure (200 dpi; used by the report, slides and notebooks alike)."""
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return directory / f"{name}.png"


def bar_with_gap(ax, x, heights, width, colour, label=None):
    """Thin bars with a 2 px-equivalent gap and rounded feel (flat top, no edge)."""
    return ax.bar(x, heights, width=width * 0.92, color=colour, label=label, linewidth=0)


def hide_spines(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def annotate_bars(ax, bars, fmt="{:.2f}", fontsize=7):
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom",
                        fontsize=fontsize, color=INK2, xytext=(0, 2), textcoords="offset points")
