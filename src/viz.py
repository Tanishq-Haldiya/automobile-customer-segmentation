"""Shared chart style and figure helpers.

Follows the project's visualisation rules: recessive grid and axes, thin marks,
a surface-coloured gap between adjacent/stacked fills, direct labels rather than
a number on every mark, text in ink tokens (never a series colour), one y-axis
per chart, and a fixed categorical hue order that is never cycled.

Palettes are validated by `src/palette_check.py`; see the note in config.py.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import config as C

SEQ_CMAP = LinearSegmentedColormap.from_list("seq", ["#f4f9fd", C.SEQUENTIAL_HUE])
DIV_CMAP = LinearSegmentedColormap.from_list("div", list(C.DIVERGING))


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": C.SURFACE,
        "axes.facecolor": C.SURFACE,
        "savefig.facecolor": C.SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": C.INK,
        "axes.labelcolor": C.INK_SECONDARY,
        "axes.labelsize": 9.5,
        "axes.edgecolor": C.GRID,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": C.GRID,
        "grid.linewidth": 0.7,
        "xtick.color": C.INK_SECONDARY,
        "ytick.color": C.INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
    })


def despine(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.grid(axis="x", visible=False)


def save(fig, name: str) -> str:
    path = C.FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(C.ROOT)).replace("\\", "/")


def cluster_colors(n: int) -> list:
    if n > len(C.CLUSTER_COLORS):
        raise ValueError("palette holds %d hues; fold extras into 'Other'"
                         % len(C.CLUSTER_COLORS))
    return C.CLUSTER_COLORS[:n]


# --------------------------------------------------------------------------- #
# mark helpers
# --------------------------------------------------------------------------- #
def bar_labels(ax, bars, fmt="{:.0f}", pad=3, color=None) -> None:
    """Direct-label bars in an ink token, never the series colour."""
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, pad), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9,
                    color=color or C.INK_SECONDARY)


def stacked_shares(ax, index, share_df, colors, label_min=0.08) -> None:
    """100% stacked bars with a 2px surface gap between segments.

    Only segments above `label_min` are labelled, so the chart is never a wall
    of numbers.
    """
    bottom = np.zeros(len(index))
    for col, color in zip(share_df.columns, colors):
        vals = share_df[col].to_numpy()
        ax.bar(index, vals, bottom=bottom, color=color, width=0.68,
               edgecolor=C.SURFACE, linewidth=2.0, label=str(col))
        for x, v, b in zip(range(len(index)), vals, bottom):
            if v >= label_min:
                ax.text(x, b + v / 2, "%.0f%%" % (v * 100), ha="center",
                        va="center", fontsize=8.5, color="white", weight="bold")
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])


def heatmap(ax, matrix: pd.DataFrame, cmap=None, fmt="{:.0f}", cbar_label=None,
            vmin=None, vmax=None, center_text_threshold=0.6):
    cmap = cmap or SEQ_CMAP
    data = matrix.to_numpy(dtype=float)
    vmin = np.nanmin(data) if vmin is None else vmin
    vmax = np.nanmax(data) if vmax is None else vmax
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), [str(c) for c in matrix.columns])
    ax.set_yticks(range(matrix.shape[0]), [str(i) for i in matrix.index])
    span = (vmax - vmin) or 1.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            frac = (data[i, j] - vmin) / span
            ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                    fontsize=9,
                    color="white" if frac > center_text_threshold else C.INK)
    ax.grid(False)
    for side in ax.spines:
        ax.spines[side].set_visible(False)
    if cbar_label:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(cbar_label, color=C.INK_SECONDARY, fontsize=9)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=C.INK_SECONDARY, labelsize=8)
    return im


def note(fig, text: str, y: float = -0.03) -> None:
    fig.text(0.0, y, text, ha="left", va="top", fontsize=8.5,
             color=C.INK_MUTED, wrap=True)
