"""Plots that need no photo: 2-D maps, a confusion heat map, the filter funnel.

Each takes a table and only draws (see `coralid.viz`, which re-exports them). matplotlib is
imported inside the functions, so `import coralid` stays light.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def _natural_key(text: str) -> tuple[int, int, str]:
    """Sort key that puts numbers (as text) in numeric order and `noise` last."""
    if text.lstrip("-").isdigit():
        return (0, int(text), "")
    return (1, 0, text)


def category_palette(values: Sequence[Any] | pd.Series) -> dict[str, tuple[float, ...]]:
    """One color per distinct value (as text), in a stable order: pass it to every panel.

    Up to 20 categories use `tab20`; more are spread over `nipy_spectral`.
    """
    from matplotlib import colormaps

    names = sorted({str(v) for v in values}, key=_natural_key)
    if len(names) <= 20:
        cycle = colormaps["tab20"]
        return {name: cycle(i) for i, name in enumerate(names)}
    cycle = colormaps["nipy_spectral"]
    return {name: cycle(i / max(1, len(names) - 1)) for i, name in enumerate(names)}


def embedding_scatter(
    table: pd.DataFrame,
    x: str = "x",
    y: str = "y",
    color: str | None = None,
    ax: Any = None,
    title: str = "",
    s: float = 10,
    alpha: float = 0.6,
    cmap: str = "nipy_spectral",
    palette: dict[str, Any] | None = None,
    legend: bool = True,
) -> Any:
    """A 2-D map of the rows of `table` (columns `x` and `y`), colored by `color`; returns a Figure.

    A numeric `color` column is drawn on the color map `cmap`. Any other column (the predicted
    genus, say) is drawn category by category with a legend that counts each; `palette`
    (`category_palette`) fixes the colors, so that several panels agree, and a value it does not
    have is drawn light gray. A number column gets a color bar. `legend=False` leaves out the
    legend and the color bar (for the first of two panels that share them). Pass `ax` to draw
    into a panel of your own figure (the figure of that axes is returned).
    """
    from matplotlib.figure import Figure

    if ax is None:
        ax = Figure(figsize=(8, 7)).subplots()
    if color is not None and not pd.api.types.is_numeric_dtype(table[color]):
        labels = table[color].astype(str)
        colors = palette or category_palette(labels)
        for name in sorted(set(labels), key=_natural_key):
            rows = (labels == name).to_numpy()
            ax.scatter(
                table[x][rows],
                table[y][rows],
                color=colors.get(name, "lightgray"),
                s=s,
                alpha=alpha,
                label=f"{name} ({int(rows.sum())})",
            )
        if legend:
            ax.legend(
                title=color,
                fontsize=7,
                markerscale=2,
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                frameon=False,
            )
    else:
        points = ax.scatter(
            table[x],
            table[y],
            c=table[color] if color else "royalblue",
            cmap=cmap if color else None,
            s=s,
            alpha=alpha,
        )
        if color is not None and legend:
            ax.figure.colorbar(points, ax=ax, shrink=0.8, label=color)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    return ax.figure


def funnel_chart(funnel: pd.DataFrame, title: str = "") -> Any:
    """The filter funnel of `coralid.funnel.filter_funnel` as two panels; returns a Figure.

    Left: the masks left after each stage (all photos). Right: the share of each orientation's
    masks that each filter removed (the stages between the first and the last row).
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=(14, 4.5))
    left, right = fig.subplots(1, 2)
    stages = list(funnel.index)
    left.barh(stages[::-1], funnel["all_remaining"].to_numpy()[::-1], color="steelblue")
    for position, value in enumerate(funnel["all_remaining"].to_numpy()[::-1]):
        left.text(value, position, f" {int(value):,}", va="center", fontsize=8)
    left.set_xlabel("masks left")
    left.set_title("Masks left after each filter")
    reasons = stages[1:-1]
    height = 0.4
    for offset, group in zip((-height / 2, height / 2), ("landscape", "portrait"), strict=True):
        total = max(1, int(funnel.loc["all masks", f"{group}_remaining"]))
        share = 100 * funnel.loc[reasons, f"{group}_removed"].to_numpy(dtype=float) / total
        right.barh(
            [i + offset for i in range(len(reasons))],
            share,
            height=height,
            label=f"{group} ({total:,} masks)",
        )
    right.set_yticks(range(len(reasons)), reasons)
    right.invert_yaxis()
    right.set_xlabel("% of the orientation's masks removed")
    right.set_title("Removed by each filter, by photo orientation")
    right.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def confusion_heatmap(
    confusion: pd.DataFrame, ax: Any = None, normalize: bool = False, title: str = ""
) -> Any:
    """A confusion matrix (true in rows, predicted in columns) as a heat map; returns a Figure.

    `normalize` shows each row as shares of its true genus. Counts are printed in the cells of a
    matrix with at most 25 classes.
    """
    from matplotlib.figure import Figure

    if ax is None:
        ax = Figure(figsize=(9, 8)).subplots()
    values = confusion.to_numpy(dtype=float)
    if normalize:
        totals = values.sum(axis=1, keepdims=True)
        values = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
    image = ax.imshow(values, cmap="Blues")
    ax.figure.colorbar(image, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(confusion.columns)), list(confusion.columns), rotation=90)
    ax.set_yticks(range(len(confusion.index)), list(confusion.index))
    ax.set_xlabel(confusion.columns.name or "predicted")
    ax.set_ylabel(confusion.index.name or "true")
    if len(confusion) <= 25:
        for (i, j), v in np.ndenumerate(values):
            if v:
                text = f"{v:.2f}" if normalize else f"{int(v)}"
                ax.text(j, i, text, ha="center", va="center", fontsize=7)
    if title:
        ax.set_title(title)
    return ax.figure
