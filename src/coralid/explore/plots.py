"""Plots for exploring embeddings: a 2-D map colored by a number or by a category.

Each function takes a table and only draws. matplotlib is imported inside the functions, so
`import coralid.explore.plots` stays light.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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
