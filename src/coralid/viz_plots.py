"""Plots that need no photo: a confusion heat map and the filter funnel.

Each takes a table and only draws (see `coralid.viz`, which re-exports them). matplotlib is
imported inside the functions, so `import coralid` stays light.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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
