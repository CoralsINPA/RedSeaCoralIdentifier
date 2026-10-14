"""Photo orientation, and splitting any table or metric by it.

The mask JSON frame has its long side at 2560: 1,493 photos are 2560x1920 (landscape) and 221 are
1920x2560 (portrait). The legacy code mis-cropped every portrait photo, so results are reported
per orientation. The orientation of a photo comes from the frame in the manifest
(`frame_w`, `frame_h`): `frame_w > frame_h` is landscape, `frame_w < frame_h` is portrait, and a
square frame is its own value.

- `image_orientations(frames)` maps each image to its orientation.
- `add_orientation(df, frames)` adds an `orientation` column to any table with `image_name`.
- `split_by_orientation(df, frames)` splits such a table.
- `by_orientation(df, frames, func)` runs a metric on everything and on each orientation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

LANDSCAPE = "landscape"
PORTRAIT = "portrait"
SQUARE = "square"
UNKNOWN = "unknown"  # an image that has no frame in `frames`
ORIENTATIONS = (LANDSCAPE, PORTRAIT)  # the two the reports always show, even when empty


def orientation_of(frame_w: int, frame_h: int) -> str:
    """`landscape` if the frame is wider than tall, `portrait` if taller, else `square`."""
    if frame_w > frame_h:
        return LANDSCAPE
    if frame_w < frame_h:
        return PORTRAIT
    return SQUARE


def image_orientations(frames: pd.DataFrame) -> pd.Series:
    """The orientation of each image: a Series indexed by `image_name`.

    `frames` is a manifest (or any table) with `image_name`, `frame_w` and `frame_h`, one or
    more rows per image. An image whose rows disagree about the frame raises `ValueError`.
    """
    missing = {"image_name", "frame_w", "frame_h"} - set(frames.columns)
    if missing:
        raise ValueError(f"frames need image_name, frame_w and frame_h; missing {sorted(missing)}")
    unique = frames[["image_name", "frame_w", "frame_h"]].drop_duplicates()
    clash = unique[unique.duplicated("image_name", keep=False)]
    if len(clash):
        raise ValueError(
            f"images with more than one frame size: {sorted(set(clash['image_name']))[:5]}"
        )
    sizes = zip(unique["frame_w"], unique["frame_h"], strict=True)
    values = [orientation_of(int(w), int(h)) for w, h in sizes]
    return pd.Series(values, index=pd.Index(unique["image_name"].to_numpy(), name="image_name"))


def add_orientation(df: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """A copy of `df` with an `orientation` column (`unknown` for images without a frame)."""
    if "image_name" not in df.columns:
        raise ValueError("the table needs an image_name column")
    lookup = image_orientations(frames)
    out = df.copy()
    out["orientation"] = out["image_name"].map(lookup).fillna(UNKNOWN).astype(object)
    return out


def split_by_orientation(df: pd.DataFrame, frames: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split `df` by orientation: `{"landscape": ..., "portrait": ...}` (empty tables included).

    Other values (`square`, `unknown`) appear only if some row has them.
    """
    tagged = add_orientation(df, frames)
    parts = {name: tagged[tagged["orientation"] == name] for name in ORIENTATIONS}
    for name in tagged["orientation"].unique():
        if name not in parts:
            parts[name] = tagged[tagged["orientation"] == name]
    return {name: part.drop(columns="orientation") for name, part in parts.items()}


def by_orientation(
    df: pd.DataFrame,
    frames: pd.DataFrame,
    func: Callable[[pd.DataFrame], Any],
    include_all: bool = True,
) -> dict[str, Any]:
    """Run `func` on all of `df` (key `all`) and on each orientation's rows.

    `func` also runs on an empty subset (a portrait-free sample, say), so it must accept one.
    """
    results: dict[str, Any] = {"all": func(df)} if include_all else {}
    for name, part in split_by_orientation(df, frames).items():
        results[name] = func(part)
    return results
