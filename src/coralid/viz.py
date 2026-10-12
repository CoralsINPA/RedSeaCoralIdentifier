"""Plots. Each function takes a table (plus an `AnnotationStore` for pictures) and only draws.

Nothing here computes: rows to show are chosen by the caller (`coralid.clustering` has the
samplers and the projections) and arrive in a table keyed by `image_name` and `coral_id`.

- `crop_grid_pdf`: a PDF of crop grids with one caption each (the review sheets' format);
- `crop_gallery_pdf`: the same, with the rows grouped by a column, one section per group (the
  K-Means cluster gallery of CM cell 14, a genus gallery);
- `mask_overlay`: a photo with its masks drawn, optionally colored by a column (CM cell 8);
- `embedding_scatter`, `category_palette`, `confusion_heatmap` and `funnel_chart`: the plots that
  need no photo (a 2-D map colored by a number or a category, a confusion matrix, the filter
  funnel). They live in `coralid.viz_plots` and are importable from here too.

Crops are always cut from the photo loaded in the mask frame. matplotlib is imported inside the
functions, so `import coralid` stays light.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from coralid._atomic import atomic_path
from coralid.config import CropConfig
from coralid.crops import bbox_crop, masked_crop, masked_square_crop
from coralid.images import ImageReadError, load_in_mask_frame
from coralid.masks import AnnotationStore, decode_rle
from coralid.viz_plots import category_palette, confusion_heatmap, embedding_scatter, funnel_chart

__all__ = [
    "CROP_VIEWS",
    "category_palette",
    "confusion_heatmap",
    "crop_gallery_pdf",
    "crop_grid_pdf",
    "cut_crops",
    "embedding_scatter",
    "funnel_chart",
    "mask_overlay",
]

CROP_VIEWS = ("masked", "square", "bbox")


def _thumbnail(crop: np.ndarray, max_side: int) -> np.ndarray:
    scale = max_side / max(crop.shape[:2])
    if scale >= 1:
        return crop
    size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def cut_crops(
    items: pd.DataFrame,
    store: AnnotationStore,
    images_dir: str | os.PathLike[str],
    crops: CropConfig | None = None,
    max_side: int = 512,
    view: str = "masked",
) -> list[np.ndarray | None]:
    """The crop of every row of `items` (`image_name`, `coral_id`), in row order.

    `view` is `masked` (the classifier's crop, the default), `square` (the masked square crop
    the DINO models saw) or `bbox` (the padded box with the background left in).
    Each photo is read once, in its mask frame. A photo that is missing or unreadable, or a
    crop that is empty, gives None for that row. Crops are shrunk to `max_side` pixels.
    """
    if view not in CROP_VIEWS:
        raise ValueError(f"view must be one of {CROP_VIEWS}, got {view!r}")
    crops = crops or CropConfig()
    out: list[np.ndarray | None] = [None] * len(items)
    positions: dict[str, list[int]] = {}
    for i, name in enumerate(items["image_name"]):
        positions.setdefault(str(name), []).append(i)
    for name, rows in positions.items():
        try:
            image = load_in_mask_frame(Path(images_dir) / name, *store.frame(name))
        except (ImageReadError, FileNotFoundError):
            continue
        by_id = store.load(name).by_id()
        for i in rows:
            anno = by_id.get(int(items["coral_id"].iloc[i]))
            if anno is None:
                continue
            if view == "masked":
                crop = masked_crop(
                    image,
                    anno["bbox"],
                    anno["segmentation"],
                    padding_pct=crops.padding_pct,
                    background=crops.background,
                    min_visible=crops.min_visible,
                )
            elif view == "square":
                crop = masked_square_crop(image, anno["bbox"], anno["segmentation"])
            else:
                crop = bbox_crop(image, anno["bbox"], crops.bbox_padding_pct)
            out[i] = None if crop is None else _thumbnail(crop, max_side)
    return out


def _add_pages(
    pdf: Any,
    thumbs: Sequence[np.ndarray | None],
    captions: Sequence[str],
    rows: int,
    cols: int,
    title: str,
) -> None:
    """Draw `rows` x `cols` thumbnails per page into an open `PdfPages`."""
    from matplotlib.figure import Figure

    per_page = rows * cols
    n_pages = max(1, -(-len(thumbs) // per_page))
    for page in range(n_pages):
        fig = Figure(figsize=(12, 16))
        axes = fig.subplots(rows, cols).flatten()
        for slot, ax in enumerate(axes):
            ax.axis("off")
            i = page * per_page + slot
            if i >= len(thumbs):
                continue
            if thumbs[i] is None:
                ax.text(0.5, 0.5, "crop unavailable", ha="center", va="center", fontsize=8)
            else:
                ax.imshow(thumbs[i])
            ax.set_title(captions[i], fontsize=8, pad=3)
        if title:
            fig.suptitle(f"{title} (page {page + 1} of {n_pages})", fontsize=12)
        fig.tight_layout()
        pdf.savefig(fig, dpi=110)


def crop_grid_pdf(
    items: pd.DataFrame,
    path: str | os.PathLike[str],
    store: AnnotationStore,
    images_dir: str | os.PathLike[str],
    crops: CropConfig | None = None,
    rows: int = 4,
    cols: int = 3,
    title: str = "",
    max_side: int = 512,
    overwrite: bool = False,
) -> Path:
    """Write a multi-page PDF with `rows` x `cols` masked crops per page.

    `items` needs `image_name`, `coral_id` and `caption`; the caption is printed above the crop
    and is the only text, so a blind review shows nothing the caller does not put there. An
    existing file is an error unless `overwrite`. Returns the path.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    if Path(path).exists() and not overwrite:
        raise FileExistsError(f"{path} exists; refusing to overwrite it")
    items = items.reset_index(drop=True)
    thumbs = cut_crops(items, store, images_dir, crops, max_side)
    captions = [str(c) for c in items["caption"]]
    with atomic_path(path) as tmp, PdfPages(tmp) as pdf:
        _add_pages(pdf, thumbs, captions, rows, cols, title)
    return Path(path)


def crop_gallery_pdf(
    table: pd.DataFrame,
    path: str | os.PathLike[str],
    store: AnnotationStore,
    images_dir: str | os.PathLike[str],
    group_col: str | None = None,
    caption_col: str | None = None,
    crops: CropConfig | None = None,
    rows: int = 4,
    cols: int = 3,
    view: str = "masked",
    title: str = "",
    max_side: int = 512,
    overwrite: bool = False,
) -> Path:
    """Write a PDF of crops with one section per value of `group_col` (all in one if None).

    `table` needs `image_name` and `coral_id`. It is drawn in row order and the groups appear in
    the order of their first row, so the caller chooses and orders what is shown
    (`coralid.clustering.sample_per_group` draws the rows of a cluster gallery). The caption is
    `caption_col` if given, else the key `IMG_4033.JPG#14`. Each section starts on a new page and
    is titled with its group value. An existing file is an error unless `overwrite`.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    if Path(path).exists() and not overwrite:
        raise FileExistsError(f"{path} exists; refusing to overwrite it")
    table = table.reset_index(drop=True)
    thumbs = cut_crops(table, store, images_dir, crops, max_side, view)
    if caption_col is not None:
        captions = [str(c) for c in table[caption_col]]
    else:
        keys = zip(table["image_name"], table["coral_id"], strict=True)
        captions = [f"{name}#{int(coral_id)}" for name, coral_id in keys]
    groups = table[group_col].astype(str) if group_col else pd.Series("", index=table.index)
    with atomic_path(path) as tmp, PdfPages(tmp) as pdf:
        for value in dict.fromkeys(groups):
            at = np.flatnonzero((groups == value).to_numpy())
            heading = f"{group_col} = {value} ({len(at)} shown)" if group_col else title
            if group_col and title:
                heading = f"{title}: {heading}"
            _add_pages(pdf, [thumbs[i] for i in at], [captions[i] for i in at], rows, cols, heading)
    return Path(path)


def mask_overlay(
    table: pd.DataFrame,
    store: AnnotationStore,
    images_dir: str | os.PathLike[str],
    image_name: str,
    color_col: str | None = None,
    max_masks: int = 128,
    alpha: float = 0.35,
) -> Any:
    """A photo, in its mask frame, with the masks of `table` drawn over it; returns a Figure.

    `table` rows for `image_name` (`coral_id`, and `area` to draw the largest first) are shown, at
    most `max_masks`. Without `color_col` all are purple, as in CM cell 8; with it each value gets
    its own color and a legend (an empty value, such as a kept mask's `reject_reason`, is `kept`).
    """
    from matplotlib import colormaps
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    shown = table[table["image_name"] == image_name]
    if "area" in shown.columns:
        shown = shown.sort_values("area", ascending=False)
    shown = shown.head(max_masks)
    width, height = store.frame(image_name)
    photo = load_in_mask_frame(Path(images_dir) / image_name, width, height)
    by_id = store.load(image_name).by_id()
    if color_col is None:
        values = pd.Series("masks", index=shown.index)
        palette = {"masks": (138 / 255, 43 / 255, 226 / 255)}
    else:
        values = shown[color_col].where(shown[color_col].notna(), "kept").astype(str)
        cycle = colormaps["tab10"]
        palette = {v: cycle(i % 10)[:3] for i, v in enumerate(sorted(set(values)))}
    overlay = np.zeros((height, width, 4))
    for coral_id, value in zip(shown["coral_id"], values, strict=True):
        mask = decode_rle(by_id[int(coral_id)]["segmentation"])
        overlay[mask, :3] = palette[value]
        overlay[mask, 3] = alpha
    fig = Figure(figsize=(12, 12 * height / width))
    ax = fig.subplots()
    ax.imshow(photo)
    ax.imshow(overlay)
    ax.axis("off")
    ax.set_title(f"{image_name}: {len(shown)} masks")
    if color_col is not None:
        ax.legend(handles=[Patch(color=c, label=v) for v, c in palette.items()], loc="upper right")
    return fig
