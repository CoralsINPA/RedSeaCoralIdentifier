"""Plots. Each function takes a table plus an `AnnotationStore` and only draws what it is given.

Stage B starts this module with the one function the review sheets need: a PDF of crop grids
(the format of the legacy Millepora PDFs). Stage C adds the rest (per-genus galleries, mask
overlays). matplotlib is imported inside the function, so `import coralid` stays light.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from coralid._atomic import atomic_path
from coralid.config import CropConfig
from coralid.crops import masked_crop
from coralid.images import ImageReadError, load_in_mask_frame
from coralid.masks import AnnotationStore


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
) -> list[np.ndarray | None]:
    """The masked crop of every row of `items` (`image_name`, `coral_id`), in row order.

    Each photo is read once, in its mask frame. A photo that is missing or unreadable, or a
    crop that is empty, gives None for that row. Crops are shrunk to `max_side` pixels.
    """
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
            crop = masked_crop(
                image,
                anno["bbox"],
                anno["segmentation"],
                padding_pct=crops.padding_pct,
                background=crops.background,
                min_visible=crops.min_visible,
            )
            out[i] = None if crop is None else _thumbnail(crop, max_side)
    return out


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
    from matplotlib.figure import Figure

    if Path(path).exists() and not overwrite:
        raise FileExistsError(f"{path} exists; refusing to overwrite it")
    items = items.reset_index(drop=True)
    thumbs = cut_crops(items, store, images_dir, crops, max_side)
    per_page = rows * cols
    n_pages = max(1, -(-len(items) // per_page))
    with atomic_path(path) as tmp, PdfPages(tmp) as pdf:
        for page in range(n_pages):
            fig = Figure(figsize=(12, 16))
            axes = fig.subplots(rows, cols).flatten()
            for slot, ax in enumerate(axes):
                ax.axis("off")
                i = page * per_page + slot
                if i >= len(items):
                    continue
                thumb = thumbs[i]
                if thumb is None:
                    ax.text(0.5, 0.5, "crop unavailable", ha="center", va="center", fontsize=8)
                else:
                    ax.imshow(thumb)
                ax.set_title(str(items["caption"].iloc[i]), fontsize=8, pad=3)
            if title:
                fig.suptitle(f"{title} (page {page + 1} of {n_pages})", fontsize=12)
            fig.tight_layout()
            pdf.savefig(fig, dpi=110)
    return Path(path)
