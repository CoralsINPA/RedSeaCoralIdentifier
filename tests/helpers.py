"""Synthetic masks, annotations and mask JSON folders for the tests (no real data)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from coralid.masks import encode_mask, write_mask_json


def rect_mask(shape: tuple[int, int], y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """A boolean mask with rows y0:y1 and columns x0:x1 set."""
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def ellipse_mask(shape: tuple[int, int], cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
    """A boolean ellipse mask."""
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def mask_bbox(mask: np.ndarray) -> list[float]:
    """The tight `[x, y, w, h]` box of a non-empty mask, as floats (like the SAM output)."""
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def make_annotation(
    mask: np.ndarray,
    ann_id: int,
    area: float | None = None,
    bbox: list[float] | None = None,
) -> dict[str, Any]:
    """An annotation dict in the mask-JSON layout, built from a boolean mask."""
    return {
        "segmentation": encode_mask(mask),
        "bbox": bbox if bbox is not None else mask_bbox(mask),
        "area": int(mask.sum()) if area is None else area,
        "predicted_iou": 0.9,
        "crop_box": [0, 0, mask.shape[1], mask.shape[0]],
        "stability_score": 0.8,
        "point_coords": [[float(mask.shape[1] // 2), float(mask.shape[0] // 2)]],
        "id": ann_id,
    }


def random_mask(rng: np.random.RandomState, shape: tuple[int, int]) -> np.ndarray:
    """A random non-empty blob: a rectangle, an ellipse or a union of two of them."""
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    for _ in range(rng.randint(1, 3)):
        if rng.rand() < 0.5:
            y0, x0 = rng.randint(0, h - 4), rng.randint(0, w - 4)
            y1, x1 = rng.randint(y0 + 3, h + 1), rng.randint(x0 + 3, w + 1)
            mask |= rect_mask(shape, y0, y1, x0, x1)
        else:
            mask |= ellipse_mask(
                shape,
                rng.uniform(0, h),
                rng.uniform(0, w),
                rng.uniform(2, h / 2),
                rng.uniform(2, w / 2),
            )
    if not mask.any():
        mask[h // 2, w // 2] = True
    return mask


def write_json_dir(
    json_dir: Path,
    entries: list[tuple[str, int, int, list[dict[str, Any]], str | None]],
) -> Path:
    """Write mask JSONs: each entry is `(image_name, width, height, annotations, note)`."""
    from coralid.masks import json_name_for

    json_dir.mkdir(parents=True, exist_ok=True)
    for image_name, width, height, annos, note in entries:
        write_mask_json(
            json_dir / json_name_for(image_name), image_name, width, height, annos, note
        )
    return json_dir
