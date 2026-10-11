"""Cutting corals out of a photo. All functions expect an RGB array in the mask frame."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from coralid.masks import Rle, decode_rle, decode_segmentation

BACKGROUND_VALUES = {"black": 0, "gray": 128, "white": 255}


def _background_value(background: str) -> int:
    try:
        return BACKGROUND_VALUES[background]
    except KeyError:
        raise ValueError(
            f"Unknown background {background!r}; use one of {sorted(BACKGROUND_VALUES)}"
        ) from None


def _padded_bounds(
    shape: tuple[int, ...], bbox: Sequence[float], padding_pct: float
) -> tuple[int, int, int, int]:
    """Return `(y1, y2, x1, x2)` of the padded bbox, clamped to the image."""
    x, y, w, h = map(int, bbox)
    pad_w = int(w * padding_pct)
    pad_h = int(h * padding_pct)
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(shape[1], x + w + pad_w)
    y2 = min(shape[0], y + h + pad_h)
    return y1, y2, x1, x2


def bbox_crop(
    img: np.ndarray, bbox: Sequence[float], padding_pct: float = 0.15
) -> np.ndarray | None:
    """Crop the padded bbox out of the image (no mask). Returns None for an empty crop."""
    y1, y2, x1, x2 = _padded_bounds(img.shape, bbox, padding_pct)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def masked_crop_stats(
    img: np.ndarray,
    bbox: Sequence[float],
    rle: Rle,
    padding_pct: float = 0.05,
    background: str = "black",
) -> tuple[np.ndarray | None, int]:
    """Cut out one mask and count its visible pixels.

    Returns `(crop, visible_px)`: the padded bbox crop with everything outside the mask set to
    the background value (None if the crop is empty), and the number of pixels in it that
    differ from the background in any channel. Applies no visibility cutoff.
    """
    bg = _background_value(background)
    mask = decode_rle(rle)
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    y1, y2, x1, x2 = _padded_bounds(img.shape, bbox, padding_pct)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, 0
    crop = crop.copy()
    crop[~mask[y1:y2, x1:x2]] = bg
    visible = int(np.any(crop != bg, axis=2).sum())
    return crop, visible


def masked_crop(
    img: np.ndarray,
    bbox: Sequence[float],
    rle: Rle,
    padding_pct: float = 0.05,
    background: str = "black",
    min_visible: int = 500,
) -> np.ndarray | None:
    """The pipeline's crop: the mask on a plain background, padded and clamped.

    Returns None if the crop is empty or has fewer than `min_visible` non-background pixels.
    A mask whose shape differs from the image is resized with nearest-neighbour first.
    """
    crop, visible = masked_crop_stats(img, bbox, rle, padding_pct, background)
    if crop is None or visible < min_visible:
        return None
    return crop


def masked_square_crop(
    img: np.ndarray, bbox: Sequence[float], segmentation: Any
) -> np.ndarray | None:
    """Mask the photo, crop the bbox, and pad to a black square (CM cell 9, for DINO).

    No padding around the bbox and no visibility cutoff. The segmentation may be a compressed
    RLE, an uncompressed RLE or polygons. Returns None for an empty crop.
    """
    height, width = img.shape[:2]
    mask = decode_segmentation(segmentation, height, width)
    if mask.shape != (height, width):
        raise ValueError(f"Mask shape {mask.shape} differs from image shape {(height, width)}")
    masked = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8) * 255)
    x, y, w, h = map(int, bbox)
    x = max(0, x)
    y = max(0, y)
    w = min(w, width - x)
    h = min(h, height - y)
    crop = masked[y : y + h, x : x + w]
    if crop.size == 0:
        return None
    h_c, w_c = crop.shape[:2]
    side = max(h_c, w_c)
    pad_y = (side - h_c) // 2
    pad_x = (side - w_c) // 2
    return cv2.copyMakeBorder(
        crop,
        pad_y,
        side - h_c - pad_y,
        pad_x,
        side - w_c - pad_x,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
