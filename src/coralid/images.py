"""Image loading. Only the loaders here resize a photo.

The mask JSONs live in a frame (`image.width` x `image.height`) that is usually smaller than
the photo. Cutting crops out of the full-size photo with bboxes from the smaller frame lands
on the wrong pixels, so photos are only ever loaded through `load_in_mask_frame` (or, for the
segmentation job that creates the frame, `load_for_segmentation`). These are the only functions
that resize a photo.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


class ImageReadError(RuntimeError):
    """Raised when a photo is missing or cannot be decoded."""


def load_rgb(path: str | os.PathLike[str]) -> np.ndarray:
    """Read an image file as an RGB uint8 array of shape (H, W, 3)."""
    if not os.path.isfile(path):
        raise ImageReadError(f"Image missing: {path}")
    bgr = cv2.imread(os.fspath(path))
    if bgr is None:
        raise ImageReadError(f"Image unreadable: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def frame_for(width: int, height: int, max_side: int = 2560) -> tuple[int, int]:
    """The mask frame `(width, height)` of a photo of this size (CM cell 8).

    A photo whose long side exceeds `max_side` is scaled so that the long side is `max_side`,
    with `int()` truncation of both sides (4:3 gives 2560x1920, 3:2 gives 2560x1706, a portrait
    4:3 photo gives 1920x2560); a smaller photo keeps its size.
    """
    longest = max(width, height)
    if longest <= max_side:
        return (width, height)
    scale = max_side / longest
    return (int(width * scale), int(height * scale))


def _to_frame(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize to the frame with `cv2.resize` and its default interpolation, as the legacy code."""
    if img.shape[1] != width or img.shape[0] != height:
        img = cv2.resize(img, (width, height))
    return img


def load_in_mask_frame(path: str | os.PathLike[str], width: int, height: int) -> np.ndarray:
    """Read a photo as RGB and resize it to the mask frame `width` x `height`.

    Uses `cv2.resize` with its default interpolation, like the legacy
    `cv2.resize(image, (2560, 1920))`. A photo already in the frame is returned as read.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid frame size {width}x{height}")
    return _to_frame(load_rgb(path), width, height)


def load_for_segmentation(
    path: str | os.PathLike[str], max_side: int = 2560
) -> tuple[np.ndarray, tuple[int, int]]:
    """Read a photo for segmentation: `(rgb, (width, height))` with the image already in its frame.

    The frame is `frame_for` of the photo's own size, so the mask JSON written from this image
    and `load_in_mask_frame(path, *frame)` later see the same pixels.
    """
    img = load_rgb(path)
    width, height = frame_for(img.shape[1], img.shape[0], max_side)
    return _to_frame(img, width, height), (width, height)
