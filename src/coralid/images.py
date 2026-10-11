"""Image loading. `load_in_mask_frame` is the only code that resizes a photo.

The mask JSONs live in a frame (`image.width` x `image.height`) that is usually smaller than
the photo. Cutting crops out of the full-size photo with bboxes from the smaller frame lands
on the wrong pixels, so photos are only ever loaded through `load_in_mask_frame`.
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


def load_in_mask_frame(path: str | os.PathLike[str], width: int, height: int) -> np.ndarray:
    """Read a photo as RGB and resize it to the mask frame `width` x `height`.

    Uses `cv2.resize` with its default interpolation, like the legacy
    `cv2.resize(image, (2560, 1920))`. A photo already in the frame is returned as read.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid frame size {width}x{height}")
    img = load_rgb(path)
    if img.shape[1] != width or img.shape[0] != height:
        img = cv2.resize(img, (width, height))
    return img
