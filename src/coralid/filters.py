"""Mask quality filters with the legacy semantics, but containment computed on the RLE.

`filter_annotations` reproduces the notebooks' filter (WC cell 8, V2 cell 6) exactly,
quirks included:

1. keep `area >= min_area`;
2. keep `aspect_ratio <= max_aspect_ratio`;
3. if one candidate or none is left, return it;
4. pairwise containment: for every ordered pair `(small=i, large=j)` with `i != j`, where
   `j` is not already removed, `large.area > small.area` and the bboxes intersect:
   `containment = |small & large| / small["area"]` and, if it reaches the threshold, the
   **larger** mask is removed. The denominator is the annotation's `area` field, and a mask
   that has itself been removed still takes part as `small`;
5. return the survivors in their original order.

The only difference from the notebooks is how `|small & large|` is computed: with
pycocotools on the RLE (`area(merge([a, b], intersect=True))`) instead of decoding two
full-frame masks for every overlapping pair. The result is the same integer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import pycocotools.mask as mask_util

from coralid.masks import Annotation, Rle, rle_to_bytes

REASON_AREA = "area"
REASON_ASPECT = "aspect"
REASON_CONTAINMENT = "containment"


@dataclass(frozen=True)
class Rejection:
    """Why a mask was removed. `contained_by` is set for `containment` only."""

    reason: str
    contained_by: int | None = None  # id of the smaller mask that caused the removal


def aspect_ratio(bbox: Sequence[float]) -> float:
    """Long side over short side of an `[x, y, w, h]` box (short side at least 1)."""
    _, _, w, h = bbox
    return max(w, h) / max(min(w, h), 1)


def bbox_intersection(b1: Sequence[float], b2: Sequence[float]) -> float:
    """Area of the intersection of two `[x, y, w, h]` boxes (0 if they do not overlap)."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return ix * iy


def rle_intersection_area(a: Rle, b: Rle) -> int:
    """Number of pixels two RLE masks share, computed without decoding them."""
    if list(a["size"]) != list(b["size"]):
        raise ValueError(f"RLE frames differ: {a['size']} vs {b['size']}")
    merged = mask_util.merge([rle_to_bytes(a), rle_to_bytes(b)], intersect=True)
    return int(mask_util.area(merged))


def filter_annotations_with_reasons(
    annotations: Sequence[Annotation],
    min_area: float = 25000,
    containment_thresh: float = 0.80,
    max_aspect_ratio: float = 3.0,
) -> tuple[list[Annotation], dict[int, Rejection | None]]:
    """Filter annotations and say why each removed one went.

    Returns `(kept, reasons)`. `kept` is exactly what `filter_annotations` returns.
    `reasons` maps every annotation id to None (kept) or a `Rejection` with reason
    `area`, `aspect` or `containment`; for `containment` it carries the id of the smaller
    mask that caused the removal (the first one, in list order, to reach the threshold).
    """
    reasons: dict[int, Rejection | None] = {}
    candidates: list[Annotation] = []
    for a in annotations:
        if a["area"] >= min_area:
            candidates.append(a)
        else:
            reasons[a["id"]] = Rejection(REASON_AREA)
    survivors: list[Annotation] = []
    for a in candidates:
        if aspect_ratio(a["bbox"]) <= max_aspect_ratio:
            survivors.append(a)
        else:
            reasons[a["id"]] = Rejection(REASON_ASPECT)
    candidates = survivors

    removed_by: dict[int, int] = {}
    if len(candidates) > 1:
        n = len(candidates)
        for i in range(n):
            for j in range(n):
                if i == j or candidates[j]["id"] in removed_by:
                    continue
                small, large = candidates[i], candidates[j]
                if large["area"] <= small["area"]:
                    continue
                if bbox_intersection(small["bbox"], large["bbox"]) == 0:
                    continue
                if small["area"] <= 0:
                    continue  # no meaningful denominator (cannot happen with min_area > 0)
                shared = rle_intersection_area(small["segmentation"], large["segmentation"])
                if shared / small["area"] >= containment_thresh:
                    removed_by[large["id"]] = small["id"]

    kept: list[Annotation] = []
    for a in candidates:
        if a["id"] in removed_by:
            reasons[a["id"]] = Rejection(REASON_CONTAINMENT, removed_by[a["id"]])
        else:
            reasons[a["id"]] = None
            kept.append(a)
    return kept, reasons


def filter_annotations(
    annotations: Sequence[Annotation],
    min_area: float = 25000,
    containment_thresh: float = 0.80,
    max_aspect_ratio: float = 3.0,
) -> list[Annotation]:
    """Filter annotations by area, aspect ratio and containment (legacy semantics)."""
    return filter_annotations_with_reasons(
        annotations, min_area, containment_thresh, max_aspect_ratio
    )[0]


def laplacian_variance(crop_rgb: np.ndarray) -> float:
    """Variance of the Laplacian of the grayscale crop: the sharpness measure.

    Which crop it is measured on is a setting (`FilterConfig.sharpness_crop`): the pipeline
    measures the masked crop on a black background (threshold 80), the Millepora audit measured
    the unmasked box (threshold 150).

    Caution: on a masked crop this also responds to the artificial edge between the coral and
    the plain background, so a flat but sharply outlined mask can pass. With correct crops the
    masked measure rejected only 117 of 16,197 candidates, which suggests the outline dominates.
    The unmasked box does not have this problem but includes the surroundings of the mask.
    """
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_sharp_enough(crop: np.ndarray, threshold: float) -> bool:
    """True if the crop's Laplacian variance is at least `threshold`.

    The threshold has no default on purpose: the notebooks disagreed (50 in the helper,
    80 in the settings). Pass `FilterConfig.sharpness_cutoff`, which follows `sharpness_crop`.
    """
    return bool(laplacian_variance(crop) >= threshold)
