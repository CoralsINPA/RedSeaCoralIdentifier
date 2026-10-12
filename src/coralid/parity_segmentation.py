"""The segmentation check: does the real CoralSCOP, run through `coralid`, reproduce saved masks?

`coralid.segmentation` has never run the real CoralSCOP on the real photos (its tests use a fake
generator, and one smoke test used random weights). This module checks it on a handful of photos
without touching the active run:

1. `check_config` makes a copy of the run's config whose run name carries a prefix
   (`segcheck_iou60_sta55_pts32`), so its mask JSONs go to a scratch run folder of their own;
2. `run_segmentation_check` segments `n` photos into it (the photos come from a small folder such
   as `data_50`; the ones that have a JSON in the active run are used, with at least one portrait
   photo if there is one) and compares the new JSONs with the active run's for the same photos;
3. `compare_segmentations` is that comparison, on files only (no model), so it is tested with
   synthetic masks.

**What is compared.** For each photo: the frame, the number of masks, and the masks themselves.
Masks are matched one to one by IoU (greedy, highest IoU first); a saved mask is "found again" if
a new mask has IoU >= 0.9 with it. SAM on another GPU or library version need not give the same
pixels, so no equality is demanded. Small masks near the `pred_iou` and `stability` thresholds
may flip, so the numbers are given twice: for all masks and for the large ones (area >= the
run's `min_area`, 25,000 px), which are the ones that survive the filters and matter.

**The verdict** (`verdict_for`; the limits are guesses to be revised after the first real run):

- FAIL: a frame differs; a photo could not be segmented; fewer than half of the large saved
  masks are found again; or the new run has fewer than half or more than twice as many masks.
  These mean the code or the weights differ, not floating-point noise.
- WARN: fewer than 90% of the large masks are found again, fewer than 80% of all masks, or the
  mask count differs by more than a quarter. Look at the per-photo table and an overlay.
- PASS: otherwise.

Nothing here writes into the active run: `check_config` refuses a config whose JSON folder is the
active one, and `write_segmentation_check` creates a new report folder.
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util

from coralid.config import Config, ConfigError
from coralid.masks import MaskFile, json_name_for, load_mask_json, rle_to_bytes
from coralid.orientation import PORTRAIT, orientation_of
from coralid.reports import new_report_folder
from coralid.segmentation import (
    SegmentationResult,
    list_images,
    run_segmentation,
)

logger = logging.getLogger(__name__)

CHECK_PREFIX = "segcheck_"
IOU_MATCH = 0.9  # a saved mask is "found again" at this IoU or more

# The verdict's limits (see the module docstring).
LARGE_SHARE_WARN, LARGE_SHARE_FAIL = 0.90, 0.50
ALL_SHARE_WARN = 0.80
COUNT_RATIO_WARN = (0.80, 1.25)
COUNT_RATIO_FAIL = (0.50, 2.00)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
COMPARED = "compared"
FRAME_DIFFERS = "frame differs"
NOT_SEGMENTED = "not segmented"
NO_SAVED_JSON = "no saved JSON"


# --------------------------------------------------------------------------- the config


def check_config(cfg: Config, prefix: str = CHECK_PREFIX, images_dir: str | None = None) -> Config:
    """A copy of `cfg` for a scratch run: the same settings, a prefixed run name, other photos.

    `images_dir` (relative to the data root, such as `data_50`) replaces the photo folder.
    Raises `ConfigError` if the copy would write to the same JSON folder as `cfg`.
    """
    scratch = Config.from_dict(cfg.to_dict())
    scratch.paths.run_prefix = prefix
    if images_dir is not None:
        scratch.paths.images_dir = images_dir
    if scratch.json_dir.resolve() == cfg.json_dir.resolve():
        raise ConfigError(
            f"The check would write into the active run's JSON folder {cfg.json_dir}. "
            "Give it a non-empty prefix."
        )
    return scratch


# --------------------------------------------------------------------------- the photos


def _frame_of(json_dir: Path, image_name: str) -> tuple[int, int]:
    return load_mask_json(json_dir / json_name_for(image_name)).frame


def choose_check_photos(
    candidates: Sequence[str],
    active_json_dir: str | os.PathLike[str],
    n: int = 5,
    seed: int = 0,
) -> list[str]:
    """`n` of the `candidates` that have a JSON in the active run, shuffled by `seed`.

    A portrait photo (a different frame, so a different code path) is included when the
    candidates have one among the first `max(4n, 20)` examined; the choice is reproducible.
    """
    root = Path(active_json_dir)
    have = [name for name in candidates if (root / json_name_for(name)).is_file()]
    random.Random(seed).shuffle(have)
    examined = have[: max(4 * n, 20)]
    chosen = examined[:n]
    portrait = [name for name in examined if orientation_of(*_frame_of(root, name)) == PORTRAIT]
    if portrait and not set(portrait) & set(chosen) and chosen:
        chosen[-1] = portrait[0]  # swap the last pick for a portrait photo
    return sorted(chosen)


# --------------------------------------------------------------------------- one photo


def _large(annotations: Sequence[dict[str, Any]], min_area: float) -> list[bool]:
    return [float(a["area"]) >= min_area for a in annotations]


def iou_matrix(active: MaskFile, new: MaskFile) -> np.ndarray:
    """IoU of every saved mask (rows) with every new mask (columns); the frames must be equal."""
    if not active.annotations or not new.annotations:
        return np.zeros((len(active.annotations), len(new.annotations)))
    saved = [rle_to_bytes(a["segmentation"]) for a in active.annotations]
    fresh = [rle_to_bytes(a["segmentation"]) for a in new.annotations]
    matrix = mask_util.iou(saved, fresh, [False] * len(fresh))
    return np.asarray(matrix, dtype=float)


def match_masks(iou: np.ndarray, threshold: float = IOU_MATCH) -> list[tuple[int, int, float]]:
    """One-to-one matches `(saved index, new index, IoU)` at `threshold` or more, best first."""
    if iou.size == 0:
        return []
    rows, cols = np.unravel_index(np.argsort(-iou, axis=None, kind="stable"), iou.shape)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    pairs = []
    for r, c in zip(rows, cols, strict=True):
        value = float(iou[r, c])
        if value < threshold:
            break
        if int(r) in used_rows or int(c) in used_cols:
            continue
        used_rows.add(int(r))
        used_cols.add(int(c))
        pairs.append((int(r), int(c), value))
    return pairs


def compare_masks(
    active: MaskFile, new: MaskFile, min_area: float = 25000, threshold: float = IOU_MATCH
) -> dict[str, Any]:
    """The comparison row of one photo (`status` is `compared` or `frame differs`)."""
    row: dict[str, Any] = {
        "orientation": orientation_of(*active.frame),
        "frame_saved": f"{active.width}x{active.height}",
        "frame_new": f"{new.width}x{new.height}",
        "n_saved": len(active.annotations),
        "n_new": len(new.annotations),
    }
    if active.frame != new.frame:
        return {**row, "status": FRAME_DIFFERS}
    large = np.array(_large(active.annotations, min_area), dtype=bool)
    iou = iou_matrix(active, new)
    pairs = match_masks(iou, threshold)
    found = np.zeros(len(active.annotations), dtype=bool)
    for r, _, _ in pairs:
        found[r] = True
    best = iou.max(axis=1) if iou.size else np.zeros(len(active.annotations))
    n_large = int(large.sum())
    return {
        **row,
        "status": COMPARED,
        "n_saved_large": n_large,
        "n_new_large": int(sum(_large(new.annotations, min_area))),
        "found": int(found.sum()),
        "found_large": int((found & large).sum()),
        "share_found": float(found.mean()) if len(found) else None,
        "share_found_large": float((found & large).sum() / n_large) if n_large else None,
        "mean_best_iou": float(best.mean()) if len(best) else None,
        "mean_best_iou_large": float(best[large].mean()) if n_large else None,
    }


# --------------------------------------------------------------------------- all photos


@dataclass
class SegmentationComparison:
    """The saved masks against the new ones, per photo and in total, with a verdict."""

    photos: pd.DataFrame
    summary: dict[str, Any]
    verdict: str
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly summary (the per-photo table is saved as CSV)."""
        return {"verdict": self.verdict, "reasons": self.reasons, **self.summary}

    def report_lines(self, min_area: float = 25000) -> list[str]:
        """The verdict, its reasons and the totals as lines of text (None-safe, for a notebook)."""
        s = self.summary

        def pct(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.1%}"

        ratio = "n/a" if s["count_ratio"] is None else f"x{s['count_ratio']:.2f}"
        iou = s["mean_best_iou"]
        lines = [
            f"{self.verdict}: {s['n_compared']} of {s['n_photos']} photos compared "
            f"({s['status_counts']})",
            f"masks: {s.get('n_saved', 0)} saved, {s.get('n_new', 0)} new ({ratio})",
            f"found again (IoU >= {IOU_MATCH}): all masks {pct(s['share_found'])}, "
            f"large masks (area >= {min_area:,.0f}) {pct(s['share_found_large'])}",
            "mean best IoU of a saved mask: " + ("n/a" if iou is None else f"{iou:.3f}"),
        ]
        return lines + [f"  - {reason}" for reason in self.reasons]


def _ratio(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def summarize(photos: pd.DataFrame) -> dict[str, Any]:
    """Totals over the compared photos (frames equal); shares are of masks, not of photos."""
    done = photos[photos["status"] == COMPARED]
    totals = {
        key: int(done[key].sum())
        for key in ("n_saved", "n_new", "n_saved_large", "n_new_large", "found", "found_large")
        if key in done
    }
    get = totals.get
    return {
        "n_photos": len(photos),
        "n_compared": len(done),
        "status_counts": {str(k): int(v) for k, v in photos["status"].value_counts().items()},
        **totals,
        "count_ratio": _ratio(get("n_new", 0), get("n_saved", 0)),
        "share_found": _ratio(get("found", 0), get("n_saved", 0)),
        "share_found_large": _ratio(get("found_large", 0), get("n_saved_large", 0)),
        "mean_best_iou": _mean_weighted(done, "mean_best_iou", "n_saved"),
        "mean_best_iou_large": _mean_weighted(done, "mean_best_iou_large", "n_saved_large"),
    }


def _mean_weighted(done: pd.DataFrame, value: str, weight: str) -> float | None:
    """Mean over masks (not over photos) of a per-photo mean, weighted by the mask count."""
    if value not in done:
        return None
    usable = done[done[value].notna() & (done[weight] > 0)]
    total = usable[weight].sum()
    return float((usable[value] * usable[weight]).sum() / total) if total else None


def verdict_for(summary: dict[str, Any]) -> tuple[str, list[str]]:
    """PASS, WARN or FAIL for a `summarize` result, with the reasons (see the module docstring)."""
    fail: list[str] = []
    warn: list[str] = []
    counts = summary["status_counts"]
    if summary["n_compared"] == 0:
        fail.append("no photo could be compared")
    for status in (FRAME_DIFFERS, NOT_SEGMENTED, NO_SAVED_JSON):
        if counts.get(status):
            fail.append(f"{counts[status]} photo(s): {status}")
    large, ratio = summary["share_found_large"], summary["count_ratio"]
    everything = summary["share_found"]
    if large is not None:
        text = f"{large:.0%} of the large saved masks are found again (IoU >= {IOU_MATCH})"
        if large < LARGE_SHARE_FAIL:
            fail.append(text)
        elif large < LARGE_SHARE_WARN:
            warn.append(text)
    if everything is not None and everything < ALL_SHARE_WARN:
        warn.append(f"{everything:.0%} of all saved masks are found again")
    if ratio is not None:
        text = f"the new run has {ratio:.2f} times as many masks"
        if not COUNT_RATIO_FAIL[0] <= ratio <= COUNT_RATIO_FAIL[1]:
            fail.append(text)
        elif not COUNT_RATIO_WARN[0] <= ratio <= COUNT_RATIO_WARN[1]:
            warn.append(text)
    if fail:
        return FAIL, fail + warn
    return (WARN, warn) if warn else (PASS, [])


def compare_segmentations(
    saved_dir: str | os.PathLike[str],
    new_dir: str | os.PathLike[str],
    image_names: Sequence[str],
    min_area: float = 25000,
    threshold: float = IOU_MATCH,
) -> SegmentationComparison:
    """Compare the mask JSONs of `image_names` in two folders (files only, no model)."""
    saved_dir, new_dir = Path(saved_dir), Path(new_dir)
    rows = []
    for name in image_names:
        saved_file, new_file = saved_dir / json_name_for(name), new_dir / json_name_for(name)
        if not saved_file.is_file():
            rows.append({"image_name": name, "status": NO_SAVED_JSON})
        elif not new_file.is_file():
            rows.append({"image_name": name, "status": NOT_SEGMENTED})
        else:
            row = compare_masks(
                load_mask_json(saved_file), load_mask_json(new_file), min_area, threshold
            )
            rows.append({"image_name": name, **row})
    photos = pd.DataFrame(rows, columns=["image_name", "status"] if not rows else None)
    summary = summarize(photos)
    verdict, reasons = verdict_for(summary)
    return SegmentationComparison(photos, summary, verdict, reasons)


# --------------------------------------------------------------------------- the whole check


@dataclass
class SegmentationCheck:
    """What `run_segmentation_check` did and found."""

    comparison: SegmentationComparison
    segmentation: SegmentationResult
    config: Config  # the scratch run's config
    image_names: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """PASS, WARN or FAIL."""
        return self.comparison.verdict


def run_segmentation_check(
    cfg: Config,
    n: int = 5,
    *,
    photos_dir: str | None = "data_50",
    prefix: str = CHECK_PREFIX,
    image_names: Sequence[str] | None = None,
    seed: int = 0,
    generator: Any = None,
    device: str | None = None,
    progress: bool = False,
) -> SegmentationCheck:
    """Segment `n` photos into a scratch run and compare them with the active run's masks.

    `photos_dir` is the photo folder to draw from (default `data_50`; `None` uses the run's own);
    `image_names` fixes the photos instead. Photos with no JSON in the active run are not used.
    The scratch run is `results_all/<prefix><run name>/`; a photo already segmented there is
    not segmented again (nothing is overwritten), so changing `prefix` redoes the check.
    `generator` is for tests; by default CoralSCOP is loaded (a GPU runtime).
    """
    scratch = check_config(cfg, prefix, photos_dir)
    candidates = list(image_names) if image_names is not None else list_images(scratch.images_dir)
    names = choose_check_photos(candidates, cfg.json_dir, n, seed)
    if not names:
        raise FileNotFoundError(
            f"None of the {len(candidates)} photos in {scratch.images_dir} has a mask JSON in "
            f"{cfg.json_dir}, so there is nothing to compare with. Try another photo folder "
            "(for example data_all)."
        )
    logger.info("Segmentation check on %d photo(s): %s", len(names), names)
    result = run_segmentation(
        scratch, generator=generator, image_names=names, device=device, progress=progress
    )
    comparison = compare_segmentations(
        cfg.json_dir, scratch.json_dir, names, min_area=cfg.filters.min_area
    )
    if result.failures:
        shown = "; ".join(f"{f['image']}: {f['error']}" for f in result.failures[:3])
        comparison.reasons.append(f"segmentation errors: {shown}")
    return SegmentationCheck(comparison, result, scratch, names)


def write_segmentation_check(
    check: SegmentationCheck, reports_dir: str | os.PathLike[str], date: str | None = None
) -> Path:
    """Save the check as new files in `<reports_dir>/segcheck_<date>/`; return that folder.

    `photos.csv` (one row per photo) and `summary.json` (the verdict, its reasons, the totals and
    the photos). An existing folder gets a `_2`, `_3`... suffix; nothing is overwritten.
    """
    folder = new_report_folder(reports_dir, "segcheck", date=date)
    check.comparison.photos.to_csv(folder / "photos.csv", index=False)
    body = {
        **check.comparison.to_dict(),
        "scratch_run": str(check.config.run_dir),
        "image_names": check.image_names,
    }
    with open(folder / "summary.json", "x", encoding="utf-8") as f:
        f.write(json.dumps(body, indent=2, default=str) + "\n")
    return folder
