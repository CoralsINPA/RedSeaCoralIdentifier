"""The segmentation check: does the real CoralSCOP, run through `coralid`, reproduce saved masks?

`coralid.segmentation` has never run the real CoralSCOP on the real photos (its tests use a fake
generator, and one smoke test used random weights). This module checks it on a handful of photos
without touching the active run:

1. `check_config` makes a copy of the run's config whose run name carries a prefix
   (`segcheck_iou60_sta55_pts32`), so its mask JSONs go to a scratch run folder of their own;
2. `run_segmentation_check` segments `n` photos into it (the photos come from a small folder such
   as `data_50`; the ones that have a JSON in the active run are used, with at least one portrait
   photo if there is one) and compares the new JSONs with the active run's for the same photos
   with `coralid.mask_compare` (what is compared, and the PASS / WARN / FAIL rule, are described
   there);
3. `write_segmentation_check` saves the result as a new folder under `reports/`.

Nothing here writes into the active run: `check_config` refuses a config whose JSON folder is the
active one, a photo already in the scratch run is not segmented again, and the report goes to a
new folder.
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

from coralid.config import Config, ConfigError
from coralid.mask_compare import SegmentationComparison, compare_segmentations
from coralid.masks import json_name_for, load_mask_json
from coralid.orientation import PORTRAIT, orientation_of
from coralid.reports import new_report_folder
from coralid.segmentation import SegmentationResult, list_images, run_segmentation

logger = logging.getLogger(__name__)

CHECK_PREFIX = "segcheck_"


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
