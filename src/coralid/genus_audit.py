"""The blind, mixed review sheet of one genus, drawn from the 20-genus baseline.

Built for the Millepora check, but nothing in it is specific to Millepora. There is no valid
Millepora precision figure yet: this sheet is how it is made. It contains

- a random sample of the crops the baseline calls the genus, across all confidence levels;
- about as many random crops it did *not* call the genus;
- shuffled, with no prediction, confidence or runner-up shown: the sheet has only `key`,
  `is_<genus>`, `actual_genus` (required when the answer is no) and `notes`.

About 40 of the crops go on a second sheet for the expert, to measure how often the analyst
agrees with him. The hidden answer key records why each crop was chosen, what the baseline said
and which predictions file it came from (`coralid.review_source`); `coralid.review_scoring`
scores the answers.

The crops are drawn only from the predictions given, which must be the saved 20-genus baseline
(`coralid.review_source.baseline_source` checks that), and every sampled crop must be a kept row
of the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from coralid.labeling import ACTUAL_GENUS, NOTES, answer_column
from coralid.orientation import add_orientation
from coralid.predictions import normalize_keys
from coralid.review_source import PredictionSource, ReviewSourceError

# why a crop is on the sheet (the `selection` column of the answer key)
PRED_RANDOM = "pred_random"  # a random crop the baseline called the genus
OTHER_RANDOM = "other_random"  # a random crop the baseline called something else
SELECTIONS = (PRED_RANDOM, OTHER_RANDOM)


@dataclass
class BlindReview:
    """A blind review: the sheets to hand out and the key to keep hidden."""

    sheet: pd.DataFrame  # for the analyst: key, is_<genus>, actual_genus, notes
    assaf_sheet: pd.DataFrame  # the subset for the expert, in its own order
    key: pd.DataFrame  # hidden: why each crop is there, what the baseline said
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def answer(self) -> str:
        """The name of the answer column."""
        return answer_column(self.meta["genus"])


def _draw(rng: np.random.Generator, pool: pd.DataFrame, n: int) -> pd.DataFrame:
    if n >= len(pool):
        return pool
    return pool.iloc[np.sort(rng.choice(len(pool), size=n, replace=False))]


def build_blind_review(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    genus: str = "Millepora",
    n_pred: int = 150,
    n_other: int = 150,
    n_assaf: int = 40,
    seed: int = 0,
    source: PredictionSource | None = None,
) -> BlindReview:
    """Sample a blind, mixed review of `genus` from the baseline `predictions`.

    `n_pred` crops are drawn at random from those the baseline calls `genus` and `n_other` from
    the rest, then everything is shuffled. `n_assaf` crops of the sheet go on the expert's.
    The same `seed` gives the same sheets. `source` (from `baseline_source`) is recorded in the
    review's meta, and so in the answer key's meta file. Raises `ReviewSourceError` if a sampled
    crop is not a kept row of `manifest`, or if the predictions are not the `source` file's.
    """
    rng = np.random.default_rng(seed)
    pred = normalize_keys(predictions)
    pred["key"] = pred["image_name"].astype(str) + "#" + pred["coral_id"].astype(str)
    if source is not None and source.n_rows is not None and source.n_rows != len(pred):
        raise ReviewSourceError(
            f"The predictions have {len(pred)} rows but {source.file} has {source.n_rows}: "
            "they are not that file."
        )
    pred = add_orientation(pred, manifest)
    pred["is_target"] = pred["predicted_genus"] == genus
    pool_pred, pool_other = pred[pred["is_target"]], pred[~pred["is_target"]]

    items = pd.concat(
        [
            _draw(rng, pool_pred, n_pred).assign(selection=PRED_RANDOM),
            _draw(rng, pool_other, n_other).assign(selection=OTHER_RANDOM),
        ],
        ignore_index=True,
    )
    items = items.iloc[rng.permutation(len(items))].reset_index(drop=True)
    items["sheet_order"] = np.arange(1, len(items) + 1)
    _require_kept(items["key"], manifest)

    extra = _draw(rng, items, n_assaf)["key"] if n_assaf else pd.Series([], dtype=object)
    items["for_assaf"] = items["key"].isin(extra)

    sheet = _sheet(items["key"], genus)
    assaf_keys = items.loc[items["for_assaf"], "key"]
    assaf_keys = assaf_keys.iloc[rng.permutation(len(assaf_keys))]
    keep = ["key", "image_name", "coral_id", "orientation", "selection", "predicted_genus"]
    keep += [c for c in ("confidence", "runner_up") if c in items.columns]
    keep += ["for_assaf", "sheet_order"]
    key = items[keep].rename(columns={"predicted_genus": "baseline_pred"})
    meta = {
        "genus": genus,
        "seed": seed,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_predictions": len(pred),
        "n_pool_pred": len(pool_pred),
        "n_pool_other": len(pool_other),
        "n_requested": {"pred": n_pred, "other": n_other, "assaf": n_assaf},
        "n_on_sheet": len(items),
        "n_by_selection": items["selection"].value_counts().to_dict(),
        "n_for_assaf": int(items["for_assaf"].sum()),
        "source": source.to_dict() if source is not None else None,
    }
    return BlindReview(sheet, _sheet(assaf_keys, genus), key, meta)


def _require_kept(keys: pd.Series, manifest: pd.DataFrame) -> None:
    """Raise unless every sampled key is a `keep == True` row of the manifest."""
    if "keep" not in manifest.columns:
        raise ReviewSourceError("The manifest has no 'keep' column, so kept rows cannot be told.")
    kept = manifest[manifest["keep"].astype(bool)]
    allowed = set(kept["image_name"].astype(str) + "#" + kept["coral_id"].astype(int).astype(str))
    stray = sorted(set(keys) - allowed)
    if stray:
        raise ReviewSourceError(
            f"{len(stray)} sampled crop(s) are not kept rows of the manifest (for example "
            f"{stray[:3]}): the predictions do not belong to this manifest."
        )


def _sheet(keys: pd.Series, genus: str) -> pd.DataFrame:
    """The blank sheet for some keys: nothing but the key and the empty answer columns."""
    sheet = pd.DataFrame({"key": keys.to_numpy()})
    for col in (answer_column(genus), ACTUAL_GENUS, NOTES):
        sheet[col] = ""
    return sheet
