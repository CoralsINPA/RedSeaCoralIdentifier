"""Auditing one genus on the new baseline: the old audit, and a blind, mixed review sheet.

Built for the Millepora re-analysis, but nothing in it is specific to Millepora.

**The old audit** (`v2_pipeline/millepora_sharp_audit.csv`, 129 rows) was judged yes / no / ? by
a non-expert analyst who saw the model's answer, on V2's 24-genus predictions and on crops cut by
the legacy code. `load_legacy_audit` reads it; `audit_agreement` joins it to the new predictions
and reports a **non-expert agreement** rate (not a precision) as a range, with the unsure counted
as wrong and as right. Rows on portrait photos are counted apart: their crops were mis-cut, so
their judgments are not comparable and are never reused.

**The blind sheet** (`build_blind_review`) is what the next review is made of:

- a random sample of the crops the new baseline calls the genus, across all confidence levels;
- about as many random crops it did *not* call the genus;
- plus the old audit's unsure crops and its portrait-photo rows, which are re-reviewed;
- crops already judged on a landscape photo are left out (their judgment is reused as it is);
- shuffled, with no prediction, confidence or runner-up shown: the sheet has only `key`,
  `is_<genus>`, `actual_genus` (required when the answer is no) and `notes`.

About 40 of the crops, including every one the old audit marked unsure, go on a second sheet for
the expert, to measure how often the analyst agrees with him. The hidden answer key records why
each crop was chosen and what the model said; `coralid.review_scoring` scores the answers.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from coralid.labeling import ACTUAL_GENUS, NO, NOTES, UNSURE, YES, answer_column, ingest_review
from coralid.orientation import LANDSCAPE, PORTRAIT, add_orientation
from coralid.predictions import normalize_keys

logger = logging.getLogger(__name__)

LEGACY_ANSWER = "is_true_millepora"
LABEL = "non-expert agreement (not a precision)"

# why a crop is on the sheet (the `selection` column of the answer key)
PRED_RANDOM = "pred_random"  # a random crop the model called the genus
OTHER_RANDOM = "other_random"  # a random crop the model called something else
FORCED_UNSURE = "forced_unsure"  # the old audit was unsure about it
FORCED_PORTRAIT = "forced_portrait"  # the old audit judged it on a mis-cut portrait crop
PRIOR_DECIDED = "prior_decided"  # judged yes / no before, on a landscape photo: not on the sheet
SELECTIONS = (PRED_RANDOM, OTHER_RANDOM, FORCED_UNSURE, FORCED_PORTRAIT, PRIOR_DECIDED)


def load_legacy_audit(path: str | os.PathLike[str], answer: str = LEGACY_ANSWER) -> pd.DataFrame:
    """The old audit sheet as a table: `key`, `answer` (yes / no / unsure / None), `actual_genus`...

    Other columns (`confidence`, `runner_up`, `sharpness`, `area`, `bbox`) are kept; the numeric
    ones are converted.
    """
    review = ingest_review(path, answer)
    if len(review.problems):
        kinds = review.problems["problem"].value_counts().to_dict()
        logger.warning(
            "%s: %d problem(s) in the old audit sheet: %s", path, len(review.problems), kinds
        )
    table = review.table
    for col in ("confidence", "sharpness", "area"):
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")
    return table


# --------------------------------------------------------------------------- old audit vs new


def _tally(part: pd.DataFrame) -> dict[str, Any]:
    """Counts and agreement of the old judgments with the new predictions, for some audit rows."""
    overlap = part[part["in_baseline"]]
    judged = overlap[overlap["answer"].isin([YES, NO, UNSURE])]
    grid = {
        a: {
            "genus": int(((judged["answer"] == a) & judged["new_is_target"]).sum()),
            "other": int(((judged["answer"] == a) & ~judged["new_is_target"].astype(bool)).sum()),
        }
        for a in (YES, NO, UNSURE)
    }
    agree = grid[YES]["genus"] + grid[NO]["other"]
    unsure = grid[UNSURE]["genus"] + grid[UNSURE]["other"]
    n = len(judged)
    calls = judged[judged["new_is_target"].astype(bool)]
    n_calls = len(calls)
    calls_yes = int((calls["answer"] == YES).sum())
    calls_unsure = int((calls["answer"] == UNSURE).sum())
    return {
        "audit_rows": len(part),
        "in_baseline": len(overlap),
        "not_in_baseline": len(part) - len(overlap),
        "judged": n,
        "pending": int((overlap["answer"].isna()).sum()),
        "analyst_answer_by_new_call": grid,
        "agreement": {
            "n": n,
            "unsure_as_wrong": agree / n if n else None,
            "unsure_as_right": (agree + unsure) / n if n else None,
        },
        "new_genus_calls": {
            "n": n_calls,
            "analyst_yes": calls_yes,
            "analyst_unsure": calls_unsure,
            "analyst_no": n_calls - calls_yes - calls_unsure,
            "share_yes_unsure_wrong": calls_yes / n_calls if n_calls else None,
            "share_yes_unsure_right": (calls_yes + calls_unsure) / n_calls if n_calls else None,
        },
    }


def audit_agreement(
    audit: pd.DataFrame,
    new_predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    genus: str = "Millepora",
) -> dict[str, Any]:
    """The old analyst judgments against the new predictions, by photo orientation.

    For every audit row with a new prediction: the analyst said yes and the baseline calls the
    genus, or the analyst said no and it does not, counts as agreement. The unsure are counted as
    wrong (`unsure_as_wrong`) and as right (`unsure_as_right`). `new_genus_calls` is the share of
    the baseline's genus calls that the analyst said yes to. This is **non-expert agreement**, not
    precision. `landscape` is the comparison to use; `portrait` is listed apart because the old
    crops there were mis-cut.
    """
    new = normalize_keys(new_predictions)[["image_name", "coral_id", "predicted_genus"]]
    rows = normalize_keys(audit).merge(new, on=["image_name", "coral_id"], how="left")
    rows = add_orientation(rows, manifest)
    rows["in_baseline"] = rows["predicted_genus"].notna()
    rows["new_is_target"] = (rows["predicted_genus"] == genus) & rows["in_baseline"]
    landscape = _tally(rows[rows["orientation"] == LANDSCAPE])
    portrait = _tally(rows[rows["orientation"] == PORTRAIT])
    portrait["comparable"] = False
    portrait["why"] = "the old crops on portrait photos were cut from the wrong region"
    return {"label": LABEL, "genus": genus, "landscape": landscape, "portrait": portrait}


# --------------------------------------------------------------------------- the blind sheet


@dataclass
class BlindReview:
    """A blind review: the sheets to hand out and the key to keep hidden."""

    sheet: pd.DataFrame  # for the analyst: key, is_<genus>, actual_genus, notes
    assaf_sheet: pd.DataFrame  # the subset for the expert, in its own order
    key: pd.DataFrame  # hidden: why each crop is there, what the model said
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def answer(self) -> str:
        """The name of the answer column."""
        return answer_column(self.meta["genus"])


def _draw(rng: np.random.Generator, pool: pd.DataFrame, n: int) -> pd.DataFrame:
    if n >= len(pool):
        return pool
    return pool.iloc[np.sort(rng.choice(len(pool), size=n, replace=False))]


def _forced(rows: pd.DataFrame) -> pd.Series:
    """Audit rows that must be reviewed again, with the reason."""
    reason = pd.Series(None, index=rows.index, dtype=object)
    reason[rows["answer"] == UNSURE] = FORCED_UNSURE
    reason[rows["orientation"] == PORTRAIT] = FORCED_PORTRAIT  # a portrait row wins
    return reason


def build_blind_review(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    genus: str = "Millepora",
    audit: pd.DataFrame | None = None,
    n_pred: int = 150,
    n_other: int = 150,
    n_assaf: int = 40,
    seed: int = 0,
) -> BlindReview:
    """Sample a blind, mixed review of `genus` from the baseline `predictions`.

    `audit` is the old audit (`load_legacy_audit`); without it nothing is excluded or forced.
    `n_pred` crops are drawn at random from those the baseline calls `genus` and `n_other` from
    the rest, neither from crops already judged on a landscape photo nor from the forced ones.
    Everything is shuffled. `n_assaf` is a target: the forced unsure crops are always included
    and the rest is drawn from the sheet. The same `seed` gives the same sheets.
    """
    rng = np.random.default_rng(seed)
    pred = normalize_keys(predictions)
    pred["key"] = pred["image_name"].astype(str) + "#" + pred["coral_id"].astype(str)
    pred = add_orientation(pred, manifest)
    pred["is_target"] = pred["predicted_genus"] == genus

    prior = pd.DataFrame(columns=["key", "answer"])
    forced = pd.DataFrame(columns=["key", "selection", "prior_answer"])
    not_in_baseline = 0
    if audit is not None:
        old = add_orientation(normalize_keys(audit), manifest)
        old["selection"] = _forced(old)
        in_baseline = old["key"].isin(pred["key"])
        not_in_baseline = int((~in_baseline).sum())
        old = old[in_baseline]
        decided = (old["orientation"] == LANDSCAPE) & old["answer"].isin([YES, NO])
        prior = old[decided][["key", "answer"]]
        forced = old[old["selection"].notna()][["key", "selection", "answer"]].rename(
            columns={"answer": "prior_answer"}
        )

    taken = set(prior["key"]) | set(forced["key"])
    free = pred[~pred["key"].isin(taken)]
    pool_pred, pool_other = free[free["is_target"]], free[~free["is_target"]]
    drawn = [
        _draw(rng, pool_pred, n_pred).assign(selection=PRED_RANDOM),
        _draw(rng, pool_other, n_other).assign(selection=OTHER_RANDOM),
        pred[pred["key"].isin(forced["key"])].merge(forced, on="key"),
    ]
    items = pd.concat(drawn, ignore_index=True)
    items = items.iloc[rng.permutation(len(items))].reset_index(drop=True)
    items["sheet_order"] = np.arange(1, len(items) + 1)

    must = items["selection"].isin([FORCED_UNSURE]) | (items["prior_answer"] == UNSURE)
    rest = items[~must]
    want = max(0, n_assaf - int(must.sum()))
    extra = _draw(rng, rest, want)["key"] if want else pd.Series([], dtype=object)
    items["for_assaf"] = must | items["key"].isin(extra)

    sheet = _sheet(items["key"], genus)
    assaf_keys = items.loc[items["for_assaf"], "key"]
    assaf_keys = assaf_keys.iloc[rng.permutation(len(assaf_keys))]
    keep = ["key", "image_name", "coral_id", "orientation", "selection", "predicted_genus"]
    keep += [c for c in ("confidence", "runner_up") if c in items.columns]
    keep += ["prior_answer", "for_assaf", "sheet_order"]
    key = items[keep].rename(columns={"predicted_genus": "new_pred"})
    done = prior.merge(
        pred[["key", "image_name", "coral_id", "orientation", "predicted_genus"]], on="key"
    )
    done = done.rename(columns={"answer": "prior_answer", "predicted_genus": "new_pred"})
    done = done.assign(selection=PRIOR_DECIDED, for_assaf=False, sheet_order=np.nan)
    key = pd.concat([key, done], ignore_index=True)
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
        "n_prior_decided": len(prior),
        "audit_rows_not_in_baseline": not_in_baseline,
    }
    return BlindReview(sheet, _sheet(assaf_keys, genus), key, meta)


def _sheet(keys: pd.Series, genus: str) -> pd.DataFrame:
    """The blank sheet for some keys: nothing but the key and the empty answer columns."""
    sheet = pd.DataFrame({"key": keys.to_numpy()})
    for col in (answer_column(genus), ACTUAL_GENUS, NOTES):
        sheet[col] = ""
    return sheet
