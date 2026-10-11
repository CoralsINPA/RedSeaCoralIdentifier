"""Scoring predictions against the expert test set.

Expert labels are normalized to a *status*:

- `in_set`: a genus of the label set (counts towards the headline accuracy);
- `out_of_set`: a genus the model cannot predict (soft corals such as Cladiella, Sinularia,
  Litophyton, Sarcophyton). Counted as wrong in `accuracy_in_plus_out`, never in `accuracy_in_set`;
- `unidentifiable`: the expert could not name the genus (`unidentifiable`, `?`);
- `missing`: no label (`MISSING_LABEL`, blank).

In the real `assaf_expert_test_set.csv` the `expert_genus` cell is empty for these last two
kinds, and the `is_correct` column holds the word (`unidentifiable` or `MISSING_LABEL`). So when
`expert_genus` is empty the status comes from `is_correct`.

`compare_predictions`, `load_legacy_predictions` and `distribution_vs_ground_truth` live in
`coralid.predictions` and are re-exported here.

`evaluate` joins predictions and expert labels on `(image_name, coral_id)` (the coral id cast to
int on both sides) and reports coverage, counts per status, the two accuracies, per-genus recall
and precision, the full confusion matrix and the top confusion pairs.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from coralid.genera import load_genera, resolve_genera
from coralid.predictions import (
    PredictionComparison,
    compare_predictions,
    distribution_vs_ground_truth,
    load_legacy_predictions,
    normalize_keys,
)

__all__ = [
    "IN_SET",
    "MISSING",
    "OUT_OF_SET",
    "STATUSES",
    "UNIDENTIFIABLE",
    "EvaluationReport",
    "PredictionComparison",
    "compare_predictions",
    "confusion_cells",
    "distribution_vs_ground_truth",
    "effective_label",
    "evaluate",
    "expert_status",
    "load_expert_set",
    "load_legacy_predictions",
    "wilson_interval",
]

logger = logging.getLogger(__name__)

IN_SET = "in_set"
OUT_OF_SET = "out_of_set"
UNIDENTIFIABLE = "unidentifiable"
MISSING = "missing"
STATUSES = (IN_SET, OUT_OF_SET, UNIDENTIFIABLE, MISSING)

_UNIDENTIFIABLE_LABELS = {"unidentifiable", "unidentified", "?", "unknown"}
_MISSING_LABELS = {"missing_label", "missing", "", "nan", "none"}


def _canonical_names(genera: Sequence[str]) -> dict[str, str]:
    """Lower-case name -> canonical spelling, for the label set and every known genus."""
    names = {g.lower(): g for g in load_genera().known_genera()}
    names.update({g.lower(): g for g in genera})
    return names


def _text(value: Any) -> str:
    """The stripped text of a cell, "" for None / NaN / pd.NA."""
    blank = value is None or (not isinstance(value, str) and bool(pd.isna(value)))
    return "" if blank else str(value).strip()


def effective_label(value: Any, is_correct: Any = None) -> str | None:
    """The label a row really carries: `expert_genus`, else the marker word in `is_correct`.

    Returns None if both are empty or `is_correct` is a plain True / False.
    """
    text = _text(value)
    if text:
        return text
    marker = _text(is_correct)
    if marker.lower() in _UNIDENTIFIABLE_LABELS | _MISSING_LABELS - {""}:
        return marker
    return None


def expert_status(
    value: Any, genera: Sequence[str], is_correct: Any = None
) -> tuple[str, str | None]:
    """Return `(status, genus)` for one `expert_genus` value (`genus` is None if none).

    If `value` is empty, `is_correct` decides: `unidentifiable` gives that status, and
    `MISSING_LABEL`, True, False or empty give `missing`.
    """
    text = effective_label(value, is_correct) or ""
    key = text.lower()
    if key in _MISSING_LABELS:
        return MISSING, None
    if key in _UNIDENTIFIABLE_LABELS:
        return UNIDENTIFIABLE, None
    genus = _canonical_names(genera).get(key, text)
    return (IN_SET if genus in set(genera) else OUT_OF_SET), genus


def load_expert_set(path: str | os.PathLike[str], label_set: str | Sequence[str] = "hard20"):
    """Read `assaf_expert_test_set.csv` and normalize the expert labels.

    Returns one row per coral: `image_name, coral_id` (int), `expert_raw` (as in the file),
    `expert_genus` (canonical spelling, None unless the status is in_set or out_of_set),
    `status`, and `review_source` when the file has it.
    """
    genera = resolve_genera(label_set)
    raw = pd.read_csv(path, dtype={"expert_genus": "string", "is_correct": "string"})
    missing = {"image_name", "coral_id", "expert_genus"} - set(raw.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
    df = normalize_keys(raw)
    if df.duplicated(["image_name", "coral_id"]).any():
        raise ValueError(f"{path}: the expert set has duplicate (image_name, coral_id) keys")
    flags = df["is_correct"] if "is_correct" in df.columns else [None] * len(df)
    labels = [effective_label(v, f) for v, f in zip(df["expert_genus"], flags, strict=True)]
    parsed = [expert_status(label, genera) for label in labels]
    from_flag = sum(
        1 for v, label in zip(df["expert_genus"], labels, strict=True) if not _text(v) and label
    )
    if from_flag:
        logger.info("%d row(s) have no expert_genus; their status comes from is_correct", from_flag)
    out = df[["image_name", "coral_id"]].copy()
    out["expert_raw"] = pd.Series(labels, dtype=object)
    out["expert_genus"] = [genus for _, genus in parsed]
    out["status"] = [status for status, _ in parsed]
    if "review_source" in df.columns:
        out["review_source"] = df["review_source"]
    return out.reset_index(drop=True)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval of a proportion `k / n` (None if `n` is 0)."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _accuracy(correct: int, total: int) -> dict[str, Any]:
    interval = wilson_interval(correct, total)
    return {
        "correct": int(correct),
        "total": int(total),
        "accuracy": correct / total if total else None,
        "ci95": list(interval) if interval else None,
    }


@dataclass
class EvaluationReport:
    """The result of `evaluate`."""

    coverage: dict[str, int]
    status_counts: dict[str, int]  # all expert rows
    status_counts_joined: dict[str, int]  # expert rows that have a prediction
    accuracy_in_set: dict[str, Any]
    accuracy_in_plus_out: dict[str, Any]
    per_genus: pd.DataFrame
    confusion: pd.DataFrame  # true (rows) x predicted (columns), identifiable rows only
    top_confusions: pd.DataFrame
    joined: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def confusion_cells(self) -> dict[tuple[str, str], int]:
        """The non-zero cells of the confusion matrix as `{(true, predicted): count}`."""
        return confusion_cells(self.confusion)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly summary (without the joined table)."""
        return {
            "coverage": self.coverage,
            "status_counts": self.status_counts,
            "status_counts_joined": self.status_counts_joined,
            "accuracy_in_set": self.accuracy_in_set,
            "accuracy_in_plus_out": self.accuracy_in_plus_out,
            "per_genus": self.per_genus.reset_index().to_dict("records"),
            "top_confusions": self.top_confusions.to_dict("records"),
            "confusion_cells": [[t, p, n] for (t, p), n in sorted(self.confusion_cells().items())],
        }


def confusion_cells(confusion: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Non-zero cells of a confusion matrix as `{(true, predicted): count}`."""
    return {
        (str(t), str(p)): int(n) for t, row in confusion.iterrows() for p, n in row.items() if n
    }


def _nan_if_zero(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.where(den > 0)).astype(float)


def evaluate(
    predictions: pd.DataFrame,
    expert: pd.DataFrame,
    label_set: str | Sequence[str] = "hard20",
    top_n: int = 10,
) -> EvaluationReport:
    """Score `predictions` (needs `image_name, coral_id, predicted_genus`) against `expert`.

    `expert` needs `image_name, coral_id` and `expert_raw` (as `load_expert_set` gives) or
    `expert_genus` (a hand-built table). Statuses are derived from the label set.
    """
    genera = resolve_genera(label_set)
    pred = normalize_keys(predictions)[["image_name", "coral_id", "predicted_genus"]]
    if pred.duplicated(["image_name", "coral_id"]).any():
        raise ValueError("predictions have duplicate (image_name, coral_id) keys")
    exp = normalize_keys(expert)
    if exp.duplicated(["image_name", "coral_id"]).any():
        raise ValueError("the expert set has duplicate (image_name, coral_id) keys")
    # a table from load_expert_set has already lost the unidentifiable / missing wording in
    # expert_genus; its expert_raw column keeps it, so statuses are derived from that
    raw_labels = exp["expert_raw"] if "expert_raw" in exp.columns else exp["expert_genus"]
    exp = exp[["image_name", "coral_id"]]
    parsed = [expert_status(v, genera) for v in raw_labels]
    exp = exp.assign(status=[s for s, _ in parsed], expert_genus=[g for _, g in parsed])

    joined = exp.merge(pred, on=["image_name", "coral_id"], how="left", indicator=True)
    has_pred = joined["_merge"] == "both"
    joined = joined.drop(columns="_merge")
    n_pred_only = len(pred) - int(has_pred.sum())
    coverage = {
        "n_expert": len(exp),
        "n_predictions": len(pred),
        "n_joined": int(has_pred.sum()),
        "n_expert_without_prediction": int((~has_pred).sum()),
        "n_predictions_without_expert": n_pred_only,
    }
    status_counts = {s: int((exp["status"] == s).sum()) for s in STATUSES}
    scored = joined[has_pred]
    status_joined = {s: int((scored["status"] == s).sum()) for s in STATUSES}

    ident = scored[scored["status"].isin([IN_SET, OUT_OF_SET])].copy()
    ident["correct"] = ident["expert_genus"] == ident["predicted_genus"]
    in_set = ident[ident["status"] == IN_SET]
    acc_in = _accuracy(int(in_set["correct"].sum()), len(in_set))
    acc_all = _accuracy(int(ident["correct"].sum()), len(ident))

    seen = dict.fromkeys([*ident["expert_genus"], *ident["predicted_genus"]])
    labels = [*genera, *[g for g in seen if g not in set(genera)]]  # label set first, then extras
    counts = ident.groupby(["expert_genus", "predicted_genus"]).size()
    confusion = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for (true, predicted), n in counts.items():
        confusion.at[true, predicted] = int(n)
    confusion.index.name, confusion.columns.name = "true", "predicted"

    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    tp = pd.Series([confusion.at[g, g] for g in labels], index=labels)
    per_genus = pd.DataFrame(
        {
            "support": support,
            "predicted": predicted,
            "tp": tp,
            "recall": _nan_if_zero(tp, support),
            "precision": _nan_if_zero(tp, predicted),
        }
    )
    per_genus.index.name = "genus"

    pairs = [
        {"true": t, "predicted": p, "count": n}
        for (t, p), n in confusion_cells(confusion).items()
        if t != p
    ]
    order = {g: i for i, g in enumerate(labels)}
    pairs.sort(key=lambda r: (-r["count"], order[r["true"]], order[r["predicted"]]))
    top = pd.DataFrame(pairs[:top_n], columns=["true", "predicted", "count"])

    return EvaluationReport(
        coverage, status_counts, status_joined, acc_in, acc_all, per_genus, confusion, top, joined
    )
