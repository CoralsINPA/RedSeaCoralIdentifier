"""A linear probe on stored embeddings: V2 Phase 2, redone so that its numbers can be trusted.

A logistic regression on a features file (`uids` + `X`, from `coralid.baseline` or
`coralid.embed`), trained on labels that someone gives it. What V2 Phase 2 got wrong, and what
is different here (see `docs/algorithm_ideas.md`, idea 4):

- **Keys.** Labels join to features on `(image_name, coral_id)`, written `IMG_4033.JPG#14`,
  never on `coral_id` alone (it restarts at 0 in every photo).
- **Held-out only.** Cross-validation is grouped by photo (`GroupKFold`), because crops of one
  photo are alike: a crop is always predicted by a model that never saw its photo. The report is
  built from those held-out predictions only. There is no accuracy on the training set.
- **Class weights** (`balanced` by default), because the genera are very unevenly common.
- **Labels are an input.** This module never makes labels: no blank cell counts as "confirmed",
  and zero-shot predictions are not turned into training labels. A genus whose precision is
  unknown or low (Millepora, 58.9% on a non-expert's check) must not teach the probe its own
  mistakes. Give it expert labels.

The report reuses `coralid.evaluation`: accuracy with a Wilson interval, per-genus precision and
recall, the confusion matrix and the top confusions. scikit-learn is imported inside the
functions that need it.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from coralid.evaluation import (
    IN_SET,
    OUT_OF_SET,
    EvaluationReport,
    evaluate,
    expert_status,
    wilson_interval,
)
from coralid.genera import resolve_genera
from coralid.predictions import normalize_keys

logger = logging.getLogger(__name__)

MIN_LABELS_PER_CLASS = 5  # fewer than this and a class gets a warning in the report


class ProbeError(ValueError):
    """The probe cannot be trained as asked."""


@dataclass
class ProbeData:
    """Labeled crops: embeddings, labels and the photo each crop is from."""

    uids: list[str]
    X: np.ndarray
    y: np.ndarray  # genus names (object)
    groups: np.ndarray  # image names, the unit of cross-validation
    dropped: dict[str, int] = field(default_factory=dict)  # labels left out, by reason

    def __len__(self) -> int:
        return len(self.uids)


# --------------------------------------------------------------------------- labels and join


def load_labels(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a labels table (CSV or parquet). It needs `image_name` and `coral_id` columns."""
    table = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    return normalize_keys(table)


def join_labels(
    uids: list[str],
    X: np.ndarray,
    labels: pd.DataFrame,
    label_column: str = "genus",
    genera: str | list[str] | None = None,
) -> ProbeData:
    """Join labels to embeddings by `(image_name, coral_id)`; return only the labeled crops.

    A label counts if it names a genus. With `genera` (a label-set name or a list) the name is
    put in its canonical spelling and anything outside the set is left out; without it every
    named genus is a class. Rows left out are counted by reason in `dropped`: `blank`
    (empty, `unidentifiable`, `MISSING_LABEL`), `outside_label_set`, `no_features` (no embedding
    for that key) and `duplicate` (the same key and label twice). Two different labels for one key
    are an error. Output rows follow the order of the features file.
    """
    if label_column not in labels.columns:
        raise ProbeError(f"the labels have no '{label_column}' column (they have {list(labels)})")
    if len(uids) != len(X):
        raise ProbeError(f"{len(uids)} uids for {len(X)} feature rows")
    names = resolve_genera(genera) if genera is not None else []
    table = normalize_keys(labels)
    table["uid"] = table["image_name"].astype(str) + "#" + table["coral_id"].astype(str)
    dropped = {"blank": 0, "outside_label_set": 0, "no_features": 0, "duplicate": 0}
    kept: dict[str, str] = {}
    position = {uid: i for i, uid in enumerate(uids)}
    for uid, value in zip(table["uid"], table[label_column], strict=True):
        status, genus = expert_status(value, names)
        if status not in (IN_SET, OUT_OF_SET):
            dropped["blank"] += 1
        elif names and status == OUT_OF_SET:
            dropped["outside_label_set"] += 1
        elif uid not in position:
            dropped["no_features"] += 1
        elif uid in kept and kept[uid] != genus:
            raise ProbeError(f"{uid} has two different labels: {kept[uid]} and {genus}")
        elif uid in kept:
            dropped["duplicate"] += 1
        else:
            kept[uid] = str(genus)
    order = sorted(kept, key=position.__getitem__)
    rows = [position[uid] for uid in order]
    groups = np.array([uid.rpartition("#")[0] for uid in order], dtype=object)
    y = np.array([kept[uid] for uid in order], dtype=object)
    return ProbeData(order, np.asarray(X)[rows], y, groups, dropped)


# --------------------------------------------------------------------------- the probe


def make_probe(
    C: float = 1.0,
    class_weight: str | dict[str, float] | None = "balanced",
    standardize: bool = True,
    max_iter: int = 2000,
    seed: int = 0,
) -> Any:
    """An unfitted scikit-learn pipeline: optional standardization, then logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    steps: list[Any] = [StandardScaler()] if standardize else []
    steps.append(
        LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter, random_state=seed)
    )
    return make_pipeline(*steps)


def fit_probe(data: ProbeData, **params: Any) -> Any:
    """Fit the probe on every labeled crop, for predicting new ones. Never report its accuracy."""
    return make_probe(**params).fit(data.X, data.y)


@dataclass
class ProbeReport:
    """The held-out result of `cross_validate_probe`."""

    evaluation: EvaluationReport
    folds: list[dict[str, Any]]
    held_out: pd.DataFrame  # uid, image_name, coral_id, true, predicted, fold
    n_labeled: int
    n_images: int
    class_counts: dict[str, int]
    majority_baseline: float  # accuracy of always answering the most common class
    zero_shot: dict[str, Any] | None
    params: dict[str, Any]
    dropped: dict[str, int]
    warnings: list[str]

    @property
    def accuracy(self) -> dict[str, Any]:
        """Held-out accuracy with its Wilson 95% interval."""
        return self.evaluation.accuracy_in_set

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly summary (the tables are saved as CSV by `write_probe_report`)."""
        return {
            "kind": "linear_probe_held_out",
            "accuracy": self.accuracy,
            "majority_baseline": self.majority_baseline,
            "zero_shot_on_the_same_crops": self.zero_shot,
            "n_labeled": self.n_labeled,
            "n_images": self.n_images,
            "class_counts": self.class_counts,
            "folds": self.folds,
            "params": self.params,
            "labels_dropped": self.dropped,
            "warnings": self.warnings,
            "per_genus": self.evaluation.per_genus.reset_index().to_dict("records"),
            "top_confusions": self.evaluation.top_confusions.to_dict("records"),
        }


def _fit_predict(model: Any, X: np.ndarray, y: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    if len(set(y)) < 2:  # a fold trained on one class can only answer that class
        return np.array([y[0]] * len(X_test), dtype=object)
    return model.fit(X, y).predict(X_test)


def cross_validate_probe(
    data: ProbeData,
    n_splits: int = 5,
    zero_shot: pd.DataFrame | None = None,
    **params: Any,
) -> ProbeReport:
    """Grouped cross-validation of the probe; the report holds held-out predictions only.

    `params` override the defaults of `make_probe` (`C`, `class_weight`, `standardize`,
    `max_iter`, `seed`) and are recorded in the report. `zero_shot` (a predictions table with
    `image_name, coral_id, predicted_genus`) is scored on the same crops, so the probe has
    something to beat.
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import GroupKFold

    n_images = len(set(data.groups))
    notes: list[str] = []
    if n_images < 2:
        raise ProbeError("cross-validation needs labeled crops from at least 2 photos")
    if n_splits > n_images:
        notes.append(f"only {n_images} photos have labels, so n_splits was lowered to {n_images}")
        n_splits = n_images
    counts = pd.Series(data.y).value_counts()
    few = [f"{g} ({n})" for g, n in counts.items() if n < MIN_LABELS_PER_CLASS]
    if few:
        notes.append(f"classes with fewer than {MIN_LABELS_PER_CLASS} labels: {', '.join(few)}")

    settings = {
        "C": 1.0,
        "class_weight": "balanced",
        "standardize": True,
        "max_iter": 2000,
        "seed": 0,
        **params,
    }
    predicted = np.empty(len(data), dtype=object)
    fold_of = np.zeros(len(data), dtype=int)
    folds = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        splitter = GroupKFold(n_splits=n_splits)
        for fold, (train, test) in enumerate(splitter.split(data.X, data.y, data.groups)):
            model = make_probe(**settings)
            predicted[test] = _fit_predict(model, data.X[train], data.y[train], data.X[test])
            fold_of[test] = fold
            accuracy = float(np.mean(predicted[test] == data.y[test]))
            folds.append(
                {
                    "fold": fold,
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_test_images": len(set(data.groups[test])),
                    "accuracy": accuracy,
                }
            )
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        notes.append("the logistic regression did not converge in some folds (raise max_iter)")

    keys = pd.DataFrame(
        {
            "uid": data.uids,
            "image_name": data.groups,
            "coral_id": [int(u.rpartition("#")[2]) for u in data.uids],
        }
    )
    held_out = keys.assign(true=data.y, predicted=predicted, fold=fold_of)
    classes = sorted(set(data.y))
    evaluation = evaluate(
        held_out.rename(columns={"predicted": "predicted_genus"}),
        held_out.rename(columns={"true": "expert_genus"}),
        classes,
    )
    return ProbeReport(
        evaluation=evaluation,
        folds=folds,
        held_out=held_out,
        n_labeled=len(data),
        n_images=n_images,
        class_counts={str(g): int(n) for g, n in counts.items()},
        majority_baseline=float(counts.iloc[0] / len(data)),
        zero_shot=_zero_shot_accuracy(held_out, zero_shot),
        params={"n_splits": n_splits, **settings},
        dropped=data.dropped,
        warnings=notes,
    )


def _zero_shot_accuracy(held_out: pd.DataFrame, zero_shot: pd.DataFrame | None) -> dict | None:
    if zero_shot is None:
        return None
    pred = normalize_keys(zero_shot)[["image_name", "coral_id", "predicted_genus"]]
    joined = held_out.merge(pred, on=["image_name", "coral_id"], how="inner")
    correct = int((joined["predicted_genus"] == joined["true"]).sum())
    interval = wilson_interval(correct, len(joined))
    return {
        "correct": correct,
        "total": len(joined),
        "accuracy": correct / len(joined) if len(joined) else None,
        "ci95": list(interval) if interval else None,
    }
