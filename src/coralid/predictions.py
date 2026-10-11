"""Reading, comparing and summarizing prediction tables.

These helpers are re-exported by `coralid.evaluation`.

- `load_legacy_predictions` reads the old `coral_weak_labels_*.csv` files, the V2 CSVs and the
  predictions stored in `assaf_expert_test_set.csv`.
- `compare_predictions` is the parity check: same keys, same top-1, close probabilities.
- `distribution_vs_ground_truth` sets the predicted genus mix beside the survey counts.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from coralid.genera import load_genera, resolve_genera
from coralid.masks import as_int, parse_bbox

logger = logging.getLogger(__name__)

KEY = ["image_name", "coral_id"]


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of `df` with a stripped string `image_name` and an int `coral_id`."""
    missing = set(KEY) - set(df.columns)
    if missing:
        raise ValueError(f"missing column(s) {sorted(missing)}")
    out = df.copy()
    out["image_name"] = out["image_name"].astype(str).str.strip()
    out["coral_id"] = [as_int(v) for v in out["coral_id"]]
    return out


def _uid(df: pd.DataFrame) -> pd.Series:
    return df["image_name"] + "#" + df["coral_id"].astype(str)


# --------------------------------------------------------------------------- legacy adapter


def load_legacy_predictions(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a legacy predictions CSV into the canonical table.

    Works for `coral_weak_labels_*.csv`, the V2 CSVs and `assaf_expert_test_set.csv`. The genus
    columns are the columns that are known genera (file order is kept). `coral_id` becomes an
    int and `bbox` a list of floats, parsed without `eval`. Other columns are dropped.
    Columns: `image_name, coral_id, [bbox, area], <genera>, predicted_genus, confidence` and
    `runner_up` when the file has it.
    """
    raw = normalize_keys(pd.read_csv(path))
    known = set(load_genera().known_genera())
    genus_cols = [c for c in raw.columns if c in known]
    if not genus_cols:
        raise ValueError(f"{path}: no genus probability columns found")
    if "predicted_genus" not in raw.columns:
        raise ValueError(f"{path}: no predicted_genus column")
    out = raw[KEY].copy()
    if "bbox" in raw.columns:
        try:
            out["bbox"] = [list(parse_bbox(v)) for v in raw["bbox"]]
        except ValueError as err:
            raise ValueError(f"{path}: {err}") from err
    for col in ("area", *genus_cols, "predicted_genus", "confidence", "runner_up"):
        if col in raw.columns:
            out[col] = raw[col]
    return out


# --------------------------------------------------------------------------- comparison


@dataclass
class PredictionComparison:
    """How two prediction tables differ (see `compare_predictions`)."""

    n_a: int
    n_b: int
    n_common: int
    only_in_a: int
    only_in_b: int
    only_in_a_sample: list[str]
    only_in_b_sample: list[str]
    top1_agreement: float | None  # over the common keys
    n_disagree: int
    max_abs_diff: float | None  # over the common keys and genus columns
    mean_abs_diff: float | None
    worst: dict[str, Any] | None
    genus_columns: list[str]
    disagreements: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def keys_equal(self) -> bool:
        """True if both tables hold exactly the same set of `(image_name, coral_id)` keys."""
        return self.only_in_a == 0 and self.only_in_b == 0

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly summary."""
        return {
            "n_a": self.n_a,
            "n_b": self.n_b,
            "n_common": self.n_common,
            "keys_equal": self.keys_equal,
            "only_in_a": self.only_in_a,
            "only_in_b": self.only_in_b,
            "only_in_a_sample": self.only_in_a_sample,
            "only_in_b_sample": self.only_in_b_sample,
            "top1_agreement": self.top1_agreement,
            "n_disagree": self.n_disagree,
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "worst": self.worst,
        }


def _top1(df: pd.DataFrame, genus_cols: list[str]) -> pd.Series:
    if "predicted_genus" in df.columns:
        return df["predicted_genus"]
    return pd.Series(np.array(genus_cols, dtype=object)[df[genus_cols].to_numpy().argmax(axis=1)])


def compare_predictions(
    a: pd.DataFrame, b: pd.DataFrame, max_listed: int = 20
) -> PredictionComparison:
    """Compare two prediction tables joined on `(image_name, coral_id)`.

    Reports key-set equality, top-1 agreement and the maximum and mean absolute difference of
    the genus probabilities (over the genus columns both tables have). Duplicate keys raise.
    """
    a, b = normalize_keys(a), normalize_keys(b)
    for name, df in (("a", a), ("b", b)):
        if df.duplicated(KEY).any():
            raise ValueError(f"table {name} has duplicate (image_name, coral_id) keys")
    known = set(load_genera().known_genera())
    genus_cols = [c for c in a.columns if c in known and c in b.columns]
    if not genus_cols:
        raise ValueError("the tables share no genus probability columns")
    top_a, top_b = _top1(a, genus_cols), _top1(b, genus_cols)
    a = a.assign(_top=top_a.to_numpy())
    b = b.assign(_top=top_b.to_numpy())
    merged = a.merge(b, on=KEY, how="outer", suffixes=("_a", "_b"), indicator=True)
    uid = _uid(merged)
    left = merged["_merge"] == "left_only"
    right = merged["_merge"] == "right_only"
    both = merged[merged["_merge"] == "both"].reset_index(drop=True)
    uid_both = uid[merged["_merge"] == "both"].reset_index(drop=True)

    result: dict[str, Any] = {
        "top1_agreement": None,
        "n_disagree": 0,
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "worst": None,
    }
    disagreements = pd.DataFrame(columns=["uid", "predicted_a", "predicted_b"])
    if len(both):
        agree = both["_top_a"] == both["_top_b"]
        result["top1_agreement"] = float(agree.mean())
        result["n_disagree"] = int((~agree).sum())
        disagreements = pd.DataFrame(
            {
                "uid": uid_both[~agree],
                "predicted_a": both.loc[~agree, "_top_a"],
                "predicted_b": both.loc[~agree, "_top_b"],
            }
        ).head(max_listed)
        pa = both[[f"{g}_a" for g in genus_cols]].to_numpy(dtype=float)
        pb = both[[f"{g}_b" for g in genus_cols]].to_numpy(dtype=float)
        diff = np.abs(pa - pb)
        result["max_abs_diff"] = float(diff.max())
        result["mean_abs_diff"] = float(diff.mean())
        row, col = np.unravel_index(diff.argmax(), diff.shape)
        result["worst"] = {
            "uid": str(uid_both[row]),
            "genus": genus_cols[col],
            "a": float(pa[row, col]),
            "b": float(pb[row, col]),
        }
    return PredictionComparison(
        n_a=len(a),
        n_b=len(b),
        n_common=len(both),
        only_in_a=int(left.sum()),
        only_in_b=int(right.sum()),
        only_in_a_sample=uid[left].head(5).tolist(),
        only_in_b_sample=uid[right].head(5).tolist(),
        genus_columns=genus_cols,
        disagreements=disagreements.reset_index(drop=True),
        **result,
    )


# --------------------------------------------------------------------------- distribution

_GENUS_COLUMNS = ("genus", "observation")
_COUNT_COLUMNS = ("count", "#")


def _find_column(columns: Sequence[str], candidates: Sequence[str], what: str) -> str:
    lowered = {str(c).strip().lower(): c for c in columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    raise ValueError(f"No {what} column; looked for {list(candidates)} in {list(columns)}")


def distribution_vs_ground_truth(
    predictions: pd.DataFrame,
    gt_csv: str | os.PathLike[str],
    label_set: str | Sequence[str] = "hard20",
) -> pd.DataFrame:
    """Predicted genus mix against the survey counts, within the label set.

    The ground-truth file may name its columns `genus`/`count` or `Observation`/`#`; genera
    outside the label set (soft corals and so on) are left out and listed in
    `result.attrs["survey_excluded"]` (genus -> count). Shares are within the label set.
    Columns: `predicted_n, predicted_share, survey_n, survey_share, difference_pp` (predicted
    minus survey, in percentage points) and `ratio` (predicted share over survey share).
    """
    genera = resolve_genera(label_set)
    canonical = {g.lower(): g for g in genera}
    gt = pd.read_csv(gt_csv)
    genus_col = _find_column(gt.columns, _GENUS_COLUMNS, "genus")
    count_col = _find_column(gt.columns, _COUNT_COLUMNS, "count")
    counts = pd.to_numeric(gt[count_col], errors="coerce")
    if counts.isna().any():
        logger.warning(
            "%d survey row(s) have no numeric count and are ignored", counts.isna().sum()
        )
    names = gt[genus_col].astype(str).str.strip()
    survey: dict[str, float] = {g: 0.0 for g in genera}
    excluded: dict[str, float] = {}
    for name, n in zip(names, counts, strict=True):
        if pd.isna(n):
            continue
        genus = canonical.get(name.lower())
        if genus is None:
            excluded[name] = excluded.get(name, 0.0) + float(n)
        else:
            survey[genus] += float(n)

    predicted = predictions["predicted_genus"].value_counts()
    table = pd.DataFrame(index=pd.Index(genera, name="genus"))
    table["predicted_n"] = [int(predicted.get(g, 0)) for g in genera]
    table["survey_n"] = [survey[g] for g in genera]
    pred_total, survey_total = table["predicted_n"].sum(), table["survey_n"].sum()
    table["predicted_share"] = table["predicted_n"] / pred_total if pred_total else np.nan
    table["survey_share"] = table["survey_n"] / survey_total if survey_total else np.nan
    table["difference_pp"] = 100 * (table["predicted_share"] - table["survey_share"])
    table["ratio"] = table["predicted_share"] / table["survey_share"].where(
        table["survey_share"] > 0
    )
    table = table[
        ["predicted_n", "predicted_share", "survey_n", "survey_share", "difference_pp", "ratio"]
    ]
    table.attrs["survey_excluded"] = excluded
    table.attrs["predicted_outside_label_set"] = int(len(predictions) - pred_total)
    return table
