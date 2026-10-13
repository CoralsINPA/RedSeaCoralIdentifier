"""Scoring a blind review (`coralid.genus_audit.build_blind_review`).

Inputs: the analyst's completed sheet, the hidden answer key and, optionally, the expert's
completed sheet for his subset. The key records which predictions file the crops were drawn
from; scoring **stops** unless that is the baseline as it is now (`coralid.review_source`).
Outputs:

- **Precision** of the baseline's genus calls and the **miss rate** among its other calls (the
  share of crops it did not call the genus that are the genus), then **recall**, each with an
  interval. The crops are random samples of the baseline's two groups of calls; the interval is
  the Wilson interval of the sample, scaled to the whole group, and the recall interval combines
  the ends of the two, so it is wide on purpose.
- Every figure comes twice, with the unsure counted as **no** and as **yes**; the true value lies
  in between (for recall, the two ends cross, so read the range).
- **Precision by confidence band**, from the random sample of the baseline's calls (the band
  edges are 0.5, 0.7 and 0.85).
- **Analyst against expert**: agreement on the crops both answered, by answer and by Cohen's
  kappa. If it is high the analyst can label for the expert; if it is low the figures above are
  the analyst's opinion only.
- **The confusion lists** from `actual_genus`: what the "no" crops really are (the false
  positives), and what the baseline called the genus crops it missed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from coralid.evaluation import wilson_interval
from coralid.genus_audit import OTHER_RANDOM, PRED_RANDOM
from coralid.labeling import ACTUAL_GENUS, ANSWERS, NO, UNSURE, YES, ReviewIngest
from coralid.review_source import PredictionSource, check_same_source

NOTES = [
    "The analyst is not an expert and the figures are the analyst's opinion until the "
    "agreement with the expert (expert_agreement) is known.",
    "The figures are about the predictions named in 'source' (the 20-genus baseline of that "
    "commit and time), not about any other run.",
]


def pooled_rate(sample: tuple[int, int], pool_size: int) -> dict[str, Any] | None:
    """The share of yes in a pool of crops, from a random sample of it.

    `sample` is `(yes, n)` over the random sample drawn from a pool of `pool_size` crops.
    Returns the point estimate and the Wilson interval of the sample, scaled to the pool; if the
    whole pool was reviewed there is no sampling error and the interval is the point. None if
    nothing was answered.
    """
    yes, n = sample
    if n == 0:
        return None
    if n >= pool_size:  # the "sample" is the whole pool: reviewed in full
        rate = yes / n
        return _rate(rate, rate, rate, n, n)
    interval = wilson_interval(yes, n)
    assert interval is not None
    return _rate(yes / n, interval[0], interval[1], n, pool_size)


def _rate(point: float, low: float, high: float, n_sample: int, pool: int) -> dict:
    return {"point": point, "lo": low, "hi": high, "n_sample": n_sample, "pool": pool}


def _yes_n(answers: pd.Series, unsure_as: str) -> tuple[int, int]:
    """`(yes, answered)`, with the unsure counted as `unsure_as` (`yes` or `no`)."""
    given = answers.dropna()
    yes = int((given == YES).sum()) + (int((given == UNSURE).sum()) if unsure_as == YES else 0)
    return yes, len(given)


def _population(
    key: pd.DataFrame, genus: str, is_target: bool, pool: int, unsure_as: str
) -> dict[str, Any] | None:
    """The yes-rate among the baseline's genus calls (`is_target`) or among its other calls."""
    mine = key[(key["baseline_pred"] == genus) == is_target]
    sampled = mine[mine["selection"] == (PRED_RANDOM if is_target else OTHER_RANDOM)]["answer"]
    return pooled_rate(_yes_n(sampled, unsure_as), pool)


def _recall(precision: dict | None, miss: dict | None, n_called: int, n_other: int) -> Any:
    """Recall from the precision and the miss rate; the interval combines their ends."""
    if precision is None or miss is None:
        return None

    def share(p: float, q: float) -> float | None:
        hits, misses = p * n_called, q * n_other
        return hits / (hits + misses) if hits + misses else None

    return {
        "point": share(precision["point"], miss["point"]),
        "lo": share(precision["lo"], miss["hi"]),
        "hi": share(precision["hi"], miss["lo"]),
    }


def _agreement(analyst: pd.DataFrame, expert: pd.DataFrame) -> dict[str, Any]:
    """How often the analyst's answer equals the expert's on the crops both answered."""
    both = analyst[["key", "answer"]].merge(
        expert[["key", "answer"]], on="key", suffixes=("_a", "_e")
    )
    both = both.dropna()
    n = len(both)
    grid = pd.crosstab(
        both["answer_a"].astype(pd.CategoricalDtype(ANSWERS)),
        both["answer_e"].astype(pd.CategoricalDtype(ANSWERS)),
    )
    agree = int((both["answer_a"] == both["answer_e"]).sum())
    decided = both[both["answer_a"].isin([YES, NO]) & both["answer_e"].isin([YES, NO])]
    expected = (
        sum((both["answer_a"] == a).mean() * (both["answer_e"] == a).mean() for a in ANSWERS)
        if n
        else 0.0
    )
    observed = agree / n if n else 0.0
    kappa = (observed - expected) / (1 - expected) if n and expected < 1 else None
    return {
        "n_both_answered": n,
        "agree": agree,
        "agree_share": observed if n else None,
        "agree_share_ci95": wilson_interval(agree, n),
        "n_decided_by_both": len(decided),
        "decided_agree_share": float((decided["answer_a"] == decided["answer_e"]).mean())
        if len(decided)
        else None,
        "kappa": kappa,
        "analyst_rows_expert_columns": grid.to_dict("index"),
    }


CONFIDENCE_EDGES = (0.0, 0.5, 0.7, 0.85, 1.0 + 1e-9)


def _by_confidence(sample: pd.DataFrame) -> pd.DataFrame:
    """The yes-rate of the random sample of genus calls, per confidence band."""
    rows = []
    bands = zip(CONFIDENCE_EDGES[:-1], CONFIDENCE_EDGES[1:], strict=True)
    for low, high in bands:
        part = sample[(sample["confidence"] >= low) & (sample["confidence"] < high)]
        yes, no, unsure = (int((part["answer"] == a).sum()) for a in (YES, NO, UNSURE))
        decided = yes + no
        interval = wilson_interval(yes, decided)
        rows.append(
            {
                "band": f"{low:.2f}-{min(high, 1.0):.2f}",
                "answered": yes + no + unsure,
                "yes": yes,
                "no": no,
                "unsure": unsure,
                "share_yes_of_decided": yes / decided if decided else None,
                "ci95_lo": interval[0] if interval else None,
                "ci95_hi": interval[1] if interval else None,
            }
        )
    return pd.DataFrame(rows)


def _calls_sample(key: pd.DataFrame) -> pd.DataFrame:
    """The random sample of the baseline's genus calls that has a confidence."""
    sample = key[key["selection"] == PRED_RANDOM]
    if "confidence" not in key.columns:
        return sample.iloc[0:0].assign(confidence=pd.Series(dtype=float))
    return sample[sample["confidence"].notna()]


@dataclass
class BlindScore:
    """The result of `score_blind_review`."""

    genus: str
    n_on_sheet: int
    answered: dict[str, int]  # yes / no / unsure / pending, on the analyst's sheet
    estimates: dict[str, dict[str, Any]]  # unsure_as_no / unsure_as_yes -> precision, miss, recall
    expert_agreement: dict[str, Any] | None
    precision_by_confidence: pd.DataFrame  # the random sample of genus calls, by band
    false_positive_genera: pd.DataFrame  # what the "no" crops really are
    missed_as: pd.DataFrame  # what the baseline called the genus crops it missed
    source: dict[str, Any] = field(default_factory=dict)  # the predictions the key was drawn from
    notes: list[str] = field(default_factory=lambda: list(NOTES))

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly form."""
        return {
            "genus": self.genus,
            "source": self.source,
            "n_on_sheet": self.n_on_sheet,
            "answered": self.answered,
            "estimates": self.estimates,
            "expert_agreement": self.expert_agreement,
            "precision_by_confidence": self.precision_by_confidence.to_dict("records"),
            "false_positive_genera": self.false_positive_genera.to_dict("records"),
            "missed_as": self.missed_as.to_dict("records"),
            "notes": self.notes,
        }


def _table(review: ReviewIngest | pd.DataFrame | None) -> pd.DataFrame | None:
    if review is None:
        return None
    return review.table if isinstance(review, ReviewIngest) else review


def score_blind_review(
    analyst: ReviewIngest | pd.DataFrame,
    key: pd.DataFrame,
    meta: dict[str, Any],
    expert: ReviewIngest | pd.DataFrame | None = None,
    *,
    current_source: PredictionSource,
) -> BlindScore:
    """Score the analyst's answers (and, if given, the expert's) against the answer key.

    `meta` is the dict that `build_blind_review` returned (pool sizes, genus, source) and
    `current_source` is `baseline_source(cfg)` now. Raises `ReviewSourceError` unless the key's
    recorded source is that baseline. Crops the analyst has not answered yet are left out of
    every estimate.
    """
    check_same_source(meta.get("source"), current_source)
    genus = meta["genus"]
    a = _table(analyst)
    e = _table(expert)
    key = key.merge(a[["key", "answer", ACTUAL_GENUS]], on="key", how="left")
    on_sheet = key
    answered = {x: int((on_sheet["answer"] == x).sum()) for x in ANSWERS} | {
        "pending": int(on_sheet["answer"].isna().sum())
    }

    estimates = {}
    for unsure_as in (NO, YES):
        precision = _population(key, genus, True, meta["n_pool_pred"], unsure_as)
        miss = _population(key, genus, False, meta["n_pool_other"], unsure_as)
        called, other = _population_size(precision), _population_size(miss)
        estimates[f"unsure_as_{unsure_as}"] = {
            "precision": precision,
            "miss_rate": miss,
            "recall": _recall(precision, miss, called, other),
        }

    final = on_sheet.assign(final=on_sheet["answer"], final_genus=on_sheet[ACTUAL_GENUS])
    if e is not None:
        said = e.set_index("key")
        by_expert = final["key"].map(said["answer"])
        genus_by_expert = final["key"].map(said[ACTUAL_GENUS])
        final["final"] = by_expert.where(by_expert.notna(), final["final"])
        final["final_genus"] = genus_by_expert.where(by_expert.notna(), final["final_genus"])
    wrong_calls = final[(final["baseline_pred"] == genus) & (final["final"] == NO)]
    missed = final[(final["baseline_pred"] != genus) & (final["final"] == YES)]
    return BlindScore(
        genus=genus,
        n_on_sheet=len(on_sheet),
        answered=answered,
        estimates=estimates,
        expert_agreement=_agreement(a, e) if e is not None else None,
        precision_by_confidence=_by_confidence(_calls_sample(key)),
        false_positive_genera=_count(
            wrong_calls["final_genus"].fillna("(not given)"), "actual_genus"
        ),
        missed_as=_count(missed["baseline_pred"], "called_instead"),
        source=dict(meta["source"]),
    )


def _population_size(rate: dict[str, Any] | None) -> int:
    return 0 if rate is None else rate["pool"]


def _count(values: pd.Series, name: str) -> pd.DataFrame:
    counts = values.value_counts()
    return pd.DataFrame({name: counts.index.to_list(), "count": counts.to_numpy()})


def write_blind_score(
    score: BlindScore,
    reports_dir: str | os.PathLike[str],
    name: str,
    date: str | None = None,
) -> Path:
    """Write a score as new files in `<reports_dir>/blind_score_<name>_<date>/`.

    `summary.json`, the precision by confidence band and the two confusion lists as CSV. An
    existing folder gets a `_2`, `_3`... suffix; no file is overwritten. Returns the folder.
    """
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    folder = Path(reports_dir) / f"blind_score_{name}_{date}"
    n = 1
    while folder.exists():
        n += 1
        folder = Path(reports_dir) / f"blind_score_{name}_{date}_{n}"
    folder.mkdir(parents=True)
    score.precision_by_confidence.to_csv(folder / "precision_by_confidence.csv", index=False)
    score.false_positive_genera.to_csv(folder / "false_positive_genera.csv", index=False)
    score.missed_as.to_csv(folder / "missed_as.csv", index=False)
    with open(folder / "summary.json", "x", encoding="utf-8") as f:
        f.write(json.dumps(score.to_dict(), indent=2, default=str) + "\n")
    return folder
