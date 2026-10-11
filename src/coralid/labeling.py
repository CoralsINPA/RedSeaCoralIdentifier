"""Review sheets: export a sheet for a person to fill in, ingest the completed one by key, and
audit one genus.

A review answers one question per crop, usually "is this genus X?", with `yes`, `no` or `unsure`
(blank means not reviewed yet). A `no` must say what the coral is instead, in `actual_genus`.
Sheets are CSV files that people edit by hand; the coral key (`IMG_4033.JPG#14`, see
`coralid.masks.CoralKey`) is in the first column, so a completed sheet joins to predictions by key
and never by `coral_id` alone.

- `export_review_sheet` writes a blank sheet (a new file, never over an existing one).
- `ingest_review` reads a completed sheet, normalizes the answers and lists the problems.
- `audit_sheet` and `summarize_audit` are the per-genus audit of the Millepora notebook for any
  genus: the sheet lists the crops the model called that genus, the summary counts the answers.

The blind, mixed sheet of the Millepora re-analysis is in `coralid.genus_audit`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from coralid._atomic import atomic_path
from coralid.evaluation import wilson_interval
from coralid.genera import load_genera, resolve_genera
from coralid.masks import CoralKey, as_int

logger = logging.getLogger(__name__)

YES, NO, UNSURE = "yes", "no", "unsure"
ANSWERS = (YES, NO, UNSURE)
_ANSWER_WORDS = {
    YES: {"yes", "y", "true", "1"},
    NO: {"no", "n", "false", "0"},
    UNSURE: {"?", "??", "unsure", "unknown", "maybe", "not sure", "u"},
}
_BLANK = {"", "nan", "none", "<na>"}

ACTUAL_GENUS = "actual_genus"
NOTES = "notes"

PROBLEM_DUPLICATE = "duplicate_key"
PROBLEM_INVALID = "invalid_answer"
PROBLEM_NO_GENUS = "missing_actual_genus"
PROBLEM_UNKNOWN_GENUS = "unknown_genus"


def answer_column(genus: str) -> str:
    """The name of the answer column for a genus: `is_millepora` for Millepora."""
    return f"is_{genus.lower()}"


def normalize_answer(value: Any) -> str | None:
    """`yes`, `no` or `unsure` for a cell, None for a blank one; anything else raises.

    Accepts the common spellings: yes / y / true, no / n / false, `?` / unsure / unknown.
    """
    blank = value is None or (not isinstance(value, str) and bool(pd.isna(value)))
    text = "" if blank else str(value).strip().lower()
    if text in _BLANK:
        return None
    for answer, words in _ANSWER_WORDS.items():
        if text in words:
            return answer
    raise ValueError(f"not an answer: {value!r} (use yes, no or unsure)")


# --------------------------------------------------------------------------- export


def export_review_sheet(
    keys: Sequence[str],
    path: str | os.PathLike[str],
    answer: str,
    extra: pd.DataFrame | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a blank review sheet with columns `key`, [extra...], `answer`, `actual_genus`, `notes`.

    `keys` are coral uids, in the order the reviewer sees them. `extra` holds additional
    columns to show (same length and order as `keys`). Nothing else is written, so a blind sheet
    shows no prediction. An existing file is an error unless `overwrite`.
    """
    if len(set(keys)) != len(keys):
        raise ValueError("a review sheet needs unique keys")
    if Path(path).exists() and not overwrite:
        raise FileExistsError(f"{path} exists; refusing to overwrite it")
    sheet = pd.DataFrame({"key": list(keys)})
    if extra is not None:
        if len(extra) != len(keys):
            raise ValueError(f"extra has {len(extra)} rows for {len(keys)} keys")
        for col in extra.columns:
            sheet[col] = extra[col].to_numpy()
    for col in (answer, ACTUAL_GENUS, NOTES):
        sheet[col] = ""
    with atomic_path(path) as tmp:
        sheet.to_csv(tmp, index=False)
    return Path(path)


# --------------------------------------------------------------------------- ingest


@dataclass
class ReviewIngest:
    """A completed review sheet, normalized.

    `table` has one row per key: `key`, `image_name`, `coral_id`, `answer` (yes / no / unsure /
    None while pending), `actual_genus` (canonical spelling when the genus is known), `notes` and
    the sheet's other columns. `problems` lists what to fix: `key`, `problem`, `detail`.
    """

    table: pd.DataFrame
    problems: pd.DataFrame = field(default_factory=pd.DataFrame)

    def counts(self) -> dict[str, int]:
        """Answers per kind, with `pending` for rows not answered yet."""
        answers = self.table["answer"]
        return {a: int((answers == a).sum()) for a in ANSWERS} | {
            "pending": int(answers.isna().sum())
        }


def _canonical_genus(text: str, known: dict[str, str]) -> tuple[str | None, bool]:
    """`(genus, is_known)`: the spelling to use, None for an empty cell."""
    text = text.strip()
    if text.lower() in _BLANK:
        return None, True
    if text.lower() in known:
        return known[text.lower()], True
    return text, False


def ingest_review(
    source: str | os.PathLike[str] | pd.DataFrame,
    answer: str,
    label_set: str | Sequence[str] = "hard20",
) -> ReviewIngest:
    """Read a completed review sheet by key and normalize it.

    `answer` is the answer column (`is_millepora`). The key comes from the `key` column, or from
    `image_name` and `coral_id` for older sheets. Problems, none of which raises: a duplicate
    key (the first row is kept), an answer that is not yes / no / unsure (treated as pending),
    a `no` without `actual_genus`, and an `actual_genus` that is not a known genus (kept as
    written, so a typo shows up here).
    """
    if _is_path(source):
        text = {col: str for col in ("key", answer, ACTUAL_GENUS, NOTES)}
        raw = pd.read_csv(source, dtype=text)
    else:
        raw = source
    raw = raw.copy().reset_index(drop=True)
    if answer not in raw.columns:
        raise ValueError(f"the sheet has no '{answer}' column; columns: {list(raw.columns)}")
    if "key" not in raw.columns:
        if not {"image_name", "coral_id"} <= set(raw.columns):
            raise ValueError("the sheet needs a 'key' column, or image_name and coral_id")
        raw["key"] = [
            CoralKey(str(n).strip(), as_int(c)).uid
            for n, c in zip(raw["image_name"], raw["coral_id"], strict=True)
        ]
    raw["key"] = raw["key"].astype(str).str.strip()
    known = {g.lower(): g for g in [*load_genera().known_genera(), *resolve_genera(label_set)]}

    problems: list[dict[str, str]] = []
    dup = raw["key"].duplicated(keep="first")
    for key in raw.loc[dup, "key"]:
        problems.append({"key": key, "problem": PROBLEM_DUPLICATE, "detail": "later rows ignored"})
    raw = raw[~dup].reset_index(drop=True)

    answers: list[str | None] = []
    genera: list[str | None] = []
    for key, value, genus_cell in zip(
        raw["key"], raw[answer], raw.get(ACTUAL_GENUS, [""] * len(raw)), strict=True
    ):
        try:
            normalized = normalize_answer(value)
        except ValueError as err:
            problems.append({"key": key, "problem": PROBLEM_INVALID, "detail": str(err)})
            normalized = None
        genus, is_known = _canonical_genus("" if pd.isna(genus_cell) else str(genus_cell), known)
        if normalized == NO and genus is None:
            problems.append({"key": key, "problem": PROBLEM_NO_GENUS, "detail": "what is it?"})
        if genus is not None and not is_known:
            problems.append({"key": key, "problem": PROBLEM_UNKNOWN_GENUS, "detail": genus})
        answers.append(normalized)
        genera.append(genus)

    uid = [CoralKey.from_uid(k) for k in raw["key"]]
    table = pd.DataFrame(
        {
            "key": raw["key"],
            "image_name": [k.image_name for k in uid],
            "coral_id": [k.coral_id for k in uid],
            "answer": pd.Series(answers, dtype=object),
            ACTUAL_GENUS: pd.Series(genera, dtype=object),
            NOTES: raw[NOTES].fillna("").astype(str) if NOTES in raw.columns else "",
        }
    )
    skip = {"key", "image_name", "coral_id", answer, ACTUAL_GENUS, NOTES}
    for col in raw.columns:
        if col not in skip:
            table[col] = raw[col]
    return ReviewIngest(table, pd.DataFrame(problems, columns=["key", "problem", "detail"]))


def _is_path(source: Any) -> bool:
    return isinstance(source, (str, os.PathLike))


# --------------------------------------------------------------------------- per-genus audit


def audit_sheet(
    predictions: pd.DataFrame,
    genus: str,
    min_confidence: float | None = None,
    columns: Sequence[str] = ("image_name", "coral_id", "bbox", "area", "confidence", "runner_up"),
) -> pd.DataFrame:
    """The crops the model called `genus`, most confident first, with blank answer columns.

    The generic form of the Millepora audit sheet (MA cell 7). It shows the model's answer, so
    it is not a blind sheet. Columns: `key`, those of `columns` that exist, then `is_<genus>`,
    `actual_genus` and `notes`.
    """
    chosen = predictions[predictions["predicted_genus"] == genus]
    if min_confidence is not None:
        chosen = chosen[chosen["confidence"] >= min_confidence]
    chosen = chosen.sort_values("confidence", ascending=False, kind="stable")
    sheet = pd.DataFrame(
        {
            "key": [
                CoralKey(n, c).uid
                for n, c in zip(chosen["image_name"], chosen["coral_id"], strict=True)
            ]
        }
    )
    for col in columns:
        if col in chosen.columns:
            sheet[col] = chosen[col].to_numpy()
    for col in (answer_column(genus), ACTUAL_GENUS, NOTES):
        sheet[col] = ""
    return sheet.reset_index(drop=True)


@dataclass
class AuditSummary:
    """The answers of a per-genus audit (see `summarize_audit`)."""

    genus: str
    n_rows: int
    counts: dict[str, int]  # yes, no, unsure, pending
    precision_decided: float | None  # yes / (yes + no)
    precision_decided_ci95: tuple[float, float] | None
    precision_unsure_wrong: float | None  # yes / (yes + no + unsure)
    precision_unsure_right: float | None  # (yes + unsure) / (yes + no + unsure)
    actual_genera: dict[str, int]  # what the `no` crops really are

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly form."""
        return dict(self.__dict__)


def summarize_audit(review: ReviewIngest | pd.DataFrame, genus: str) -> AuditSummary:
    """Count the answers of an audit and give the share of `yes` as a range.

    The figure is the share of reviewed crops the reviewer agreed with. It is **not** a
    precision unless the reviewer was an expert, did not see the model's answer and the crops
    were a random sample. `precision_unsure_wrong` and `precision_unsure_right` count the
    unsure crops as wrong and as right.
    """
    table = review.table if isinstance(review, ReviewIngest) else review
    answers = table["answer"]
    counts = {a: int((answers == a).sum()) for a in ANSWERS} | {
        "pending": int(answers.isna().sum())
    }
    yes, no, unsure = counts[YES], counts[NO], counts[UNSURE]
    decided, reviewed = yes + no, yes + no + unsure
    ci = wilson_interval(yes, decided)
    wrong = table.loc[answers == NO, ACTUAL_GENUS].dropna()
    return AuditSummary(
        genus=genus,
        n_rows=len(table),
        counts=counts,
        precision_decided=yes / decided if decided else None,
        precision_decided_ci95=ci,
        precision_unsure_wrong=yes / reviewed if reviewed else None,
        precision_unsure_right=(yes + unsure) / reviewed if reviewed else None,
        actual_genera=wrong.value_counts().to_dict(),
    )
