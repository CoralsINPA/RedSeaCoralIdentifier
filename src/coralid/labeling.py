"""Review sheets: export a sheet for a person to fill in, and ingest the completed one by key.

A review answers one question per crop, usually "is this genus X?", with `yes`, `no` or `unsure`
(blank means not reviewed yet). A `no` must say what the coral is instead, in `actual_genus`.
Sheets are CSV files that people edit by hand; the coral key (`IMG_4033.JPG#14`, see
`coralid.masks.CoralKey`) is in the first column, so a completed sheet joins to predictions by key
and never by `coral_id` alone.

- `export_review_sheet` writes a blank sheet (a new file, never over an existing one).
- `ingest_review` reads a completed sheet, normalizes the answers and lists the problems.

The blind, mixed sheet of one genus (the Millepora check) is in `coralid.genus_audit`.
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
from coralid.genera import load_genera, resolve_genera
from coralid.masks import CoralKey

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

    `answer` is the answer column (`is_millepora`). The key comes from the `key` column (a sheet
    without one is an error: a row is never identified by `coral_id` alone). Problems, none of
    which raises: a duplicate key (the first row is kept), an answer that is not yes / no /
    unsure (treated as pending), a `no` without `actual_genus`, and an `actual_genus` that is not
    a known genus (kept as written, so a typo shows up here).
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
        raise ValueError("the sheet needs a 'key' column (IMG_4033.JPG#14)")
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
