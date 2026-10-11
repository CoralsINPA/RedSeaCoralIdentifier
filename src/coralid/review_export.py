"""Writing a blind review to disk, and reading its hidden key back.

`write_blind_review` saves, as new files only (it refuses to overwrite and writes nothing if any
target exists):

- in `exports_dir`, to hand out: `<name>_analyst.csv` and `.pdf`, `<name>_assaf.csv` and `.pdf`.
  The CSV is the blank sheet (`key`, `is_<genus>`, `actual_genus`, `notes`); the PDF shows the
  masked crops with the caption `#<row> <key>` and nothing else;
- in `keys_dir`, **to keep to yourself**: `<name>_answer_key.csv` (why each crop is there and
  what the model said) and `<name>_answer_key.meta.json` (seed, pool sizes, counts).

`read_blind_key` loads the key and its meta for scoring later, possibly in another session.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from coralid._atomic import atomic_path, atomic_write_text
from coralid.config import CropConfig
from coralid.genus_audit import BlindReview
from coralid.labeling import export_review_sheet
from coralid.masks import AnnotationStore, CoralKey
from coralid.viz import crop_grid_pdf


def blind_review_name(genus: str, seed: int, date: str | None = None) -> str:
    """The base file name: `millepora_blind_2026-10-12_seed0`."""
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"{genus.lower()}_blind_{date}_seed{seed}"


def _items(keys: pd.Series) -> pd.DataFrame:
    """The table `crop_grid_pdf` takes, with the numbered caption of each sheet row."""
    parsed = [CoralKey.from_uid(k) for k in keys]
    return pd.DataFrame(
        {
            "image_name": [k.image_name for k in parsed],
            "coral_id": [k.coral_id for k in parsed],
            "caption": [f"#{i} {k.uid}" for i, k in enumerate(parsed, start=1)],
        }
    )


def write_blind_review(
    review: BlindReview,
    exports_dir: str | os.PathLike[str],
    keys_dir: str | os.PathLike[str],
    store: AnnotationStore,
    images_dir: str | os.PathLike[str],
    name: str | None = None,
    crops: CropConfig | None = None,
    pdf: bool = True,
) -> dict[str, Path]:
    """Write the sheets, their PDFs and the hidden key; return `{what: path}`.

    Raises `FileExistsError` before writing anything if a target exists. To make another sheet
    change the seed (it is in the name) or pass a different `name`.
    """
    name = name or blind_review_name(review.meta["genus"], review.meta["seed"])
    exports, keys = Path(exports_dir), Path(keys_dir)
    targets = {
        "key": keys / f"{name}_answer_key.csv",
        "key_meta": keys / f"{name}_answer_key.meta.json",
        "analyst_csv": exports / f"{name}_analyst.csv",
        "assaf_csv": exports / f"{name}_assaf.csv",
    }
    if pdf:
        targets["analyst_pdf"] = exports / f"{name}_analyst.pdf"
        targets["assaf_pdf"] = exports / f"{name}_assaf.pdf"
    taken = [str(p) for p in targets.values() if p.exists()]
    if taken:
        raise FileExistsError(f"refusing to overwrite {taken}; change the seed or the name")

    with atomic_path(targets["key"]) as tmp:
        review.key.to_csv(tmp, index=False)
    meta = {**review.meta, "name": name, "answer_column": review.answer}
    atomic_write_text(targets["key_meta"], json.dumps(meta, indent=2, default=str) + "\n")
    for who, sheet in (("analyst", review.sheet), ("assaf", review.assaf_sheet)):
        export_review_sheet(sheet["key"].tolist(), targets[f"{who}_csv"], review.answer)
        if pdf:
            title = f"Blind review of {review.meta['genus']} ({who})"
            crop_grid_pdf(
                _items(sheet["key"]), targets[f"{who}_pdf"], store, images_dir, crops, title=title
            )
    return targets


def read_blind_key(
    keys_dir: str | os.PathLike[str], name: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read `<name>_answer_key.csv` and its meta back: `(key, meta)` ready for scoring."""
    folder = Path(keys_dir)
    key = pd.read_csv(
        folder / f"{name}_answer_key.csv",
        dtype={
            "key": str,
            "image_name": str,
            "selection": str,
            "new_pred": str,
            "prior_answer": str,
        },
    )
    key["coral_id"] = key["coral_id"].astype(int)
    key["for_assaf"] = key["for_assaf"].astype(bool)
    meta = json.loads((folder / f"{name}_answer_key.meta.json").read_text(encoding="utf-8"))
    return key, meta
