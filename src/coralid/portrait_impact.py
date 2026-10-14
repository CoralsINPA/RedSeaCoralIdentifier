"""The portrait-impact report: which reviewed crops sit on portrait photos, and what changed.

The legacy code mis-cut every crop on the 221 portrait photos (1920x2560 frames). The expert set
was judged on legacy crops, so for portrait photos the expert's label may describe a
different-looking crop than the one `coralid` now cuts. `expert_portrait_impact` lists those
rows: every row of the expert set on a portrait photo, with the label and status, the legacy
prediction stored in the file and the new prediction.

It takes the manifest (it knows each photo's frame) and the new predictions (the baseline).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from coralid.evaluation import IN_SET, OUT_OF_SET, STATUSES, UNIDENTIFIABLE
from coralid.orientation import LANDSCAPE, PORTRAIT, add_orientation, image_orientations
from coralid.predictions import normalize_keys

CAUTION = (
    "The expert judged crops cut by the legacy code. On portrait photos those were cut from the "
    "wrong region, so the label may describe a different-looking crop than the one coralid cuts "
    "now."
)


def _key(df: pd.DataFrame) -> pd.Series:
    return df["image_name"].astype(str) + "#" + df["coral_id"].astype(str)


def _prediction_columns(predictions: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Key, predicted genus and confidence of a prediction table under a column prefix."""
    pred = normalize_keys(predictions)
    cols = {"predicted_genus": f"{prefix}_pred"}
    if "confidence" in pred.columns:
        cols["confidence"] = f"{prefix}_conf"
    out = pred[["image_name", "coral_id", *cols]].rename(columns=cols)
    if f"{prefix}_conf" not in out.columns:
        out[f"{prefix}_conf"] = float("nan")
    return out


def _counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().items()}


# --------------------------------------------------------------------------- expert set


def expert_portrait_impact(
    expert: pd.DataFrame,
    legacy_predictions: pd.DataFrame,
    new_predictions: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Every expert-set row on a portrait photo, and counts for all rows by orientation.

    `expert` comes from `load_expert_set`; `legacy_predictions` are the predictions stored in the
    expert CSV (`load_legacy_predictions`); `new_predictions` are the baseline's. Returns
    `(rows, summary)`. `rows` has `image_name, coral_id, key, status, expert_raw, expert_genus,
    legacy_pred, legacy_conf, new_pred, new_conf, in_new_baseline, prediction_changed,
    legacy_correct, new_correct`; the correctness columns are None unless the expert named a
    genus.
    """
    table = normalize_keys(expert)[
        ["image_name", "coral_id", "status", "expert_raw", "expert_genus"]
    ]
    table = table.merge(
        _prediction_columns(legacy_predictions, "legacy"), on=["image_name", "coral_id"], how="left"
    )
    table = table.merge(
        _prediction_columns(new_predictions, "new"), on=["image_name", "coral_id"], how="left"
    )
    table = add_orientation(table, manifest)
    table["key"] = _key(table)
    table["in_new_baseline"] = table["new_pred"].notna()
    both = table["in_new_baseline"] & table["legacy_pred"].notna()
    table["prediction_changed"] = pd.Series(
        (table["legacy_pred"] != table["new_pred"]).where(both), dtype=object
    )
    named = table["status"].isin([IN_SET, OUT_OF_SET])
    for side in ("legacy", "new"):
        correct = table["expert_genus"] == table[f"{side}_pred"]
        table[f"{side}_correct"] = pd.Series(
            correct.where(named & table[f"{side}_pred"].notna()), dtype=object
        )

    portrait = table[table["orientation"] == PORTRAIT]
    rows = portrait.drop(columns="orientation").sort_values(
        ["image_name", "coral_id"], ignore_index=True
    )
    return rows, _expert_summary(table, portrait, manifest)


def _correct(series: pd.Series) -> dict[str, int]:
    known = series.dropna()
    return {"correct": int(known.astype(bool).sum()), "scored": len(known)}


def _expert_summary(
    table: pd.DataFrame, portrait: pd.DataFrame, manifest: pd.DataFrame
) -> dict[str, Any]:
    orientations = image_orientations(manifest)
    share_photos = float((orientations == PORTRAIT).mean()) if len(orientations) else None
    unid = table[table["status"] == UNIDENTIFIABLE]
    status_by_orientation = {
        name: {s: int((part["status"] == s).sum()) for s in STATUSES}
        for name, part in table.groupby("orientation")
    }
    changed = portrait["prediction_changed"].dropna().astype(bool)
    return {
        "n_expert_rows": len(table),
        "rows_by_orientation": _counts(table["orientation"]),
        "status_by_orientation": status_by_orientation,
        "portrait_rows": len(portrait),
        "portrait_share_of_expert_rows": len(portrait) / len(table) if len(table) else None,
        "portrait_share_of_photos": share_photos,
        "unidentifiable_total": len(unid),
        "unidentifiable_on_portrait": int((unid["orientation"] == PORTRAIT).sum()),
        "portrait_status": {s: int((portrait["status"] == s).sum()) for s in STATUSES},
        "portrait_in_new_baseline": int(portrait["in_new_baseline"].sum()),
        "portrait_prediction_changed": {"changed": int(changed.sum()), "compared": len(changed)},
        "portrait_legacy": _correct(portrait["legacy_correct"]),
        "portrait_new": _correct(portrait["new_correct"]),
        "landscape_prediction_changed": int(
            table.loc[table["orientation"] == LANDSCAPE, "prediction_changed"]
            .dropna()
            .astype(bool)
            .sum()
        ),
        "caution": CAUTION,
    }


# --------------------------------------------------------------------------- saving


def write_portrait_report(
    expert_rows: pd.DataFrame,
    expert_summary: dict[str, Any],
    reports_dir: str | os.PathLike[str],
    date: str | None = None,
) -> Path:
    """Write the report as new files in a new folder `<reports_dir>/portrait_impact_<date>/`.

    Files: `summary.json` and `expert_portrait_rows.csv`. If the folder exists a `_2`, `_3`...
    is used; nothing is overwritten.
    """
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    folder = Path(reports_dir) / f"portrait_impact_{date}"
    n = 1
    while folder.exists():
        n += 1
        folder = Path(reports_dir) / f"portrait_impact_{date}_{n}"
    folder.mkdir(parents=True)
    expert_rows.to_csv(folder / "expert_portrait_rows.csv", index=False)
    with open(folder / "summary.json", "x", encoding="utf-8") as f:
        f.write(json.dumps({"expert": expert_summary}, indent=2, default=str) + "\n")
    return folder
