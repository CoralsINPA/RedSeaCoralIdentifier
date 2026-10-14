"""Labels for the probe, from the expert set: the rows that can be trusted for the new crops.

`coralid.probe` never makes its own labels; it is given a table. Today the only table is the
expert test set (`assaf_expert_test_set.csv`, 77 rows). `expert_labels` turns it into a labels
table, keeping:

- rows whose expert genus is one of the label set's genera (`in_set`). The soft corals,
  `unidentifiable` and `MISSING_LABEL` rows are left out and counted;
- rows on photos of one orientation, **landscape** by default. On the 221 portrait photos the
  legacy code cut the wrong region, and Assaf labeled those mis-cut crops, so his label does not
  describe the crop `coralid` classifies now (`coralid.portrait_impact`). Using those rows would
  teach the probe the wrong image for the label.

Zero-shot predictions are never used here: a model's own guesses are not labels.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import pandas as pd

from coralid.evaluation import (
    IN_SET,
    MISSING,
    OUT_OF_SET,
    UNIDENTIFIABLE,
    load_expert_set,
)
from coralid.orientation import LANDSCAPE, add_orientation


def expert_labels(
    expert_csv: str | os.PathLike[str],
    manifest: pd.DataFrame,
    label_set: str | Sequence[str] = "hard20",
    orientation: str = LANDSCAPE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The probe's labels from the expert set: `(labels, summary)`.

    `labels` has `image_name`, `coral_id` and `genus` (canonical spelling), one row per usable
    expert row. `manifest` supplies each photo's orientation (from its frame). The summary counts
    the expert rows, their status and orientation, what was kept per genus, and why each row that
    was left out was left out (`dropped`; every expert row is in exactly one place).
    """
    expert = add_orientation(load_expert_set(expert_csv, label_set), manifest)
    by_status = expert["status"].value_counts()
    usable = expert[expert["status"] == IN_SET]
    kept = usable[usable["orientation"] == orientation]
    elsewhere = usable[usable["orientation"] != orientation]
    labels = kept[["image_name", "coral_id"]].copy()
    labels["genus"] = kept["expert_genus"].astype(str)
    summary = {
        "expert_rows": len(expert),
        "by_status": {str(k): int(v) for k, v in by_status.items()},
        "by_orientation": {str(k): int(v) for k, v in expert["orientation"].value_counts().items()},
        "orientation_kept": orientation,
        "n_labels": len(labels),
        "n_photos": int(labels["image_name"].nunique()),
        "genera": {str(k): int(v) for k, v in labels["genus"].value_counts().items()},
        "dropped": {
            "outside_the_label_set": int(by_status.get(OUT_OF_SET, 0)),
            UNIDENTIFIABLE: int(by_status.get(UNIDENTIFIABLE, 0)),
            MISSING: int(by_status.get(MISSING, 0)),
            f"on_other_orientation_than_{orientation}": int(
                (elsewhere["orientation"] != "unknown").sum()
            ),
            "photo_not_in_the_manifest": int((elsewhere["orientation"] == "unknown").sum()),
        },
    }
    return labels.reset_index(drop=True), summary
