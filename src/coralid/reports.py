"""The baseline report: the expert-set evaluation and the genus mix, broken down by orientation.

`baseline_report` takes the baseline predictions, the manifest (which knows each photo's frame),
the expert set and the survey counts, and gives:

- the evaluation of the new predictions against the expert labels, for all rows and separately
  for expert rows on landscape and on portrait photos;
- the same for the predictions stored in the expert CSV (the legacy ones), so the two can be
  compared on the same rows;
- the genus distribution against `ground_truth_genus_counts.csv` (the label set's genera only);
- a per-orientation summary.

**Read the portrait numbers with care.** Assaf judged crops cut by the legacy code. On portrait
photos those were cut from the wrong region, so his label may describe a different-looking crop
than the one `coralid` classifies. The fair comparison with the legacy result is the landscape
one; the portrait one is informational. `write_baseline_report` saves everything as new files
in a new folder under the run's `reports/`. `write_probe_report` does the same for the held-out
report of the linear probe (`coralid.probe`).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from coralid.evaluation import (
    IN_SET,
    EvaluationReport,
    distribution_vs_ground_truth,
    evaluate,
)
from coralid.genera import resolve_genera
from coralid.orientation import ORIENTATIONS, add_orientation, image_orientations
from coralid.portrait_impact import CAUTION
from coralid.predictions import normalize_keys
from coralid.probe import ProbeReport

GROUPS = ("all", *ORIENTATIONS)


@dataclass
class BaselineReport:
    """The result of `baseline_report`; every dict is keyed `all`, `landscape`, `portrait`."""

    label_set: str
    expert_new: dict[str, EvaluationReport]
    expert_legacy: dict[str, EvaluationReport]
    expert_paired: pd.DataFrame  # new against legacy on the same rows, per group
    distribution: dict[str, pd.DataFrame]
    genus_by_orientation: pd.DataFrame
    orientation_summary: pd.DataFrame
    caution: str = CAUTION
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly summary (the tables are saved as CSV by `write_baseline_report`)."""
        return _clean(
            {
                "label_set": self.label_set,
                "caution": self.caution,
                "notes": self.notes,
                "expert_new": {k: v.to_dict() for k, v in self.expert_new.items()},
                "expert_legacy": {k: v.to_dict() for k, v in self.expert_legacy.items()},
                "expert_paired": self.expert_paired.to_dict("records"),
                "orientation_summary": self.orientation_summary.to_dict("records"),
            }
        )


def _clean(obj: Any) -> Any:
    """Make a structure JSON-safe: NaN becomes None, numpy numbers become Python numbers."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _rows_in(df: pd.DataFrame, frames: pd.DataFrame, group: str) -> pd.DataFrame:
    """The rows of `df` on photos of one group (`all` keeps everything)."""
    if group == "all":
        return df
    tagged = add_orientation(df, frames)
    return tagged[tagged["orientation"] == group].drop(columns="orientation")


def _paired(expert: pd.DataFrame, legacy: pd.DataFrame, new: pd.DataFrame) -> dict[str, Any]:
    """New and legacy compared on the expert rows that have both predictions."""
    both = normalize_keys(legacy)[["image_name", "coral_id", "predicted_genus"]].merge(
        normalize_keys(new)[["image_name", "coral_id", "predicted_genus"]],
        on=["image_name", "coral_id"],
        suffixes=("_legacy", "_new"),
    )
    rows = normalize_keys(expert).merge(both, on=["image_name", "coral_id"])
    named = rows[rows["status"] == IN_SET]
    n = len(named)
    legacy_ok = int((named["expert_genus"] == named["predicted_genus_legacy"]).sum())
    new_ok = int((named["expert_genus"] == named["predicted_genus_new"]).sum())
    changed = int((rows["predicted_genus_legacy"] != rows["predicted_genus_new"]).sum())
    return {
        "expert_rows": len(expert),
        "with_both_predictions": len(rows),
        "in_set_paired": n,
        "legacy_correct": legacy_ok,
        "new_correct": new_ok,
        "legacy_accuracy": legacy_ok / n if n else None,
        "new_accuracy": new_ok / n if n else None,
        "predictions_changed": changed,
    }


def baseline_report(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    expert: pd.DataFrame,
    legacy_predictions: pd.DataFrame,
    ground_truth_csv: str | os.PathLike[str],
    label_set: str | list[str] = "hard20",
) -> BaselineReport:
    """Evaluate the baseline against the expert set and the survey, by orientation.

    `expert` comes from `load_expert_set`, `legacy_predictions` from `load_legacy_predictions`
    on the expert CSV, `manifest` is the run's image manifest.
    """
    genera = resolve_genera(label_set)
    expert = normalize_keys(expert)
    new_eval, legacy_eval, paired, distribution = {}, {}, [], {}
    for group in GROUPS:
        subset = _rows_in(expert, manifest, group)
        new_eval[group] = evaluate(predictions, subset, genera)
        legacy_eval[group] = evaluate(legacy_predictions, subset, genera)
        paired.append({"group": group} | _paired(subset, legacy_predictions, predictions))
        distribution[group] = distribution_vs_ground_truth(
            _rows_in(predictions, manifest, group), ground_truth_csv, genera
        )
    return BaselineReport(
        label_set=label_set if isinstance(label_set, str) else ",".join(genera),
        expert_new=new_eval,
        expert_legacy=legacy_eval,
        expert_paired=pd.DataFrame(paired),
        distribution=distribution,
        genus_by_orientation=_genus_mix(predictions, manifest, genera),
        orientation_summary=_orientation_summary(predictions, manifest),
    )


def _orientation_summary(predictions: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Photos, masks, kept crops and prediction statistics per orientation."""
    orient = image_orientations(manifest)
    tagged = add_orientation(manifest, manifest)
    preds = add_orientation(predictions, manifest)
    rows = []
    for group in GROUPS:
        masks = tagged if group == "all" else tagged[tagged["orientation"] == group]
        mine = preds if group == "all" else preds[preds["orientation"] == group]
        photos = len(orient) if group == "all" else int((orient == group).sum())
        rows.append(
            {
                "group": group,
                "photos": photos,
                "masks": len(masks),
                "kept_in_manifest": int(masks["keep"].sum()),
                "predictions": len(mine),
                "mean_confidence": float(mine["confidence"].mean()) if len(mine) else None,
                "share_confidence_ge_085": float((mine["confidence"] >= 0.85).mean())
                if len(mine)
                else None,
                "mean_margin": float(mine["margin"].mean())
                if len(mine) and "margin" in mine
                else None,
            }
        )
    return pd.DataFrame(rows)


def _genus_mix(
    predictions: pd.DataFrame, manifest: pd.DataFrame, genera: list[str]
) -> pd.DataFrame:
    """Share of the predictions per genus, for all photos and each orientation."""
    columns = {}
    for group in GROUPS:
        part = _rows_in(predictions, manifest, group)
        counts = part["predicted_genus"].value_counts()
        columns[f"{group}_n"] = [int(counts.get(g, 0)) for g in genera]
        total = max(1, len(part))
        columns[f"{group}_share"] = [counts.get(g, 0) / total for g in genera]
    return pd.DataFrame(columns, index=pd.Index(genera, name="genus"))


# --------------------------------------------------------------------------- saving


def new_report_folder(
    reports_dir: str | os.PathLike[str], kind: str, name: str = "", date: str | None = None
) -> Path:
    """Create and return `<reports_dir>/<kind>_<name>_<date>/` (the name may be empty).

    An existing folder is never reused: the next free `_2`, `_3`... suffix is taken, so every
    report lands in a folder of its own and no file is ever overwritten.
    """
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    stem = "_".join(part for part in (kind, name, date) if part)
    folder = Path(reports_dir) / stem
    n = 1
    while folder.exists():
        n += 1
        folder = Path(reports_dir) / f"{stem}_{n}"
    folder.mkdir(parents=True)
    return folder


def write_baseline_report(
    report: BaselineReport,
    reports_dir: str | os.PathLike[str],
    name: str,
    date: str | None = None,
) -> Path:
    """Write the report as new files in `<reports_dir>/baseline_<name>_<date>/`.

    `summary.json`, `expert_paired.csv`, `orientation_summary.csv`, `genus_by_orientation.csv`
    and, per group, `expert_<group>_per_genus.csv`, `expert_<group>_confusion.csv` and
    `distribution_<group>.csv`. An existing folder gets a `_2`, `_3`... suffix; no file is
    ever overwritten. Returns the folder.
    """
    folder = new_report_folder(reports_dir, "baseline", name, date)
    report.expert_paired.to_csv(folder / "expert_paired.csv", index=False)
    report.orientation_summary.to_csv(folder / "orientation_summary.csv", index=False)
    report.genus_by_orientation.to_csv(folder / "genus_by_orientation.csv")
    for group in GROUPS:
        report.expert_new[group].per_genus.to_csv(folder / f"expert_{group}_per_genus.csv")
        report.expert_new[group].confusion.to_csv(folder / f"expert_{group}_confusion.csv")
        report.distribution[group].to_csv(folder / f"distribution_{group}.csv")
    with open(folder / "summary.json", "x", encoding="utf-8") as f:
        f.write(json.dumps(report.to_dict(), indent=2) + "\n")
    return folder


def probe_report_lines(report: ProbeReport, dropped: dict[str, int] | None = None) -> list[str]:
    """The printed summary of a probe report (the command line and the notebook show the same)."""
    acc = report.accuracy
    low, high = acc["ci95"] or (float("nan"), float("nan"))
    share = acc["accuracy"] if acc["accuracy"] is not None else float("nan")
    lines = [
        f"Held-out accuracy {acc['correct']}/{acc['total']} = {share:.3f} ({low:.3f}-{high:.3f}) "
        f"on {report.n_labeled} labeled crops from {report.n_images} photos; always answering "
        f"the commonest genus gives {report.majority_baseline:.3f}"
    ]
    if report.zero_shot and report.zero_shot["total"]:
        shot = report.zero_shot
        done = f"{shot['correct']}/{shot['total']} = {shot['accuracy']:.3f}"
        lines.append(f"  zero-shot on the same crops: {done}")
    lines += [f"  note: {note}" for note in report.warnings]
    lines.append(f"  labels left out: {dropped if dropped is not None else report.dropped}")
    return lines


def write_probe_report(
    report: ProbeReport,
    reports_dir: str | os.PathLike[str],
    name: str,
    date: str | None = None,
) -> Path:
    """Write the report as new files in `<reports_dir>/probe_<name>_<date>/` and return it.

    `summary.json`, `per_genus.csv`, `confusion.csv`, `folds.csv` and `held_out.csv`. An existing
    folder gets a `_2`, `_3`... suffix; no file is ever overwritten.
    """
    folder = new_report_folder(reports_dir, "probe", name, date)
    report.evaluation.per_genus.to_csv(folder / "per_genus.csv")
    report.evaluation.confusion.to_csv(folder / "confusion.csv")
    pd.DataFrame(report.folds).to_csv(folder / "folds.csv", index=False)
    report.held_out.to_csv(folder / "held_out.csv", index=False)
    with open(folder / "summary.json", "x", encoding="utf-8") as f:
        f.write(json.dumps(report.to_dict(), indent=2, default=str) + "\n")
    return folder
