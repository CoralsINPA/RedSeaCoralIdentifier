"""The pass/fail logic of the Colab parity notebook (`notebooks/parity_check.ipynb`).

The notebook only runs things and prints. Every decision lives here, where it is unit-tested:
each `check_*` function turns measured numbers into a list of `CheckResult` rows for checks
A (geometric), B (zero-shot parity), C (image manifest) and D (expert set). Folding the rows
into a summary and saving it is in `coralid.parity_report`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from coralid.evaluation import EvaluationReport
from coralid.masks import JsonAudit
from coralid.predictions import PredictionComparison

PASS, FAIL, INFO, NOT_RUN = "PASS", "FAIL", "INFO", "NOT RUN"

# Numbers from the legacy runs (docs/agent_prompts/implement_organization_plan.md).
EXPECTED = {
    "n_json_files": 1714,
    "n_masks": 44211,
    "after_geometric": 16197,
    "kept": 14224,
    "files_without_survivors": 4,
    "mean_survivors_per_file": 9.4,
    "frame": "2560x1920",
    "sample_masks": 2337,
    "sample_after_filter": 1054,
    "sample_crops": 842,
    "min_top1_agreement": 0.995,
    "max_abs_diff": 1e-3,
}


@dataclass
class CheckResult:
    """One row of the summary table. `kind="cell"` rows (one per confusion cell) are detail."""

    check: str
    name: str
    status: str
    expected: Any = None
    actual: Any = None
    detail: str = ""
    kind: str = "check"


def equal(check: str, name: str, actual: Any, expected: Any, detail: str = "") -> CheckResult:
    """PASS if `actual == expected`, else FAIL."""
    return CheckResult(check, name, PASS if actual == expected else FAIL, expected, actual, detail)


def info(check: str, name: str, actual: Any, expected: Any = None, detail: str = "") -> CheckResult:
    """A row that reports a number without judging it."""
    return CheckResult(check, name, INFO, expected, actual, detail)


# --------------------------------------------------------------------------- A


def check_json_audit(audit: JsonAudit, expected: Mapping[str, Any] = EXPECTED) -> list[CheckResult]:
    """Check A, part 1: file and mask counts, frames, failed images."""
    off_frame = {k: n for k, n in audit.as_dict()["frame_sizes"].items() if k != expected["frame"]}
    return [
        equal("A", "A1 mask JSON files", audit.n_files, expected["n_json_files"]),
        equal("A", "A2 masks in total", audit.n_masks, expected["n_masks"]),
        equal("A", "A3 unreadable JSONs", len(audit.unreadable), 0, str(audit.unreadable[:3])),
        info(
            "A",
            "A4 frame sizes",
            audit.as_dict()["frame_sizes"],
            expected["frame"],
            f"{sum(off_frame.values())} file(s) not {expected['frame']}"
            if off_frame
            else "all as expected",
        ),
        info(
            "A",
            "A5 JSONs with a note (failed images)",
            len(audit.noted),
            detail=str(audit.noted[:5]),
        ),
        info("A", "A6 JSON name differs from image name", len(audit.name_mismatches)),
    ]


def check_geometric(
    manifest: pd.DataFrame, timings: Mapping[str, float], expected: Mapping[str, Any] = EXPECTED
) -> list[CheckResult]:
    """Check A, part 2: how many masks survive area, aspect and containment, and how fast."""
    survivors = manifest["stage1_reason"].isna()
    per_file = survivors.groupby(manifest["image_name"]).sum()
    reasons = manifest["stage1_reason"].value_counts().to_dict()
    return [
        equal(
            "A",
            "A7 masks after area/aspect/containment",
            int(survivors.sum()),
            expected["after_geometric"],
        ),
        info("A", "A8 removed per reason", reasons),
        info(
            "A",
            "A9 files with no survivor / mean survivors per file",
            [int((per_file == 0).sum()), round(float(per_file.mean()), 2)],
            [expected["files_without_survivors"], expected["mean_survivors_per_file"]],
        ),
        info(
            "A",
            "A10 time: reading JSONs / filters with RLE containment (s)",
            [round(timings["load_s"], 1), round(timings["filter_s"], 1)],
            detail="the legacy decode-based containment took hours",
        ),
    ]


# --------------------------------------------------------------------------- B


def manifest_counts(manifest: pd.DataFrame) -> dict[str, int]:
    """Counts the legacy loop printed: masks before the filter, after it, and crops kept.

    The legacy loop skipped an absent or unreadable photo without counting its masks, so the
    images with an `image_missing` row are left out here too.
    """
    absent = set(manifest.loc[manifest["reject_reason"] == "image_missing", "image_name"])
    seen = manifest[~manifest["image_name"].isin(absent)]
    return {
        "masks": len(seen),
        "after_filter": int(seen["stage1_reason"].isna().sum()),
        "crops": int(seen["keep"].sum()),
    }


def check_sample_counts(
    legacy: Mapping[str, int], new: Mapping[str, int], expected: Mapping[str, Any] = EXPECTED
) -> list[CheckResult]:
    """Check B, counts: masks before / after the filter and crops, for both implementations.

    B1 (required) compares coralid with the legacy code run live. B2 compares that live run
    with the numbers recorded in 2026-10: PASS when they match, INFO (never FAIL) when the
    Drive data has changed since.
    """
    keys = ("masks", "after_filter", "crops")
    recorded = [expected["sample_masks"], expected["sample_after_filter"], expected["sample_crops"]]
    new_counts, legacy_counts = [new[k] for k in keys], [legacy[k] for k in keys]
    differs = "" if legacy_counts == recorded else "the Drive data may have changed since 2026-10"
    return [
        equal(
            "B",
            "B1 masks / after filter / crops: coralid = live legacy run",
            new_counts,
            legacy_counts,
        ),
        CheckResult(
            "B",
            "B2 live legacy run reproduces the recorded numbers",
            PASS if legacy_counts == recorded else INFO,  # a difference is data drift, not a bug
            recorded,
            legacy_counts,
            differs,
        ),
    ]


def check_zero_shot_parity(
    comparison: PredictionComparison,
    min_agreement: float = EXPECTED["min_top1_agreement"],
    max_abs_diff: float = EXPECTED["max_abs_diff"],
) -> list[CheckResult]:
    """Check B, parity: identical keys, top-1 agreement and probability difference."""
    c = comparison
    agreement, diff = c.top1_agreement, c.max_abs_diff
    return [
        CheckResult(
            "B",
            "B3 identical (image_name, coral_id) keys",
            PASS if c.keys_equal else FAIL,
            "legacy rows == coralid rows",
            f"{c.n_a} vs {c.n_b} rows; only legacy {c.only_in_a}, only coralid {c.only_in_b}",
            f"{c.only_in_a_sample} {c.only_in_b_sample}" if not c.keys_equal else "",
        ),
        CheckResult(
            "B",
            "B4 top-1 agreement",
            PASS if agreement is not None and agreement >= min_agreement else FAIL,
            f">= {min_agreement}",
            agreement,
            f"{c.n_disagree} of {c.n_common} differ",
        ),
        CheckResult(
            "B",
            "B5 max absolute probability difference",
            PASS if diff is not None and diff <= max_abs_diff else FAIL,
            f"<= {max_abs_diff}",
            diff,
            f"mean {c.mean_abs_diff}; worst {c.worst}",
        ),
    ]


def check_against_drive_csv(comparison: PredictionComparison) -> list[CheckResult]:
    """Check B, informational: coralid (a) against the file saved on Drive (b)."""
    c = comparison
    return [
        info(
            "B",
            "B6 coralid vs Drive coral_weak_labels_..._best.csv (rewritten 2026-10-11)",
            {
                "rows_coralid_drive": [c.n_a, c.n_b],
                "keys_equal": c.keys_equal,
                "top1": c.top1_agreement,
                "max_abs_diff": c.max_abs_diff,
            },
            detail="informational: a mismatch here is not a failure",
        )
    ]


# --------------------------------------------------------------------------- C


def check_image_manifest(
    manifest: pd.DataFrame,
    unresolved_failures: Sequence[Mapping[str, Any]] = (),
    expected: Mapping[str, Any] = EXPECTED,
) -> list[CheckResult]:
    """Check C: the manifest is complete and keeps 14,224 masks."""
    reasons = manifest["reject_reason"].value_counts().to_dict()
    missing = manifest.loc[manifest["reject_reason"] == "image_missing", "image_name"].nunique()
    return [
        equal("C", "C1 masks in the manifest", len(manifest), expected["n_masks"]),
        equal("C", "C2 images that failed and are still unresolved", len(unresolved_failures), 0),
        equal("C", "C3 masks kept", int(manifest["keep"].sum()), expected["kept"]),
        info("C", "C4 rejected per reason", reasons),
        info("C", "C5 images missing or unreadable", int(missing)),
    ]


# --------------------------------------------------------------------------- D


def check_expert_confusion(
    report: EvaluationReport,
    expected_cells: Mapping[str, Mapping[str, int]],
    expected_status: Mapping[str, int],
    expected_correct: tuple[int, int, int],
) -> list[CheckResult]:
    """Check D: counts, accuracy and every confusion cell against the authoritative matrix.

    `expected_cells` is `{true: {predicted: count}}`; `expected_correct` is
    `(correct, in_set total, in_set + out_of_set total)`.
    """
    expected_pairs = {(t, p): n for t, row in expected_cells.items() for p, n in row.items()}
    actual_pairs = report.confusion_cells()
    results = [
        equal(
            "D",
            "D1 expert rows with a prediction",
            report.coverage["n_joined"],
            sum(expected_status.values()),
        ),
        equal("D", "D2 rows per status", report.status_counts_joined, dict(expected_status)),
        equal(
            "D",
            "D3 correct / in-set / in-set + out-of-set",
            [
                report.accuracy_in_set["correct"],
                report.accuracy_in_set["total"],
                report.accuracy_in_plus_out["total"],
            ],
            list(expected_correct),
        ),
    ]
    cells = sorted(set(expected_pairs) | set(actual_pairs))
    wrong = 0
    for true, predicted in cells:
        want, got = expected_pairs.get((true, predicted), 0), actual_pairs.get((true, predicted), 0)
        wrong += want != got
        cell = equal("D", f"cell {true} -> {predicted}", got, want)
        cell.kind = "cell"
        results.append(cell)
    results.insert(
        3,
        equal(
            "D",
            f"D4 confusion matrix cells equal ({len(cells)} compared)",
            wrong,
            0,
            f"{wrong} differ",
        ),
    )
    return results
