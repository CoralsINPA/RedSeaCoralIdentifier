"""The pass/fail logic of the Colab parity notebook (`notebooks/parity_check.ipynb`).

The notebook only runs things and prints. Every decision lives here, where it is unit-tested:
each `check_*` function turns measured numbers into a list of `CheckResult` rows for checks
A (geometric), B (zero-shot parity), C (image manifest) and D (expert set). Folding the rows
into a summary and saving it is in `coralid.parity_report`.

This module holds the row type, Check A and Check B. Check C is in `coralid.parity_manifest`
and Check D in `coralid.parity_expert`.

Checks B and C are orientation-aware. The legacy code squashed the 221 portrait photos
(1920x2560 frames) to 2560x1920 without moving the boxes, so its crops on them were wrong. The
refactor is correct there and the legacy output is not, so only landscape photos are held to
parity (exact match required); portrait photos are reported as information.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from coralid.masks import JsonAudit
from coralid.orientation import LANDSCAPE, PORTRAIT, split_by_orientation
from coralid.predictions import PredictionComparison, compare_predictions

PASS, FAIL, INFO, NOT_RUN = "PASS", "FAIL", "INFO", "NOT RUN"

# Numbers from the legacy runs (docs/agent_prompts/implement_organization_plan.md).
EXPECTED = {
    "n_json_files": 1714,
    "n_masks": 44211,
    "after_geometric": 16197,
    "kept": 14224,
    "files_without_survivors": 4,
    "mean_survivors_per_file": 9.4,
    "frame_sizes": {"2560x1920": 1493, "1920x2560": 221},  # landscape, portrait (2026-10-11)
    "kept_recorded": 16080,  # Check C on 2026-10-11 (the legacy count was 14,224)
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
    sizes = audit.as_dict()["frame_sizes"]
    portrait = sum(n for k, n in sizes.items() if int(k.split("x")[0]) < int(k.split("x")[1]))
    landscape = sum(n for k, n in sizes.items() if int(k.split("x")[0]) > int(k.split("x")[1]))
    return [
        equal("A", "A1 mask JSON files", audit.n_files, expected["n_json_files"]),
        equal("A", "A2 masks in total", audit.n_masks, expected["n_masks"]),
        equal("A", "A3 unreadable JSONs", len(audit.unreadable), 0, str(audit.unreadable[:3])),
        equal(
            "A",
            "A4 frame sizes (files per width x height)",
            sizes,
            dict(expected["frame_sizes"]),
            f"{landscape} landscape, {portrait} portrait photos",
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


def counts_by_orientation(table: pd.DataFrame, frames: pd.DataFrame) -> dict[str, int]:
    """Rows of `table` (one per crop) on landscape and on portrait photos."""
    return {name: len(part) for name, part in split_by_orientation(table, frames).items()}


def compare_by_orientation(
    legacy: pd.DataFrame, new: pd.DataFrame, frames: pd.DataFrame
) -> dict[str, PredictionComparison]:
    """`compare_predictions(legacy, new)` on the landscape photos and on the portrait photos.

    `frames` (the manifest of the sample) says which photo is which. Both tables are split by
    the same frames, so a crop that only one side has still lands in its photo's group.
    """
    legacy_parts = split_by_orientation(legacy, frames)
    new_parts = split_by_orientation(new, frames)
    return {
        name: compare_predictions(legacy_parts[name], new_parts[name])
        for name in (LANDSCAPE, PORTRAIT)
    }


def check_sample_counts(
    legacy: Mapping[str, int],
    new: Mapping[str, int],
    legacy_crops: Mapping[str, int],
    new_crops: Mapping[str, int],
    expected: Mapping[str, Any] = EXPECTED,
) -> list[CheckResult]:
    """Check B, counts: masks before / after the filter, and crops per orientation.

    `legacy` and `new` hold `masks`, `after_filter` and `crops` for the whole sample;
    `legacy_crops` and `new_crops` hold the crops per orientation. B1 and B2 compare coralid
    with the legacy code run live: the filters are blind to orientation, so B1 covers all
    photos, while crops are compared on landscape photos only (B2). B3 compares that live run
    with the numbers recorded in 2026-10: PASS when they match, INFO (never FAIL) when the Drive
    data has changed since. B4 reports the portrait crops, where the legacy count is wrong.
    """
    keys = ("masks", "after_filter", "crops")
    recorded = [expected["sample_masks"], expected["sample_after_filter"], expected["sample_crops"]]
    legacy_counts = [legacy[k] for k in keys]
    differs = "" if legacy_counts == recorded else "the Drive data may have changed since 2026-10"
    return [
        equal(
            "B",
            "B1 masks / after filter: coralid = live legacy run (all photos)",
            [new["masks"], new["after_filter"]],
            [legacy["masks"], legacy["after_filter"]],
        ),
        equal(
            "B",
            "B2 crops kept on landscape photos: coralid = live legacy run",
            new_crops.get(LANDSCAPE, 0),
            legacy_crops.get(LANDSCAPE, 0),
        ),
        CheckResult(
            "B",
            "B3 live legacy run reproduces the recorded numbers",
            PASS if legacy_counts == recorded else INFO,  # a difference is data drift, not a bug
            recorded,
            legacy_counts,
            differs,
        ),
        info(
            "B",
            "B4 crops kept on portrait photos: legacy vs coralid",
            [legacy_crops.get(PORTRAIT, 0), new_crops.get(PORTRAIT, 0)],
            detail="informational: the legacy code cut the wrong region on portrait photos",
        ),
    ]


def check_zero_shot_parity(
    comparisons: Mapping[str, PredictionComparison],
    min_agreement: float = EXPECTED["min_top1_agreement"],
    max_abs_diff: float = EXPECTED["max_abs_diff"],
    portrait_images: int | None = None,
) -> list[CheckResult]:
    """Check B, parity: landscape photos must match exactly; portrait photos are information.

    `comparisons` is `{orientation: PredictionComparison}` from `compare_by_orientation`
    (a = legacy, b = coralid). On landscape photos: identical `(image_name, coral_id)` keys
    (B5), top-1 agreement of at least `min_agreement` (B6) and a maximum absolute probability
    difference of at most `max_abs_diff` (B7). The portrait row (B8) never fails.
    """
    results = _landscape_parity(comparisons.get(LANDSCAPE), min_agreement, max_abs_diff)
    p = comparisons.get(PORTRAIT)
    if p is not None:
        results.append(
            info(
                "B",
                "B8 portrait photos: legacy vs coralid",
                {
                    "images": portrait_images,
                    "crops_legacy_coralid": [p.n_a, p.n_b],
                    "only_legacy": p.only_in_a,
                    "only_coralid": p.only_in_b,
                    "top1": p.top1_agreement,
                    "max_abs_diff": p.max_abs_diff,
                },
                detail="informational: the legacy crops on portrait photos were mis-cut",
            )
        )
    return results


def _landscape_parity(
    c: PredictionComparison | None, min_agreement: float, max_abs_diff: float
) -> list[CheckResult]:
    if c is None or c.n_a + c.n_b == 0:
        why = "no landscape photo in the sample"
        return [
            CheckResult("B", name, NOT_RUN, detail=why)
            for name in (
                "B5 identical (image_name, coral_id) keys on landscape photos",
                "B6 top-1 agreement on landscape photos",
                "B7 max absolute probability difference on landscape photos",
            )
        ]
    agreement, diff = c.top1_agreement, c.max_abs_diff
    return [
        CheckResult(
            "B",
            "B5 identical (image_name, coral_id) keys on landscape photos",
            PASS if c.keys_equal else FAIL,
            "legacy rows == coralid rows",
            f"{c.n_a} vs {c.n_b} rows; only legacy {c.only_in_a}, only coralid {c.only_in_b}",
            f"{c.only_in_a_sample} {c.only_in_b_sample}" if not c.keys_equal else "",
        ),
        CheckResult(
            "B",
            "B6 top-1 agreement on landscape photos",
            PASS if agreement is not None and agreement >= min_agreement else FAIL,
            f">= {min_agreement}",
            agreement,
            f"{c.n_disagree} of {c.n_common} differ",
        ),
        CheckResult(
            "B",
            "B7 max absolute probability difference on landscape photos",
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
            "B9 coralid vs Drive coral_weak_labels_..._best.csv (rewritten 2026-10-11)",
            {
                "rows_coralid_drive": [c.n_a, c.n_b],
                "keys_equal": c.keys_equal,
                "top1": c.top1_agreement,
                "max_abs_diff": c.max_abs_diff,
            },
            detail="informational: a mismatch here is not a failure",
        )
    ]
