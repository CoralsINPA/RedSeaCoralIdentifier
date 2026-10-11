"""Check C of the parity notebook: the image manifest on all images.

The legacy V2 run kept 14,224 crops, but it cut the 221 portrait photos (1920x2560 frames) from
the wrong region, so its total cannot be a target. The check that holds is per orientation: on
landscape photos the keys the new manifest keeps must equal the keys V2 kept
(`v2_pipeline/v2_weak_labels_expanded.csv`: same filters, 24 genera, but the keep decision does
not depend on the label set). Portrait photos are reported as information.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from coralid.orientation import LANDSCAPE, PORTRAIT, add_orientation, image_orientations
from coralid.parity import EXPECTED, NOT_RUN, CheckResult, equal, info
from coralid.predictions import normalize_keys


def load_v2_keys(path: str | os.PathLike[str]) -> pd.DataFrame:
    """The `(image_name, coral_id)` keys of a legacy predictions CSV (`coral_id` as int)."""
    return normalize_keys(pd.read_csv(path, usecols=["image_name", "coral_id"]))


def _keyset(df: pd.DataFrame) -> set[tuple[str, int]]:
    return set(zip(df["image_name"], df["coral_id"], strict=True))


def compare_kept_with_v2(
    manifest: pd.DataFrame, v2_keys: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Kept keys of the manifest against V2's keys, for landscape and for portrait photos.

    Per orientation: `images`, `kept` (new manifest), `v2`, `both`, `only_new`, `only_v2` and up
    to five example uids of each difference. V2 rows on images the manifest does not know are
    counted under `unknown_images` in the result (outside the two orientations).
    """
    orientations = image_orientations(manifest)
    kept = add_orientation(manifest[manifest["keep"].astype(bool)], manifest)
    v2 = add_orientation(normalize_keys(v2_keys), manifest)
    out: dict[str, dict[str, Any]] = {}
    for name in (LANDSCAPE, PORTRAIT):
        mine = _keyset(kept[kept["orientation"] == name])
        theirs = _keyset(v2[v2["orientation"] == name])
        out[name] = {
            "images": int((orientations == name).sum()),
            "kept": len(mine),
            "v2": len(theirs),
            "both": len(mine & theirs),
            "only_new": len(mine - theirs),
            "only_v2": len(theirs - mine),
            "only_new_sample": [f"{n}#{c}" for n, c in sorted(mine - theirs)[:5]],
            "only_v2_sample": [f"{n}#{c}" for n, c in sorted(theirs - mine)[:5]],
        }
    out["unknown_images"] = {"v2_rows": int((v2["orientation"] == "unknown").sum())}
    return out


def check_image_manifest(
    manifest: pd.DataFrame,
    unresolved_failures: Sequence[Mapping[str, Any]] = (),
    v2_keys: pd.DataFrame | None = None,
    expected: Mapping[str, Any] = EXPECTED,
) -> list[CheckResult]:
    """Check C: the manifest is complete, and on landscape photos it keeps what V2 kept.

    C1 and C2 are completeness (44,211 masks, no unresolved failures). C3 compares the kept
    keys on landscape photos with V2's (`v2_keys`, from `load_v2_keys`; NOT RUN without it).
    C4 to C7 are information: portrait photos, the total against the recorded 16,080, the
    reasons for rejection per orientation, and missing photos.
    """
    tagged = add_orientation(manifest, manifest)
    portrait_kept = int(tagged.loc[tagged["orientation"] == PORTRAIT, "keep"].sum())
    reasons = manifest["reject_reason"].value_counts().to_dict()
    missing = manifest.loc[manifest["reject_reason"] == "image_missing", "image_name"].nunique()
    per_orientation = {
        name: tagged.loc[tagged["orientation"] == name, "reject_reason"]
        .fillna("kept")
        .value_counts()
        .to_dict()
        for name in (LANDSCAPE, PORTRAIT)
    }
    rows = [
        equal("C", "C1 masks in the manifest", len(manifest), expected["n_masks"]),
        equal("C", "C2 images that failed and are still unresolved", len(unresolved_failures), 0),
    ]
    if v2_keys is None:
        rows.append(
            CheckResult(
                "C",
                "C3 kept keys on landscape photos = V2's kept keys",
                NOT_RUN,
                detail="v2_pipeline/v2_weak_labels_expanded.csv was not found",
            )
        )
        v2_info: dict[str, Any] = {}
    else:
        v2_info = compare_kept_with_v2(manifest, v2_keys)
        land = v2_info[LANDSCAPE]
        rows.append(
            equal(
                "C",
                "C3 kept keys on landscape photos = V2's kept keys",
                [land["only_new"], land["only_v2"]],
                [0, 0],
                f"{land['kept']} kept, {land['v2']} in V2, {land['both']} in both; "
                f"only new {land['only_new_sample']}, only V2 {land['only_v2_sample']}",
            )
        )
    rows += [
        info(
            "C",
            "C4 portrait photos: kept by coralid vs V2",
            v2_info.get(PORTRAIT, {"kept": portrait_kept}),
            detail="informational: V2 cut the wrong region on portrait photos",
        ),
        info(
            "C",
            "C5 masks kept in total",
            int(manifest["keep"].sum()),
            expected["kept_recorded"],
            f"the legacy V2 total was {expected['kept']}, with portrait photos mis-cut",
        ),
        info("C", "C6 rejected per reason, per orientation", per_orientation, detail=str(reasons)),
        info("C", "C7 images missing or unreadable", int(missing)),
    ]
    return rows
