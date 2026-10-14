"""Check C of the parity notebook: the image manifest on all images.

The legacy run kept 14,224 crops, but it cut the 221 portrait photos (1920x2560 frames) from the
wrong region, so its total cannot be a target. Check C is therefore about completeness (every mask
of the 1,714 JSONs is in the manifest and no photo failed) and reports the kept counts, per
orientation, as information.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from coralid.orientation import LANDSCAPE, PORTRAIT, add_orientation
from coralid.parity import EXPECTED, CheckResult, equal, info


def check_image_manifest(
    manifest: pd.DataFrame,
    unresolved_failures: Sequence[Mapping[str, Any]] = (),
    expected: Mapping[str, Any] = EXPECTED,
) -> list[CheckResult]:
    """Check C: the manifest is complete.

    C1 and C2 are completeness (44,211 masks, no unresolved failures). C3 to C6 are
    information: the crops kept on portrait photos, the total against the recorded 16,080, the
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
    return [
        equal("C", "C1 masks in the manifest", len(manifest), expected["n_masks"]),
        equal("C", "C2 images that failed and are still unresolved", len(unresolved_failures), 0),
        info(
            "C",
            "C3 portrait photos: crops kept by coralid",
            {"kept": portrait_kept},
            detail="informational: the legacy code cut the wrong region on portrait photos",
        ),
        info(
            "C",
            "C4 masks kept in total",
            int(manifest["keep"].sum()),
            expected["kept_recorded"],
            f"the legacy total was {expected['kept']}, counted with portrait photos mis-cut",
        ),
        info("C", "C5 rejected per reason, per orientation", per_orientation, detail=str(reasons)),
        info("C", "C6 images missing or unreadable", int(missing)),
    ]
