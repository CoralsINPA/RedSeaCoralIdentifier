"""The filter funnel: how many masks each filter removes, by photo orientation.

The manifest records one reason per rejected mask, and the filters act in a fixed order (area,
aspect ratio, containment, then what needs the photo: missing photo, visibility, blur), so the
reasons add up to a funnel: from all masks down to the kept crops. `filter_funnel` turns a
manifest into that table for all photos and for landscape and portrait photos apart.
"""

from __future__ import annotations

import pandas as pd

from coralid.orientation import ORIENTATIONS, add_orientation

REASON_ORDER = ("area", "aspect", "containment", "image_missing", "not_visible", "blurry")
GROUPS = ("all", *ORIENTATIONS)


def filter_funnel(manifest: pd.DataFrame) -> pd.DataFrame:
    """The masks left after each filter, for all photos and by orientation.

    One row per stage: `all masks` (the start), then each reject reason that occurs, in the order
    the filters act, then `kept` (`passed geometric filters` for a manifest made without photos).
    For each group (`all`, `landscape`, `portrait`) two columns: `<group>_removed` (masks that
    stage removed; the start row shows the total) and `<group>_remaining` (masks left after it).
    The reasons are exclusive: a mask counts under the first filter that removed it.
    """
    image_level = "keep" in manifest.columns
    column = "reject_reason" if image_level else "stage1_reason"
    for needed in ("image_name", "frame_w", "frame_h", column):
        if needed not in manifest.columns:
            raise ValueError(f"the manifest needs a '{needed}' column")
    tagged = add_orientation(manifest, manifest)
    seen = [r for r in manifest[column].dropna().unique()]
    reasons = [r for r in REASON_ORDER if r in seen] + sorted(set(seen) - set(REASON_ORDER))
    last = "kept" if image_level else "passed geometric filters"
    stages = ["all masks", *reasons, last]
    table: dict[str, list[int]] = {}
    for group in GROUPS:
        part = tagged if group == "all" else tagged[tagged["orientation"] == group]
        counts = part[column].value_counts()
        remaining = len(part)
        removed_col, remaining_col = [remaining], [remaining]
        for reason in reasons:
            removed = int(counts.get(reason, 0))
            remaining -= removed
            removed_col.append(removed)
            remaining_col.append(remaining)
        removed_col.append(0)
        remaining_col.append(remaining)
        table[f"{group}_removed"] = removed_col
        table[f"{group}_remaining"] = remaining_col
    out = pd.DataFrame(table, index=pd.Index(stages, name="stage"))
    if image_level:  # the last row must agree with the manifest's own decision
        for group in GROUPS:
            part = tagged if group == "all" else tagged[tagged["orientation"] == group]
            if int(out.loc[last, f"{group}_remaining"]) != int(part["keep"].sum()):
                raise ValueError("reject_reason and keep disagree in the manifest")
    return out
