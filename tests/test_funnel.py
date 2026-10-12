import pandas as pd
import pytest

from coralid.funnel import REASON_ORDER, filter_funnel


def manifest(rows):
    """Rows `(image_name, frame_w, frame_h, reject_reason)` as an image manifest."""
    df = pd.DataFrame(rows, columns=["image_name", "frame_w", "frame_h", "reject_reason"])
    df["keep"] = df["reject_reason"].isna()
    return df


LAND, PORT = (2560, 1920), (1920, 2560)


def make():
    landscape = ["area"] * 5 + ["aspect"] * 2 + ["containment"] * 3 + [None] * 4
    rows = [("L1.JPG", *LAND, reason) for reason in landscape]
    rows += [("P1.JPG", *PORT, r) for r in ["area", "area", "blurry", "not_visible", None, None]]
    return manifest(rows)


def test_the_funnel_counts_each_filter_in_the_order_they_act():
    funnel = filter_funnel(make())
    assert list(funnel.index) == [
        "all masks",
        "area",
        "aspect",
        "containment",
        "not_visible",
        "blurry",
        "kept",
    ]
    assert list(funnel["all_removed"]) == [20, 7, 2, 3, 1, 1, 0]
    assert list(funnel["all_remaining"]) == [20, 13, 11, 8, 7, 6, 6]
    assert list(funnel["landscape_remaining"]) == [14, 9, 7, 4, 4, 4, 4]
    assert list(funnel["portrait_remaining"]) == [6, 4, 4, 4, 3, 2, 2]


def test_the_groups_add_up_to_the_whole():
    funnel = filter_funnel(make())
    for stage in funnel.index:
        row = funnel.loc[stage]
        assert row["all_removed"] == row["landscape_removed"] + row["portrait_removed"]
        assert row["all_remaining"] == row["landscape_remaining"] + row["portrait_remaining"]


def test_only_the_reasons_that_occur_are_rows_and_unknown_ones_come_last():
    df = manifest([("A.JPG", *LAND, "area"), ("A.JPG", *LAND, "weird"), ("A.JPG", *LAND, None)])
    funnel = filter_funnel(df)
    assert list(funnel.index) == ["all masks", "area", "weird", "kept"]
    assert set(REASON_ORDER) >= {"area", "blurry"}


def test_a_manifest_without_photos_ends_at_the_geometric_filters():
    df = pd.DataFrame(
        {
            "image_name": ["A.JPG"] * 4,
            "frame_w": [2560] * 4,
            "frame_h": [1920] * 4,
            "stage1_reason": ["area", "containment", None, None],
        }
    )
    funnel = filter_funnel(df)
    assert list(funnel.index) == ["all masks", "area", "containment", "passed geometric filters"]
    assert list(funnel["all_remaining"]) == [4, 3, 2, 2]
    assert list(funnel["portrait_remaining"]) == [0, 0, 0, 0]  # a group with no photos is zeros


def test_a_manifest_whose_keep_disagrees_with_the_reasons_is_refused():
    df = make()
    df.loc[0, "keep"] = True  # kept, yet it has a reject reason
    with pytest.raises(ValueError, match="disagree"):
        filter_funnel(df)


def test_missing_columns_are_named():
    with pytest.raises(ValueError, match="frame_w"):
        filter_funnel(make().drop(columns="frame_w"))
