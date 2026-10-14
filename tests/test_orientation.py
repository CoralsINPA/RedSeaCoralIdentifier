import pandas as pd
import pytest

from coralid.orientation import (
    LANDSCAPE,
    PORTRAIT,
    SQUARE,
    UNKNOWN,
    add_orientation,
    by_orientation,
    image_orientations,
    orientation_of,
    split_by_orientation,
)


def _frames():
    """A manifest-like table: two masks per photo."""
    rows = []
    for name, w, h in [("L1.JPG", 2560, 1920), ("P1.JPG", 1920, 2560), ("L2.JPG", 2560, 1706)]:
        rows += [{"image_name": name, "coral_id": i, "frame_w": w, "frame_h": h} for i in range(2)]
    return pd.DataFrame(rows)


def test_orientation_of():
    assert orientation_of(2560, 1920) == LANDSCAPE
    assert orientation_of(2560, 1706) == LANDSCAPE  # 3:2 photos are landscape too
    assert orientation_of(1920, 2560) == PORTRAIT
    assert orientation_of(100, 100) == SQUARE


def test_image_orientations_gives_one_entry_per_image():
    out = image_orientations(_frames())
    assert out.to_dict() == {"L1.JPG": LANDSCAPE, "P1.JPG": PORTRAIT, "L2.JPG": LANDSCAPE}
    assert out.index.name == "image_name"


def test_image_orientations_rejects_a_photo_with_two_frames_and_missing_columns():
    clash = pd.DataFrame(
        {"image_name": ["A", "A"], "frame_w": [2560, 1920], "frame_h": [1920, 2560]}
    )
    with pytest.raises(ValueError, match="more than one frame"):
        image_orientations(clash)
    with pytest.raises(ValueError, match="frame_h"):
        image_orientations(pd.DataFrame({"image_name": ["A"], "frame_w": [1]}))


def test_add_orientation_keeps_rows_and_marks_unknown_images():
    table = pd.DataFrame({"image_name": ["P1.JPG", "L1.JPG", "X.JPG"], "value": [1, 2, 3]})
    out = add_orientation(table, _frames())
    assert out["orientation"].tolist() == [PORTRAIT, LANDSCAPE, UNKNOWN]
    assert out["value"].tolist() == [1, 2, 3] and "orientation" not in table.columns
    with pytest.raises(ValueError, match="image_name"):
        add_orientation(pd.DataFrame({"x": [1]}), _frames())


def test_split_by_orientation_always_has_both_keys():
    table = pd.DataFrame({"image_name": ["L1.JPG", "L1.JPG", "L2.JPG"], "value": [1, 2, 3]})
    parts = split_by_orientation(table, _frames())
    assert list(parts) == [LANDSCAPE, PORTRAIT]
    assert parts[LANDSCAPE]["value"].tolist() == [1, 2, 3]
    assert parts[PORTRAIT].empty and list(parts[PORTRAIT].columns) == ["image_name", "value"]
    extra = split_by_orientation(pd.DataFrame({"image_name": ["X.JPG"]}), _frames())
    assert list(extra) == [LANDSCAPE, PORTRAIT, UNKNOWN] and len(extra[UNKNOWN]) == 1


def test_by_orientation_runs_a_metric_on_everything_and_each_orientation():
    table = pd.DataFrame(
        {"image_name": ["L1.JPG", "P1.JPG", "P1.JPG", "L2.JPG"], "ok": [True, False, True, True]}
    )
    result = by_orientation(table, _frames(), lambda df: int(df["ok"].sum()))
    assert result == {"all": 3, LANDSCAPE: 2, PORTRAIT: 1}
    only = by_orientation(table, _frames(), len, include_all=False)
    assert only == {LANDSCAPE: 2, PORTRAIT: 2}
    empty = by_orientation(table.iloc[:1], _frames(), len)
    assert empty == {"all": 1, LANDSCAPE: 1, PORTRAIT: 0}
