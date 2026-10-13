import re

import cv2
import numpy as np
import pandas as pd
import pytest

from coralid.labeling import (
    NO,
    UNSURE,
    YES,
    answer_column,
    export_review_sheet,
    ingest_review,
    normalize_answer,
)
from coralid.masks import AnnotationStore
from tests.helpers import make_annotation, rect_mask, write_json_dir

# ----------------------------------------------------------------------------- answers


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("yes", YES),
        (" Yes ", YES),
        ("Y", YES),
        (True, YES),
        ("no", NO),
        ("N", NO),
        (False, NO),
        ("?", UNSURE),
        ("Unsure", UNSURE),
        ("unknown", UNSURE),
        ("", None),
        ("  ", None),
        (None, None),
        (float("nan"), None),
        (pd.NA, None),
    ],
)
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


def test_normalize_answer_rejects_anything_else():
    with pytest.raises(ValueError, match="not an answer"):
        normalize_answer("probably")


def test_answer_column_names_the_genus():
    assert answer_column("Millepora") == "is_millepora"
    assert answer_column("Dipsastraea") == "is_dipsastraea"


# ----------------------------------------------------------------------------- export


def test_export_review_sheet_is_blank_and_never_overwrites(tmp_path):
    path = tmp_path / "sub" / "sheet.csv"
    keys = ["IMG_1.JPG#3", "IMG_2.JPG#0"]
    export_review_sheet(keys, path, "is_millepora")
    sheet = pd.read_csv(path, keep_default_na=False)
    assert list(sheet.columns) == ["key", "is_millepora", "actual_genus", "notes"]
    assert sheet["key"].tolist() == keys and (sheet.iloc[:, 1:] == "").all().all()
    with pytest.raises(FileExistsError):
        export_review_sheet(keys, path, "is_millepora")
    assert pd.read_csv(path)["key"].tolist() == keys  # untouched
    with pytest.raises(ValueError, match="unique"):
        export_review_sheet(["a#1", "a#1"], tmp_path / "dup.csv", "is_x")


def test_export_review_sheet_can_show_extra_columns(tmp_path):
    extra = pd.DataFrame({"image_name": ["A.JPG", "B.JPG"], "coral_id": [1, 2]})
    export_review_sheet(["A.JPG#1", "B.JPG#2"], tmp_path / "s.csv", "is_x", extra=extra)
    assert list(pd.read_csv(tmp_path / "s.csv").columns) == [
        "key",
        "image_name",
        "coral_id",
        "is_x",
        "actual_genus",
        "notes",
    ]
    with pytest.raises(ValueError, match="extra has"):
        export_review_sheet(["a#1"], tmp_path / "t.csv", "is_x", extra=extra)


# ----------------------------------------------------------------------------- ingest


def _completed():
    return pd.DataFrame(
        {
            "key": [f"IMG_{i}.JPG#{i}" for i in range(8)] + ["IMG_0.JPG#0"],
            "is_millepora": ["yes", "no", "no", "?", "", "probably", "no", "YES", "no"],
            "actual_genus": ["", "echinopora", "", "", "", "", "Echinopra", "", "Porites"],
            "notes": ["", "x", "", "", "", "", "", "", "second row of a duplicate"],
        }
    )


def test_ingest_normalizes_answers_and_genus_spelling():
    review = ingest_review(_completed(), "is_millepora")
    t = review.table.set_index("key")
    assert t["answer"].tolist() == [YES, NO, NO, UNSURE, None, None, NO, YES]  # 8 rows, dup gone
    assert t.loc["IMG_1.JPG#1", "actual_genus"] == "Echinopora"  # canonical spelling
    assert t.loc["IMG_6.JPG#6", "actual_genus"] == "Echinopra"  # a typo is kept as written
    assert (
        t.loc["IMG_0.JPG#0", "image_name"] == "IMG_0.JPG" and t.loc["IMG_0.JPG#0", "coral_id"] == 0
    )
    assert review.counts() == {YES: 2, NO: 3, UNSURE: 1, "pending": 2}


def test_ingest_lists_every_problem_without_raising():
    review = ingest_review(_completed(), "is_millepora")
    found = {(p.key, p.problem) for p in review.problems.itertuples()}
    assert found == {
        ("IMG_0.JPG#0", "duplicate_key"),
        ("IMG_2.JPG#2", "missing_actual_genus"),  # a "no" must say what it is
        ("IMG_5.JPG#5", "invalid_answer"),
        ("IMG_6.JPG#6", "unknown_genus"),
    }
    assert list(review.problems.columns) == ["key", "problem", "detail"]
    clean = ingest_review(_completed().iloc[[0, 1]], "is_millepora")
    assert clean.problems.empty


def test_ingest_reads_a_csv_by_key_and_keeps_the_other_columns(tmp_path):
    path = tmp_path / "sheet.csv"
    pd.DataFrame(
        {
            "key": ["A.JPG#4", "B.JPG#5", "C.JPG#6"],
            "confidence": [0.99, 0.9, 0.86],
            "is_millepora": ["yes", "no", ""],
            "actual_genus": ["", "Sinularia", ""],
            "notes": ["", "", ""],
        }
    ).to_csv(path, index=False)
    review = ingest_review(path, "is_millepora")
    assert review.table["key"].tolist() == ["A.JPG#4", "B.JPG#5", "C.JPG#6"]
    assert review.table["answer"].tolist() == [YES, NO, None]
    assert review.table["coral_id"].tolist() == [4, 5, 6]
    assert review.table["confidence"].tolist() == [0.99, 0.9, 0.86]  # other columns are kept
    assert review.problems.empty  # Sinularia is a known (soft coral) genus


def test_a_sheet_without_a_key_column_is_refused_not_matched_on_coral_id():
    sheet = pd.DataFrame({"image_name": ["A.JPG"], "coral_id": [4], "is_millepora": ["yes"]})
    with pytest.raises(ValueError, match="needs a 'key' column"):
        ingest_review(sheet, "is_millepora")


# ----------------------------------------------------------------------------- crop grids


@pytest.fixture
def store_and_images(tmp_path):
    """Two photos in a 200x150 mask frame (one is 2x that size) with two masks each."""
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.RandomState(1)
    for name, scale in (("A.JPG", 2), ("B.JPG", 1)):
        photo = rng.randint(0, 256, size=(150 * scale, 200 * scale, 3), dtype=np.uint8)
        assert cv2.imwrite(str(images / name), photo)
    shape = (150, 200)
    annos = [
        make_annotation(rect_mask(shape, 10, 80, 10, 90), 0),
        make_annotation(rect_mask(shape, 90, 140, 100, 190), 1),
    ]
    json_dir = write_json_dir(
        tmp_path / "json", [("A.JPG", 200, 150, annos, None), ("B.JPG", 200, 150, annos, None)]
    )
    return AnnotationStore(json_dir), images


def _items():
    return pd.DataFrame(
        {
            "image_name": ["B.JPG", "A.JPG", "A.JPG", "Z.JPG", "A.JPG"],
            "coral_id": [1, 0, 1, 0, 99],
            "caption": ["#1", "#2", "#3", "#4 photo missing", "#5 no such mask"],
        }
    )


def test_cut_crops_follows_the_row_order_and_loads_in_the_mask_frame(store_and_images):
    from coralid.config import CropConfig

    store, images = store_and_images
    crops = cut_crops_for_test(store, images, CropConfig(padding_pct=0.0, min_visible=10))
    assert [c is None for c in crops] == [False, False, False, True, True]
    assert crops[0].shape[:2] == (50, 90) and crops[1].shape[:2] == (70, 80)  # bbox sizes
    assert crops[2].shape[:2] == (50, 90)
    assert store.loads == 2  # A and B are each read once (Z has no JSON)


def cut_crops_for_test(store, images, crops):
    from coralid.viz import cut_crops

    return cut_crops(_items(), store, images, crops, max_side=512)


def test_cut_crops_shrinks_large_crops(store_and_images):
    from coralid.config import CropConfig
    from coralid.viz import cut_crops

    store, images = store_and_images
    crop = cut_crops(
        _items().iloc[[1]], store, images, CropConfig(padding_pct=0.0, min_visible=10), max_side=40
    )[0]
    assert max(crop.shape[:2]) == 40


def test_crop_grid_pdf_writes_pages_and_never_overwrites(store_and_images, tmp_path):
    pytest.importorskip("matplotlib")
    from coralid.viz import crop_grid_pdf

    store, images = store_and_images
    items = pd.concat([_items()] * 3, ignore_index=True)  # 15 items: two pages of 12
    path = crop_grid_pdf(items, tmp_path / "out" / "grid.pdf", store, images, title="Blind review")
    data = path.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(re.findall(rb"/Type\s*/Page(?![a-z])", data)) == 2
    with pytest.raises(FileExistsError):
        crop_grid_pdf(items, path, store, images)
    assert not list(path.parent.glob(".*tmp*"))  # no temporary file left behind
