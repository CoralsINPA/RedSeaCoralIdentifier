import re

import cv2
import numpy as np
import pandas as pd
import pytest

from coralid.labeling import (
    NO,
    UNSURE,
    YES,
    ReviewIngest,
    answer_column,
    audit_sheet,
    export_review_sheet,
    ingest_review,
    normalize_answer,
    summarize_audit,
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


def test_ingest_reads_a_csv_and_the_old_audit_layout(tmp_path):
    """The legacy audit sheet has image_name and coral_id, no key column."""
    path = tmp_path / "audit.csv"
    pd.DataFrame(
        {
            "image_name": ["A.JPG", "B.JPG", "C.JPG"],
            "coral_id": [4, 5, 6],
            "confidence": [0.99, 0.9, 0.86],
            "is_true_millepora": ["yes", "no", ""],
            "actual_genus": ["", "Sinularia", ""],
            "notes": ["", "", ""],
        }
    ).to_csv(path, index=False)
    review = ingest_review(path, "is_true_millepora")
    assert review.table["key"].tolist() == ["A.JPG#4", "B.JPG#5", "C.JPG#6"]
    assert review.table["answer"].tolist() == [YES, NO, None]
    assert review.table["confidence"].tolist() == [0.99, 0.9, 0.86]  # other columns are kept
    assert review.problems.empty  # Sinularia is a known (soft coral) genus


def test_ingest_needs_the_answer_column_and_a_key():
    with pytest.raises(ValueError, match="no 'is_x' column"):
        ingest_review(pd.DataFrame({"key": ["a#1"]}), "is_x")
    with pytest.raises(ValueError, match="key"):
        ingest_review(pd.DataFrame({"is_x": ["yes"]}), "is_x")


def test_export_fill_ingest_round_trip(tmp_path):
    path = tmp_path / "sheet.csv"
    keys = [f"IMG_{i}.JPG#{i}" for i in range(4)]
    export_review_sheet(keys, path, "is_millepora")
    sheet = pd.read_csv(path, keep_default_na=False)
    sheet["is_millepora"] = ["yes", "no", "?", ""]
    sheet.loc[1, "actual_genus"] = "Porites"
    sheet.to_csv(path, index=False)
    review = ingest_review(path, "is_millepora")
    assert review.table["answer"].tolist() == [YES, NO, UNSURE, None]
    assert review.table.loc[1, "actual_genus"] == "Porites" and review.problems.empty


# ----------------------------------------------------------------------------- the audit


def _predictions():
    return pd.DataFrame(
        {
            "image_name": ["A.JPG", "A.JPG", "B.JPG", "B.JPG", "C.JPG"],
            "coral_id": [0, 1, 0, 1, 0],
            "bbox": [[1, 2, 3, 4]] * 5,
            "area": [30000] * 5,
            "predicted_genus": ["Millepora", "Porites", "Millepora", "Millepora", "Acropora"],
            "confidence": [0.9, 0.99, 0.95, 0.6, 0.8],
            "runner_up": ["Acropora"] * 5,
        }
    )


def test_audit_sheet_lists_the_genus_most_confident_first():
    sheet = audit_sheet(_predictions(), "Millepora")
    assert sheet["key"].tolist() == ["B.JPG#0", "A.JPG#0", "B.JPG#1"]
    assert list(sheet.columns) == [
        "key",
        "image_name",
        "coral_id",
        "bbox",
        "area",
        "confidence",
        "runner_up",
        "is_millepora",
        "actual_genus",
        "notes",
    ]
    assert (sheet[["is_millepora", "actual_genus", "notes"]] == "").all().all()
    assert audit_sheet(_predictions(), "Millepora", min_confidence=0.9)["key"].tolist() == [
        "B.JPG#0",
        "A.JPG#0",
    ]
    assert audit_sheet(_predictions(), "Porites", columns=("confidence",)).shape == (1, 5)
    assert audit_sheet(_predictions(), "Pavona").empty


def test_the_old_millepora_figures_as_a_range():
    """66 yes / 46 no / 17 unsure: 58.9% of the decided, 51-64% counting the unsure."""
    answers = [YES] * 66 + [NO] * 46 + [UNSURE] * 17 + [None] * 3
    genus = [None] * 66 + ["Echinopora"] * 30 + ["Sinularia"] * 16 + [None] * 20
    table = pd.DataFrame({"answer": pd.Series(answers, dtype=object), "actual_genus": genus})
    summary = summarize_audit(table, "Millepora")
    assert summary.counts == {YES: 66, NO: 46, UNSURE: 17, "pending": 3}
    assert summary.precision_decided == pytest.approx(66 / 112)
    assert summary.precision_unsure_wrong == pytest.approx(66 / 129)
    assert summary.precision_unsure_right == pytest.approx(83 / 129)
    low, high = summary.precision_decided_ci95
    assert 0.49 < low < 0.589 < high < 0.68
    assert summary.actual_genera == {"Echinopora": 30, "Sinularia": 16}
    assert summary.to_dict()["n_rows"] == 132


def test_summarize_audit_with_nothing_reviewed():
    review = ingest_review(pd.DataFrame({"key": ["a.jpg#1"], "is_millepora": [""]}), "is_millepora")
    assert isinstance(review, ReviewIngest)
    summary = summarize_audit(review, "Millepora")
    assert summary.precision_decided is None and summary.precision_unsure_right is None
    assert summary.counts["pending"] == 1 and summary.actual_genera == {}


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
