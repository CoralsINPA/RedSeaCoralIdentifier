import json

import numpy as np
import pandas as pd
import pytest

from coralid.evaluation import (
    IN_SET,
    MISSING,
    OUT_OF_SET,
    UNIDENTIFIABLE,
    compare_predictions,
    confusion_cells,
    distribution_vs_ground_truth,
    evaluate,
    expert_status,
    load_expert_set,
    load_legacy_predictions,
    wilson_interval,
)
from coralid.genera import load_genera
from tests.assaf_matrix import ASSAF, ASSAF_CSV, ASSAF_PRECISION, ASSAF_RECALL

HARD20 = load_genera().genera("hard20")

# ----------------------------------------------------------------------------- status


@pytest.mark.parametrize(
    ("raw", "status", "genus"),
    [
        ("Acropora", IN_SET, "Acropora"),
        ("  acropora ", IN_SET, "Acropora"),
        ("PORITES", IN_SET, "Porites"),
        ("Cladiella", OUT_OF_SET, "Cladiella"),
        ("sinularia", OUT_OF_SET, "Sinularia"),
        ("Sarcophyton", OUT_OF_SET, "Sarcophyton"),
        ("Xenia", OUT_OF_SET, "Xenia"),  # any other name is a genus the model cannot predict
        ("unidentifiable", UNIDENTIFIABLE, None),
        ("Unidentifiable", UNIDENTIFIABLE, None),
        ("?", UNIDENTIFIABLE, None),
        ("MISSING_LABEL", MISSING, None),
        ("", MISSING, None),
        (None, MISSING, None),
        (float("nan"), MISSING, None),
        (pd.NA, MISSING, None),
    ],
)
def test_expert_status(raw, status, genus):
    assert expert_status(raw, HARD20) == (status, genus)


@pytest.mark.parametrize(
    ("raw", "is_correct", "status"),
    [
        (None, "unidentifiable", UNIDENTIFIABLE),  # the real file: the word sits in is_correct
        (float("nan"), "Unidentifiable", UNIDENTIFIABLE),
        (pd.NA, "MISSING_LABEL", MISSING),
        (None, "True", MISSING),  # a plain flag says nothing about the genus
        (None, "False", MISSING),
        (None, None, MISSING),
        ("", pd.NA, MISSING),
        ("Acropora", "unidentifiable", IN_SET),  # a genus in expert_genus always wins
        ("unidentifiable", "True", UNIDENTIFIABLE),
    ],
)
def test_expert_status_falls_back_to_is_correct_when_the_genus_is_empty(raw, is_correct, status):
    assert expert_status(raw, HARD20, is_correct=is_correct)[0] == status


def test_status_depends_on_the_label_set():
    assert expert_status("Pavona", ["Acropora", "Porites"]) == (OUT_OF_SET, "Pavona")
    assert expert_status("pavona", HARD20 + ["Pavona"])[0] == IN_SET


# --------------------------------------------------------------------------- the file


def _expert_csv(tmp_path, rows):
    path = tmp_path / "expert.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_expert_set_normalizes_every_status(tmp_path):
    path = _expert_csv(
        tmp_path,
        {
            "image_name": ["A.JPG", "A.JPG", "B.JPG", "B.JPG", "C.JPG", "C.JPG", "D.JPG"],
            "coral_id": [0, 1, 0, 1, 0, 1, 2.0],  # a float id arrives as a float column
            "expert_genus": [
                "Acropora",
                "dipsastraea",
                "Cladiella",
                "unidentifiable",
                "MISSING_LABEL",
                None,
                "Litophyton",
            ],
            "review_source": ["r1"] * 7,
        },
    )
    out = load_expert_set(path)
    assert list(out.columns) == [
        "image_name",
        "coral_id",
        "expert_raw",
        "expert_genus",
        "status",
        "review_source",
    ]
    assert (
        out["coral_id"].tolist() == [0, 1, 0, 1, 0, 1, 2]
        and out["coral_id"].map(type).eq(int).all()
    )
    assert out["status"].tolist() == [
        IN_SET,
        IN_SET,
        OUT_OF_SET,
        UNIDENTIFIABLE,
        MISSING,
        MISSING,
        OUT_OF_SET,
    ]
    genus = out["expert_genus"]
    assert genus[[0, 1, 2, 6]].tolist() == ["Acropora", "Dipsastraea", "Cladiella", "Litophyton"]
    assert genus[[3, 4, 5]].isna().all()  # no genus for unidentifiable / missing
    assert out["expert_raw"].tolist()[:2] == ["Acropora", "dipsastraea"]


def test_load_expert_set_reads_the_status_from_is_correct(tmp_path):
    """The real file leaves expert_genus empty and puts the word in is_correct."""
    path = _expert_csv(
        tmp_path,
        {
            "image_name": ["A.JPG"] * 6,
            "coral_id": [0, 1, 2, 3, 4, 5],
            "expert_genus": ["Acropora", "Sinularia", None, None, None, None],
            "is_correct": [True, False, "unidentifiable", "MISSING_LABEL", False, None],
        },
    )
    out = load_expert_set(path)
    assert out["status"].tolist() == [IN_SET, OUT_OF_SET, UNIDENTIFIABLE, MISSING, MISSING, MISSING]
    assert out["expert_raw"].tolist() == [
        "Acropora",
        "Sinularia",
        "unidentifiable",
        "MISSING_LABEL",
        None,
        None,
    ]
    assert out["expert_genus"].isna().tolist() == [False, False, True, True, True, True]


def test_load_expert_set_rejects_bad_files(tmp_path):
    dup = _expert_csv(
        tmp_path, {"image_name": ["A", "A"], "coral_id": [1, 1], "expert_genus": ["x", "y"]}
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_expert_set(dup)
    nocol = _expert_csv(tmp_path, {"image_name": ["A"], "coral_id": [1]})
    with pytest.raises(ValueError, match="expert_genus"):
        load_expert_set(nocol)


# ------------------------------------------------------------------------- evaluate


def _frames():
    """A small expert set with every status, and predictions for all but one of its corals."""
    expert = pd.DataFrame(
        {
            "image_name": ["A", "A", "A", "B", "B", "B", "C", "C"],
            "coral_id": [0, 1, 2, 0, 1, 2, 0, 1],
            "expert_genus": [
                "Acropora",
                "Acropora",
                "Porites",
                "Cladiella",
                "unidentifiable",
                "MISSING_LABEL",
                "Porites",
                "Millepora",  # this one has no prediction
            ],
        }
    )
    predictions = pd.DataFrame(
        {
            "image_name": ["A", "A", "A", "B", "B", "B", "C", "Z"],
            "coral_id": ["0", 1.0, 2, 0, 1, 2, 0, 9],  # str / float ids are cast to int
            "predicted_genus": [
                "Acropora",
                "Porites",
                "Porites",
                "Acropora",
                "Acropora",
                "Acropora",
                "Pavona",
                "Acropora",
            ],
        }
    )
    return predictions, expert


def test_evaluate_counts_coverage_statuses_and_accuracy():
    predictions, expert = _frames()
    report = evaluate(predictions, expert, "hard20")
    assert report.coverage == {
        "n_expert": 8,
        "n_predictions": 8,
        "n_joined": 7,
        "n_expert_without_prediction": 1,
        "n_predictions_without_expert": 1,
    }
    assert report.status_counts == {IN_SET: 5, OUT_OF_SET: 1, UNIDENTIFIABLE: 1, MISSING: 1}
    assert report.status_counts_joined == {IN_SET: 4, OUT_OF_SET: 1, UNIDENTIFIABLE: 1, MISSING: 1}
    # in set (joined): Acropora ok, Acropora->Porites, Porites ok, Porites->Pavona = 2 / 4
    assert report.accuracy_in_set["correct"] == 2 and report.accuracy_in_set["total"] == 4
    assert report.accuracy_in_set["accuracy"] == 0.5
    lo, hi = report.accuracy_in_set["ci95"]
    assert 0.15 < lo < 0.5 < hi < 0.85
    # in + out: the Cladiella crop is always wrong -> 2 / 5
    assert report.accuracy_in_plus_out["correct"] == 2 and report.accuracy_in_plus_out["total"] == 5
    assert report.accuracy_in_plus_out["accuracy"] == pytest.approx(0.4)


def test_evaluate_confusion_per_genus_and_top_pairs():
    predictions, expert = _frames()
    report = evaluate(predictions, expert, "hard20")
    assert report.confusion_cells() == {
        ("Acropora", "Acropora"): 1,
        ("Acropora", "Porites"): 1,
        ("Porites", "Porites"): 1,
        ("Porites", "Pavona"): 1,
        ("Cladiella", "Acropora"): 1,
    }
    cm = report.confusion
    assert list(cm.index) == [*HARD20, "Cladiella"] == list(cm.columns)  # square, extras last
    assert cm.index.name == "true" and cm.columns.name == "predicted"
    assert cm.to_numpy().sum() == 5
    pg = report.per_genus
    assert pg.loc["Acropora", ["support", "predicted", "tp"]].tolist() == [2, 2, 1]
    assert pg.loc["Acropora", "recall"] == 0.5 and pg.loc["Acropora", "precision"] == 0.5
    assert pg.loc["Porites", "recall"] == 0.5 and pg.loc["Porites", "precision"] == 0.5
    assert pg.loc["Cladiella", "recall"] == 0.0 and np.isnan(pg.loc["Cladiella", "precision"])
    assert np.isnan(pg.loc["Millepora", "recall"]) and np.isnan(pg.loc["Millepora", "precision"])
    assert pg.loc["Pavona", "precision"] == 0.0
    top = report.top_confusions
    assert list(top.columns) == ["true", "predicted", "count"]
    assert set(map(tuple, top[["true", "predicted"]].to_numpy())) == {
        ("Acropora", "Porites"),
        ("Porites", "Pavona"),
        ("Cladiella", "Acropora"),
    }
    json.dumps(report.to_dict())  # the summary is JSON-friendly


def test_evaluate_with_nothing_to_score():
    expert = pd.DataFrame(
        {"image_name": ["A"], "coral_id": [0], "expert_genus": ["unidentifiable"]}
    )
    predictions = pd.DataFrame(
        {"image_name": ["A"], "coral_id": [0], "predicted_genus": ["Porites"]}
    )
    report = evaluate(predictions, expert, HARD20)
    assert report.accuracy_in_set == {"correct": 0, "total": 0, "accuracy": None, "ci95": None}
    assert report.confusion.to_numpy().sum() == 0 and report.top_confusions.empty
    assert report.status_counts[UNIDENTIFIABLE] == 1


def test_evaluate_rejects_duplicate_keys():
    predictions, expert = _frames()
    with pytest.raises(ValueError, match="predictions have duplicate"):
        evaluate(pd.concat([predictions, predictions.head(1)]), expert)
    with pytest.raises(ValueError, match="expert set has duplicate"):
        evaluate(predictions, pd.concat([expert, expert.head(1)]))


def test_wilson_interval():
    assert wilson_interval(0, 0) is None
    lo, hi = wilson_interval(39, 57)  # Assaf's in-set result, 68.4%
    assert (round(lo, 3), round(hi, 3)) == (0.555, 0.790)  # the interval in the memory notes
    assert wilson_interval(5, 5)[1] == 1.0 and wilson_interval(0, 5)[0] == 0.0


# --------------------------------------- Assaf's review, rebuilt from the authoritative matrix


def _assaf_csv(tmp_path):
    """A file shaped like assaf_expert_test_set.csv that holds exactly the authoritative matrix:
    62 identifiable crops, 16 unidentifiable ones (predicted as anything)."""
    rows = []
    for true, row in ASSAF.items():
        for predicted, n in row.items():
            rows += [(true, predicted)] * n
    rows += [("unidentifiable", HARD20[i % 20]) for i in range(16)]
    records = []
    for i, (true, predicted) in enumerate(rows):
        probs = np.full(20, 0.01)
        probs[HARD20.index(predicted)] = 0.81
        records.append(
            {
                "image_name": f"IMG_{i // 3}.JPG",
                "coral_id": i % 3,
                "bbox": f"[{i}.0, 10.0, 200.0, 150.5]",
                "area": 30000 + i,
                **dict(zip(HARD20, probs, strict=True)),
                "predicted_genus": predicted,
                "confidence": 0.81,
                "expert_genus": true,
                "is_correct": true == predicted,
                "review_source": "assaf",
            }
        )
    path = tmp_path / "assaf_expert_test_set.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def test_the_real_review_pipeline_reproduces_the_authoritative_matrix(tmp_path):
    """The Check D of the parity notebook, on a file built to match the matrix."""
    path = _assaf_csv(tmp_path)
    predictions = load_legacy_predictions(path)
    expert = load_expert_set(path)
    assert len(predictions) == len(expert) == 78
    report = evaluate(predictions, expert, "hard20")

    assert report.status_counts == {IN_SET: 57, OUT_OF_SET: 5, UNIDENTIFIABLE: 16, MISSING: 0}
    assert report.coverage["n_joined"] == 78
    expected_cells = {(t, p): n for t, row in ASSAF.items() for p, n in row.items()}
    assert report.confusion_cells() == expected_cells  # cell for cell
    assert (report.accuracy_in_set["correct"], report.accuracy_in_set["total"]) == (39, 57)
    assert report.accuracy_in_set["accuracy"] == pytest.approx(0.684, abs=5e-4)
    assert (report.accuracy_in_plus_out["correct"], report.accuracy_in_plus_out["total"]) == (
        39,
        62,
    )
    assert report.accuracy_in_plus_out["accuracy"] == pytest.approx(0.629, abs=5e-4)
    pg = report.per_genus
    for genus, percent in ASSAF_RECALL.items():
        assert round(100 * pg.loc[genus, "recall"]) == percent, genus
    for genus, percent in ASSAF_PRECISION.items():
        assert round(100 * pg.loc[genus, "precision"]) == percent, genus
    assert np.isnan(pg.loc["Lobophyllia", "recall"])  # no true Lobophyllia in the set
    # the most common confusion: Dipsastraea -> Favia and -> Leptastrea (2 each)
    assert set(report.top_confusions.head(2)["true"]) == {"Dipsastraea"}
    assert report.top_confusions.head(2)["count"].tolist() == [2, 2]


def _real_shaped_csv(tmp_path):
    """A file shaped like the real assaf_expert_test_set.csv: 77 rows, and for the 18 rows
    without a genus an empty expert_genus with the word in is_correct (16 + 2)."""
    rows = [(t, p) for t, row in ASSAF_CSV.items() for p, n in row.items() for _ in range(n)]
    rows += [("unidentifiable", HARD20[i % 20]) for i in range(16)]
    rows += [("MISSING_LABEL", HARD20[i % 20]) for i in range(2)]
    records = []
    for i, (true, predicted) in enumerate(rows):
        marker = true in ("unidentifiable", "MISSING_LABEL")
        records.append(
            {
                "image_name": f"IMG_{i // 3}.JPG",
                "coral_id": i % 3,
                "bbox": f"[{i}.0, 10.0, 200.0, 150.5]",
                "area": 30000 + i,
                **{g: (0.81 if g == predicted else 0.01) for g in HARD20},
                "predicted_genus": predicted,
                "confidence": 0.81,
                "expert_genus": None if marker else true,
                "is_correct": true if marker else true == predicted,
                "review_source": "assaf",
            }
        )
    path = tmp_path / "assaf_expert_test_set.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def test_the_real_file_shape_gives_77_rows_54_in_set_and_36_correct(tmp_path):
    path = _real_shaped_csv(tmp_path)
    report = evaluate(load_legacy_predictions(path), load_expert_set(path), "hard20")
    assert report.coverage["n_expert"] == report.coverage["n_joined"] == 77
    assert report.status_counts == {IN_SET: 54, OUT_OF_SET: 5, UNIDENTIFIABLE: 16, MISSING: 2}
    assert (report.accuracy_in_set["correct"], report.accuracy_in_set["total"]) == (36, 54)
    assert report.accuracy_in_set["accuracy"] == pytest.approx(0.667, abs=5e-4)
    assert (report.accuracy_in_plus_out["correct"], report.accuracy_in_plus_out["total"]) == (
        36,
        59,
    )
    assert report.confusion_cells() == {
        (t, p): n for t, r in ASSAF_CSV.items() for p, n in r.items()
    }


# ------------------------------------------------------------------------ the adapter


def test_load_legacy_predictions_reads_bboxes_without_eval(tmp_path):
    cols = {g: [0.1, 0.9] for g in ["Acropora", "Porites"]}
    path = tmp_path / "legacy.csv"
    pd.DataFrame(
        {
            "image_name": ["A.JPG", "B.JPG"],
            "coral_id": [3, 4],
            "bbox": ["[10.5, 20.0, 300.0, 400.25]", "[ 10  20 300.5 400]"],  # list / numpy style
            "area": [111, 222],
            **cols,
            "Cladiella": [0.0, 0.0],  # not a genus the model predicts: not a probability column
            "predicted_genus": ["Porites", "Porites"],
            "confidence": [0.9, 0.9],
            "runner_up": ["Acropora", "Acropora"],
            "expert_genus": ["x", "y"],  # not part of the canonical table
        }
    ).to_csv(path, index=False)
    out = load_legacy_predictions(path)
    assert list(out.columns) == [
        "image_name",
        "coral_id",
        "bbox",
        "area",
        "Acropora",
        "Porites",
        "predicted_genus",
        "confidence",
        "runner_up",
    ]
    assert out["bbox"].tolist() == [[10.5, 20.0, 300.0, 400.25], [10.0, 20.0, 300.5, 400.0]]
    assert out["coral_id"].tolist() == [3, 4]

    bad = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "image_name": ["A"],
            "coral_id": [1],
            "bbox": ["__import__('os')"],
            "Acropora": [1],
            "predicted_genus": ["Acropora"],
        }
    ).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="needs 4 numbers"):
        load_legacy_predictions(bad)
    nogenus = tmp_path / "none.csv"
    pd.DataFrame({"image_name": ["A"], "coral_id": [1], "predicted_genus": ["x"]}).to_csv(
        nogenus, index=False
    )
    with pytest.raises(ValueError, match="no genus probability columns"):
        load_legacy_predictions(nogenus)


# ----------------------------------------------------------------------- comparison


def _table(n=6, seed=0):
    rng = np.random.RandomState(seed)
    probs = rng.dirichlet(np.ones(20) * 0.3, size=n)
    df = pd.DataFrame(probs, columns=HARD20)
    df.insert(0, "coral_id", [i % 3 for i in range(n)])
    df.insert(0, "image_name", [f"IMG_{i // 3}.JPG" for i in range(n)])
    df["predicted_genus"] = df[HARD20].idxmax(axis=1)
    return df


def test_identical_tables_compare_perfectly():
    a = _table()
    result = compare_predictions(a, a.sample(frac=1, random_state=1))  # row order is irrelevant
    assert result.keys_equal and result.n_common == 6 and result.n_disagree == 0
    assert result.top1_agreement == 1.0
    assert result.max_abs_diff == 0.0 and result.mean_abs_diff == 0.0
    assert result.disagreements.empty


def test_small_probability_noise_is_measured_and_top1_flips_are_listed():
    a = _table()
    b = a.copy()
    b.loc[2, HARD20[0]] += 4e-4
    b.loc[2, HARD20[1]] -= 1e-4
    result = compare_predictions(a, b)
    assert result.max_abs_diff == pytest.approx(4e-4)
    assert result.mean_abs_diff == pytest.approx((4e-4 + 1e-4) / (6 * 20))
    assert result.worst["uid"] == "IMG_0.JPG#2" and result.worst["genus"] == HARD20[0]
    assert result.top1_agreement == 1.0

    c = a.copy()
    flipped = "Pavona" if a.loc[1, "predicted_genus"] != "Pavona" else "Porites"
    c.loc[1, "predicted_genus"] = flipped
    result = compare_predictions(a, c)
    assert result.n_disagree == 1 and result.top1_agreement == pytest.approx(5 / 6)
    assert result.disagreements["uid"].tolist() == ["IMG_0.JPG#1"]
    assert result.disagreements["predicted_b"].tolist() == [flipped]


def test_key_set_differences_are_reported():
    a, b = _table(), _table().iloc[1:].copy()
    b.loc[b.index[0], "image_name"] = "OTHER.JPG"
    result = compare_predictions(a, b)
    assert not result.keys_equal
    assert result.only_in_a == 2 and result.only_in_b == 1
    assert "IMG_0.JPG#0" in result.only_in_a_sample and result.only_in_b_sample == ["OTHER.JPG#1"]
    assert result.n_common == 4 and result.n_a == 6 and result.n_b == 5
    d = result.to_dict()
    assert d["keys_equal"] is False
    json.dumps(d)


def test_compare_uses_shared_genus_columns_and_derives_top1_when_missing():
    a = _table()
    b = a.drop(columns=["predicted_genus", HARD20[5]])
    result = compare_predictions(a, b)
    assert HARD20[5] not in result.genus_columns and len(result.genus_columns) == 19
    assert result.max_abs_diff == 0.0
    with pytest.raises(ValueError, match="share no genus"):
        compare_predictions(a, a[["image_name", "coral_id"]])
    with pytest.raises(ValueError, match="duplicate"):
        compare_predictions(pd.concat([a, a.head(1)]), a)


# ------------------------------------------------------------------ distribution


@pytest.mark.parametrize(("genus_col", "count_col"), [("genus", "count"), ("Observation", "#")])
def test_distribution_accepts_both_ground_truth_layouts(tmp_path, genus_col, count_col):
    gt = tmp_path / "gt.csv"
    pd.DataFrame(
        {
            "Category": ["hard", "hard", "hard", "soft", "hard"],
            genus_col: ["Acropora", "porites ", "Pavona", "Sinularia", "Acropora"],
            count_col: [60, 30, 5, 100, 5],  # Acropora appears twice: 65 in all
            "Pi": [0.3] * 5,
        }
    ).to_csv(gt, index=False)
    predictions = pd.DataFrame({"predicted_genus": ["Acropora"] * 6 + ["Porites"] * 3 + ["Pavona"]})
    table = distribution_vs_ground_truth(predictions, gt, "hard20")
    assert list(table.index) == HARD20
    assert table.loc["Acropora", ["predicted_n", "survey_n"]].tolist() == [6, 65]
    assert table.loc["Acropora", "predicted_share"] == pytest.approx(0.6)
    assert table.loc["Acropora", "survey_share"] == pytest.approx(65 / 100)  # soft coral left out
    assert table.loc["Acropora", "difference_pp"] == pytest.approx(-5.0)
    assert table.loc["Pavona", "ratio"] == pytest.approx(0.1 / 0.05)
    assert np.isnan(table.loc["Montipora", "ratio"])  # no survey count -> no ratio
    assert table.attrs["survey_excluded"] == {"Sinularia": 100.0}
    assert table["predicted_share"].sum() == pytest.approx(1.0)
    assert table["survey_share"].sum() == pytest.approx(1.0)


def test_distribution_needs_recognisable_columns(tmp_path):
    gt = tmp_path / "gt.csv"
    pd.DataFrame({"name": ["Acropora"], "n": [3]}).to_csv(gt, index=False)
    with pytest.raises(ValueError, match="No genus column"):
        distribution_vs_ground_truth(pd.DataFrame({"predicted_genus": ["Acropora"]}), gt)


def test_confusion_cells_of_a_plain_frame():
    cm = pd.DataFrame([[2, 0], [1, 0]], index=["a", "b"], columns=["a", "b"])
    assert confusion_cells(cm) == {("a", "a"): 2, ("b", "a"): 1}
