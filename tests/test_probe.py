import json

import numpy as np
import pandas as pd
import pytest

from coralid.probe import (
    ProbeData,
    ProbeError,
    cross_validate_probe,
    fit_probe,
    join_labels,
    load_labels,
)
from coralid.reports import write_probe_report

GENERA = ["Acropora", "Porites", "Pocillopora"]


def synthetic(n_images=24, per_image=3, dim=6, separation=4.0, seed=0, labels=GENERA):
    """Embeddings around one centre per genus, `per_image` crops in each of `n_images` photos."""
    rng = np.random.RandomState(seed)
    centres = rng.randn(len(labels), dim) * separation
    uids, rows, genus = [], [], []
    for image in range(n_images):
        for coral in range(per_image):
            g = (image + coral) % len(labels)
            uids.append(f"IMG_{image}.JPG#{coral}")
            rows.append(centres[g] + rng.randn(dim))
            genus.append(labels[g])
    labels_table = pd.DataFrame(
        {
            "image_name": [u.rpartition("#")[0] for u in uids],
            "coral_id": [int(u.rpartition("#")[2]) for u in uids],
            "genus": genus,
        }
    )
    return uids, np.array(rows, dtype=np.float32), labels_table


# ------------------------------------------------------------------------- labels and the join


def test_join_aligns_labels_to_features_by_image_and_coral_id():
    uids = ["A.JPG#0", "A.JPG#1", "B.JPG#0", "B.JPG#1"]
    X = np.arange(8, dtype=np.float32).reshape(4, 2)
    labels = pd.DataFrame(
        {"image_name": ["B.JPG", "A.JPG"], "coral_id": [1.0, 0], "genus": ["porites", "Acropora"]}
    )
    data = join_labels(uids, X, labels, "genus", "hard20")
    assert data.uids == ["A.JPG#0", "B.JPG#1"]  # the order of the features file
    assert data.y.tolist() == ["Acropora", "Porites"]  # canonical spelling
    assert data.groups.tolist() == ["A.JPG", "B.JPG"]
    np.testing.assert_array_equal(data.X, X[[0, 3]])
    # coral_id 0 of A is not coral_id 0 of B: the key is the pair
    only_a = labels.iloc[[1]]
    assert join_labels(uids, X, only_a, "genus", "hard20").uids == ["A.JPG#0"]


def test_join_counts_what_it_leaves_out():
    uids = ["A.JPG#0", "A.JPG#1", "B.JPG#0", "B.JPG#1", "C.JPG#0"]
    X = np.zeros((5, 2), np.float32)
    labels = pd.DataFrame(
        {
            "image_name": ["A.JPG", "A.JPG", "B.JPG", "B.JPG", "Z.JPG", "A.JPG", "C.JPG"],
            "coral_id": [0, 1, 0, 1, 0, 0, 0],
            "genus": [
                "Acropora",
                "unidentifiable",  # blank
                "Sinularia",  # a soft coral: outside hard20
                "MISSING_LABEL",  # blank
                "Porites",  # no features for this key
                "Acropora",  # the same key and label twice
                None,  # blank
            ],
        }
    )
    data = join_labels(uids, X, labels, "genus", "hard20")
    assert data.uids == ["A.JPG#0"]
    assert data.dropped == {"blank": 3, "outside_label_set": 1, "no_features": 1, "duplicate": 1}


def test_join_without_a_label_set_keeps_every_named_genus():
    uids = ["A.JPG#0", "B.JPG#0"]
    labels = pd.DataFrame(
        {"image_name": ["A.JPG", "B.JPG"], "coral_id": [0, 0], "genus": ["Acropora", "Sinularia"]}
    )
    data = join_labels(uids, np.zeros((2, 2)), labels)
    assert data.y.tolist() == ["Acropora", "Sinularia"]


def test_two_labels_for_one_crop_are_an_error():
    labels = pd.DataFrame(
        {"image_name": ["A.JPG"] * 2, "coral_id": [0, 0], "genus": ["Acropora", "Porites"]}
    )
    with pytest.raises(ProbeError, match="two different labels"):
        join_labels(["A.JPG#0"], np.zeros((1, 2)), labels, "genus", "hard20")


def test_the_label_column_must_exist_and_labels_come_from_a_file(tmp_path):
    table = pd.DataFrame({"image_name": ["A.JPG"], "coral_id": [0], "expert_genus": ["Acropora"]})
    with pytest.raises(ProbeError, match="no 'genus' column"):
        join_labels(["A.JPG#0"], np.zeros((1, 2)), table)
    table.to_csv(tmp_path / "labels.csv", index=False)
    table.to_parquet(tmp_path / "labels.parquet")
    for path in ("labels.csv", "labels.parquet"):
        loaded = load_labels(tmp_path / path)
        assert loaded["coral_id"].tolist() == [0] and loaded["image_name"].tolist() == ["A.JPG"]
    assert join_labels(["A.JPG#0"], np.zeros((1, 2)), loaded, "expert_genus").y.tolist() == [
        "Acropora"
    ]


# ------------------------------------------------------------------------- cross-validation


def test_every_crop_is_predicted_once_by_a_model_that_never_saw_its_photo():
    uids, X, labels = synthetic()
    data = join_labels(uids, X, labels, "genus", "hard20")
    report = cross_validate_probe(data, n_splits=4)
    held = report.held_out
    assert sorted(held["uid"]) == sorted(uids) and len(held) == 72  # once each
    photos_per_fold = held.groupby("fold")["image_name"].apply(set)
    for a in range(4):
        for b in range(a + 1, 4):
            assert not photos_per_fold[a] & photos_per_fold[b]  # a photo is in one fold only
    assert [f["n_test"] for f in report.folds] == [18, 18, 18, 18]
    assert report.evaluation.accuracy_in_set["total"] == 72


def test_separable_data_gives_high_held_out_accuracy_with_an_interval():
    uids, X, labels = synthetic()
    report = cross_validate_probe(join_labels(uids, X, labels, "genus", "hard20"), n_splits=4)
    acc = report.accuracy
    assert acc["accuracy"] > 0.95 and acc["ci95"][0] > 0.85
    assert report.majority_baseline == pytest.approx(1 / 3)
    per_genus = report.evaluation.per_genus
    assert per_genus.loc["Porites", "support"] == 24 and per_genus.loc["Porites", "recall"] > 0.9
    assert list(report.evaluation.confusion.index[:3]) == sorted(GENERA)
    assert report.class_counts == {"Acropora": 24, "Pocillopora": 24, "Porites": 24}


def test_the_report_is_held_out_not_training_accuracy():
    """Labels that carry no signal: the probe fits them perfectly but cannot predict them."""
    rng = np.random.RandomState(1)
    uids, X, labels = synthetic(n_images=30, dim=60)
    X = rng.randn(*X.shape).astype(np.float32)  # noise: the features say nothing about the genus
    data = join_labels(uids, X, labels, "genus", "hard20")
    training = fit_probe(data, C=100.0)
    assert (
        training.predict(data.X) == data.y
    ).mean() > 0.95  # the training accuracy, which is never reported
    report = cross_validate_probe(data, n_splits=5, C=100.0)
    assert report.accuracy["accuracy"] < 0.6  # chance is 1/3
    assert report.accuracy["total"] == len(data)


def test_class_weights_help_the_rare_genus():
    rng = np.random.RandomState(2)
    n_images, per_image = 40, 5
    uids, rows, genus = [], [], []
    for image in range(n_images):
        for coral in range(per_image):
            rare = (image * per_image + coral) % 10 == 0
            uids.append(f"IMG_{image}.JPG#{coral}")
            genus.append("Millepora" if rare else "Acropora")
            rows.append([rng.randn() + (1.0 if rare else -1.0), rng.randn()])
    labels = pd.DataFrame(
        {
            "image_name": [u.rpartition("#")[0] for u in uids],
            "coral_id": [int(u.rpartition("#")[2]) for u in uids],
            "genus": genus,
        }
    )
    data = join_labels(uids, np.array(rows, np.float32), labels, "genus", "hard20")

    def recall(weights):
        report = cross_validate_probe(data, n_splits=5, class_weight=weights)
        return report.evaluation.per_genus.loc["Millepora", "recall"]

    assert recall("balanced") > recall(None)


def test_the_report_compares_with_the_zero_shot_predictions_on_the_same_crops():
    uids, X, labels = synthetic()
    data = join_labels(uids, X, labels, "genus", "hard20")
    zero_shot = labels.rename(columns={"genus": "predicted_genus"}).copy()
    zero_shot["predicted_genus"] = "Acropora"  # a zero-shot model that always says Acropora
    zero_shot = zero_shot.iloc[:-3]  # and has no prediction for the last crops
    report = cross_validate_probe(data, n_splits=4, zero_shot=zero_shot)
    shot = report.zero_shot
    assert shot["total"] == 69 and shot["correct"] == 23 and 0.3 < shot["accuracy"] < 0.4
    assert cross_validate_probe(data, n_splits=4).zero_shot is None


def test_few_photos_few_labels_and_a_single_class_are_handled():
    uids, X, labels = synthetic(n_images=3, per_image=4)
    data = join_labels(uids, X, labels, "genus", "hard20")
    report = cross_validate_probe(data, n_splits=10)
    assert report.params["n_splits"] == 3 and len(report.folds) == 3
    assert any("n_splits was lowered to 3" in w for w in report.warnings)
    assert any("fewer than 5 labels" in w for w in report.warnings)
    one_photo = ProbeData(
        ["A#0"], np.zeros((1, 2)), np.array(["Acropora"], object), np.array(["A"])
    )
    with pytest.raises(ProbeError, match="at least 2 photos"):
        cross_validate_probe(one_photo)
    # a genus that exists in one photo only: some folds train on a single class
    uids = ["A.JPG#0", "A.JPG#1", "B.JPG#0", "B.JPG#1"]
    X = np.array([[0, 0], [0, 1], [5, 5], [5, 6]], np.float32)
    labels = pd.DataFrame(
        {
            "image_name": ["A.JPG", "A.JPG", "B.JPG", "B.JPG"],
            "coral_id": [0, 1, 0, 1],
            "genus": ["Acropora", "Acropora", "Porites", "Porites"],
        }
    )
    tiny = cross_validate_probe(join_labels(uids, X, labels, "genus", "hard20"), n_splits=2)
    assert len(tiny.held_out) == 4  # no crash; the unseen genus is simply missed


# ------------------------------------------------------------------------- saving


def test_the_report_is_saved_as_new_files_in_a_new_folder(tmp_path):
    uids, X, labels = synthetic()
    report = cross_validate_probe(join_labels(uids, X, labels, "genus", "hard20"), n_splits=4)
    folder = write_probe_report(report, tmp_path / "reports", "test", date="2026-10-12")
    assert folder.name == "probe_test_2026-10-12"
    assert sorted(p.name for p in folder.iterdir()) == [
        "confusion.csv",
        "folds.csv",
        "held_out.csv",
        "per_genus.csv",
        "summary.json",
    ]
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["kind"] == "linear_probe_held_out" and summary["n_labeled"] == 72
    assert summary["accuracy"]["total"] == 72 and summary["params"]["class_weight"] == "balanced"
    before = (folder / "summary.json").read_bytes()
    second = write_probe_report(report, tmp_path / "reports", "test", date="2026-10-12")
    assert second.name == "probe_test_2026-10-12_2"  # never over the first
    assert (folder / "summary.json").read_bytes() == before
