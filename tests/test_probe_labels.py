import pandas as pd
import pytest

from coralid.probe_labels import expert_labels

GENERA = ["Acropora", "Porites", "Pavona"]


def manifest():
    """Two landscape photos and one portrait photo."""
    return pd.DataFrame(
        {
            "image_name": ["L1.JPG", "L2.JPG", "P1.JPG"],
            "frame_w": [2560, 2560, 1920],
            "frame_h": [1920, 1920, 2560],
        }
    )


def expert_csv(tmp_path):
    """The real file's shape: empty expert_genus with a word in is_correct for no-genus rows."""
    rows = [
        ("L1.JPG", 0, "Acropora", True),
        ("L1.JPG", 1, "Porites", False),
        ("L2.JPG", 0, "Pavona", True),
        ("L2.JPG", 1, "Sinularia", False),  # a soft coral: outside the label set
        ("L2.JPG", 2, None, "unidentifiable"),
        ("L2.JPG", 3, None, "MISSING_LABEL"),
        ("P1.JPG", 0, "Acropora", True),  # in the label set, but on a portrait photo
        ("P1.JPG", 1, None, "unidentifiable"),
        ("GONE.JPG", 0, "Porites", True),  # a photo the manifest does not have
    ]
    path = tmp_path / "expert.csv"
    pd.DataFrame(rows, columns=["image_name", "coral_id", "expert_genus", "is_correct"]).to_csv(
        path, index=False
    )
    return path


def test_only_in_set_landscape_rows_become_labels(tmp_path):
    labels, summary = expert_labels(expert_csv(tmp_path), manifest(), GENERA)
    assert list(labels.columns) == ["image_name", "coral_id", "genus"]
    assert list(zip(labels["image_name"], labels["coral_id"], labels["genus"], strict=True)) == [
        ("L1.JPG", 0, "Acropora"),
        ("L1.JPG", 1, "Porites"),
        ("L2.JPG", 0, "Pavona"),
    ]
    assert summary["n_labels"] == 3 and summary["n_photos"] == 2
    assert summary["genera"] == {"Acropora": 1, "Porites": 1, "Pavona": 1}


def test_every_expert_row_is_accounted_for(tmp_path):
    _, summary = expert_labels(expert_csv(tmp_path), manifest(), GENERA)
    assert summary["expert_rows"] == 9
    assert summary["by_status"] == {
        "in_set": 5,
        "unidentifiable": 2,
        "out_of_set": 1,
        "missing": 1,
    }
    assert summary["by_orientation"] == {"landscape": 6, "portrait": 2, "unknown": 1}
    dropped = summary["dropped"]
    assert dropped == {
        "outside_the_label_set": 1,
        "unidentifiable": 2,
        "missing": 1,
        "on_other_orientation_than_landscape": 1,  # the portrait Acropora
        "photo_not_in_the_manifest": 1,
    }
    assert summary["n_labels"] + sum(dropped.values()) == summary["expert_rows"]


def test_portrait_rows_never_become_labels_unless_asked(tmp_path):
    csv = expert_csv(tmp_path)
    labels, _ = expert_labels(csv, manifest(), GENERA)
    assert "P1.JPG" not in set(labels["image_name"])
    portrait, summary = expert_labels(csv, manifest(), GENERA, orientation="portrait")
    assert list(portrait["image_name"]) == ["P1.JPG"]
    assert summary["dropped"]["on_other_orientation_than_portrait"] == 3


def test_a_label_set_name_works_and_unknown_genus_spelling_is_canonical(tmp_path):
    path = tmp_path / "e.csv"
    pd.DataFrame(
        {
            "image_name": ["L1.JPG"],
            "coral_id": [4.0],
            "expert_genus": ["porites"],
            "is_correct": [True],
        }
    ).to_csv(path, index=False)
    labels, _ = expert_labels(path, manifest(), "hard20")
    assert list(labels["genus"]) == ["Porites"] and labels["coral_id"].dtype == "int64"


def test_the_no_label_columns_of_the_file_are_required(tmp_path):
    path = tmp_path / "e.csv"
    pd.DataFrame({"image_name": ["L1.JPG"], "coral_id": [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="expert_genus"):
        expert_labels(path, manifest(), GENERA)
