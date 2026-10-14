import json

import pandas as pd
import pytest

from coralid.portrait_impact import (
    expert_portrait_impact,
    write_portrait_report,
)


def _manifest():
    """Two landscape photos (L1, L2) and two portrait photos (P1, P2)."""
    rows = [("L1.JPG", 2560, 1920), ("L2.JPG", 2560, 1920)]
    rows += [("P1.JPG", 1920, 2560), ("P2.JPG", 1920, 2560)]
    return pd.DataFrame(
        [{"image_name": n, "coral_id": 0, "frame_w": w, "frame_h": h} for n, w, h in rows]
    )


def _expert():
    keys = [("L1.JPG", 0), ("L1.JPG", 1), ("L2.JPG", 0)]
    keys += [("P1.JPG", 0), ("P1.JPG", 1), ("P2.JPG", 0), ("P2.JPG", 3)]
    status = ["in_set", "unidentifiable", "out_of_set", "unidentifiable", "in_set", "in_set"]
    status += ["missing"]
    genus = ["Acropora", None, "Sinularia", None, "Porites", "Galaxea", None]
    return pd.DataFrame(
        {
            "image_name": [k[0] for k in keys],
            "coral_id": [k[1] for k in keys],
            "status": status,
            "expert_raw": [g or s for g, s in zip(genus, status, strict=True)],
            "expert_genus": genus,
        }
    )


def _legacy():
    """The predictions stored in the expert CSV."""
    keys = [("L1.JPG", 0), ("L1.JPG", 1), ("L2.JPG", 0)]
    keys += [("P1.JPG", 0), ("P1.JPG", 1), ("P2.JPG", 0), ("P2.JPG", 3)]
    genera = ["Acropora", "Pavona", "Acropora", "Porites", "Porites", "Pavona", "Porites"]
    return pd.DataFrame(
        {
            "image_name": [k[0] for k in keys],
            "coral_id": [k[1] for k in keys],
            "predicted_genus": genera,
            "confidence": [0.9, 0.5, 0.8, 0.99, 0.7, 0.6, 0.4],
        }
    )


def _new():
    """The new baseline: the landscape rows agree; two portrait predictions changed;
    P2#3 is not in the baseline (not kept)."""
    keys = [
        ("L1.JPG", 0),
        ("L1.JPG", 1),
        ("L2.JPG", 0),
        ("P1.JPG", 0),
        ("P1.JPG", 1),
        ("P2.JPG", 0),
    ]
    genera = ["Acropora", "Pavona", "Acropora", "Echinopora", "Porites", "Galaxea"]
    return pd.DataFrame(
        {
            "image_name": [k[0] for k in keys],
            "coral_id": [k[1] for k in keys],
            "predicted_genus": genera,
            "confidence": [0.9, 0.5, 0.8, 0.55, 0.7, 0.9],
        }
    )


def test_expert_portrait_rows_list_every_row_on_a_portrait_photo():
    rows, summary = expert_portrait_impact(_expert(), _legacy(), _new(), _manifest())
    assert rows["key"].tolist() == ["P1.JPG#0", "P1.JPG#1", "P2.JPG#0", "P2.JPG#3"]
    assert "orientation" not in rows.columns
    by = rows.set_index("key")
    assert by.loc["P1.JPG#0", ["status", "legacy_pred", "new_pred"]].tolist() == [
        "unidentifiable",
        "Porites",
        "Echinopora",
    ]
    assert bool(by.loc["P1.JPG#0", "prediction_changed"])
    assert not bool(by.loc["P1.JPG#1", "prediction_changed"])
    # the expert named Galaxea; the legacy file said Pavona, the new baseline says Galaxea
    assert not bool(by.loc["P2.JPG#0", "legacy_correct"])
    assert bool(by.loc["P2.JPG#0", "new_correct"])
    # unidentifiable / missing rows are never scored
    assert pd.isna(by.loc["P1.JPG#0", "legacy_correct"]) and pd.isna(
        by.loc["P2.JPG#3", "new_correct"]
    )
    # a row the baseline does not have
    assert not by.loc["P2.JPG#3", "in_new_baseline"] and pd.isna(by.loc["P2.JPG#3", "new_pred"])
    assert pd.isna(by.loc["P2.JPG#3", "prediction_changed"])


def test_expert_portrait_summary_counts_the_unidentifiable_rows():
    _, s = expert_portrait_impact(_expert(), _legacy(), _new(), _manifest())
    assert s["n_expert_rows"] == 7 and s["rows_by_orientation"] == {"portrait": 4, "landscape": 3}
    assert s["portrait_rows"] == 4 and s["portrait_share_of_expert_rows"] == pytest.approx(4 / 7)
    assert s["portrait_share_of_photos"] == 0.5
    assert s["unidentifiable_total"] == 2 and s["unidentifiable_on_portrait"] == 1
    assert s["portrait_status"] == {"in_set": 2, "out_of_set": 0, "unidentifiable": 1, "missing": 1}
    assert s["status_by_orientation"]["landscape"]["out_of_set"] == 1
    assert s["portrait_in_new_baseline"] == 3
    assert s["portrait_prediction_changed"] == {"changed": 2, "compared": 3}
    assert s["portrait_legacy"] == {"correct": 1, "scored": 2}  # Porites right, Pavona wrong
    assert s["portrait_new"] == {"correct": 2, "scored": 2}  # Galaxea is now right
    assert s["landscape_prediction_changed"] == 0
    assert "different-looking crop" in s["caution"]
    json.dumps(s)


def test_the_report_is_written_as_a_new_folder_every_time(tmp_path):
    rows, s = expert_portrait_impact(_expert(), _legacy(), _new(), _manifest())
    first = write_portrait_report(rows, s, tmp_path, date="2026-10-12")
    second = write_portrait_report(rows, s, tmp_path, date="2026-10-12")
    assert (
        first.name == "portrait_impact_2026-10-12" and second.name == "portrait_impact_2026-10-12_2"
    )
    for folder in (first, second):
        assert sorted(p.name for p in folder.iterdir()) == [
            "expert_portrait_rows.csv",
            "summary.json",
        ]
    body = json.loads((first / "summary.json").read_text())
    assert list(body) == ["expert"] and body["expert"]["portrait_rows"] == 4
    assert len(pd.read_csv(first / "expert_portrait_rows.csv")) == 4
