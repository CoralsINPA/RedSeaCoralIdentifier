import json

import numpy as np
import pandas as pd
import pytest

from coralid.evaluation import IN_SET, OUT_OF_SET, load_expert_set, load_legacy_predictions
from coralid.genera import load_genera
from coralid.reports import GROUPS, baseline_report, write_baseline_report
from coralid.zero_shot import predictions_table

HARD20 = load_genera().genera("hard20")


def _manifest():
    """Photos L1, L2 (landscape) and P1 (portrait), a few masks each; some are not kept."""
    rows = []
    for name, w, h, n in [
        ("L1.JPG", 2560, 1920, 4),
        ("L2.JPG", 2560, 1920, 3),
        ("P1.JPG", 1920, 2560, 3),
    ]:
        for i in range(n):
            rows.append(
                {
                    "image_name": name,
                    "coral_id": i,
                    "frame_w": w,
                    "frame_h": h,
                    "keep": i < n - 1,
                }
            )
    return pd.DataFrame(rows)


def _predictions(genus_of):
    """A baseline for the kept masks of the manifest; `genus_of(i)` names row i's genus."""
    kept = _manifest()
    kept = kept[kept["keep"]].reset_index(drop=True)
    probs = np.full((len(kept), 20), 0.01)
    for i in range(len(kept)):
        probs[i, HARD20.index(genus_of(i))] = 0.81
    meta = kept[["image_name", "coral_id"]].assign(bbox=[[0.0, 0.0, 1.0, 1.0]] * len(kept), area=1)
    return predictions_table(meta, probs / probs.sum(axis=1, keepdims=True), HARD20)


GENERA_OF_KEPT = ["Acropora", "Porites", "Millepora", "Pavona", "Acropora", "Echinopora", "Porites"]


def _expert_csv(tmp_path, legacy_genus):
    """The expert CSV: keys on every photo, its own (legacy) predictions and expert labels."""
    rows = [
        ("L1.JPG", 0, "Acropora", legacy_genus["L1.JPG#0"]),
        ("L1.JPG", 1, "Porites", legacy_genus["L1.JPG#1"]),
        ("L2.JPG", 0, "Sinularia", legacy_genus["L2.JPG#0"]),
        ("L2.JPG", 1, None, legacy_genus["L2.JPG#1"]),  # unidentifiable
        ("P1.JPG", 0, "Echinopora", legacy_genus["P1.JPG#0"]),
        ("P1.JPG", 1, "Porites", legacy_genus["P1.JPG#1"]),
    ]
    records = []
    for name, cid, expert, legacy in rows:
        record = {"image_name": name, "coral_id": cid}
        record |= {g: (0.81 if g == legacy else 0.01) for g in HARD20}
        record |= {"predicted_genus": legacy, "confidence": 0.81}
        record |= {
            "expert_genus": expert,
            "is_correct": "unidentifiable" if expert is None else expert == legacy,
        }
        records.append(record)
    path = tmp_path / "expert.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def _ground_truth(tmp_path):
    path = tmp_path / "gt.csv"
    rng = np.random.RandomState(0)
    genera = [*HARD20, "Sinularia"]
    pd.DataFrame(
        {"Category": "x", "genus": genera, "count": rng.randint(5, 50, len(genera)), "Pi": 0.0}
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def report(tmp_path):
    # baseline rows (kept): L1#0 Acropora, L1#1 Porites, L1#2 Millepora, L2#0 Pavona,
    #                       L2#1 Acropora, P1#0 Echinopora, P1#1 Porites
    legacy = {
        "L1.JPG#0": "Acropora",  # right, and the baseline agrees
        "L1.JPG#1": "Porites",  # right, agrees
        "L2.JPG#0": "Acropora",  # a soft coral, always wrong
        "L2.JPG#1": "Pavona",  # unidentifiable
        "P1.JPG#0": "Pavona",  # legacy wrong; the new baseline says Echinopora: right
        "P1.JPG#1": "Pavona",  # legacy wrong; the new baseline says Porites: right
    }
    path = _expert_csv(tmp_path, legacy)
    return baseline_report(
        _predictions(lambda i: GENERA_OF_KEPT[i]),
        _manifest(),
        load_expert_set(path),
        load_legacy_predictions(path),
        _ground_truth(tmp_path),
    )


def test_the_expert_evaluation_is_split_by_orientation(report):
    assert list(report.expert_new) == list(GROUPS) == ["all", "landscape", "portrait"]
    land, port, everything = (report.expert_new[g] for g in ("landscape", "portrait", "all"))
    assert land.coverage["n_expert"] == 4 and port.coverage["n_expert"] == 2
    assert everything.coverage["n_expert"] == 6 and everything.coverage["n_joined"] == 6
    assert (land.accuracy_in_set["correct"], land.accuracy_in_set["total"]) == (2, 2)
    assert (port.accuracy_in_set["correct"], port.accuracy_in_set["total"]) == (2, 2)
    assert land.status_counts[OUT_OF_SET] == 1 and land.status_counts["unidentifiable"] == 1
    assert (
        everything.accuracy_in_plus_out["total"] == 5 and everything.accuracy_in_set["total"] == 4
    )


def test_the_legacy_predictions_of_the_csv_are_scored_on_the_same_rows(report):
    legacy_land, legacy_port = report.expert_legacy["landscape"], report.expert_legacy["portrait"]
    assert (legacy_land.accuracy_in_set["correct"], legacy_land.accuracy_in_set["total"]) == (2, 2)
    assert (legacy_port.accuracy_in_set["correct"], legacy_port.accuracy_in_set["total"]) == (0, 2)


def test_the_paired_table_compares_new_and_legacy_row_by_row(report):
    paired = report.expert_paired.set_index("group")
    assert paired.loc["landscape", ["in_set_paired", "legacy_correct", "new_correct"]].tolist() == [
        2,
        2,
        2,
    ]
    assert paired.loc["landscape", "predictions_changed"] == 2  # L2#0 and L2#1 (not in-set rows)
    assert paired.loc["portrait", ["in_set_paired", "legacy_correct", "new_correct"]].tolist() == [
        2,
        0,
        2,
    ]
    assert paired.loc["portrait", "predictions_changed"] == 2
    assert (
        paired.loc["portrait", "new_accuracy"] == 1.0
        and paired.loc["portrait", "legacy_accuracy"] == 0.0
    )
    assert paired.loc["all", "with_both_predictions"] == 6


def test_the_distribution_is_computed_for_each_group(report):
    assert set(report.distribution) == set(GROUPS)
    assert report.distribution["all"]["predicted_n"].sum() == 7
    assert report.distribution["landscape"]["predicted_n"].sum() == 5
    assert report.distribution["portrait"]["predicted_n"].sum() == 2
    assert set(report.distribution["all"].attrs["survey_excluded"]) == {"Sinularia"}
    mix = report.genus_by_orientation
    assert mix.loc["Porites", ["all_n", "landscape_n", "portrait_n"]].tolist() == [2, 1, 1]
    assert mix["all_share"].sum() == pytest.approx(1.0)


def test_the_orientation_summary(report):
    summary = report.orientation_summary.set_index("group")
    assert summary["photos"].to_dict() == {"all": 3, "landscape": 2, "portrait": 1}
    assert summary["masks"].to_dict() == {"all": 10, "landscape": 7, "portrait": 3}
    assert summary["kept_in_manifest"].to_dict() == {"all": 7, "landscape": 5, "portrait": 2}
    assert summary["predictions"].to_dict() == {"all": 7, "landscape": 5, "portrait": 2}
    assert summary.loc["all", "mean_confidence"] == pytest.approx(0.81)


def test_the_report_carries_the_caution_and_is_json_safe(report):
    assert "different-looking crop" in report.caution
    body = report.to_dict()
    json.dumps(body, allow_nan=False)  # NaN would be invalid JSON
    assert body["expert_new"]["landscape"]["accuracy_in_set"]["correct"] == 2


def test_the_report_files_are_new_and_never_overwritten(report, tmp_path):
    first = write_baseline_report(report, tmp_path / "reports", "fb_taxonomy_v1", date="2026-10-12")
    second = write_baseline_report(
        report, tmp_path / "reports", "fb_taxonomy_v1", date="2026-10-12"
    )
    assert first.name == "baseline_fb_taxonomy_v1_2026-10-12"
    assert second.name == "baseline_fb_taxonomy_v1_2026-10-12_2"
    names = sorted(p.name for p in first.iterdir())
    assert "summary.json" in names and "expert_paired.csv" in names
    for group in GROUPS:
        for stem in ("expert_{}_per_genus", "expert_{}_confusion", "distribution_{}"):
            assert stem.format(group) + ".csv" in names
    assert json.loads((first / "summary.json").read_text())["label_set"] == "hard20"
    confusion = pd.read_csv(first / "expert_landscape_confusion.csv", index_col=0)
    assert confusion.loc["Acropora", "Acropora"] == 1


def test_a_group_without_expert_rows_still_reports(tmp_path):
    legacy = {
        k: "Porites"
        for k in ("L1.JPG#0", "L1.JPG#1", "L2.JPG#0", "L2.JPG#1", "P1.JPG#0", "P1.JPG#1")
    }
    path = _expert_csv(tmp_path, legacy)
    expert = load_expert_set(path)
    expert = expert[~expert["image_name"].str.startswith("P")]  # no portrait expert rows
    report = baseline_report(
        _predictions(lambda i: GENERA_OF_KEPT[i]),
        _manifest(),
        expert,
        load_legacy_predictions(path),
        _ground_truth(tmp_path),
    )
    assert report.expert_new["portrait"].coverage["n_expert"] == 0
    assert report.expert_new["portrait"].accuracy_in_set["total"] == 0
    assert report.expert_paired.set_index("group").loc["portrait", "in_set_paired"] == 0
    assert IN_SET in report.expert_new["all"].status_counts


def test_new_report_folders_are_never_reused(tmp_path):
    from coralid.reports import new_report_folder

    reports = tmp_path / "reports"  # created on demand
    first = new_report_folder(reports, "segcheck", date="2026-10-12")
    second = new_report_folder(reports, "segcheck", date="2026-10-12")
    named = new_report_folder(reports, "probe", "feats", "2026-10-12")
    assert first.name == "segcheck_2026-10-12" and second.name == "segcheck_2026-10-12_2"
    assert named.name == "probe_feats_2026-10-12"
    assert all(p.is_dir() for p in (first, second, named))


def test_the_probe_report_lines_are_what_the_command_prints():
    from types import SimpleNamespace

    from coralid.reports import probe_report_lines

    report = SimpleNamespace(
        accuracy={"correct": 3, "total": 5, "accuracy": 0.6, "ci95": (0.23, 0.88)},
        n_labeled=5,
        n_images=4,
        majority_baseline=0.4,
        zero_shot={"correct": 2, "total": 5, "accuracy": 0.4},
        warnings=["classes with fewer than 5 labels: Pavona (1)"],
        dropped={"blank": 1},
    )
    assert probe_report_lines(report) == [
        "Held-out accuracy 3/5 = 0.600 (0.230-0.880) on 5 labeled crops from 4 photos; "
        "always answering the commonest genus gives 0.400",
        "  zero-shot on the same crops: 2/5 = 0.400",
        "  note: classes with fewer than 5 labels: Pavona (1)",
        "  labels left out: {'blank': 1}",
    ]
    report.zero_shot = None
    assert len(probe_report_lines(report, {"x": 2})) == 3
