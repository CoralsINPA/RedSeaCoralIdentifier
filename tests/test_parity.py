import json

import numpy as np
import pandas as pd
import pytest

from coralid.evaluation import compare_predictions, evaluate
from coralid.genera import load_genera
from coralid.masks import JsonAudit
from coralid.parity import (
    EXPECTED,
    FAIL,
    INFO,
    NOT_RUN,
    PASS,
    CheckResult,
    check_against_drive_csv,
    check_expert_confusion,
    check_geometric,
    check_image_manifest,
    check_json_audit,
    check_sample_counts,
    check_zero_shot_parity,
    equal,
)
from coralid.parity_report import (
    format_summary,
    format_table,
    next_free_path,
    summarize,
    write_report,
)
from tests.assaf_matrix import ASSAF

HARD20 = load_genera().genera("hard20")


def _by_name(results):
    return {r.name.split(" ")[0]: r for r in results}


# ----------------------------------------------------------------------------- A


def _audit(**overrides):
    values = {
        "n_files": 1714,
        "n_masks": 44211,
        "frame_sizes": {(2560, 1920): 1714},
        "noted": [],
        "name_mismatches": [],
        "unreadable": [],
    }
    values.update(overrides)
    return JsonAudit(**values)


def test_json_audit_passes_on_the_recorded_numbers():
    rows = _by_name(check_json_audit(_audit()))
    assert [rows[k].status for k in ("A1", "A2", "A3")] == [PASS, PASS, PASS]
    assert rows["A4"].status == INFO and rows["A4"].detail == "all as expected"


def test_json_audit_fails_on_different_counts_and_reports_odd_frames():
    audit = _audit(
        n_files=1700,
        frame_sizes={(2560, 1920): 1690, (2560, 1706): 8, (1920, 2560): 2},
        unreadable=[("IMG_1.json", "boom")],
        noted=[("IMG_2.json", "Skipped due to: x")],
    )
    rows = _by_name(check_json_audit(audit))
    assert rows["A1"].status == FAIL and rows["A1"].actual == 1700
    assert rows["A2"].status == PASS
    assert rows["A3"].status == FAIL
    assert rows["A4"].status == INFO and rows["A4"].detail == "10 file(s) not 2560x1920"
    assert rows["A5"].actual == 1


def _geometric_manifest(survivors_per_file):
    rows = []
    for i, n_keep in enumerate(survivors_per_file):
        for _ in range(n_keep):
            rows.append({"image_name": f"I{i}.JPG", "stage1_reason": None})
        rows.append({"image_name": f"I{i}.JPG", "stage1_reason": "area"})
    return pd.DataFrame(rows)


def test_geometric_checks():
    expected = {**EXPECTED, "after_geometric": 12, "files_without_survivors": 1}
    manifest = _geometric_manifest([5, 7, 0])
    rows = _by_name(check_geometric(manifest, {"load_s": 3.14, "filter_s": 0.5}, expected))
    assert rows["A7"].status == PASS and rows["A7"].actual == 12
    assert rows["A8"].actual == {"area": 3}
    assert rows["A9"].actual == [1, 4.0]  # one file without survivors, 12 / 3 per file
    assert rows["A10"].actual == [3.1, 0.5]
    wrong = check_geometric(_geometric_manifest([5, 6, 0]), {"load_s": 0, "filter_s": 0}, expected)
    assert _by_name(wrong)["A7"].status == FAIL


# ----------------------------------------------------------------------------- B


def test_sample_counts_compare_both_implementations_and_the_record():
    recorded = {"masks": 2337, "after_filter": 1054, "crops": 842}
    rows = _by_name(check_sample_counts(recorded, dict(recorded)))
    assert rows["B1"].status == PASS and rows["B2"].status == INFO and rows["B2"].detail == ""
    drift = {"masks": 2300, "after_filter": 1040, "crops": 830}
    rows = _by_name(check_sample_counts(drift, dict(drift)))
    assert rows["B1"].status == PASS  # the two implementations still agree
    assert "may have changed" in rows["B2"].detail  # but the data is not what it was
    rows = _by_name(check_sample_counts(recorded, {**recorded, "crops": 841}))
    assert rows["B1"].status == FAIL


def _table(n=200, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(rng.dirichlet(np.ones(20) * 0.3, size=n), columns=HARD20)
    df.insert(0, "coral_id", [i % 7 for i in range(n)])
    df.insert(0, "image_name", [f"IMG_{i // 7}.JPG" for i in range(n)])
    df["predicted_genus"] = df[HARD20].idxmax(axis=1)
    return df


def test_zero_shot_parity_passes_within_tolerance():
    a = _table()
    b = a.copy()
    b[HARD20[3]] += 5e-4
    rows = _by_name(check_zero_shot_parity(compare_predictions(a, b)))
    assert [rows[k].status for k in ("B3", "B4", "B5")] == [PASS, PASS, PASS]


def test_zero_shot_parity_fails_on_each_criterion():
    a = _table()
    noisy = a.copy()
    noisy[HARD20[3]] += 2e-3
    assert _by_name(check_zero_shot_parity(compare_predictions(a, noisy)))["B5"].status == FAIL

    flipped = a.copy()
    flipped.loc[:2, "predicted_genus"] = "Pavona"  # 3 of 200 = 1.5% disagree (some may agree)
    flipped.loc[:2, "predicted_genus"] = [
        ("Pavona" if g != "Pavona" else "Porites") for g in a.loc[:2, "predicted_genus"]
    ]
    result = _by_name(check_zero_shot_parity(compare_predictions(a, flipped)))
    assert result["B4"].status == FAIL and result["B4"].actual == pytest.approx(197 / 200)

    # exactly at the limit still passes: 1 of 200 differs = 99.5%
    one = a.copy()
    one.loc[0, "predicted_genus"] = (
        "Pavona" if a.loc[0, "predicted_genus"] != "Pavona" else "Porites"
    )
    assert _by_name(check_zero_shot_parity(compare_predictions(a, one)))["B4"].status == PASS

    missing = _by_name(check_zero_shot_parity(compare_predictions(a, a.iloc[1:])))
    assert missing["B3"].status == FAIL and "only legacy 1" in missing["B3"].actual


def test_drive_csv_comparison_is_informational_only():
    a = _table()
    (row,) = check_against_drive_csv(compare_predictions(a, a.iloc[5:]))
    assert row.status == INFO and row.actual["keys_equal"] is False


# ----------------------------------------------------------------------------- C


def _image_manifest(n_keep=14224, n_masks=44211, failed_images=0):
    reasons = ["area"] * (n_masks - n_keep - failed_images) + ["image_missing"] * failed_images
    keep = [True] * n_keep + [False] * (n_masks - n_keep)
    return pd.DataFrame(
        {
            "image_name": [f"I{i % 100}.JPG" for i in range(n_masks)],
            "keep": keep,
            "reject_reason": [None] * n_keep + reasons,
        }
    )


def test_image_manifest_check():
    rows = _by_name(check_image_manifest(_image_manifest()))
    assert [rows[k].status for k in ("C1", "C2", "C3")] == [PASS, PASS, PASS]
    assert rows["C5"].actual == 0
    rows = _by_name(
        check_image_manifest(_image_manifest(n_keep=14000, failed_images=300), [{"image": "X"}])
    )
    assert rows["C3"].status == FAIL and rows["C3"].actual == 14000
    assert rows["C2"].status == FAIL
    assert rows["C4"].actual["image_missing"] == 300 and rows["C5"].actual > 0


# ----------------------------------------------------------------------------- D


def _assaf_report():
    rows = [(t, p) for t, r in ASSAF.items() for p, n in r.items() for _ in range(n)]
    rows += [("unidentifiable", "Acropora")] * 16
    keys = pd.DataFrame(
        {"image_name": [f"I{i}" for i in range(len(rows))], "coral_id": list(range(len(rows)))}
    )
    expert = keys.assign(expert_genus=[t for t, _ in rows])
    predictions = keys.assign(predicted_genus=[p for _, p in rows])
    return evaluate(predictions, expert, "hard20")


EXPECTED_STATUS = {"in_set": 57, "out_of_set": 5, "unidentifiable": 16, "missing": 0}


def test_expert_confusion_passes_cell_for_cell_on_the_authoritative_matrix():
    results = check_expert_confusion(_assaf_report(), ASSAF, EXPECTED_STATUS, (39, 57, 62))
    assert all(r.status == PASS for r in results)
    cells = [r for r in results if r.kind == "cell"]
    n_cells = sum(len(r) for r in ASSAF.values())
    assert n_cells == 34 and len(cells) == n_cells
    assert any(r.name == "cell Dipsastraea -> Favia" and r.expected == 2 for r in cells)
    assert _by_name(results)["D4"].name == "D4 confusion matrix cells equal (34 compared)"


def test_expert_confusion_names_the_cells_that_differ():
    expected = {t: dict(r) for t, r in ASSAF.items()}
    expected["Dipsastraea"]["Favia"] = 3  # the file says 2
    expected["Acropora"]["Pavona"] = 1  # a cell the file does not have
    results = check_expert_confusion(_assaf_report(), expected, EXPECTED_STATUS, (39, 57, 62))
    failing = {r.name: (r.expected, r.actual) for r in results if r.status == FAIL}
    assert failing["cell Dipsastraea -> Favia"] == (3, 2)
    assert failing["cell Acropora -> Pavona"] == (1, 0)
    assert failing["D4 confusion matrix cells equal (35 compared)"] == (0, 2)
    table = format_table(results)
    assert "cell Dipsastraea -> Favia" in table and "cell Porites -> Porites" not in table
    assert "cell Porites -> Porites" in format_table(results, show_cells=True)


# ----------------------------------------------------------------------- summary


def test_summarize_statuses():
    results = [
        equal("A", "A1", 1, 1),
        CheckResult("A", "A2", INFO),
        equal("B", "B1", 1, 2),
        CheckResult("C", "C1", INFO),
    ]
    summary = summarize(results)
    assert summary["checks"]["A"]["status"] == PASS
    assert summary["checks"]["B"]["status"] == FAIL
    assert summary["checks"]["C"]["status"] == INFO
    assert summary["checks"]["D"]["status"] == NOT_RUN
    assert summary["overall"] == FAIL
    assert summarize([equal("A", "A1", 1, 1)])["overall"] == "PARTIAL"
    assert summarize([])["overall"] == NOT_RUN
    all_four = [equal(c, f"{c}1", 1, 1) for c in "ABCD"]
    assert summarize(all_four)["overall"] == PASS
    json.dumps(summary)


def test_summary_text_lists_the_failures():
    summary = summarize([equal("A", "A1 files", 3, 4), equal("B", "B3 keys", 1, 1)])
    text = format_summary(summary)
    assert text.splitlines()[0] == "PARITY SUMMARY: FAIL"
    assert "A  FAIL" in text and "B  PASS" in text and "D  NOT RUN" in text
    assert "FAILED A1 files: expected 4, got 3" in text


def test_reports_are_new_files_and_never_overwrite(tmp_path):
    summary = summarize([equal("A", "A1", 1, 1)])
    first = write_report(summary, tmp_path / "reports", date="2026-10-12")
    second = write_report(summary, tmp_path / "reports", date="2026-10-12")
    third = write_report(summary, tmp_path / "reports", date="2026-10-12")
    assert [p.name for p in (first, second, third)] == [
        "parity_2026-10-12.json",
        "parity_2026-10-12_2.json",
        "parity_2026-10-12_3.json",
    ]
    body = json.loads(first.read_text())
    assert body["overall"] == "PARTIAL" and "packages" in body["environment"]
    assert body["checks"]["A"]["rows"][0]["name"] == "A1"
    assert next_free_path(tmp_path / "nothing.json") == tmp_path / "nothing.json"
    with pytest.raises(FileExistsError):  # even a direct write cannot replace a report
        with open(first, "x"):
            pass
