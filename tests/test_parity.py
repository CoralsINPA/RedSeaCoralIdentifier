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
    check_geometric,
    check_json_audit,
    check_sample_counts,
    check_zero_shot_parity,
    compare_by_orientation,
    counts_by_orientation,
    equal,
    manifest_counts,
)
from coralid.parity_expert import (
    EXPERT_CSV_CELLS,
    EXPERT_CSV_EXPECTED,
    check_expert_confusion,
)
from coralid.parity_manifest import check_image_manifest, compare_kept_with_v2, load_v2_keys
from coralid.parity_report import (
    format_summary,
    format_table,
    next_free_path,
    paste_block,
    summarize,
    write_report,
)
from tests.assaf_matrix import ASSAF, ASSAF_CSV

HARD20 = load_genera().genera("hard20")


def _by_name(results):
    return {r.name.split(" ")[0]: r for r in results}


# ----------------------------------------------------------------------------- A


def _audit(**overrides):
    values = {
        "n_files": 1714,
        "n_masks": 44211,
        "frame_sizes": {(2560, 1920): 1493, (1920, 2560): 221},
        "noted": [],
        "name_mismatches": [],
        "unreadable": [],
    }
    values.update(overrides)
    return JsonAudit(**values)


def test_json_audit_passes_on_the_recorded_numbers():
    rows = _by_name(check_json_audit(_audit()))
    assert [rows[k].status for k in ("A1", "A2", "A3")] == [PASS, PASS, PASS]
    assert rows["A4"].status == PASS and rows["A4"].detail == "1493 landscape, 221 portrait photos"


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
    assert rows["A4"].status == FAIL and rows["A4"].detail == "1698 landscape, 2 portrait photos"
    assert rows["A4"].actual == {"2560x1920": 1690, "2560x1706": 8, "1920x2560": 2}
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


def _frames(landscape=("L1.JPG", "L2.JPG"), portrait=("P1.JPG",)):
    rows = [{"image_name": n, "frame_w": 2560, "frame_h": 1920} for n in landscape]
    rows += [{"image_name": n, "frame_w": 1920, "frame_h": 2560} for n in portrait]
    return pd.DataFrame(rows)


def _table(n=200, seed=0, images=None):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(rng.dirichlet(np.ones(20) * 0.3, size=n), columns=HARD20)
    names = images or [f"IMG_{i // 7}.JPG" for i in range(n)]
    df.insert(0, "image_name", names)
    df.insert(1, "coral_id", df.groupby("image_name").cumcount())
    df["predicted_genus"] = df[HARD20].idxmax(axis=1)
    return df


def _sample_counts(legacy_total=(2337, 1054, 842), legacy_land=773, new_total=(2337, 1054, 1051)):
    legacy = dict(zip(("masks", "after_filter", "crops"), legacy_total, strict=True))
    new = dict(zip(("masks", "after_filter", "crops"), new_total, strict=True))
    legacy_crops = {"landscape": legacy_land, "portrait": legacy["crops"] - legacy_land}
    new_crops = {"landscape": legacy_land, "portrait": new["crops"] - legacy_land}
    return legacy, new, legacy_crops, new_crops


def test_sample_counts_hold_landscape_to_parity_and_report_portrait():
    rows = _by_name(check_sample_counts(*_sample_counts()))
    assert [rows[k].status for k in ("B1", "B2", "B3", "B4")] == [PASS, PASS, PASS, INFO]
    assert rows["B4"].actual == [69, 278]  # the first Colab run: 69 legacy vs 278 coralid crops
    legacy, new, legacy_crops, new_crops = _sample_counts()
    bad = _by_name(check_sample_counts(legacy, new, legacy_crops, {**new_crops, "landscape": 770}))
    assert bad["B2"].status == FAIL and bad["B4"].status == INFO
    bad = _by_name(
        check_sample_counts(legacy, {**new, "after_filter": 1050}, legacy_crops, new_crops)
    )
    assert bad["B1"].status == FAIL


def test_sample_counts_note_drift_of_the_recorded_numbers_without_failing():
    drift = _sample_counts(
        legacy_total=(2300, 1040, 830), legacy_land=760, new_total=(2300, 1040, 1020)
    )
    rows = _by_name(check_sample_counts(*drift))
    assert rows["B1"].status == PASS and rows["B2"].status == PASS  # both still agree
    assert rows["B3"].status == INFO and "may have changed" in rows["B3"].detail


def test_manifest_counts_skip_images_whose_photo_is_missing():
    manifest = pd.DataFrame(
        {
            "image_name": ["A", "A", "A", "B", "B", "C"],
            "stage1_reason": [None, None, "area", None, "aspect", None],
            "keep": [True, False, False, False, False, False],
            "reject_reason": [None, "blurry", "area", "image_missing", "aspect", "image_missing"],
        }
    )
    # B and C have a missing photo: the legacy loop never counted their masks
    assert manifest_counts(manifest) == {"masks": 3, "after_filter": 2, "crops": 1}
    assert manifest_counts(manifest.iloc[:3]) == {"masks": 3, "after_filter": 2, "crops": 1}


def test_counts_by_orientation_counts_crops_per_photo_orientation():
    crops = pd.DataFrame({"image_name": ["L1.JPG", "L1.JPG", "P1.JPG", "L2.JPG"]})
    assert counts_by_orientation(crops, _frames()) == {"landscape": 3, "portrait": 1}
    assert counts_by_orientation(crops.iloc[:1], _frames()) == {"landscape": 1, "portrait": 0}


def _compare(legacy, new, portrait=("P1.JPG",), landscape=("L1.JPG", "L2.JPG")):
    return compare_by_orientation(legacy, new, _frames(landscape, portrait))


def _two_orientations():
    """100 crops on landscape photos and 40 on a portrait photo."""
    names = ["L1.JPG"] * 60 + ["L2.JPG"] * 40 + ["P1.JPG"] * 40
    return _table(140, images=names)


def test_landscape_parity_passes_within_tolerance_whatever_portrait_does():
    a = _two_orientations()
    b = a.copy()
    b[HARD20[3]] += 5e-4  # within 1e-3
    portrait = b["image_name"] == "P1.JPG"
    b = pd.concat([b[~portrait], b[portrait].iloc[:10]])  # portrait: 30 crops only in legacy
    b.loc[b["image_name"] == "P1.JPG", HARD20[0]] += 0.5  # and the rest are very different
    comparisons = _compare(a, b)
    rows = _by_name(check_zero_shot_parity(comparisons, portrait_images=1))
    assert [rows[k].status for k in ("B5", "B6", "B7")] == [PASS, PASS, PASS]
    assert rows["B8"].status == INFO and rows["B8"].actual["only_legacy"] == 30
    assert rows["B8"].actual["images"] == 1 and rows["B8"].actual["max_abs_diff"] > 0.4


def test_landscape_parity_fails_on_each_criterion():
    a = _two_orientations()
    noisy = a.copy()
    noisy[HARD20[3]] += 2e-3
    assert _by_name(check_zero_shot_parity(_compare(a, noisy)))["B7"].status == FAIL

    land = a.index[a["image_name"] != "P1.JPG"]
    flipped = a.copy()
    flipped.loc[land[:2], "predicted_genus"] = [  # 2 of 100 landscape crops: 98% < 99.5%
        "Pavona" if g != "Pavona" else "Porites" for g in a.loc[land[:2], "predicted_genus"]
    ]
    result = _by_name(check_zero_shot_parity(_compare(a, flipped)))
    assert result["B6"].status == FAIL and result["B6"].actual == pytest.approx(0.98)
    assert result["B5"].status == PASS and result["B7"].status == PASS

    missing = _by_name(check_zero_shot_parity(_compare(a, a.iloc[1:])))
    assert missing["B5"].status == FAIL and "only legacy 1" in missing["B5"].actual


def test_one_landscape_disagreement_in_200_is_exactly_the_limit():
    a = _table(200, images=["L1.JPG"] * 200)
    one = a.copy()
    one.loc[0, "predicted_genus"] = (
        "Pavona" if a.loc[0, "predicted_genus"] != "Pavona" else "Porites"
    )
    rows = _by_name(check_zero_shot_parity(_compare(a, one, portrait=())))
    assert rows["B6"].status == PASS  # 199 / 200 = 99.5%
    assert rows["B8"].actual["crops_legacy_coralid"] == [0, 0]


def test_a_sample_without_landscape_photos_cannot_pass_parity():
    a = _table(10, images=["P1.JPG"] * 10)
    rows = _by_name(check_zero_shot_parity(_compare(a, a)))
    assert [rows[k].status for k in ("B5", "B6", "B7")] == [NOT_RUN] * 3
    assert summarize(list(rows.values()))["checks"]["B"]["status"] == "PARTIAL"


def test_drive_csv_comparison_is_informational_only():
    a = _table()
    (row,) = check_against_drive_csv(compare_predictions(a, a.iloc[5:]))
    assert row.name.startswith("B9") and row.status == INFO and row.actual["keys_equal"] is False


# ----------------------------------------------------------------------------- C


def _image_manifest(n_land=30, n_port=10, port_kept=4, land_kept=20):
    """`n_land` masks on two landscape photos and `n_port` on one portrait photo."""
    rows = []
    for orient, n, kept, w, h in [
        ("L", n_land, land_kept, 2560, 1920),
        ("P", n_port, port_kept, 1920, 2560),
    ]:
        for i in range(n):
            reason = None if i < kept else ("image_missing" if i == n - 1 else "area")
            rows.append(
                {
                    "image_name": f"{orient}{i % 2}.JPG",
                    "coral_id": i // 2,
                    "frame_w": w,
                    "frame_h": h,
                    "keep": i < kept,
                    "reject_reason": reason,
                }
            )
    return pd.DataFrame(rows)


def _v2_keys(manifest, drop=None, add=None, portrait_extra=0):
    """V2's keys: the kept landscape keys (optionally changed) and some portrait keys."""
    kept = manifest[manifest["keep"]]
    land = kept[kept["frame_w"] > kept["frame_h"]][["image_name", "coral_id"]]
    if drop is not None:
        land = land.drop(index=land.index[drop])
    keys = [tuple(r) for r in land.to_numpy()]
    keys += [tuple(r) for r in (add or [])]
    keys += [("P0.JPG", 100 + i) for i in range(portrait_extra)]
    return pd.DataFrame(keys, columns=["image_name", "coral_id"])


def test_image_manifest_check_compares_landscape_keys_with_v2():
    manifest = _image_manifest()
    rows = _by_name(
        check_image_manifest(
            manifest,
            v2_keys=_v2_keys(manifest, portrait_extra=3),
            expected={**EXPECTED, "n_masks": 40},
        )
    )
    assert [rows[k].status for k in ("C1", "C2", "C3")] == [PASS, PASS, PASS]
    assert (
        rows["C4"].status == INFO
        and rows["C4"].actual["kept"] == 4
        and rows["C4"].actual["v2"] == 3
    )
    assert rows["C5"].actual == 24 and rows["C5"].expected == 16080 and rows["C5"].status == INFO
    assert rows["C6"].actual["portrait"] == {"kept": 4, "area": 5, "image_missing": 1}
    assert rows["C7"].actual == 2  # one photo per orientation has an image_missing mask


def test_image_manifest_check_fails_on_a_landscape_key_that_differs():
    manifest = _image_manifest()
    expected = {**EXPECTED, "n_masks": 40}
    fewer = _by_name(
        check_image_manifest(manifest, v2_keys=_v2_keys(manifest, drop=[0, 1]), expected=expected)
    )
    assert (
        fewer["C3"].status == FAIL and fewer["C3"].actual == [2, 0]  # two kept keys V2 lacks
    )
    more = _by_name(
        check_image_manifest(
            manifest, v2_keys=_v2_keys(manifest, add=[("L0.JPG", 99)]), expected=expected
        )
    )
    assert more["C3"].status == FAIL and more["C3"].actual == [0, 1]
    assert "L0.JPG#99" in more["C3"].detail
    incomplete = _by_name(check_image_manifest(manifest, [{"image": "X"}], expected=EXPECTED))
    assert incomplete["C1"].status == FAIL and incomplete["C2"].status == FAIL


def test_image_manifest_check_without_the_v2_file_is_partial():
    rows = check_image_manifest(_image_manifest(), expected={**EXPECTED, "n_masks": 40})
    by = _by_name(rows)
    assert by["C3"].status == NOT_RUN and by["C4"].actual == {"kept": 4}
    assert summarize(rows)["checks"]["C"]["status"] == "PARTIAL"


def test_compare_kept_with_v2_counts_each_orientation(tmp_path):
    manifest = _image_manifest()
    v2 = _v2_keys(manifest, drop=[0], add=[("L1.JPG", 77)], portrait_extra=2)
    out = compare_kept_with_v2(manifest, v2)
    land = out["landscape"]
    assert (land["images"], land["kept"], land["v2"], land["both"]) == (2, 20, 20, 19)
    assert (land["only_new"], land["only_v2"]) == (1, 1) and land["only_v2_sample"] == ["L1.JPG#77"]
    assert (
        out["portrait"]["images"] == 2
        and out["portrait"]["v2"] == 2
        and out["portrait"]["both"] == 0
    )
    stray = compare_kept_with_v2(
        manifest, pd.concat([v2, pd.DataFrame({"image_name": ["Z.JPG"], "coral_id": [1]})])
    )
    assert stray["unknown_images"] == {"v2_rows": 1}
    path = tmp_path / "v2.csv"
    pd.DataFrame({"image_name": ["A.JPG"], "coral_id": ["3.0"], "bbox": ["[1, 2, 3, 4]"]}).to_csv(
        path, index=False
    )
    assert load_v2_keys(path).to_dict("records") == [{"image_name": "A.JPG", "coral_id": 3}]


# ----------------------------------------------------------------------------- D


def _expert_report(cells=ASSAF_CSV, n_unidentifiable=16, n_missing=2):
    rows = [(t, p) for t, r in cells.items() for p, n in r.items() for _ in range(n)]
    rows += [("unidentifiable", "Acropora")] * n_unidentifiable
    rows += [("MISSING_LABEL", "Acropora")] * n_missing
    keys = pd.DataFrame(
        {"image_name": [f"I{i}" for i in range(len(rows))], "coral_id": list(range(len(rows)))}
    )
    expert = keys.assign(expert_genus=[t for t, _ in rows])
    predictions = keys.assign(predicted_genus=[p for _, p in rows])
    return evaluate(predictions, expert, "hard20")


def test_the_expected_cells_are_the_memory_matrix_changed_as_the_first_run_found():
    assert EXPERT_CSV_CELLS == ASSAF_CSV
    cells = [(t, p, n) for t, row in EXPERT_CSV_CELLS.items() for p, n in row.items()]
    assert sum(n for _, _, n in cells) == EXPERT_CSV_EXPECTED["identifiable"] == 59
    assert sum(n for t, p, n in cells if t == p) == EXPERT_CSV_EXPECTED["correct"] == 36
    soft = {"Cladiella", "Litophyton", "Sinularia"}
    assert sum(n for t, _, n in cells if t in soft) == EXPERT_CSV_EXPECTED["out_of_set"] == 5
    e = EXPERT_CSV_EXPECTED
    assert e["rows"] == e["identifiable"] + e["no_genus"] == e["in_set"] + e["out_of_set"] + 18
    assert e["no_genus"] == e["unidentifiable"] + e["missing"]


def test_expert_confusion_passes_on_the_real_file_numbers():
    results = check_expert_confusion(_expert_report())
    assert all(r.status in (PASS, INFO) for r in results)
    by = _by_name(results)
    assert [by[k].status for k in ("D1", "D2", "D3", "D5", "D6")] == [PASS] * 5
    assert by["D4"].status == INFO and by["D4"].actual == [16, 2]
    cells = [r for r in results if r.kind == "cell"]
    n_cells = sum(len(r) for r in ASSAF_CSV.values())
    assert n_cells == 34 and len(cells) == n_cells
    assert any(r.name == "cell Seriatopora -> Seriatopora" and r.expected == 2 for r in cells)
    assert by["D6"].name == "D6 confusion matrix cells equal (34 compared)"


def test_the_unidentifiable_missing_split_is_information_unless_strict():
    other_split = _expert_report(n_unidentifiable=15, n_missing=3)  # 18 rows without a genus
    by = _by_name(check_expert_confusion(other_split))
    assert by["D3"].status == PASS and by["D4"].status == INFO and by["D4"].actual == [15, 3]
    strict = _by_name(check_expert_confusion(other_split, strict_split=True))
    assert strict["D4"].status == FAIL
    fewer = _by_name(check_expert_confusion(_expert_report(n_unidentifiable=14, n_missing=2)))
    assert fewer["D1"].status == FAIL and fewer["D3"].status == FAIL


def test_expert_confusion_still_passes_cell_for_cell_on_the_old_matrix():
    expected = {"rows": 78, "in_set": 57, "out_of_set": 5, "no_genus": 16, "unidentifiable": 16}
    expected |= {"missing": 0, "correct": 39, "identifiable": 62}
    results = check_expert_confusion(
        _expert_report(ASSAF, 16, 0), ASSAF, expected, strict_split=True
    )
    assert all(r.status == PASS for r in results)
    assert len([r for r in results if r.kind == "cell"]) == 34


def test_expert_confusion_names_the_cells_that_differ():
    expected = {t: dict(r) for t, r in EXPERT_CSV_CELLS.items()}
    expected["Dipsastraea"]["Favia"] = 3  # the file says 2
    expected["Acropora"]["Pavona"] = 1  # a cell the file does not have
    results = check_expert_confusion(_expert_report(), expected)
    failing = {r.name: (r.expected, r.actual) for r in results if r.status == FAIL}
    assert failing["cell Dipsastraea -> Favia"] == (3, 2)
    assert failing["cell Acropora -> Pavona"] == (1, 0)
    assert failing["D6 confusion matrix cells equal (35 compared)"] == (0, 2)
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


def test_paste_block_is_compact_json_without_confusion_cells():
    results = check_expert_confusion(_expert_report())
    summary = summarize([*results, equal("A", "A1", 1714, 1714)])
    block = json.loads(paste_block(summary, {"commit": "abc123"}))
    assert block["overall"] == "PARTIAL" and block["commit"] == "abc123"
    names = [r["name"] for r in block["checks"]["D"]["rows"]]
    assert names[0].startswith("D1") and not any(n.startswith("cell ") for n in names)
    assert block["checks"]["D"]["status"] == "PASS"
    assert block["checks"]["A"]["rows"] == [
        {"name": "A1", "status": "PASS", "expected": 1714, "actual": 1714}
    ]
    detailed = summarize([equal("B", "B9", 1, 2, "worst crop IMG_1.JPG#3")])
    (row,) = json.loads(paste_block(detailed))["checks"]["B"]["rows"]
    assert row["detail"] == "worst crop IMG_1.JPG#3"
    assert block["checks"]["B"] == {"status": "NOT RUN", "rows": []}


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
