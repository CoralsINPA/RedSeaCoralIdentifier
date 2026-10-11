import json

import numpy as np
import pandas as pd
import pytest

from coralid.genus_audit import (
    FORCED_PORTRAIT,
    FORCED_UNSURE,
    LABEL,
    OTHER_RANDOM,
    PRED_RANDOM,
    PRIOR_DECIDED,
    audit_agreement,
    build_blind_review,
    load_legacy_audit,
)
from coralid.labeling import ingest_review

GENUS = "Millepora"
OTHERS = ["Acropora", "Porites", "Pavona", "Montipora", "Echinopora"]
N_LAND, N_PORT, PER_IMAGE = 10, 4, 25


def _world(seed=0):
    """A manifest and baseline predictions: 10 landscape and 4 portrait photos, 25 crops each,
    with about 20% of the crops called Millepora."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N_LAND + N_PORT):
        portrait = i >= N_LAND
        name = f"{'P' if portrait else 'L'}{i:02d}.JPG"
        for cid in range(PER_IMAGE):
            called = GENUS if rng.random() < 0.2 else OTHERS[int(rng.integers(len(OTHERS)))]
            rows.append(
                {
                    "image_name": name,
                    "coral_id": cid,
                    "frame_w": 1920 if portrait else 2560,
                    "frame_h": 2560 if portrait else 1920,
                    "predicted_genus": called,
                    "confidence": float(rng.uniform(0.3, 1.0)),
                    "runner_up": "Acropora" if called != "Acropora" else "Porites",
                }
            )
    table = pd.DataFrame(rows)
    return table[["image_name", "coral_id", "frame_w", "frame_h"]], table.drop(
        columns=["frame_w", "frame_h"]
    )


def _audit_file(tmp_path, rows):
    """A legacy audit CSV: (image, coral_id, answer, actual genus)."""
    path = tmp_path / "millepora_sharp_audit.csv"
    pd.DataFrame(
        {
            "image_name": [r[0] for r in rows],
            "coral_id": [r[1] for r in rows],
            "bbox": "[1, 2, 3, 4]",
            "area": 30000,
            "confidence": 0.95,
            "runner_up": "Acropora",
            "sharpness": 200.0,
            "is_true_millepora": [r[2] for r in rows],
            "actual_genus": [r[3] for r in rows],
            "notes": "",
        }
    ).to_csv(path, index=False)
    return path


ROWS = [
    ("L00.JPG", 0, "yes", ""),
    ("L00.JPG", 1, "yes", ""),
    ("L01.JPG", 0, "no", "Echinopora"),
    ("L01.JPG", 1, "?", ""),
    ("L02.JPG", 0, "?", ""),
    ("L03.JPG", 0, "yes", ""),
    ("L03.JPG", 1, "no", "Porites"),
    ("P10.JPG", 0, "yes", ""),
    ("P10.JPG", 1, "no", "Pavona"),
    ("P11.JPG", 0, "?", ""),
    ("P11.JPG", 99, "yes", ""),  # not in the baseline
]


def test_load_legacy_audit_reads_the_judgments_and_numbers(tmp_path):
    audit = load_legacy_audit(_audit_file(tmp_path, ROWS))
    assert audit["key"].iloc[0] == "L00.JPG#0"
    assert audit["answer"].tolist()[:5] == ["yes", "yes", "no", "unsure", "unsure"]
    assert audit["confidence"].dtype == float and audit["sharpness"].iloc[0] == 200.0


def _set_calls(preds, calls):
    """Force the baseline's call for some crops: {(image, id): genus}."""
    preds = preds.copy()
    for (name, cid), genus in calls.items():
        preds.loc[(preds["image_name"] == name) & (preds["coral_id"] == cid), "predicted_genus"] = (
            genus
        )
    return preds


def test_audit_agreement_is_a_range_and_keeps_portrait_apart(tmp_path):
    manifest, preds = _world()
    preds = _set_calls(
        preds,
        {
            ("L00.JPG", 0): GENUS,  # yes, and called Millepora: agree
            ("L00.JPG", 1): "Porites",  # yes, but called Porites: disagree
            ("L01.JPG", 0): "Porites",  # no, and not Millepora: agree
            ("L01.JPG", 1): GENUS,  # unsure, called Millepora
            ("L02.JPG", 0): "Acropora",  # unsure, not called
            ("L03.JPG", 0): GENUS,  # yes, called: agree
            ("L03.JPG", 1): GENUS,  # no, but called Millepora: disagree
            ("P10.JPG", 0): GENUS,
            ("P10.JPG", 1): GENUS,
            ("P11.JPG", 0): "Porites",
        },
    )
    audit = load_legacy_audit(_audit_file(tmp_path, ROWS))
    out = audit_agreement(audit, preds, manifest, GENUS)
    assert out["label"] == LABEL and "not a precision" in LABEL and out["genus"] == GENUS
    land = out["landscape"]
    assert (land["audit_rows"], land["in_baseline"], land["judged"]) == (7, 7, 7)
    assert land["agreement"] == {
        "n": 7,
        "unsure_as_wrong": pytest.approx(3 / 7),  # three agreements
        "unsure_as_right": pytest.approx(5 / 7),  # and the two unsure
    }
    grid = land["analyst_answer_by_new_call"]
    assert grid["yes"] == {"genus": 2, "other": 1} and grid["no"] == {"genus": 1, "other": 1}
    assert grid["unsure"] == {"genus": 1, "other": 1}
    calls = land["new_genus_calls"]
    assert (calls["n"], calls["analyst_yes"], calls["analyst_no"], calls["analyst_unsure"]) == (
        4,
        2,
        1,
        1,
    )
    assert calls["share_yes_unsure_wrong"] == 0.5 and calls["share_yes_unsure_right"] == 0.75
    port = out["portrait"]
    assert port["comparable"] is False and "wrong region" in port["why"]
    assert (port["audit_rows"], port["in_baseline"], port["not_in_baseline"]) == (4, 3, 1)
    json.dumps(out)


# ----------------------------------------------------------------------------- the blind sheet


@pytest.fixture
def world_and_audit(tmp_path):
    manifest, preds = _world()
    audit = load_legacy_audit(_audit_file(tmp_path, ROWS))
    return manifest, preds, audit


def test_the_sheet_shows_nothing_but_the_key_and_blank_answers(world_and_audit):
    manifest, preds, audit = world_and_audit
    review = build_blind_review(preds, manifest, GENUS, audit, n_pred=15, n_other=15, seed=3)
    assert list(review.sheet.columns) == ["key", "is_millepora", "actual_genus", "notes"]
    assert review.answer == "is_millepora"
    assert (review.sheet[["is_millepora", "actual_genus", "notes"]] == "").all().all()
    assert review.sheet["key"].is_unique
    text = review.sheet.to_csv()
    assert "confidence" not in text and "runner_up" not in text and "Acropora" not in text


def test_the_sample_is_random_mixed_and_leaves_out_what_was_judged(world_and_audit):
    manifest, preds, audit = world_and_audit
    review = build_blind_review(preds, manifest, GENUS, audit, n_pred=15, n_other=15, seed=3)
    key = review.key
    on_sheet = key[key["selection"] != PRIOR_DECIDED]
    assert on_sheet["key"].tolist() == review.sheet["key"].tolist()  # sheet order is the key's
    by = on_sheet["selection"].value_counts().to_dict()
    # forced: L01#1 and L02#0 (old unsure, landscape) and the 3 portrait rows in the baseline
    assert by[FORCED_UNSURE] == 2 and by[FORCED_PORTRAIT] == 3
    assert by[PRED_RANDOM] == 15 and by[OTHER_RANDOM] == 15
    random_pred = on_sheet[on_sheet["selection"] == PRED_RANDOM]
    random_other = on_sheet[on_sheet["selection"] == OTHER_RANDOM]
    assert (random_pred["new_pred"] == GENUS).all() and (random_other["new_pred"] != GENUS).all()
    # already judged on a landscape photo (yes / no): not on the sheet, kept in the key
    prior = key[key["selection"] == PRIOR_DECIDED]
    assert sorted(prior["key"]) == ["L00.JPG#0", "L00.JPG#1", "L01.JPG#0", "L03.JPG#0", "L03.JPG#1"]
    assert prior["sheet_order"].isna().all() and not set(prior["key"]) & set(review.sheet["key"])
    # portrait rows are reviewed again whatever they were judged, except the one not in the baseline
    port = on_sheet[on_sheet["selection"] == FORCED_PORTRAIT]
    assert sorted(port["key"]) == ["P10.JPG#0", "P10.JPG#1", "P11.JPG#0"]
    assert port.set_index("key").loc["P10.JPG#0", "prior_answer"] == "yes"
    assert review.meta["audit_rows_not_in_baseline"] == 1 and review.meta["n_prior_decided"] == 5
    assert len(review.sheet) == review.meta["n_on_sheet"] == 35


def test_the_pools_exclude_judged_and_forced_crops(world_and_audit):
    manifest, preds, audit = world_and_audit
    review = build_blind_review(preds, manifest, GENUS, audit, n_pred=15, n_other=15, seed=3)
    called = int((preds["predicted_genus"] == GENUS).sum())
    judged_or_forced = set(audit["key"]) - {"P11.JPG#99"}
    taken_called = sum(
        1
        for k in judged_or_forced
        if preds.set_index(preds["image_name"] + "#" + preds["coral_id"].astype(str)).loc[
            k, "predicted_genus"
        ]
        == GENUS
    )
    assert review.meta["n_pool_pred"] == called - taken_called
    assert review.meta["n_pool_pred"] + review.meta["n_pool_other"] == len(preds) - len(
        judged_or_forced
    )


def test_the_expert_subset_has_every_unsure_crop_and_about_forty(world_and_audit):
    manifest, preds, audit = world_and_audit
    review = build_blind_review(preds, manifest, GENUS, audit, n_pred=60, n_other=60, seed=1)
    key = review.key.set_index("key")
    assaf = review.assaf_sheet
    assert set(assaf["key"]) <= set(review.sheet["key"]) and assaf["key"].is_unique
    assert len(assaf) == 40 == review.meta["n_for_assaf"]
    for unsure in ("L01.JPG#1", "L02.JPG#0", "P11.JPG#0"):  # the old audit's unsure crops
        assert unsure in set(assaf["key"])
    assert key.loc[list(assaf["key"]), "for_assaf"].all()
    assert list(assaf.columns) == list(review.sheet.columns)
    assert assaf["key"].tolist() != sorted(assaf["key"]) or len(assaf) < 3  # shuffled
    many_unsure = build_blind_review(preds, manifest, GENUS, audit, n_pred=5, n_other=5, n_assaf=2)
    assert len(many_unsure.assaf_sheet) >= 3  # the unsure crops are never dropped


def test_the_same_seed_gives_the_same_sheets_and_another_seed_does_not(world_and_audit):
    manifest, preds, audit = world_and_audit
    a = build_blind_review(preds, manifest, GENUS, audit, n_pred=20, n_other=20, seed=5)
    b = build_blind_review(preds, manifest, GENUS, audit, n_pred=20, n_other=20, seed=5)
    c = build_blind_review(preds, manifest, GENUS, audit, n_pred=20, n_other=20, seed=6)
    pd.testing.assert_frame_equal(a.sheet, b.sheet)
    pd.testing.assert_frame_equal(a.assaf_sheet, b.assaf_sheet)
    pd.testing.assert_frame_equal(a.key, b.key)
    assert a.sheet["key"].tolist() != c.sheet["key"].tolist()


def test_without_an_audit_nothing_is_excluded_or_forced(world_and_audit):
    manifest, preds, _ = world_and_audit
    review = build_blind_review(preds, manifest, GENUS, None, n_pred=10, n_other=10, seed=2)
    assert review.key["selection"].isin([PRED_RANDOM, OTHER_RANDOM]).all()
    assert len(review.sheet) == 20 and review.meta["n_prior_decided"] == 0
    assert review.meta["n_pool_pred"] + review.meta["n_pool_other"] == len(preds)


def test_a_pool_smaller_than_the_request_is_taken_whole(world_and_audit):
    manifest, preds, _ = world_and_audit
    called = int((preds["predicted_genus"] == GENUS).sum())
    review = build_blind_review(preds, manifest, GENUS, None, n_pred=10_000, n_other=3, seed=2)
    assert review.meta["n_by_selection"][PRED_RANDOM] == called
    assert review.meta["n_by_selection"][OTHER_RANDOM] == 3


def test_the_completed_sheet_comes_back_through_ingest(world_and_audit, tmp_path):
    from coralid.labeling import export_review_sheet

    manifest, preds, audit = world_and_audit
    review = build_blind_review(preds, manifest, GENUS, audit, n_pred=6, n_other=6, seed=4)
    path = export_review_sheet(review.sheet["key"].tolist(), tmp_path / "s.csv", review.answer)
    sheet = pd.read_csv(path, keep_default_na=False)
    sheet.loc[0, "is_millepora"] = "yes"
    sheet.loc[1, "is_millepora"] = "no"
    sheet.loc[1, "actual_genus"] = "Porites"
    sheet.to_csv(path, index=False)
    back = ingest_review(path, review.answer)
    assert back.table["key"].tolist() == review.sheet["key"].tolist() and back.problems.empty
    assert back.counts()["yes"] == 1 and back.counts()["pending"] == len(sheet) - 2
