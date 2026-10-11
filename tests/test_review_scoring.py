import json

import numpy as np
import pandas as pd
import pytest

from coralid.evaluation import wilson_interval
from coralid.genus_audit import (
    FORCED_PORTRAIT,
    FORCED_UNSURE,
    OTHER_RANDOM,
    PRED_RANDOM,
    PRIOR_DECIDED,
    build_blind_review,
)
from coralid.labeling import ingest_review
from coralid.review_scoring import pooled_rate, score_blind_review

GENUS = "Millepora"

# ----------------------------------------------------------------------------- pooled_rate


def test_pooled_rate_scales_the_wilson_interval_of_the_sample_to_the_pool():
    out = pooled_rate(census=(4, 6), sample=(6, 10), pool_size=100)
    lo, hi = wilson_interval(6, 10)
    assert out["point"] == pytest.approx((4 + 100 * 0.6) / 106)
    assert out["lo"] == pytest.approx((4 + 100 * lo) / 106)
    assert out["hi"] == pytest.approx((4 + 100 * hi) / 106)
    assert (out["n_sample"], out["n_census"], out["pool"]) == (10, 6, 100)


def test_a_sample_that_is_the_whole_pool_has_no_sampling_error():
    out = pooled_rate(census=(2, 4), sample=(3, 6), pool_size=6)  # all six pool crops reviewed
    assert out["point"] == out["lo"] == out["hi"] == pytest.approx(5 / 10)
    assert (out["n_census"], out["pool"]) == (10, 0)


def test_pooled_rate_edge_cases():
    assert pooled_rate((3, 4), (0, 0), 0)["point"] == 0.75  # everything reviewed in full
    assert pooled_rate((0, 0), (1, 2), 0) is None  # nothing at all
    assert pooled_rate((3, 4), (0, 0), 50) is None  # a pool but no sample answers yet
    none_yes = pooled_rate((0, 0), (0, 20), 1000)
    assert none_yes["point"] == 0 and none_yes["lo"] == 0 and 0 < none_yes["hi"] < 0.2


# ----------------------------------------------------------------------------- hand-made key


def _key(rows):
    """rows: (key, selection, new_pred, prior_answer) -> an answer key."""
    return pd.DataFrame(rows, columns=["key", "selection", "new_pred", "prior_answer"])


def _analyst(answers, genera=None):
    """A completed analyst sheet from {key: answer} (and {key: actual genus})."""
    genera = genera or {}
    sheet = pd.DataFrame(
        {
            "key": list(answers),
            "is_millepora": [answers[k] for k in answers],
            "actual_genus": [genera.get(k, "") for k in answers],
            "notes": "",
        }
    )
    return ingest_review(sheet, "is_millepora")


def _scenario():
    """The hand example of the docs: precision census 3 yes + 1 no (prior) and 2 forced (yes,
    unsure), a sample of 10 (6 yes, 3 no, 1 unsure) from a pool of 100; the other population
    has a prior census of 6 no, one forced no, and a sample of 20 (1 yes) from a pool of 900."""
    rows, answers, genera = [], {}, {}
    for i, a in enumerate(["yes", "yes", "yes", "no"]):
        rows.append((f"A.JPG#{i}", PRIOR_DECIDED, GENUS, a))
    rows += [
        ("F.JPG#0", FORCED_UNSURE, GENUS, "unsure"),
        ("F.JPG#1", FORCED_PORTRAIT, GENUS, "yes"),
    ]
    answers |= {"F.JPG#0": "yes", "F.JPG#1": "unsure"}
    for i, a in enumerate(["yes"] * 6 + ["no"] * 3 + ["unsure"]):
        rows.append((f"S.JPG#{i}", PRED_RANDOM, GENUS, None))
        answers[f"S.JPG#{i}"] = a
        if a == "no":
            genera[f"S.JPG#{i}"] = ["Echinopora", "Echinopora", "Sinularia"][i - 6]
    for i in range(6):
        rows.append((f"B.JPG#{i}", PRIOR_DECIDED, "Porites", "no"))
    rows.append(("G.JPG#0", FORCED_PORTRAIT, "Pavona", "yes"))
    answers["G.JPG#0"] = "no"
    genera["G.JPG#0"] = "Pavona"
    for i in range(20):
        rows.append((f"O.JPG#{i}", OTHER_RANDOM, "Acropora" if i < 12 else "Porites", None))
        answers[f"O.JPG#{i}"] = "yes" if i == 3 else "no"
        if i != 3:
            genera[f"O.JPG#{i}"] = "Acropora"
    meta = {"genus": GENUS, "n_pool_pred": 100, "n_pool_other": 900}
    return _key(rows), _analyst(answers, genera), meta


def test_precision_miss_rate_and_recall_follow_the_hand_calculation():
    key, analyst, meta = _scenario()
    score = score_blind_review(analyst, key, meta)
    low = score.estimates["unsure_as_no"]
    # precision: census yes 3 + 1 = 4 of 6; sample 6 of 10 over a pool of 100
    assert low["precision"]["point"] == pytest.approx((4 + 100 * 0.6) / 106)
    # miss rate: census 0 yes of 7, sample 1 of 20 over a pool of 900
    assert low["miss_rate"]["point"] == pytest.approx((0 + 900 * 0.05) / 907)
    hits, misses = 4 + 60, 0 + 45
    assert low["recall"]["point"] == pytest.approx(hits / (hits + misses))
    assert low["recall"]["lo"] < low["recall"]["point"] < low["recall"]["hi"]
    high = score.estimates["unsure_as_yes"]  # forced unsure + 1 sample unsure count as yes
    assert high["precision"]["point"] == pytest.approx((5 + 100 * 0.7) / 106)
    assert high["precision"]["point"] > low["precision"]["point"]
    assert high["miss_rate"]["point"] == low["miss_rate"]["point"]  # no unsure among the others
    json.dumps(score.to_dict(), allow_nan=False)


def test_pending_answers_are_left_out_not_counted_as_no():
    key, analyst, meta = _scenario()
    answers = analyst.table.set_index("key")["answer"].to_dict()
    for i in range(6, 10):
        answers[f"S.JPG#{i}"] = ""  # the last four sample crops are not answered yet
    pending = _analyst(answers)
    score = score_blind_review(pending, key, meta)
    assert score.answered["pending"] == 4
    precision = score.estimates["unsure_as_no"]["precision"]
    assert precision["n_sample"] == 6 and precision["point"] == pytest.approx((4 + 100 * 1.0) / 106)


def test_a_sheet_with_no_sample_answers_gives_no_estimate_yet():
    key, _, meta = _scenario()
    empty = _analyst({"S.JPG#0": ""})
    score = score_blind_review(empty, key, meta)
    assert score.estimates["unsure_as_no"]["precision"] is None
    assert score.estimates["unsure_as_no"]["recall"] is None
    assert score.answered["pending"] == 33 and score.expert_agreement is None


def test_the_confusion_lists_come_from_actual_genus():
    key, analyst, meta = _scenario()
    score = score_blind_review(analyst, key, meta)
    # three "no" among the sample's genus calls: Echinopora x2, Sinularia x1
    assert score.false_positive_genera.set_index("actual_genus")["count"].to_dict() == {
        "Echinopora": 2,
        "Sinularia": 1,
    }
    # one other call was really the genus: the baseline called it Acropora
    assert score.missed_as.to_dict("records") == [{"called_instead": "Acropora", "count": 1}]


def test_an_expert_answer_replaces_the_analysts_in_the_confusion_lists():
    key, analyst, meta = _scenario()
    expert = _analyst({"S.JPG#6": "yes", "S.JPG#7": "no"}, {"S.JPG#7": "Porites"})
    score = score_blind_review(analyst, key, meta, expert)
    counts = score.false_positive_genera.set_index("actual_genus")["count"].to_dict()
    # the expert said #6 is the genus (no longer a false positive) and #7 is Porites, not Echinopora
    assert counts == {"Sinularia": 1, "Porites": 1}


def test_a_no_without_an_actual_genus_is_listed_as_not_given():
    key, _, meta = _scenario()
    analyst = _analyst({"S.JPG#6": "no"})
    score = score_blind_review(analyst, key, meta)
    assert score.false_positive_genera.to_dict("records") == [
        {"actual_genus": "(not given)", "count": 1}
    ]


# ----------------------------------------------------------------------------- analyst vs expert


def test_agreement_with_the_expert_by_count_and_kappa():
    key, _, meta = _scenario()
    ids = [f"S.JPG#{i}" for i in range(10)]
    analyst = ["yes", "yes", "no", "no", "unsure", "yes", "no", "yes", "no", "unsure"]
    expert = ["yes", "no", "no", "no", "yes", "yes", "no", "yes", "unsure", "unsure"]
    score = score_blind_review(
        _analyst(dict(zip(ids, analyst, strict=True))),
        key,
        meta,
        _analyst(dict(zip(ids, expert, strict=True))),
    )
    agreement = score.expert_agreement
    assert (agreement["n_both_answered"], agreement["agree"]) == (10, 7)
    assert agreement["agree_share"] == pytest.approx(0.7)
    assert (agreement["n_decided_by_both"], agreement["decided_agree_share"]) == (
        7,
        pytest.approx(6 / 7),
    )
    assert agreement["kappa"] == pytest.approx((0.7 - 0.36) / 0.64)
    assert agreement["analyst_rows_expert_columns"]["yes"] == {"yes": 3, "no": 1, "unsure": 0}
    assert agreement["agree_share_ci95"] == wilson_interval(7, 10)


def test_agreement_only_counts_crops_both_people_answered():
    key, _, meta = _scenario()
    analyst = _analyst({"S.JPG#0": "yes", "S.JPG#1": "no", "S.JPG#2": "yes"})
    expert = _analyst({"S.JPG#0": "yes", "S.JPG#1": "yes", "S.JPG#9": "no"})
    agreement = score_blind_review(analyst, key, meta, expert).expert_agreement
    assert agreement["n_both_answered"] == 2 and agreement["agree"] == 1
    assert (
        score_blind_review(analyst, key, meta, _analyst({"S.JPG#9": "no"})).expert_agreement[
            "agree_share"
        ]
        is None
    )


# ----------------------------------------------------------------------------- end to end


def test_a_perfect_simulated_analyst_gets_the_truth_inside_the_interval():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        for cid in range(40):
            rows.append(
                {
                    "image_name": f"L{i:02d}.JPG",
                    "coral_id": cid,
                    "frame_w": 2560,
                    "frame_h": 1920,
                    "predicted_genus": GENUS if rng.random() < 0.15 else "Acropora",
                    "confidence": float(rng.random()),
                }
            )
    table = pd.DataFrame(rows)
    manifest = table[["image_name", "coral_id", "frame_w", "frame_h"]]
    preds = table.drop(columns=["frame_w", "frame_h"])
    # the truth: 80% of the genus calls are right, and 3% of the others are the genus
    called = preds["predicted_genus"] == GENUS
    truth = np.where(called, rng.random(len(preds)) < 0.8, rng.random(len(preds)) < 0.03)
    uid = preds["image_name"] + "#" + preds["coral_id"].astype(str)
    is_true = dict(zip(uid, truth, strict=True))

    review = build_blind_review(preds, manifest, GENUS, None, n_pred=150, n_other=300, seed=7)
    answers = {k: ("yes" if is_true[k] else "no") for k in review.sheet["key"]}
    genera = {k: "Porites" for k in answers if not is_true[k]}
    score = score_blind_review(_analyst(answers, genera), review.key, review.meta)

    true_precision = truth[called.to_numpy()].mean()
    true_miss = truth[~called.to_numpy()].mean()
    true_recall = truth[called.to_numpy()].sum() / truth.sum()
    est = score.estimates["unsure_as_no"]
    assert est["precision"]["lo"] <= true_precision <= est["precision"]["hi"]
    assert est["miss_rate"]["lo"] <= true_miss <= est["miss_rate"]["hi"]
    assert est["recall"]["lo"] <= true_recall <= est["recall"]["hi"]
    assert est["precision"]["point"] == pytest.approx(true_precision, abs=0.08)
    assert est["precision"] == score.estimates["unsure_as_yes"]["precision"]  # no unsure answers
    assert score.false_positive_genera["actual_genus"].tolist() == ["Porites"]


def test_the_score_is_saved_as_a_new_folder_every_time(tmp_path):
    from coralid.review_scoring import write_blind_score

    key, analyst, meta = _scenario()
    score = score_blind_review(analyst, key, meta)
    first = write_blind_score(score, tmp_path, "millepora_blind_x", date="2026-10-12")
    second = write_blind_score(score, tmp_path, "millepora_blind_x", date="2026-10-12")
    assert first.name == "blind_score_millepora_blind_x_2026-10-12"
    assert second.name == "blind_score_millepora_blind_x_2026-10-12_2"
    assert sorted(p.name for p in first.iterdir()) == [
        "false_positive_genera.csv",
        "missed_as.csv",
        "summary.json",
    ]
    body = json.loads((first / "summary.json").read_text())
    assert body["genus"] == GENUS and body["answered"]["yes"] > 0
