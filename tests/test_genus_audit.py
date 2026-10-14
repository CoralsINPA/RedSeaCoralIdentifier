from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from coralid.genus_audit import OTHER_RANDOM, PRED_RANDOM, build_blind_review
from coralid.labeling import ingest_review
from coralid.review_source import PredictionSource, ReviewSourceError

GENUS = "Millepora"
OTHERS = ["Acropora", "Porites", "Pavona", "Montipora", "Echinopora"]
N_LAND, N_PORT, PER_IMAGE = 10, 4, 25
SOURCE = PredictionSource(
    file="finetuned-bioclip__taxonomy_v1__hard20__all.parquet",
    path="/drive/run/predictions/finetuned-bioclip__taxonomy_v1__hard20__all.parquet",
    git_commit="abc123",
    created_utc="2026-10-12T20:00:00Z",
    model="finetuned-bioclip",
    prompt_set="taxonomy_v1",
    label_set="hard20",
    n_rows=(N_LAND + N_PORT) * PER_IMAGE,
)


SOURCE_SMALL = replace(SOURCE, n_rows=6)


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
    manifest = table[["image_name", "coral_id", "frame_w", "frame_h"]].assign(keep=True)
    return manifest, table.drop(columns=["frame_w", "frame_h"])


# ----------------------------------------------------------------------------- the blind sheet


@pytest.fixture
def world():
    return _world()


def test_the_sheet_shows_nothing_but_the_key_and_blank_answers(world):
    manifest, preds = world
    review = build_blind_review(preds, manifest, GENUS, n_pred=15, n_other=15, seed=3)
    assert list(review.sheet.columns) == ["key", "is_millepora", "actual_genus", "notes"]
    assert review.answer == "is_millepora"
    assert (review.sheet[["is_millepora", "actual_genus", "notes"]] == "").all().all()
    assert review.sheet["key"].is_unique
    text = review.sheet.to_csv()
    assert "confidence" not in text and "runner_up" not in text and "Acropora" not in text


def test_the_sample_is_random_mixed_and_drawn_from_the_predictions_only(world):
    manifest, preds = world
    review = build_blind_review(preds, manifest, GENUS, n_pred=15, n_other=15, seed=3)
    key = review.key
    assert key["key"].tolist() == review.sheet["key"].tolist()  # sheet order is the key's
    assert key["selection"].value_counts().to_dict() == {PRED_RANDOM: 15, OTHER_RANDOM: 15}
    called = key[key["selection"] == PRED_RANDOM]
    other = key[key["selection"] == OTHER_RANDOM]
    assert (called["baseline_pred"] == GENUS).all() and (other["baseline_pred"] != GENUS).all()
    assert set(key["key"]) <= set(preds["image_name"] + "#" + preds["coral_id"].astype(str))
    assert key["orientation"].isin(["landscape", "portrait"]).all()
    assert review.meta["n_pool_pred"] + review.meta["n_pool_other"] == len(preds)
    assert len(review.sheet) == review.meta["n_on_sheet"] == 30
    assert "prior_answer" not in key.columns and "audit_rows_not_in_baseline" not in review.meta


def test_the_expert_subset_is_forty_crops_of_the_sheet_in_its_own_order(world):
    manifest, preds = world
    review = build_blind_review(preds, manifest, GENUS, n_pred=60, n_other=60, seed=1)
    key = review.key.set_index("key")
    assaf = review.assaf_sheet
    assert set(assaf["key"]) <= set(review.sheet["key"]) and assaf["key"].is_unique
    assert len(assaf) == 40 == review.meta["n_for_assaf"]
    assert key.loc[list(assaf["key"]), "for_assaf"].all()
    assert list(assaf.columns) == list(review.sheet.columns)
    assert assaf["key"].tolist() != sorted(assaf["key"])  # shuffled
    small = build_blind_review(preds, manifest, GENUS, n_pred=5, n_other=5, n_assaf=2)
    assert len(small.assaf_sheet) == 2
    assert (
        len(build_blind_review(preds, manifest, GENUS, n_pred=5, n_other=5, n_assaf=0).assaf_sheet)
        == 0
    )


def test_the_same_seed_gives_the_same_sheets_and_another_seed_does_not(world):
    manifest, preds = world
    a = build_blind_review(preds, manifest, GENUS, n_pred=20, n_other=20, seed=5)
    b = build_blind_review(preds, manifest, GENUS, n_pred=20, n_other=20, seed=5)
    c = build_blind_review(preds, manifest, GENUS, n_pred=20, n_other=20, seed=6)
    pd.testing.assert_frame_equal(a.sheet, b.sheet)
    pd.testing.assert_frame_equal(a.assaf_sheet, b.assaf_sheet)
    pd.testing.assert_frame_equal(a.key, b.key)
    assert a.sheet["key"].tolist() != c.sheet["key"].tolist()


def test_a_pool_smaller_than_the_request_is_taken_whole(world):
    manifest, preds = world
    called = int((preds["predicted_genus"] == GENUS).sum())
    review = build_blind_review(preds, manifest, GENUS, n_pred=10_000, n_other=3, seed=2)
    assert review.meta["n_by_selection"][PRED_RANDOM] == called
    assert review.meta["n_by_selection"][OTHER_RANDOM] == 3


def test_the_completed_sheet_comes_back_through_ingest(world, tmp_path):
    from coralid.labeling import export_review_sheet

    manifest, preds = world
    review = build_blind_review(preds, manifest, GENUS, n_pred=6, n_other=6, seed=4)
    path = export_review_sheet(review.sheet["key"].tolist(), tmp_path / "s.csv", review.answer)
    sheet = pd.read_csv(path, keep_default_na=False)
    sheet.loc[0, "is_millepora"] = "yes"
    sheet.loc[1, "is_millepora"] = "no"
    sheet.loc[1, "actual_genus"] = "Porites"
    sheet.to_csv(path, index=False)
    back = ingest_review(path, review.answer)
    assert back.table["key"].tolist() == review.sheet["key"].tolist() and back.problems.empty
    assert back.counts()["yes"] == 1 and back.counts()["pending"] == len(sheet) - 2


# ----------------------------------------------------------------------------- the source


def test_the_review_records_where_its_predictions_came_from(world):
    manifest, preds = world
    review = build_blind_review(preds, manifest, GENUS, n_pred=5, n_other=5, source=SOURCE)
    assert review.meta["source"] == SOURCE.to_dict()
    assert review.meta["source"]["label_set"] == "hard20"
    assert build_blind_review(preds, manifest, GENUS, n_pred=5, n_other=5).meta["source"] is None


def test_a_sampled_crop_that_is_not_a_kept_manifest_row_is_refused(world):
    manifest, preds = world
    dropped = manifest.copy()
    dropped.loc[dropped["image_name"] == "L00.JPG", "keep"] = False
    with pytest.raises(ReviewSourceError, match="not kept rows of the manifest"):
        build_blind_review(preds, dropped, GENUS, n_pred=10_000, n_other=10_000)
    with pytest.raises(ReviewSourceError, match="no 'keep' column"):
        build_blind_review(preds, manifest.drop(columns="keep"), GENUS, n_pred=3, n_other=3)


def test_predictions_that_are_not_the_source_file_are_refused(world):
    manifest, preds = world
    with pytest.raises(ReviewSourceError, match="they are not that file"):
        build_blind_review(preds.head(100), manifest, GENUS, n_pred=5, n_other=5, source=SOURCE)


# ----------------------------------------------------------------------------- writing it out


def test_write_blind_review_hands_out_sheets_and_keeps_the_key_apart(tmp_path):
    pytest.importorskip("matplotlib")
    import cv2

    from coralid.masks import AnnotationStore
    from coralid.review_export import blind_review_name, read_blind_key, write_blind_review
    from coralid.review_scoring import score_blind_review
    from tests.helpers import make_annotation, rect_mask, write_json_dir

    # a tiny world with real photos: one landscape and one portrait image, 3 crops each
    images = tmp_path / "images"
    images.mkdir()
    entries, rows = [], []
    for name, (w, h) in (("L00.JPG", (200, 150)), ("P01.JPG", (150, 200))):
        photo = np.random.RandomState(0).randint(0, 256, (h, w, 3), dtype=np.uint8)
        assert cv2.imwrite(str(images / name), photo)
        masks = [rect_mask((h, w), 10 + 40 * i, 40 + 40 * i, 10, 80) for i in range(3)]
        entries.append((name, w, h, [make_annotation(m, i) for i, m in enumerate(masks)], None))
        for i in range(3):
            rows.append((name, i, w, h, GENUS if i == 0 else "Acropora"))
    store = AnnotationStore(write_json_dir(tmp_path / "json", entries))
    table = pd.DataFrame(
        rows, columns=["image_name", "coral_id", "frame_w", "frame_h", "predicted_genus"]
    )
    manifest = table[["image_name", "coral_id", "frame_w", "frame_h"]]
    preds = table[["image_name", "coral_id", "predicted_genus"]].assign(confidence=0.9)
    manifest = manifest.assign(keep=True)
    review = build_blind_review(
        preds, manifest, GENUS, n_pred=2, n_other=2, n_assaf=2, seed=1, source=SOURCE_SMALL
    )

    name = blind_review_name(GENUS, 1, "2026-10-12")
    assert name == "millepora_blind_2026-10-12_seed1"
    paths = write_blind_review(
        review, tmp_path / "exports", tmp_path / "keys", store, images, name=name
    )
    assert sorted(p.name for p in (tmp_path / "exports").iterdir()) == [
        f"{name}_analyst.csv",
        f"{name}_analyst.pdf",
        f"{name}_assaf.csv",
        f"{name}_assaf.pdf",
    ]
    assert sorted(p.name for p in (tmp_path / "keys").iterdir()) == [
        f"{name}_answer_key.csv",
        f"{name}_answer_key.meta.json",
    ]
    assert paths["analyst_pdf"].read_bytes().startswith(b"%PDF")
    sheet = pd.read_csv(paths["analyst_csv"], keep_default_na=False)
    assert sheet["key"].tolist() == review.sheet["key"].tolist()
    assert "baseline_pred" not in sheet.columns and "selection" not in sheet.columns
    key, meta = read_blind_key(tmp_path / "keys", name)
    assert (
        meta["genus"] == GENUS and meta["name"] == name and meta["answer_column"] == "is_millepora"
    )
    pd.testing.assert_frame_equal(
        key[["key", "selection", "baseline_pred"]],
        review.key[["key", "selection", "baseline_pred"]],
        check_dtype=False,
    )
    # nothing is overwritten, and nothing is written when a target exists
    with pytest.raises(FileExistsError, match="change the seed"):
        write_blind_review(
            review, tmp_path / "exports", tmp_path / "keys", store, images, name=name
        )
    other = tmp_path / "other"
    paths["analyst_csv"].rename(tmp_path / "moved.csv")
    with pytest.raises(FileExistsError):
        write_blind_review(review, tmp_path / "exports", other, store, images, name=name)
    assert not other.exists()  # the existing PDF stopped it before the key was written
    # the key read back from disk scores like the one in memory
    answers = {k: "yes" for k in review.sheet["key"]}
    sheet_in = pd.DataFrame(
        {"key": list(answers), "is_millepora": list(answers.values()), "actual_genus": ""}
    )
    done = ingest_review(sheet_in, "is_millepora")
    from_disk = score_blind_review(done, key, meta, current_source=SOURCE_SMALL)
    in_memory = score_blind_review(done, review.key, review.meta, current_source=SOURCE_SMALL)
    assert from_disk.estimates == in_memory.estimates
    assert meta["source"] == SOURCE_SMALL.to_dict()  # the key's meta file records the source
