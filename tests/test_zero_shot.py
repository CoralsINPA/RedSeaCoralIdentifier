import math
from dataclasses import replace

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from coralid.batch import ResumableRun
from coralid.config import CropConfig, FilterConfig
from coralid.crops import masked_crop
from coralid.genera import PromptSet, load_genera
from coralid.images import load_in_mask_frame
from coralid.manifest import image_manifest
from coralid.masks import load_mask_json
from coralid.zero_shot import (
    CropError,
    build_text_features,
    l2_normalize,
    load_embeddings,
    predictions_table,
    run_zero_shot,
    save_embeddings,
    score,
)
from tests.fake_encoder import FakeEncoder
from tests.helpers import make_annotation, rect_mask, write_json_dir

HARD20 = load_genera().genera("hard20")

# ----------------------------------------------------------------------------- scoring


def test_l2_normalize_rows():
    x = np.array([[3.0, 4.0], [0.0, 2.0], [0.0, 0.0]])
    out = l2_normalize(x)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:2], [[0.6, 0.8], [0.0, 1.0]])
    assert np.isfinite(out).all()  # the zero row stays zero, not NaN


def test_score_matches_a_hand_computed_softmax():
    """One image at angle 0, three genera whose cosines to it are 0.95, 0.94 and 0.90."""
    cosines = [0.95, 0.94, 0.90]
    text = np.array([[c, math.sqrt(1 - c * c)] for c in cosines])
    image = np.array([[2.0, 0.0]])  # not unit length on purpose: score normalizes
    expected = [math.exp(100 * c) for c in cosines]
    expected = [e / sum(expected) for e in expected]  # = [1, e^-1, e^-5] / (1 + e^-1 + e^-5)
    probs = score(image, text * 3.0)  # text not unit length either
    assert probs.dtype == np.float32 and probs.shape == (1, 3)
    np.testing.assert_allclose(probs[0], expected, rtol=1e-4)
    assert expected[0] == pytest.approx(1 / (1 + math.exp(-1) + math.exp(-5)))


def test_score_scale_and_batches():
    text = np.eye(3)
    images = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    probs = score(images, text, logit_scale=1.0)
    assert probs.sum(axis=1) == pytest.approx([1.0, 1.0])
    e = math.exp(1.0)
    np.testing.assert_allclose(probs[0], [e / (e + 2), 1 / (e + 2), 1 / (e + 2)], rtol=1e-5)
    s = math.exp(math.sqrt(0.5))
    np.testing.assert_allclose(
        probs[1], [1 / (1 + 2 * s), s / (1 + 2 * s), s / (1 + 2 * s)], rtol=1e-5
    )
    assert score(images, text)[0, 0] == pytest.approx(1.0)  # scale 100 is the default


def test_argmax_tie_goes_to_the_first_genus_in_label_order():
    text = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])  # genera 0 and 2 are identical
    probs = score(np.array([[1.0, 0.0]]), text)
    assert probs[0, 0] == probs[0, 2]
    table = predictions_table(_meta(1), probs, ["Zeta", "Alpha", "Mid"])
    assert table.loc[0, "predicted_genus"] == "Zeta"  # first in label order, not alphabetical
    assert table.loc[0, "runner_up"] == "Mid"  # the tied one that lost
    assert table.loc[0, "margin"] == 0.0


def _meta(n):
    return pd.DataFrame(
        {
            "image_name": [f"I{i}.JPG" for i in range(n)],
            "coral_id": list(range(n)),
            "bbox": [[1.0, 2.0, 3.0, 4.0]] * n,
            "area": [100 + i for i in range(n)],
        }
    )


def test_predictions_table_columns_and_values():
    probs = np.zeros((3, 20), np.float32)
    probs[0, [1, 4, 0]] = [0.7, 0.2, 0.1]
    probs[1, [19, 3]] = [0.55, 0.45]
    probs[2, :] = 0.05
    table = predictions_table(_meta(3), probs, HARD20)
    assert list(table.columns) == [
        "image_name",
        "coral_id",
        "bbox",
        "area",
        *HARD20,
        "predicted_genus",
        "confidence",
        "runner_up",
        "margin",
    ]
    assert table["predicted_genus"].tolist() == ["Acropora", "Pavona", "Platygyra"]
    assert table["runner_up"].tolist() == ["Favites", "Millepora", "Acropora"]
    np.testing.assert_allclose(table["confidence"], [0.7, 0.55, 0.05])
    np.testing.assert_allclose(table["margin"], [0.5, 0.10, 0.0], atol=1e-6)
    assert table.loc[0, "bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert table["area"].tolist() == [100, 101, 102]


def test_predictions_table_checks_its_inputs():
    with pytest.raises(ValueError, match="shape"):
        predictions_table(_meta(2), np.zeros((3, 20)), HARD20)
    with pytest.raises(ValueError, match="at least two"):
        predictions_table(_meta(1), np.ones((1, 1)), ["A"])
    empty = predictions_table(_meta(0), np.zeros((0, 20)), HARD20)
    assert len(empty) == 0 and "margin" in empty.columns


# ------------------------------------------------------------------------- text features


def test_taxonomy_text_features_are_the_normalized_prompt_embeddings():
    enc = FakeEncoder(dim=6)
    feats = build_text_features(enc, "hard20", "taxonomy")
    prompts = load_genera().prompt_set("taxonomy_v1")
    assert enc.text_calls == [[prompts.prompts[g] for g in HARD20]]  # label order
    assert feats.shape == (20, 6) and feats.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(feats, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(feats, l2_normalize(enc.encode_texts(enc.text_calls[0])), atol=1e-6)


def test_another_taxonomy_prompt_set_changes_the_prompts():
    base = load_genera()
    variant = PromptSet(
        "taxonomy_test", "taxonomy", "a test", dict(base.prompt_set("taxonomy_v1").prompts)
    )
    variant.prompts["Favites"] = "Animalia Cnidaria Anthozoa Scleractinia Merulinidae Favites"
    config = replace(base, prompt_sets={**base.prompt_sets, "taxonomy_test": variant})
    enc = FakeEncoder()
    build_text_features(enc, "hard20", "taxonomy", "taxonomy_test", config)
    assert "Merulinidae Favites" in enc.text_calls[0][HARD20.index("Favites")]
    build_text_features(enc, "hard20", "taxonomy", "taxonomy_v1", config)
    assert "Faviidae Favites" in enc.text_calls[1][HARD20.index("Favites")]


def test_ensemble_is_the_renormalized_mean_of_the_normalized_components():
    enc = FakeEncoder(dim=6)
    feats = build_text_features(enc, "hard20", "ensemble")
    assert len(enc.text_calls) == 3  # taxonomy, descriptive, common
    parts = [l2_normalize(enc.encode_texts(c)) for c in list(enc.text_calls)]
    expected = l2_normalize((parts[0] + parts[1] + parts[2]) / 3.0)
    np.testing.assert_allclose(feats, expected, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(feats, axis=1), 1.0, atol=1e-6)
    assert not np.allclose(feats, parts[0])


def test_siglip_rules_are_kept(caplog):
    cfg = load_genera()
    desc = [cfg.prompt_set("descriptive").prompts[g] for g in HARD20]
    common = [cfg.prompt_set("common").prompts[g] for g in HARD20]
    enc = FakeEncoder(family="siglip")
    ens = build_text_features(enc, "hard20", "ensemble")
    assert enc.text_calls == [desc, common]  # no taxonomy for SigLIP
    parts = [l2_normalize(enc.encode_texts(c)) for c in (desc, common)]
    np.testing.assert_allclose(ens, l2_normalize((parts[0] + parts[1]) / 2.0), atol=1e-6)

    enc = FakeEncoder(family="siglip")
    with caplog.at_level("WARNING", logger="coralid.genera"):
        tax = build_text_features(enc, "hard20", "taxonomy")
    assert enc.text_calls == [desc]  # silently replaced by descriptive
    np.testing.assert_allclose(tax, l2_normalize(enc.encode_texts(desc)), atol=1e-6)
    assert "descriptive" in caplog.text


def test_hybrid_with_hard20_raises():
    with pytest.raises(Exception, match="10 of the 20 genera"):
        build_text_features(FakeEncoder(), "hard20", "hybrid")


# ------------------------------------------------------------------- the batch loop

FRAME_W, FRAME_H = 200, 150
SHAPE = (FRAME_H, FRAME_W)
FILTERS = FilterConfig(min_area=100, containment_thresh=0.8, max_aspect_ratio=3.0)
CROPS = CropConfig(padding_pct=0.0, min_visible=50)
GENERA = HARD20[:4]


def _photo(seed):
    """A photo twice the frame size: noise on the left half (sharp), flat on the right."""
    rng = np.random.RandomState(seed)
    photo = np.full((2 * FRAME_H, 2 * FRAME_W, 3), 120, np.uint8)
    photo[:, :FRAME_W] = rng.randint(0, 256, size=(2 * FRAME_H, FRAME_W, 3), dtype=np.uint8)
    return photo


@pytest.fixture
def dataset(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    layout = {
        "IMG_1.JPG": [(10, 50, 10, 60), (60, 100, 10, 80), (10, 50, 120, 180)],  # last: flat
        "IMG_2.JPG": [(20, 70, 20, 90)],
        "IMG_3.JPG": [(10, 40, 10, 50), (50, 90, 20, 90), (100, 140, 20, 60)],
        "IMG_4.JPG": [(20, 60, 20, 60)],  # no photo on disk
    }
    entries = []
    for seed, (name, boxes) in enumerate(layout.items()):
        annos = [make_annotation(rect_mask(SHAPE, *b), i) for i, b in enumerate(boxes)]
        entries.append((name, FRAME_W, FRAME_H, annos, None))
        if name != "IMG_4.JPG":
            assert cv2.imwrite(str(images / name), cv2.cvtColor(_photo(seed), cv2.COLOR_RGB2BGR))
    json_dir = write_json_dir(tmp_path / "json", entries)
    manifest = image_manifest(json_dir, images, FILTERS, CROPS)
    return json_dir, images, manifest


def _text(seed=0):
    return l2_normalize(np.random.RandomState(seed).randn(len(GENERA), 8))


def _direct(enc, manifest, json_dir, images, text):
    """The kept crops scored one at a time, without run_zero_shot."""
    probs = []
    for row in manifest[manifest["keep"]].itertuples():
        mask_file = load_mask_json(json_dir / row.image_name.replace(".JPG", ".json"))
        img = load_in_mask_frame(images / row.image_name, *mask_file.frame)
        anno = mask_file.by_id()[row.coral_id]
        crop = masked_crop(img, anno["bbox"], anno["segmentation"], 0.0, "black", 50)
        feats = enc.encode_images([enc.preprocess(Image.fromarray(crop))])
        probs.append(score(l2_normalize(feats), text)[0])
    return np.array(probs)


def test_the_synthetic_dataset_exercises_keep_and_reject(dataset):
    _, _, manifest = dataset
    kept = manifest[manifest["keep"]]
    assert kept["image_name"].tolist() == ["IMG_1.JPG"] * 2 + ["IMG_2.JPG"] + ["IMG_3.JPG"] * 3
    assert manifest.loc[manifest["image_name"] == "IMG_4.JPG", "reject_reason"].tolist() == [
        "image_missing"
    ]
    assert "blurry" in manifest["reject_reason"].tolist()


def test_run_zero_shot_scores_every_kept_crop_and_nothing_else(dataset):
    json_dir, images, manifest = dataset
    enc, text = FakeEncoder(dim=8), _text()
    result = run_zero_shot(enc, text, GENERA, manifest, json_dir, images, crops=CROPS)
    kept = manifest[manifest["keep"]]
    table = result.predictions
    assert table["image_name"].tolist() == kept["image_name"].tolist()
    assert table["coral_id"].tolist() == kept["coral_id"].tolist()  # same keys, same order
    assert list(table.columns)[:4] == ["image_name", "coral_id", "bbox", "area"]
    assert result.embeddings is None and result.failures == [] and result.n_images == 3
    np.testing.assert_allclose(
        table[GENERA].to_numpy(),
        _direct(FakeEncoder(dim=8), manifest, json_dir, images, text),
        atol=1e-6,
    )
    # the legacy meta columns come from the annotation
    first = load_mask_json(json_dir / "IMG_1.json").by_id()[0]
    assert table.loc[0, "bbox"] == list(first["bbox"]) and table.loc[0, "area"] == first["area"]
    np.testing.assert_allclose(table["confidence"], table[GENERA].max(axis=1))
    assert (table["predicted_genus"] == table[GENERA].idxmax(axis=1)).all()


def test_batches_have_at_most_batch_size_crops(dataset):
    json_dir, images, manifest = dataset
    enc = FakeEncoder()
    run_zero_shot(enc, _text(), GENERA, manifest, json_dir, images, crops=CROPS, batch_size=4)
    assert enc.image_batch_sizes == [4, 2]  # six crops in one group
    enc = FakeEncoder()
    run_zero_shot(
        enc, _text(), GENERA, manifest, json_dir, images, crops=CROPS, batch_size=4, group_size=1
    )
    assert enc.image_batch_sizes == [2, 1, 3]  # one encode per image group


def test_embeddings_are_normalized_and_row_aligned(dataset, tmp_path):
    json_dir, images, manifest = dataset
    enc, text = FakeEncoder(), _text()
    result = run_zero_shot(
        enc, text, GENERA, manifest, json_dir, images, crops=CROPS, keep_embeddings=True
    )
    X = result.embeddings
    assert X.shape == (6, 8)
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(score(X, text), result.predictions[GENERA].to_numpy(), atol=1e-6)
    uids = [
        f"{n}#{c}"
        for n, c in zip(
            result.predictions["image_name"], result.predictions["coral_id"], strict=True
        )
    ]
    path = save_embeddings(tmp_path / "e.npz", uids, X)
    back_uids, back_X = load_embeddings(path)
    assert back_uids == uids and back_X.dtype == np.float32
    np.testing.assert_array_equal(back_X, X)
    with pytest.raises(FileExistsError):
        save_embeddings(path, uids, X)
    save_embeddings(path, uids, X, overwrite=True)
    with pytest.raises(ValueError, match="uids"):
        save_embeddings(tmp_path / "bad.npz", uids[:-1], X)


def test_a_tie_in_the_text_features_goes_to_the_first_genus(dataset):
    json_dir, images, manifest = dataset
    text = _text()
    text[2] = text[0]  # genera 0 and 2 can no longer be told apart
    result = run_zero_shot(FakeEncoder(), text, GENERA, manifest, json_dir, images, crops=CROPS)
    tied = result.predictions[GENERA[0]] == result.predictions[GENERA[2]]
    assert tied.all()
    assert GENERA[2] not in result.predictions["predicted_genus"].tolist()


def test_resume_skips_done_images_and_retries_failed_ones(dataset, tmp_path):
    json_dir, images, manifest = dataset
    text = _text()
    clean = run_zero_shot(FakeEncoder(), text, GENERA, manifest, json_dir, images, crops=CROPS)

    state = {"calls": 0, "fail_second": True}

    def flaky(items):
        state["calls"] += 1
        return state["fail_second"] and state["calls"] == 2

    run = ResumableRun(tmp_path / "run", "zs", flush_every=1)
    first = run_zero_shot(
        FakeEncoder(fail_when=flaky), text, GENERA, manifest, json_dir, images,
        crops=CROPS, run=run, group_size=1,
    )  # fmt: skip
    assert first.n_images == 2  # IMG_2 failed, the other two were recorded
    assert run.done == {"IMG_1.JPG", "IMG_3.JPG"}
    assert [f["image"] for f in first.failures] == ["IMG_2.JPG"]
    assert "out of memory" in first.failures[0]["error"]
    assert sorted(first.predictions["image_name"].unique()) == ["IMG_1.JPG", "IMG_3.JPG"]

    # resume with a healthy encoder: only the failed image is processed
    state["fail_second"] = False
    enc = FakeEncoder()
    second = run_zero_shot(
        enc, text, GENERA, manifest, json_dir, images,
        crops=CROPS, run=ResumableRun(tmp_path / "run", "zs"), group_size=1,
    )  # fmt: skip
    assert second.n_images == 1 and enc.image_batch_sizes == [1]
    assert second.failures == []
    # all three images are in the result now, and the numbers equal a run that never failed
    got = second.predictions.sort_values(["image_name", "coral_id"], ignore_index=True)
    want = clean.predictions.sort_values(["image_name", "coral_id"], ignore_index=True)
    assert got["coral_id"].tolist() == want["coral_id"].tolist()
    np.testing.assert_allclose(got[GENERA].to_numpy(), want[GENERA].to_numpy(), atol=1e-7)
    assert got["predicted_genus"].tolist() == want["predicted_genus"].tolist()

    # a third call has nothing left to do
    enc3 = FakeEncoder()
    third = run_zero_shot(
        enc3, text, GENERA, manifest, json_dir, images,
        crops=CROPS, run=ResumableRun(tmp_path / "run", "zs"),
    )  # fmt: skip
    assert enc3.image_batch_sizes == [] and third.n_images == 0 and len(third.predictions) == 6


def test_resumable_embeddings_are_merged_with_the_predictions(dataset, tmp_path):
    json_dir, images, manifest = dataset
    run = ResumableRun(tmp_path / "run", "zs", flush_every=2)
    result = run_zero_shot(
        FakeEncoder(), _text(), GENERA, manifest, json_dir, images,
        crops=CROPS, run=run, keep_embeddings=True, group_size=1,
    )  # fmt: skip
    assert result.embeddings.shape == (6, 8) and len(result.predictions) == 6
    plain = run_zero_shot(
        FakeEncoder(),
        _text(),
        GENERA,
        manifest,
        json_dir,
        images,
        crops=CROPS,
        keep_embeddings=True,
    )
    np.testing.assert_allclose(result.embeddings, plain.embeddings, atol=1e-7)


def test_the_run_stops_after_too_many_consecutive_failures(dataset, tmp_path):
    json_dir, images, manifest = dataset
    run = ResumableRun(tmp_path / "run", "zs")
    enc = FakeEncoder(fail_when=lambda items: True)
    with pytest.raises(RuntimeError, match="2 consecutive failures"):
        run_zero_shot(
            enc, _text(), GENERA, manifest, json_dir, images,
            crops=CROPS, run=run, group_size=1, max_consecutive_failures=2,
        )  # fmt: skip
    assert run.done == set()  # nothing marked done
    assert {f["image"] for f in run.failures()} == {"IMG_1.JPG", "IMG_2.JPG"}


def test_a_manifest_that_disagrees_with_the_crop_config_is_reported(dataset, tmp_path):
    json_dir, images, manifest = dataset
    strict = CropConfig(padding_pct=0.0, min_visible=10**9)
    run = ResumableRun(tmp_path / "run", "zs")
    with pytest.raises(RuntimeError, match="consecutive failures"):
        run_zero_shot(
            FakeEncoder(), _text(), GENERA, manifest, json_dir, images,
            crops=strict, run=run, max_consecutive_failures=3,
        )  # fmt: skip
    assert "kept in the manifest" in run.failures()[0]["error"]
    assert issubclass(CropError, RuntimeError)


def test_input_checks(dataset):
    json_dir, images, manifest = dataset
    enc = FakeEncoder()
    with pytest.raises(ValueError, match="'keep' column"):
        run_zero_shot(enc, _text(), GENERA, manifest.drop(columns="keep"), json_dir, images)
    with pytest.raises(NotImplementedError, match="masked crops only"):
        run_zero_shot(
            enc, _text(), GENERA, manifest, json_dir, images, crops=CropConfig(mode="bbox")
        )
    with pytest.raises(ValueError, match="text features"):
        run_zero_shot(enc, _text()[:3], GENERA, manifest, json_dir, images, crops=CROPS)


def test_nothing_to_classify_gives_an_empty_table_with_the_columns(dataset):
    json_dir, images, manifest = dataset
    none_kept = manifest.assign(keep=False)
    result = run_zero_shot(
        FakeEncoder(),
        _text(),
        GENERA,
        none_kept,
        json_dir,
        images,
        crops=CROPS,
        keep_embeddings=True,
    )
    assert len(result.predictions) == 0 and result.embeddings is None
    assert list(result.predictions.columns)[-4:] == [
        "predicted_genus",
        "confidence",
        "runner_up",
        "margin",
    ]
