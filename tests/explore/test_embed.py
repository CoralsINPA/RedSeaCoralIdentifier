import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from coralid.crops import masked_square_crop
from coralid.explore.embed import (
    EmbedError,
    EmbedIncomplete,
    embed_paths,
    load_features,
    run_embeddings,
)
from coralid.images import load_in_mask_frame
from coralid.masks import load_mask_json
from tests.fake_encoder import FakeEncoder
from tests.run_data import make_run

MODEL = "dinov2-s14"


@pytest.fixture
def cfg(tmp_path):
    return make_run(tmp_path / "data")


def _uids(df):
    return [f"{n}#{c}" for n, c in zip(df["image_name"], df["coral_id"], strict=True)]


def test_the_paths_follow_the_features_convention(cfg):
    paths = embed_paths(cfg)
    assert paths.features == cfg.run_dir / "features" / "dinov2-s14__masked_square.npz"
    assert paths.parts_dir == cfg.run_dir / "features" / "parts"
    assert embed_paths(cfg, "dinovdeau-l").name == "dinovdeau-l__masked_square"


def test_features_are_the_square_crops_encoded_and_saved_with_meta(cfg):
    encoder = FakeEncoder(dim=8)
    result = run_embeddings(cfg, MODEL, encoder=encoder)
    assert result.status == "computed" and result.n_images_processed == 3
    manifest = pd.read_parquet(cfg.manifest_path)
    kept = manifest[manifest["keep"]]
    assert result.uids == _uids(kept)  # manifest order
    assert result.features.shape == (6, 8) and result.features.dtype == np.float32
    # the features are those of the masked square crop of each kept mask, not normalized
    fresh = FakeEncoder(dim=8)
    expected = []
    for name, rows in kept.groupby("image_name", sort=False):
        mask_file = load_mask_json(cfg.json_dir / name.replace(".JPG", ".json"))
        image = load_in_mask_frame(cfg.images_dir / name, *mask_file.frame)
        for coral_id in rows["coral_id"]:
            anno = mask_file.by_id()[int(coral_id)]
            crop = masked_square_crop(image, anno["bbox"], anno["segmentation"])
            expected.append(fresh.preprocess(Image.fromarray(crop)))
    np.testing.assert_allclose(result.features, fresh.encode_images(expected), atol=1e-6)
    assert not np.allclose(np.linalg.norm(result.features, axis=1), 1.0)

    with np.load(embed_paths(cfg).features) as data:
        assert data["uids"].tolist() == result.uids
        np.testing.assert_array_equal(data["X"], result.features)
    meta = json.loads(embed_paths(cfg).features.with_suffix(".meta.json").read_text())
    assert meta["kind"] == "image_features" and meta["n_rows"] == 6 and meta["dim"] == 8
    assert meta["model"] == MODEL and meta["crop_mode"] == "masked_square"
    assert meta["normalized"] == "none" and meta["n_manifest_keep"] == 6
    assert meta["config"]["embed"]["model"] == "dinov2-s14" and "git_commit" in meta


def test_the_batch_size_comes_from_the_config(cfg):
    cfg.embed.batch_size = 4
    encoder = FakeEncoder()
    run_embeddings(cfg, MODEL, encoder=encoder)
    assert encoder.image_batch_sizes == [4, 2]


def test_existing_features_are_loaded_never_recomputed_or_overwritten(cfg):
    run_embeddings(cfg, MODEL, encoder=FakeEncoder())
    path = embed_paths(cfg).features
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    encoder = FakeEncoder()
    again = run_embeddings(cfg, MODEL, encoder=encoder)
    assert again.status == "already_done" and encoder.image_batch_sizes == []
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert load_features(cfg, MODEL).uids == again.uids
    path.unlink()
    with pytest.raises(FileNotFoundError):
        load_features(cfg, MODEL)


def test_a_failed_image_is_retried_and_the_result_equals_an_uninterrupted_run(cfg, tmp_path):
    reference = run_embeddings(make_run(tmp_path / "ref"), MODEL, encoder=FakeEncoder())
    calls = {"n": 0}

    def fail_second_group(items):
        calls["n"] += 1
        return calls["n"] == 2

    flaky = FakeEncoder(fail_when=fail_second_group)
    with pytest.raises(EmbedIncomplete) as err:
        run_embeddings(cfg, MODEL, encoder=flaky, group_size=1)
    assert [f["image"] for f in err.value.failures] == ["IMG_2.JPG"]
    assert not embed_paths(cfg).features.exists()  # written only when every image is done
    log = embed_paths(cfg).parts_dir / f"{embed_paths(cfg).name}.failures.jsonl"
    assert "CUDA out of memory" in log.read_text()

    retry = FakeEncoder()
    result = run_embeddings(cfg, MODEL, encoder=retry, group_size=1)
    assert result.n_images_processed == 1 and retry.image_batch_sizes == [1]
    assert result.uids == reference.uids
    np.testing.assert_allclose(result.features, reference.features, atol=1e-6)


def test_a_crash_resumes_with_only_the_missing_images(cfg):
    class Crash(FakeEncoder):
        def encode_images(self, batch):
            if len(self.image_batch_sizes) == 1:
                raise KeyboardInterrupt
            return super().encode_images(batch)

    with pytest.raises(KeyboardInterrupt):
        run_embeddings(cfg, MODEL, encoder=Crash(), group_size=1, flush_every=1)
    resumed = FakeEncoder()
    result = run_embeddings(cfg, MODEL, encoder=resumed, group_size=1)
    assert result.n_images_processed == 2 and len(result.uids) == 6


def test_parts_made_from_other_inputs_are_refused(cfg):
    flaky = FakeEncoder(fail_when=lambda items: True)
    with pytest.raises(RuntimeError, match="consecutive failures"):
        run_embeddings(cfg, MODEL, encoder=flaky, group_size=1, max_consecutive_failures=2)
    record = embed_paths(cfg).parts_dir / f"{embed_paths(cfg).name}.input.json"
    assert record.exists()
    record.write_text(json.dumps({"fingerprint": "another manifest or model"}))
    with pytest.raises(EmbedError, match="other inputs"):
        run_embeddings(cfg, MODEL, encoder=FakeEncoder())


def test_clip_family_models_are_not_embedded_here(cfg):
    with pytest.raises(EmbedError, match="zero-shot"):
        run_embeddings(cfg, "finetuned-bioclip")


def test_the_manifest_is_required(tmp_path):
    cfg = make_run(tmp_path / "data", manifest=False)
    with pytest.raises(EmbedError, match="manifest"):
        run_embeddings(cfg, MODEL, encoder=FakeEncoder())
