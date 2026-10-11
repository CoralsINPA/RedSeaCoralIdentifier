import json

import cv2
import numpy as np
import pandas as pd
import pytest

from coralid import baseline as baseline_module
from coralid.baseline import (
    BaselineError,
    BaselineIncomplete,
    baseline_paths,
    load_baseline,
    prompts_tag,
    run_baseline,
)
from coralid.cli import main
from coralid.config import Config, CropConfig, FilterConfig, Paths, save_config
from coralid.genera import load_genera
from coralid.manifest import image_manifest, write_manifest
from tests.fake_encoder import FakeEncoder
from tests.helpers import make_annotation, rect_mask, write_json_dir

FRAME_W, FRAME_H = 200, 150
SHAPE = (FRAME_H, FRAME_W)
HARD20 = load_genera().genera("hard20")


def _photo(seed):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(FRAME_H, FRAME_W, 3), dtype=np.uint8)


def _config(tmp_path):
    return Config(
        paths=Paths(data_root=str(tmp_path)),
        filters=FilterConfig(min_area=100),
        crops=CropConfig(padding_pct=0.0, min_visible=50),
    )


def _make_dataset(root):
    """A data root in the real layout: photos, mask JSONs and the manifest of three images."""
    cfg = _config(root)
    cfg.images_dir.mkdir(parents=True)
    layout = {
        "IMG_1.JPG": [(10, 50, 10, 60), (60, 100, 10, 80)],
        "IMG_2.JPG": [(20, 70, 20, 90)],
        "IMG_3.JPG": [(10, 40, 10, 50), (50, 90, 20, 90), (100, 140, 20, 60)],
    }
    entries = []
    for seed, (name, boxes) in enumerate(layout.items()):
        annos = [make_annotation(rect_mask(SHAPE, *b), i) for i, b in enumerate(boxes)]
        entries.append((name, FRAME_W, FRAME_H, annos, None))
        assert cv2.imwrite(
            str(cfg.images_dir / name), cv2.cvtColor(_photo(seed), cv2.COLOR_RGB2BGR)
        )
    write_json_dir(cfg.json_dir, entries)
    manifest = image_manifest(cfg.json_dir, cfg.images_dir, cfg.filters, cfg.crops)
    assert manifest["keep"].sum() == 6
    write_manifest(manifest, cfg.manifest_path, config=cfg)
    return cfg


@pytest.fixture
def cfg(tmp_path):
    return _make_dataset(tmp_path / "data")


def _uids(df):
    return [f"{n}#{c}" for n, c in zip(df["image_name"], df["coral_id"], strict=True)]


def test_the_paths_follow_the_documented_names(cfg):
    paths = baseline_paths(cfg)
    run = cfg.run_dir
    assert (
        paths.predictions
        == run / "predictions" / "finetuned-bioclip__taxonomy_v1__hard20__all.parquet"
    )
    assert paths.features == run / "features" / "finetuned-bioclip__masked.npz"
    assert paths.parts_dir == run / "predictions" / "parts"
    assert baseline_paths(cfg, "sample50").name.endswith("__hard20__sample50")
    cfg.zero_shot.prompt_format = "ensemble"
    assert prompts_tag(cfg) == "ensemble_taxonomy_v1"
    cfg.zero_shot.prompt_format = "descriptive"
    assert prompts_tag(cfg) == "descriptive"


def test_run_baseline_writes_predictions_and_embeddings_with_meta(cfg):
    result = run_baseline(cfg, encoder=FakeEncoder(dim=8))
    assert result.status == "computed" and result.n_images_processed == 3
    paths = result.paths
    table = pd.read_parquet(paths.predictions)
    assert len(table) == 6 and list(table.columns)[:4] == ["image_name", "coral_id", "bbox", "area"]
    assert list(table.columns)[4:24] == HARD20
    assert list(table.columns)[24:] == ["predicted_genus", "confidence", "runner_up", "margin"]
    manifest = pd.read_parquet(cfg.manifest_path)
    assert _uids(table) == _uids(manifest[manifest["keep"]])
    with np.load(paths.features) as data:
        assert list(data) == ["uids", "X"] or sorted(data) == ["X", "uids"]
        assert data["uids"].tolist() == _uids(table) and data["X"].shape == (6, 8)
        np.testing.assert_allclose(np.linalg.norm(data["X"], axis=1), 1.0, atol=1e-6)
    meta = json.loads(paths.predictions.with_suffix(".meta.json").read_text())
    assert meta["kind"] == "predictions" and meta["n_rows"] == 6
    assert (meta["model"], meta["label_set"], meta["prompt_set"]) == (
        "finetuned-bioclip",
        "hard20",
        "taxonomy_v1",
    )
    assert meta["crop_mode"] == "masked" and meta["n_manifest_keep"] == 6
    assert meta["config"]["zero_shot"]["logit_scale"] == 100.0 and "git_commit" in meta
    fmeta = json.loads(paths.features.with_suffix(".meta.json").read_text())
    assert fmeta["kind"] == "image_features" and fmeta["dim"] == 8 and fmeta["normalized"] == "l2"
    # the embeddings score to the saved probabilities
    from coralid.zero_shot import build_text_features, score

    text = build_text_features(FakeEncoder(dim=8), "hard20", "taxonomy", "taxonomy_v1")
    with np.load(paths.features) as data:
        np.testing.assert_allclose(score(data["X"], text), table[HARD20].to_numpy(), atol=1e-5)


def test_a_finished_baseline_is_loaded_and_never_recomputed_or_overwritten(cfg):
    first = run_baseline(cfg, encoder=FakeEncoder())
    before = first.paths.predictions.read_bytes()
    enc = FakeEncoder()
    second = run_baseline(cfg, encoder=enc)
    assert second.status == "already_done" and enc.image_batch_sizes == [] and not enc.text_calls
    assert first.paths.predictions.read_bytes() == before
    pd.testing.assert_frame_equal(second.predictions, first.predictions)
    loaded = load_baseline(cfg)
    assert loaded.uids == first.uids and loaded.features.shape == (6, 8)


def test_a_lone_result_file_is_an_error(cfg):
    result = run_baseline(cfg, encoder=FakeEncoder())
    result.paths.features.unlink()
    with pytest.raises(BaselineError, match="Only .*parquet exists"):
        run_baseline(cfg, encoder=FakeEncoder())
    with pytest.raises(FileNotFoundError):
        load_baseline(cfg)


def test_a_failed_image_is_retried_and_the_result_equals_an_uninterrupted_run(cfg, tmp_path):
    expected = run_baseline(_make_dataset(tmp_path / "reference"), encoder=FakeEncoder(dim=8))

    calls = []

    def fail_second_image(items):
        calls.append(len(items))
        return len(calls) == 2  # image 2 (one crop per group of one image) fails once

    flaky = FakeEncoder(dim=8, fail_when=fail_second_image)
    with pytest.raises(BaselineIncomplete, match="1 image.*failed"):
        run_baseline(cfg, encoder=flaky, flush_every=1, group_size=1)
    paths = baseline_paths(cfg)
    assert not paths.predictions.exists() and not paths.features.exists()  # nothing half-written
    log = (paths.parts_dir / f"{paths.name}.failures.jsonl").read_text()
    assert "IMG_2.JPG" in log and "fake CUDA out of memory" in log

    retry = FakeEncoder(dim=8)
    resumed = run_baseline(cfg, encoder=retry, flush_every=1, group_size=1)
    assert retry.image_batch_sizes == [1]  # only image 2 is redone
    assert resumed.status == "computed" and resumed.n_images_processed == 1
    pd.testing.assert_frame_equal(
        resumed.predictions.reset_index(drop=True), expected.predictions.reset_index(drop=True)
    )
    np.testing.assert_allclose(resumed.features, expected.features, atol=1e-6)
    assert resumed.uids == expected.uids


def test_resuming_after_a_crash_skips_the_finished_images(cfg, tmp_path):
    expected = run_baseline(_make_dataset(tmp_path / "reference"), encoder=FakeEncoder(dim=8))

    class CrashingEncoder(FakeEncoder):
        def encode_images(self, batch):
            if len(self.image_batch_sizes) == 1:  # the second image group
                raise KeyboardInterrupt("simulated crash")
            return super().encode_images(batch)

    with pytest.raises(KeyboardInterrupt):
        run_baseline(cfg, encoder=CrashingEncoder(dim=8), group_size=1)
    assert not baseline_paths(cfg).predictions.exists()
    again = FakeEncoder(dim=8)
    result = run_baseline(cfg, encoder=again, group_size=1)
    assert again.image_batch_sizes == [1, 3]  # images 2 and 3; image 1 was saved before the crash
    assert result.n_images_processed == 2
    pd.testing.assert_frame_equal(
        result.predictions.reset_index(drop=True), expected.predictions.reset_index(drop=True)
    )


def test_parts_from_other_inputs_are_refused(cfg):
    with pytest.raises(BaselineIncomplete):
        run_baseline(cfg, encoder=FakeEncoder(fail_when=lambda items: True))
    cfg.zero_shot.logit_scale = 50.0  # the same file names, other settings
    with pytest.raises(BaselineError, match="other inputs"):
        run_baseline(cfg, encoder=FakeEncoder())


def test_a_missing_manifest_says_where_it_comes_from(tmp_path):
    cfg = _config(tmp_path)
    with pytest.raises(BaselineError, match="Check C"):
        run_baseline(cfg, encoder=FakeEncoder())


# ----------------------------------------------------------------------------- the CLI


@pytest.fixture
def cli_args(cfg, tmp_path, monkeypatch):
    path = tmp_path / "run.yaml"
    config = _config(tmp_path / "data")
    config.paths.data_root = None  # the command line gives the root
    save_config(config, path)
    monkeypatch.setattr(baseline_module, "load_encoder", lambda name: FakeEncoder(dim=8))
    return ["zero-shot", "--config", str(path), "--data-root", str(cfg.data_root), "--quiet"]


def test_cli_zero_shot_runs_the_baseline_and_is_idempotent(cli_args, cfg, capsys):
    assert main(cli_args) == 0
    out = capsys.readouterr().out
    assert "Wrote: 6 crops" in out and "finetuned-bioclip__taxonomy_v1__hard20__all.parquet" in out
    assert baseline_paths(cfg).predictions.exists()
    assert main(cli_args) == 0
    assert "Loaded the saved baseline: 6 crops" in capsys.readouterr().out


def test_cli_reports_an_incomplete_run_with_exit_code_2(cli_args, cfg, monkeypatch, capsys):
    monkeypatch.setattr(
        baseline_module, "load_encoder", lambda name: FakeEncoder(fail_when=lambda items: True)
    )
    assert main(cli_args) == 2
    assert "INCOMPLETE" in capsys.readouterr().err
    assert not baseline_paths(cfg).predictions.exists()


def test_cli_reports_a_missing_manifest_and_needs_a_command(tmp_path, monkeypatch, capsys):
    cfg_path = tmp_path / "run.yaml"
    save_config(_config(tmp_path), cfg_path)
    assert main(["zero-shot", "--config", str(cfg_path), "--quiet"]) == 2
    assert "STOPPED" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main([])
