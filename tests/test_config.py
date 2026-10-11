import json
from pathlib import Path

import pytest
import yaml

import coralid
from coralid.config import (
    DATA_ROOT_ENV,
    Config,
    ConfigError,
    CropConfig,
    FilterConfig,
    SamConfig,
    ZeroShotConfig,
    load_config,
    meta_path,
    run_name,
    save_config,
    write_meta,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_run_name_for_current_values():
    assert run_name() == "iou60_sta55_pts32"
    assert run_name(SamConfig()) == "iou60_sta55_pts32"
    assert Config().run_name == "iou60_sta55_pts32"


def test_run_name_rounds_instead_of_truncating():
    # int(0.57 * 100) == 56 because 0.57 * 100 == 56.99999999999999
    assert int(0.57 * 100) == 56
    assert run_name(SamConfig(pred_iou_thresh=0.57)) == "iou57_sta55_pts32"
    assert run_name(SamConfig(stability_score_thresh=0.57, points_per_side=16)) == (
        "iou60_sta57_pts16"
    )


def test_defaults_match_the_legacy_notebooks():
    f = FilterConfig()
    assert (f.min_area, f.containment_thresh, f.max_aspect_ratio) == (25000, 0.80, 3.0)
    assert f.sharpness_threshold == 80
    c = CropConfig()
    assert (c.mode, c.background, c.padding_pct, c.min_visible) == ("masked", "black", 0.05, 500)
    assert c.bbox_padding_pct == 0.15
    s = SamConfig()
    assert (s.points_per_side, s.pred_iou_thresh, s.stability_score_thresh) == (32, 0.60, 0.55)
    assert (s.crop_n_layers, s.crop_n_points_downscale_factor) == (1, 2)
    assert (s.min_mask_region_area, s.max_side, s.model_type) == (100, 2560, "vit_b")
    z = ZeroShotConfig()
    assert (z.model, z.label_set, z.prompt_format, z.batch_size) == (
        "finetuned-bioclip",
        "hard20",
        "taxonomy",
        64,
    )
    assert z.logit_scale == 100.0


def test_shipped_yaml_equals_the_defaults():
    path = REPO_ROOT / "configs" / "iou60_sta55_pts32.yaml"
    cfg = load_config(path)
    assert cfg == Config()
    assert cfg.run_name == path.stem


def test_yaml_round_trip(tmp_path):
    cfg = Config()
    cfg.filters.min_area = 12345
    cfg.sam.pred_iou_thresh = 0.57
    cfg.crops.background = "gray"
    out = save_config(cfg, tmp_path / "sub" / "cfg.yaml")
    assert out.exists()
    assert load_config(out) == cfg


def test_unknown_keys_and_sections_are_rejected(tmp_path):
    bad_key = tmp_path / "a.yaml"
    bad_key.write_text(yaml.safe_dump({"filters": {"min_aera": 1}}))
    with pytest.raises(ConfigError, match="min_aera"):
        load_config(bad_key)
    bad_section = tmp_path / "b.yaml"
    bad_section.write_text(yaml.safe_dump({"filter": {}}))
    with pytest.raises(ConfigError, match="filter"):
        load_config(bad_section)


def test_invalid_crop_settings_are_rejected():
    with pytest.raises(ConfigError):
        CropConfig(background="grey")
    with pytest.raises(ConfigError):
        CropConfig(mode="circle")


def test_empty_yaml_gives_defaults(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert load_config(p) == Config()


def test_data_root_comes_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    cfg = Config()
    assert cfg.data_root == tmp_path
    assert cfg.images_dir == tmp_path / "data_all"
    assert cfg.run_dir == tmp_path / "results_all" / "iou60_sta55_pts32"
    assert cfg.json_dir == cfg.run_dir / "json"
    assert cfg.manifest_path == cfg.run_dir / "manifest" / "manifest.parquet"
    assert cfg.checkpoint_path == tmp_path / "CoralSCOP" / "checkpoints" / "vit_b_coralscop.pth"
    assert cfg.ground_truth_path == tmp_path / "ground_truth_genus_counts.csv"


def test_explicit_data_root_wins_over_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path / "env"))
    cfg = Config()
    cfg.paths.data_root = str(tmp_path / "explicit")
    assert cfg.data_root == tmp_path / "explicit"


def test_missing_data_root_is_a_clear_error(monkeypatch):
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    with pytest.raises(ConfigError, match=DATA_ROOT_ENV):
        _ = Config().run_dir


def test_absolute_paths_are_kept(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    cfg = Config()
    cfg.paths.images_dir = "/somewhere/else"
    assert cfg.images_dir == Path("/somewhere/else")


def test_meta_path():
    assert meta_path("a/manifest.parquet") == Path("a/manifest.meta.json")
    assert meta_path("a/x.npz") == Path("a/x.meta.json")
    assert meta_path("a/x.meta.json") == Path("a/x.meta.json")


def test_write_meta_records_the_provenance(tmp_path):
    cfg = Config()
    out = write_meta(
        tmp_path / "out" / "preds.parquet", config=cfg, n_rows=42, label_set="hard20", note="t"
    )
    assert out == tmp_path / "out" / "preds.meta.json"
    meta = json.loads(out.read_text())
    assert meta["package_version"] == coralid.__version__
    assert meta["n_rows"] == 42
    assert meta["label_set"] == "hard20"
    assert meta["config"] == cfg.to_dict()
    assert meta["created_utc"].endswith("Z") and len(meta["created_utc"]) == 20
    # The repo is a git checkout in development; elsewhere the commit is simply None.
    assert meta["git_commit"] is None or len(meta["git_commit"]) == 40
    assert not list(out.parent.glob(".*tmp*"))  # no temp files left behind


def test_write_meta_handles_numpy_values(tmp_path):
    import numpy as np

    out = write_meta(tmp_path / "x.parquet", n_rows=np.int64(3), shape=np.array([2, 3]))
    meta = json.loads(out.read_text())
    assert meta["n_rows"] == 3
    assert meta["shape"] == [2, 3]
    assert meta["config"] is None
