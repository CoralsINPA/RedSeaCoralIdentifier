"""A blind review is drawn only from the 20-genus baseline; scoring checks it is still that one."""

import json
from dataclasses import replace

import pytest

from coralid.baseline import baseline_paths, run_baseline
from coralid.config import meta_path
from coralid.jobs import export_blind_review
from coralid.review_source import (
    ReviewSourceError,
    baseline_source,
    check_same_source,
)
from tests.fake_encoder import FakeEncoder
from tests.run_data import make_run


@pytest.fixture
def cfg(tmp_path):
    config = make_run(tmp_path / "data")
    run_baseline(config, encoder=FakeEncoder(dim=8))
    return config


def _edit_meta(cfg, **changes):
    path = meta_path(baseline_paths(cfg).predictions)
    meta = json.loads(path.read_text())
    meta.update(changes)
    path.write_text(json.dumps(meta))


def test_the_source_is_read_from_the_baseline_through_baseline_paths(cfg):
    source = baseline_source(cfg)
    paths = baseline_paths(cfg)
    assert source.path == str(paths.predictions) and source.file == paths.predictions.name
    assert (source.model, source.prompt_set, source.label_set) == (
        "finetuned-bioclip",
        "taxonomy_v1",
        "hard20",
    )
    meta = json.loads(meta_path(paths.predictions).read_text())
    assert source.git_commit == meta["git_commit"] and source.created_utc == meta["created_utc"]
    assert source.n_rows == meta["n_rows"] == 6
    assert any("hard20" in line for line in source.lines())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"label_set": "soft_corals"}, "label_set is 'soft_corals', it must be 'hard20'"),
        ({"label_set": None}, "label_set is None"),
        ({"model": "bioclip2"}, "model is 'bioclip2', it must be 'finetuned-bioclip'"),
        ({"prompt_set": "ensemble_taxonomy_v1"}, "prompt_set is .ensemble_taxonomy_v1."),
    ],
)
def test_a_baseline_with_another_label_set_model_or_prompt_set_is_refused(cfg, change, message):
    _edit_meta(cfg, **change)
    with pytest.raises(ReviewSourceError, match=message):
        baseline_source(cfg)
    with pytest.raises(ReviewSourceError):  # and no review is written
        export_blind_review(cfg, "Acropora", n_pred=2, n_other=2, n_assaf=2, pdf=False)
    assert not (cfg.reviews_dir / "exports").exists() and not (cfg.reviews_dir / "keys").exists()


def test_a_config_that_asks_for_another_label_set_does_not_find_the_baseline(cfg):
    cfg.zero_shot.label_set = "hard12"
    with pytest.raises(ReviewSourceError, match="not found"):
        baseline_source(cfg)


def test_predictions_without_a_meta_file_are_refused(cfg):
    meta_path(baseline_paths(cfg).predictions).unlink()
    with pytest.raises(ReviewSourceError, match="nobody can tell which model"):
        baseline_source(cfg)


def test_no_baseline_at_all_says_to_run_it(tmp_path):
    with pytest.raises(ReviewSourceError, match="run the 20-genus baseline first"):
        baseline_source(make_run(tmp_path / "empty"))


def test_the_same_source_passes_and_any_change_of_identity_is_named(cfg):
    source = baseline_source(cfg)
    check_same_source(source.to_dict(), source)
    check_same_source(replace(source, path="/elsewhere").to_dict(), source)  # another mount
    for field, value in (
        ("created_utc", "2030-01-01T00:00:00Z"),
        ("git_commit", "ffff"),
        ("n_rows", 7),
        ("file", "other.parquet"),
    ):
        with pytest.raises(ReviewSourceError, match=field):
            check_same_source(replace(source, **{field: value}).to_dict(), source)
    with pytest.raises(ReviewSourceError, match="records no source"):
        check_same_source(None, source)
