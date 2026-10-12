import json

import pandas as pd
import pytest

from coralid.baseline import run_baseline
from coralid.embed import embed_paths, run_embeddings
from coralid.jobs import (
    JobError,
    ManifestIncomplete,
    build_manifest,
    evaluate_baseline,
    export_blind_review,
    train_probe,
)
from coralid.probe import ProbeError
from tests.fake_encoder import FakeEncoder
from tests.run_data import labels_for, make_run, write_expert_inputs


@pytest.fixture
def run(tmp_path):
    return make_run(tmp_path / "data", manifest=False)


@pytest.fixture
def baseline_run(tmp_path):
    """A run with its manifest and the baseline (6 crops on 3 photos) already saved."""
    cfg = make_run(tmp_path / "data")
    run_baseline(cfg, encoder=FakeEncoder(dim=8))
    return cfg


# ------------------------------------------------------------------------- build_manifest


def test_the_manifest_is_built_saved_with_meta_and_then_loaded(run):
    result = build_manifest(run)
    assert result.status == "computed" and result.path == run.manifest_path
    assert len(result.manifest) == 6 and int(result.manifest["keep"].sum()) == 6
    meta = json.loads(run.manifest_path.with_suffix(".meta.json").read_text())
    assert meta["kind"] == "image" and meta["n_keep"] == 6 and meta["run_name"] == run.run_name
    before = run.manifest_path.read_bytes()
    again = build_manifest(run)
    assert again.status == "already_done" and run.manifest_path.read_bytes() == before


def test_a_manifest_made_with_other_settings_is_not_silently_reused(run):
    build_manifest(run)
    run.filters.min_area = 5000
    with pytest.raises(JobError, match=r"filters\.min_area: saved 100, now 5000"):
        build_manifest(run)
    run.filters.min_area = 100
    run.crops.min_visible = 400
    with pytest.raises(JobError, match=r"crops\.min_visible"):
        build_manifest(run)


def test_a_failing_image_stops_the_manifest_and_nothing_is_written(run):
    (run.json_dir / "IMG_9.json").write_text("{}")  # a malformed mask JSON
    with pytest.raises(ManifestIncomplete) as err:
        build_manifest(run)
    assert [f["image"] for f in err.value.failures] == ["IMG_9.json"]
    assert not run.manifest_path.exists()
    assert (run.manifest_path.parent / "parts" / "manifest.done.json").exists()  # the rest is kept


def test_the_geometric_manifest_needs_no_photos_and_has_its_own_file(run):
    for photo in run.images_dir.iterdir():
        photo.unlink()
    result = build_manifest(run, geometric=True)
    assert result.path.name == "manifest_geometric.parquet" and "keep" not in result.manifest
    assert not run.manifest_path.exists()
    assert json.loads(result.path.with_suffix(".meta.json").read_text())["kind"] == "geometric"


# ------------------------------------------------------------------------- evaluate


def test_evaluate_writes_a_new_report_folder_for_the_saved_baseline(baseline_run):
    cfg = baseline_run
    predictions = pd.read_parquet(
        cfg.predictions_dir / next(p.name for p in cfg.predictions_dir.glob("*.parquet"))
    )
    write_expert_inputs(cfg, predictions)
    report, folder = evaluate_baseline(cfg, date="2026-10-12")
    assert (
        folder
        == cfg.reports_dir / "baseline_finetuned-bioclip__taxonomy_v1__hard20__all_2026-10-12"
    )
    assert (folder / "summary.json").exists() and (folder / "expert_paired.csv").exists()
    everything = report.expert_new["all"]
    assert everything.accuracy_in_set["correct"] == 3 and everything.accuracy_in_set["total"] == 4
    assert (
        everything.status_counts["out_of_set"] == 1
        and everything.status_counts["unidentifiable"] == 1
    )
    again = evaluate_baseline(cfg, date="2026-10-12")[1]
    assert again.name.endswith("_2")  # a second report never overwrites the first


def test_evaluate_without_a_baseline_says_to_run_it(run):
    build_manifest(run)
    with pytest.raises(FileNotFoundError, match="run the baseline first"):
        evaluate_baseline(run)


# ------------------------------------------------------------------------- export-review


def test_export_review_writes_hand_out_files_and_a_private_key(baseline_run):
    cfg = baseline_run
    review, files = export_blind_review(cfg, "Acropora", n_pred=2, n_other=2, n_assaf=2, pdf=False)
    assert review.meta["n_on_sheet"] == len(review.sheet)
    exports, keys = cfg.reviews_dir / "exports", cfg.reviews_dir / "keys"
    assert files["analyst_csv"].parent == exports and files["key"].parent == keys
    sheet = pd.read_csv(files["analyst_csv"])
    assert list(sheet.columns) == ["key", "is_acropora", "actual_genus", "notes"]  # blind
    with pytest.raises(FileExistsError):  # the same seed would overwrite
        export_blind_review(cfg, "Acropora", n_pred=2, n_other=2, n_assaf=2, pdf=False)


# ------------------------------------------------------------------------- train-probe


def test_train_probe_needs_labels_it_never_makes_its_own(baseline_run):
    with pytest.raises(ProbeError, match="never makes its own"):
        train_probe(baseline_run)
    baseline_run.probe.labels = "missing.csv"
    with pytest.raises(FileNotFoundError):
        train_probe(baseline_run)


def test_train_probe_reports_held_out_accuracy_and_saves_a_new_folder(baseline_run):
    cfg = baseline_run
    labels = labels_for(cfg, ["Acropora", "Porites"])
    cfg.probe.n_splits = 3
    report, data, folder = train_probe(cfg, labels, name="unit", date="2026-10-12")
    assert folder == cfg.reports_dir / "probe_unit_2026-10-12"
    assert len(data) == 6 and report.n_labeled == 6 and report.n_images == 3
    assert (
        report.zero_shot is not None and report.zero_shot["total"] == 6
    )  # the baseline, same crops
    assert report.accuracy["total"] == 6
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["labels_dropped"] == data.dropped and summary["params"]["n_splits"] == 3
    # labels can come from the config's path instead of an argument
    path = cfg.data_root / "labels.csv"
    labels.to_csv(path, index=False)
    cfg.probe.labels = "labels.csv"
    assert train_probe(cfg, name="unit", date="2026-10-12")[2].name == "probe_unit_2026-10-12_2"


def test_train_probe_can_use_the_dino_features(baseline_run):
    cfg = baseline_run
    run_embeddings(cfg, "dinov2-s14", encoder=FakeEncoder(dim=8))
    cfg.probe.features = embed_paths(cfg, "dinov2-s14").name
    cfg.probe.n_splits = 3
    report, _, _ = train_probe(cfg, labels_for(cfg, ["Acropora", "Porites"]), date="2026-10-12")
    assert report.params["standardize"] is True and report.n_labeled == 6
