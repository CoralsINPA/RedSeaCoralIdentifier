import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from coralid import cli
from coralid.baseline import baseline_paths, run_baseline
from coralid.cli import COMMANDS, build_parser, main
from coralid.config import Config, Paths, save_config
from coralid.explore.embed import EmbedIncomplete, embed_paths
from coralid.segmentation import SegmentationResult
from tests.fake_encoder import FakeEncoder
from tests.run_data import labels_for, make_run, write_expert_inputs

COMMAND_NAMES = [
    "segment",
    "build-manifest",
    "embed",
    "zero-shot",
    "evaluate",
    "export-review",
    "train-probe",
]


@pytest.fixture
def cfg(tmp_path):
    return make_run(tmp_path / "data", manifest=False)


@pytest.fixture
def args(cfg, tmp_path):
    """`--config` and `--data-root` for the run; the config file itself has no data root."""
    path = tmp_path / "run.yaml"
    config = Config(filters=cfg.filters, crops=cfg.crops, paths=Paths())
    save_config(config, path)
    return ["--config", str(path), "--data-root", str(cfg.data_root), "--quiet"]


def _with_manifest(cfg):
    from coralid.jobs import build_manifest

    build_manifest(cfg)
    return cfg


# ------------------------------------------------------------------------- parsing


def test_every_command_of_the_plan_exists_and_dispatches():
    assert set(COMMANDS) == set(COMMAND_NAMES)
    parser = build_parser()
    for name in COMMAND_NAMES:
        parsed = parser.parse_args([name, "--config", "c.yaml"])
        assert parsed.command == name and parsed.config == "c.yaml" and parsed.data_root is None
        with pytest.raises(SystemExit):  # --config is required by every command
            parser.parse_args([name])
    with pytest.raises(SystemExit):
        parser.parse_args(["tier", "--config", "c.yaml"])  # not a command (it was dropped)


def test_options_of_each_command_parse():
    p = build_parser()
    a = p.parse_args(
        [
            "segment",
            "--config",
            "c",
            "--limit",
            "5",
            "--images",
            "A.JPG",
            "B.JPG",
            "--device",
            "cpu",
        ]
    )
    assert (a.limit, a.images, a.device, a.dry_run, a.save_features) == (
        5,
        ["A.JPG", "B.JPG"],
        "cpu",
        False,
        False,
    )
    assert p.parse_args(["segment", "--config", "c", "--dry-run", "--save-features"]).dry_run
    assert p.parse_args(["build-manifest", "--config", "c", "--geometric"]).geometric
    e = p.parse_args(["embed", "--config", "c", "--model", "dinovdeau-l", "--batch-size", "8"])
    assert (e.model, e.batch_size) == ("dinovdeau-l", 8)
    r = p.parse_args(["export-review", "--config", "c", "--genus", "Porites", "--no-pdf"])
    assert (r.genus, r.n_pred, r.n_other, r.n_assaf, r.seed, r.no_pdf) == (
        "Porites",
        150,
        150,
        40,
        0,
        True,
    )
    t = p.parse_args(["train-probe", "--config", "c", "--class-weight", "none", "--C", "0.1"])
    assert (t.class_weight, t.C, t.n_splits, t.no_standardize) == ("none", 0.1, None, False)
    with pytest.raises(SystemExit):
        p.parse_args(["train-probe", "--config", "c", "--class-weight", "heavy"])


def test_help_and_version_work_without_torch_or_matplotlib():
    code = (
        "import sys; from coralid.cli import main\n"
        "try: main(['--help'])\n"
        "except SystemExit as e: assert e.code == 0\n"
        "heavy = {'torch', 'torchvision', 'open_clip', 'transformers', 'umap', 'hdbscan', "
        "'sklearn', 'matplotlib', 'segment_anything'}\n"
        "assert not heavy & set(sys.modules), heavy & set(sys.modules)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
    out = subprocess.run(
        [sys.executable, "-m", "coralid", "--help"], check=True, capture_output=True, text=True
    ).stdout
    for name in COMMAND_NAMES:
        assert name in out


# ------------------------------------------------------------------------- dispatch with fakes


def test_segment_passes_its_options_to_the_job(args, monkeypatch, capsys):
    seen = {}

    def fake(cfg, **kwargs):
        seen.update(kwargs, save_features=cfg.sam.save_features)
        return SegmentationResult(n_images=10, n_already_done=7, written=["a", "b"], n_empty=1)

    monkeypatch.setattr(cli, "run_segmentation", fake)
    code = main(
        [
            "segment",
            *args,
            "--limit",
            "3",
            "--images",
            "A.JPG",
            "--device",
            "cpu",
            "--save-features",
        ]
    )
    assert code == 0
    assert seen == {
        "image_names": ["A.JPG"],
        "limit": 3,
        "device": "cpu",
        "progress": False,
        "save_features": True,
    }
    assert "2 written (1 with no masks)" in capsys.readouterr().out


def test_segment_exits_2_and_names_the_failed_photos(args, monkeypatch, capsys):
    failed = SegmentationResult(failures=[{"image": "IMG_2.JPG", "error": "RuntimeError: OOM"}])
    monkeypatch.setattr(cli, "run_segmentation", lambda cfg, **kw: failed)
    assert main(["segment", *args]) == 2
    assert "INCOMPLETE" in capsys.readouterr().err


def test_embed_passes_the_model_and_the_batch_size(args, monkeypatch, capsys):
    seen = {}

    def fake(cfg, model, **kwargs):
        seen.update(model=model, batch_size=cfg.embed.batch_size, **kwargs)
        from coralid.explore.embed import EmbedPaths, EmbedResult

        paths = EmbedPaths("x", cfg.features_dir / "x.npz", cfg.features_dir / "parts")
        return EmbedResult(["A#0"], np.zeros((1, 4), np.float32), paths, "computed")

    monkeypatch.setattr(cli, "run_embeddings", fake)
    assert main(["embed", *args, "--model", "dinovdeau-l", "--batch-size", "8"]) == 0
    assert seen == {"model": "dinovdeau-l", "batch_size": 8, "progress": False}
    assert "Wrote: 1 crops, 4 dimensions" in capsys.readouterr().out


def test_an_incomplete_job_is_exit_code_2_with_the_reason(args, monkeypatch, capsys):
    def incomplete(cfg, model, **kwargs):
        raise EmbedIncomplete([{"image": "IMG_1.JPG", "error": "boom"}])

    monkeypatch.setattr(cli, "run_embeddings", incomplete)
    assert main(["embed", *args]) == 2
    assert "INCOMPLETE" in capsys.readouterr().err


def test_train_probe_options_override_the_config(args, monkeypatch):
    seen = {}

    def fake(cfg, labels, **kwargs):
        seen.update(vars(cfg.probe), labels=labels, **kwargs)
        raise cli.ProbeError("stop here")

    monkeypatch.setattr(cli, "train_probe", fake)
    code = main(
        [
            "train-probe",
            *args,
            "--labels",
            "l.csv",
            "--label-column",
            "expert_genus",
            "--features",
            "dinov2-s14__masked_square",
            "--n-splits",
            "3",
            "--C",
            "0.1",
            "--class-weight",
            "none",
            "--no-standardize",
            "--no-zero-shot",
            "--name",
            "t",
        ]
    )
    assert code == 2
    assert (seen["labels"], seen["label_column"], seen["features"]) == (
        "l.csv",
        "expert_genus",
        "dinov2-s14__masked_square",
    )
    assert (seen["n_splits"], seen["C"], seen["class_weight"]) == (3, 0.1, None)
    assert seen["standardize"] is False and seen["compare_zero_shot"] is False
    assert seen["name"] == "t"


def test_a_bad_config_or_data_root_is_a_message_not_a_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CORALID_DATA_ROOT", raising=False)
    path = tmp_path / "run.yaml"
    save_config(Config(), path)
    assert main(["build-manifest", "--config", str(path)]) == 2
    assert "CORALID_DATA_ROOT" in capsys.readouterr().err
    assert main(["build-manifest", "--config", str(tmp_path / "missing.yaml")]) == 2
    assert "STOPPED" in capsys.readouterr().err


# ------------------------------------------------------------------------- the real jobs


def test_segment_on_a_full_run_folder_changes_nothing(cfg, args, capsys):
    """The active run has all 1,714 JSONs: `segment` with its config must be harmless."""
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in cfg.json_dir.iterdir()}
    assert main(["segment", *args]) == 0  # no checkpoint, no model: nothing to do
    assert "3 already had a JSON (untouched), 0 written" in capsys.readouterr().out
    after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in cfg.json_dir.iterdir()}
    assert after == before and not cfg.segmentation_dir.exists()
    assert main(["segment", *args, "--dry-run"]) == 0
    assert "3 photos: 3 have a JSON (never rewritten), 0 to segment" in capsys.readouterr().out


def test_segment_without_the_checkpoint_stops_cleanly(cfg, args, capsys):
    (cfg.json_dir / "IMG_3.json").unlink()
    assert main(["segment", *args]) == 2
    assert "STOPPED" in capsys.readouterr().err  # CoralSCOP or its checkpoint is missing
    assert not (cfg.json_dir / "IMG_3.json").exists()  # and no empty JSON was made up


def test_build_manifest_command(cfg, args, capsys):
    assert main(["build-manifest", *args]) == 0
    out = capsys.readouterr().out
    assert "Wrote: 6 masks on 3 photos, 6 kept" in out and "kept 6" in out
    assert main(["build-manifest", *args]) == 0
    assert "Loaded the saved manifest" in capsys.readouterr().out
    assert main(["build-manifest", *args, "--geometric"]) == 0
    assert "manifest_geometric.parquet" in capsys.readouterr().out


def test_embed_command(cfg, args, monkeypatch, capsys):
    _with_manifest(cfg)
    from coralid.explore import embed

    monkeypatch.setattr(embed, "load_dino_encoder", lambda name: FakeEncoder(dim=8))
    assert main(["embed", *args]) == 0
    assert "Wrote: 6 crops, 8 dimensions" in capsys.readouterr().out
    assert embed_paths(cfg).features.exists()
    assert main(["embed", *args]) == 0
    assert "Loaded the saved features" in capsys.readouterr().out
    assert main(["embed", *args, "--model", "finetuned-bioclip"]) == 2
    assert "zero-shot" in capsys.readouterr().err


def test_evaluate_and_export_review_and_train_probe_commands(cfg, args, capsys):
    _with_manifest(cfg)
    run_baseline(cfg, encoder=FakeEncoder(dim=8))
    predictions = pd.read_parquet(baseline_paths(cfg).predictions)
    write_expert_inputs(cfg, predictions)

    assert main(["evaluate", *args]) == 0
    out = capsys.readouterr().out
    assert "correct 3/4" in out and "report:" in out
    assert any((cfg.reports_dir).glob("baseline_*/summary.json"))

    assert (
        main(
            [
                "export-review",
                *args,
                "--genus",
                "Acropora",
                "--no-pdf",
                "--n-pred",
                "2",
                "--n-other",
                "2",
                "--n-assaf",
                "2",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "KEEP" in out and "PRIVATE" in out
    assert any((cfg.reviews_dir / "keys").glob("*_answer_key.csv"))
    assert (
        main(
            [
                "export-review",
                *args,
                "--genus",
                "Acropora",
                "--no-pdf",
                "--n-pred",
                "2",
                "--n-other",
                "2",
                "--n-assaf",
                "2",
            ]
        )
        == 2
    )  # same seed: would overwrite
    assert "STOPPED" in capsys.readouterr().err

    labels = cfg.data_root / "labels.csv"
    labels_for(cfg, ["Acropora", "Porites"]).to_csv(labels, index=False)
    assert main(["train-probe", *args, "--labels", str(labels), "--n-splits", "3"]) == 0
    out = capsys.readouterr().out
    assert "Held-out accuracy" in out and "zero-shot on the same crops" in out
    assert main(["train-probe", *args]) == 2  # no labels: it never makes its own
    assert "never makes its own" in capsys.readouterr().err
    summary = next(cfg.reports_dir.glob("probe_*/summary.json"))
    assert json.loads(summary.read_text())["kind"] == "linear_probe_held_out"
