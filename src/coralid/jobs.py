"""Whole-run jobs behind the command line: each takes a `Config` and does one step of the pipeline.

They add no logic of their own, only the wiring of the config's paths to the library functions:

- `build_manifest`: the run's `manifest/manifest.parquet`, resumable, never overwritten;
- `evaluate_baseline`: the expert-set evaluation and genus mix of the saved baseline, as a new
  folder under `reports/`;
- `export_blind_review`: a blind review sheet of one genus (hand-out files and a hidden key);
- `train_probe`: the linear probe's held-out report, as a new folder under `reports/`.

The segmentation, embedding and zero-shot jobs have modules of their own
(`coralid.segmentation`, `coralid.embed`, `coralid.baseline`).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from coralid.baseline import baseline_paths, load_baseline
from coralid.batch import ResumableRun
from coralid.config import Config, meta_path
from coralid.evaluation import load_expert_set, load_legacy_predictions
from coralid.genus_audit import BlindReview, build_blind_review, load_legacy_audit
from coralid.manifest import geometric_manifest, image_manifest, read_manifest, write_manifest
from coralid.masks import AnnotationStore
from coralid.probe import (
    ProbeData,
    ProbeError,
    ProbeReport,
    cross_validate_probe,
    join_labels,
    load_labels,
)
from coralid.reports import (
    BaselineReport,
    baseline_report,
    write_baseline_report,
    write_probe_report,
)
from coralid.review_export import write_blind_review
from coralid.zero_shot import load_embeddings

logger = logging.getLogger(__name__)

EXPERT_CSV = "assaf_expert_test_set.csv"  # in the run folder
LEGACY_AUDIT = {"Millepora": "v2_pipeline/millepora_sharp_audit.csv"}  # in the run folder


class JobError(RuntimeError):
    """A job cannot run or finish as asked."""


class ManifestIncomplete(JobError):
    """Some images failed. Finished images are saved; run the job again to retry the others."""

    def __init__(self, failures: list[dict[str, Any]]) -> None:
        self.failures = failures
        shown = "; ".join(f"{f['image']}: {f['error']}" for f in failures[:5])
        super().__init__(
            f"{len(failures)} image(s) failed and were not saved as done ({shown}). "
            "No manifest was written. Run the job again to retry them."
        )


# --------------------------------------------------------------------------- the manifest


@dataclass
class ManifestResult:
    """What `build_manifest` did."""

    manifest: pd.DataFrame
    path: Path
    status: str  # "computed" (written now) or "already_done" (loaded)


def _config_differences(path: Path, cfg: Config) -> list[str]:
    """What differs between the filter and crop settings that made a saved manifest and now."""
    meta_file = meta_path(path)
    if not meta_file.exists():
        logger.warning("%s has no meta file; its settings cannot be checked", path)
        return []
    saved = json.loads(meta_file.read_text(encoding="utf-8")).get("config") or {}
    now = cfg.to_dict()
    return [
        f"{section}.{key}: saved {saved[section][key]!r}, now {value!r}"
        for section in ("filters", "crops")
        for key, value in now[section].items()
        if section in saved and key in saved[section] and saved[section][key] != value
    ]


def build_manifest(
    cfg: Config, *, geometric: bool = False, progress: bool = False, flush_every: int = 25
) -> ManifestResult:
    """Build the run's manifest and save it as a new file (never over an existing one).

    The image manifest (visibility, sharpness, `keep`) reads every photo and is resumable:
    progress is kept in `manifest/parts/`, a failed image is retried on the next call, and the
    file is written only when every image is done. If a manifest exists it is loaded, unless it
    was made with other filter or crop settings, which is an error. With `geometric` the
    manifest needs no photos and goes to `manifest/manifest_geometric.parquet`.
    """
    path = cfg.manifest_path
    if geometric:
        path = path.with_name("manifest_geometric.parquet")
    if path.exists():
        diff = [] if geometric else _config_differences(path, cfg)
        if diff:
            raise JobError(
                f"{path} was made with other settings ({'; '.join(diff)}). "
                "Move it aside to build a new one."
            )
        logger.info("The manifest is already saved; loading %s", path)
        return ManifestResult(read_manifest(path), path, "already_done")
    if geometric:
        table = geometric_manifest(cfg.json_dir, cfg.filters, progress=progress)
    else:
        run = ResumableRun(path.parent / "parts", "manifest", flush_every=flush_every)
        table = image_manifest(
            cfg.json_dir, cfg.images_dir, cfg.filters, cfg.crops, progress=progress, run=run
        )
        if run.unresolved_failures():
            raise ManifestIncomplete(run.unresolved_failures())
    write_manifest(table, path, config=cfg, overwrite=False, run_name=cfg.run_name)
    return ManifestResult(table, path, "computed")


# --------------------------------------------------------------------------- reports


def evaluate_baseline(
    cfg: Config,
    expert_csv: str | os.PathLike[str] | None = None,
    date: str | None = None,
) -> tuple[BaselineReport, Path]:
    """Evaluate the saved baseline against the expert set and the survey; save a report folder.

    Reads the baseline predictions, the manifest, the expert set (default: the run folder's
    `assaf_expert_test_set.csv`, whose own stored predictions are scored on the same rows as the
    legacy result) and the survey counts. Returns the report and the folder it was written to.
    """
    expert_csv = Path(expert_csv) if expert_csv else cfg.run_dir / EXPERT_CSV
    label_set = cfg.zero_shot.label_set
    baseline = load_baseline(cfg)
    report = baseline_report(
        baseline.predictions,
        read_manifest(cfg.manifest_path),
        load_expert_set(expert_csv, label_set),
        load_legacy_predictions(expert_csv),
        cfg.ground_truth_path,
        label_set,
    )
    folder = write_baseline_report(report, cfg.reports_dir, baseline.paths.name, date)
    return report, folder


def export_blind_review(
    cfg: Config,
    genus: str = "Millepora",
    audit_csv: str | os.PathLike[str] | None = None,
    *,
    n_pred: int = 150,
    n_other: int = 150,
    n_assaf: int = 40,
    seed: int = 0,
    pdf: bool = True,
) -> tuple[BlindReview, dict[str, Path]]:
    """Sample a blind review of `genus` from the baseline and write its sheets and hidden key.

    Hand out `reviews/exports/`; keep `reviews/keys/` private. The old audit (default for
    Millepora: `v2_pipeline/millepora_sharp_audit.csv`) excludes crops already judged and forces
    the unsure ones back in. Files are new; an existing target is an error.
    """
    predictions = load_baseline(cfg).predictions
    if audit_csv is None and genus in LEGACY_AUDIT:
        candidate = cfg.run_dir / LEGACY_AUDIT[genus]
        audit_csv = candidate if candidate.exists() else None
    audit = load_legacy_audit(audit_csv) if audit_csv else None
    review = build_blind_review(
        predictions,
        read_manifest(cfg.manifest_path),
        genus,
        audit,
        n_pred=n_pred,
        n_other=n_other,
        n_assaf=n_assaf,
        seed=seed,
    )
    files = write_blind_review(
        review,
        cfg.reviews_dir / "exports",
        cfg.reviews_dir / "keys",
        AnnotationStore(cfg.json_dir),
        cfg.images_dir,
        crops=cfg.crops,
        pdf=pdf,
    )
    return review, files


# --------------------------------------------------------------------------- the probe


def train_probe(
    cfg: Config,
    labels: pd.DataFrame | str | os.PathLike[str] | None = None,
    *,
    name: str | None = None,
    compare_zero_shot: bool = True,
    date: str | None = None,
) -> tuple[ProbeReport, ProbeData, Path]:
    """Cross-validate the probe the config describes and save its held-out report.

    Features: `<run>/features/<probe.features>.npz`. Labels: the argument, else `probe.labels`;
    there is no default, because the probe never makes labels. The baseline's predictions, if
    saved, are scored on the same crops. Returns the report, the labeled data and the new folder.
    """
    p = cfg.probe
    labels = labels if labels is not None else cfg.probe_labels_path
    if labels is None:
        raise ProbeError(
            "No labels: give a labels table (--labels, or probe.labels in the config) with "
            f"image_name, coral_id and a '{p.label_column}' column. The probe never makes its own."
        )
    table = labels if isinstance(labels, pd.DataFrame) else load_labels(labels)
    features = cfg.features_dir / f"{p.features}.npz"
    if not features.exists():
        raise FileNotFoundError(f"{features} not found; run the job that makes these features")
    uids, X = load_embeddings(features)
    data = join_labels(uids, X, table, p.label_column, p.label_set)
    logger.info(
        "%d labeled crops on %d photos; labels left out: %s",
        len(data),
        len(set(data.groups)),
        data.dropped,
    )
    zero_shot = None
    if compare_zero_shot and baseline_paths(cfg).predictions.exists():
        zero_shot = pd.read_parquet(baseline_paths(cfg).predictions)
    report = cross_validate_probe(
        data,
        n_splits=p.n_splits,
        zero_shot=zero_shot,
        C=p.C,
        class_weight=p.class_weight,
        standardize=p.standardize,
        max_iter=p.max_iter,
        seed=p.seed,
    )
    folder = write_probe_report(report, cfg.reports_dir, name or p.features, date)
    return report, data, folder
