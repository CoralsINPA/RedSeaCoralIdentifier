"""The 20-genus baseline job: zero-shot labels and image embeddings for every kept crop.

One configuration (by default the fine-tuned BioCLIP, `taxonomy_v1` prompts, `hard20`, masked
crops) over every `keep == True` row of the run's saved manifest. It writes two files, each with
a `*.meta.json` sidecar, into the run folder:

- `predictions/<model>__<prompts>__<label set>__<subset>.parquet`: the legacy columns plus
  `runner_up` and `margin`;
- `features/<model>__<crop mode>.npz`: `uids` and `X`, the L2-normalized image features.

The job is resumable through `ResumableRun`: finished images are saved in
`predictions/parts/` as it goes, a failed image is logged and retried on the next call, and the
two result files are written only when every image is done. It never overwrites a file: if both
results exist it just loads them. Run it from Python (`run_baseline(cfg)`), a notebook, or
`python -m coralid zero-shot --config ...`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coralid._atomic import atomic_path, atomic_write_text
from coralid.batch import ResumableRun
from coralid.config import Config, write_meta
from coralid.encoders import Encoder, load_encoder
from coralid.genera import load_genera
from coralid.manifest import read_manifest
from coralid.zero_shot import build_text_features, load_embeddings, run_zero_shot, save_embeddings

logger = logging.getLogger(__name__)


class BaselineError(RuntimeError):
    """The baseline cannot run or finish as asked."""


class BaselineIncomplete(BaselineError):
    """Some images failed. Finished images are saved; run the job again to retry the others."""

    def __init__(self, failures: list[dict[str, Any]]) -> None:
        self.failures = failures
        shown = "; ".join(f"{f['image']}: {f['error']}" for f in failures[:5])
        super().__init__(
            f"{len(failures)} image(s) failed and were not saved as done ({shown}). "
            "Nothing was written. Run the job again to retry them."
        )


@dataclass(frozen=True)
class BaselinePaths:
    """Where the baseline's files go."""

    name: str
    predictions: Path
    features: Path
    parts_dir: Path


@dataclass
class BaselineResult:
    """What `run_baseline` returns."""

    predictions: pd.DataFrame
    uids: list[str]
    features: np.ndarray
    paths: BaselinePaths
    status: str  # "computed" (written now) or "already_done" (loaded)
    n_images_processed: int = 0  # images classified in this call


def prompts_tag(cfg: Config) -> str:
    """The prompts part of the file name: `taxonomy_v1`, or e.g. `ensemble_taxonomy_v1`."""
    zs = cfg.zero_shot
    if zs.prompt_format == "taxonomy":
        return zs.prompt_set
    if zs.prompt_format == "ensemble":
        return f"ensemble_{zs.prompt_set}"
    return zs.prompt_format


def baseline_paths(cfg: Config, subset: str = "all") -> BaselinePaths:
    """File locations of a configuration: `finetuned-bioclip__taxonomy_v1__hard20__all.parquet`."""
    zs = cfg.zero_shot
    name = f"{zs.model}__{prompts_tag(cfg)}__{zs.label_set}__{subset}"
    return BaselinePaths(
        name=name,
        predictions=cfg.predictions_dir / f"{name}.parquet",
        features=cfg.features_dir / f"{zs.model}__{cfg.crops.mode}.npz",
        parts_dir=cfg.predictions_dir / "parts",
    )


def load_baseline(cfg: Config, subset: str = "all") -> BaselineResult:
    """Read the saved baseline of a configuration (`FileNotFoundError` if it is not there)."""
    paths = baseline_paths(cfg, subset)
    for path in (paths.predictions, paths.features):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run the baseline first")
    predictions = pd.read_parquet(paths.predictions)
    uids, features = load_embeddings(paths.features)
    return BaselineResult(predictions, uids, features, paths, "already_done")


def _fingerprint(kept: pd.DataFrame, cfg: Config, genera: list[str]) -> str:
    """A short hash of everything that decides the results, to refuse mixed-up parts."""
    zs = cfg.zero_shot
    payload = {
        "uids": sorted(
            f"{n}#{c}" for n, c in zip(kept["image_name"], kept["coral_id"], strict=True)
        ),
        "zero_shot": asdict(zs),
        "genera": genera,
        "crops": asdict(cfg.crops),
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _check_inputs(paths: BaselinePaths, fingerprint: str, n_keep: int, run: ResumableRun) -> None:
    """Record the inputs next to the parts; refuse to resume parts made from other inputs."""
    path = paths.parts_dir / f"{paths.name}.input.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("fingerprint") != fingerprint:
            raise BaselineError(
                f"{paths.parts_dir} holds parts made from other inputs (another manifest, model, "
                "prompts or crop settings). Move that folder aside to start again."
            )
        return
    if run.done:
        raise BaselineError(
            f"{paths.parts_dir} has finished images but no record of their inputs; "
            "move that folder aside to start again."
        )
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = {"fingerprint": fingerprint, "n_keep": n_keep, "created_utc": created}
    atomic_write_text(path, json.dumps(body, indent=1))


def run_baseline(
    cfg: Config,
    *,
    encoder: Encoder | None = None,
    subset: str = "all",
    progress: bool = False,
    flush_every: int = 25,
    group_size: int = 32,
) -> BaselineResult:
    """Classify every kept crop of the run's manifest and save predictions and embeddings.

    Resumes from `predictions/parts/` if an earlier call was interrupted. If both result files
    exist they are loaded and nothing is recomputed or overwritten; if only one exists that is
    an error. Raises `BaselineIncomplete` (and writes nothing) if any image failed.
    `encoder` is for tests; by default the configured model is loaded. `group_size` is how
    many images are cut and encoded together.
    """
    paths = baseline_paths(cfg, subset)
    exists = [paths.predictions.exists(), paths.features.exists()]
    if all(exists):
        logger.info("The baseline is already saved; loading %s", paths.predictions)
        return load_baseline(cfg, subset)
    if any(exists):
        done = paths.predictions if exists[0] else paths.features
        raise BaselineError(f"Only {done} exists, not its partner; move it aside and run again.")

    manifest_path = cfg.manifest_path
    if not manifest_path.exists():
        raise BaselineError(
            f"{manifest_path} not found. The manifest comes from the parity notebook's Check C."
        )
    manifest = read_manifest(manifest_path)
    kept = manifest[manifest["keep"].astype(bool)]
    if kept.empty:
        raise BaselineError(f"{manifest_path} keeps no crops")

    zs = cfg.zero_shot
    genera = load_genera().genera(zs.label_set)
    run = ResumableRun(paths.parts_dir, paths.name, flush_every=flush_every)
    _check_inputs(paths, _fingerprint(kept, cfg, genera), len(kept), run)

    encoder = encoder or load_encoder(zs.model)
    text = build_text_features(encoder, zs.label_set, zs.prompt_format, zs.prompt_set)
    result = run_zero_shot(
        encoder,
        text,
        genera,
        manifest,
        cfg.json_dir,
        cfg.images_dir,
        crops=cfg.crops,
        batch_size=zs.batch_size,
        logit_scale=zs.logit_scale,
        run=run,
        keep_embeddings=True,
        group_size=group_size,
        progress=progress,
    )
    if result.failures or run.pending(list(kept["image_name"].unique())):
        raise BaselineIncomplete(run.unresolved_failures() or result.failures)

    rows, features = run.merge()
    expected = [f"{n}#{int(c)}" for n, c in zip(kept["image_name"], kept["coral_id"], strict=True)]
    uids = [f"{n}#{int(c)}" for n, c in zip(rows["image_name"], rows["coral_id"], strict=True)]
    if features is None or len(rows) != len(kept) or set(uids) != set(expected):
        raise BaselineError(
            f"The saved parts hold {len(rows)} crops, the manifest keeps {len(kept)}. "
            "Move the parts folder aside and run again."
        )
    # parts are in the order the images were finished; save in manifest order so that an
    # interrupted and resumed run gives the same files as an uninterrupted one
    position = {uid: i for i, uid in enumerate(expected)}
    order = np.argsort([position[uid] for uid in uids], kind="stable")
    rows = rows.iloc[order].reset_index(drop=True)
    features = features[order]
    uids = [uids[i] for i in order]
    with atomic_path(paths.predictions) as tmp:
        rows.to_parquet(tmp, index=False)
    save_embeddings(paths.features, uids, features)
    _write_metas(cfg, paths, subset, rows, features, kept)
    logger.info("Baseline written: %s and %s", paths.predictions, paths.features)
    return BaselineResult(rows, uids, features, paths, "computed", result.n_images)


def _write_metas(
    cfg: Config,
    paths: BaselinePaths,
    subset: str,
    rows: pd.DataFrame,
    features: np.ndarray,
    kept: pd.DataFrame,
) -> None:
    zs = cfg.zero_shot
    common: dict[str, Any] = {
        "run_name": cfg.run_name,
        "model": zs.model,
        "label_set": zs.label_set,
        "prompt_format": zs.prompt_format,
        "prompt_set": zs.prompt_set,
        "crop_mode": cfg.crops.mode,
        "subset": subset,
        "manifest": str(cfg.manifest_path),
        "n_manifest_keep": len(kept),
        "n_images": int(kept["image_name"].nunique()),
    }
    write_meta(
        paths.predictions,
        config=cfg,
        n_rows=len(rows),
        kind="predictions",
        columns=list(rows.columns),
        features_file=paths.features.name,
        **common,
    )
    write_meta(
        paths.features,
        config=cfg,
        n_rows=len(features),
        kind="image_features",
        dim=int(features.shape[1]),
        normalized="l2",
        arrays={"uids": "coral uids, row-aligned with X", "X": "float32 image features"},
        predictions_file=paths.predictions.name,
        **common,
    )
