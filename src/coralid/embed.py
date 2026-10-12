"""The embedding job: DINO features of the masked square crop of every kept mask.

DINOv2 ViT-S/14 (384-d) and DinoVdeau-large (1024-d), as in CM cells 9 and 11, but run over the
run's saved manifest: every `keep == True` row, cut with `crops.masked_square_crop` from the photo
loaded in the mask frame (the legacy cells used every mask, or filters of their own). The result
follows the convention of the zero-shot baseline:

- `features/<model>__masked_square.npz` with `uids` (`IMG_4033.JPG#14`) and `X`, the raw
  (not normalized) float32 features, and its `*.meta.json`;
- progress in `features/parts/` (a `ResumableRun`): finished images are kept as it goes, a failed
  image goes to the failure log and is retried on the next call;
- the result file is written only when every image is done, and never over an existing file: if
  it exists it is loaded. Parts made from other inputs are refused.

The CLIP-family models are not run here: their features come with the predictions of
`coralid.baseline` (`features/<model>__masked.npz`).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coralid._atomic import atomic_write_text
from coralid.batch import ResumableRun
from coralid.config import Config, write_meta
from coralid.crops import masked_square_crop
from coralid.encoders import DINO_MODELS, ImageEncoder, load_dino_encoder
from coralid.images import load_in_mask_frame
from coralid.manifest import read_manifest
from coralid.masks import json_name_for, load_mask_json
from coralid.zero_shot import load_embeddings, save_embeddings

logger = logging.getLogger(__name__)

CROP_MODE = "masked_square"


class EmbedError(RuntimeError):
    """The embedding job cannot run or finish as asked."""


class EmbedIncomplete(EmbedError):
    """Some images failed. Finished images are saved; run the job again to retry the others."""

    def __init__(self, failures: list[dict[str, Any]]) -> None:
        self.failures = failures
        shown = "; ".join(f"{f['image']}: {f['error']}" for f in failures[:5])
        super().__init__(
            f"{len(failures)} image(s) failed and were not saved as done ({shown}). "
            "Nothing was written. Run the job again to retry them."
        )


@dataclass(frozen=True)
class EmbedPaths:
    """Where the embedding job's files go."""

    name: str
    features: Path
    parts_dir: Path


@dataclass
class EmbedResult:
    """What `run_embeddings` returns."""

    uids: list[str]
    features: np.ndarray
    paths: EmbedPaths
    status: str  # "computed" (written now) or "already_done" (loaded)
    n_images_processed: int = 0  # images embedded in this call


def embed_paths(cfg: Config, model: str | None = None) -> EmbedPaths:
    """File locations of a model: `features/dinov2-s14__masked_square.npz`."""
    name = f"{model or cfg.embed.model}__{CROP_MODE}"
    return EmbedPaths(name, cfg.features_dir / f"{name}.npz", cfg.features_dir / "parts")


def load_features(cfg: Config, model: str | None = None) -> EmbedResult:
    """Read the saved features of a model (`FileNotFoundError` if they are not there)."""
    paths = embed_paths(cfg, model)
    if not paths.features.exists():
        raise FileNotFoundError(f"{paths.features} not found; run the embedding job first")
    uids, features = load_embeddings(paths.features)
    return EmbedResult(uids, features, paths, "already_done")


def _uid_list(df: pd.DataFrame) -> list[str]:
    return [f"{n}#{int(c)}" for n, c in zip(df["image_name"], df["coral_id"], strict=True)]


def _record_inputs(paths: EmbedPaths, fingerprint: str, n_keep: int, run: ResumableRun) -> None:
    """Save what the parts are made from; refuse to resume parts made from other inputs."""
    path = paths.parts_dir / f"{paths.name}.input.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")).get("fingerprint") != fingerprint:
            raise EmbedError(
                f"{paths.parts_dir} holds parts made from other inputs (another manifest or "
                "model). Move that folder aside to start again."
            )
        return
    if run.done:
        raise EmbedError(
            f"{paths.parts_dir} has finished images but no record of their inputs; "
            "move that folder aside to start again."
        )
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = {"fingerprint": fingerprint, "n_keep": n_keep, "created_utc": created}
    atomic_write_text(path, json.dumps(body, indent=1))


def _square_items(
    encoder: ImageEncoder, image_name: str, keep_rows: pd.DataFrame, cfg: Config
) -> list[Any]:
    """Cut and preprocess the masked square crop of one image's kept masks, in row order."""
    from PIL import Image

    mask_file = load_mask_json(cfg.json_dir / json_name_for(image_name))
    image = load_in_mask_frame(cfg.images_dir / image_name, *mask_file.frame)
    by_id = mask_file.by_id()
    items = []
    for coral_id in keep_rows["coral_id"]:
        anno = by_id[int(coral_id)]
        crop = masked_square_crop(image, anno["bbox"], anno["segmentation"])
        if crop is None:
            raise EmbedError(f"{image_name}#{int(coral_id)}: kept in the manifest, empty crop")
        items.append(encoder.preprocess(Image.fromarray(crop)))
    return items


def run_embeddings(
    cfg: Config,
    model: str | None = None,
    *,
    encoder: ImageEncoder | None = None,
    progress: bool = False,
    flush_every: int = 25,
    group_size: int = 16,
    max_consecutive_failures: int = 5,
) -> EmbedResult:
    """Embed the masked square crop of every kept mask and save `features/<model>__<crop>.npz`.

    Resumes from `features/parts/`. If the result exists it is loaded and nothing is recomputed
    or overwritten. Raises `EmbedIncomplete` (and writes nothing) if any image failed.
    `encoder` is for tests; by default the configured DINO model is loaded. `group_size` is how
    many images are cut and encoded together; the batch size is `cfg.embed.batch_size` or the
    model's own (DINOv2 64, DinoVdeau 16).
    """
    model = model or cfg.embed.model
    if encoder is None and model not in DINO_MODELS:
        raise EmbedError(
            f"'{model}' is not a DINO model ({sorted(DINO_MODELS)}). The features of the "
            "CLIP-family models come with their predictions: python -m coralid zero-shot."
        )
    paths = embed_paths(cfg, model)
    if paths.features.exists():
        logger.info("The features are already saved; loading %s", paths.features)
        return load_features(cfg, model)
    if not cfg.manifest_path.exists():
        raise EmbedError(f"{cfg.manifest_path} not found. Build the manifest first.")
    manifest = read_manifest(cfg.manifest_path)
    kept = manifest[manifest["keep"].astype(bool)]
    if kept.empty:
        raise EmbedError(f"{cfg.manifest_path} keeps no crops")
    expected = _uid_list(kept)

    run = ResumableRun(paths.parts_dir, paths.name, flush_every=flush_every)
    text = json.dumps({"uids": sorted(expected), "model": model, "crop": CROP_MODE})
    _record_inputs(paths, hashlib.sha256(text.encode()).hexdigest()[:16], len(kept), run)
    encoder = encoder or load_dino_encoder(model)
    batch_size = cfg.embed.batch_size or getattr(encoder, "batch_size", 64)

    by_image = {name: rows for name, rows in kept.groupby("image_name", sort=False)}
    todo = run.pending(list(by_image))
    groups = [todo[i : i + group_size] for i in range(0, len(todo), group_size)]
    iterator: Any = groups
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(groups, desc=f"Embedding with {model}")
    n_images = consecutive = 0
    try:
        for group in iterator:
            prepared: list[tuple[str, pd.DataFrame, list[Any]]] = []
            for name in group:
                try:
                    items = _square_items(encoder, name, by_image[name], cfg)
                except Exception as err:  # logged per image, retried on resume
                    run.fail(name, err)
                    consecutive = _count(consecutive + 1, max_consecutive_failures)
                    continue
                prepared.append((name, by_image[name], items))
            if not prepared:
                continue
            flat = [item for _, _, items in prepared for item in items]
            try:
                X = np.vstack(
                    [
                        encoder.encode_images(flat[i : i + batch_size])
                        for i in range(0, len(flat), batch_size)
                    ]
                ).astype(np.float32)
            except Exception as err:  # the whole group is retried on resume
                for name, _, _ in prepared:
                    run.fail(name, err)
                consecutive = _count(consecutive + len(prepared), max_consecutive_failures)
                continue
            consecutive = start = 0
            for name, keep_rows, items in prepared:
                stop = start + len(items)
                rows = keep_rows[["image_name", "coral_id"]].reset_index(drop=True)
                run.add(name, rows, X[start:stop])
                start = stop
                n_images += 1
    finally:
        run.flush()
    if run.pending(list(by_image)):
        raise EmbedIncomplete(run.unresolved_failures())

    rows, matrix = run.merge()
    uids = _uid_list(rows)
    if matrix is None or len(rows) != len(kept) or set(uids) != set(expected):
        raise EmbedError(
            f"The saved parts hold {len(rows)} crops, the manifest keeps {len(kept)}. "
            "Move the parts folder aside and run again."
        )
    # parts are in the order the images were finished: save in manifest order
    position = {uid: i for i, uid in enumerate(expected)}
    order = np.argsort([position[uid] for uid in uids], kind="stable")
    matrix, uids = matrix[order], [uids[i] for i in order]
    save_embeddings(paths.features, uids, matrix)
    write_meta(
        paths.features,
        config=cfg,
        n_rows=len(matrix),
        kind="image_features",
        model=model,
        model_source=DINO_MODELS[model]["source"] if model in DINO_MODELS else None,
        crop_mode=CROP_MODE,
        dim=int(matrix.shape[1]),
        normalized="none",
        batch_size=batch_size,
        arrays={"uids": "coral uids, row-aligned with X", "X": "float32 raw model features"},
        manifest=str(cfg.manifest_path),
        n_manifest_keep=len(kept),
        n_images=int(kept["image_name"].nunique()),
    )
    logger.info("Features written: %s %s", paths.features, matrix.shape)
    return EmbedResult(uids, matrix, paths, "computed", n_images)


def _count(consecutive: int, limit: int) -> int:
    if consecutive >= limit:
        raise RuntimeError(
            f"Stopping after {consecutive} consecutive failures; see the failure log. "
            "Finished images are saved; fix the cause and run again to resume."
        )
    return consecutive
