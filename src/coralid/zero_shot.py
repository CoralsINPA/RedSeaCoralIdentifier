"""Zero-shot genus scoring: text features, softmax scores, prediction tables, the batch loop.

The scoring is the legacy one (WC cell 14): image and text features are L2-normalized, and
`probs = softmax(100 * image @ text.T)` over the label set. The predicted genus is the argmax
(the first genus in label order on ties, like pandas `idxmax`) and the confidence is its
probability. The table adds `runner_up` and `margin` (top-1 minus top-2 probability).

`run_zero_shot` does not filter masks itself. It takes the image manifest of
`coralid.manifest` and classifies the rows with `keep == True`, so the keep decision (area,
aspect, containment, visibility, sharpness) is made once and reused.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coralid._atomic import atomic_path
from coralid.batch import ResumableRun
from coralid.config import CropConfig
from coralid.crops import masked_crop
from coralid.encoders import Encoder
from coralid.genera import GeneraConfig, build_prompts
from coralid.images import load_in_mask_frame
from coralid.masks import json_name_for, load_mask_json

logger = logging.getLogger(__name__)

META_COLUMNS = ["image_name", "coral_id", "bbox", "area"]


class CropError(RuntimeError):
    """The manifest keeps a mask whose crop could not be cut."""


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Scale each row to unit length (float32)."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, np.float32(1e-12))


def build_text_features(
    encoder: Encoder,
    label_set: str,
    prompt_format: str,
    prompt_set: str = "taxonomy_v1",
    config: GeneraConfig | None = None,
) -> np.ndarray:
    """Normalized text features, one row per genus of the label set, in label-set order.

    `ensemble` is the mean of the L2-normalized features of its prompt sets, renormalized.
    """
    prompt_lists = build_prompts(label_set, prompt_format, prompt_set, encoder.family, config)
    parts = [l2_normalize(encoder.encode_texts(prompts)) for prompts in prompt_lists]
    if len(parts) == 1:
        return parts[0]
    return l2_normalize(sum(parts) / len(parts))


def score(
    image_feats: np.ndarray, text_feats: np.ndarray, logit_scale: float = 100.0
) -> np.ndarray:
    """Probabilities `softmax(logit_scale * cos)` of each image over the genera (float32)."""
    logits = np.float32(logit_scale) * (l2_normalize(image_feats) @ l2_normalize(text_feats).T)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


def predictions_table(
    rows: pd.DataFrame | Sequence[dict[str, Any]], probs: np.ndarray, genera: Sequence[str]
) -> pd.DataFrame:
    """The legacy table plus `runner_up` and `margin`.

    Columns: `image_name, coral_id, bbox, area`, one probability column per genus (label
    order), `predicted_genus, confidence, runner_up, margin`.
    """
    meta = pd.DataFrame(rows)[META_COLUMNS].reset_index(drop=True)
    probs = np.asarray(probs, dtype=np.float32)
    genera = list(genera)
    if probs.shape != (len(meta), len(genera)):
        raise ValueError(f"probs has shape {probs.shape}, expected {(len(meta), len(genera))}")
    if len(genera) < 2:
        raise ValueError("A prediction table needs at least two genera")
    order = np.argsort(-probs, axis=1, kind="stable")  # ties: earlier genus first
    first, second = order[:, 0], order[:, 1]
    index = np.arange(len(meta))
    names = np.array(genera, dtype=object)
    table = pd.DataFrame(probs, columns=genera)
    table["predicted_genus"] = names[first]
    table["confidence"] = probs[index, first]
    table["runner_up"] = names[second]
    table["margin"] = probs[index, first] - probs[index, second]
    return pd.concat([meta, table], axis=1)


def save_embeddings(
    path: str | os.PathLike[str], uids: Sequence[str], X: np.ndarray, overwrite: bool = False
) -> Path:
    """Write embeddings as `.npz` with `uids` (strings) and `X` (n, d). Atomic; no overwrite."""
    if len(uids) != len(X):
        raise ValueError(f"{len(uids)} uids for {len(X)} embedding rows")
    if Path(path).exists() and not overwrite:
        raise FileExistsError(f"{path} exists; refusing to overwrite it")
    with atomic_path(path) as tmp:
        np.savez_compressed(tmp, uids=np.array(list(uids)), X=np.asarray(X))
    return Path(path)


def load_embeddings(path: str | os.PathLike[str]) -> tuple[list[str], np.ndarray]:
    """Read an embeddings `.npz`: `(uids, X)`."""
    with np.load(path) as data:
        return [str(u) for u in data["uids"]], data["X"]


# --------------------------------------------------------------------------- the batch loop


@dataclass
class ZeroShotResult:
    """What `run_zero_shot` returns."""

    predictions: pd.DataFrame
    embeddings: np.ndarray | None  # L2-normalized, row-aligned with `predictions`
    n_images: int = 0  # images processed in this call
    failures: list[dict[str, Any]] = field(default_factory=list)  # still unresolved


class _MemorySink:
    """Collects results in memory when no `ResumableRun` is given."""

    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []
        self.matrices: list[np.ndarray] = []
        self.failures: list[dict[str, Any]] = []

    def pending(self, names: list[str]) -> list[str]:
        return list(names)

    def add(self, name: str, rows: pd.DataFrame, embeddings: np.ndarray | None = None) -> None:
        self.frames.append(rows)
        if embeddings is not None:
            self.matrices.append(embeddings)

    def fail(self, name: str, error: BaseException) -> None:
        self.failures.append({"image": name, "error": f"{type(error).__name__}: {error}"})
        logger.warning("Image failed: %s: %s", name, error)

    def flush(self) -> None:
        pass

    def unresolved_failures(self) -> list[dict[str, Any]]:
        return self.failures

    def merge(self) -> tuple[pd.DataFrame, np.ndarray | None]:
        rows = pd.concat(self.frames, ignore_index=True) if self.frames else pd.DataFrame()
        return rows, (np.vstack(self.matrices) if self.matrices else None)


def _image_items(
    encoder: Encoder,
    image_name: str,
    keep_rows: pd.DataFrame,
    json_dir: Path,
    images_dir: Path,
    crops: CropConfig,
) -> list[Any]:
    """Cut and preprocess the crops of one image's kept masks, in row order."""
    from PIL import Image

    mask_file = load_mask_json(json_dir / json_name_for(image_name))
    image = load_in_mask_frame(images_dir / image_name, *mask_file.frame)
    by_id = mask_file.by_id()
    items = []
    for coral_id in keep_rows["coral_id"]:
        anno = by_id[int(coral_id)]
        crop = masked_crop(
            image,
            anno["bbox"],
            anno["segmentation"],
            padding_pct=crops.padding_pct,
            background=crops.background,
            min_visible=crops.min_visible,
        )
        if crop is None:
            raise CropError(
                f"{image_name}#{int(coral_id)}: kept in the manifest, but the crop is empty or "
                f"has fewer than {crops.min_visible} visible pixels (manifest from another "
                "crop config?)"
            )
        items.append(encoder.preprocess(Image.fromarray(crop)))
    return items


def _meta_rows(keep_rows: pd.DataFrame) -> pd.DataFrame:
    """The legacy meta columns of kept manifest rows (`bbox` as `[x, y, w, h]`)."""
    boxes = keep_rows[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "image_name": keep_rows["image_name"].to_numpy(),
            "coral_id": keep_rows["coral_id"].to_numpy(),
            "bbox": [list(box) for box in boxes],
            "area": keep_rows["area"].to_numpy(),
        }
    )


def run_zero_shot(
    encoder: Encoder,
    text_features: np.ndarray,
    genera: Sequence[str],
    manifest: pd.DataFrame,
    json_dir: str | os.PathLike[str],
    images_dir: str | os.PathLike[str],
    *,
    crops: CropConfig | None = None,
    batch_size: int = 64,
    logit_scale: float = 100.0,
    run: ResumableRun | None = None,
    keep_embeddings: bool = False,
    group_size: int = 32,
    max_consecutive_failures: int = 5,
    progress: bool = False,
) -> ZeroShotResult:
    """Classify the kept masks of an image manifest, image by image.

    For each image: load it in the mask frame, cut the masked crop of every `keep` row,
    preprocess, and encode in batches of `batch_size` (images are processed in groups of
    `group_size`, so a batch never waits for the next group). With a `ResumableRun`, done
    images are skipped, each image is recorded as it completes and errors go to the run's
    failure log (the image is retried on the next call). Without one, results stay in memory.
    After `max_consecutive_failures` failures in a row the run stops with a `RuntimeError`.

    `keep_embeddings` also returns the L2-normalized image features.
    """
    crops = crops or CropConfig()
    if crops.mode != "masked":
        raise NotImplementedError(
            "run_zero_shot supports masked crops only; bbox crops arrive with the "
            "sharpness_crop option in Stage B"
        )
    if "keep" not in manifest.columns:
        raise ValueError("run_zero_shot needs an image manifest (with a 'keep' column)")
    genera = list(genera)
    if text_features.shape[0] != len(genera):
        raise ValueError(f"{text_features.shape[0]} text features for {len(genera)} genera")
    json_dir, images_dir = Path(json_dir), Path(images_dir)
    sink: Any = run if run is not None else _MemorySink()

    kept = manifest[manifest["keep"].astype(bool)]
    by_image = {name: rows for name, rows in kept.groupby("image_name", sort=False)}
    todo = sink.pending(list(by_image))
    groups = [todo[i : i + group_size] for i in range(0, len(todo), group_size)]
    iterator: Any = groups
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(groups, desc="Zero-shot groups")

    n_images = 0
    consecutive = 0
    try:
        for group in iterator:
            prepared: list[tuple[str, pd.DataFrame, list[Any]]] = []
            for name in group:
                try:
                    items = _image_items(encoder, name, by_image[name], json_dir, images_dir, crops)
                except Exception as err:  # logged per image, retried on resume
                    sink.fail(name, err)
                    consecutive += 1
                    _check_failures(consecutive, max_consecutive_failures)
                    continue
                prepared.append((name, by_image[name], items))
            if not prepared:
                continue
            items = [item for _, _, image_items in prepared for item in image_items]
            try:
                feats = [
                    encoder.encode_images(items[i : i + batch_size])
                    for i in range(0, len(items), batch_size)
                ]
                X = l2_normalize(np.vstack(feats))
                probs = score(X, text_features, logit_scale)
            except Exception as err:  # the whole group is retried on resume
                for name, _, _ in prepared:
                    sink.fail(name, err)
                consecutive += len(prepared)
                _check_failures(consecutive, max_consecutive_failures)
                continue
            consecutive = 0
            start = 0
            for name, keep_rows, image_items in prepared:
                stop = start + len(image_items)
                table = predictions_table(_meta_rows(keep_rows), probs[start:stop], genera)
                sink.add(name, table, X[start:stop] if keep_embeddings else None)
                start = stop
                n_images += 1
    finally:
        sink.flush()
    rows, matrix = sink.merge()
    if rows.empty:
        rows = predictions_table(
            pd.DataFrame(columns=META_COLUMNS), np.zeros((0, len(genera))), genera
        )
    return ZeroShotResult(rows, matrix, n_images, list(sink.unresolved_failures()))


def _check_failures(consecutive: int, limit: int) -> None:
    if consecutive >= limit:
        raise RuntimeError(
            f"Stopping after {consecutive} consecutive failures; see the failure log. "
            "Finished images are saved; fix the cause and run again to resume."
        )
