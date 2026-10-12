"""CoralSCOP segmentation: photos in, one mask JSON per photo out (CM cells 6-8).

`run_segmentation` is the job behind `python -m coralid segment`. It is safe to point at a run
folder that is already full (the active run's `json/` holds 1,714 files):

- **An image whose JSON exists is skipped**, whatever the file holds, and no JSON is ever
  rewritten. A new JSON is created with `write_mask_json(..., overwrite=False)`: atomic and
  exclusive, so even a second process working on the same folder cannot be overwritten. With
  nothing to do the job loads no model and creates no folder.
- **An error never becomes a JSON.** The legacy cell wrote an empty JSON for any error, CUDA
  out-of-memory included, and its smart-skip then treated the image as done for good. Here a photo
  that cannot be read, or a generator that raises, goes to the failure log of a `ResumableRun`
  (`<run>/segmentation/segment.failures.jsonl`); no JSON is written, so the next call retries it.
- An image that really has no masks gets an empty JSON with `"note": "no masks"`.

The JSON layout, the frame and the generator settings are the legacy ones: the photo is scaled so
its long side is at most `sam.max_side` (2560), segmented as RGB in that frame, and each mask is
stored as `segmentation` (RLE with str counts), `bbox`, `area`, `predicted_iou`, `crop_box`,
`stability_score`, `point_coords` and `id` (0..N-1 in generator order). CoralSCOP's extra outputs
(`cate_preds`, `fc_features`, `similarity`) are never put in the JSON; with `sam.save_features`
the first two go to `<run>/segmentation/extras/<stem>.npz`.

CoralSCOP is not stock SAM: its `segment_anything` has a coral/background head. It is imported
from the `third_party/CoralSCOP` submodule, in `load_coralscop` and nowhere else, and that
function refuses to go on if the module came from anywhere else (pip's `segment-anything`, say).
The checkpoint is loaded strictly: there is no `strict=False` fallback.
"""

from __future__ import annotations

import gc
import importlib
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from coralid._atomic import atomic_path
from coralid.batch import ResumableRun
from coralid.config import Config, SamConfig, write_meta
from coralid.images import load_for_segmentation
from coralid.masks import encode_mask, json_name_for, write_mask_json

logger = logging.getLogger(__name__)

CORALSCOP_ENV = "CORALID_CORALSCOP_DIR"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")  # CM cell 8
NOTE_NO_MASKS = "no masks"
RUN_NAME = "segment"


class SegmentationError(RuntimeError):
    """CoralSCOP cannot be loaded as required, or the job cannot go on."""


class MaskGenerator(Protocol):
    """What the job needs from the SAM generator: `generate(rgb)` gives mask records."""

    def generate(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Return SAM's mask records for an RGB uint8 image (binary-mask output)."""


# --------------------------------------------------------------------------- CoralSCOP


def coralscop_dir() -> Path:
    """The CoralSCOP folder: `$CORALID_CORALSCOP_DIR`, else `third_party/CoralSCOP` of the repo."""
    override = os.environ.get(CORALSCOP_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "third_party" / "CoralSCOP"


def load_coralscop(root: str | os.PathLike[str] | None = None) -> Any:
    """Import CoralSCOP's own `segment_anything` and check it is the one that was imported.

    This is the only place in the package that puts the submodule on `sys.path`.
    """
    folder = Path(root).resolve() if root else coralscop_dir().resolve()
    if not (folder / "segment_anything" / "__init__.py").is_file():
        raise SegmentationError(
            f"CoralSCOP not found in {folder}. If it is the submodule, run "
            "`git submodule update --init` in the repository."
        )
    if sys.path[:1] != [str(folder)]:
        sys.path.insert(0, str(folder))  # before any pip-installed segment_anything
    try:
        module = importlib.import_module("segment_anything")
    except ImportError as err:
        raise SegmentationError(
            f"CoralSCOP's segment_anything could not be imported ({err}). It needs torch and "
            "torchvision: install the 'segment' extra (pip install -e '.[segment]')."
        ) from err
    origin = Path(module.__file__).resolve()
    if folder not in origin.parents:
        raise SegmentationError(
            f"segment_anything was imported from {origin}, not from {folder}. pip's "
            "`segment-anything` has no coral/background head. Uninstall it "
            "(`pip uninstall segment-anything`), restart the runtime and try again."
        )
    return module


def build_generator(
    sam: SamConfig,
    checkpoint: str | os.PathLike[str],
    *,
    device: str | None = None,
    coralscop_root: str | os.PathLike[str] | None = None,
) -> MaskGenerator:
    """Load the CoralSCOP checkpoint and build the automatic mask generator of CM cell 7.

    The checkpoint is loaded with `strict=True` by CoralSCOP's own builder; a mismatch raises.
    A missing file is an error too (the builder would silently return an untrained model).
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CoralSCOP checkpoint not found: {checkpoint}")
    segment_anything = load_coralscop(coralscop_root)
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        builder = segment_anything.sam_model_registry[sam.model_type]
    except KeyError:
        known = sorted(segment_anything.sam_model_registry)
        raise SegmentationError(f"Unknown sam.model_type {sam.model_type!r}; use {known}") from None
    logger.info("Loading CoralSCOP %s from %s on %s", sam.model_type, checkpoint, device)
    model = builder(checkpoint=str(checkpoint))
    if not hasattr(model.mask_decoder, "cate_prediction_head"):
        raise SegmentationError("This segment_anything model has no coral/background head")
    model.to(device=device)
    model.eval()
    return segment_anything.SamAutomaticMaskGenerator(
        model=model,
        points_per_side=sam.points_per_side,
        pred_iou_thresh=sam.pred_iou_thresh,
        stability_score_thresh=sam.stability_score_thresh,
        crop_n_layers=sam.crop_n_layers,
        crop_n_points_downscale_factor=sam.crop_n_points_downscale_factor,
        min_mask_region_area=sam.min_mask_region_area,
    )


# --------------------------------------------------------------------------- one image


def annotations_from_masks(masks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """SAM's mask records as the JSON annotations of CM cell 8 (same keys, same order)."""
    annotations = []
    for anno_id, m in enumerate(masks):
        points = m["point_coords"]
        annotations.append(
            {
                "segmentation": encode_mask(m["segmentation"]),
                "bbox": m["bbox"],
                "area": m["area"],
                "predicted_iou": float(m["predicted_iou"]),
                "crop_box": m["crop_box"],
                "stability_score": float(m["stability_score"]),
                "point_coords": points.tolist() if isinstance(points, np.ndarray) else points,
                "id": anno_id,
            }
        )
    return annotations


def extras_from_masks(masks: Sequence[dict[str, Any]]) -> dict[str, np.ndarray] | None:
    """CoralSCOP's `cate_preds` and `fc_features` per mask, row-aligned with the annotation ids.

    None if the generator did not return them (a generator that is not CoralSCOP's).
    """
    if not masks or not all("fc_features" in m and "cate_preds" in m for m in masks):
        return None
    return {
        "ids": np.arange(len(masks), dtype=np.int64),
        "cate_preds": np.array([m["cate_preds"] for m in masks], dtype=np.int64),
        "fc_features": np.array([m["fc_features"] for m in masks], dtype=np.float32),
    }


# --------------------------------------------------------------------------- the job


def list_images(images_dir: str | os.PathLike[str]) -> list[str]:
    """The photo names in a folder (the legacy extensions, any case), sorted; no hidden files."""
    root = Path(images_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"The photo folder {root} does not exist")
    names = (
        n
        for n in os.listdir(root)
        if not n.startswith(".") and n.lower().endswith(IMAGE_EXTENSIONS)
    )
    return sorted(n for n in names if (root / n).is_file())


@dataclass
class SegmentationPlan:
    """What a call would do: which photos need a JSON, which have one, which clash."""

    todo: list[str] = field(default_factory=list)
    done: list[str] = field(default_factory=list)  # a JSON exists: skipped, never rewritten
    collisions: dict[str, list[str]] = field(default_factory=dict)  # JSON name -> later photos


def plan_segmentation(
    image_names: Sequence[str], json_dir: str | os.PathLike[str]
) -> SegmentationPlan:
    """Split photos into those that still need a JSON and those that have one.

    Two photos with the same stem and different extensions would share one JSON name; the first
    one (in order) keeps it and the others are listed in `collisions` and left alone.
    """
    root = Path(json_dir)
    existing = set(os.listdir(root)) if root.is_dir() else set()
    plan, owner = SegmentationPlan(), {}
    for name in image_names:
        json_name = json_name_for(name)
        if json_name in owner:
            plan.collisions.setdefault(json_name, [owner[json_name]]).append(name)
            continue
        owner[json_name] = name
        (plan.done if json_name in existing else plan.todo).append(name)
    return plan


@dataclass
class SegmentationResult:
    """What `run_segmentation` did."""

    n_images: int = 0
    n_already_done: int = 0  # had a JSON: skipped
    written: list[str] = field(default_factory=list)  # JSONs created in this call
    n_empty: int = 0  # of those, images with no masks
    raced: list[str] = field(default_factory=list)  # a JSON appeared while we worked on it
    n_remaining: int = 0  # photos still without a JSON (failed, or beyond `limit`)
    failures: list[dict[str, Any]] = field(default_factory=list)  # unresolved
    collisions: dict[str, list[str]] = field(default_factory=dict)


def unresolved_failures(run: ResumableRun, json_dir: Path) -> list[dict[str, Any]]:
    """Logged failures whose image still has no JSON (the last record per image)."""
    latest = {r["image"]: r for r in run.failures()}
    return [r for name, r in latest.items() if not (json_dir / json_name_for(name)).exists()]


def _free_memory() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_extras(path: Path, extras: dict[str, np.ndarray], frame: tuple[int, int]) -> None:
    # An extras file without its JSON is an orphan of an earlier attempt, so it may be replaced.
    with atomic_path(path) as tmp:
        np.savez_compressed(tmp, frame=np.array(frame), **extras)


def run_segmentation(
    cfg: Config,
    *,
    generator: MaskGenerator | None = None,
    image_names: Sequence[str] | None = None,
    limit: int | None = None,
    device: str | None = None,
    progress: bool = False,
    max_consecutive_failures: int = 5,
) -> SegmentationResult:
    """Create the missing mask JSONs of a run; never touch an existing one. Resumable.

    `image_names` restricts the photos, `limit` caps how many are segmented in this call.
    `generator` is for tests; by default CoralSCOP is loaded from `cfg.checkpoint_path`.
    After `max_consecutive_failures` failures in a row the call stops with a `SegmentationError`.
    """
    images = list_images(cfg.images_dir)
    if image_names is not None:
        unknown = sorted(set(image_names) - set(images))
        if unknown:
            raise FileNotFoundError(f"Not in {cfg.images_dir}: {unknown[:5]}")
        images = [n for n in images if n in set(image_names)]
    plan = plan_segmentation(images, cfg.json_dir)
    for json_name, names in plan.collisions.items():
        logger.warning("%s would be shared by %s; only the first is segmented", json_name, names)
    todo = plan.todo[:limit] if limit is not None else plan.todo
    result = SegmentationResult(
        n_images=len(images), n_already_done=len(plan.done), collisions=plan.collisions
    )
    run = ResumableRun(cfg.segmentation_dir, RUN_NAME)
    if not todo:
        logger.info("Nothing to segment: %d photos, %d with a JSON", len(images), len(plan.done))
        result.failures = unresolved_failures(run, cfg.json_dir)
        return result

    generator = generator or build_generator(cfg.sam, cfg.checkpoint_path, device=device)
    cfg.json_dir.mkdir(parents=True, exist_ok=True)
    iterator: Any = todo
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(todo, desc="Segmenting")
    extras_dir = cfg.segmentation_dir / "extras"
    consecutive = 0
    for name in iterator:
        try:
            rgb, frame = load_for_segmentation(cfg.images_dir / name, cfg.sam.max_side)
            masks = generator.generate(rgb)
            annotations = annotations_from_masks(masks)
            extras = extras_from_masks(masks) if cfg.sam.save_features else None
            if cfg.sam.save_features and annotations and extras is None:
                logger.warning("%s: the generator returned no cate_preds / fc_features", name)
            if extras is not None:
                _write_extras(extras_dir / f"{Path(json_name_for(name)).stem}.npz", extras, frame)
        except Exception as err:  # logged per image and retried on the next call
            run.fail(name, err)
            _free_memory()
            consecutive += 1
            if consecutive >= max_consecutive_failures:
                raise SegmentationError(
                    f"Stopping after {consecutive} failures in a row; see {run.failures_path}. "
                    "Finished images are saved; fix the cause and run again to resume."
                ) from err
            continue
        consecutive = 0
        try:
            write_mask_json(
                cfg.json_dir / json_name_for(name),
                name,
                frame[0],
                frame[1],
                annotations,
                note=None if annotations else NOTE_NO_MASKS,
                overwrite=False,
            )
        except FileExistsError:
            result.raced.append(name)
            logger.warning("%s: a JSON appeared meanwhile; left as it is", name)
        else:
            result.written.append(name)
            result.n_empty += 0 if annotations else 1
        _free_memory()
    result.failures = unresolved_failures(run, cfg.json_dir)
    result.n_remaining = len(plan.todo) - len(result.written) - len(result.raced)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_meta(
        cfg.segmentation_dir / f"segment_{stamp}.meta.json",
        config=cfg,
        n_rows=len(result.written),
        kind="segmentation_run",
        n_attempted=len(todo),
        n_empty=result.n_empty,
        n_failed=len(result.failures),
        save_features=cfg.sam.save_features,
        checkpoint=str(cfg.checkpoint_path),
    )
    return result
