"""The per-run coral table: one row per mask, with the filter decisions and their reasons.

Two flavours:
- `geometric_manifest(json_dir)` needs no photos and is fast. It applies the area, aspect
  and containment filters and records why each mask was removed.
- `image_manifest(json_dir, images_dir)` adds what needs the photo: visible pixels,
  sharpness and the final `keep` decision (`reject_reason` says why a mask was dropped).

Every downstream step should read `keep == True` from the manifest instead of re-filtering.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from coralid._atomic import atomic_path
from coralid.config import Config, CropConfig, FilterConfig, write_meta
from coralid.crops import masked_crop_stats
from coralid.filters import aspect_ratio, filter_annotations_with_reasons, laplacian_variance
from coralid.images import ImageReadError, load_in_mask_frame
from coralid.masks import (
    CoralKey,
    MaskFile,
    list_mask_files,
    load_mask_json,
    parse_bbox,
)

REASON_IMAGE_MISSING = "image_missing"
REASON_NOT_VISIBLE = "not_visible"
REASON_BLURRY = "blurry"

GEOMETRIC_COLUMNS = [
    "image_name",
    "coral_id",
    "uid",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "area",
    "predicted_iou",
    "stability_score",
    "aspect",
    "frame_w",
    "frame_h",
    "stage1_reason",
    "contained_by_smaller",
]
IMAGE_COLUMNS = [*GEOMETRIC_COLUMNS, "visible_px", "sharpness", "keep", "reject_reason"]


def geometric_rows(mask_file: MaskFile, filters: FilterConfig) -> list[dict[str, Any]]:
    """Manifest rows of one image from its masks alone (no photo needed)."""
    _, reasons = filter_annotations_with_reasons(
        mask_file.annotations,
        filters.min_area,
        filters.containment_thresh,
        filters.max_aspect_ratio,
    )
    rows = []
    for a in mask_file.annotations:
        x, y, w, h = parse_bbox(a["bbox"])
        rejection = reasons[a["id"]]
        rows.append(
            {
                "image_name": mask_file.image_name,
                "coral_id": int(a["id"]),
                "uid": CoralKey(mask_file.image_name, a["id"]).uid,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "area": a["area"],
                "predicted_iou": a.get("predicted_iou"),
                "stability_score": a.get("stability_score"),
                "aspect": aspect_ratio((x, y, w, h)),
                "frame_w": mask_file.width,
                "frame_h": mask_file.height,
                "stage1_reason": rejection.reason if rejection else None,
                "contained_by_smaller": rejection.contained_by if rejection else None,
            }
        )
    return rows


def image_rows(
    mask_file: MaskFile,
    images_dir: str | os.PathLike[str],
    filters: FilterConfig,
    crops: CropConfig,
) -> list[dict[str, Any]]:
    """Manifest rows of one image, including visibility, sharpness and the keep decision.

    A mask removed by the geometric filters keeps that reason. The others are judged on the
    masked crop (the pipeline's own: black background, Laplacian variance at least the
    threshold). If the photo is missing or unreadable they get `image_missing`.
    """
    rows = geometric_rows(mask_file, filters)
    for row in rows:
        row.update(visible_px=None, sharpness=None, keep=False, reject_reason=row["stage1_reason"])
    survivors = [r for r in rows if r["stage1_reason"] is None]
    if not survivors:
        return rows
    try:
        img = load_in_mask_frame(Path(images_dir) / mask_file.image_name, *mask_file.frame)
    except ImageReadError:
        for row in survivors:
            row["reject_reason"] = REASON_IMAGE_MISSING
        return rows
    by_id = mask_file.by_id()
    for row in survivors:
        anno = by_id[row["coral_id"]]
        crop, visible = masked_crop_stats(
            img,
            anno["bbox"],
            anno["segmentation"],
            padding_pct=crops.padding_pct,
            background=crops.background,
        )
        row["visible_px"] = visible
        if crop is None or visible < crops.min_visible:
            row["reject_reason"] = REASON_NOT_VISIBLE
            continue
        row["sharpness"] = laplacian_variance(crop)
        if row["sharpness"] < filters.sharpness_threshold:
            row["reject_reason"] = REASON_BLURRY
        else:
            row["keep"] = True
            row["reject_reason"] = None
    return rows


def _to_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    """Build the manifest DataFrame with stable column order and dtypes."""
    df = pd.DataFrame(rows, columns=columns)
    for col in ("coral_id", "frame_w", "frame_h"):
        df[col] = df[col].astype("int64")
    for col in ("bbox_x", "bbox_y", "bbox_w", "bbox_h", "aspect"):
        df[col] = df[col].astype("float64")
    for col in ("predicted_iou", "stability_score"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df["area"] = pd.to_numeric(df["area"])
    df["contained_by_smaller"] = pd.array(df["contained_by_smaller"].tolist(), dtype="Int64")
    if "keep" in df:
        df["visible_px"] = pd.array(df["visible_px"].tolist(), dtype="Int64")
        df["sharpness"] = pd.to_numeric(df["sharpness"], errors="coerce").astype("float64")
        df["keep"] = df["keep"].astype(bool)
    return df


def _mask_files(json_dir: str | os.PathLike[str], image_names: Sequence[str] | None) -> list[Path]:
    if image_names is None:
        return list_mask_files(json_dir)
    from coralid.masks import json_name_for

    paths = [Path(json_dir) / json_name_for(name) for name in image_names]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"No mask JSON for {len(missing)} image(s), e.g. {missing[:3]}")
    return paths


def _progress(items: Iterable[Path], enabled: bool, desc: str) -> Iterable[Path]:
    if not enabled:
        return items
    from tqdm.auto import tqdm

    return tqdm(list(items), desc=desc)


def geometric_manifest(
    json_dir: str | os.PathLike[str],
    filters: FilterConfig | None = None,
    image_names: Sequence[str] | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """One row per mask with geometry and the area / aspect / containment decisions.

    Needs no photos. `stage1_reason` is `area`, `aspect`, `containment` or None, and
    `contained_by_smaller` is the id of the smaller mask that removed a `containment` mask.
    """
    filters = filters or FilterConfig()
    rows: list[dict[str, Any]] = []
    for path in _progress(_mask_files(json_dir, image_names), progress, "Geometric manifest"):
        rows.extend(geometric_rows(load_mask_json(path), filters))
    return _to_frame(rows, GEOMETRIC_COLUMNS)


def image_manifest(
    json_dir: str | os.PathLike[str],
    images_dir: str | os.PathLike[str],
    filters: FilterConfig | None = None,
    crops: CropConfig | None = None,
    image_names: Sequence[str] | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """The geometric manifest plus `visible_px`, `sharpness`, `keep` and `reject_reason`.

    `reject_reason` is `area`, `aspect`, `containment`, `image_missing` (photo absent or
    unreadable), `not_visible`, `blurry`, or None for kept masks.
    """
    filters = filters or FilterConfig()
    crops = crops or CropConfig()
    rows: list[dict[str, Any]] = []
    for path in _progress(_mask_files(json_dir, image_names), progress, "Image manifest"):
        rows.extend(image_rows(load_mask_json(path), images_dir, filters, crops))
    return _to_frame(rows, IMAGE_COLUMNS)


def write_manifest(
    df: pd.DataFrame,
    path: str | os.PathLike[str],
    config: Config | None = None,
    **info: Any,
) -> Path:
    """Write the manifest as parquet (atomically) with its `*.meta.json` sidecar."""
    kind = "image" if "keep" in df.columns else "geometric"
    with atomic_path(path) as tmp:
        df.to_parquet(tmp, index=False)
    extra: dict[str, Any] = {"kind": kind, "n_images": int(df["image_name"].nunique())}
    if kind == "image":
        extra["n_keep"] = int(df["keep"].sum())
    extra.update(info)
    write_meta(path, config=config, n_rows=len(df), **extra)
    return Path(path)


def read_manifest(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a manifest written by `write_manifest`."""
    return pd.read_parquet(path)
