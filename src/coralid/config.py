"""Run configuration: dataclasses, YAML load/save, run name and `*.meta.json` sidecars.

Nothing here knows about Colab. Paths are relative to a data root that comes from the
config or from the `CORALID_DATA_ROOT` environment variable.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from coralid import __version__
from coralid._atomic import atomic_write_text

DATA_ROOT_ENV = "CORALID_DATA_ROOT"
BACKGROUNDS = ("black", "gray", "white")
CROP_MODES = ("masked", "bbox")
SHARPNESS_CROPS = ("masked", "bbox")


class ConfigError(ValueError):
    """Raised for an invalid or incomplete configuration."""


@dataclass
class Paths:
    """Where things live, relative to the data root (absolute entries are kept as they are)."""

    data_root: str | None = None  # None: take it from $CORALID_DATA_ROOT
    images_dir: str = "data_all"
    results_dir: str = "results_all"
    json_dir: str = "json"
    manifest_dir: str = "manifest"
    features_dir: str = "features"
    predictions_dir: str = "predictions"
    reviews_dir: str = "reviews"
    reports_dir: str = "reports"
    ground_truth_csv: str = "ground_truth_genus_counts.csv"

    def resolve_root(self) -> Path:
        """Return the data root from the config, else from `$CORALID_DATA_ROOT`."""
        root = self.data_root or os.environ.get(DATA_ROOT_ENV)
        if not root:
            raise ConfigError(
                f"No data root: set paths.data_root in the config or the {DATA_ROOT_ENV} "
                "environment variable."
            )
        return Path(root)


@dataclass
class SamConfig:
    """CoralSCOP / SAM automatic mask generator settings (CM cells 7-8)."""

    model_type: str = "vit_b"
    checkpoint: str = "CoralSCOP/checkpoints/vit_b_coralscop.pth"  # relative to the data root
    points_per_side: int = 32
    pred_iou_thresh: float = 0.60
    stability_score_thresh: float = 0.55
    crop_n_layers: int = 1
    crop_n_points_downscale_factor: int = 2
    min_mask_region_area: int = 100
    max_side: int = 2560  # photos larger than this are downscaled before segmentation


@dataclass
class FilterConfig:
    """Mask quality filters (WC cell 8, V2 cell 4).

    Sharpness is the Laplacian variance of a crop, and *which* crop is a choice:
    - `masked` (the pipeline's definition, the default): the masked crop on a black background,
      kept if it reaches `sharpness_threshold` (80).
    - `bbox` (the Millepora audit's definition, MA cells 5 and 9): the unmasked box of the mask,
      without padding, kept if it reaches `sharpness_threshold_bbox` (150).
    The default is unchanged; switching it is an algorithm decision, not a refactoring one.
    """

    min_area: int = 25000
    containment_thresh: float = 0.80
    max_aspect_ratio: float = 3.0
    sharpness_threshold: float = 80.0
    sharpness_crop: str = "masked"
    sharpness_threshold_bbox: float = 150.0

    def __post_init__(self) -> None:
        if self.sharpness_crop not in SHARPNESS_CROPS:
            raise ConfigError(
                f"filters.sharpness_crop must be one of {SHARPNESS_CROPS}, "
                f"got {self.sharpness_crop!r}"
            )

    @property
    def sharpness_cutoff(self) -> float:
        """The threshold that goes with `sharpness_crop`."""
        if self.sharpness_crop == "bbox":
            return self.sharpness_threshold_bbox
        return self.sharpness_threshold


@dataclass
class CropConfig:
    """How a mask is cut out of the photo."""

    mode: str = "masked"  # "masked" (mask on a plain background) or "bbox"
    background: str = "black"
    padding_pct: float = 0.05  # masked crops
    bbox_padding_pct: float = 0.15  # bbox crops
    min_visible: int = 500  # minimum non-background pixels in a masked crop

    def __post_init__(self) -> None:
        if self.mode not in CROP_MODES:
            raise ConfigError(f"crops.mode must be one of {CROP_MODES}, got {self.mode!r}")
        if self.background not in BACKGROUNDS:
            raise ConfigError(
                f"crops.background must be one of {BACKGROUNDS}, got {self.background!r}"
            )


@dataclass
class ZeroShotConfig:
    """Zero-shot classification settings (WC cell 14)."""

    model: str = "finetuned-bioclip"
    label_set: str = "hard20"
    prompt_format: str = "taxonomy"
    prompt_set: str = "taxonomy_v1"
    batch_size: int = 64
    logit_scale: float = 100.0


_SECTIONS: dict[str, type] = {
    "paths": Paths,
    "sam": SamConfig,
    "filters": FilterConfig,
    "crops": CropConfig,
    "zero_shot": ZeroShotConfig,
}


def run_name(sam: SamConfig | None = None) -> str:
    """Return the run folder name, e.g. `iou60_sta55_pts32` for the default settings.

    Uses `round()`: `int(0.57 * 100)` is 56, which would name the run after the wrong value.
    """
    sam = sam or SamConfig()
    iou = round(sam.pred_iou_thresh * 100)
    sta = round(sam.stability_score_thresh * 100)
    return f"iou{iou}_sta{sta}_pts{int(sam.points_per_side)}"


@dataclass
class Config:
    """The full run configuration."""

    paths: Paths = field(default_factory=Paths)
    sam: SamConfig = field(default_factory=SamConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    crops: CropConfig = field(default_factory=CropConfig)
    zero_shot: ZeroShotConfig = field(default_factory=ZeroShotConfig)

    @property
    def run_name(self) -> str:
        """Run folder name derived from the SAM settings."""
        return run_name(self.sam)

    @property
    def data_root(self) -> Path:
        """The resolved data root."""
        return self.paths.resolve_root()

    @property
    def images_dir(self) -> Path:
        """Folder with the source photos."""
        return self._under_root(self.paths.images_dir)

    @property
    def run_dir(self) -> Path:
        """Folder of this segmentation run (results_all/<run name>)."""
        return self._under_root(self.paths.results_dir) / self.run_name

    @property
    def json_dir(self) -> Path:
        """Folder with the mask JSONs of this run."""
        return self.run_dir / self.paths.json_dir

    @property
    def manifest_path(self) -> Path:
        """Where the run's manifest.parquet goes."""
        return self.run_dir / self.paths.manifest_dir / "manifest.parquet"

    @property
    def features_dir(self) -> Path:
        """Folder for embedding files of this run."""
        return self.run_dir / self.paths.features_dir

    @property
    def predictions_dir(self) -> Path:
        """Folder for prediction tables of this run."""
        return self.run_dir / self.paths.predictions_dir

    @property
    def reviews_dir(self) -> Path:
        """Folder for review sheets of this run."""
        return self.run_dir / self.paths.reviews_dir

    @property
    def reports_dir(self) -> Path:
        """Folder for evaluation reports of this run."""
        return self.run_dir / self.paths.reports_dir

    @property
    def ground_truth_path(self) -> Path:
        """The survey genus counts file."""
        return self._under_root(self.paths.ground_truth_csv)

    @property
    def checkpoint_path(self) -> Path:
        """The CoralSCOP checkpoint."""
        return self._under_root(self.sam.checkpoint)

    def _under_root(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else self.data_root / path

    def to_dict(self) -> dict[str, Any]:
        """Return the config as plain nested dicts (what the YAML and the meta file hold)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Build a config from nested dicts, rejecting unknown sections and keys."""
        unknown = set(data) - set(_SECTIONS)
        if unknown:
            raise ConfigError(f"Unknown config sections: {sorted(unknown)}")
        sections = {}
        for name, section_cls in _SECTIONS.items():
            values = data.get(name) or {}
            valid = {f.name for f in dataclasses.fields(section_cls)}
            bad = set(values) - valid
            if bad:
                raise ConfigError(f"Unknown keys in '{name}': {sorted(bad)}")
            sections[name] = section_cls(**values)
        return cls(**sections)


def load_config(path: str | os.PathLike[str]) -> Config:
    """Load a config from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return Config.from_dict(data)


def save_config(config: Config, path: str | os.PathLike[str]) -> Path:
    """Write a config to a YAML file."""
    text = yaml.safe_dump(config.to_dict(), sort_keys=False, default_flow_style=False)
    return atomic_write_text(path, text)


def meta_path(path: str | os.PathLike[str]) -> Path:
    """Return the `*.meta.json` sidecar path for a data file (`x.parquet` -> `x.meta.json`)."""
    p = Path(path)
    if p.name.endswith(".meta.json"):
        return p
    return p.with_suffix(".meta.json")


def git_commit(cwd: str | os.PathLike[str] | None = None) -> str | None:
    """Return the current git commit hash, or None if git or a repository is unavailable."""
    cwd = Path(cwd) if cwd is not None else Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _json_default(obj: Any) -> Any:
    """Make numpy scalars, arrays and paths JSON serializable."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def write_meta(
    path: str | os.PathLike[str],
    *,
    config: Config | None = None,
    n_rows: int | None = None,
    **info: Any,
) -> Path:
    """Write the `*.meta.json` sidecar for an output file and return its path.

    Records the resolved config, git commit (if available), package version, UTC timestamp
    and row count, plus any extra keyword arguments (label set, prompt set, sample, ...).
    """
    meta: dict[str, Any] = {
        "package_version": __version__,
        "git_commit": git_commit(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_rows": n_rows,
        "config": config.to_dict() if config is not None else None,
    }
    meta.update(info)
    out = meta_path(path)
    return atomic_write_text(out, json.dumps(meta, indent=2, default=_json_default) + "\n")
