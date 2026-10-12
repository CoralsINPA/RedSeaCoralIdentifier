"""A small synthetic data root in the real layout, for the job and command tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from coralid.config import Config, CropConfig, FilterConfig, Paths
from coralid.genera import load_genera
from coralid.manifest import image_manifest, write_manifest
from tests.helpers import make_annotation, rect_mask, write_json_dir

HARD20 = load_genera().genera("hard20")
FRAME_W, FRAME_H = 200, 150
SHAPE = (FRAME_H, FRAME_W)
LAYOUT = {
    "IMG_1.JPG": [(10, 50, 10, 60), (60, 100, 10, 80)],
    "IMG_2.JPG": [(20, 70, 20, 90)],
    "IMG_3.JPG": [(10, 40, 10, 50), (50, 90, 20, 90), (100, 140, 20, 60)],
}


def photo(seed: int) -> np.ndarray:
    """A random RGB photo in the frame."""
    return np.random.RandomState(seed).randint(0, 256, size=(FRAME_H, FRAME_W, 3), dtype=np.uint8)


def make_config(root: str | Path) -> Config:
    """A config whose filters suit the tiny frame (every rectangle of LAYOUT is kept)."""
    return Config(
        paths=Paths(data_root=str(root)),
        filters=FilterConfig(min_area=100),
        crops=CropConfig(padding_pct=0.0, min_visible=50),
    )


def make_run(root: str | Path, manifest: bool = True) -> Config:
    """Photos, mask JSONs and (by default) the saved manifest of three images, 6 kept masks."""
    cfg = make_config(root)
    cfg.images_dir.mkdir(parents=True)
    entries = []
    for seed, (name, boxes) in enumerate(LAYOUT.items()):
        annos = [make_annotation(rect_mask(SHAPE, *b), i) for i, b in enumerate(boxes)]
        entries.append((name, FRAME_W, FRAME_H, annos, None))
        assert cv2.imwrite(str(cfg.images_dir / name), cv2.cvtColor(photo(seed), cv2.COLOR_RGB2BGR))
    write_json_dir(cfg.json_dir, entries)
    if manifest:
        table = image_manifest(cfg.json_dir, cfg.images_dir, cfg.filters, cfg.crops)
        assert table["keep"].sum() == 6
        write_manifest(table, cfg.manifest_path, config=cfg)
    return cfg


def write_expert_inputs(cfg: Config, predictions: pd.DataFrame) -> None:
    """An expert CSV in the real layout (with its own stored predictions) and survey counts.

    Of the 6 crops: 3 the expert agrees with, 1 wrong, 1 a soft coral, 1 unidentifiable.
    """
    columns = ["image_name", "coral_id", *HARD20, "predicted_genus", "confidence"]
    legacy = predictions[columns].copy()
    wrong = "Pavona" if predictions["predicted_genus"].iloc[3] != "Pavona" else "Porites"
    legacy["expert_genus"] = [*predictions["predicted_genus"].iloc[:3], wrong, "Sinularia", None]
    legacy["is_correct"] = [True, True, True, False, False, "unidentifiable"]
    legacy.to_csv(cfg.run_dir / "assaf_expert_test_set.csv", index=False)
    genera = [*HARD20, "Sinularia"]
    pd.DataFrame(
        {"Category": "x", "genus": genera, "count": np.arange(len(genera)) + 5, "Pi": 0.0}
    ).to_csv(cfg.ground_truth_path, index=False)


def labels_for(cfg: Config, genera: list[str]) -> pd.DataFrame:
    """A labels table for the kept crops of the run, cycling through `genera`."""
    kept = pd.read_parquet(cfg.manifest_path)
    kept = kept[kept["keep"]].reset_index(drop=True)
    return pd.DataFrame(
        {
            "image_name": kept["image_name"],
            "coral_id": kept["coral_id"],
            "genus": [genera[i % len(genera)] for i in range(len(kept))],
        }
    )
