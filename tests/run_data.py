"""A small synthetic data root in the real layout, for the job and command tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from coralid.config import Config, CropConfig, FilterConfig, Paths
from coralid.manifest import image_manifest, write_manifest
from tests.helpers import make_annotation, rect_mask, write_json_dir

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
