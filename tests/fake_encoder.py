"""A deterministic stand-in for a CLIP-family model, so tests need no torch and no weights."""

from __future__ import annotations

import zlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


class FakeEncoder:
    """Implements the `coralid.encoders.Encoder` protocol with numpy only.

    - `preprocess` returns the mean RGB of the crop (a 3-vector in [0, 1]), so features depend
      on the pixels and a crop that is blank gives the zero vector.
    - `encode_images` projects those with a fixed random matrix.
    - `encode_texts` gives every distinct text its own reproducible vector (seeded by a CRC),
      unless `text_vectors` pins it.

    The call logs (`image_batch_sizes`, `text_calls`) let tests check batching and prompts.
    `fail_when` can make `encode_images` raise, to test failure handling.
    """

    def __init__(
        self,
        dim: int = 8,
        family: str = "open_clip",
        text_vectors: dict[str, Sequence[float]] | None = None,
        fail_when: Callable[[Sequence[Any]], bool] | None = None,
    ) -> None:
        self.name = f"fake-{family}"
        self.family = family
        self.dim = dim
        self.text_vectors = {
            k: np.asarray(v, dtype=np.float32) for k, v in (text_vectors or {}).items()
        }
        self.fail_when = fail_when
        self._projection = np.random.RandomState(0).randn(3, dim).astype(np.float32)
        self.image_batch_sizes: list[int] = []
        self.text_calls: list[list[str]] = []

    def preprocess(self, image: Any) -> np.ndarray:
        """Mean RGB of the PIL image, scaled to [0, 1]."""
        return (np.asarray(image, dtype=np.float32).mean(axis=(0, 1)) / 255.0).astype(np.float32)

    def encode_images(self, batch: Any) -> np.ndarray:
        """Project a batch (list or stacked array) of preprocessed crops."""
        items = list(batch)
        if self.fail_when is not None and self.fail_when(items):
            raise RuntimeError("fake CUDA out of memory")
        self.image_batch_sizes.append(len(items))
        return (np.stack(items) @ self._projection).astype(np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """One reproducible vector per distinct text."""
        texts = list(texts)
        self.text_calls.append(texts)
        rows = []
        for text in texts:
            if text in self.text_vectors:
                rows.append(self.text_vectors[text])
            else:
                rows.append(np.random.RandomState(zlib.crc32(text.encode())).randn(self.dim))
        return np.asarray(rows, dtype=np.float32)
