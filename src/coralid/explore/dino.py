"""The DINO image encoders of the exploration direction (not part of the pipeline).

DINOv2 ViT-S/14 (CM cell 9) and DinoVdeau-large (CM cell 11) are image-only and run as the legacy
cells did, in fp16 autocast on a GPU; their features come back raw (not normalized). torch,
transformers and torchvision are imported inside the classes, so importing this module stays
light.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from coralid.encoders import _as_batch, _to_numpy, pick_device

logger = logging.getLogger(__name__)


@runtime_checkable
class ImageEncoder(Protocol):
    """What the embedding job needs from a model: no text side."""

    name: str

    def preprocess(self, image: Any) -> Any:
        """Turn one PIL image into a model input (a tensor)."""

    def encode_images(self, batch: Any) -> np.ndarray:
        """Encode a batch (a stacked tensor, or a list of `preprocess` outputs) to (n, d)."""


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# name -> where the model comes from, the input size and the batch size of the legacy cell.
# Kept apart from MODEL_IDS: these are not CLIP-family models and have no text encoder.
DINO_MODELS: dict[str, dict[str, Any]] = {
    "dinov2-s14": {
        "source": "torch.hub facebookresearch/dinov2 dinov2_vits14",
        "input_size": 224,
        "batch_size": 64,
    },
    "dinovdeau-l": {
        "source": (
            "lombardata/DinoVdeau-large-2024_04_03-with_data_aug_batch-size32_epochs150_freeze"
        ),
        "input_size": 518,
        "batch_size": 16,
    },
}


class _DinoEncoder:
    """Shared by the DINO encoders: ImageNet preprocessing and fp16 autocast on a GPU."""

    family = "dino"
    crop = "masked_square"  # the crop the legacy cells fed these models (crops.masked_square_crop)

    def __init__(self, name: str, model: Any, device: str | None) -> None:
        self.name = name
        self.device = device or pick_device()
        self.input_size: int = DINO_MODELS[name]["input_size"]
        self.batch_size: int = DINO_MODELS[name]["batch_size"]
        self.model = model.to(self.device).eval()
        self._transform: Any = None

    def preprocess(self, image: Any) -> Any:
        """Resize a PIL crop to the model's size, then ImageNet normalization."""
        if self._transform is None:
            from torchvision import transforms

            size = (self.input_size, self.input_size)
            self._transform = transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
                ]
            )
        return self._transform(image)

    def _forward(self, batch: Any) -> Any:
        raise NotImplementedError

    def encode_images(self, batch: Any) -> np.ndarray:
        """Raw features of a batch, float32."""
        import torch

        on_gpu = str(self.device).startswith("cuda")
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda" if on_gpu else "cpu", dtype=torch.float16, enabled=on_gpu
            ),
        ):
            return _to_numpy(self._forward(_as_batch(batch, self.device)))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """DINO has no text encoder."""
        raise NotImplementedError(f"{self.name} is an image-only model")


class DinoV2Encoder(_DinoEncoder):
    """DINOv2 ViT-S/14 from torch.hub (CM cell 9): the 384-d class-token features."""

    def __init__(self, device: str | None = None, *, model: Any = None) -> None:
        """Load the model. `model` skips the download (for tests)."""
        if model is None:
            import torch

            logger.info("Loading DINOv2 ViT-S/14 from torch.hub")
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        super().__init__("dinov2-s14", model, device)

    def _forward(self, batch: Any) -> Any:
        return self.model(batch)


class DinoVdeauEncoder(_DinoEncoder):
    """DinoVdeau-large from Hugging Face (CM cell 11): the 1024-d `pooler_output`."""

    def __init__(self, device: str | None = None, *, model: Any = None) -> None:
        """Load the model. `model` skips the download (for tests)."""
        if model is None:
            from transformers import AutoModel

            source = DINO_MODELS["dinovdeau-l"]["source"]
            logger.info("Loading %s (transformers)", source)
            model = AutoModel.from_pretrained(source)
        super().__init__("dinovdeau-l", model, device)

    def _forward(self, batch: Any) -> Any:
        return self.model(batch).pooler_output


def load_dino_encoder(model: str, device: str | None = None) -> ImageEncoder:
    """Load a DINO model by name: `dinov2-s14` or `dinovdeau-l`."""
    if model == "dinov2-s14":
        return DinoV2Encoder(device)
    if model == "dinovdeau-l":
        return DinoVdeauEncoder(device)
    raise ValueError(f"Unknown DINO model '{model}'; choose from {sorted(DINO_MODELS)}")
