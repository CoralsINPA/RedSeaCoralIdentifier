"""Vision-language encoders behind one small interface.

An encoder turns a PIL crop into a model input (`preprocess`), a batch of those inputs into
feature vectors (`encode_images`) and prompts into feature vectors (`encode_texts`). Features
come back as float32 numpy arrays, **not** normalized: `coralid.zero_shot` normalizes them.

All models run in fp32 under `torch.no_grad()` on the best available device. torch,
open_clip and transformers are imported inside the classes, so `import coralid.encoders`
stays light. DINO encoders arrive in Stage C.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# model name -> (family, model id). The family decides how prompts are built (genera.py).
MODEL_IDS: dict[str, tuple[str, str]] = {
    "finetuned-bioclip": ("open_clip", "hf-hub:ReefNet/finetuned-bioclip"),
    "bioclip2": ("open_clip", "hf-hub:imageomics/bioclip-2"),
    "siglip2": ("siglip", "google/siglip2-so400m-patch14-384"),
}


@runtime_checkable
class Encoder(Protocol):
    """What `zero_shot` needs from a model."""

    name: str
    family: str  # "open_clip" or "siglip"

    def preprocess(self, image: Any) -> Any:
        """Turn one PIL image into a model input (a tensor)."""

    def encode_images(self, batch: Any) -> np.ndarray:
        """Encode a batch (a stacked tensor, or a list of `preprocess` outputs) to (n, d)."""

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode prompts to (n, d)."""


def pick_device() -> str:
    """`cuda` if a GPU is available, else `cpu`."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _as_batch(batch: Any, device: str) -> Any:
    import torch

    if isinstance(batch, (list, tuple)):
        batch = torch.stack(list(batch))
    return batch.to(device)


def _features_tensor(feats: Any) -> Any:
    """Some transformers versions return an output object instead of a tensor."""
    import torch

    return feats if isinstance(feats, torch.Tensor) else feats.pooler_output


def _to_numpy(feats: Any) -> np.ndarray:
    return feats.detach().float().cpu().numpy()


class OpenClipEncoder:
    """An open_clip model (BioCLIP, the ReefNet fine-tune of BioCLIP, BioCLIP 2)."""

    family = "open_clip"

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        *,
        parts: tuple[Any, Any, Any] | None = None,
    ) -> None:
        """Load `model_id`. `parts=(model, preprocess, tokenizer)` skips loading (for tests)."""
        self.name = model_id
        self.device = device or pick_device()
        if parts is None:
            import open_clip

            logger.info("Loading %s (open_clip) on %s", model_id, self.device)
            model, _, preprocess = open_clip.create_model_and_transforms(model_id)
            tokenizer = open_clip.get_tokenizer(model_id)
        else:
            model, preprocess, tokenizer = parts
        self.model = model.to(self.device).eval()
        self._preprocess = preprocess
        self._tokenizer = tokenizer

    def preprocess(self, image: Any) -> Any:
        """The model's own validation transform."""
        return self._preprocess(image)

    def encode_images(self, batch: Any) -> np.ndarray:
        """Image features of a batch, float32, not normalized."""
        import torch

        with torch.no_grad():
            return _to_numpy(self.model.encode_image(_as_batch(batch, self.device)))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Text features of prompts, float32, not normalized."""
        import torch

        tokens = self._tokenizer(list(texts)).to(self.device)
        with torch.no_grad():
            return _to_numpy(self.model.encode_text(tokens))


class SiglipEncoder:
    """A SigLIP 2 model through transformers (`AutoProcessor` / `AutoModel`)."""

    family = "siglip"

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        *,
        parts: tuple[Any, Any] | None = None,
    ) -> None:
        """Load `model_id`. `parts=(model, processor)` skips loading (for tests)."""
        self.name = model_id
        self.device = device or pick_device()
        if parts is None:
            from transformers import AutoModel, AutoProcessor

            logger.info("Loading %s (transformers) on %s", model_id, self.device)
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id)
        else:
            model, processor = parts
        self.model = model.to(self.device).eval()
        self._processor = processor

    def preprocess(self, image: Any) -> Any:
        """Pixel values of one image, without the batch dimension."""
        return self._processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    def encode_images(self, batch: Any) -> np.ndarray:
        """Image features of a batch, float32, not normalized."""
        import torch

        with torch.no_grad():
            feats = self.model.get_image_features(pixel_values=_as_batch(batch, self.device))
            return _to_numpy(_features_tensor(feats))

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Text features of prompts (padded to max length, as SigLIP was trained)."""
        import torch

        inputs = self._processor(text=list(texts), return_tensors="pt", padding="max_length")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            return _to_numpy(_features_tensor(self.model.get_text_features(**inputs)))


def load_encoder(model: str, device: str | None = None) -> Encoder:
    """Load one of the known models by name: finetuned-bioclip, bioclip2 or siglip2."""
    try:
        family, model_id = MODEL_IDS[model]
    except KeyError:
        raise ValueError(f"Unknown model '{model}'; choose from {sorted(MODEL_IDS)}") from None
    encoder = (
        OpenClipEncoder(model_id, device)
        if family == "open_clip"
        else SiglipEncoder(model_id, device)
    )
    encoder.name = model
    return encoder
