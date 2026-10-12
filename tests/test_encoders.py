import numpy as np
import pytest

from coralid.encoders import MODEL_IDS, Encoder, load_encoder
from tests.fake_encoder import FakeEncoder


def test_model_ids_are_the_legacy_ones():
    assert MODEL_IDS == {
        "finetuned-bioclip": ("open_clip", "hf-hub:ReefNet/finetuned-bioclip"),
        "bioclip2": ("open_clip", "hf-hub:imageomics/bioclip-2"),
        "siglip2": ("siglip", "google/siglip2-so400m-patch14-384"),
    }


def test_unknown_model_name_is_rejected_before_anything_is_loaded():
    with pytest.raises(ValueError, match="Unknown model 'clip-9000'"):
        load_encoder("clip-9000")


def test_fake_encoder_satisfies_the_protocol():
    assert isinstance(FakeEncoder(), Encoder)


# The real encoders need torch. CI installs only the dev extra, so these run where torch is
# available (the Colab parity notebook exercises the real weights).


def _tiny_open_clip_parts():
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_linear = torch.nn.Linear(3, 5)
            self.embedding = torch.nn.Embedding(10, 5)

        def encode_image(self, x):
            assert not torch.is_grad_enabled()  # must run under no_grad
            return self.image_linear(x)

        def encode_text(self, tokens):
            assert not torch.is_grad_enabled()
            return self.embedding(tokens).mean(dim=1)

    def preprocess(pil):
        return torch.tensor(np.asarray(pil, dtype=np.float32).mean(axis=(0, 1)) / 255.0)

    def tokenizer(texts):
        return torch.tensor([[len(t) % 10, (len(t) * 3) % 10] for t in texts])

    torch.manual_seed(0)
    return Model(), preprocess, tokenizer


def test_open_clip_encoder_batches_texts_and_dtypes():
    from PIL import Image

    from coralid.encoders import OpenClipEncoder

    model, preprocess, tokenizer = _tiny_open_clip_parts()
    enc = OpenClipEncoder("tiny", device="cpu", parts=(model, preprocess, tokenizer))
    assert isinstance(enc, Encoder) and enc.family == "open_clip"
    images = [Image.fromarray(np.full((8, 8, 3), v, np.uint8)) for v in (10, 100, 200)]
    items = [enc.preprocess(i) for i in images]
    from_list = enc.encode_images(items)
    import torch

    from_stack = enc.encode_images(torch.stack(items))
    assert from_list.shape == (3, 5) and from_list.dtype == np.float32
    np.testing.assert_array_equal(from_list, from_stack)
    with torch.no_grad():
        np.testing.assert_allclose(from_list, model.encode_image(torch.stack(items)).numpy())
    texts = enc.encode_texts(["abc", "defgh"])
    assert texts.shape == (2, 5) and texts.dtype == np.float32


def test_siglip_encoder_accepts_tensors_and_output_objects():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from PIL import Image

    from coralid.encoders import SiglipEncoder

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 4)

        def get_image_features(self, pixel_values):
            return self.linear(pixel_values)  # a plain tensor (older transformers)

        def get_text_features(self, input_ids, **kwargs):
            # newer transformers return an output object
            return SimpleNamespace(pooler_output=input_ids.float().unsqueeze(-1).repeat(1, 4))

    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        def __call__(self, images=None, text=None, return_tensors=None, padding=None):
            if images is not None:
                mean = np.asarray(images, dtype=np.float32).mean(axis=(0, 1)) / 255.0
                return Inputs(pixel_values=torch.tensor(mean).unsqueeze(0))
            assert padding == "max_length"  # as in the legacy cell
            return Inputs(input_ids=torch.tensor([len(t) for t in text]))

    enc = SiglipEncoder("tiny", device="cpu", parts=(Model(), Processor()))
    assert enc.family == "siglip"
    item = enc.preprocess(Image.fromarray(np.full((6, 6, 3), 90, np.uint8)))
    assert tuple(item.shape) == (3,)  # the batch dimension is squeezed away
    feats = enc.encode_images([item, item])
    assert feats.shape == (2, 4) and feats.dtype == np.float32
    texts = enc.encode_texts(["ab", "cdef"])
    np.testing.assert_array_equal(texts, [[2.0] * 4, [4.0] * 4])


# ------------------------------------------------------------------------- DINO (Stage C)


def test_dino_models_are_the_legacy_ones_and_apart_from_the_clip_family():
    from coralid.encoders import DINO_MODELS, ImageEncoder

    assert set(DINO_MODELS) == {"dinov2-s14", "dinovdeau-l"}
    assert DINO_MODELS["dinov2-s14"]["input_size"] == 224  # CM cell 9
    assert DINO_MODELS["dinov2-s14"]["batch_size"] == 64
    assert DINO_MODELS["dinovdeau-l"]["input_size"] == 518  # CM cell 11
    assert DINO_MODELS["dinovdeau-l"]["batch_size"] == 16
    assert DINO_MODELS["dinovdeau-l"]["source"] == (
        "lombardata/DinoVdeau-large-2024_04_03-with_data_aug_batch-size32_epochs150_freeze"
    )
    assert not set(DINO_MODELS) & set(MODEL_IDS)  # no text encoder, so not a zero-shot model
    assert isinstance(FakeEncoder(), ImageEncoder)


def test_an_unknown_dino_model_is_rejected_before_anything_is_loaded():
    from coralid.encoders import load_dino_encoder

    with pytest.raises(ValueError, match="Unknown DINO model 'dino-9'"):
        load_dino_encoder("dino-9")


def _tiny_dino_model(torch, out_dim=5, pooled=False):
    from types import SimpleNamespace

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, out_dim)

        def forward(self, x):
            assert not torch.is_grad_enabled()  # must run under no_grad
            feats = self.linear(x.mean(dim=(2, 3)))
            return SimpleNamespace(pooler_output=feats) if pooled else feats

    torch.manual_seed(0)
    return Model()


def test_dinov2_encoder_preprocesses_like_the_legacy_transform():
    torch = pytest.importorskip("torch")
    transforms = pytest.importorskip("torchvision").transforms
    from PIL import Image

    from coralid.encoders import DinoV2Encoder, ImageEncoder

    enc = DinoV2Encoder("cpu", model=_tiny_dino_model(torch))
    assert isinstance(enc, ImageEncoder) and enc.name == "dinov2-s14" and enc.batch_size == 64
    crop = np.random.RandomState(0).randint(0, 256, size=(40, 30, 3), dtype=np.uint8)
    legacy = transforms.Compose(  # CM cell 9, verbatim
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(crop)
    item = enc.preprocess(Image.fromarray(crop))
    assert tuple(item.shape) == (3, 224, 224)
    assert torch.equal(item, legacy)


def test_dinov2_encoder_gives_float32_features_for_lists_and_stacks():
    torch = pytest.importorskip("torch")
    from PIL import Image

    from coralid.encoders import DinoV2Encoder

    enc = DinoV2Encoder("cpu", model=_tiny_dino_model(torch))
    items = [enc.preprocess(Image.fromarray(np.full((9, 9, 3), v, np.uint8))) for v in (5, 90, 250)]
    from_list = enc.encode_images(items)
    assert from_list.shape == (3, 5) and from_list.dtype == np.float32
    np.testing.assert_array_equal(from_list, enc.encode_images(torch.stack(items)))
    with pytest.raises(NotImplementedError, match="image-only"):
        enc.encode_texts(["a coral"])


def test_dinovdeau_encoder_takes_the_pooler_output_at_518():
    torch = pytest.importorskip("torch")
    from PIL import Image

    from coralid.encoders import DinoVdeauEncoder

    enc = DinoVdeauEncoder("cpu", model=_tiny_dino_model(torch, out_dim=7, pooled=True))
    assert enc.name == "dinovdeau-l" and enc.batch_size == 16
    item = enc.preprocess(Image.fromarray(np.full((12, 12, 3), 128, np.uint8)))
    assert tuple(item.shape) == (3, 518, 518)
    feats = enc.encode_images([item, item])
    assert feats.shape == (2, 7) and feats.dtype == np.float32
