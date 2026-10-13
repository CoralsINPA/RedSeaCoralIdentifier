import numpy as np
import pytest

from coralid.encoders import MODEL_IDS
from tests.fake_encoder import FakeEncoder


def test_dino_models_are_the_legacy_ones_and_apart_from_the_clip_family():
    from coralid.explore.dino import DINO_MODELS, ImageEncoder

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
    from coralid.explore.dino import load_dino_encoder

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

    from coralid.explore.dino import DinoV2Encoder, ImageEncoder

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

    from coralid.explore.dino import DinoV2Encoder

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

    from coralid.explore.dino import DinoVdeauEncoder

    enc = DinoVdeauEncoder("cpu", model=_tiny_dino_model(torch, out_dim=7, pooled=True))
    assert enc.name == "dinovdeau-l" and enc.batch_size == 16
    item = enc.preprocess(Image.fromarray(np.full((12, 12, 3), 128, np.uint8)))
    assert tuple(item.shape) == (3, 518, 518)
    feats = enc.encode_images([item, item])
    assert feats.shape == (2, 7) and feats.dtype == np.float32
