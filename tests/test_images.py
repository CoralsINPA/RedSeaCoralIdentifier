import cv2
import numpy as np
import pytest

from coralid.crops import bbox_crop, masked_crop
from coralid.images import ImageReadError, load_in_mask_frame, load_rgb
from coralid.masks import encode_mask
from tests.helpers import rect_mask


def _write_png(path, rgb):
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return path


def test_load_rgb_returns_rgb_order(tmp_path):
    rgb = np.zeros((6, 8, 3), np.uint8)
    rgb[..., 0] = 255  # pure red
    path = _write_png(tmp_path / "red.png", rgb)
    out = load_rgb(path)
    assert out.shape == (6, 8, 3) and out.dtype == np.uint8
    assert tuple(out[0, 0]) == (255, 0, 0)


def test_load_rgb_raises_a_clear_error(tmp_path):
    with pytest.raises(ImageReadError):
        load_rgb(tmp_path / "missing.jpg")
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"this is not an image")
    with pytest.raises(ImageReadError):
        load_rgb(corrupt)


def test_load_in_mask_frame_resizes_like_the_legacy_code(tmp_path):
    rng = np.random.RandomState(0)
    rgb = rng.randint(0, 256, size=(90, 120, 3), dtype=np.uint8)
    path = _write_png(tmp_path / "p.png", rgb)
    out = load_in_mask_frame(path, 80, 60)
    assert out.shape == (60, 80, 3)
    legacy = cv2.cvtColor(cv2.resize(cv2.imread(str(path)), (80, 60)), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(out, legacy)


def test_a_photo_already_in_the_frame_is_returned_unchanged(tmp_path):
    rgb = np.random.RandomState(1).randint(0, 256, size=(60, 80, 3), dtype=np.uint8)
    path = _write_png(tmp_path / "p.png", rgb)
    np.testing.assert_array_equal(load_in_mask_frame(path, 80, 60), rgb)


def test_4_to_3_photo_gives_the_same_pixels_as_the_legacy_2560x1920_resize(tmp_path):
    # IMG_4033.JPG is 3648x2736; the legacy code resized every photo to (2560, 1920).
    yy, xx = np.mgrid[:2736, :3648]
    rgb = np.stack([xx % 256, yy % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
    path = _write_png(tmp_path / "big.png", rgb)
    out = load_in_mask_frame(path, 2560, 1920)
    legacy = cv2.cvtColor(cv2.resize(cv2.imread(str(path)), (2560, 1920)), cv2.COLOR_BGR2RGB)
    assert out.shape == (1920, 2560, 3)
    np.testing.assert_array_equal(out, legacy)


def test_invalid_frame_sizes_are_rejected(tmp_path):
    path = _write_png(tmp_path / "p.png", np.zeros((4, 4, 3), np.uint8))
    for w, h in ((0, 4), (4, 0), (-1, 4)):
        with pytest.raises(ValueError):
            load_in_mask_frame(path, w, h)


def test_crop_from_an_image_loaded_in_the_mask_frame_lands_on_the_mask_pixels(tmp_path):
    """Regression test for the coordinate-frame bug (plan section 2.1, #1).

    The photo is twice the size of the mask frame. A bright patch sits at
    x 400-600, y 300-450 in the photo, which is x 200-300, y 150-225 in the mask frame.
    """
    rng = np.random.RandomState(0)
    photo = rng.randint(0, 100, size=(600, 800, 3), dtype=np.uint8)  # dark noise
    photo[300:450, 400:600] = 250  # the "coral"
    path = _write_png(tmp_path / "photo.png", photo)

    frame_w, frame_h = 400, 300
    mask = rect_mask((frame_h, frame_w), 153, 222, 203, 297)  # inside the patch, inset by 3 px
    bbox = [203, 153, 94, 69]
    rle = encode_mask(mask)

    in_frame = load_in_mask_frame(path, frame_w, frame_h)
    crop = masked_crop(in_frame, bbox, rle, padding_pct=0.0, min_visible=1)
    assert crop is not None
    assert (crop[mask[153:222, 203:297]] == 250).all()  # every mask pixel is bright coral

    # The legacy bug: crop the full-size photo with the smaller frame's bbox. That region
    # holds only dark noise, not the coral.
    wrong = bbox_crop(photo, bbox, padding_pct=0.0)
    assert wrong.max() < 100
