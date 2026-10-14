import cv2
import numpy as np
import pytest

from coralid.crops import bbox_crop, masked_crop, masked_crop_stats, masked_square_crop
from coralid.masks import encode_mask
from tests import legacy_reference as legacy
from tests.helpers import random_mask, rect_mask


def _photo(h=100, w=120, seed=0):
    """A random photo with no pure-black pixels, so black means "masked out"."""
    return np.random.RandomState(seed).randint(1, 256, size=(h, w, 3), dtype=np.uint8)


# ------------------------------------------------------------------------- masked_crop


def test_masked_crop_keeps_mask_pixels_and_blacks_out_the_rest():
    img = _photo()
    mask = rect_mask((100, 120), 30, 50, 40, 80)
    crop = masked_crop(img, [40, 30, 40, 20], encode_mask(mask), padding_pct=0.0)
    assert crop.shape == (20, 40, 3)
    np.testing.assert_array_equal(crop, img[30:50, 40:80])  # all inside the mask
    # a mask smaller than its bbox: the corner outside the mask becomes black
    mask2 = rect_mask((100, 120), 30, 40, 40, 80)
    crop2 = masked_crop(img, [40, 30, 40, 20], encode_mask(mask2), padding_pct=0.0, min_visible=1)
    np.testing.assert_array_equal(crop2[:10], img[30:40, 40:80])
    assert not crop2[10:].any()


def test_masked_crop_padding_is_int_truncated_per_axis():
    img = _photo()
    mask = np.ones((100, 120), bool)
    # w=40 -> pad_w = int(40 * 0.05) = 2, h=20 -> pad_h = int(20 * 0.05) = 1
    crop = masked_crop(img, [40, 30, 40, 20], encode_mask(mask))
    assert crop.shape == (20 + 2 * 1, 40 + 2 * 2, 3)
    np.testing.assert_array_equal(crop, img[29:51, 38:82])
    # float bbox values are truncated by int(), like the legacy code
    crop_f = masked_crop(img, [40.9, 30.9, 40.9, 20.9], encode_mask(mask), padding_pct=0.0)
    assert crop_f.shape == (20, 40, 3)


def test_masked_crop_clamps_at_the_image_edges():
    img = _photo(100, 120)
    mask = np.ones((100, 120), bool)
    rle = encode_mask(mask)
    top_left = masked_crop(img, [0, 0, 40, 20], rle, padding_pct=0.5)
    assert top_left.shape == (20 + 10, 40 + 20, 3)  # padding only right and below
    np.testing.assert_array_equal(top_left, img[0:30, 0:60])
    bottom_right = masked_crop(img, [100, 90, 40, 20], rle, padding_pct=0.5)
    np.testing.assert_array_equal(bottom_right, img[80:100, 80:120])
    # a bbox that starts outside the image on the right has nothing to cut
    assert masked_crop(img, [130, 10, 10, 10], rle) is None
    assert masked_crop(img, [10, 110, 10, 10], rle) is None


def test_masked_crop_min_visible_cutoff():
    img = _photo()
    mask = rect_mask((100, 120), 30, 40, 40, 50)  # 100 visible pixels
    rle = encode_mask(mask)
    bbox = [40, 30, 10, 10]
    assert masked_crop(img, bbox, rle, padding_pct=0.0, min_visible=100) is not None
    assert masked_crop(img, bbox, rle, padding_pct=0.0, min_visible=101) is None
    assert masked_crop(img, bbox, rle, padding_pct=0.0) is None  # default cutoff is 500
    crop, visible = masked_crop_stats(img, bbox, rle, padding_pct=0.0)
    assert visible == 100 and crop is not None


def test_visible_pixels_are_those_that_differ_from_the_background():
    img = _photo()
    img[35, 45] = 0  # a genuinely black photo pixel inside the mask counts as not visible
    mask = rect_mask((100, 120), 30, 40, 40, 50)
    _, visible = masked_crop_stats(img, [40, 30, 10, 10], encode_mask(mask), padding_pct=0.0)
    assert visible == 99


def test_masked_crop_background_values():
    img = _photo()
    mask = rect_mask((100, 120), 30, 40, 40, 50)
    rle = encode_mask(mask)
    for name, value in (("black", 0), ("gray", 128), ("white", 255)):
        crop = masked_crop(
            img, [40, 30, 20, 20], rle, padding_pct=0.0, background=name, min_visible=1
        )
        assert (crop[10:] == value).all()
        np.testing.assert_array_equal(crop[:10, :10], img[30:40, 40:50])
    with pytest.raises(ValueError, match="background"):
        masked_crop(img, [40, 30, 20, 20], rle, background="grey")


def test_masked_crop_resizes_a_mask_of_a_different_shape():
    img = _photo(100, 120)
    small = rect_mask((50, 60), 15, 25, 20, 40)  # half the resolution of the photo
    rle = encode_mask(small)
    expected_mask = cv2.resize(small.astype(np.uint8), (120, 100), interpolation=cv2.INTER_NEAREST)
    expected_mask = expected_mask.astype(bool)
    assert expected_mask.sum() == 4 * small.sum()
    crop = masked_crop(img, [40, 30, 40, 20], rle, padding_pct=0.0, min_visible=1)
    expected = img[30:50, 40:80].copy()
    expected[~expected_mask[30:50, 40:80]] = 0
    np.testing.assert_array_equal(crop, expected)


def test_masked_crop_does_not_modify_the_input():
    img = _photo()
    before = img.copy()
    masked_crop(img, [40, 30, 20, 20], encode_mask(rect_mask((100, 120), 30, 40, 40, 50)))
    np.testing.assert_array_equal(img, before)


def test_masked_crop_matches_the_legacy_function_on_random_inputs():
    rng = np.random.RandomState(1)
    for trial in range(150):
        h, w = int(rng.randint(40, 90)), int(rng.randint(40, 90))
        img = rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
        mask = random_mask(rng, (h, w))
        if trial % 5 == 0:  # exercise the mask-resize branch too
            mask_rle = encode_mask(random_mask(rng, (h // 2, w // 2)))
        else:
            mask_rle = encode_mask(mask)
        bbox = [rng.uniform(-5, w), rng.uniform(-5, h), rng.uniform(1, w), rng.uniform(1, h)]
        bbox[0], bbox[1] = max(bbox[0], 0.0), max(bbox[1], 0.0)  # SAM boxes are never negative
        kwargs = {
            "padding_pct": float(rng.choice([0.0, 0.05, 0.3])),
            "background": str(rng.choice(["black", "gray", "white"])),
            "min_visible": int(rng.choice([1, 50, 500])),
        }
        ours = masked_crop(img, bbox, mask_rle, **kwargs)
        ref = legacy.get_masked_crop(img, bbox, mask_rle, **kwargs)
        if ref is None:
            assert ours is None
        else:
            np.testing.assert_array_equal(ours, ref)


# ------------------------------------------------------------------------- bbox_crop


def test_bbox_crop_padding_and_clamping():
    img = _photo(100, 120)
    crop = bbox_crop(img, [40, 30, 40, 20])  # pad_w = int(40*.15) = 6, pad_h = 3
    np.testing.assert_array_equal(crop, img[27:53, 34:86])
    np.testing.assert_array_equal(bbox_crop(img, [0, 0, 40, 20]), img[0:23, 0:46])
    np.testing.assert_array_equal(bbox_crop(img, [100, 90, 40, 20]), img[87:100, 94:120])
    assert bbox_crop(img, [130, 10, 10, 10]) is None
    assert bbox_crop(img, [40, 30, 40, 20], padding_pct=0.0).shape == (20, 40, 3)


def test_bbox_crop_matches_the_legacy_function():
    rng = np.random.RandomState(2)
    for _ in range(100):
        img = _photo(60, 80, seed=int(rng.randint(0, 1000)))
        bbox = [rng.uniform(0, 90), rng.uniform(0, 70), rng.uniform(1, 60), rng.uniform(1, 60)]
        pad = float(rng.choice([0.0, 0.15, 0.4]))
        ours, ref = bbox_crop(img, bbox, pad), legacy.get_crop(img, bbox, pad)
        if ref is None:
            assert ours is None
        else:
            np.testing.assert_array_equal(ours, ref)


# ------------------------------------------------------------------------- masked_square_crop


def test_masked_square_crop_pads_to_a_black_square():
    img = _photo(100, 120)
    mask = np.ones((100, 120), bool)
    anno = {"segmentation": encode_mask(mask), "bbox": [10, 20, 15, 10]}  # w=15, h=10
    sq = masked_square_crop(img, anno["bbox"], anno["segmentation"])
    assert sq.shape == (15, 15, 3)
    # pad_y = (15 - 10) // 2 = 2 on top, 3 at the bottom
    assert not sq[:2].any() and not sq[12:].any()
    np.testing.assert_array_equal(sq[2:12], img[20:30, 10:25])
    # taller than wide: padding goes left (smaller half) and right (the remainder)
    tall = masked_square_crop(img, [10, 20, 10, 15], encode_mask(mask))
    assert tall.shape == (15, 15, 3)
    assert not tall[:, :2].any() and not tall[:, 12:].any()
    np.testing.assert_array_equal(tall[:, 2:12], img[20:35, 10:20])


def test_masked_square_crop_blacks_out_the_background_and_clamps():
    img = _photo(100, 120)
    mask = rect_mask((100, 120), 90, 100, 100, 120)
    sq = masked_square_crop(img, [100, 90, 50, 50], encode_mask(mask))  # bbox runs off the image
    assert sq.shape == (20, 20, 3)  # clamped to w = 120 - 100, h = 100 - 90
    assert masked_square_crop(img, [130, 10, 10, 10], encode_mask(mask)) is None


def test_masked_square_crop_rejects_a_mask_of_the_wrong_shape():
    img = _photo(100, 120)
    with pytest.raises(ValueError, match="differs"):
        masked_square_crop(img, [0, 0, 10, 10], encode_mask(np.ones((50, 60), bool)))


def test_masked_square_crop_accepts_polygons_and_uncompressed_rle():
    img = _photo(100, 120)
    poly = [[10, 20, 40, 20, 40, 50, 10, 50]]
    anno = {"segmentation": poly, "bbox": [10, 20, 30, 30]}
    ref = legacy.get_clean_masked_square_crop(img, anno, 100, 120)
    np.testing.assert_array_equal(masked_square_crop(img, anno["bbox"], poly), ref)


def test_masked_square_crop_matches_the_legacy_function_on_random_inputs():
    rng = np.random.RandomState(3)
    for _ in range(100):
        h, w = int(rng.randint(40, 90)), int(rng.randint(40, 90))
        img = rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
        anno = {
            "segmentation": encode_mask(random_mask(rng, (h, w))),
            "bbox": [rng.uniform(0, w), rng.uniform(0, h), rng.uniform(1, w), rng.uniform(1, h)],
        }
        ref = legacy.get_clean_masked_square_crop(img, anno, h, w)
        ours = masked_square_crop(img, anno["bbox"], anno["segmentation"])
        if ref is None:
            assert ours is None
        else:
            np.testing.assert_array_equal(ours, ref)


def test_sam_bbox_excludes_the_last_row_and_column_of_the_mask():
    """SAM writes `w = right - left` (indices are inclusive), so its bbox is one pixel short.

    Without padding the legacy crops lose the last row and column of the mask; the default
    5% padding gives that pixel back for any coral wider than 20 px. Reproduced, not fixed.
    """
    img = _photo()
    mask = rect_mask((100, 120), 20, 60, 40, 100)  # 40 rows x 60 columns
    sam_bbox = [40, 20, 59, 39]  # x, y, right - left, bottom - top
    rle = encode_mask(mask)
    assert masked_crop_stats(img, sam_bbox, rle, padding_pct=0.0)[1] == 39 * 59
    assert masked_crop_stats(img, sam_bbox, rle, padding_pct=0.05)[1] == 40 * 60
    square = masked_square_crop(img, sam_bbox, rle)
    assert square.shape == (59, 59, 3)  # 39 x 59 content, padded to a square
