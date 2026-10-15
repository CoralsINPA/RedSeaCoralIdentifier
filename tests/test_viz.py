import re

import cv2
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from coralid.config import CropConfig  # noqa: E402
from coralid.crops import bbox_crop  # noqa: E402
from coralid.masks import AnnotationStore  # noqa: E402
from coralid.viz import (  # noqa: E402
    confusion_heatmap,
    crop_gallery_pdf,
    cut_crops,
    funnel_chart,
    mask_overlay,
    photo_panels,
)
from tests.helpers import make_annotation, rect_mask, write_json_dir  # noqa: E402

SHAPE = (150, 200)
CROPS = CropConfig(padding_pct=0.0, min_visible=10)


@pytest.fixture
def store_and_images(tmp_path):
    """Two photos in a 200x150 mask frame (A is 2x that size) with two masks each."""
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.RandomState(1)
    for name, scale in (("A.JPG", 2), ("B.JPG", 1)):
        photo = rng.randint(0, 256, size=(150 * scale, 200 * scale, 3), dtype=np.uint8)
        assert cv2.imwrite(str(images / name), photo)
    annos = [
        make_annotation(rect_mask(SHAPE, 10, 80, 10, 90), 0),
        make_annotation(rect_mask(SHAPE, 90, 140, 100, 190), 1),
    ]
    json_dir = write_json_dir(
        tmp_path / "json", [("A.JPG", 200, 150, annos, None), ("B.JPG", 200, 150, annos, None)]
    )
    return AnnotationStore(json_dir), images


def _pages(path):
    return len(re.findall(rb"/Type\s*/Page(?![a-z])", path.read_bytes()))


def _keyed(uids, **columns):
    """A table keyed by `(image_name, coral_id)`, the way the plots and galleries take it."""
    table = pd.DataFrame(
        {
            "image_name": [u.rpartition("#")[0] for u in uids],
            "coral_id": [int(u.rpartition("#")[2]) for u in uids],
        }
    )
    for name, values in columns.items():
        table[name] = list(values)
    return table


def _table(n_per_cluster=(2, 14)):
    names = ["A.JPG", "B.JPG"]
    uids, clusters = [], []
    for cluster, n in enumerate(n_per_cluster):
        for i in range(n):
            uids.append(f"{names[i % 2]}#{i % 2}")
            clusters.append(cluster)
    return _keyed(uids, cluster=clusters)


# ------------------------------------------------------------------------- crops


def test_cut_crops_views(store_and_images):
    store, images = store_and_images
    items = pd.DataFrame({"image_name": ["A.JPG"], "coral_id": [0]})
    masked = cut_crops(items, store, images, CROPS)[0]
    square = cut_crops(items, store, images, CROPS, view="square")[0]
    box = cut_crops(items, store, images, CROPS, view="bbox")[0]
    assert masked.shape[:2] == (70, 80)  # the mask's bbox, no padding
    assert square.shape[:2] == (80, 80)  # padded to a square with black
    image = cv2.cvtColor(cv2.imread(str(images / "A.JPG")), cv2.COLOR_BGR2RGB)
    frame = cv2.resize(image, (200, 150))
    np.testing.assert_array_equal(box, bbox_crop(frame, [10, 10, 80, 70], CROPS.bbox_padding_pct))
    with pytest.raises(ValueError, match="view must be one of"):
        cut_crops(items, store, images, CROPS, view="circle")


# ------------------------------------------------------------------------- galleries


def test_gallery_starts_a_new_page_for_each_group(store_and_images, tmp_path):
    store, images = store_and_images
    table = _table((2, 14))  # 2 crops in cluster 0, 14 in cluster 1 (12 fit a page)
    path = crop_gallery_pdf(
        table, tmp_path / "out" / "g.pdf", store, images, group_col="cluster", crops=CROPS
    )
    assert path.read_bytes().startswith(b"%PDF") and _pages(path) == 3
    flat = crop_gallery_pdf(table, tmp_path / "flat.pdf", store, images, crops=CROPS)
    assert _pages(flat) == 2  # 16 crops, 12 per page, no groups
    with pytest.raises(FileExistsError):
        crop_gallery_pdf(table, path, store, images, group_col="cluster")
    assert not list(path.parent.glob(".*tmp*"))


def test_gallery_survives_missing_photos_and_uses_caption_column(store_and_images, tmp_path):
    store, images = store_and_images
    table = _keyed(["A.JPG#0", "Z.JPG#0", "B.JPG#99"], genus=["Acropora"] * 3)
    table["caption"] = ["a", "b", "c"]
    path = crop_gallery_pdf(
        table, tmp_path / "g.pdf", store, images, group_col="genus", caption_col="caption"
    )
    assert _pages(path) == 1  # the missing photo and the missing mask are "crop unavailable"


def test_gallery_reads_each_photo_once(store_and_images, tmp_path):
    store, images = store_and_images
    crop_gallery_pdf(_table((6, 6)), tmp_path / "g.pdf", store, images, group_col="cluster")
    assert store.loads == 2


# ------------------------------------------------------------------------- overlay and maps


def test_mask_overlay_draws_the_masks_of_the_table(store_and_images):
    store, images = store_and_images
    table = pd.DataFrame(
        {
            "image_name": ["A.JPG", "A.JPG", "B.JPG"],
            "coral_id": [0, 1, 0],
            "area": [70 * 80, 50 * 90, 70 * 80],
            "reject_reason": [None, "blurry", None],
        }
    )
    fig = mask_overlay(table, store, images, "A.JPG")
    ax = fig.axes[0]
    overlay = ax.get_images()[1].get_array()
    assert overlay.shape == (150, 200, 4)
    assert overlay[40, 40, 3] == pytest.approx(0.35) and overlay[0, 0, 3] == 0
    np.testing.assert_allclose(overlay[40, 40, :3], np.array([138, 43, 226]) / 255)
    assert ax.get_title() == "A.JPG: 2 masks"
    by_reason = mask_overlay(table, store, images, "A.JPG", color_col="reject_reason")
    legend = by_reason.axes[0].get_legend()
    assert sorted(t.get_text() for t in legend.get_texts()) == ["blurry", "kept"]
    only_one = mask_overlay(table, store, images, "A.JPG", max_masks=1)
    assert only_one.axes[0].get_title() == "A.JPG: 1 masks"
    assert only_one.axes[0].get_images()[1].get_array()[120, 150, 3] == 0  # the smaller one is out


def test_photo_panels_shows_original_masks_and_boxes(store_and_images):
    store, images = store_and_images
    table = pd.DataFrame({"image_name": ["A.JPG", "A.JPG", "B.JPG"], "coral_id": [0, 1, 0]})
    fig = photo_panels(table, store, images, "A.JPG")
    left, middle, right = fig.axes
    original = left.get_images()[0].get_array()
    painted = middle.get_images()[0].get_array()
    assert original.shape == painted.shape == (150, 200, 3)  # the photo, in its mask frame
    assert (painted[40, 40] != original[40, 40]).any()  # inside a mask: tinted
    np.testing.assert_array_equal(painted[0, 0], original[0, 0])  # outside: unchanged
    assert middle.get_title() == "Masks (2)"
    assert len(right.patches) == 2


def test_confusion_heatmap_shows_counts_or_row_shares():
    confusion = pd.DataFrame(
        [[3, 1], [0, 0]], index=pd.Index(["A", "B"], name="true"), columns=["A", "B"]
    )
    fig = confusion_heatmap(confusion, title="c")
    ax = fig.axes[0]
    np.testing.assert_array_equal(ax.get_images()[0].get_array(), [[3, 1], [0, 0]])
    assert [t.get_text() for t in ax.texts] == ["3", "1"]
    shares = confusion_heatmap(confusion, normalize=True).axes[0].get_images()[0].get_array()
    np.testing.assert_allclose(shares, [[0.75, 0.25], [0, 0]])  # an empty row stays 0


def test_the_funnel_chart_has_the_stages_and_the_orientation_shares():
    from coralid.funnel import filter_funnel

    manifest = pd.DataFrame(
        {
            "image_name": ["L.JPG"] * 6 + ["P.JPG"] * 4,
            "frame_w": [2560] * 6 + [1920] * 4,
            "frame_h": [1920] * 6 + [2560] * 4,
            "reject_reason": [
                "area",
                "area",
                "aspect",
                None,
                None,
                None,
                "area",
                "blurry",
                None,
                None,
            ],
        }
    )
    manifest["keep"] = manifest["reject_reason"].isna()
    fig = funnel_chart(filter_funnel(manifest), title="funnel")
    left, right = fig.axes
    assert [t.get_text() for t in left.get_yticklabels()] == [
        "kept",
        "blurry",
        "aspect",
        "area",
        "all masks",
    ]
    assert [p.get_width() for p in left.patches] == [5, 5, 6, 7, 10]  # masks left, bottom up
    assert [t.get_text() for t in right.get_yticklabels()] == ["area", "aspect", "blurry"]
    widths = [round(p.get_width(), 1) for p in right.patches]
    assert widths == [33.3, 16.7, 0.0, 25.0, 0.0, 25.0]  # landscape x3, then portrait x3
