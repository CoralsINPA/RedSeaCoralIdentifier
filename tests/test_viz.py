import re

import cv2
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from coralid.clustering import embedding_table  # noqa: E402
from coralid.config import CropConfig  # noqa: E402
from coralid.crops import bbox_crop  # noqa: E402
from coralid.masks import AnnotationStore  # noqa: E402
from coralid.viz import (  # noqa: E402
    confusion_heatmap,
    crop_gallery_pdf,
    cut_crops,
    embedding_scatter,
    mask_overlay,
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


def _table(n_per_cluster=(2, 14)):
    names = ["A.JPG", "B.JPG"]
    uids, clusters = [], []
    for cluster, n in enumerate(n_per_cluster):
        for i in range(n):
            uids.append(f"{names[i % 2]}#{i % 2}")
            clusters.append(cluster)
    return embedding_table(uids, cluster=clusters)


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
    table = embedding_table(["A.JPG#0", "Z.JPG#0", "B.JPG#99"], genus=["Acropora"] * 3)
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


def test_embedding_scatter_draws_the_table_columns():
    table = embedding_table(
        [f"I.JPG#{i}" for i in range(5)],
        x=[0, 1, 2, 3, 4.0],
        y=[4, 3, 2, 1, 0.0],
        c=[0, 0, 1, 1, 2],
    )
    fig = embedding_scatter(table, color="c", title="map")
    ax = fig.axes[0]
    np.testing.assert_array_equal(ax.collections[0].get_offsets(), table[["x", "y"]].to_numpy())
    np.testing.assert_array_equal(ax.collections[0].get_array(), [0, 0, 1, 1, 2])
    assert ax.get_title() == "map" and ax.get_xlabel() == "x"
    again = embedding_scatter(table, ax=ax)
    assert again is fig  # drawn into the panel it was given


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
