import json

import cv2
import numpy as np
import pandas as pd
import pytest

from coralid.config import Config, ConfigError, CropConfig, FilterConfig
from coralid.filters import filter_annotations
from coralid.manifest import (
    GEOMETRIC_COLUMNS,
    IMAGE_COLUMNS,
    geometric_manifest,
    image_manifest,
    read_manifest,
    write_manifest,
)
from coralid.masks import load_mask_json
from tests.helpers import make_annotation, rect_mask, write_json_dir

FRAME_W, FRAME_H = 200, 150
SHAPE = (FRAME_H, FRAME_W)
FILTERS = FilterConfig(min_area=100, containment_thresh=0.8, max_aspect_ratio=3.0)


def _records(df):
    """Rows as dicts with missing values as None, so dtype details do not matter."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def _values(series):
    """A column as a list with missing values as None (pandas versions differ on None/NaN)."""
    return series.astype(object).where(series.notna(), None).tolist()


def _geometry_annotations():
    large = rect_mask(SHAPE, 10, 100, 10, 150)  # contains `small`, so it is removed
    small = rect_mask(SHAPE, 20, 60, 20, 80)
    tiny = rect_mask(SHAPE, 120, 125, 160, 165)  # area 25 < 100
    sliver = rect_mask(SHAPE, 130, 134, 20, 120)  # 400 px, aspect 25
    lone = rect_mask(SHAPE, 105, 145, 100, 190)
    return [make_annotation(m, i) for i, m in enumerate([large, small, tiny, sliver, lone])]


@pytest.fixture
def geometry_dir(tmp_path):
    annos = _geometry_annotations()
    return write_json_dir(
        tmp_path / "json",
        [
            ("IMG_B.JPG", FRAME_W, FRAME_H, annos[:2], None),  # large + small
            ("IMG_A.JPG", FRAME_W, FRAME_H, annos, None),
            ("IMG_C.JPG", FRAME_W, FRAME_H, [], "Skipped due to: boom"),
        ],
    )


def test_geometric_manifest_columns_and_row_order(geometry_dir):
    df = geometric_manifest(geometry_dir, FILTERS)
    assert list(df.columns) == GEOMETRIC_COLUMNS
    assert len(df) == 5 + 2  # the noted, empty JSON adds no rows
    # JSON files in name order (A before B), masks in generator order
    assert df["image_name"].tolist() == ["IMG_A.JPG"] * 5 + ["IMG_B.JPG"] * 2
    assert df["coral_id"].tolist() == [0, 1, 2, 3, 4, 0, 1]
    assert df["uid"].tolist()[:2] == ["IMG_A.JPG#0", "IMG_A.JPG#1"]
    assert df["uid"].is_unique  # coral_id repeats across images, the uid does not
    assert df["frame_w"].unique().tolist() == [FRAME_W] and df["frame_h"].unique().tolist() == [
        FRAME_H
    ]


def test_geometric_manifest_records_reasons(geometry_dir):
    df = geometric_manifest(geometry_dir, FILTERS)
    a = df[df["image_name"] == "IMG_A.JPG"].set_index("coral_id")
    assert _values(a["stage1_reason"]) == ["containment", None, "area", "aspect", None]
    assert a.loc[0, "contained_by_smaller"] == 1
    assert a.loc[[1, 2, 3, 4], "contained_by_smaller"].isna().all()
    assert a.loc[3, "aspect"] == 100 / 4
    assert (a.loc[0, "bbox_x"], a.loc[0, "bbox_y"], a.loc[0, "bbox_w"]) == (10.0, 10.0, 140.0)
    assert a.loc[1, "area"] == 40 * 60
    assert a.loc[1, "predicted_iou"] == 0.9 and a.loc[1, "stability_score"] == 0.8


def test_geometric_manifest_agrees_with_filter_annotations(geometry_dir):
    df = geometric_manifest(geometry_dir, FILTERS)
    for name, group in df.groupby("image_name"):
        annos = load_mask_json(geometry_dir / f"{name[:-4]}.json").annotations
        kept = filter_annotations(annos, 100, 0.8, 3.0)
        survivors = group.loc[group["stage1_reason"].isna(), "coral_id"].tolist()
        assert survivors == [a["id"] for a in kept]


def test_geometric_manifest_uses_the_default_filters_and_can_be_restricted(geometry_dir):
    default = geometric_manifest(geometry_dir)  # min_area 25,000 removes everything here
    assert (default["stage1_reason"] == "area").all()
    only_b = geometric_manifest(geometry_dir, FILTERS, image_names=["IMG_B.JPG"])
    assert only_b["image_name"].unique().tolist() == ["IMG_B.JPG"]
    with pytest.raises(FileNotFoundError):
        geometric_manifest(geometry_dir, FILTERS, image_names=["NOPE.JPG"])


def test_geometric_manifest_of_an_empty_folder_has_the_columns(tmp_path):
    df = geometric_manifest(tmp_path)
    assert len(df) == 0 and list(df.columns) == GEOMETRIC_COLUMNS


# ----------------------------------------------------------------------- image manifest


def _write_photo(path, rgb):
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


@pytest.fixture
def image_setup(tmp_path):
    """A photo twice the size of the mask frame: noise on the left, a flat patch on the right."""
    rng = np.random.RandomState(0)
    photo = np.full((2 * FRAME_H, 2 * FRAME_W, 3), 120, np.uint8)
    photo[:, :FRAME_W] = rng.randint(0, 256, size=(2 * FRAME_H, FRAME_W, 3), dtype=np.uint8)
    images = tmp_path / "images"
    images.mkdir()
    _write_photo(images / "IMG_1.JPG", photo)
    (images / "IMG_3.JPG").write_bytes(b"not an image")  # present but unreadable

    sharp = rect_mask(SHAPE, 20, 60, 20, 80)  # on the noisy left half (frame x < 100)
    flat = rect_mask(SHAPE, 20, 60, 120, 180)  # on the flat right half
    speck = np.zeros(SHAPE, bool)
    speck[30:34, 30:35] = True  # 20 visible pixels, but a declared area that passes the filter
    tiny = rect_mask(SHAPE, 100, 105, 20, 25)
    annos1 = [
        make_annotation(sharp, 0),
        make_annotation(flat, 1),
        make_annotation(speck, 2, area=500),
        make_annotation(tiny, 3),
    ]
    annos2 = [make_annotation(sharp, 0)]
    json_dir = write_json_dir(
        tmp_path / "json",
        [
            ("IMG_1.JPG", FRAME_W, FRAME_H, annos1, None),
            ("IMG_2.JPG", FRAME_W, FRAME_H, annos2, None),  # photo absent
            ("IMG_3.JPG", FRAME_W, FRAME_H, annos2, None),  # photo unreadable
        ],
    )
    crops = CropConfig(padding_pct=0.0, min_visible=50)
    return json_dir, images, crops


def test_image_manifest_decisions(image_setup, caplog):
    json_dir, images, crops = image_setup
    with caplog.at_level("WARNING", logger="coralid.manifest"):
        df = image_manifest(json_dir, images, FILTERS, crops)
    assert sum("image_missing" in r.getMessage() for r in caplog.records) == 2
    assert list(df.columns) == IMAGE_COLUMNS
    assert len(df) == 4 + 1 + 1
    one = df[df["image_name"] == "IMG_1.JPG"].set_index("coral_id")
    assert _values(one["reject_reason"]) == [None, "blurry", "not_visible", "area"]
    assert one["keep"].tolist() == [True, False, False, False]
    assert one.loc[0, "visible_px"] == 40 * 60
    assert one.loc[0, "sharpness"] > FILTERS.sharpness_threshold
    assert one.loc[1, "visible_px"] == 40 * 60 and one.loc[1, "sharpness"] == 0.0
    assert one.loc[2, "visible_px"] == 20 and pd.isna(one.loc[2, "sharpness"])
    assert pd.isna(one.loc[3, "visible_px"]) and pd.isna(one.loc[3, "sharpness"])
    assert one.loc[3, "stage1_reason"] == "area"
    # missing and unreadable photos: the masks that passed the geometric filters are
    # "image_missing", never silently dropped
    others = df[df["image_name"] != "IMG_1.JPG"]
    assert others["reject_reason"].tolist() == ["image_missing", "image_missing"]
    assert not others["keep"].any()
    assert df["keep"].sum() == 1


def test_image_manifest_keeps_the_geometric_columns(image_setup):
    json_dir, images, crops = image_setup
    geo = geometric_manifest(json_dir, FILTERS)
    full = image_manifest(json_dir, images, FILTERS, crops)
    assert _records(full[GEOMETRIC_COLUMNS]) == _records(geo)


def test_image_manifest_keep_matches_the_legacy_per_crop_decision(image_setup):
    """keep == survives the filters, then masked crop (min_visible) and sharpness."""
    from coralid.crops import masked_crop
    from coralid.filters import is_sharp_enough
    from coralid.images import load_in_mask_frame

    json_dir, images, crops = image_setup
    df = image_manifest(json_dir, images, FILTERS, crops)
    mf = load_mask_json(json_dir / "IMG_1.json")
    img = load_in_mask_frame(images / "IMG_1.JPG", FRAME_W, FRAME_H)
    expected = []
    for anno in filter_annotations(mf.annotations, 100, 0.8, 3.0):
        crop = masked_crop(
            img, anno["bbox"], anno["segmentation"], crops.padding_pct, "black", crops.min_visible
        )
        if crop is not None and is_sharp_enough(crop, FILTERS.sharpness_threshold):
            expected.append(anno["id"])
    kept = df[(df["image_name"] == "IMG_1.JPG") & df["keep"]]["coral_id"].tolist()
    assert kept == expected == [0]


def test_image_manifest_can_be_restricted_to_some_images(image_setup):
    json_dir, images, crops = image_setup
    df = image_manifest(json_dir, images, FILTERS, crops, image_names=["IMG_1.JPG"])
    assert df["image_name"].unique().tolist() == ["IMG_1.JPG"]


# ----------------------------------------------------------------------- parquet + meta


def test_manifest_parquet_round_trip_with_meta(image_setup, tmp_path):
    json_dir, images, crops = image_setup
    df = image_manifest(json_dir, images, FILTERS, crops)
    out = write_manifest(df, tmp_path / "run" / "manifest" / "manifest.parquet", config=Config())
    assert out.exists()
    back = read_manifest(out)
    assert list(back.columns) == IMAGE_COLUMNS
    assert _records(back) == _records(df)
    meta = json.loads((out.parent / "manifest.meta.json").read_text())
    assert meta["n_rows"] == len(df) == 6
    assert meta["kind"] == "image" and meta["n_keep"] == 1 and meta["n_images"] == 3
    assert meta["config"]["filters"]["min_area"] == 25000
    assert not list(out.parent.glob(".*tmp*"))


def test_geometric_manifest_parquet_round_trip(geometry_dir, tmp_path):
    df = geometric_manifest(geometry_dir, FILTERS)
    out = write_manifest(df, tmp_path / "geo.parquet")
    assert _records(read_manifest(out)) == _records(df)
    meta = json.loads((tmp_path / "geo.meta.json").read_text())
    assert meta["kind"] == "geometric" and meta["n_rows"] == 7


# ----------------------------------------------------------------------- resumable + timings


def test_geometric_manifest_reports_timings(geometry_dir):
    timings = {}
    df = geometric_manifest(geometry_dir, FILTERS, timings=timings)
    assert set(timings) == {"load_s", "filter_s"} and min(timings.values()) >= 0.0
    assert _records(df) == _records(geometric_manifest(geometry_dir, FILTERS))


def test_resumable_image_manifest_equals_the_plain_one(image_setup, tmp_path):
    from coralid.batch import ResumableRun

    json_dir, images, crops = image_setup
    plain = image_manifest(json_dir, images, FILTERS, crops)
    run = ResumableRun(tmp_path / "work", "manifest", flush_every=2)
    resumable = image_manifest(json_dir, images, FILTERS, crops, run=run)
    assert list(resumable.columns) == IMAGE_COLUMNS
    assert _records(resumable) == _records(plain)
    assert run.done == {"IMG_1.JPG", "IMG_2.JPG", "IMG_3.JPG"} and not run.unresolved_failures()


def test_resumable_image_manifest_skips_done_images_and_retries_failed_ones(
    image_setup, tmp_path, monkeypatch
):
    import coralid.manifest as manifest_module
    from coralid.batch import ResumableRun

    json_dir, images, crops = image_setup
    plain = image_manifest(json_dir, images, FILTERS, crops)
    real_rows = manifest_module.image_rows
    seen = []

    def flaky(mask_file, *args):
        seen.append(mask_file.image_name)
        if mask_file.image_name == "IMG_2.JPG":
            raise OSError("Drive timed out")
        return real_rows(mask_file, *args)

    monkeypatch.setattr(manifest_module, "image_rows", flaky)
    run = ResumableRun(tmp_path / "work", "manifest")
    partial = image_manifest(json_dir, images, FILTERS, crops, run=run)
    assert seen == ["IMG_1.JPG", "IMG_2.JPG", "IMG_3.JPG"]
    assert sorted(partial["image_name"].unique()) == ["IMG_1.JPG", "IMG_3.JPG"]
    assert [f["image"] for f in run.unresolved_failures()] == ["IMG_2.JPG"]

    seen.clear()
    monkeypatch.setattr(
        manifest_module,
        "image_rows",
        lambda mf, *a: (seen.append(mf.image_name), real_rows(mf, *a))[1],
    )
    resumed_run = ResumableRun(tmp_path / "work", "manifest")
    full = image_manifest(json_dir, images, FILTERS, crops, run=resumed_run)
    assert seen == ["IMG_2.JPG"]  # only the failed image was redone
    assert not resumed_run.unresolved_failures()
    key = ["image_name", "coral_id"]
    assert _records(full.sort_values(key, ignore_index=True)) == _records(
        plain.sort_values(key, ignore_index=True)
    )


def test_resumable_image_manifest_logs_a_corrupt_json_and_goes_on(image_setup, tmp_path):
    from coralid.batch import ResumableRun

    json_dir, images, crops = image_setup
    (json_dir / "IMG_0.json").write_text("{ not json")
    run = ResumableRun(tmp_path / "work", "manifest")
    df = image_manifest(json_dir, images, FILTERS, crops, run=run)
    assert len(df) == 6  # the good images are all there
    (failure,) = run.unresolved_failures()
    assert failure["image"] == "IMG_0.json"  # the image name is unknown when the JSON is unreadable


def test_write_manifest_can_refuse_to_overwrite(geometry_dir, tmp_path):
    df = geometric_manifest(geometry_dir, FILTERS)
    out = write_manifest(df, tmp_path / "m.parquet", overwrite=False)
    before = out.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_manifest(df.head(1), out, overwrite=False)
    assert out.read_bytes() == before
    write_manifest(df.head(1), out)  # the default still replaces, as in Stage A1
    assert len(read_manifest(out)) == 1


# ----------------------------------------------------------------- the sharpness crop option


def test_sharpness_crop_defaults_keep_the_pipeline_definition():
    f = FilterConfig()
    assert (f.sharpness_crop, f.sharpness_cutoff) == ("masked", 80.0)
    bbox = FilterConfig(sharpness_crop="bbox")
    assert (bbox.sharpness_crop, bbox.sharpness_cutoff) == ("bbox", 150.0)
    assert (
        FilterConfig(sharpness_crop="bbox", sharpness_threshold_bbox=99.0).sharpness_cutoff == 99.0
    )
    with pytest.raises(ConfigError, match="sharpness_crop"):
        FilterConfig(sharpness_crop="outline")


def test_the_default_manifest_is_unchanged_by_the_option(image_setup):
    json_dir, images, crops = image_setup
    default = image_manifest(json_dir, images, FILTERS, crops)
    explicit = image_manifest(
        json_dir, images, FilterConfig(**{**FILTERS.__dict__, "sharpness_crop": "masked"}), crops
    )
    assert _records(default) == _records(explicit)


def test_bbox_sharpness_is_the_laplacian_variance_of_the_plain_box(image_setup):
    """The alternative measure: the unmasked, unpadded box in the mask frame."""
    json_dir, images, crops = image_setup
    bbox_filters = FilterConfig(**{**FILTERS.__dict__, "sharpness_crop": "bbox"})
    df = image_manifest(json_dir, images, bbox_filters, crops)
    one = df[df["image_name"] == "IMG_1.JPG"].set_index("coral_id")
    mf = load_mask_json(json_dir / "IMG_1.json")
    raw = cv2.resize(cv2.imread(str(images / "IMG_1.JPG")), (FRAME_W, FRAME_H))
    for coral_id in (0, 1):
        x, y, w, h = map(int, mf.by_id()[coral_id]["bbox"])
        box = raw[max(0, y) : y + h, max(0, x) : x + w]
        reference = cv2.Laplacian(cv2.cvtColor(box, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        assert one.loc[coral_id, "sharpness"] == pytest.approx(reference)
    assert one.loc[0, "keep"] and not one.loc[1, "keep"]  # noise passes, the flat patch does not


def test_masked_sharpness_responds_to_the_outline_but_bbox_sharpness_does_not(tmp_path):
    """A flat grey photo with an elliptical mask: the black corners of the masked crop are an
    edge, so the masked measure passes while the box of flat grey is not sharp at all."""
    from tests.helpers import ellipse_mask

    images = tmp_path / "images"
    images.mkdir()
    _write_photo(images / "FLAT.JPG", np.full((FRAME_H, FRAME_W, 3), 120, np.uint8))
    mask = ellipse_mask(SHAPE, 75, 100, 40, 80)
    json_dir = write_json_dir(
        tmp_path / "json", [("FLAT.JPG", FRAME_W, FRAME_H, [make_annotation(mask, 0)], None)]
    )
    crops = CropConfig(padding_pct=0.0, min_visible=50)
    masked = image_manifest(json_dir, images, FILTERS, crops).iloc[0]
    bbox = image_manifest(
        json_dir, images, FilterConfig(**{**FILTERS.__dict__, "sharpness_crop": "bbox"}), crops
    ).iloc[0]
    assert masked["sharpness"] > 80 and masked["keep"]
    assert bbox["sharpness"] < 1 and not bbox["keep"] and bbox["reject_reason"] == "blurry"
    assert masked["visible_px"] == bbox["visible_px"]  # visibility is judged the same way
