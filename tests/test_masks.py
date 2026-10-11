import json

import numpy as np
import pycocotools.mask as mask_util
import pytest

from coralid.masks import (
    AnnotationStore,
    CoralKey,
    MaskJsonError,
    audit_json_dir,
    decode_rle,
    decode_segmentation,
    encode_mask,
    json_name_for,
    list_mask_files,
    load_mask_json,
    parse_bbox,
    rle_to_bytes,
    write_mask_json,
)
from tests.helpers import make_annotation, random_mask, rect_mask, write_json_dir

# ------------------------------------------------------------------------- RLE


def test_rle_round_trip_on_random_masks():
    rng = np.random.RandomState(0)
    for _ in range(25):
        mask = random_mask(rng, (37, 53))
        rle = encode_mask(mask)
        assert rle["size"] == [37, 53]
        assert isinstance(rle["counts"], str)
        np.testing.assert_array_equal(decode_rle(rle), mask)


def test_rle_round_trip_edge_masks():
    for mask in (np.zeros((10, 12), bool), np.ones((10, 12), bool)):
        np.testing.assert_array_equal(decode_rle(encode_mask(mask)), mask)


def test_rle_counts_may_be_str_or_bytes():
    mask = rect_mask((20, 30), 2, 9, 4, 20)
    as_str = encode_mask(mask)
    as_bytes = {"size": as_str["size"], "counts": as_str["counts"].encode("utf-8")}
    assert isinstance(as_bytes["counts"], bytes)
    np.testing.assert_array_equal(decode_rle(as_str), mask)
    np.testing.assert_array_equal(decode_rle(as_bytes), mask)
    assert rle_to_bytes(as_str)["counts"] == as_bytes["counts"]
    assert rle_to_bytes(as_bytes)["counts"] == as_bytes["counts"]


def test_decode_rle_returns_bool():
    assert decode_rle(encode_mask(rect_mask((8, 8), 1, 4, 1, 4))).dtype == bool


def test_encode_matches_pycocotools_layout():
    mask = rect_mask((20, 30), 2, 9, 4, 20)
    ref = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    ours = encode_mask(mask)
    assert ours["counts"] == ref["counts"].decode("utf-8")
    assert ours["size"] == list(ref["size"])


def test_encode_rejects_non_2d():
    with pytest.raises(ValueError):
        encode_mask(np.zeros((2, 3, 4), bool))


def test_decode_segmentation_handles_all_three_formats():
    h, w = 20, 30
    mask = rect_mask((h, w), 2, 9, 3, 12)
    # compressed RLE
    np.testing.assert_array_equal(decode_segmentation(encode_mask(mask), h, w), mask)
    # uncompressed RLE (column-major run lengths, starting with background)
    flat = mask.T.reshape(-1).astype(np.uint8)
    runs, prev, count = [], 0, 0
    for v in flat:
        if v == prev:
            count += 1
        else:
            runs.append(count)
            prev, count = v, 1
    runs.append(count)
    uncompressed = {"size": [h, w], "counts": runs}
    np.testing.assert_array_equal(decode_segmentation(uncompressed, h, w), mask)
    # polygons (a rectangle given by its corners); merged into one mask
    poly = [[3, 2, 12, 2, 12, 9, 3, 9]]
    decoded = decode_segmentation(poly, h, w)
    assert decoded.shape == (h, w) and decoded.dtype == bool
    assert decoded.sum() > 0
    ref = mask_util.decode(mask_util.merge(mask_util.frPyObjects(poly, h, w))).astype(bool)
    np.testing.assert_array_equal(decoded, ref)


def test_decode_segmentation_rejects_other_types():
    with pytest.raises(TypeError):
        decode_segmentation("nope", 4, 4)


# ------------------------------------------------------------------------- keys


def test_coral_key_uid_round_trip():
    key = CoralKey("IMG_4033.JPG", 14)
    assert key.uid == "IMG_4033.JPG#14"
    assert CoralKey.from_uid("IMG_4033.JPG#14") == key


def test_coral_key_casts_ids_to_int():
    assert CoralKey("a.jpg", np.int64(3)).coral_id == 3
    assert CoralKey("a.jpg", 3.0).coral_id == 3
    assert CoralKey("a.jpg", "3").coral_id == 3
    assert isinstance(CoralKey("a.jpg", np.int32(3)).coral_id, int)
    with pytest.raises(ValueError):
        CoralKey("a.jpg", 3.5)
    with pytest.raises(ValueError):
        CoralKey("a.jpg", float("nan"))


def test_coral_key_from_uid_rejects_garbage():
    for bad in ("IMG_4033.JPG", "#4", "IMG#x"):
        with pytest.raises(ValueError):
            CoralKey.from_uid(bad)


def test_coral_keys_are_unique_across_images():
    # The same coral_id appears in every image, so only the pair identifies a coral.
    keys = {CoralKey(f"IMG_{i}.JPG", cid) for i in range(5) for cid in range(4)}
    assert len(keys) == 20
    assert len({k.coral_id for k in keys}) == 4
    assert len({k.uid for k in keys}) == 20


# ------------------------------------------------------------------------- parse_bbox


@pytest.mark.parametrize(
    "value",
    [
        [1, 2, 3, 4],
        (1.0, 2.0, 3.0, 4.0),
        np.array([1, 2, 3, 4]),
        "[1, 2, 3, 4]",
        "[1.0, 2.0, 3.0, 4.0]",
        "[1 2 3 4]",
        "[ 1.   2.   3.   4.]",
        "1,2,3,4",
        "(1, 2, 3, 4)",
        "  [1,2,   3,4 ]  ",
        "[1.0e0, 2e0, 3.0, 4.0E0]",
        "[1.\n 2.\n 3.\n 4.]",
    ],
)
def test_parse_bbox_formats(value):
    assert parse_bbox(value) == (1.0, 2.0, 3.0, 4.0)


def test_parse_bbox_float_strings():
    assert parse_bbox("[1234.56, 78.9, 300.25, 400.125]") == (1234.56, 78.9, 300.25, 400.125)
    assert parse_bbox("[-1.5, 0, 2, 3]") == (-1.5, 0.0, 2.0, 3.0)


@pytest.mark.parametrize(
    "bad",
    [
        "[1, 2, 3]",
        "[1, 2, 3, 4, 5]",
        "",
        "abc",
        "[1, 2, x, 4]",
        "__import__('os')",
        None,
        5,
        [1, 2, 3, None],
        "[1, 2, nan, 4]",
    ],
)
def test_parse_bbox_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_bbox(bad)


def test_parse_bbox_does_not_execute_code(tmp_path):
    marker = tmp_path / "pwned"
    with pytest.raises(ValueError):
        parse_bbox(f"__import__('pathlib').Path({str(marker)!r}).touch()")
    assert not marker.exists()


# ------------------------------------------------------------------------- JSON files


def test_json_name_is_image_stem_plus_json():
    assert json_name_for("IMG_4033.JPG") == "IMG_4033.json"
    assert json_name_for("a.b.jpeg") == "a.b.json"


def test_write_and_load_mask_json(tmp_path):
    annos = [make_annotation(rect_mask((20, 30), 2, 9, 3, 12), 0)]
    path = write_mask_json(tmp_path / "IMG_1.json", "IMG_1.JPG", 30, 20, annos)
    raw = json.loads(path.read_text())
    assert raw["image"] == {"image_id": 0, "width": 30, "height": 20, "file_name": "IMG_1.JPG"}
    assert "note" not in raw
    mf = load_mask_json(path)
    assert (mf.image_name, mf.width, mf.height, mf.frame) == ("IMG_1.JPG", 30, 20, (30, 20))
    assert mf.annotations == annos
    assert mf.note is None
    path2 = write_mask_json(tmp_path / "IMG_2.json", "IMG_2.JPG", 30, 20, [], note="no masks")
    assert load_mask_json(path2).note == "no masks"


def test_load_mask_json_rejects_malformed_files(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"annotations": []}))
    with pytest.raises(MaskJsonError):
        load_mask_json(tmp_path / "a.json")
    dup = {
        "image": {"width": 3, "height": 3, "file_name": "x.jpg"},
        "annotations": [{"id": 1}, {"id": 1}],
    }
    (tmp_path / "b.json").write_text(json.dumps(dup))
    with pytest.raises(MaskJsonError, match="unique"):
        load_mask_json(tmp_path / "b.json")


def test_list_mask_files_is_sorted_and_json_only(tmp_path):
    for name in ("b.json", "a.json", "c.txt", "d.json"):
        (tmp_path / name).write_text("{}")
    assert [p.name for p in list_mask_files(tmp_path)] == ["a.json", "b.json", "d.json"]


def _store_dir(tmp_path, n_images=4):
    entries = []
    for i in range(n_images):
        annos = [make_annotation(rect_mask((20, 30), 1 + c, 8 + c, 2, 12), c) for c in range(3)]
        entries.append((f"IMG_{i}.JPG", 30, 20, annos, None))
    return write_json_dir(tmp_path / "json", entries)


def test_annotation_store_get_and_annotations(tmp_path):
    store = AnnotationStore(_store_dir(tmp_path))
    annos = store.annotations("IMG_2.JPG")
    assert [a["id"] for a in annos] == [0, 1, 2]
    assert store.get(CoralKey("IMG_2.JPG", 1)) is annos[1]
    assert store.get("IMG_2.JPG#2")["id"] == 2
    assert store.frame("IMG_2.JPG") == (30, 20)
    with pytest.raises(KeyError):
        store.get(CoralKey("IMG_2.JPG", 99))
    with pytest.raises(KeyError):
        store.get(CoralKey("MISSING.JPG", 0))
    with pytest.raises(FileNotFoundError):
        store.annotations("MISSING.JPG")


def test_annotation_store_is_lazy_and_caches_with_lru(tmp_path):
    store = AnnotationStore(_store_dir(tmp_path, n_images=4), cache_size=2)
    assert store.loads == 0
    store.annotations("IMG_0.JPG")
    store.annotations("IMG_0.JPG")
    assert store.loads == 1  # second call hit the cache
    store.annotations("IMG_1.JPG")
    store.annotations("IMG_2.JPG")  # evicts IMG_0 (least recently used)
    assert store.loads == 3
    store.annotations("IMG_1.JPG")  # still cached
    assert store.loads == 3
    store.annotations("IMG_0.JPG")  # was evicted, read again
    assert store.loads == 4


def test_annotation_store_rejects_bad_cache_size(tmp_path):
    with pytest.raises(ValueError):
        AnnotationStore(tmp_path, cache_size=0)


# ------------------------------------------------------------------------- audit


def test_audit_json_dir_on_a_temp_directory(tmp_path):
    annos3 = [make_annotation(rect_mask((20, 30), 1, 8, 2, 12), c) for c in range(3)]
    annos1 = annos3[:1]
    json_dir = write_json_dir(
        tmp_path / "json",
        [
            ("A.JPG", 2560, 1920, annos3, None),
            ("B.JPG", 2560, 1920, annos1, None),
            ("C.JPG", 2560, 1706, annos3, None),
            ("D.JPG", 1920, 2560, [], "Skipped due to: CUDA out of memory"),
        ],
    )
    (json_dir / "broken.json").write_text("{not json")
    write_mask_json(json_dir / "Z.json", "OTHER.JPG", 2560, 1920, [])
    audit = audit_json_dir(json_dir)
    assert audit.n_files == 6
    assert audit.n_masks == 3 + 1 + 3 + 0 + 0
    assert audit.frame_sizes == {(2560, 1920): 3, (2560, 1706): 1, (1920, 2560): 1}
    assert audit.noted == [("D.json", "Skipped due to: CUDA out of memory")]
    assert [name for name, _ in audit.unreadable] == ["broken.json"]
    assert audit.name_mismatches == [("Z.json", "OTHER.JPG")]
    summary = audit.as_dict()
    assert summary["frame_sizes"]["2560x1920"] == 3
    json.dumps(summary)  # serializable


def test_audit_of_an_empty_directory(tmp_path):
    audit = audit_json_dir(tmp_path)
    assert (audit.n_files, audit.n_masks, audit.frame_sizes, audit.noted) == (0, 0, {}, [])
