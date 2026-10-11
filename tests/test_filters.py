import inspect

import numpy as np
import pytest

from coralid.filters import (
    Rejection,
    aspect_ratio,
    bbox_intersection,
    filter_annotations,
    filter_annotations_with_reasons,
    is_sharp_enough,
    laplacian_variance,
    rle_intersection_area,
)
from coralid.masks import decode_rle, encode_mask
from tests import legacy_reference as legacy
from tests.helpers import ellipse_mask, make_annotation, random_mask, rect_mask


def _random_annotations(rng, shape, k):
    """k random masks; some are carved out of earlier ones so containment actually happens."""
    masks = []
    for _ in range(k):
        if masks and rng.rand() < 0.5:
            base = masks[rng.randint(len(masks))]
            ys, xs = np.nonzero(base)
            cy, cx = ys[rng.randint(len(ys))], xs[rng.randint(len(xs))]
            sub = base & ellipse_mask(shape, cy, cx, rng.uniform(2, 20), rng.uniform(2, 20))
            if rng.rand() < 0.5:  # grow it a little so it is only partly inside the base
                sub = sub | random_mask(rng, shape) & (rng.rand(*shape) < 0.3)
            masks.append(sub if sub.any() else random_mask(rng, shape))
        else:
            masks.append(random_mask(rng, shape))
    return masks, [make_annotation(m, i) for i, m in enumerate(masks)]


# --------------------------------------------------------------- RLE containment == decode


def test_rle_intersection_equals_the_decode_based_reference_on_random_mask_sets():
    rng = np.random.RandomState(0)
    n_sets = n_pairs = n_nonzero = n_nested = 0
    for _ in range(250):
        shape = (int(rng.randint(12, 70)), int(rng.randint(12, 70)))
        masks, annos = _random_annotations(rng, shape, int(rng.randint(2, 7)))
        n_sets += 1
        for i, a in enumerate(annos):
            for j, b in enumerate(annos):
                if i == j:
                    continue
                # reference: decode both masks and count the shared pixels
                ref = int(
                    np.logical_and(
                        decode_rle(a["segmentation"]), decode_rle(b["segmentation"])
                    ).sum()
                )
                assert rle_intersection_area(a["segmentation"], b["segmentation"]) == ref
                assert ref == int((masks[i] & masks[j]).sum())
                # the containment ratio (what the filter compares with the threshold) agrees too
                assert ref / a["area"] == int((masks[i] & masks[j]).sum()) / a["area"]
                n_pairs += 1
                n_nonzero += ref > 0
                n_nested += ref == a["area"]
    assert n_sets >= 200
    # the random sets must be non-trivial: lots of overlap and some fully nested pairs
    assert n_nonzero > n_pairs // 4
    assert n_nested > 50


def test_rle_intersection_accepts_str_and_bytes_counts_and_checks_frames():
    a = encode_mask(rect_mask((20, 30), 2, 10, 3, 15))
    b = encode_mask(rect_mask((20, 30), 5, 15, 8, 25))
    as_bytes = {"size": a["size"], "counts": a["counts"].encode("utf-8")}
    expected = rect_mask((20, 30), 5, 10, 8, 15).sum()
    assert rle_intersection_area(a, b) == rle_intersection_area(as_bytes, b) == expected
    with pytest.raises(ValueError, match="frames differ"):
        rle_intersection_area(a, encode_mask(np.ones((10, 10), bool)))


# --------------------------------------------------------------- filter == legacy loop


def test_filter_annotations_equals_the_legacy_loop_on_random_sets():
    rng = np.random.RandomState(1)
    trials = containment_trials = 0
    for _ in range(300):
        shape = (int(rng.randint(15, 60)), int(rng.randint(15, 60)))
        _, annos = _random_annotations(rng, shape, int(rng.randint(2, 9)))
        params = {
            "min_area": float(rng.choice([1, 20, 60])),
            "containment_thresh": float(rng.choice([0.5, 0.8, 0.95, 1.0])),
            "max_aspect_ratio": float(rng.choice([2.0, 3.0, 10.0])),
        }
        ref = legacy.filter_annotations(annos, **params)
        ours = filter_annotations(annos, **params)
        assert [a["id"] for a in ours] == [a["id"] for a in ref]
        assert ours == ref  # the same annotation dicts, in the same order
        kept, reasons = filter_annotations_with_reasons(annos, **params)
        assert [a["id"] for a in kept] == [a["id"] for a in ref]
        assert set(reasons) == {a["id"] for a in annos}
        assert [a["id"] for a in annos if reasons[a["id"]] is None] == [a["id"] for a in ref]
        trials += 1
        containment_trials += any(r and r.reason == "containment" for r in reasons.values())
    assert trials >= 200
    assert containment_trials > 80  # the comparison is not vacuous


def test_reasons_are_consistent_with_the_decode_based_containment():
    rng = np.random.RandomState(2)
    by_id_checked = 0
    for _ in range(100):
        shape = (int(rng.randint(15, 50)), int(rng.randint(15, 50)))
        masks, annos = _random_annotations(rng, shape, int(rng.randint(2, 8)))
        thresh = float(rng.choice([0.6, 0.8, 0.95]))
        _, reasons = filter_annotations_with_reasons(
            annos, min_area=10, containment_thresh=thresh, max_aspect_ratio=4.0
        )
        for aid, rej in reasons.items():
            if rej is None:
                continue
            assert rej.reason in {"area", "aspect", "containment"}
            if rej.reason == "containment":
                small, large = annos[rej.contained_by], annos[aid]
                assert small["area"] < large["area"]
                shared = int((masks[rej.contained_by] & masks[aid]).sum())
                assert shared / small["area"] >= thresh
                by_id_checked += 1
            else:
                assert rej.contained_by is None
    assert by_id_checked > 20


# --------------------------------------------------------------- hand-built cases

SHAPE = (60, 100)


def _annos(*masks, **kw):
    return [make_annotation(m, i, **kw) for i, m in enumerate(masks)]


def _ids(annos):
    return [a["id"] for a in annos]


def test_a_small_mask_inside_a_large_one_removes_the_large_one():
    large = rect_mask(SHAPE, 0, 50, 0, 80)
    small = rect_mask(SHAPE, 10, 30, 10, 30)
    annos = _annos(large, small)
    kept, reasons = filter_annotations_with_reasons(annos, min_area=1)
    assert _ids(kept) == [1]
    assert reasons[0] == Rejection("containment", contained_by=1)
    assert reasons[1] is None
    # the order of the input does not change which one is removed
    assert _ids(filter_annotations(_annos(small, large), min_area=1)) == [0]


def test_equal_areas_never_remove_each_other():
    a = rect_mask(SHAPE, 10, 30, 10, 30)
    same = rect_mask(SHAPE, 10, 30, 10, 30)
    shifted = rect_mask(SHAPE, 11, 31, 10, 30)  # same area, 95% overlap
    assert _ids(filter_annotations(_annos(a, same), min_area=1)) == [0, 1]
    assert _ids(filter_annotations(_annos(a, shifted), min_area=1)) == [0, 1]


def test_non_overlapping_bboxes_are_skipped():
    small = rect_mask(SHAPE, 0, 10, 0, 10)
    large = rect_mask(SHAPE, 20, 55, 40, 95)
    assert bbox_intersection([0, 0, 10, 10], [40, 20, 55, 35]) == 0
    assert _ids(filter_annotations(_annos(small, large), min_area=1)) == [0, 1]
    # bboxes that overlap but masks that do not: containment is 0, nothing is removed
    ring = rect_mask(SHAPE, 0, 50, 0, 80) & ~rect_mask(SHAPE, 5, 45, 5, 75)
    inner = rect_mask(SHAPE, 20, 30, 20, 30)
    assert _ids(filter_annotations(_annos(ring, inner), min_area=1)) == [0, 1]


def test_chain_a_in_b_in_c_keeps_only_the_innermost():
    a = rect_mask(SHAPE, 20, 30, 20, 30)
    b = rect_mask(SHAPE, 10, 40, 10, 50)
    c = rect_mask(SHAPE, 0, 55, 0, 90)
    kept, reasons = filter_annotations_with_reasons(_annos(a, b, c), min_area=1)
    assert _ids(kept) == [0]
    assert reasons[1] == Rejection("containment", 0)
    assert reasons[2] == Rejection("containment", 0)  # A reaches C before B does
    # reversed input order: each one is removed by the next smaller one instead
    kept, reasons = filter_annotations_with_reasons(_annos(c, b, a), min_area=1)
    assert _ids(kept) == [2]
    assert reasons[0] == Rejection("containment", 1)  # C removed by B
    assert reasons[1] == Rejection("containment", 2)  # B removed by A


def test_a_removed_mask_still_removes_larger_ones():
    """Legacy quirk: `i` is not skipped when it has already been removed.

    A sits in the part of B that lies outside C. B is 85% inside C. The legacy loop removes
    B (by A) and then still lets B remove C, leaving only A.
    """
    shape = (40, 100)
    c = rect_mask(shape, 0, 40, 0, 60)  # area 2400
    b = rect_mask(shape, 0, 40, 43, 63)  # area 800, 17 of 20 columns inside C
    a = rect_mask(shape, 5, 35, 60, 63)  # area 90, entirely outside C
    annos = _annos(a, b, c)
    expected = legacy.filter_annotations(annos, 1, 0.80, 100.0)
    assert _ids(expected) == [0]
    kept, reasons = filter_annotations_with_reasons(annos, min_area=1, max_aspect_ratio=100.0)
    assert _ids(kept) == [0]
    assert reasons[1] == Rejection("containment", 0)
    assert reasons[2] == Rejection("containment", 1)  # removed by an already-removed mask


def test_only_the_first_mask_that_triggers_is_recorded():
    large = rect_mask(SHAPE, 0, 50, 0, 80)
    s1 = rect_mask(SHAPE, 10, 20, 10, 20)
    s2 = rect_mask(SHAPE, 25, 40, 30, 45)
    kept, reasons = filter_annotations_with_reasons(_annos(s1, s2, large), min_area=1)
    assert _ids(kept) == [0, 1]
    assert reasons[2] == Rejection("containment", 0)


def test_the_threshold_is_inclusive():
    shape = (40, 100)
    c = rect_mask(shape, 0, 40, 0, 60)
    b = rect_mask(shape, 0, 40, 44, 64)  # exactly 16 of 20 columns inside C: containment 0.8
    assert _ids(filter_annotations(_annos(b, c), min_area=1, containment_thresh=0.80)) == [0]
    assert _ids(filter_annotations(_annos(b, c), min_area=1, containment_thresh=0.81)) == [0, 1]


def test_the_denominator_is_the_area_field_not_the_pixel_count():
    large = rect_mask(SHAPE, 0, 50, 0, 80)
    small = rect_mask(SHAPE, 10, 30, 10, 30)  # fully inside, 400 px
    inflated = [make_annotation(large, 0), make_annotation(small, 1, area=800)]  # 400/800 = 0.5
    assert _ids(filter_annotations(inflated, min_area=1)) == [0, 1]
    assert _ids(legacy.filter_annotations(inflated, 1, 0.8, 3.0)) == [0, 1]
    # a small mask that only half overlaps, but whose area field is halved: 200/200 = 1.0
    half_in = rect_mask(SHAPE, 40, 60, 10, 30)  # rows 40-49 inside the large mask
    deflated = [make_annotation(large, 0), make_annotation(half_in, 1, area=200)]
    assert _ids(filter_annotations(deflated, min_area=1)) == [1]
    assert _ids(legacy.filter_annotations(deflated, 1, 0.8, 3.0)) == [1]


def test_area_and_aspect_filters_run_before_containment():
    large = rect_mask(SHAPE, 0, 50, 0, 80)
    tiny = rect_mask(SHAPE, 10, 14, 10, 14)  # inside the large mask but below min_area
    sliver = rect_mask(SHAPE, 20, 22, 10, 70)  # inside the large mask, aspect 30
    annos = _annos(large, tiny, sliver)
    kept, reasons = filter_annotations_with_reasons(annos, min_area=20, max_aspect_ratio=3.0)
    assert _ids(kept) == [0]  # neither tiny nor sliver can remove the large mask
    assert reasons[1] == Rejection("area")
    assert reasons[2] == Rejection("aspect")
    # a mask that fails both is reported as an area rejection
    both = [make_annotation(rect_mask(SHAPE, 0, 1, 0, 50), 0)]
    assert filter_annotations_with_reasons(both, min_area=100)[1][0] == Rejection("area")


def test_zero_or_one_candidates_are_returned_as_they_are():
    assert filter_annotations([], min_area=1) == []
    single = _annos(rect_mask(SHAPE, 0, 10, 0, 10))
    assert filter_annotations(single, min_area=1) == single
    assert filter_annotations(single, min_area=1000) == []


def test_original_order_is_kept_and_inputs_are_not_modified():
    masks = [rect_mask(SHAPE, 0, 10 + 3 * i, 5 * i, 5 * i + 12) for i in range(5)]
    annos = _annos(*masks)
    snapshot = [dict(a) for a in annos]
    kept = filter_annotations(annos, min_area=1, containment_thresh=1.01)
    assert _ids(kept) == [0, 1, 2, 3, 4]
    assert kept is not annos
    assert annos == snapshot


def test_default_arguments_are_the_current_pipeline_values():
    sig = inspect.signature(filter_annotations)
    assert sig.parameters["min_area"].default == 25000
    assert sig.parameters["containment_thresh"].default == 0.80
    assert sig.parameters["max_aspect_ratio"].default == 3.0


# --------------------------------------------------------------- small helpers


def test_aspect_ratio_and_bbox_intersection():
    assert aspect_ratio([0, 0, 10, 2]) == 5
    assert aspect_ratio([0, 0, 2, 10]) == 5
    assert aspect_ratio([0, 0, 10, 0]) == 10  # the short side counts as at least 1
    assert aspect_ratio([0, 0, 7, 7]) == 1
    assert bbox_intersection([0, 0, 10, 10], [5, 5, 10, 10]) == 25
    assert bbox_intersection([0, 0, 10, 10], [10, 0, 10, 10]) == 0  # touching only
    assert bbox_intersection([0, 0, 10, 10], [20, 20, 5, 5]) == 0


# --------------------------------------------------------------- sharpness


def test_laplacian_variance_separates_flat_from_textured_crops():
    flat = np.full((40, 40, 3), 120, np.uint8)
    noise = np.random.RandomState(0).randint(0, 256, size=(40, 40, 3), dtype=np.uint8)
    assert laplacian_variance(flat) == 0.0
    assert laplacian_variance(noise) > 1000
    assert isinstance(laplacian_variance(noise), float)


def test_is_sharp_enough_matches_the_legacy_helper_and_is_inclusive():
    rng = np.random.RandomState(1)
    for _ in range(30):
        crop = rng.randint(0, 256, size=(30, 30, 3), dtype=np.uint8)
        crop = (crop * rng.uniform(0.0, 0.3)).astype(np.uint8)  # vary the contrast
        for threshold in (1.0, 50.0, 80.0, 150.0):
            assert is_sharp_enough(crop, threshold) == legacy.is_sharp_enough(crop, threshold)
    crop = np.random.RandomState(2).randint(0, 256, size=(30, 30, 3), dtype=np.uint8)
    value = laplacian_variance(crop)
    assert is_sharp_enough(crop, value)  # exactly at the threshold counts as sharp
    assert not is_sharp_enough(crop, value + 1e-6)


def test_is_sharp_enough_has_no_default_threshold():
    assert (
        inspect.signature(is_sharp_enough).parameters["threshold"].default
        is inspect.Parameter.empty
    )
    with pytest.raises(TypeError):
        is_sharp_enough(np.zeros((8, 8, 3), np.uint8))  # type: ignore[call-arg]
