import json

import cv2
import numpy as np
import pandas as pd
import pytest

from coralid.config import Config, ConfigError, Paths, SamConfig
from coralid.mask_compare import (
    COMPARED,
    FAIL,
    FRAME_DIFFERS,
    NOT_SEGMENTED,
    PASS,
    WARN,
    SegmentationComparison,
    compare_masks,
    compare_segmentations,
    iou_matrix,
    match_masks,
    summarize,
    verdict_for,
)
from coralid.masks import load_mask_json
from coralid.parity_segmentation import (
    check_config,
    choose_check_photos,
    run_segmentation_check,
    write_segmentation_check,
)
from tests.helpers import make_annotation, rect_mask, write_json_dir

W, H = 100, 75  # the mask frame; photos are 160x120 and max_side is 100
MIN_AREA = 600  # "large" in these small frames
BIG = [(5, 40, 5, 50), (45, 70, 30, 90), (10, 30, 60, 95)]  # y0, y1, x0, x1; each 600+ px
SMALL = (60, 66, 2, 12)  # 60 px: not large


def annos(boxes, shift=0, shape=(H, W)):
    """Annotations of rectangles; `shift` moves each one `shift` pixels to the right."""
    out = []
    for i, (y0, y1, x0, x1) in enumerate(boxes):
        out.append(make_annotation(rect_mask(shape, y0, y1, x0 + shift, x1 + shift), i))
    return out


def mask_file(tmp_path, name, boxes, shift=0, folder="saved", w=W, h=H):
    write_json_dir(tmp_path / folder, [(name, w, h, annos(boxes, shift, (h, w)), None)])
    return load_mask_json(tmp_path / folder / (name.rsplit(".", 1)[0] + ".json"))


# ------------------------------------------------------------------------- the config


def test_the_check_config_is_a_prefixed_copy_with_other_photos(tmp_path):
    cfg = Config(paths=Paths(data_root=str(tmp_path)))
    scratch = check_config(cfg, images_dir="data_50")
    assert scratch.run_name == "segcheck_iou60_sta55_pts32"
    assert scratch.json_dir == tmp_path / "results_all" / "segcheck_iou60_sta55_pts32" / "json"
    assert scratch.images_dir == tmp_path / "data_50"
    assert scratch.sam == cfg.sam and scratch.filters == cfg.filters  # the same settings
    assert cfg.paths.run_prefix == "" and cfg.run_name == "iou60_sta55_pts32"  # untouched
    assert cfg.images_dir == tmp_path / "data_all"


def test_the_check_config_refuses_to_share_the_active_json_folder(tmp_path):
    cfg = Config(paths=Paths(data_root=str(tmp_path)))
    with pytest.raises(ConfigError, match="active run's JSON folder"):
        check_config(cfg, prefix="")


# ------------------------------------------------------------------------- matching


def test_matching_is_one_to_one_and_best_first():
    iou = np.array([[0.95, 0.92], [0.60, 0.97]])  # saved 0 likes both, saved 1 likes new 1
    assert match_masks(iou) == [(1, 1, 0.97), (0, 0, 0.95)]
    two_for_one = np.array([[0.91, 0.99]])  # one saved mask, two new masks: only the best pairs
    assert match_masks(two_for_one) == [(0, 1, 0.99)]
    assert match_masks(np.array([[0.89]])) == []  # below the threshold
    assert match_masks(np.zeros((0, 3))) == []


def test_the_iou_matrix_is_saved_by_new(tmp_path):
    saved = mask_file(tmp_path, "A.JPG", BIG[:2])
    fresh = mask_file(tmp_path, "A.JPG", BIG, folder="new")
    iou = iou_matrix(saved, fresh)
    assert iou.shape == (2, 3)
    assert iou[0, 0] == pytest.approx(1.0) and iou[1, 1] == pytest.approx(1.0)
    assert iou[0, 2] == 0.0


# ------------------------------------------------------------------------- one photo


def test_identical_masks_are_all_found_with_iou_one(tmp_path):
    saved = mask_file(tmp_path, "A.JPG", [*BIG, SMALL])
    fresh = mask_file(tmp_path, "A.JPG", [*BIG, SMALL], folder="new")
    row = compare_masks(saved, fresh, MIN_AREA)
    assert row["status"] == COMPARED and row["orientation"] == "landscape"
    assert (row["n_saved"], row["n_new"], row["n_saved_large"], row["n_new_large"]) == (4, 4, 3, 3)
    assert (row["found"], row["found_large"]) == (4, 3)
    assert row["share_found"] == 1.0 and row["share_found_large"] == 1.0
    assert row["mean_best_iou"] == pytest.approx(1.0)


def test_a_one_pixel_shift_still_matches_but_a_big_one_does_not(tmp_path):
    saved = mask_file(tmp_path, "A.JPG", BIG)
    near = mask_file(tmp_path, "A.JPG", BIG, shift=1, folder="near")
    far = mask_file(tmp_path, "A.JPG", BIG, shift=20, folder="far")
    near_row, far_row = compare_masks(saved, near, MIN_AREA), compare_masks(saved, far, MIN_AREA)
    assert near_row["found"] == 3 and 0.9 < near_row["mean_best_iou"] < 1.0
    assert far_row["found"] < 3 and far_row["mean_best_iou"] < 0.9


def test_a_different_frame_is_reported_not_compared(tmp_path):
    saved = mask_file(tmp_path, "A.JPG", BIG)
    other = mask_file(tmp_path, "A.JPG", BIG, folder="new", w=120, h=90)
    row = compare_masks(saved, other, MIN_AREA)
    assert row["status"] == FRAME_DIFFERS
    assert (row["frame_saved"], row["frame_new"]) == ("100x75", "120x90")
    assert "found" not in row


def test_a_portrait_photo_is_labeled_by_its_saved_frame(tmp_path):
    saved = mask_file(tmp_path, "P.JPG", [(5, 60, 5, 40)], w=75, h=100)
    assert compare_masks(saved, saved, MIN_AREA)["orientation"] == "portrait"


def test_a_photo_with_no_masks_on_either_side_compares_cleanly(tmp_path):
    empty = mask_file(tmp_path, "E.JPG", [])
    row = compare_masks(empty, empty, MIN_AREA)
    assert row["status"] == COMPARED and row["found"] == 0 and row["share_found"] is None


# ------------------------------------------------------------------------- the verdict


def photos_table(**overrides):
    """A one-row `photos` table in the shape `compare_segmentations` builds."""
    row = {
        "image_name": "A.JPG",
        "status": COMPARED,
        "n_saved": 10,
        "n_new": 10,
        "n_saved_large": 4,
        "n_new_large": 4,
        "found": 10,
        "found_large": 4,
        "mean_best_iou": 0.99,
        "mean_best_iou_large": 0.99,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_a_faithful_run_passes():
    summary = summarize(photos_table())
    assert verdict_for(summary) == (PASS, [])
    assert summary["share_found_large"] == 1.0 and summary["count_ratio"] == 1.0


def test_losing_a_few_large_masks_warns():
    verdict, reasons = verdict_for(summarize(photos_table(found_large=3)))  # 75% of the large
    assert verdict == WARN and "75% of the large saved masks" in reasons[0]


def test_losing_most_large_masks_fails():
    verdict, reasons = verdict_for(summarize(photos_table(found_large=1, found=7)))
    assert verdict == FAIL and "25% of the large saved masks" in reasons[0]


def test_a_different_mask_count_warns_then_fails():
    assert verdict_for(summarize(photos_table(n_new=14)))[0] == WARN  # 1.4 times as many
    verdict, reasons = verdict_for(summarize(photos_table(n_new=25)))
    assert verdict == FAIL and "2.50 times as many masks" in reasons[0]


def test_many_small_masks_flipping_alone_warns_not_fails():
    verdict, reasons = verdict_for(summarize(photos_table(found=6)))  # large all found, 60% of all
    assert verdict == WARN and "60% of all saved masks" in reasons[0]


def test_a_frame_mismatch_a_missing_photo_or_nothing_compared_fail():
    table = pd.concat(
        [photos_table(), photos_table(image_name="B.JPG", status=FRAME_DIFFERS)],
        ignore_index=True,
    )
    verdict, reasons = verdict_for(summarize(table))
    assert verdict == FAIL and "1 photo(s): frame differs" in reasons
    missing = pd.DataFrame([{"image_name": "A.JPG", "status": NOT_SEGMENTED}])
    verdict, reasons = verdict_for(summarize(missing))
    assert verdict == FAIL and "no photo could be compared" in reasons
    assert verdict_for(summarize(pd.DataFrame(columns=["image_name", "status"])))[0] == FAIL


def test_totals_count_masks_not_photos():
    table = pd.concat(
        [
            photos_table(n_saved=100, n_new=100, found=50, n_saved_large=10, found_large=10),
            photos_table(
                image_name="B.JPG", n_saved=2, n_new=2, found=2, n_saved_large=1, found_large=1
            ),
        ],
        ignore_index=True,
    )
    summary = summarize(table)
    assert summary["share_found"] == pytest.approx(52 / 102)  # not the mean of 0.5 and 1.0
    assert summary["share_found_large"] == 1.0
    assert summary["n_saved"] == 102 and summary["n_compared"] == 2


# ------------------------------------------------------------------------- folders of JSONs


def test_compare_segmentations_reads_the_two_folders(tmp_path):
    for name, shift in (("A.JPG", 0), ("B.JPG", 1)):
        mask_file(tmp_path, name, BIG, folder="saved")
        mask_file(tmp_path, name, BIG, shift=shift, folder="new")
    mask_file(tmp_path, "C.JPG", BIG, folder="saved")  # never segmented in the new folder
    result = compare_segmentations(
        tmp_path / "saved", tmp_path / "new", ["A.JPG", "B.JPG", "C.JPG", "D.JPG"], MIN_AREA
    )
    status = dict(zip(result.photos["image_name"], result.photos["status"], strict=True))
    assert status == {
        "A.JPG": COMPARED,
        "B.JPG": COMPARED,
        "C.JPG": NOT_SEGMENTED,
        "D.JPG": "no saved JSON",
    }
    assert result.verdict == FAIL
    assert result.to_dict()["verdict"] == FAIL and result.to_dict()["n_compared"] == 2


# ------------------------------------------------------------------------- choosing photos


def run_with_photos(tmp_path, n_landscape=6, n_portrait=2):
    """An active run with landscape and portrait JSONs, and the photo names."""
    cfg = Config(paths=Paths(data_root=str(tmp_path)))
    cfg.sam = SamConfig(max_side=100)
    entries, names = [], []
    for i in range(n_landscape + n_portrait):
        portrait = i >= n_landscape
        name = f"IMG_{i:02d}.JPG"
        names.append(name)
        w, h = (75, 100) if portrait else (100, 75)
        entries.append((name, w, h, annos(BIG[:2], shape=(h, w)), None))
    write_json_dir(cfg.json_dir, entries)
    return cfg, names


def test_the_chosen_photos_include_a_portrait_one_and_are_reproducible(tmp_path):
    cfg, names = run_with_photos(tmp_path)
    for seed in range(8):
        chosen = choose_check_photos(names, cfg.json_dir, n=3, seed=seed)
        assert len(chosen) == 3 and chosen == sorted(chosen)
        assert set(chosen) & {"IMG_06.JPG", "IMG_07.JPG"}, (seed, chosen)  # a portrait photo
        assert chosen == choose_check_photos(names, cfg.json_dir, n=3, seed=seed)
    assert len({tuple(choose_check_photos(names, cfg.json_dir, 3, s)) for s in range(8)}) > 1


def test_photos_without_a_saved_json_are_not_chosen(tmp_path):
    cfg, names = run_with_photos(tmp_path, n_landscape=2, n_portrait=0)
    chosen = choose_check_photos([*names, "NO_JSON.JPG"], cfg.json_dir, n=5)
    assert chosen == names  # fewer than n: everything that can be compared
    assert choose_check_photos(["NO_JSON.JPG"], cfg.json_dir) == []


# ------------------------------------------------------------------------- the whole check


class ScriptedGenerator:
    """Stands in for CoralSCOP: answers per call, in the order the photos are segmented."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def generate(self, image):
        answer = self.answers[self.calls]
        self.calls += 1
        return answer


def sam_records(annotations):
    """SAM-style mask records for the masks of saved annotations (what `generate` returns)."""
    from coralid.masks import decode_rle

    records = []
    for a in annotations:
        mask = decode_rle(a["segmentation"])
        records.append(
            {
                "segmentation": mask,
                "area": int(mask.sum()),
                "bbox": list(a["bbox"]),
                "predicted_iou": 0.9,
                "point_coords": [[1.0, 1.0]],
                "stability_score": 0.8,
                "crop_box": [0, 0, mask.shape[1], mask.shape[0]],
            }
        )
    return records


def photo_files(cfg, folder, names):
    """160x120 photos (frame 100x75 at max_side 100) in `<root>/<folder>`."""
    out = cfg.data_root / folder
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for name in names:
        rgb = rng.randint(0, 256, size=(120, 160, 3), dtype=np.uint8)
        assert cv2.imwrite(str(out / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def active_snapshot(cfg):
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in cfg.json_dir.iterdir()}


def saved_records(cfg, names):
    """SAM-style records of the saved masks of each photo, in the order of `names`."""
    return [
        sam_records(load_mask_json(cfg.json_dir / f"{n.rsplit('.', 1)[0]}.json").annotations)
        for n in names
    ]


class Raising:
    def generate(self, image):
        raise RuntimeError("CUDA out of memory")


def test_the_check_segments_a_scratch_run_and_leaves_the_active_run_alone(tmp_path):
    cfg, names = run_with_photos(tmp_path, n_landscape=3, n_portrait=0)
    cfg.filters.min_area = MIN_AREA
    photo_files(cfg, "data_50", names)
    before = active_snapshot(cfg)
    chosen = choose_check_photos(names, cfg.json_dir, n=2, seed=1)
    generator = ScriptedGenerator(saved_records(cfg, chosen))  # photos are segmented in name order

    check = run_segmentation_check(
        cfg, n=2, photos_dir="data_50", image_names=names, generator=generator, seed=1
    )
    assert check.image_names == chosen
    assert check.verdict == PASS and check.comparison.summary["share_found"] == 1.0
    scratch = tmp_path / "results_all" / "segcheck_iou60_sta55_pts32"
    assert check.config.run_dir == scratch
    assert sorted(p.name for p in (scratch / "json").iterdir()) == [
        f"{n.rsplit('.', 1)[0]}.json" for n in chosen
    ]
    assert active_snapshot(cfg) == before  # not a byte, not a timestamp
    assert not (cfg.run_dir / "segmentation").exists()  # no log, no sidecar in the active run

    # a second call segments nothing again (the JSONs exist in the scratch run)
    again = ScriptedGenerator([])
    second = run_segmentation_check(
        cfg, n=2, photos_dir="data_50", image_names=names, generator=again, seed=1
    )
    assert again.calls == 0 and second.verdict == PASS
    assert active_snapshot(cfg) == before


def test_a_run_that_differs_from_the_saved_masks_is_reported(tmp_path):
    cfg, names = run_with_photos(tmp_path, n_landscape=2, n_portrait=0)
    cfg.filters.min_area = MIN_AREA
    photo_files(cfg, "data_50", names)
    other = [make_annotation(rect_mask((H, W), 40, 74, 0, 20), 0)]  # nothing like the saved masks
    generator = ScriptedGenerator([sam_records(other)] * 2)
    check = run_segmentation_check(
        cfg, n=2, photos_dir="data_50", image_names=names, generator=generator
    )
    assert check.verdict == FAIL
    assert check.comparison.summary["share_found_large"] == 0.0
    assert any("large saved masks" in reason for reason in check.comparison.reasons)


def test_a_segmentation_error_fails_the_check_and_says_why(tmp_path):
    cfg, names = run_with_photos(tmp_path, n_landscape=2, n_portrait=0)
    photo_files(cfg, "data_50", names)
    check = run_segmentation_check(
        cfg, n=2, photos_dir="data_50", image_names=names, generator=Raising()
    )
    assert check.verdict == FAIL
    reasons = check.comparison.reasons
    assert any("segmentation errors" in r and "CUDA out of memory" in r for r in reasons)
    assert not list(check.config.json_dir.glob("*.json"))  # an error never becomes a JSON


def test_no_photo_with_a_saved_json_is_a_clear_error(tmp_path):
    cfg, _ = run_with_photos(tmp_path, n_landscape=1, n_portrait=0)
    photo_files(cfg, "data_50", ["OTHER.JPG"])
    with pytest.raises(FileNotFoundError, match="nothing to compare with"):
        run_segmentation_check(cfg, photos_dir="data_50", generator=ScriptedGenerator([]))


def test_the_check_is_saved_as_new_files(tmp_path):
    cfg, names = run_with_photos(tmp_path, n_landscape=2, n_portrait=0)
    cfg.filters.min_area = MIN_AREA
    photo_files(cfg, "data_50", names)
    check = run_segmentation_check(
        cfg,
        n=2,
        photos_dir="data_50",
        image_names=names,
        generator=ScriptedGenerator(saved_records(cfg, sorted(names))),
    )
    first = write_segmentation_check(check, cfg.reports_dir, date="2026-10-12")
    second = write_segmentation_check(check, cfg.reports_dir, date="2026-10-12")
    assert first.name == "segcheck_2026-10-12" and second.name == "segcheck_2026-10-12_2"
    summary = json.loads((first / "summary.json").read_text())
    assert summary["verdict"] == PASS and summary["image_names"] == check.image_names
    assert summary["scratch_run"] == str(check.config.run_dir)
    assert len(pd.read_csv(first / "photos.csv")) == 2


def comparison_of(table):
    summary = summarize(table)
    verdict, reasons = verdict_for(summary)
    return SegmentationComparison(table, summary, verdict, reasons)


def test_the_report_lines_are_safe_when_nothing_could_be_compared():
    nothing = pd.DataFrame([{"image_name": "A.JPG", "status": NOT_SEGMENTED}])
    lines = comparison_of(nothing).report_lines()
    assert lines[0].startswith("FAIL: 0 of 1 photos compared")
    assert "all masks n/a, large masks (area >= 25,000) n/a" in lines[2]
    assert "  - no photo could be compared" in lines


def test_the_report_lines_give_the_numbers_a_person_pastes_back():
    lines = comparison_of(photos_table()).report_lines()
    assert lines[0].startswith("PASS: 1 of 1 photos compared")
    assert lines[1] == "masks: 10 saved, 10 new (x1.00)"
    assert "all masks 100.0%, large masks (area >= 25,000) 100.0%" in lines[2]
    assert lines[3] == "mean best IoU of a saved mask: 0.990"
