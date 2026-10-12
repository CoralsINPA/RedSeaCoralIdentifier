import json
import sys
import types

import cv2
import numpy as np
import pytest

from coralid import segmentation
from coralid.config import Config, Paths, SamConfig
from coralid.masks import audit_json_dir, load_mask_json
from coralid.segmentation import (
    NOTE_NO_MASKS,
    SegmentationError,
    annotations_from_masks,
    build_generator,
    coralscop_dir,
    extras_from_masks,
    list_images,
    load_coralscop,
    plan_segmentation,
    run_segmentation,
)
from tests.helpers import rect_mask
from tests.legacy_reference import legacy_export_json

PHOTO_W, PHOTO_H = 160, 120  # the stand-in for a 3648x2736 photo
MAX_SIDE = 100  # the stand-in for 2560, so the frame is 100x75


def sam_record(mask, with_extras=True, points=None):
    """A mask record as CoralSCOP's `generate()` returns it (binary-mask output)."""
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    record = {
        "segmentation": mask,
        "area": int(mask.sum()),
        "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],  # SAM: right - left
        "predicted_iou": 0.9123456789,
        "point_coords": [[float(x0 + 3), float(y0 + 3)]] if points is None else points,
        "stability_score": 0.8765432109,
        "crop_box": [0, 0, mask.shape[1], mask.shape[0]],
    }
    if with_extras:
        record.update(
            cate_preds=1,
            fc_features=[0.25 * i for i in range(6)],
            similarity=[-np.inf, 0.5, 0.25],
        )
    return record


class FakeGenerator:
    """Stands in for `SamAutomaticMaskGenerator`: one scripted answer per call, in call order."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.images = []

    def generate(self, image):
        self.images.append(image)
        answer = self.answers[len(self.images) - 1] if self.answers else []
        if isinstance(answer, Exception):
            raise answer
        return answer(image) if callable(answer) else answer


def frame_masks(count=2):
    """`count` rectangular masks in the 100x75 frame."""
    shape = (75, 100)
    boxes = [(5, 40, 5, 50), (45, 70, 30, 90), (10, 30, 60, 95)]
    return [sam_record(rect_mask(shape, *b)) for b in boxes[:count]]


def make_cfg(tmp_path, names=("IMG_1.JPG", "IMG_2.JPG", "IMG_3.JPG"), save_features=False):
    cfg = Config(paths=Paths(data_root=str(tmp_path)))
    cfg.sam = SamConfig(max_side=MAX_SIDE, save_features=save_features)
    cfg.images_dir.mkdir(parents=True)
    rng = np.random.RandomState(0)
    for name in names:
        rgb = rng.randint(0, 256, size=(PHOTO_H, PHOTO_W, 3), dtype=np.uint8)
        assert cv2.imwrite(str(cfg.images_dir / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return cfg


# ------------------------------------------------------------------------- the JSON


def test_the_json_equals_the_legacy_export_byte_for_byte(tmp_path):
    cfg = make_cfg(tmp_path, ["IMG_1.JPG"])
    masks = [
        *frame_masks(2),
        sam_record(rect_mask((75, 100), 50, 70, 10, 60), points=np.array([[12.5, 33.0]])),
    ]
    run_segmentation(cfg, generator=FakeGenerator([masks]))
    written = (cfg.json_dir / "IMG_1.json").read_text(encoding="utf-8")
    legacy = json.dumps(legacy_export_json(masks, "IMG_1.JPG", 100, 75), indent=2)
    assert written == legacy
    # and it reads back with the package's own loader
    mask_file = load_mask_json(cfg.json_dir / "IMG_1.json")
    assert [a["id"] for a in mask_file.annotations] == [0, 1, 2]
    assert mask_file.frame == (100, 75) and mask_file.note is None
    assert list(mask_file.annotations[0]) == [
        "segmentation",
        "bbox",
        "area",
        "predicted_iou",
        "crop_box",
        "stability_score",
        "point_coords",
        "id",
    ]
    assert mask_file.annotations[0]["segmentation"]["size"] == [75, 100]


def test_coralscop_extras_never_reach_the_json(tmp_path):
    cfg = make_cfg(tmp_path, ["IMG_1.JPG"])
    run_segmentation(cfg, generator=FakeGenerator([frame_masks(2)]))
    text = (cfg.json_dir / "IMG_1.json").read_text()
    for forbidden in ("cate_preds", "fc_features", "similarity", "Infinity"):
        assert forbidden not in text
    assert not cfg.segmentation_dir.joinpath("extras").exists()  # off by default


def test_annotations_use_the_generator_order_for_ids():
    masks = frame_masks(3)
    annotations = annotations_from_masks(masks)
    assert [a["id"] for a in annotations] == [0, 1, 2]
    assert [a["area"] for a in annotations] == [m["area"] for m in masks]
    assert annotations_from_masks([]) == []


def test_the_frame_and_the_pixels_the_generator_sees(tmp_path):
    cfg = make_cfg(tmp_path, ["IMG_1.JPG"])
    generator = FakeGenerator([frame_masks(1)])
    run_segmentation(cfg, generator=generator)
    (seen,) = generator.images
    bgr = cv2.resize(cv2.imread(str(cfg.images_dir / "IMG_1.JPG")), (100, 75))
    np.testing.assert_array_equal(seen, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    assert seen.shape == (75, 100, 3) and seen.dtype == np.uint8
    mask_file = load_mask_json(cfg.json_dir / "IMG_1.json")
    assert mask_file.frame == (100, 75) and mask_file.image_name == "IMG_1.JPG"


def test_a_portrait_photo_gets_a_portrait_frame(tmp_path):
    cfg = make_cfg(tmp_path, [])
    rgb = np.random.RandomState(1).randint(0, 256, size=(PHOTO_W, PHOTO_H, 3), dtype=np.uint8)
    assert cv2.imwrite(str(cfg.images_dir / "IMG_P.JPG"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    generator = FakeGenerator([[]])
    run_segmentation(cfg, generator=generator)
    assert load_mask_json(cfg.json_dir / "IMG_P.json").frame == (75, 100)
    assert generator.images[0].shape == (100, 75, 3)


# ------------------------------------------------------------------------- never overwrite


def test_an_existing_json_is_skipped_and_never_rewritten(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.json_dir.mkdir(parents=True)
    sentinel = cfg.json_dir / "IMG_2.json"
    sentinel.write_bytes(b"an earlier run's file, valid or not")
    before = sentinel.stat()
    generator = FakeGenerator([frame_masks(1), frame_masks(2)])
    result = run_segmentation(cfg, generator=generator)
    assert len(generator.images) == 2  # IMG_2 was not even segmented
    assert sentinel.read_bytes() == b"an earlier run's file, valid or not"
    assert sentinel.stat().st_mtime_ns == before.st_mtime_ns
    assert result.written == ["IMG_1.JPG", "IMG_3.JPG"] and result.n_already_done == 1
    assert sorted(p.name for p in cfg.json_dir.iterdir()) == [
        "IMG_1.json",
        "IMG_2.json",
        "IMG_3.json",
    ]


def test_a_full_run_folder_is_left_alone_and_needs_no_model(tmp_path):
    """The active run has all its JSONs: running `segment` on it must do nothing at all."""
    cfg = make_cfg(tmp_path)
    cfg.json_dir.mkdir(parents=True)
    for name in ("IMG_1", "IMG_2", "IMG_3"):
        (cfg.json_dir / f"{name}.json").write_text(f"original {name}")
    snapshot = {p.name: p.read_bytes() for p in cfg.json_dir.iterdir()}
    result = run_segmentation(cfg)  # no generator and no checkpoint: it must not try to load one
    assert result.written == [] and result.n_already_done == 3 and result.failures == []
    assert {p.name: p.read_bytes() for p in cfg.json_dir.iterdir()} == snapshot
    assert not cfg.segmentation_dir.exists()  # not even a log folder


def test_a_json_that_appears_while_segmenting_is_not_overwritten(tmp_path):
    cfg = make_cfg(tmp_path, ["IMG_1.JPG", "IMG_2.JPG"])
    rival = cfg.json_dir / "IMG_1.json"

    def rival_writes_first(image):
        rival.write_text("written by another process")
        return frame_masks(1)

    generator = FakeGenerator([rival_writes_first, frame_masks(1)])
    result = run_segmentation(cfg, generator=generator)
    assert rival.read_text() == "written by another process"
    assert result.raced == ["IMG_1.JPG"] and result.written == ["IMG_2.JPG"]
    assert not any(p.name.startswith(".") for p in cfg.json_dir.iterdir())


def test_two_photos_sharing_a_stem_are_reported_not_merged(tmp_path):
    cfg = make_cfg(tmp_path, ["IMG_1.JPG"])
    rgb = np.zeros((PHOTO_H, PHOTO_W, 3), np.uint8)
    assert cv2.imwrite(str(cfg.images_dir / "IMG_1.png"), rgb)
    generator = FakeGenerator([frame_masks(1)])
    result = run_segmentation(cfg, generator=generator)
    assert len(generator.images) == 1 and result.written == ["IMG_1.JPG"]
    assert result.collisions == {"IMG_1.json": ["IMG_1.JPG", "IMG_1.png"]}


# ------------------------------------------------------------------------- errors


def test_an_error_goes_to_the_failure_log_and_never_to_a_json(tmp_path):
    cfg = make_cfg(tmp_path)
    boom = RuntimeError("CUDA out of memory")
    generator = FakeGenerator([frame_masks(1), boom, frame_masks(1)])
    result = run_segmentation(cfg, generator=generator)
    assert result.written == ["IMG_1.JPG", "IMG_3.JPG"]
    assert not (cfg.json_dir / "IMG_2.json").exists()  # not even an empty one
    assert [f["image"] for f in result.failures] == ["IMG_2.JPG"]
    log = cfg.segmentation_dir / "segment.failures.jsonl"
    (record,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert record["image"] == "IMG_2.JPG" and "CUDA out of memory" in record["error"]
    assert "Traceback" in record["traceback"] and record["time_utc"].endswith("Z")
    assert audit_json_dir(cfg.json_dir).noted == []  # no JSON carries a failure note


def test_a_failed_image_is_retried_and_finished_ones_are_not_redone(tmp_path):
    cfg = make_cfg(tmp_path)
    run_segmentation(
        cfg, generator=FakeGenerator([frame_masks(1), RuntimeError("OOM"), frame_masks(1)])
    )
    again = FakeGenerator([frame_masks(2)])
    result = run_segmentation(cfg, generator=again)
    assert len(again.images) == 1  # only IMG_2
    assert result.written == ["IMG_2.JPG"] and result.failures == []
    assert len(load_mask_json(cfg.json_dir / "IMG_2.json").annotations) == 2
    assert result.n_already_done == 2


def test_an_unreadable_photo_is_a_failure_not_an_empty_json(tmp_path):
    cfg = make_cfg(tmp_path, ["IMG_1.JPG"])
    (cfg.images_dir / "IMG_2.JPG").write_bytes(b"this is not a jpeg")
    generator = FakeGenerator([frame_masks(1)])
    result = run_segmentation(cfg, generator=generator)
    assert result.written == ["IMG_1.JPG"] and len(generator.images) == 1
    assert [f["image"] for f in result.failures] == ["IMG_2.JPG"]
    assert "unreadable" in result.failures[0]["error"]
    assert not (cfg.json_dir / "IMG_2.json").exists()


def test_a_photo_with_no_masks_gets_an_empty_json_with_a_note(tmp_path):
    cfg = make_cfg(tmp_path, ["IMG_1.JPG", "IMG_2.JPG"])
    result = run_segmentation(cfg, generator=FakeGenerator([[], frame_masks(1)]))
    empty = json.loads((cfg.json_dir / "IMG_1.json").read_text())
    assert empty["annotations"] == [] and empty["note"] == NOTE_NO_MASKS == "no masks"
    assert empty["image"] == {"image_id": 0, "width": 100, "height": 75, "file_name": "IMG_1.JPG"}
    assert result.n_empty == 1 and result.failures == []
    again = FakeGenerator([])
    run_segmentation(cfg, generator=again)
    assert again.images == []  # an empty image is done, not retried


def test_the_job_stops_after_too_many_failures_in_a_row(tmp_path):
    cfg = make_cfg(tmp_path)
    generator = FakeGenerator([RuntimeError("CUDA error")] * 3)
    with pytest.raises(SegmentationError, match="3 failures in a row"):
        run_segmentation(cfg, generator=generator, max_consecutive_failures=3)
    assert not any(cfg.json_dir.glob("*.json"))
    log = (cfg.segmentation_dir / "segment.failures.jsonl").read_text().splitlines()
    assert len(log) == 3


def test_a_success_resets_the_failure_streak(tmp_path):
    cfg = make_cfg(tmp_path)
    answers = [RuntimeError("a"), RuntimeError("b"), frame_masks(1)]
    result = run_segmentation(cfg, generator=FakeGenerator(answers), max_consecutive_failures=3)
    assert result.written == ["IMG_3.JPG"] and len(result.failures) == 2


# ------------------------------------------------------------------------- options


def test_limit_and_image_names(tmp_path):
    cfg = make_cfg(tmp_path)
    result = run_segmentation(cfg, generator=FakeGenerator([frame_masks(1)] * 3), limit=1)
    assert result.written == ["IMG_1.JPG"] and result.n_remaining == 2
    only = run_segmentation(
        cfg, generator=FakeGenerator([frame_masks(1)]), image_names=["IMG_3.JPG"]
    )
    assert only.written == ["IMG_3.JPG"] and only.n_images == 1
    with pytest.raises(FileNotFoundError):
        run_segmentation(cfg, generator=FakeGenerator([]), image_names=["NOPE.JPG"])


def test_list_images_finds_photos_only(tmp_path):
    for name in ("b.JPG", "a.jpeg", "c.Png", "d.tif", "notes.txt", ".hidden.jpg", "._IMG.JPG"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "folder.jpg").mkdir()
    assert list_images(tmp_path) == ["a.jpeg", "b.JPG", "c.Png", "d.tif"]
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path / "missing")


def test_plan_segmentation(tmp_path):
    (tmp_path / "A.json").write_text("{}")
    plan = plan_segmentation(["A.JPG", "B.JPG", "B.png", "C.JPG"], tmp_path)
    assert plan.done == ["A.JPG"] and plan.todo == ["B.JPG", "C.JPG"]
    assert plan.collisions == {"B.json": ["B.JPG", "B.png"]}
    assert plan_segmentation(["A.JPG"], tmp_path / "missing").todo == ["A.JPG"]


def test_a_run_record_is_written_next_to_the_log_and_not_in_json(tmp_path):
    cfg = make_cfg(tmp_path, ["IMG_1.JPG"])
    run_segmentation(cfg, generator=FakeGenerator([frame_masks(1)]))
    (meta_file,) = cfg.segmentation_dir.glob("segment_*.meta.json")
    meta = json.loads(meta_file.read_text())
    assert meta["kind"] == "segmentation_run" and meta["n_rows"] == 1
    assert meta["config"]["sam"]["pred_iou_thresh"] == 0.6
    assert [p.name for p in cfg.json_dir.iterdir()] == ["IMG_1.json"]  # json/ holds JSONs only


# ------------------------------------------------------------------------- optional extras


def test_extras_go_to_a_sidecar_and_leave_the_json_unchanged(tmp_path):
    plain = make_cfg(tmp_path / "plain", ["IMG_1.JPG", "IMG_2.JPG"])
    masks = [frame_masks(2), []]
    run_segmentation(plain, generator=FakeGenerator(masks))

    withx = make_cfg(tmp_path / "withx", ["IMG_1.JPG", "IMG_2.JPG"], save_features=True)
    run_segmentation(withx, generator=FakeGenerator(masks))
    for name in ("IMG_1.json", "IMG_2.json"):  # the JSON format stays identical
        assert (withx.json_dir / name).read_bytes() == (plain.json_dir / name).read_bytes()
    extras = withx.segmentation_dir / "extras"
    assert [p.name for p in extras.iterdir()] == ["IMG_1.npz"]  # none for the empty image
    with np.load(extras / "IMG_1.npz") as data:
        assert data["ids"].tolist() == [0, 1] and data["cate_preds"].tolist() == [1, 1]
        assert data["fc_features"].shape == (2, 6) and data["fc_features"].dtype == np.float32
        assert data["frame"].tolist() == [100, 75]


def test_extras_from_masks_needs_both_outputs():
    assert extras_from_masks([]) is None
    assert extras_from_masks(frame_masks(1)) is not None
    assert extras_from_masks([sam_record(rect_mask((9, 9), 1, 5, 1, 5), with_extras=False)]) is None


# ------------------------------------------------------------------------- CoralSCOP import


@pytest.fixture
def clean_modules(monkeypatch):
    """Isolate sys.path / sys.modules changes made by `load_coralscop`."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "segment_anything", raising=False)
    yield
    sys.modules.pop("segment_anything", None)


def _fake_coralscop(root):
    package = root / "segment_anything"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("IS_CORALSCOP = True\n")
    return root


def test_load_coralscop_puts_the_folder_first_on_the_path(tmp_path, clean_modules):
    root = _fake_coralscop(tmp_path / "CoralSCOP")
    module = load_coralscop(root)
    assert module.IS_CORALSCOP and sys.path[0] == str(root.resolve())
    assert load_coralscop(root) is module  # idempotent, path not extended again
    assert sys.path.count(str(root.resolve())) == 1


def test_load_coralscop_refuses_another_segment_anything(tmp_path, clean_modules):
    pip_version = types.ModuleType("segment_anything")
    pip_version.__file__ = str(tmp_path / "site-packages" / "segment_anything" / "__init__.py")
    sys.modules["segment_anything"] = pip_version
    root = _fake_coralscop(tmp_path / "CoralSCOP")
    with pytest.raises(SegmentationError, match="no coral/background head"):
        load_coralscop(root)


def test_load_coralscop_explains_missing_torch(tmp_path, clean_modules):
    root = tmp_path / "CoralSCOP"
    (root / "segment_anything").mkdir(parents=True)
    (root / "segment_anything" / "__init__.py").write_text("import torch_that_is_not_installed\n")
    with pytest.raises(SegmentationError, match=r"needs torch.*'\.\[segment\]'"):
        load_coralscop(root)


def test_load_coralscop_explains_a_missing_submodule(tmp_path, clean_modules):
    with pytest.raises(SegmentationError, match="git submodule update --init"):
        load_coralscop(tmp_path / "empty")


def test_the_default_folder_is_the_submodule_or_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv(segmentation.CORALSCOP_ENV, raising=False)
    assert coralscop_dir().parts[-2:] == ("third_party", "CoralSCOP")
    monkeypatch.setenv(segmentation.CORALSCOP_ENV, str(tmp_path))
    assert coralscop_dir() == tmp_path


def test_the_real_submodule_has_the_coral_head():
    """CoralSCOP's builder differs from stock SAM by its 2-class head (`cate_num=2`)."""
    root = coralscop_dir()
    build = root / "segment_anything" / "build_sam.py"
    if not build.is_file():
        pytest.skip("the CoralSCOP submodule is not checked out")
    assert "cate_num=2" in build.read_text().replace(" ", "")


# ------------------------------------------------------------------------- the generator


def _stub_segment_anything(root, calls):
    """A stand-in `segment_anything` that records how the model and generator are built."""

    class Model:
        mask_decoder = types.SimpleNamespace(cate_prediction_head=object())

        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

    def builder(checkpoint=None):
        calls["checkpoint"] = checkpoint
        return Model()

    class Generator:
        def __init__(self, **kwargs):
            calls["generator"] = kwargs

    package = types.ModuleType("segment_anything")
    package.__file__ = str(root / "segment_anything" / "__init__.py")
    package.sam_model_registry = {"vit_b": builder}
    package.SamAutomaticMaskGenerator = Generator
    return package


def test_build_generator_uses_the_legacy_settings_and_a_strict_load(tmp_path, clean_modules):
    root = _fake_coralscop(tmp_path / "CoralSCOP")
    checkpoint = tmp_path / "vit_b_coralscop.pth"
    checkpoint.write_bytes(b"weights")
    calls = {}
    sys.modules["segment_anything"] = _stub_segment_anything(root, calls)
    build_generator(SamConfig(), checkpoint, device="cpu", coralscop_root=root)
    assert calls["checkpoint"] == str(checkpoint)  # the builder loads it with strict=True
    assert calls["device"] == "cpu" and calls["eval"] is True
    assert calls["generator"] == {
        "model": calls["generator"]["model"],
        "points_per_side": 32,
        "pred_iou_thresh": 0.60,
        "stability_score_thresh": 0.55,
        "crop_n_layers": 1,
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 100,
    }


def test_build_generator_needs_the_checkpoint_and_the_coral_head(tmp_path, clean_modules):
    root = _fake_coralscop(tmp_path / "CoralSCOP")
    calls = {}
    sys.modules["segment_anything"] = _stub_segment_anything(root, calls)
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        build_generator(SamConfig(), tmp_path / "missing.pth", device="cpu", coralscop_root=root)
    assert "checkpoint" not in calls  # the builder was never called (it would return a blank model)
    checkpoint = tmp_path / "w.pth"
    checkpoint.write_bytes(b"w")
    stock = sys.modules["segment_anything"]
    stock.sam_model_registry["vit_b"] = lambda checkpoint=None: types.SimpleNamespace(
        mask_decoder=types.SimpleNamespace()
    )
    with pytest.raises(SegmentationError, match="coral/background head"):
        build_generator(SamConfig(), checkpoint, device="cpu", coralscop_root=root)
    with pytest.raises(SegmentationError, match="Unknown sam.model_type"):
        build_generator(
            SamConfig(model_type="vit_x"), checkpoint, device="cpu", coralscop_root=root
        )
