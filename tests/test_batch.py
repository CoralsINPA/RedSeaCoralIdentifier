import json

import numpy as np
import pandas as pd
import pytest

from coralid.batch import ResumableRun, RunStateError


def _rows(image, n, start=0):
    return pd.DataFrame(
        {
            "image_name": [image] * n,
            "coral_id": list(range(start, start + n)),
            "value": [float(i) for i in range(start, start + n)],
        }
    )


def _emb(n, seed=0):
    return np.random.RandomState(seed).randn(n, 4).astype(np.float32)


def test_add_flush_and_merge_round_trip(tmp_path):
    run = ResumableRun(tmp_path, "r", flush_every=100)
    run.add("A.JPG", _rows("A.JPG", 2), _emb(2, 1))
    run.add("B.JPG", _rows("B.JPG", 3), _emb(3, 2))
    assert run.done == {"A.JPG", "B.JPG"}  # buffered counts as done within the process
    assert not list(tmp_path.glob("r.part_*"))  # nothing written yet
    assert run.flush() == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "r.done.json",
        "r.part_0000.npz",
        "r.part_0000.parquet",
    ]
    rows, matrix = run.merge()
    assert rows["image_name"].tolist() == ["A.JPG"] * 2 + ["B.JPG"] * 3
    assert rows["coral_id"].tolist() == [0, 1, 0, 1, 2]
    assert matrix.shape == (5, 4)
    np.testing.assert_array_equal(matrix[:2], _emb(2, 1))
    np.testing.assert_array_equal(matrix[2:], _emb(3, 2))


def test_auto_flush_every_n_images_numbers_the_parts(tmp_path):
    run = ResumableRun(tmp_path, "r", flush_every=2)
    for i in range(5):
        run.add(f"I{i}.JPG", _rows(f"I{i}.JPG", 1))
    # 5 images, flush every 2: parts 0 and 1 are written, the fifth image is still buffered
    assert sorted(p.name for p in tmp_path.glob("r.part_*")) == [
        "r.part_0000.parquet",
        "r.part_0001.parquet",
    ]
    state = json.loads((tmp_path / "r.done.json").read_text())
    assert sorted(state["done"]) == ["I0.JPG", "I1.JPG", "I2.JPG", "I3.JPG"]
    assert state["parts"] == [0, 1]
    rows, matrix = run.merge()  # merge flushes the last image first
    assert matrix is None
    assert rows["image_name"].tolist() == [f"I{i}.JPG" for i in range(5)]


def test_resume_skips_done_images_and_continues_the_numbering(tmp_path):
    first = ResumableRun(tmp_path, "r")
    first.add("A.JPG", _rows("A.JPG", 2))
    first.add("B.JPG", _rows("B.JPG", 1))
    first.flush()

    second = ResumableRun(tmp_path, "r")
    assert second.done == {"A.JPG", "B.JPG"}
    assert second.pending(["B.JPG", "C.JPG", "A.JPG", "D.JPG"]) == ["C.JPG", "D.JPG"]
    second.add("C.JPG", _rows("C.JPG", 4))
    second.flush()
    assert sorted(p.name for p in tmp_path.glob("r.part_*")) == [
        "r.part_0000.parquet",
        "r.part_0001.parquet",
    ]
    rows, _ = ResumableRun(tmp_path, "r").merge()
    assert rows["image_name"].tolist() == ["A.JPG"] * 2 + ["B.JPG"] + ["C.JPG"] * 4
    assert rows["value"].tolist() == [0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0]


def test_adding_a_done_image_again_is_an_error(tmp_path):
    run = ResumableRun(tmp_path, "r")
    run.add("A.JPG", _rows("A.JPG", 1))
    with pytest.raises(ValueError, match="already recorded"):
        run.add("A.JPG", _rows("A.JPG", 1))
    run.flush()
    with pytest.raises(ValueError, match="already recorded"):
        ResumableRun(tmp_path, "r").add("A.JPG", _rows("A.JPG", 1))


def test_failed_images_are_logged_not_done_and_retried(tmp_path):
    run = ResumableRun(tmp_path, "r")
    run.add("A.JPG", _rows("A.JPG", 1))
    try:
        raise MemoryError("CUDA out of memory")
    except MemoryError as err:
        run.fail("B.JPG", err)
    run.flush()
    assert run.done == {"A.JPG"}
    (record,) = run.failures()
    assert record["image"] == "B.JPG"
    assert record["error"] == "MemoryError: CUDA out of memory"
    assert "Traceback" in record["traceback"] and record["time_utc"].endswith("Z")
    assert [r["image"] for r in run.unresolved_failures()] == ["B.JPG"]

    resumed = ResumableRun(tmp_path, "r")
    assert resumed.pending(["A.JPG", "B.JPG"]) == ["B.JPG"]  # retried
    resumed.add("B.JPG", _rows("B.JPG", 2))
    resumed.flush()
    assert resumed.unresolved_failures() == []  # fixed ...
    assert len(resumed.failures()) == 1  # ... but the log keeps the history
    assert resumed.summary() == {"done": 2, "parts": 2, "unresolved_failures": 0}


def test_a_torn_last_failure_line_is_skipped(tmp_path):
    run = ResumableRun(tmp_path, "r")
    run.fail("A.JPG", ValueError("x"))
    with open(run.failures_path, "a", encoding="utf-8") as f:
        f.write('{"image": "B.JP')  # a crash in the middle of an append
    assert [r["image"] for r in run.failures()] == ["A.JPG"]


def test_images_without_rows_are_done_without_a_part(tmp_path):
    run = ResumableRun(tmp_path, "r")
    run.add("EMPTY.JPG", None)
    run.add("ALSO.JPG", pd.DataFrame({"image_name": [], "coral_id": []}))
    assert run.flush() is None
    assert not list(tmp_path.glob("r.part_*"))
    assert ResumableRun(tmp_path, "r").done == {"EMPTY.JPG", "ALSO.JPG"}
    rows, matrix = run.merge()
    assert rows.empty and matrix is None


def test_a_crash_between_part_and_done_set_loses_nothing_and_orphans_are_ignored(
    tmp_path, monkeypatch
):
    run = ResumableRun(tmp_path, "r")
    run.add("A.JPG", _rows("A.JPG", 1))
    run.flush()

    crashing = ResumableRun(tmp_path, "r")
    crashing.add("B.JPG", _rows("B.JPG", 2))

    def boom(*args):
        raise OSError("disk went away")

    with monkeypatch.context() as patch:
        patch.setattr(crashing, "_save_state", boom)
        with pytest.raises(OSError):
            crashing.flush()
    assert (tmp_path / "r.part_0001.parquet").exists()  # the orphan
    assert crashing.done == {"A.JPG", "B.JPG"}  # still buffered, not lost
    retried = crashing.flush()  # the same process can simply retry
    assert retried == 2 and crashing.merge()[0]["image_name"].tolist() == [
        "A.JPG",
        "B.JPG",
        "B.JPG",
    ]
    (tmp_path / "r.part_0002.parquet").unlink()
    (tmp_path / "r.done.json").write_text(
        json.dumps(
            {"version": 1, "name": "r", "embeddings": False, "parts": [0], "done": {"A.JPG": 0}}
        )
    )  # rewind the files to the state right after the crash

    resumed = ResumableRun(tmp_path, "r")
    assert resumed.done == {"A.JPG"}  # B is not done: it will be redone
    resumed.add("B.JPG", _rows("B.JPG", 2))
    resumed.flush()
    assert (tmp_path / "r.part_0002.parquet").exists()  # the orphan is never overwritten
    rows, _ = resumed.merge()
    assert rows["image_name"].tolist() == ["A.JPG", "B.JPG", "B.JPG"]  # B once, not twice


def test_a_failed_write_leaves_no_temporary_files(tmp_path, monkeypatch):
    run = ResumableRun(tmp_path, "r")
    run.add("A.JPG", _rows("A.JPG", 1))

    def boom(self, *args, **kwargs):
        raise OSError("no space left")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError):
        run.flush()
    assert list(tmp_path.iterdir()) == []
    assert ResumableRun(tmp_path, "r").done == set()


def test_embeddings_must_match_the_rows_and_be_consistent(tmp_path):
    run = ResumableRun(tmp_path, "r")
    with pytest.raises(ValueError, match="3 embeddings for 2 rows"):
        run.add("A.JPG", _rows("A.JPG", 2), _emb(3))
    run.add("A.JPG", _rows("A.JPG", 2), _emb(2))
    with pytest.raises(ValueError, match="without them"):
        run.add("B.JPG", _rows("B.JPG", 1))
    plain = ResumableRun(tmp_path / "plain", "r")
    plain.add("A.JPG", _rows("A.JPG", 1))
    with pytest.raises(ValueError, match="with them"):
        plain.add("B.JPG", _rows("B.JPG", 1), _emb(1))


def test_part_embeddings_carry_uids(tmp_path):
    run = ResumableRun(tmp_path, "r")
    run.add("A.JPG", _rows("A.JPG", 2), _emb(2))
    run.flush()
    with np.load(tmp_path / "r.part_0000.npz") as data:
        assert data["uids"].tolist() == ["A.JPG#0", "A.JPG#1"]
        assert data["X"].shape == (2, 4)


def test_merge_writes_files_atomically_and_never_overwrites(tmp_path):
    run = ResumableRun(tmp_path / "work", "r")
    run.add("A.JPG", _rows("A.JPG", 2), _emb(2))
    out_parquet, out_npz = tmp_path / "out" / "all.parquet", tmp_path / "out" / "all.npz"
    run.merge(out_parquet, out_npz)
    assert pd.read_parquet(out_parquet)["coral_id"].tolist() == [0, 1]
    with np.load(out_npz) as data:
        assert data["uids"].tolist() == ["A.JPG#0", "A.JPG#1"] and data["X"].shape == (2, 4)
    assert not list((tmp_path / "out").glob(".*tmp*"))

    before = out_parquet.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run.merge(out_parquet)
    assert out_parquet.read_bytes() == before
    run.merge(out_parquet, overwrite=True)  # explicit opt-in

    plain = ResumableRun(tmp_path / "plain", "r")
    plain.add("A.JPG", _rows("A.JPG", 1))
    with pytest.raises(ValueError, match="no embeddings"):
        plain.merge(npz_path=tmp_path / "x.npz")


def test_a_part_whose_embeddings_do_not_match_is_detected(tmp_path):
    run = ResumableRun(tmp_path, "r")
    run.add("A.JPG", _rows("A.JPG", 2), _emb(2))
    run.flush()
    np.savez_compressed(tmp_path / "r.part_0000.npz", uids=np.array(["X#0", "X#1"]), X=_emb(2))
    with pytest.raises(RunStateError, match="uids do not match"):
        ResumableRun(tmp_path, "r").merge()


def test_unknown_state_version_and_bad_arguments(tmp_path):
    (tmp_path / "r.done.json").write_text(json.dumps({"version": 99}))
    with pytest.raises(RunStateError, match="state version"):
        ResumableRun(tmp_path, "r")
    with pytest.raises(ValueError):
        ResumableRun(tmp_path / "z", "r", flush_every=0)


def test_runs_with_different_names_share_a_folder(tmp_path):
    a, b = ResumableRun(tmp_path, "a"), ResumableRun(tmp_path, "b")
    a.add("X.JPG", _rows("X.JPG", 1))
    b.add("Y.JPG", _rows("Y.JPG", 2))
    a.flush(), b.flush()
    assert ResumableRun(tmp_path, "a").done == {"X.JPG"}
    assert ResumableRun(tmp_path, "b").done == {"Y.JPG"}
