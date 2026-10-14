import os

import pytest

from coralid import _atomic
from coralid._atomic import atomic_write_text, atomic_write_text_exclusive


def test_exclusive_write_creates_the_file(tmp_path):
    out = atomic_write_text_exclusive(tmp_path / "sub" / "a.json", "hello")
    assert out.read_text() == "hello"
    assert [p.name for p in out.parent.iterdir()] == ["a.json"]  # the temp file is gone


def test_exclusive_write_never_replaces_an_existing_file(tmp_path):
    path = tmp_path / "a.json"
    path.write_text("original")
    with pytest.raises(FileExistsError):
        atomic_write_text_exclusive(path, "new")
    assert path.read_text() == "original"
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_exclusive_write_loses_a_race_without_overwriting(tmp_path, monkeypatch):
    """Another process creates the file after the first check: the link step must refuse."""
    path = tmp_path / "a.json"
    real_link = os.link

    def link_after_a_rival_wrote(src, dst, *args, **kwargs):
        path.write_text("the rival")
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(_atomic.os, "link", link_after_a_rival_wrote)
    with pytest.raises(FileExistsError):
        atomic_write_text_exclusive(path, "mine")
    assert path.read_text() == "the rival"
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_exclusive_write_falls_back_where_hard_links_are_unsupported(tmp_path, monkeypatch):
    def no_links(*args, **kwargs):
        raise PermissionError("hard links are not supported here")

    monkeypatch.setattr(_atomic.os, "link", no_links)
    path = tmp_path / "a.json"
    atomic_write_text_exclusive(path, "mine")
    assert path.read_text() == "mine"
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]
    path.write_text("rival")  # the fallback still refuses an existing file
    with pytest.raises(FileExistsError):
        atomic_write_text_exclusive(path, "again")
    assert path.read_text() == "rival"


def test_exclusive_write_leaves_no_temp_file_when_writing_fails(tmp_path):
    with pytest.raises(UnicodeEncodeError):
        atomic_write_text_exclusive(tmp_path / "a.json", "café", encoding="ascii")
    assert list(tmp_path.iterdir()) == []


def test_the_temp_file_of_an_exclusive_write_is_not_a_json_file(tmp_path, monkeypatch):
    """A crash between the write and the rename must not leave something a `*.json` scan reads."""
    seen = []
    real_link = os.link

    def spy(src, dst, *args, **kwargs):
        seen.extend(p.name for p in tmp_path.iterdir())
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(_atomic.os, "link", spy)
    atomic_write_text_exclusive(tmp_path / "IMG_1.json", "{}")
    temp = [n for n in seen if n != "IMG_1.json"]
    assert temp and not any(n.endswith(".json") for n in temp)


def test_plain_atomic_write_still_replaces(tmp_path):
    atomic_write_text(tmp_path / "a.txt", "one")
    atomic_write_text(tmp_path / "a.txt", "two")
    assert (tmp_path / "a.txt").read_text() == "two"
