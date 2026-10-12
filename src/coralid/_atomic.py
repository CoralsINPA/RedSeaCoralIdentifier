"""Atomic file writes: write to a temporary sibling, then rename over the target."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path


@contextlib.contextmanager
def atomic_path(path: str | os.PathLike[str]) -> Iterator[Path]:
    """Yield a temporary path next to `path` (same suffix) and rename it on success.

    The temporary file keeps the real suffix, so writers that infer the format from the
    extension (parquet, npz) behave the same. If the body raises, the temporary file is
    removed and the target is left untouched.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.stem}.tmp{os.getpid()}{target.suffix}")
    try:
        yield tmp
        os.replace(tmp, target)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def atomic_write_text(path: str | os.PathLike[str], text: str, encoding: str = "utf-8") -> Path:
    """Write `text` to `path` atomically and return the path."""
    with atomic_path(path) as tmp:
        tmp.write_text(text, encoding=encoding)
    return Path(path)


def atomic_write_text_exclusive(
    path: str | os.PathLike[str], text: str, encoding: str = "utf-8"
) -> Path:
    """Create `path` with `text` atomically, and never replace an existing file.

    The text is written to a temporary sibling first, so a crash leaves either no file or a
    complete one. The final step is a hard link, which fails with `FileExistsError` if `path`
    exists (even if another process created it a moment ago). Where hard links are not
    supported (some network and FUSE drives) it falls back to a second existence check and a
    rename. The temporary file keeps no `.json`-style suffix, so a leftover can never be mistaken
    for a real file in a folder that is scanned by extension.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"{target} exists; refusing to overwrite it")
    tmp = target.with_name(f".{target.name}.tmp{os.getpid()}")
    try:
        tmp.write_text(text, encoding=encoding)
        try:
            os.link(tmp, target)
        except FileExistsError:
            raise FileExistsError(f"{target} exists; refusing to overwrite it") from None
        except OSError:
            if target.exists():
                raise FileExistsError(f"{target} exists; refusing to overwrite it") from None
            os.replace(tmp, target)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
    return target
