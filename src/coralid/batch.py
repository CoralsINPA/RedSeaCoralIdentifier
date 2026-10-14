"""A resumable per-image runner: done-set, numbered part files, a failure log and a merge step.

Progress is saved as it goes and a run can be resumed, keyed by image:

- `add(image, rows, embeddings)` buffers one image's result. Every `flush_every` images the
  buffer is written as a numbered part (`<name>.part_0003.parquet`, plus `.npz` with `uids`
  and `X` when embeddings are given).
- The done-set lives in `<name>.done.json`. It is rewritten only **after** the part files of
  the images it lists are on disk, so a crash can orphan a part but never lose a done image.
  A resume skips exactly the images in that file and ignores orphan parts.
- `fail(image, error)` appends to `<name>.failures.jsonl` and does **not** mark the image done,
  so the next resume retries it.
- All writes go to a temporary file first and are renamed into place.

Nothing here deletes or replaces an existing file except its own done-set, and `merge` refuses
to overwrite its output unless asked to.
"""

from __future__ import annotations

import json
import logging
import os
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coralid._atomic import atomic_path, atomic_write_text

logger = logging.getLogger(__name__)

STATE_VERSION = 1


class RunStateError(RuntimeError):
    """Raised when the files of a run contradict each other."""


def _uids(df: pd.DataFrame) -> list[str]:
    return [f"{n}#{int(c)}" for n, c in zip(df["image_name"], df["coral_id"], strict=True)]


class ResumableRun:
    """Process many images, survive interruptions, and merge the parts at the end."""

    def __init__(self, out_dir: str | os.PathLike[str], name: str, flush_every: int = 25) -> None:
        if flush_every < 1:
            raise ValueError("flush_every must be at least 1")
        self.out_dir = Path(out_dir)
        self.name = name
        self.flush_every = flush_every
        self.done_path = self.out_dir / f"{name}.done.json"
        self.failures_path = self.out_dir / f"{name}.failures.jsonl"
        self._done: dict[str, int | None] = {}  # image -> part number (None: no rows)
        self._parts: list[int] = []
        self._embeddings: bool | None = None
        self._buffer: list[tuple[str, pd.DataFrame | None, np.ndarray | None]] = []
        self._buffered: set[str] = set()
        self._load_state()

    # ------------------------------------------------------------------ state

    def _load_state(self) -> None:
        if not self.done_path.exists():
            return
        with open(self.done_path, encoding="utf-8") as f:
            state = json.load(f)
        if state.get("version") != STATE_VERSION:
            raise RunStateError(f"{self.done_path}: unknown state version {state.get('version')}")
        self._done = dict(state["done"])
        self._parts = [int(p) for p in state["parts"]]
        self._embeddings = state.get("embeddings")
        logger.info(
            "Resuming run '%s': %d image(s) done in %d part(s)",
            self.name,
            len(self._done),
            len(self._parts),
        )

    def _save_state(self, parts: list[int], done: dict[str, int | None]) -> None:
        state = {
            "version": STATE_VERSION,
            "name": self.name,
            "embeddings": self._embeddings,
            "parts": parts,
            "done": done,
        }
        atomic_write_text(self.done_path, json.dumps(state, indent=1))

    def part_path(self, number: int, suffix: str) -> Path:
        """Path of part `number`; `suffix` is `parquet` or `npz`."""
        return self.out_dir / f"{self.name}.part_{number:04d}.{suffix}"

    def _next_part_number(self) -> int:
        """One more than any part on disk or in the state, so an orphan is never overwritten."""
        numbers = list(self._parts)
        pattern = re.compile(rf"^{re.escape(self.name)}\.part_(\d+)\.(parquet|npz)$")
        if self.out_dir.is_dir():
            for entry in os.listdir(self.out_dir):
                match = pattern.match(entry)
                if match:
                    numbers.append(int(match.group(1)))
        return max(numbers, default=-1) + 1

    # ------------------------------------------------------------------ progress

    @property
    def done(self) -> set[str]:
        """Images that are done (written, or buffered in this process)."""
        return set(self._done) | self._buffered

    def pending(self, image_names: list[str]) -> list[str]:
        """The images of `image_names` that are not done yet, in the same order."""
        done = self.done
        return [n for n in image_names if n not in done]

    def summary(self) -> dict[str, int]:
        """Counts for a progress line."""
        return {
            "done": len(self.done),
            "parts": len(self._parts),
            "unresolved_failures": len(self.unresolved_failures()),
        }

    # ------------------------------------------------------------------ adding results

    def add(
        self,
        image_name: str,
        rows: pd.DataFrame | None,
        embeddings: np.ndarray | None = None,
    ) -> None:
        """Record the result of one image (`rows` may be None or empty: nothing to keep).

        `rows` needs `image_name` and `coral_id` columns when embeddings are given.
        """
        if image_name in self.done:
            raise ValueError(f"Image already recorded in run '{self.name}': {image_name}")
        n_rows = 0 if rows is None else len(rows)
        if embeddings is not None and len(embeddings) != n_rows:
            raise ValueError(f"{image_name}: {len(embeddings)} embeddings for {n_rows} rows")
        if n_rows:
            has_embeddings = embeddings is not None
            if self._embeddings is None:
                self._embeddings = has_embeddings
            elif self._embeddings != has_embeddings:
                raise ValueError(
                    f"Run '{self.name}' stores embeddings: {self._embeddings}; "
                    f"{image_name} came {'with' if has_embeddings else 'without'} them"
                )
        keep_rows = rows if n_rows else None
        self._buffer.append((image_name, keep_rows, embeddings if n_rows else None))
        self._buffered.add(image_name)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def fail(self, image_name: str, error: BaseException) -> None:
        """Log a failed image. It is not marked done, so the next resume retries it."""
        record = {
            "image": image_name,
            "error": f"{type(error).__name__}: {error}",
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
            "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.failures_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.warning("Image failed (will be retried on resume): %s: %s", image_name, error)

    def failures(self) -> list[dict[str, Any]]:
        """Every logged failure, oldest first. A torn last line from a crash is skipped."""
        if not self.failures_path.exists():
            return []
        records = []
        with open(self.failures_path, encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping an unreadable line in %s", self.failures_path)
        return records

    def unresolved_failures(self) -> list[dict[str, Any]]:
        """Logged failures of images that are still not done (the last one per image)."""
        done = self.done
        latest = {r["image"]: r for r in self.failures() if r.get("image") not in done}
        return list(latest.values())

    # ------------------------------------------------------------------ writing parts

    def flush(self) -> int | None:
        """Write the buffered images as the next part and mark them done.

        Returns the part number, or None if no buffered image had rows (they are still marked
        done) or the buffer was empty.
        """
        if not self._buffer:
            return None
        frames = [rows for _, rows, _ in self._buffer if rows is not None]
        number: int | None = None
        if frames:
            number = self._next_part_number()
            part = pd.concat(frames, ignore_index=True)
            with atomic_path(self.part_path(number, "parquet")) as tmp:
                part.to_parquet(tmp, index=False)
            if self._embeddings:
                matrix = np.vstack([x for _, rows, x in self._buffer if rows is not None])
                with atomic_path(self.part_path(number, "npz")) as tmp:
                    np.savez_compressed(tmp, uids=np.array(_uids(part)), X=matrix)
        parts = [*self._parts, number] if number is not None else list(self._parts)
        done = dict(self._done)
        for image_name, rows, _ in self._buffer:
            done[image_name] = number if rows is not None else None
        self._save_state(parts, done)  # if this raises, nothing below changed: flush can be retried
        self._parts, self._done = parts, done
        logger.info("Run '%s': wrote part %s, %d image(s) done", self.name, number, len(self._done))
        self._buffer = []
        self._buffered = set()
        return number

    # ------------------------------------------------------------------ merging

    def merge(
        self,
        parquet_path: str | os.PathLike[str] | None = None,
        npz_path: str | os.PathLike[str] | None = None,
        overwrite: bool = False,
    ) -> tuple[pd.DataFrame, np.ndarray | None]:
        """Concatenate all parts (flushing the buffer first); optionally write the result.

        Returns `(rows, embeddings)`; embeddings are None for a run without them. The output
        files are written atomically and an existing file is an error unless `overwrite`.
        """
        self.flush()
        frames, matrices = [], []
        for number in self._parts:
            frame = pd.read_parquet(self.part_path(number, "parquet"))
            frames.append(frame)
            if self._embeddings:
                with np.load(self.part_path(number, "npz")) as data:
                    if list(data["uids"]) != _uids(frame):
                        raise RunStateError(f"Part {number}: embedding uids do not match the rows")
                    matrices.append(data["X"])
        rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        matrix = np.vstack(matrices) if matrices else None
        if parquet_path is not None:
            _guard(parquet_path, overwrite)
            with atomic_path(parquet_path) as tmp:
                rows.to_parquet(tmp, index=False)
        if npz_path is not None:
            if matrix is None:
                raise ValueError("This run has no embeddings to write")
            _guard(npz_path, overwrite)
            with atomic_path(npz_path) as tmp:
                np.savez_compressed(tmp, uids=np.array(_uids(rows)), X=matrix)
        return rows, matrix


def _guard(path: str | os.PathLike[str], overwrite: bool) -> None:
    if Path(path).exists() and not overwrite:
        raise FileExistsError(f"{path} exists; refusing to overwrite it")
