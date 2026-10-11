"""Masks: RLE encode/decode, mask-JSON read/write, coral keys, bbox parsing and lookup.

Conventions (see docs/organization_plan.md section 3.4):
- Every mask lives in the frame stored in its JSON (`image.width` x `image.height`).
- A coral is identified by `(image_name, coral_id)`, written `IMG_4033.JPG#14`.
  `coral_id` restarts at 0 in every image, so it is never a key on its own.
"""

from __future__ import annotations

import json
import math
import operator
import os
from collections import Counter, OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pycocotools.mask as mask_util

from coralid._atomic import atomic_write_text

Rle = dict[str, Any]
Annotation = dict[str, Any]
Bbox = tuple[float, float, float, float]


class MaskJsonError(ValueError):
    """Raised when a mask JSON file is malformed."""


# --------------------------------------------------------------------------- RLE


def rle_to_bytes(rle: Rle) -> Rle:
    """Return a compressed RLE whose `counts` is bytes (what pycocotools expects)."""
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    return {"size": list(rle["size"]), "counts": counts}


def decode_rle(rle: Rle) -> np.ndarray:
    """Decode a compressed RLE (str or bytes `counts`) into a boolean (H, W) mask."""
    return mask_util.decode(rle_to_bytes(rle)).astype(bool)


def decode_segmentation(seg: Any, height: int, width: int) -> np.ndarray:
    """Decode any COCO segmentation into a boolean (H, W) mask.

    Accepts compressed RLE, uncompressed RLE (list `counts`) and polygons, as the
    legacy masked-square-crop cell did. `height` and `width` size polygons and
    uncompressed RLE.
    """
    if isinstance(seg, list):  # polygons
        mask = mask_util.decode(mask_util.merge(mask_util.frPyObjects(seg, height, width)))
    elif isinstance(seg, dict):
        if isinstance(seg["counts"], list):  # uncompressed RLE
            mask = mask_util.decode(mask_util.frPyObjects(seg, height, width))
        else:
            mask = mask_util.decode(rle_to_bytes(seg))
    else:
        raise TypeError(f"Unsupported segmentation type: {type(seg).__name__}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(bool)


def encode_mask(mask: np.ndarray) -> Rle:
    """Encode a 2-D mask as `{"size": [h, w], "counts": <str>}`, as stored in the mask JSONs."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {mask.shape}")
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": str(counts)}


# --------------------------------------------------------------------------- keys


def as_int(value: Any) -> int:
    """Convert an integer-like value (int, numpy int, 14.0, "14") to int, else raise."""
    try:
        return operator.index(value)
    except TypeError:
        pass
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"coral_id must be an integer, got {value!r}")
    return int(number)


@dataclass(frozen=True, order=True)
class CoralKey:
    """The identity of one coral: the image it is in and its id within that image."""

    image_name: str
    coral_id: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "coral_id", as_int(self.coral_id))

    @property
    def uid(self) -> str:
        """The string form, e.g. `IMG_4033.JPG#14`."""
        return f"{self.image_name}#{self.coral_id}"

    @classmethod
    def from_uid(cls, uid: str) -> CoralKey:
        """Parse `IMG_4033.JPG#14` back into a key."""
        name, sep, cid = uid.rpartition("#")
        if not sep or not name:
            raise ValueError(f"Not a coral uid: {uid!r}")
        return cls(name, as_int(cid))


def parse_bbox(value: Any) -> Bbox:
    """Parse an `[x, y, w, h]` box without `eval`.

    Accepts lists, tuples and arrays, `"[x, y, w, h]"`, numpy-style `"[x y w h]"` and
    float strings such as `"[1.5e2, 3.0, 400.0, 250.5]"`.
    """
    if isinstance(value, str):
        text = value.strip().strip("[]()").replace(",", " ")
        parts: Sequence[Any] = text.split()
    else:
        try:
            parts = list(value)
        except TypeError as err:
            raise ValueError(f"Cannot parse bbox from {value!r}") from err
    if len(parts) != 4:
        raise ValueError(f"A bbox needs 4 numbers, got {len(parts)} in {value!r}")
    try:
        numbers = tuple(float(p) for p in parts)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Cannot parse bbox from {value!r}") from err
    x, y, w, h = numbers
    if not all(math.isfinite(n) for n in numbers):
        raise ValueError(f"bbox must be finite, got {value!r}")
    return (x, y, w, h)


# --------------------------------------------------------------------------- JSON files


@dataclass
class MaskFile:
    """One image's mask JSON: its frame, its annotations and an optional failure note."""

    image_name: str
    width: int
    height: int
    annotations: list[Annotation]
    note: str | None = None
    path: Path | None = None
    _by_id: dict[int, Annotation] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def frame(self) -> tuple[int, int]:
        """The mask frame as `(width, height)`."""
        return (self.width, self.height)

    def by_id(self) -> dict[int, Annotation]:
        """Annotations keyed by `id` (built on first use)."""
        if self._by_id is None:
            self._by_id = {int(a["id"]): a for a in self.annotations}
        return self._by_id


def json_name_for(image_name: str) -> str:
    """The mask JSON file name for an image: its stem plus `.json`."""
    return os.path.splitext(image_name)[0] + ".json"


def list_mask_files(json_dir: str | os.PathLike[str]) -> list[Path]:
    """List the `*.json` files of a run's `json/` folder, sorted by name."""
    root = Path(json_dir)
    return [root / name for name in sorted(os.listdir(root)) if name.endswith(".json")]


def load_mask_json(path: str | os.PathLike[str]) -> MaskFile:
    """Load one mask JSON, checking its structure and that annotation ids are unique."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    try:
        info = data["image"]
        width, height = int(info["width"]), int(info["height"])
        image_name = str(info["file_name"])
        annotations = list(data["annotations"])
        ids = [int(a["id"]) for a in annotations]
    except (KeyError, TypeError, ValueError, AttributeError) as err:
        raise MaskJsonError(f"{path}: malformed mask JSON ({err!r})") from err
    if len(set(ids)) != len(ids):
        raise MaskJsonError(f"{path}: annotation ids are not unique")
    return MaskFile(
        image_name=image_name,
        width=width,
        height=height,
        annotations=annotations,
        note=data.get("note"),
        path=path,
    )


def write_mask_json(
    path: str | os.PathLike[str],
    image_name: str,
    width: int,
    height: int,
    annotations: Iterable[Annotation],
    note: str | None = None,
) -> Path:
    """Write a mask JSON in the legacy layout (`image`, `annotations`, optional `note`)."""
    out: dict[str, Any] = {
        "image": {"image_id": 0, "width": width, "height": height, "file_name": image_name},
        "annotations": list(annotations),
    }
    if note is not None:
        out["note"] = note
    return atomic_write_text(path, json.dumps(out, indent=2))


class AnnotationStore:
    """Per-image lazy access to a run's mask JSONs, with a small LRU cache.

    Replaces rescanning the JSON folder for every crop: `store.get(key)` loads the image's
    JSON once and keeps it while recently used.
    """

    def __init__(self, json_dir: str | os.PathLike[str], cache_size: int = 32) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be at least 1")
        self.json_dir = Path(json_dir)
        self.cache_size = cache_size
        self._cache: OrderedDict[str, MaskFile] = OrderedDict()
        self.loads = 0  # how many JSON files were actually read (cache misses)

    def path_for(self, image_name: str) -> Path:
        """The JSON path for an image name (its stem plus `.json`)."""
        return self.json_dir / json_name_for(image_name)

    def load(self, image_name: str) -> MaskFile:
        """Return the image's `MaskFile`, reading it if it is not cached."""
        cached = self._cache.get(image_name)
        if cached is not None:
            self._cache.move_to_end(image_name)
            return cached
        mask_file = load_mask_json(self.path_for(image_name))
        self.loads += 1
        self._cache[image_name] = mask_file
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return mask_file

    def annotations(self, image_name: str) -> list[Annotation]:
        """All annotations of an image, in generator order."""
        return self.load(image_name).annotations

    def frame(self, image_name: str) -> tuple[int, int]:
        """The mask frame of an image as `(width, height)`."""
        return self.load(image_name).frame

    def get(self, key: CoralKey | str) -> Annotation:
        """Return the annotation for a `CoralKey` (or its uid string); `KeyError` if absent."""
        if isinstance(key, str):
            key = CoralKey.from_uid(key)
        try:
            mask_file = self.load(key.image_name)
        except FileNotFoundError as err:
            raise KeyError(key.uid) from err
        try:
            return mask_file.by_id()[key.coral_id]
        except KeyError:
            raise KeyError(key.uid) from None


# --------------------------------------------------------------------------- audit


@dataclass
class JsonAudit:
    """What a mask JSON folder holds: frames, masks, failed images and unreadable files."""

    n_files: int = 0
    n_masks: int = 0
    frame_sizes: dict[tuple[int, int], int] = field(default_factory=dict)  # (w, h) -> files
    noted: list[tuple[str, str]] = field(default_factory=list)  # (json name, note)
    name_mismatches: list[tuple[str, str]] = field(default_factory=list)  # (json, file_name)
    unreadable: list[tuple[str, str]] = field(default_factory=list)  # (json name, error)

    def as_dict(self) -> dict[str, Any]:
        """A JSON-friendly summary (frame sizes keyed as `"WxH"`)."""
        return {
            "n_files": self.n_files,
            "n_masks": self.n_masks,
            "frame_sizes": {f"{w}x{h}": n for (w, h), n in self.frame_sizes.items()},
            "noted": [list(item) for item in self.noted],
            "name_mismatches": [list(item) for item in self.name_mismatches],
            "unreadable": [list(item) for item in self.unreadable],
        }


def audit_json_dir(json_dir: str | os.PathLike[str], progress: bool = False) -> JsonAudit:
    """Count frame sizes, masks and files, and list JSONs with a `note` field.

    A `note` marks an image whose segmentation failed and was saved as empty (CM cell 8).
    JSONs whose `image.file_name` does not match their own file name are listed too, since
    `AnnotationStore` finds a JSON from the image name.
    """
    audit = JsonAudit()
    sizes: Counter[tuple[int, int]] = Counter()
    files: Iterable[Path] = list_mask_files(json_dir)
    if progress:
        from tqdm.auto import tqdm

        files = tqdm(list(files), desc="Auditing mask JSONs")
    for path in files:
        audit.n_files += 1
        try:
            mask_file = load_mask_json(path)
        except (OSError, ValueError) as err:
            audit.unreadable.append((path.name, str(err)))
            continue
        audit.n_masks += len(mask_file.annotations)
        sizes[mask_file.frame] += 1
        if mask_file.note is not None:
            audit.noted.append((path.name, str(mask_file.note)))
        if json_name_for(mask_file.image_name) != path.name:
            audit.name_mismatches.append((path.name, mask_file.image_name))
    audit.frame_sizes = dict(sizes.most_common())
    return audit
