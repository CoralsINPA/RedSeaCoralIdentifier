"""Label sets and prompt sets (`configs/genera.yaml`) and the prompt formats built from them.

A *label set* is an ordered list of genera (`hard20`). Its order is the column order of every
prediction table and decides argmax ties. A *prompt set* is one text prompt per genus, copied
character for character from a legacy notebook. A *prompt format* says how prompt sets turn
into text features:

- `taxonomy`: one taxonomy prompt set (`taxonomy_v1` by default, or `taxonomy_v2`);
- `descriptive`, `common`, `hybrid`: the prompt set of that name;
- `ensemble`: the mean of the L2-normalized taxonomy, descriptive and common embeddings,
  renormalized.

SigLIP quirks of the legacy notebook are kept: its `ensemble` is the mean of descriptive and
common only, and its `taxonomy` is silently replaced by `descriptive` (with a warning).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PROMPT_FORMATS = ("taxonomy", "descriptive", "common", "hybrid", "ensemble")
FAMILIES = ("open_clip", "siglip")
KIND_TAXONOMY = "taxonomy"


class GeneraError(ValueError):
    """Raised for an unknown label set or prompt set, or a prompt set that does not fit."""


@dataclass(frozen=True)
class PromptSet:
    """One text prompt per genus, from one legacy source."""

    name: str
    kind: str  # taxonomy, descriptive, common or hybrid
    source: str
    prompts: dict[str, str]
    incomplete: bool = False

    def prompts_for(self, genera: list[str], label_set: str = "") -> list[str]:
        """Return the prompts for `genera` in that order, or raise if any genus has none."""
        missing = [g for g in genera if g not in self.prompts]
        if missing:
            where = f" in label set '{label_set}'" if label_set else ""
            note = " The set is marked incomplete." if self.incomplete else ""
            raise GeneraError(
                f"Prompt set '{self.name}' has prompts for {len(genera) - len(missing)} of the "
                f"{len(genera)} genera{where} (missing: {', '.join(missing)}).{note} "
                "Use a complete prompt set."
            )
        return [self.prompts[g] for g in genera]


@dataclass(frozen=True)
class GeneraConfig:
    """The contents of `genera.yaml`."""

    label_sets: dict[str, tuple[str, ...]]
    prompt_sets: dict[str, PromptSet]
    soft_genera: tuple[str, ...] = field(default_factory=tuple)

    def genera(self, label_set: str) -> list[str]:
        """The ordered genera of a label set."""
        try:
            return list(self.label_sets[label_set])
        except KeyError:
            raise GeneraError(
                f"Unknown label set '{label_set}'; available: {sorted(self.label_sets)}"
            ) from None

    def prompt_set(self, name: str) -> PromptSet:
        """A prompt set by name."""
        try:
            return self.prompt_sets[name]
        except KeyError:
            raise GeneraError(
                f"Unknown prompt set '{name}'; available: {sorted(self.prompt_sets)}"
            ) from None

    def known_genera(self) -> list[str]:
        """Every genus the project knows: all label sets, then the soft corals (no repeats)."""
        seen: dict[str, None] = {}
        for genera in self.label_sets.values():
            seen.update(dict.fromkeys(genera))
        seen.update(dict.fromkeys(self.soft_genera))
        return list(seen)


def default_genera_path() -> Path:
    """The `configs/genera.yaml` of the clone this package was installed from (`pip install -e`)."""
    return Path(__file__).resolve().parents[2] / "configs" / "genera.yaml"


def _parse(data: dict[str, Any]) -> GeneraConfig:
    label_sets = {name: tuple(spec["genera"]) for name, spec in data["label_sets"].items()}
    prompt_sets = {
        name: PromptSet(
            name=name,
            kind=spec["kind"],
            source=spec.get("source", ""),
            prompts=dict(spec["prompts"]),
            incomplete=bool(spec.get("incomplete", False)),
        )
        for name, spec in data["prompt_sets"].items()
    }
    return GeneraConfig(label_sets, prompt_sets, tuple(data.get("soft_genera") or ()))


@lru_cache(maxsize=8)
def _load(path: str) -> GeneraConfig:
    with open(path, encoding="utf-8") as f:
        return _parse(yaml.safe_load(f))


def load_genera(path: str | os.PathLike[str] | None = None) -> GeneraConfig:
    """Load `genera.yaml` (the repository's by default). Loaded files are cached."""
    target = Path(path) if path is not None else default_genera_path()
    if not target.is_file():
        raise FileNotFoundError(
            f"{target} not found. configs/ is not part of the installed package: install from "
            "a clone (pip install -e) or pass the path to genera.yaml."
        )
    return _load(str(target.resolve()))


def prompt_set_names(
    prompt_format: str, prompt_set: str = "taxonomy_v1", family: str = "open_clip"
) -> list[str]:
    """Names of the prompt sets whose text features are averaged for a prompt format."""
    if prompt_format not in PROMPT_FORMATS:
        raise GeneraError(f"Unknown prompt format '{prompt_format}'; use one of {PROMPT_FORMATS}")
    if family not in FAMILIES:
        raise GeneraError(f"Unknown encoder family '{family}'; use one of {FAMILIES}")
    if prompt_format == "taxonomy":
        if family == "siglip":
            logger.warning("taxonomy format not recommended for SigLIP 2: switching to descriptive")
            return ["descriptive"]
        return [prompt_set]
    if prompt_format == "ensemble":
        if family == "siglip":
            return ["descriptive", "common"]  # SigLIP skips taxonomy: a 2-way ensemble
        return [prompt_set, "descriptive", "common"]
    return [prompt_format]  # descriptive, common, hybrid: the set of the same name


def build_prompts(
    label_set: str,
    prompt_format: str,
    prompt_set: str = "taxonomy_v1",
    family: str = "open_clip",
    config: GeneraConfig | None = None,
) -> list[list[str]]:
    """Return one prompt list (in label-set order) per prompt set the format averages.

    `prompt_set` names the taxonomy variant for the `taxonomy` and `ensemble` formats; the
    other formats use the set of their own name and ignore it. A prompt set that does not cover
    the label set (the incomplete `hybrid` against `hard20`) raises `GeneraError`.
    """
    config = config or load_genera()
    genera = config.genera(label_set)
    names = prompt_set_names(prompt_format, prompt_set, family)
    if family == "open_clip" and prompt_format in ("taxonomy", "ensemble"):
        kind = config.prompt_set(prompt_set).kind
        if kind != KIND_TAXONOMY:
            raise GeneraError(
                f"Prompt set '{prompt_set}' is a {kind} set, but format '{prompt_format}' needs "
                "a taxonomy set (taxonomy_v1 or taxonomy_v2)."
            )
    return [config.prompt_set(name).prompts_for(genera, label_set) for name in names]
