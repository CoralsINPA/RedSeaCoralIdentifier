"""Which predictions a blind review is drawn from, and a check that scoring uses the same ones.

A blind review of a genus (`coralid.genus_audit`) must be about the 20-genus baseline:
`<run>/predictions/<model>__<prompts>__hard20__all.parquet`, found through `baseline_paths`.
`baseline_source` reads that file's `.meta.json` and **refuses** unless the label set is `hard20`
and the model and prompt set are the config's. The source it returns (the file, the git commit
and creation time of the baseline job, the model, the prompt set, the label set and the row
count) is stored in the review's meta and in the answer key's meta file. When the reviews come
back, `check_same_source` compares that record with the baseline as it is now and stops if the
baseline was made again or is another one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from coralid.baseline import baseline_paths
from coralid.config import Config, meta_path

REQUIRED_LABEL_SET = "hard20"
# What makes two sources the same baseline. The folder is left out: the same Drive can be
# mounted at another path.
IDENTITY = ("file", "git_commit", "created_utc", "model", "prompt_set", "label_set", "n_rows")


class ReviewSourceError(RuntimeError):
    """The predictions behind a blind review are not the 20-genus baseline, or not the same one."""


@dataclass(frozen=True)
class PredictionSource:
    """The predictions file a review was drawn from, as its own meta file describes it."""

    file: str
    path: str
    git_commit: str | None
    created_utc: str | None
    model: str
    prompt_set: str
    label_set: str
    n_rows: int | None

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly form."""
        return asdict(self)

    def lines(self) -> list[str]:
        """The source as short lines for a notebook or a terminal."""
        return [
            f"predictions: {self.path}",
            f"  model {self.model} | prompt set {self.prompt_set} | label set {self.label_set} "
            f"| {self.n_rows} crops",
            f"  made by commit {self.git_commit} at {self.created_utc}",
        ]


def baseline_source(cfg: Config) -> PredictionSource:
    """The saved 20-genus baseline of the config, or `ReviewSourceError` saying why not."""
    predictions = baseline_paths(cfg).predictions
    meta_file = meta_path(predictions)
    if not predictions.exists():
        raise ReviewSourceError(
            f"{predictions} not found: run the 20-genus baseline first (python -m coralid "
            "zero-shot, or notebooks/02_zero_shot.ipynb)."
        )
    if not meta_file.exists():
        raise ReviewSourceError(
            f"{meta_file} not found: without it nobody can tell which model, prompts and label "
            "set made these predictions, so no review is drawn from them."
        )
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    zero_shot = cfg.zero_shot
    problems = []
    for key, wanted in (
        ("label_set", REQUIRED_LABEL_SET),
        ("model", zero_shot.model),
        ("prompt_set", zero_shot.prompt_set),
    ):
        if meta.get(key) != wanted:
            problems.append(f"{key} is {meta.get(key)!r}, it must be {wanted!r}")
    if problems:
        raise ReviewSourceError(
            f"{predictions.name} is not the baseline this config describes "
            f"({'; '.join(problems)}). A blind review is drawn only from the 20-genus baseline."
        )
    return PredictionSource(
        file=predictions.name,
        path=str(predictions),
        git_commit=meta.get("git_commit"),
        created_utc=meta.get("created_utc"),
        model=meta["model"],
        prompt_set=meta["prompt_set"],
        label_set=meta["label_set"],
        n_rows=meta.get("n_rows"),
    )


def check_same_source(
    recorded: Mapping[str, Any] | None, current: PredictionSource | Mapping[str, Any]
) -> None:
    """Raise `ReviewSourceError` unless the key's recorded source is the current baseline."""
    if not recorded:
        raise ReviewSourceError(
            "The answer key records no source for its predictions, so it cannot be scored "
            "against the baseline. Export the review again with this version of coralid."
        )
    now = current.to_dict() if isinstance(current, PredictionSource) else dict(current)
    differs = [
        f"{key}: the key says {recorded.get(key)!r}, the baseline now says {now.get(key)!r}"
        for key in IDENTITY
        if recorded.get(key) != now.get(key)
    ]
    if differs:
        raise ReviewSourceError(
            "The review was drawn from other predictions than the current baseline ("
            + "; ".join(differs)
            + "). Score it against the baseline it was drawn from, or review again."
        )
