"""Summary table and saved report of the parity checks (rows come from `coralid.parity`)."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from coralid.config import git_commit
from coralid.parity import FAIL, INFO, NOT_RUN, PASS, CheckResult

PARTIAL = "PARTIAL"

CHECKS = {
    "A": "geometric filters on all mask JSONs",
    "B": "zero-shot parity with the legacy code on the 50-image sample (landscape photos)",
    "C": "image manifest on all images (completeness)",
    "D": "predictions stored in the expert set against its expert labels",
}


# --------------------------------------------------------------------------- summary


def summarize(results: Sequence[CheckResult]) -> dict[str, Any]:
    """Fold rows into per-check statuses.

    FAIL if any row fails; PARTIAL if some row was NOT RUN; PASS if some pass; else INFO.
    """
    checks: dict[str, Any] = {}
    for letter, title in CHECKS.items():
        rows = [r for r in results if r.check == letter]
        if not rows:
            status = NOT_RUN
        elif any(r.status == FAIL for r in rows):
            status = FAIL
        elif any(r.status == NOT_RUN for r in rows):
            status = PARTIAL
        elif any(r.status == PASS for r in rows):
            status = PASS
        else:
            status = INFO
        checks[letter] = {"title": title, "status": status, "rows": [asdict(r) for r in rows]}
    statuses = [c["status"] for c in checks.values()]
    if FAIL in statuses:
        overall = FAIL
    elif all(s == NOT_RUN for s in statuses):
        overall = NOT_RUN
    elif NOT_RUN in statuses or PARTIAL in statuses:
        overall = PARTIAL
    else:
        overall = PASS
    return {"overall": overall, "checks": checks}


def format_table(results: Sequence[CheckResult], show_cells: bool = False) -> str:
    """A plain-text table of rows (confusion cells only if `show_cells` or they fail)."""
    lines = []
    for r in results:
        if r.kind == "cell" and r.status == PASS and not show_cells:
            continue
        lines.append(f"[{r.status:>7}] {r.name}")
        if r.status != PASS or r.kind == "check":
            lines.append(f"          expected: {r.expected!r}   actual: {r.actual!r}")
            if r.detail:
                lines.append(f"          {r.detail}")
    return "\n".join(lines)


def format_summary(summary: Mapping[str, Any]) -> str:
    """The short table the user pastes back: one line per check, then every failing row."""
    lines = [f"PARITY SUMMARY: {summary['overall']}"]
    for letter, check in summary["checks"].items():
        lines.append(f"  {letter}  {check['status']:<8} {check['title']}")
    for check in summary["checks"].values():
        for row in check["rows"]:
            if row["status"] == FAIL and row["kind"] == "check":
                lines.append(
                    f"  FAILED {row['name']}: expected {row['expected']!r}, got {row['actual']!r}"
                )
    return "\n".join(lines)


def paste_block(summary: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> str:
    """The compact JSON the user pastes into the next session.

    One entry per row with its expected and actual value (and its detail, when it has one);
    the per-cell confusion rows are left out.
    """
    body = {
        "overall": summary["overall"],
        **(extra or {}),
        "checks": {
            letter: {
                "status": check["status"],
                "rows": [
                    {
                        k: row[k]
                        for k in ("name", "status", "expected", "actual", "detail")
                        if row[k] != ""
                    }
                    for row in check["rows"]
                    if row["kind"] == "check"
                ],
            }
            for letter, check in summary["checks"].items()
        },
    }
    return json.dumps(body, indent=1, default=str)


def environment_info() -> dict[str, Any]:
    """Versions that matter for parity, read without importing the heavy packages."""
    import platform
    from importlib import metadata

    names = ["numpy", "pandas", "opencv-python", "opencv-python-headless", "torch", "torchvision"]
    names += ["open_clip_torch", "transformers", "pycocotools", "Pillow", "pyarrow"]
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "git_commit": git_commit(), "packages": versions}


def next_free_path(path: str | os.PathLike[str]) -> Path:
    """`path`, or `name_2.ext`, `name_3.ext`... if it exists. Never returns an existing file."""
    target = Path(path)
    candidate, n = target, 1
    while candidate.exists():
        n += 1
        candidate = target.with_name(f"{target.stem}_{n}{target.suffix}")
    return candidate


def write_report(
    summary: Mapping[str, Any],
    reports_dir: str | os.PathLike[str],
    date: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write `parity_<date>.json` as a new file in `reports_dir` and return its path."""
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    body = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": environment_info(),
        **summary,
        **(extra or {}),
    }
    path = next_free_path(Path(reports_dir) / f"parity_{date}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:  # "x": fail rather than overwrite
        f.write(json.dumps(body, indent=2, default=str) + "\n")
    return path
