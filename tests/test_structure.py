"""The pipeline does not depend on the exploration direction (`coralid.explore`).

The pipeline is segment -> manifest -> zero-shot baseline -> evaluation and expert review ->
linear probe. Exploring embeddings (DINO features, clustering) is a separate direction. These
tests read the source of every module as text, and import the pipeline in a fresh interpreter.
"""

import ast
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "coralid"
EXPLORE = SRC / "explore"
# The only module outside coralid/explore that may name it, and only inside a function.
LAZY_ALLOWED = {"cli.py"}


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(SRC.parent).with_suffix("").parts)


def _explore_imports(path: Path) -> list[tuple[int, bool]]:
    """`(line, inside_a_function)` for every import of coralid.explore in a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    found = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            names = [base] + [f"{base}.{alias.name}" for alias in node.names]
        if any(
            n == "coralid.explore" or n.startswith("coralid.explore.") or n.lstrip(".") == "explore"
            for n in names
        ) or any(n.lstrip(".").startswith("explore.") for n in names):
            scope, in_function = node, False
            while scope in parents:
                scope = parents[scope]
                if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef):
                    in_function = True
            found.append((node.lineno, in_function))
    return found


def test_no_pipeline_module_imports_the_exploration_subpackage():
    modules = [p for p in sorted(SRC.rglob("*.py")) if EXPLORE not in p.parents]
    assert len(modules) > 30  # the scan really sees the package
    offenders = []
    for path in modules:
        for line, in_function in _explore_imports(path):
            lazy_ok = path.name in LAZY_ALLOWED and path.parent == SRC and in_function
            if not lazy_ok:
                offenders.append(f"{_module_name(path)}:{line}")
    assert not offenders, f"pipeline modules import coralid.explore: {offenders}"


def test_the_scan_catches_every_way_of_importing_it(tmp_path):
    cases = {
        "import coralid.explore.embed\n": [(1, False)],
        "from coralid.explore import embed\n": [(1, False)],
        "from coralid import explore\n": [(1, False)],
        "from . import explore\n": [(1, False)],
        "from .explore.embed import run_embeddings\n": [(1, False)],
        "def f():\n    from coralid.explore.embed import run_embeddings\n": [(2, True)],
        "from coralid.exploration import x\n": [],
    }
    for source, expected in cases.items():
        module = tmp_path / "m.py"
        module.write_text(source)
        assert _explore_imports(module) == expected, source


def test_cli_imports_the_subpackage_only_inside_the_exploration_command():
    calls = _explore_imports(SRC / "cli.py")
    assert calls and all(in_function for _, in_function in calls)


def test_importing_the_pipeline_loads_nothing_of_the_exploration_subpackage():
    code = (
        "import sys, coralid.cli, coralid.jobs, coralid.baseline, coralid.probe, "
        "coralid.probe_labels, coralid.viz, coralid.viz_plots, coralid.encoders, "
        "coralid.evaluation, coralid.reports, coralid.review_export, coralid.config; "
        "loaded = sorted(m for m in sys.modules if m.startswith('coralid.explore')); "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_the_probe_extra_does_not_need_the_exploration_extras():
    text = (SRC.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    probe = text.split("probe = [", 1)[1].split("]", 1)[0]
    assert "umap" not in probe and "hdbscan" not in probe and "torch" not in probe
