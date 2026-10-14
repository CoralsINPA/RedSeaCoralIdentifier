"""The Colab notebooks are thin drivers; these tests hold them to that.

There are four pipeline notebooks (`01` to `04`) and one exploration notebook
(`notebooks/explore/embeddings.ipynb`, outside the pipeline).

They check the files as text (nothing is executed): valid nbformat, outputs stripped, under 15
cells, no function definitions, every code cell compiles, every `coralid` name a notebook imports
exists, the first code cell is `coralid.colab.BOOTSTRAP_CELL` with the right extras, and the cells
that could change data default to doing nothing. `notebooks/parity_check.ipynb` (a one-time check
with a verbatim legacy cell) and `notebooks/legacy/` are not held to these rules.
"""

import ast
import importlib
import json
from pathlib import Path

import pytest

from coralid.colab import BOOTSTRAP_CELL

ROOT = Path(__file__).resolve().parents[1]
BRANCH = "claude/awesome-archimedes-kvh4c7"
EXTRAS = {
    "01_segment_and_filter": "segment,viz",
    "02_zero_shot": "clip,viz",
    "03_expert_review": "viz",
    "04_supervised_probe": "probe,viz",
    "05_millepora": "viz",
    "explore/embeddings": "dino,cluster,viz",
}
PIPELINE = [name for name in EXTRAS if not name.startswith("explore/")]


def load(name):
    path = ROOT / "notebooks" / f"{name}.ipynb"
    if not path.exists():
        pytest.skip(f"{path} is not in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


def code_cells(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def runnable(source):
    """The cell without IPython `!` and `%` lines, which `ast` cannot parse."""
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("!", "%"))
    )


@pytest.mark.parametrize("name", EXTRAS)
def test_the_notebook_is_valid_and_stripped(name):
    nb = load(name)
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.from_dict(nb))
    assert len(nb["cells"]) < 15
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
        assert set(cell["metadata"]) <= {"tags"}  # no Colab output ids or collapsed-state leftovers


@pytest.mark.parametrize("name", EXTRAS)
def test_the_notebook_defines_no_functions_and_every_cell_compiles(name):
    for source in code_cells(load(name)):
        tree = ast.parse(runnable(source))
        compile(tree, f"<{name}>", "exec")
        defined = [
            type(node).__name__
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda)
        ]
        assert not defined, f"{name} defines {defined}; the logic belongs in the library"


@pytest.mark.parametrize("name", EXTRAS)
def test_the_first_cell_explains_and_the_first_code_cell_is_the_bootstrap(name):
    nb = load(name)
    intro = "".join(nb["cells"][0]["source"])
    assert nb["cells"][0]["cell_type"] == "markdown" and intro.startswith("# ")
    for needle in (
        "Safe to run and re-run",
        "What to paste back",
        "| Step | What it does | Time",
    ):
        assert needle in intro, f"{name}: the first cell lacks '{needle}'"
    assert f"blob/{BRANCH}/notebooks/{name}.ipynb" in intro  # the Colab badge
    first = code_cells(nb)[0]
    expected = BOOTSTRAP_CELL.replace('EXTRAS = "clip,viz"', f'EXTRAS = "{EXTRAS[name]}"')
    assert first == expected.rstrip("\n")


@pytest.mark.parametrize("name", EXTRAS)
def test_every_coralid_name_the_notebook_imports_exists(name):
    seen = 0
    for source in code_cells(load(name)):
        for node in ast.walk(ast.parse(runnable(source))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("coralid"):
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name), f"{node.module} has no {alias.name}"
                    seen += 1
    assert seen, f"{name} imports nothing from coralid"


def first_lines(source, n=14):
    return "\n".join(source.splitlines()[:n])


def test_cells_that_can_change_data_default_to_doing_nothing():
    cells = {name: "\n\n".join(code_cells(load(name))) for name in EXTRAS}
    assert "SEGMENT_MISSING = False" in cells["01_segment_and_filter"]
    assert "RUN_EMBED = False" in cells["explore/embeddings"]
    assert "RUN_EXPORT = False" in cells["03_expert_review"]
    # the on-hold blind review is gated, and the gate comes before the export itself
    gated = cells["03_expert_review"]
    assert gated.index("RUN_EXPORT = False") < gated.index("write_blind_review(")
    assert "if RUN_EXPORT:" in gated
    # the exploration notebook uses the BioCLIP features by default (setting at the top of the cell)
    config = next(s for s in code_cells(load("explore/embeddings")) if "FEATURES =" in s)
    assert 'FEATURES = "finetuned-bioclip__masked"' in first_lines(config)


def test_notebook_01_checks_the_segmentation_in_a_scratch_run():
    text = "\n".join(code_cells(load("01_segment_and_filter")))
    assert "run_segmentation_check(" in text and 'CHECK_PREFIX = "segcheck_"' in text
    assert "CHECK_N = 5" in text and 'CHECK_PHOTOS = "data_50"' in text
    # it never runs the segmentation job on the active run unless asked
    assert text.count("run_segmentation(cfg") == 1
    assert "if plan.todo and SEGMENT_MISSING:" in text


def test_notebook_04_never_builds_labels_from_predictions():
    text = "\n".join(code_cells(load("04_supervised_probe")))
    assert "expert_labels(" in text and "landscape" in text
    assert "predicted_genus" not in text
    markdown = "".join(
        "".join(c["source"])
        for c in load("04_supervised_probe")["cells"]
        if c["cell_type"] == "markdown"
    )
    assert "too little to mean anything" in markdown and "idea 1.1" in markdown


def test_the_numbered_notebooks_are_the_pipeline_and_nothing_else():
    numbered = sorted(p.stem for p in (ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb"))
    assert (
        numbered
        == sorted(PIPELINE)
        == [
            "01_segment_and_filter",
            "02_zero_shot",
            "03_expert_review",
            "04_supervised_probe",
            "05_millepora",
        ]
    )


def test_no_pipeline_notebook_uses_the_exploration_subpackage_and_the_other_way_round():
    for name in PIPELINE:
        for source in code_cells(load(name)):
            assert "coralid.explore" not in source, f"{name} uses coralid.explore"
    nb = load("explore/embeddings")
    assert "not a step of the pipeline" in "".join(nb["cells"][0]["source"])
    assert any("coralid.explore" in source for source in code_cells(nb))


def test_notebook_03_prints_the_source_before_it_exports_and_scoring_checks_it():
    cells = code_cells(load("03_expert_review"))
    export = next(s for s in cells if "RUN_EXPORT = False" in s)
    assert export.index("baseline_source(cfg)") < export.index("print(")
    assert export.index("print(") < export.index("build_blind_review(")
    assert "source=source" in export and ".lines()" in export
    scoring = next(s for s in cells if "score_blind_review(" in s)
    assert "current_source=current" in scoring and "baseline_source(cfg)" in scoring
    text = "\n".join(cells)
    assert "audit" not in text.replace("genus_audit", "")  # the old audit is gone
