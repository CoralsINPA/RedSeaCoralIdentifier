"""The project refers only to code it defines: the excluded V2 pipeline is gone from the code.

V2 (24 genera, tiers, the notebook `CoralClassification_v2_Pipeline`) and the old Millepora audit
made on its predictions are not a reference for anything. This test fails if a source file, a
config, a test (except this one) or a non-legacy notebook mentions them. `notebooks/legacy/` is
frozen history and is not scanned. Model names such as `dinov2`, `DINOv2`, `siglip2`, `bioclip-2`
or `BioCLIP v2` are not V2 and are not flagged.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".ipynb", ".json"}

# The capital-V word only: `\bv2\b` in lower case is a model version ("BioCLIP v2").
UPPER_V2 = re.compile(r"\bV2\b")
# Names of the V2 outputs and notebook, in any case.
V2_NAMES = re.compile(
    r"v2_pipeline|v2_weak|v2_tiered|v2_embeddings|v2_final|v2_all_corals|millepora_sharp_audit"
    r"|CoralClassification_v2|V2[ _-](?:pipeline|notebook|phase|labels|predictions|prompt|CSV)",
    re.IGNORECASE,
)


def _files():
    folders = [ROOT / "src", ROOT / "configs", ROOT / "tests"]
    paths = [p for folder in folders for p in sorted(folder.rglob("*"))]
    paths += sorted((ROOT / "notebooks").glob("*.ipynb"))
    paths += sorted((ROOT / "notebooks" / "explore").glob("*.ipynb"))
    return [
        p
        for p in paths
        if p.is_file()
        and p.suffix in SUFFIXES
        and p.resolve() != THIS_FILE
        and "__pycache__" not in p.parts
        and ".egg-info" not in str(p)
    ]


def _text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":  # the cells' text, not the JSON's escapes
        nb = json.loads(text)
        text = "\n".join("".join(cell["source"]) for cell in nb["cells"])
    return text


def _hits(text: str) -> list[str]:
    return [m.group(0) for pattern in (UPPER_V2, V2_NAMES) for m in pattern.finditer(text)]


def test_the_scan_sees_the_code_and_notebooks_and_skips_legacy():
    names = {p.relative_to(ROOT).as_posix() for p in _files()}
    assert "src/coralid/cli.py" in names and "configs/genera.yaml" in names
    assert "notebooks/03_expert_review.ipynb" in names
    assert "notebooks/explore/embeddings.ipynb" in names
    assert not any(n.startswith("notebooks/legacy/") for n in names)
    assert len(names) > 60


def test_the_patterns_catch_v2_and_leave_model_names_alone():
    for text in (
        "the V2 pipeline",
        "(V2 cell 9)",
        "v2_pipeline/x.csv",
        "v2_weak_labels_expanded.csv",
        "v2_tiered_labels",
        "millepora_sharp_audit.csv",
        "CoralClassification_v2_Pipeline.ipynb",
        "v2 pipeline",
        "the V2-style sheet",
    ):
        assert _hits(text), text
    for text in (
        "dinov2-s14",
        "DINOv2 ViT-S/14",
        "torch.hub dinov2_vits14",
        "siglip2 / SigLIP 2",
        "bioclip-2",
        "BioCLIP v2, trained on 200M images",
        "ViT-B/16",
        "taxonomy_v1",
    ):
        assert not _hits(text), text


def test_no_source_config_test_or_notebook_mentions_v2():
    found = {
        p.relative_to(ROOT).as_posix(): sorted(set(_hits(_text(p))))
        for p in _files()
        if _hits(_text(p))
    }
    assert not found, f"V2 references: {found}"
