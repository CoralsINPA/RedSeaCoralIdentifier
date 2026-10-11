import subprocess
import sys

import coralid


def test_version_is_a_string():
    assert isinstance(coralid.__version__, str)
    assert coralid.__version__.count(".") == 2


def test_import_stays_light():
    """`import coralid` must not pull in the heavy ML dependencies."""
    code = (
        "import sys, coralid; "
        "heavy = {'torch', 'open_clip', 'transformers', 'umap', 'hdbscan', 'sklearn'}; "
        "loaded = heavy & set(sys.modules); "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
