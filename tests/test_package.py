import subprocess
import sys

import coralid


def test_version_is_a_string():
    assert isinstance(coralid.__version__, str)
    assert coralid.__version__.count(".") == 2


def test_import_stays_light():
    """Importing the package and its modules must not pull in the heavy ML dependencies."""
    code = (
        "import sys, coralid; "
        "import coralid.config, coralid.masks, coralid.images, coralid.crops, "
        "coralid.filters, coralid.manifest, coralid.genera, "
        "coralid.encoders, coralid.zero_shot, coralid.batch, coralid.evaluation, "
        "coralid.predictions; "
        "heavy = {'torch', 'torchvision', 'open_clip', 'transformers', 'umap', 'hdbscan', "
        "'sklearn', 'matplotlib'}; "
        "loaded = heavy & set(sys.modules); "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
