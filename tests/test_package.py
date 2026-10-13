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
        "coralid.predictions, coralid.parity, coralid.parity_report, "
        "coralid.baseline, coralid.reports, coralid.labeling, coralid.viz, "
        "coralid.viz_plots, coralid.funnel, coralid.probe_labels, "
        "coralid.mask_compare, coralid.parity_segmentation, "
        "coralid.segmentation, coralid.probe, "
        "coralid.colab, coralid.jobs, coralid.cli, coralid.explore, coralid.explore.dino, "
        "coralid.explore.embed, coralid.explore.clustering, coralid.explore.plots, "
        "coralid.explore.analysis; "
        "heavy = {'torch', 'torchvision', 'open_clip', 'transformers', 'umap', 'hdbscan', "
        "'sklearn', 'matplotlib', 'segment_anything'}; "
        "loaded = heavy & set(sys.modules); "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
