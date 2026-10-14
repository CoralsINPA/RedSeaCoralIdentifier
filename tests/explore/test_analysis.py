import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

from matplotlib.figure import Figure  # noqa: E402

from coralid.baseline import baseline_paths  # noqa: E402
from coralid.explore.analysis import (  # noqa: E402
    cluster_genus_table,
    explore_summary,
    features_path,
    genus_cluster_table,
    load_exploration,
    write_explore_report,
)
from coralid.explore.clustering import embedding_table  # noqa: E402
from coralid.zero_shot import save_embeddings  # noqa: E402
from tests.run_data import make_run  # noqa: E402

FEATURES = "finetuned-bioclip__masked"


def kept_uids(cfg):
    manifest = pd.read_parquet(cfg.manifest_path)
    kept = manifest[manifest["keep"]]
    return [f"{n}#{c}" for n, c in zip(kept["image_name"], kept["coral_id"], strict=True)]


@pytest.fixture
def cfg(tmp_path):
    return make_run(tmp_path / "data")


def write_features(cfg, name=FEATURES, order=None):
    """Features whose first value is the crop's position in the manifest (to check alignment)."""
    manifest_order = kept_uids(cfg)
    uids = [manifest_order[i] for i in order] if order is not None else manifest_order
    X = np.array([[manifest_order.index(u), 1, 2, 3] for u in uids], dtype=np.float32)
    return uids, save_embeddings(cfg.features_dir / f"{name}.npz", uids, X)


def write_predictions(cfg):
    kept = pd.read_parquet(cfg.manifest_path).query("keep")
    genus = ["Acropora", "Porites", "Porites", "Pavona", "Acropora", "Porites"]
    predictions = pd.DataFrame(
        {
            "image_name": kept["image_name"].to_numpy(),
            "coral_id": kept["coral_id"].to_numpy(),
            "predicted_genus": genus,
            "confidence": np.linspace(0.3, 0.9, 6),
            "margin": np.linspace(0.1, 0.5, 6),
        }
    )
    path = baseline_paths(cfg).predictions
    path.parent.mkdir(parents=True)
    predictions.to_parquet(path)
    return predictions


def test_the_table_describes_each_crop_in_the_order_of_the_features(cfg):
    predictions = write_predictions(cfg)
    order = [5, 3, 0, 1, 4, 2]  # the feature rows are not in manifest order
    uids, _ = write_features(cfg, order=order)
    data = load_exploration(cfg, FEATURES)
    assert list(data.table["uid"]) == uids and data.name == FEATURES
    assert data.X.shape == (6, 4) and data.notes == []
    expected = predictions.set_index(["image_name", "coral_id"])
    for _, row in data.table.iterrows():
        truth = expected.loc[(row["image_name"], row["coral_id"])]
        assert row["predicted_genus"] == truth["predicted_genus"]
        assert row["confidence"] == pytest.approx(truth["confidence"])
    assert set(data.table["orientation"]) == {"landscape"}  # the synthetic frame is 200x150
    assert {"area", "sharpness", "visible_px"} <= set(data.table.columns)
    # the features are those of the file, row for row: the first value is the manifest position
    np.testing.assert_array_equal(data.X[:, 0], order)


def test_without_a_baseline_the_genus_is_unknown_and_a_note_says_so(cfg):
    write_features(cfg)
    data = load_exploration(cfg, FEATURES)
    assert set(data.table["predicted_genus"]) == {"unknown"}
    assert any("no baseline predictions" in note for note in data.notes)


def test_a_crop_the_baseline_does_not_know_is_unknown(cfg):
    predictions = write_predictions(cfg)
    write_features(cfg)
    data = load_exploration(cfg, FEATURES, predictions=predictions.iloc[:4])
    assert (data.table["predicted_genus"] == "unknown").sum() == 2
    assert any("2 crop(s) have no baseline prediction" in note for note in data.notes)


def test_without_a_manifest_there_is_no_orientation(cfg):
    write_features(cfg)
    cfg.manifest_path.unlink()
    data = load_exploration(cfg, FEATURES)
    assert "orientation" not in data.table.columns
    assert any("no manifest" in note for note in data.notes)


def test_a_missing_features_file_says_which_job_makes_it(cfg):
    with pytest.raises(FileNotFoundError, match="02_zero_shot"):
        load_exploration(cfg, "dinov2-s14__masked_square")


def test_the_features_are_named_by_stem_not_path(cfg):
    assert features_path(cfg, FEATURES) == cfg.features_dir / f"{FEATURES}.npz"
    for bad in (f"{FEATURES}.npz", "features/x"):
        with pytest.raises(ValueError, match="file stem"):
            features_path(cfg, bad)


def table_for_clusters():
    return embedding_table(
        [f"I.JPG#{i}" for i in range(10)],
        cluster=[10, 2, 2, 2, 2, 10, 10, -1, 2, 10],
        predicted_genus=["A", "A", "A", "A", "B", "B", "B", "B", "A", "A"],
        confidence=np.linspace(0.1, 1.0, 10),
    )


def test_a_cluster_is_described_by_its_commonest_genera():
    summary = cluster_genus_table(table_for_clusters())
    assert list(summary["cluster"]) == [-1, 2, 10]  # numeric order
    two = summary.set_index("cluster").loc[2]
    assert two["n"] == 5 and two["share"] == 0.5
    assert (two["top_genus"], two["top_share"]) == ("A", 0.8)
    assert (two["second_genus"], two["second_share"]) == ("B", 0.2)
    only = summary.set_index("cluster").loc[-1]
    assert only["second_genus"] == "" and only["top_share"] == 1.0
    assert two["mean_confidence"] == pytest.approx(np.mean([0.2, 0.3, 0.4, 0.5, 0.9]))


def test_a_genus_is_described_by_the_clusters_it_spreads_over():
    summary = genus_cluster_table(table_for_clusters()).set_index("predicted_genus")
    assert summary.loc["A", "n"] == 6 and summary.loc["A", "n_clusters"] == 2
    assert summary.loc["A", "main_cluster"] == 2
    assert summary.loc["A", "main_cluster_share"] == pytest.approx(4 / 6)


def test_the_summary_reports_what_was_run(cfg):
    write_predictions(cfg)
    write_features(cfg)
    data = load_exploration(cfg, FEATURES)
    bare = explore_summary(data)
    assert bare["n_crops"] == 6 and bare["dim"] == 4 and bare["n_photos"] == 3
    assert bare["predicted_genus_counts"] == {"Acropora": 2, "Porites": 3, "Pavona": 1}
    assert "kmeans" not in bare and "hdbscan" not in bare and "pca" not in bare
    kmeans = SimpleNamespace(
        labels=np.array([0, 1, 1, 0, 2, 2]),
        silhouette=0.12,
        davies_bouldin=2.5,
        compactness=pd.Series([0.5, 0.7, 0.9]),
    )
    hdb = SimpleNamespace(
        n_clusters=2, n_noise=1, silhouette=0.3, davies_bouldin=1.1, sizes={-1: 1, 0: 3, 1: 2}
    )
    pca = SimpleNamespace(Z=np.zeros((6, 50)), total_variance=0.81)
    full = explore_summary(data, pca, kmeans, hdb, n_map=5)
    assert full["n_in_maps"] == 5 and full["pca"] == {"n_components": 50, "variance_kept": 0.81}
    assert full["kmeans"]["n_clusters"] == 3 and full["kmeans"][
        "mean_compactness"
    ] == pytest.approx(0.7)
    assert full["hdbscan"]["sizes"] == {"-1": 1, "0": 3, "1": 2}
    json.dumps(full)  # plain data


def test_the_report_is_new_files_in_a_new_folder(tmp_path):
    small = table_for_clusters()
    big = pd.DataFrame({"x": np.arange(6000.0)})
    figure = Figure()
    figure.subplots().plot([0, 1])
    args = (tmp_path / "reports", "dinov2-s14__masked_square", {"clusters": small, "map": big})
    first = write_explore_report(
        *args, figures={"maps": figure}, summary={"a": 1}, date="2026-10-12"
    )
    second = write_explore_report(*args, date="2026-10-12")
    assert first.name == "embeddings_dinov2-s14__masked_square_2026-10-12"
    assert second.name == first.name + "_2"
    assert sorted(p.name for p in first.iterdir()) == [
        "clusters.csv",
        "map.parquet",  # more than 5,000 rows
        "maps.png",
        "summary.json",
    ]
    assert json.loads((first / "summary.json").read_text()) == {"a": 1}
    assert len(pd.read_parquet(first / "map.parquet")) == 6000
    assert sorted(p.name for p in second.iterdir()) == [
        "clusters.csv",
        "map.parquet",
        "summary.json",
    ]
