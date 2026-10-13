import sys
import types

import numpy as np
import pandas as pd
import pytest

from coralid.explore.clustering import (
    analyze_kmeans,
    cluster_sizes,
    drop_first_component,
    embedding_table,
    hdbscan_clusters,
    kmeans_cluster,
    pca_reduce,
    sample_per_group,
    subsample,
    sweep_projections,
    tsne_embed,
    umap_embed,
)


def blobs(per_blob=30, dim=8, k=3, seed=0, spread=0.3):
    rng = np.random.RandomState(seed)
    centres = rng.randn(k, dim) * 5
    X = np.vstack([c + spread * rng.randn(per_blob, dim) for c in centres])
    truth = np.repeat(np.arange(k), per_blob)
    return X.astype(np.float32), truth


@pytest.fixture
def fake_umap(monkeypatch):
    """A stand-in for umap-learn that records how UMAP was built (the real one is optional)."""
    calls = []

    class UMAP:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.n_components = kwargs.get("n_components", 2)

        def fit_transform(self, X):
            return np.asarray(X)[:, : self.n_components].astype(np.float64)

    module = types.ModuleType("umap")
    module.UMAP = UMAP
    monkeypatch.setitem(sys.modules, "umap", module)
    return calls


@pytest.fixture
def fake_hdbscan(monkeypatch):
    calls = []

    class HDBSCAN:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def fit_predict(self, X):
            X = np.asarray(X)  # two clusters, and every tenth point is noise (-1)
            side = (X[:, 0] > np.median(X[:, 0])).astype(int)
            return np.where(np.arange(len(X)) % 10 == 0, -1, side)

    module = types.ModuleType("hdbscan")
    module.HDBSCAN = HDBSCAN
    monkeypatch.setitem(sys.modules, "hdbscan", module)
    return calls


# ------------------------------------------------------------------------- the legacy settings


def test_pca_keeps_the_requested_components_and_is_reproducible():
    X, _ = blobs()
    a, b = pca_reduce(X, 5), pca_reduce(X, 5)
    assert a.Z.shape == (90, 5) and a.explained_variance_ratio.shape == (5,)
    np.testing.assert_allclose(a.Z, b.Z)
    assert 0.9 < a.total_variance <= 1.0  # three well separated blobs live in a few components


def test_tsne_uses_the_notebook_settings(monkeypatch):
    from sklearn import manifold

    seen = {}
    real = manifold.TSNE

    class Spy(real):
        def __init__(self, **kwargs):
            seen.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(manifold, "TSNE", Spy)
    X, _ = blobs(per_blob=15)
    xy = tsne_embed(X)
    assert xy.shape == (45, 2)
    assert seen == {
        "n_components": 2,
        "perplexity": 10.0,
        "random_state": 42,
        "init": "pca",
        "learning_rate": "auto",
    }
    np.testing.assert_allclose(xy, tsne_embed(X))  # the seed fixes the map


def test_umap_uses_the_notebook_settings(fake_umap):
    X, _ = blobs()
    out = umap_embed(X)
    assert out.shape == (90, 2)
    assert fake_umap == [
        {
            "n_neighbors": 5,
            "min_dist": 0.0,
            "n_components": 2,
            "metric": "euclidean",
            "random_state": 42,
        }
    ]


def test_the_sweep_is_the_one_of_cells_13_and_15(fake_umap):
    X, _ = blobs(per_blob=15)
    maps = sweep_projections(X, perplexities=(5, 10), neighbors=(5, 10))
    assert list(maps) == ["tsne_perp5", "tsne_perp10", "umap_nn5", "umap_nn10"]
    assert all(m.shape == (45, 2) for m in maps.values())
    assert [(c["n_neighbors"], c["min_dist"]) for c in fake_umap] == [(5, 0.1), (10, 0.1)]
    assert all(c["random_state"] == 42 for c in fake_umap)


def test_kmeans_recovers_blobs_with_the_notebook_seed():
    X, truth = blobs()
    labels, centers = kmeans_cluster(X, n_clusters=3)
    assert centers.shape == (3, 8)
    assert len(set(zip(truth, labels, strict=True))) == 3  # one cluster per blob
    again, _ = kmeans_cluster(X, n_clusters=3)
    np.testing.assert_array_equal(labels, again)


def test_analyze_kmeans_scores_match_a_by_hand_computation():
    X, truth = blobs(spread=0.2)
    result = analyze_kmeans(X, 3)
    unit = X.astype(np.float64)
    unit = unit / (np.linalg.norm(unit, axis=1, keepdims=True) + 1e-8)
    assert len(set(zip(truth, result.labels, strict=True))) == 3
    # compactness: mean cosine of the members to their normalized center
    for cluster, value in result.compactness.items():
        center = result.centers[cluster] / (np.linalg.norm(result.centers[cluster]) + 1e-8)
        members = unit[result.labels == cluster]
        assert value == pytest.approx(float(np.mean(members @ center)), abs=1e-6)
    # separation: cosine between centers, 1 on the diagonal
    sep = result.separation.to_numpy()
    centers = result.centers / np.linalg.norm(result.centers, axis=1, keepdims=True)
    np.testing.assert_allclose(sep, centers @ centers.T, atol=1e-6)
    np.testing.assert_allclose(np.diag(sep), 1.0)
    assert list(result.separation.columns) == ["Clus_0", "Clus_1", "Clus_2"]
    # well separated blobs: a high cosine silhouette and a low Davies-Bouldin
    assert result.silhouette > 0.8 and result.davies_bouldin < 0.3
    sampled = analyze_kmeans(X, 3, silhouette_sample=60)
    assert 0.5 < sampled.silhouette <= 1.0


def test_dropping_component_zero_removes_the_dominant_axis():
    from sklearn.decomposition import PCA

    rng = np.random.RandomState(3)
    X = rng.randn(80, 10) * 0.1
    X[:, 0] += rng.randn(80) * 10  # a loud 'lighting' axis
    cleaned = drop_first_component(X, n_components=5)
    assert cleaned.shape == X.shape
    assert cleaned[:, 0].std() < 0.5 * X[:, 0].std()
    pca = PCA(n_components=5, random_state=42).fit(X)
    np.testing.assert_allclose(pca.transform(cleaned)[:, 0], 0, atol=1e-8)


def test_hdbscan_pipeline_uses_the_notebook_settings(fake_umap, fake_hdbscan):
    X, _ = blobs(per_blob=40, dim=60)
    result = hdbscan_clusters(X)
    (u,) = fake_umap
    assert u == {
        "n_neighbors": 30,
        "min_dist": 0.0,
        "n_components": 5,
        "metric": "cosine",
        "random_state": 42,
    }
    (h,) = fake_hdbscan
    assert h == {
        "min_cluster_size": 500,
        "min_samples": 50,
        "metric": "euclidean",
        "prediction_data": False,
    }
    assert result.umap_features.shape == (120, 5)
    assert result.n_clusters == 2 and result.n_noise == 12 and sum(result.sizes.values()) == 120
    assert result.sizes[-1] == 12
    assert result.silhouette is not None and result.davies_bouldin is not None  # noise left out


def test_hdbscan_with_one_cluster_has_no_scores(fake_umap, monkeypatch):
    class OneCluster:
        def __init__(self, **kwargs):
            pass

        def fit_predict(self, X):
            return np.zeros(len(X), dtype=int)

    module = types.ModuleType("hdbscan")
    module.HDBSCAN = OneCluster
    monkeypatch.setitem(sys.modules, "hdbscan", module)
    X, _ = blobs(dim=60)
    result = hdbscan_clusters(X, remove_first_pc=False)
    assert result.n_clusters == 1 and result.silhouette is None and result.davies_bouldin is None


def test_real_umap_and_hdbscan_run_when_installed():
    pytest.importorskip("umap")
    pytest.importorskip("hdbscan")
    X, _ = blobs(per_blob=60, dim=60)
    assert umap_embed(X, n_neighbors=5).shape == (180, 2)
    result = hdbscan_clusters(X, min_cluster_size=20, min_samples=5, n_neighbors=10)
    assert result.labels.shape == (180,)


# ------------------------------------------------------------------------- tables for viz


def test_embedding_table_is_keyed_by_image_and_coral_id():
    table = embedding_table(
        ["A.JPG#0", "A.JPG#12", "B.png#3"], x=[1.0, 2.0, 3.0], cluster=np.array([0, 1, 1])
    )
    assert list(table.columns) == ["uid", "image_name", "coral_id", "x", "cluster"]
    assert table["image_name"].tolist() == ["A.JPG", "A.JPG", "B.png"]
    assert table["coral_id"].tolist() == [0, 12, 3]
    with pytest.raises(ValueError, match="'x' has 2 values for 3 crops"):
        embedding_table(["A#0", "A#1", "A#2"], x=[1, 2])


def test_subsample_is_the_notebooks_draw():
    assert subsample(10, size=100).tolist() == list(range(10))
    drawn = subsample(50000)
    assert len(drawn) == 15000 and len(set(drawn)) == 15000
    np.random.seed(42)  # CM cell 21, verbatim
    np.testing.assert_array_equal(drawn, np.random.choice(50000, size=15000, replace=False))


def test_cluster_sizes_and_sampling_per_group():
    labels = np.array([0, 0, 0, 1, 2, 2])
    sizes = cluster_sizes(labels)
    assert sizes["cluster"].tolist() == [0, 1, 2] and sizes["count"].tolist() == [3, 1, 2]
    assert sizes["percent"].round(1).tolist() == [50.0, 16.7, 33.3]
    table = embedding_table([f"IMG_{i}.JPG#0" for i in range(6)], cluster=labels)
    picked = sample_per_group(table, "cluster", n=2)
    assert picked["cluster"].tolist() == [0, 0, 1, 2, 2]  # groups sorted, at most 2 each
    again = sample_per_group(table, "cluster", n=2)
    pd.testing.assert_frame_equal(picked, again)  # the seed fixes the draw
    assert len(sample_per_group(table.iloc[:0], "cluster")) == 0


def test_pca_keeps_all_it_has_when_the_features_are_smaller_than_asked():
    pytest.importorskip("sklearn")
    from coralid.explore.clustering import drop_first_component, pca_reduce

    X = np.random.RandomState(0).randn(40, 24)  # 24 dimensions, 50 components asked for
    result = pca_reduce(X, 50)
    assert result.Z.shape == (40, 24) and result.total_variance == pytest.approx(1.0)
    assert pca_reduce(X[:10], 50).Z.shape == (10, 10)  # fewer crops than components
    assert drop_first_component(X, 50).shape == X.shape
