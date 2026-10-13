"""Dimensionality reduction and clustering of embeddings, with the settings of the legacy notebook.

These are the computations of CoralMasking.ipynb cells 12-15 and 19-22 (PCA, t-SNE, UMAP,
K-Means with its quality scores, HDBSCAN), kept apart from the plots (`coralid.explore.plots`):
nothing here draws, and nothing in the plots computes. They work on a matrix `X` of embeddings
(for example the `X` of a `features/*.npz` file) and return arrays or small result objects;
`embedding_table` turns those into a table keyed by `(image_name, coral_id)` that the plots take.

Settings reproduced from the notebook (all with `random_state=42`):

| function | settings | cells |
|---|---|---|
| `pca_reduce` | PCA, 50 components (100 before K-Means) | 13, 14 |
| `tsne_embed` | perplexity 10, `init="pca"`, `learning_rate="auto"` | 12, 13, 15 |
| `umap_embed` | `n_neighbors=5`, `min_dist=0.0` (0.1 in the sweeps), 2 components | 12, 13, 15 |
| `kmeans_cluster` | 50 clusters, `n_init=10` | 14 |
| `analyze_kmeans` | L2-normalized features, cosine silhouette, Davies-Bouldin, compactness | 19 |
| `hdbscan_clusters` | PCA 50 minus PC 0, UMAP (5 dims, cosine, 30 nn), HDBSCAN (500, 50) | 20 |

scikit-learn, umap-learn and hdbscan are imported inside the functions (extra `cluster`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

SEED = 42  # random_state of every step in the notebook


@dataclass
class PcaResult:
    """PCA scores and how much of the variance they keep."""

    Z: np.ndarray
    explained_variance_ratio: np.ndarray

    @property
    def total_variance(self) -> float:
        """Share of the variance the components keep (0 to 1)."""
        return float(self.explained_variance_ratio.sum())


def _cap_components(X: np.ndarray, n_components: int) -> int:
    """`n_components`, lowered to what the data allows (a PCA cannot exceed rows or columns)."""
    return max(1, min(n_components, X.shape[0], X.shape[1]))


def pca_reduce(X: np.ndarray, n_components: int = 50, seed: int = SEED) -> PcaResult:
    """Denoise by projecting on the first `n_components` principal components (CM cells 13, 14).

    Features with fewer than `n_components` dimensions (or fewer crops) keep all they have.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=_cap_components(X, n_components), random_state=seed)
    return PcaResult(pca.fit_transform(X), pca.explained_variance_ratio_)


def tsne_embed(X: np.ndarray, perplexity: float = 10.0, seed: int = SEED) -> np.ndarray:
    """A 2-D t-SNE map of `X` (n, 2), as CM cell 12 made it."""
    from sklearn.manifold import TSNE

    tsne = TSNE(
        n_components=2, perplexity=perplexity, random_state=seed, init="pca", learning_rate="auto"
    )
    return tsne.fit_transform(X)


def umap_embed(
    X: np.ndarray,
    n_neighbors: int = 5,
    min_dist: float = 0.0,
    n_components: int = 2,
    metric: str = "euclidean",
    seed: int = SEED,
) -> np.ndarray:
    """A UMAP map of `X` (n, n_components); the defaults are CM cell 12's."""
    import warnings

    import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=seed,
    )
    with warnings.catch_warnings():
        # a fixed seed makes UMAP single-threaded and it says so on every call
        warnings.filterwarnings("ignore", message="n_jobs value 1 overridden", category=UserWarning)
        return reducer.fit_transform(X)


def sweep_projections(
    X: np.ndarray,
    perplexities: tuple[int, ...] = (5, 10, 15, 20),
    neighbors: tuple[int, ...] = (5, 10, 15, 20),
    min_dist: float = 0.1,
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    """The parameter sweep of CM cells 13 and 15: t-SNE per perplexity, UMAP per neighbor count.

    `X` is normally the PCA-50 features. Returns `{"tsne_perp5": (n, 2), ..., "umap_nn5": ...}`.
    """
    maps = {f"tsne_perp{p}": tsne_embed(X, p, seed) for p in perplexities}
    for n in neighbors:
        maps[f"umap_nn{n}"] = umap_embed(X, n, min_dist, seed=seed)
    return maps


def kmeans_cluster(
    X: np.ndarray, n_clusters: int = 50, seed: int = SEED, n_init: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """K-Means labels and centers (CM cell 14: 50 clusters, `n_init=10`)."""
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init).fit(X)
    return model.labels_, model.cluster_centers_


@dataclass
class KMeansAnalysis:
    """K-Means on L2-normalized features and its quality scores (CM cell 19)."""

    labels: np.ndarray
    centers: np.ndarray
    silhouette: float  # cosine; closer to 1 is better
    davies_bouldin: float  # closer to 0 is better
    compactness: pd.Series  # mean cosine similarity of a cluster's members to its center
    separation: pd.DataFrame  # cosine similarity between cluster centers (1 on the diagonal)


def analyze_kmeans(
    X: np.ndarray,
    n_clusters: int,
    seed: int = SEED,
    n_init: int = 10,
    silhouette_sample: int | None = None,
) -> KMeansAnalysis:
    """Cluster `X` after L2 normalization and score the clusters, as `analyze_kmeans_embeddings`.

    `silhouette_sample` scores a random subset of that size (the full silhouette needs all
    pairwise distances, which is slow for tens of thousands of crops).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score, silhouette_score
    from sklearn.metrics.pairwise import cosine_similarity

    X = np.asarray(X, dtype=np.float64)
    normalized = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
    labels = kmeans.fit_predict(normalized)
    silhouette = silhouette_score(
        normalized, labels, metric="cosine", sample_size=silhouette_sample, random_state=seed
    )
    compactness = {}
    for cluster in np.unique(labels):
        centroid = kmeans.cluster_centers_[cluster]
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        members = normalized[labels == cluster]
        compactness[int(cluster)] = float(np.mean(cosine_similarity(members, centroid[None])))
    names = [f"Clus_{c}" for c in np.unique(labels)]
    separation = cosine_similarity(kmeans.cluster_centers_)
    np.fill_diagonal(separation, 1.0)
    return KMeansAnalysis(
        labels=labels,
        centers=kmeans.cluster_centers_,
        silhouette=float(silhouette),
        davies_bouldin=float(davies_bouldin_score(normalized, labels)),
        compactness=pd.Series(compactness, name="compactness"),
        separation=pd.DataFrame(separation, index=names, columns=names),
    )


def drop_first_component(X: np.ndarray, n_components: int = 50, seed: int = SEED) -> np.ndarray:
    """Remove principal component 0 (the 'lighting' component of CM cell 20); same space as X."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=_cap_components(X, n_components), random_state=seed)
    scores = pca.fit_transform(X)
    scores[:, 0] = 0
    return pca.inverse_transform(scores)


@dataclass
class HdbscanResult:
    """HDBSCAN clusters found in a UMAP space (CM cell 20). Label -1 is noise."""

    labels: np.ndarray
    umap_features: np.ndarray
    sizes: dict[int, int]
    n_clusters: int
    n_noise: int
    silhouette: float | None  # on the UMAP space, noise left out; None with fewer than 2 clusters
    davies_bouldin: float | None


def hdbscan_clusters(
    X: np.ndarray,
    min_cluster_size: int = 500,
    min_samples: int = 50,
    n_neighbors: int = 30,
    n_components: int = 5,
    seed: int = SEED,
    prediction_data: bool = False,
    remove_first_pc: bool = True,
) -> HdbscanResult:
    """Density-based clusters as in CM cell 20: PCA-50 without component 0, UMAP, HDBSCAN.

    `prediction_data` only enables `approximate_predict` later; the labels do not depend on it.
    """
    import hdbscan
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    cleaned = drop_first_component(X, 50, seed) if remove_first_pc else X
    projected = umap_embed(cleaned, n_neighbors, 0.0, n_components, "cosine", seed)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        prediction_data=prediction_data,
    )
    labels = np.asarray(clusterer.fit_predict(projected))
    found = [int(c) for c in np.unique(labels) if c != -1]
    clustered = labels != -1
    scores: tuple[Any, Any] = (None, None)
    if len(found) > 1:
        scores = (
            float(silhouette_score(projected[clustered], labels[clustered])),
            float(davies_bouldin_score(projected[clustered], labels[clustered])),
        )
    return HdbscanResult(
        labels=labels,
        umap_features=projected,
        sizes={int(c): int((labels == c).sum()) for c in np.unique(labels)},
        n_clusters=len(found),
        n_noise=int((~clustered).sum()),
        silhouette=scores[0],
        davies_bouldin=scores[1],
    )


# --------------------------------------------------------------------------- tables for viz


def embedding_table(uids: list[str], **columns: Any) -> pd.DataFrame:
    """A table keyed by `(image_name, coral_id)` with the given columns, row-aligned with `uids`.

    For example `embedding_table(uids, x=xy[:, 0], y=xy[:, 1], cluster=labels)`.
    """
    table = pd.DataFrame(
        {
            "uid": list(uids),
            "image_name": [u.rpartition("#")[0] for u in uids],
            "coral_id": [int(u.rpartition("#")[2]) for u in uids],
        }
    )
    for name, values in columns.items():
        if len(values) != len(table):
            raise ValueError(f"column '{name}' has {len(values)} values for {len(table)} crops")
        table[name] = np.asarray(values)
    return table


def subsample(n: int, size: int = 15000, seed: int = SEED) -> np.ndarray:
    """Row indices of a random subset for plotting (CM cell 21: 15,000 points, seed 42)."""
    if n <= size:
        return np.arange(n)
    return np.random.RandomState(seed).choice(n, size=size, replace=False)


def cluster_sizes(labels: np.ndarray) -> pd.DataFrame:
    """Items and share per cluster, as the K-Means gallery labels them (CM cell 14)."""
    counts = pd.Series(labels).value_counts().sort_index()
    return pd.DataFrame(
        {"cluster": counts.index, "count": counts.to_numpy(), "percent": 100 * counts / len(labels)}
    ).reset_index(drop=True)


def sample_per_group(
    table: pd.DataFrame, group_col: str, n: int = 5, seed: int = SEED
) -> pd.DataFrame:
    """Up to `n` random rows of each group, groups in sorted order (the rows of CM cell 14)."""
    parts = [
        rows.sample(n=min(n, len(rows)), random_state=seed)
        for _, rows in table.groupby(group_col, sort=True)
    ]
    return pd.concat(parts, ignore_index=True) if parts else table.iloc[:0]
