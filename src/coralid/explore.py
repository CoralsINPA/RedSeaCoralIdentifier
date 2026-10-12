"""Exploring stored embeddings: join them to the baseline, summarize clusters, save the results.

The computations are in `coralid.clustering` (PCA, t-SNE, UMAP, K-Means, HDBSCAN) and the plots in
`coralid.viz`; this module is the glue the exploration notebook (`02_embeddings_explore`) needs:

- `load_exploration` reads a features file of the run (`features/<name>.npz`, for example the
  baseline's `finetuned-bioclip__masked` or the DINO job's `dinov2-s14__masked_square`) and joins
  each crop to the baseline's predicted genus and confidence and to the manifest's orientation,
  area and sharpness, in the order of the feature rows;
- `cluster_genus_table` says what each cluster is made of, by the baseline's genus labels;
- `explore_summary` and `write_explore_report` save the results as a new folder under `reports/`.

The genus is the **baseline's zero-shot prediction**, not a label: a cluster that is 90% one
predicted genus shows that the features and the model agree, not that the genus is right.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from coralid.baseline import baseline_paths
from coralid.clustering import (
    HdbscanResult,
    KMeansAnalysis,
    PcaResult,
    embedding_table,
)
from coralid.config import Config
from coralid.manifest import read_manifest
from coralid.orientation import add_orientation
from coralid.predictions import normalize_keys
from coralid.reports import new_report_folder
from coralid.zero_shot import load_embeddings

PREDICTION_COLUMNS = ("predicted_genus", "confidence", "margin")
MANIFEST_COLUMNS = ("area", "sharpness", "visible_px")


@dataclass
class Exploration:
    """Features and the table that describes each crop (row-aligned with `X`)."""

    features_path: Path
    X: np.ndarray
    table: pd.DataFrame  # uid, image_name, coral_id, predicted_genus, confidence, orientation, ...
    notes: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """The features file's name without `.npz`."""
        return self.features_path.stem


def features_path(cfg: Config, features: str) -> Path:
    """`<run>/features/<features>.npz` (a bare stem, not a path)."""
    if os.sep in features or features.endswith(".npz"):
        raise ValueError(
            f"give the file stem such as 'finetuned-bioclip__masked', got {features!r}"
        )
    return cfg.features_dir / f"{features}.npz"


def load_exploration(
    cfg: Config,
    features: str,
    predictions: pd.DataFrame | None = None,
    manifest: pd.DataFrame | None = None,
) -> Exploration:
    """Read a features file and describe its crops with the baseline and the manifest.

    `predictions` (default: the saved baseline, if there is one) adds `predicted_genus`,
    `confidence` and `margin`; `manifest` (default: the run's, if saved) adds `orientation`,
    `area`, `sharpness` and `visible_px`. What is missing becomes `unknown` or NaN, and a note says
    so. Row `i` of the table is row `i` of `X`.
    """
    path = features_path(cfg, features)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The baseline's features come from notebooks/03_zero_shot.ipynb "
            "(finetuned-bioclip__masked); DINO features from the embedding job "
            "(python -m coralid embed)."
        )
    uids, X = load_embeddings(path)
    table = embedding_table(uids)
    notes: list[str] = []

    if predictions is None and baseline_paths(cfg).predictions.exists():
        predictions = pd.read_parquet(baseline_paths(cfg).predictions)
    if predictions is not None:
        wanted = [c for c in PREDICTION_COLUMNS if c in predictions.columns]
        joined = normalize_keys(predictions)[["image_name", "coral_id", *wanted]]
        table = table.merge(joined, on=["image_name", "coral_id"], how="left")
        missing = int(table["predicted_genus"].isna().sum())
        if missing:
            notes.append(f"{missing} crop(s) have no baseline prediction (genus 'unknown')")
    else:
        notes.append("no baseline predictions found: the genus is 'unknown' for every crop")
        table["predicted_genus"] = np.nan
    table["predicted_genus"] = table["predicted_genus"].fillna("unknown").astype(str)

    if manifest is None and cfg.manifest_path.exists():
        manifest = read_manifest(cfg.manifest_path)
    if manifest is not None:
        wanted = [c for c in MANIFEST_COLUMNS if c in manifest.columns]
        table = table.merge(
            manifest[["image_name", "coral_id", *wanted]], on=["image_name", "coral_id"], how="left"
        )
        table = add_orientation(table, manifest)
    else:
        notes.append("no manifest found: no orientation, area or sharpness")
    if len(table) != len(X):  # a duplicated key in the predictions or manifest would do this
        raise ValueError(f"the table has {len(table)} rows for {len(X)} feature rows")
    return Exploration(path, X, table, notes)


def cluster_genus_table(
    table: pd.DataFrame, cluster_col: str = "cluster", genus_col: str = "predicted_genus"
) -> pd.DataFrame:
    """What each cluster is made of: size, the two commonest predicted genera, mean confidence.

    One row per cluster value (numbers in numeric order, `noise` or -1 included).
    `top_share` is the purity: 1.0 means every crop of the cluster has the same predicted genus.
    """
    rows = []
    for value, part in table.groupby(cluster_col, sort=True):
        shares = part[genus_col].value_counts(normalize=True)
        second = shares.index[1] if len(shares) > 1 else ""
        rows.append(
            {
                cluster_col: value,
                "n": len(part),
                "share": len(part) / len(table),
                "top_genus": shares.index[0],
                "top_share": float(shares.iloc[0]),
                "second_genus": second,
                "second_share": float(shares.iloc[1]) if len(shares) > 1 else 0.0,
                "mean_confidence": float(part["confidence"].mean())
                if "confidence" in part and part["confidence"].notna().any()
                else None,
            }
        )
    return pd.DataFrame(rows)


def genus_cluster_table(
    table: pd.DataFrame, cluster_col: str = "cluster", genus_col: str = "predicted_genus"
) -> pd.DataFrame:
    """For each predicted genus: how many crops, and the cluster that holds most of them."""
    rows = []
    for genus, part in table.groupby(genus_col, sort=True):
        shares = part[cluster_col].value_counts(normalize=True)
        rows.append(
            {
                genus_col: genus,
                "n": len(part),
                "n_clusters": int(part[cluster_col].nunique()),
                "main_cluster": shares.index[0],
                "main_cluster_share": float(shares.iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def explore_summary(
    data: Exploration,
    pca: PcaResult | None = None,
    kmeans: KMeansAnalysis | None = None,
    hdbscan_result: HdbscanResult | None = None,
    n_map: int | None = None,
) -> dict[str, Any]:
    """A JSON-friendly summary of an exploration (what was run and its quality scores)."""
    summary: dict[str, Any] = {
        "features": str(data.features_path),
        "n_crops": len(data.table),
        "dim": int(data.X.shape[1]),
        "n_photos": int(data.table["image_name"].nunique()),
        "n_in_maps": n_map,
        "notes": data.notes,
        "predicted_genus_counts": {
            str(k): int(v) for k, v in data.table["predicted_genus"].value_counts().items()
        },
    }
    if pca is not None:
        summary["pca"] = {
            "n_components": int(pca.Z.shape[1]),
            "variance_kept": pca.total_variance,
        }
    if kmeans is not None:
        summary["kmeans"] = {
            "n_clusters": int(len(np.unique(kmeans.labels))),
            "silhouette_cosine": kmeans.silhouette,
            "davies_bouldin": kmeans.davies_bouldin,
            "mean_compactness": float(kmeans.compactness.mean()),
        }
    if hdbscan_result is not None:
        summary["hdbscan"] = {
            "n_clusters": hdbscan_result.n_clusters,
            "n_noise": hdbscan_result.n_noise,
            "silhouette": hdbscan_result.silhouette,
            "davies_bouldin": hdbscan_result.davies_bouldin,
            "sizes": {str(k): v for k, v in hdbscan_result.sizes.items()},
        }
    return summary


def write_explore_report(
    reports_dir: str | os.PathLike[str],
    name: str,
    tables: dict[str, pd.DataFrame],
    figures: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    date: str | None = None,
) -> Path:
    """Save an exploration as new files in `<reports_dir>/embeddings_<name>_<date>/`; return it.

    Every table is a `<key>.csv` (a table with more than 5,000 rows is a `<key>.parquet`),
    every figure a `<key>.png`, and `summary.json` holds the summary. An existing folder gets
    a `_2`, `_3`... suffix; no file is ever overwritten.
    """
    folder = new_report_folder(reports_dir, "embeddings", name, date)
    for key, table in tables.items():
        if len(table) > 5000:
            table.to_parquet(folder / f"{key}.parquet", index=False)
        else:
            table.to_csv(folder / f"{key}.csv", index=False)
    for key, figure in (figures or {}).items():
        figure.savefig(folder / f"{key}.png", dpi=110, bbox_inches="tight")
    with open(folder / "summary.json", "x", encoding="utf-8") as f:
        f.write(json.dumps(summary or {}, indent=2, default=str) + "\n")
    return folder
