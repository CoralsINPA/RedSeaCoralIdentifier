"""Exploration of embeddings: a separate direction, not part of the pipeline.

The pipeline is segment -> manifest -> zero-shot baseline -> evaluation and expert review ->
linear probe. Nothing in it imports this subpackage, except the `explore-embed` command of
`coralid.cli`, which imports it lazily. See `docs/exploration_embeddings.md`.

- `dino`: the DINOv2 and DinoVdeau image encoders;
- `embed`: the DINO embedding job over the manifest's kept masks;
- `clustering`: PCA, t-SNE, UMAP, K-Means and HDBSCAN with the legacy notebook's settings;
- `plots`: 2-D maps colored by a number or a category;
- `analysis`: a features file joined to the baseline's predicted genus, cluster summaries and
  the saved exploration report.

Importing the subpackage imports none of the heavy dependencies; each module loads them inside
its functions (extras `dino` and `cluster`).
"""
