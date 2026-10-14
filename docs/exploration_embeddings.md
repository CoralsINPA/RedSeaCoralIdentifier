# Exploring embeddings (a separate direction)

Embeddings and clustering are **not part of the pipeline**. The pipeline is segment, manifest (filters), zero-shot baseline, evaluation and expert review, linear probe, and none of its modules, commands or notebooks needs anything here. This page says what exists for people who want to look at embeddings, and what it could be used for.

## What exists

| What | Where |
|---|---|
| The subpackage | `src/coralid/explore/`: `dino` (the DINOv2 and DinoVdeau image encoders), `embed` (the DINO embedding job over the manifest's kept masks), `clustering` (PCA, t-SNE, UMAP, K-Means, HDBSCAN with the legacy notebook's settings), `plots` (2-D maps colored by a number or a category), `analysis` (a features file joined to the baseline's predicted genus and the manifest's orientation, cluster summaries, the saved report) |
| The notebook | `notebooks/explore/embeddings.ipynb`. It explores the baseline's BioCLIP features by default, so it needs no GPU; `RUN_EMBED = True` runs the DINO job first (GPU) |
| The command | `python -m coralid explore-embed [--model dinov2-s14\|dinovdeau-l]` writes `features/<model>__masked_square.npz` for the kept masks. Resumable, never overwrites a file |
| The config | the `explore:` section of `configs/iou60_sta55_pts32.yaml` (the DINO model and batch size). Nothing else reads it |
| The extras | `pip install -e ".[dino]"` and `".[cluster]"`. The `clip`, `segment`, `probe` and `viz` extras do not need them |

**The pipeline never depends on it.** `tests/test_structure.py` fails if a module outside `coralid/explore/` imports it. The one exception is `cli.py`, which imports it inside the `explore-embed` command only. The notebook is outside the numbered pipeline notebooks (`01` to `04`), and `tests/test_notebooks.py` checks that none of them uses `coralid.explore`.

The genus colors on the maps are the baseline's zero-shot predictions, not labels: a cluster that is mostly one predicted genus shows that the features and the model agree, not that the genus is right.

Nothing here has run on the real data yet (the DINO job never ran; the notebook was dry-run on a synthetic Drive).

## History

Iteration 1 of the project (`notebooks/legacy/CoralMasking.ipynb`) embedded the CoralSCOP masks with DINOv2 and clustered them with K-Means and HDBSCAN. It found **no genus separation**, and the project moved to zero-shot classification with BioCLIP. That result is partly confounded (some of its crop galleries showed crops cut from the wrong part of the photo, see `docs/agent_memory/pipeline_history.md`), but the DinoVdeau silhouette scores did show only weak cluster structure. Generic embeddings are not expected to separate genera; this direction is for inspection, not for classification.

## Possible uses

Each is exploration. None changes a number of the pipeline.

1. **Cluster the baseline's BioCLIP embeddings to see what the model calls Pavona.** Pavona is 15.1% of the baseline's crops against 0.44% of the survey's colonies. A cluster table and a gallery of the Pavona-predicted crops show whether they are one kind of texture (flat or encrusting colonies, for example) before any prior is corrected (algorithm idea 5).
2. **Find soft-coral or junk clusters.** Soft corals (5 of 59 identifiable expert crops), sand, shadow and partial colonies may form clusters of their own. That would help to size the reject option (algorithm idea 3).
3. **Choose a diverse sample for the next expert set.** Sampling across clusters, confidence bands and photo orientation instead of at random gives the larger blind expert set (algorithm idea 1.1) more coverage per crop Assaf has to label.
4. **DinoVdeau features as one candidate input for the probe.** `explore-embed --model dinovdeau-l` writes a features file in the same format as the baseline's. `coralid.probe` can train on it by setting `probe.features`, and compare it with the BioCLIP embeddings on the same labels. This is a comparison to run once labels exist, not a pipeline step.

## Running it

```bash
pip install -e ".[dino,cluster,viz]"
python -m coralid explore-embed --config configs/iou60_sta55_pts32.yaml        # GPU, optional
```

or open `notebooks/explore/embeddings.ipynb` in Colab (about 15-25 minutes on a CPU on the BioCLIP features; estimates, not measured). Reports go to `reports/embeddings_<features>_<date>/` in a new folder every time.

## First results (2026-10-14)
On the BioCLIP features of the 16,080 kept crops (6,000 random, UMAP and t-SNE):
- **Distinctive genera form clean islands:** Acropora, Porites, Stylophora/Seriatopora, Platygyra, Millepora, Pocillopora.
- **Pavona sits in large low-confidence regions** (baseline confidence mostly below 0.2), mixed with the massive genera. It looks like the model's default for ambiguous crops.
- **Coloring by predicted genus is circular,** since the predictions come from these same embeddings. Judge clusters from the gallery.

Details are in `docs/agent_memory/sessions/2026-10-14_end_to_end_millepora_exploration.md`.

## DinoVdeau results (2026-10-14)

The full summary is in `docs/clustering_results.md`. In short: the independent DinoVdeau clusters separate growth forms and isolate non-coral objects (bare substrate labeled "Pavona", giant clams labeled "Lobophyllia"). That is the basis of the proposed plan in `docs/algorithm_ideas.md`.
