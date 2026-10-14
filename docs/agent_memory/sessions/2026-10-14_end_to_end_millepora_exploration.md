# 2026-10-14: End-to-end check, the new Millepora notebook, and the first look at the embeddings

## What was done
1. **End-to-end run of the reorganized notebooks in Colab (passed).**
   - **`01_segment_and_filter`:** the segmentation check is a PASS. Five photos, one of them portrait (IMG_4484), were re-segmented into `results_all/segcheck_iou60_sta55_pts32/`. All 109 saved masks were found again (51 of 51 large), with identical counts and frames and a mean IoU of 1.000. The filter funnel matches the record: 44,211 → 19,457 (area) → 18,945 (aspect) → 16,197 (containment) → 16,080 kept. Details and the overlay observations are in `2026-10-14_segcheck_and_thresholds.md`.
   - **`02_zero_shot`:** it loaded the saved baseline in seconds (never overwrites, so no recompute). Its new report `reports/baseline_finetuned-bioclip__taxonomy_v1__hard20__all_2026-10-14/expert_paired.csv` is identical to the 2026-10-12 one: landscape 34/51 = 66.7% with 0 predictions changed; portrait 0/3 with 7 of 10 changed (the portrait fix). A full re-run is not needed: nothing that affects predictions changed.
   - **Conclusion:** the organization is verified on real data.
2. **The thresholds:** the conclusions are recorded in `2026-10-14_segcheck_and_thresholds.md`. In short, set blur and size from Assaf's labels (after fixing the blur measure), keep containment, and change nothing before there are labels.
3. **New `notebooks/05_millepora.ipynb` (commit `432847f`).**
   - It is the Millepora analysis on the **correct predictions**, the 20-genus baseline. It stops unless `review_source.baseline_source(cfg)` confirms the `hard20` baseline of the config.
   - It prints counts by orientation and confidence band (0-0.5, 0.5-0.85, 0.85-1), the runner-ups and a confidence histogram.
   - It writes crop PDFs: 24 random predictions per band, and the 48 likely misses (Millepora as runner-up, closest first). Everything goes into a new `reports/millepora_<date>/`.
   - It replaces `legacy/Millepora_Analysis.ipynb`, which stays in `notebooks/legacy/` as history. Precision still needs the blind review of notebook 03, which is on hold.
   - **Verified:** lint clean; 585 tests pass, 6 skipped. Its table and sampling logic were run on the real manifest with synthetic predictions. **Not yet run in Colab.**
4. **The exploration notebook (`notebooks/explore/embeddings.ipynb`) was run by the user** on the BioCLIP features (`features/finetuned-bioclip__masked.npz`): t-SNE and UMAP on 6,000 random kept crops, K-Means.

## Observations from the embedding maps (user's screenshots)
- **Colored by predicted genus:**
  - Distinctive genera form clean islands: Acropora, Porites, Stylophora/Seriatopora, Platygyra, Millepora, Pocillopora.
  - **Pavona** (910 of 6,000) has no island of its own. It is smeared through the large central mass, mixed with Leptastrea, Favia, Echinopora and Favites.
  - The massive complex overlaps heavily; Favites is relatively tight.
- **Caveat:** the coloring is circular. The predictions come from the same embeddings, so color agreement shows structure, not correctness.
- **Colored by baseline confidence (UMAP):**
  - **The large left mass and the long bottom stalk have very low confidence (mostly below 0.2).** These are the regions where Pavona and the massive genera mix.
  - The distinct islands have high confidence (above 0.8).
  - **Reading:** Pavona over-prediction (15.1% of crops against 0.44% in the survey) looks like the model's default for ambiguous, low-confidence crops, not a real Pavona population.
- **K-Means (50 clusters) splits the low-confidence regions into a few big clusters.** On the map, cluster 0 is the left mass, and the teal and blue clusters (around 13-20) are the bottom stalk. These are the clusters to inspect in the gallery PDF.
- **Compared with the old exploration** (`legacy/CoralMasking.ipynb`, DINOv2 / DinoVdeau on all 44,211 masks, no genus separation), the genera separate here because:
  - the features are BioCLIP fine-tuned on reef taxonomy, not general-purpose DINO;
  - only the 16,080 filtered crops are used;
  - crops are cut in the mask frame.
- **For the probe:** these features carry real genus structure, which is good news for the linear probe (idea 4).

## Where the original exploration lives
- `notebooks/legacy/CoralMasking.ipynb`, unchanged: CoralSCOP segmentation, DINOv2 and DinoVdeau embeddings, t-SNE, UMAP, K-Means and HDBSCAN, with its outputs.
- The new exploration is `notebooks/explore/embeddings.ipynb` and `coralid.explore` (see `docs/exploration_embeddings.md`). The pipeline never depends on it.

## Next
- **Open the cluster gallery PDF** (`reports/embeddings_finetuned-bioclip__masked_<date>/`) for the low-confidence clusters (0, and the bottom stalk) and see what the crops are: sand, rubble, soft coral, encrusting growth, blur, or real Pavona. Free; it answers the Pavona question and shows what a reject / "other" option must catch.
- **For a fairer comparison with the old DINO run,** judge cluster purity from the gallery, not from the genus coloring.
- **Run `05_millepora`** and paste the overview.
- **Decisions pending from the user:**
  - keep or delete `weak_classification.ipynb` and `CoralMasking.ipynb` in `legacy/` (they are kept for now);
  - a label per colony, or genus composition per site;
  - the optional `sharpness_crop = "bbox"` setting;
  - the Drive cleanup of `cleaned_images_crops/` and `cleaned_jsons_filtered/`.
- **Then the algorithm phase,** starting from `docs/algorithm_ideas.md`.

## Addendum: HDBSCAN on the BioCLIP features (legacy settings: min_cluster_size 500, min_samples 50)
- **Result:** 2 clusters and 65 noise crops (0.4%); silhouette 0.38, Davies-Bouldin 0.67.

  | Cluster | Crops | Top genus | Second | Mean confidence |
  |---|---|---|---|---|
  | 0 | 1,164 (7%) | Acropora 86% | Pocillopora 14% | 0.98 |
  | 1 | 14,851 (92%) | Pavona 16% | Favia 10% | 0.46 |
  | noise | 65 | Lobophyllia 60% | Dipsastraea 17% | 0.45 |

- **Reading:** with these settings, made for 44,211 masks, HDBSCAN only finds the branching corals (Acropora / Pocillopora) as a separate dense group. Everything else is one continuum at low confidence. The silhouette is high only because there are just two clusters, so it says little.
- **K-Means (50 clusters) is the more useful view.** To use HDBSCAN, lower `HDBSCAN_MIN_CLUSTER` to about 50 and `HDBSCAN_MIN_SAMPLES` to about 10.
- **Worth a look:** the tight isolated noise blob (65 crops, mostly Lobophyllia). It may be near-duplicate crops of one colony across overlapping photos, or an artifact.

## Addendum: DinoVdeau features (the independent direction)
- **Embedding job:** `explore-embed --model dinovdeau-l` ran on a T4 in Colab: 16,080 crops × 1024 → `features/dinovdeau-l__masked_square.npz`. It resumed once after a CPU-only runtime was detected. Report: `reports/embeddings_dinovdeau-l__masked_square_2026-10-14/`.
- **K-Means, 20 clusters:** cosine silhouette 0.11 and Davies-Bouldin 2.99, so the cluster structure is weak overall.
  - **Only one pure cluster:** cluster 5 (902 crops), Acropora 74%, confidence 0.88.
  - **Brain-coral clusters:** 4 and 19 (1,638 crops), Platygyra 44-51% plus Favia 27-34%. The classifier splits a single DINO look into Platygyra and Favia, which may explain part of the Favia over-prediction (1.6× the survey) and the Platygyra shortfall (0.33×).
  - **Branching clusters:** 2, 7, 13 and 16, mixing Acropora, Stylophora and Millepora.
  - **Seven clusters led by Pavona** (0, 8, 9, 10, 11, 12, 15; about 6,700 crops, 42%): Pavona at 16-37%, mean confidence 0.30-0.48. The low-confidence mass is a large part of the data.
  - **Cluster 14:** 65 crops, Lobophyllia 54%.
- **HDBSCAN (still the legacy 500 / 50 settings):** 3 clusters, no noise.
  - Brain-coral group: 1,858 crops (Platygyra 45%, Favia 27%).
  - Lobophyllia-led blob: 71 crops.
  - Everything else: 14,151 crops.
- **The same Lobophyllia-led blob of about 65-71 crops appears in both BioCLIP and DinoVdeau,** two independent models. That is about half of all Lobophyllia predictions (146) in one tight group. Likely near-duplicate crops (one colony across overlapping photos) or one repeated object. Check its image names in `crops.parquet` and the gallery: duplicates would inflate counts and leak between training and test folds in the probe.
- **Reading:** an independent model agrees with the classifier on growth forms (branching, brain, Acropora), not on genera within them. The Pavona-led low-confidence mass is real structure in both feature spaces.
