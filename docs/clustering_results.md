# Clustering results (2026-10-14)

The exploration direction (embeddings plus clustering, outside the pipeline; see `docs/exploration_embeddings.md`) was run on the 16,080 kept crops of run `iou60_sta55_pts32`. Iteration 1 of the project (DINOv2 on all 44,211 masks, with mis-cut portrait crops) concluded "no genus separation". With correct, filtered crops and a reef-trained model, **the clusters carry a strong signal**: they separate growth forms, and they isolate the classifier's main error sources (non-coral objects) without any labels.

## Setup
| | |
|---|---|
| Crops | 16,080 kept crops (manifest filters: area ≥ 25,000 px, aspect ≤ 3, containment, blur), cut in each photo's own frame, portrait photos included |
| Features | **DinoVdeau-large** (`lombardata/DinoVdeau-large…`, reef-trained DINOv2), masked square crop at 518 px, `pooler_output`, 1024-d; file `features/dinovdeau-l__masked_square.npz` (T4, about 1 h, resumable). Independent of the classifier: it never saw genus names or BioCLIP. |
| Comparison | BioCLIP-FT embeddings (`features/finetuned-bioclip__masked.npz`, the classifier's own features) |
| Methods | PCA-50, then t-SNE and UMAP on 6,000 random crops; K-Means with 20 clusters (seed 42) on all crops; HDBSCAN with the legacy settings (500 / 50) |
| Coloring | the baseline's predicted genus. It is only painted onto the map and is meaningful for DinoVdeau, but circular for BioCLIP |
| Report | `reports/embeddings_dinovdeau-l__masked_square_2026-10-14/` (tables, maps, `crops.parquet` with each crop's cluster) |

## Numbers
- **Weak geometry overall:** K-Means cosine silhouette 0.11, Davies-Bouldin 2.99. HDBSCAN at 500 / 50 gives 3 clusters (71 / 1,858 / 14,151).
- **But the clusters are interpretable:**

| Cluster | Crops | Classifier's labels (top, second) | Mean confidence | **What the crops are** (12 random crops viewed) |
|---|---|---|---|---|
| 5 | 902 | Acropora 74%, Pocillopora 11% | 0.88 | Branching / corymbose coral from above. Acropora-looking ones at confidence 1.0; the rest split between Stylophora, Seriatopora and Pocillopora at 0.3-0.6 |
| 4 | 649 | Platygyra 51%, Favia 34% | 0.46 | Meandroid brain corals, Platygyra-like. Often low confidence or mislabeled (Favia, Echinopora, Pavona) |
| 19 | 989 | Platygyra 44%, Favia 27% | 0.44 | Brain corals, including elongated and ribbon-like pieces; some labeled Acropora or Pavona |
| 8 | 1,112 | Pavona 37%, Leptastrea 18% | **0.30** | **Bare substrate:** pale rock, dead coral, rubble, encrusted surfaces. Not coral |
| 14 | 65 | Lobophyllia 54%, Dipsastraea 18% | 0.42 | **Giant clam (Tridacna) mantles** from many photos. Not coral; about half of all Lobophyllia predictions |
| 0, 9, 10, 11, 12, 15 | about 5,500 | Pavona-led (16-33%) | 0.33-0.48 | Not yet viewed; likely more substrate (to check) |
| 1, 6 | 852, 666 | Favia / Favites, Favites / Platygyra | about 0.5 | Not yet viewed (massive complex) |
| 2, 7, 13, 16 | 589-843 | Acropora / Stylophora / Millepora mixes | 0.50-0.69 | Not yet viewed (branching) |
| 3, 17, 18 | 729-884 | Echinopora / Porites, Porites | 0.54-0.59 | Not yet viewed |

## What it shows
1. **An independent model agrees with the classifier on growth forms.** DinoVdeau groups branching corals, brain corals, Porites and Favites the way the classifier labels them. Two models that share nothing seeing the same groups is real signal.
2. **Within a growth form, genera are not separated.** In the branching cluster, Acropora, Stylophora, Seriatopora and Pocillopora mix; the massive complex stays mixed. Clusters are about form, not genus.
3. **The over-predictions are mostly non-coral.**
   - "Pavona" (15.1% of crops against 0.44% of surveyed colonies) is to a large extent bare substrate (cluster 8).
   - "Lobophyllia" is inflated by giant clams (cluster 14).
   - The classifier has no "other" class, so it must force these into a genus. Their low confidence is the tell.
4. **The under-prediction of Platygyra (0.33× the survey) is visible:** clear brain corals get low confidence or another genus.
5. **Low confidence is concentrated:** the classifier's uncertain crops gather in a few DINO regions instead of being spread out.
6. **Crop quality, seen in passing:**
   - colonies cut by the photo edge (a straight black side in the square crop);
   - some blurred crops;
   - the black background of the masked crop.

## Caveats
- Each cluster was judged from 12 random crops by eye (non-expert). Cluster contents need confirmation by Assaf, especially the genus inside the brain and branching clusters.
- The genus coloring is the classifier's prediction, which is about 67% accurate on the landscape expert rows, not ground truth.
- K-Means ids depend on the features and the settings (20 clusters, seed 42). Re-running with the same file gives the same ids.

## Next
- **View the remaining Pavona-led clusters** (`INSPECT = [0, 9, 10, 11, 12, 15]` in the last cell of `notebooks/explore/embeddings.ipynb`): how much of the 42% "Pavona mass" is substrate?
- **Have Assaf confirm the cluster grids** (minutes per cluster). That turns clusters into labels for the plan in `docs/algorithm_ideas.md` ("Proposed plan").
- **A number for the signal:** cluster purity / adjusted mutual information against the predicted genus (and against the expert set), compared with random clusters.
