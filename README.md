# Red Sea Coral Identifier

Identify the genus of individual corals in Red Sea reef photos. The pipeline first cuts each coral out of a reef image (segmentation), then labels it with a genus (classification), starting zero-shot and improving with expert corrections.

## Pipeline

| Stage | Notebook | What it does |
|---|---|---|
| 1. Segmentation | `notebooks/CoralMasking.ipynb` | Runs [CoralSCOP](https://github.com/zhengziqiang/CoralSCOP) (SAM fine-tuned for coral) over 1,714 reef images to produce per-coral masks, then extracts DINOv2 / DinoVdeau embeddings, t-SNE/UMAP views and clusters. |
| 2. Zero-shot labels | `notebooks/weak_classification.ipynb` | Classifies each mask crop into 20 Red Sea hard-coral genera with CLIP-family models (fine-tuned BioCLIP by default; BioCLIP 2 and SigLIP 2 also supported). |
| 3. Tiered labels and supervised classifier | `CoralClassification_v2_Pipeline.ipynb` (*not yet in this repo*) | Expands to 24 genera (adds 4 soft corals), auto-accepts confident predictions for reliable genera, sends the rest to expert review, then trains a logistic-regression classifier on the corrected labels. |
| 4. Genus audit | `notebooks/Millepora_Analysis.ipynb` | Precision audit and false-positive/negative analysis for *Millepora* (fire coral). |

Quality filters applied to masks: minimum area 25,000 px², aspect ratio at most 3, containment threshold 0.80, Laplacian sharpness at least 80. Source images are resized to 2560 x 1920.

## Status

- Phase 1 of the v2 pipeline is done: 14,224 corals classified across 24 genera. 3,058 were auto-accepted (confidence at least 0.85, ten reliable genera) and 11,166 were flagged for expert review.
- Phase 2 (supervised classifier) is written but waiting on the expert's corrected labels.
- *Millepora* audit: precision is only 58.9% on sharp, high-confidence crops (129 reviewed: 66 correct, 46 wrong, 17 uncertain).

Open questions: whether to demote *Millepora* to expert review, how to validate the new soft-coral genera, and how to collect more expert labels.

## Repository layout

```
notebooks/            Colab notebooks (see the table above)
third_party/CoralSCOP Submodule: upstream CoralSCOP segmentation model
docs/                 Onboarding guide with the full project map
```

## Getting started

```bash
git clone --recurse-submodules https://github.com/CoralsINPA/RedSeaCoralIdentifier
# already cloned without submodules?
git submodule update --init
```

Open a notebook in Colab via File > Open notebook > GitHub, or use
`https://colab.research.google.com/github/CoralsINPA/RedSeaCoralIdentifier/blob/<branch>/notebooks/<name>.ipynb`.

Notebooks run in Colab with a GPU and mount Google Drive.

## Data and model weights (not in this repo)

The images, mask JSONs, embeddings, label CSVs and the CoralSCOP checkpoint (`vit_b_coralscop.pth`) are too large for Git and live in a shared Google Drive. See [`docs/Coral_Pipeline_Onboarding_Guide.pdf`](docs/Coral_Pipeline_Onboarding_Guide.pdf) for the folder structure and where each file comes from.

## Notes

- Notebooks are committed with their saved outputs, so they are large (14-17 MB) and GitHub may not render them.
- The quality-filter helpers are copy-pasted across notebooks rather than shared.
