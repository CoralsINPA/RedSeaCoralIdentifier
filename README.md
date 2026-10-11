# Red Sea Coral Identifier

Identify the genus of individual corals in Red Sea reef photos. The pipeline first cuts each coral out of a reef image (segmentation), then labels it with a genus (classification).

## Pipeline

| Stage | Notebook | What it does |
|---|---|---|
| 1. Segmentation | `notebooks/CoralMasking.ipynb` | Runs [CoralSCOP](https://github.com/zhengziqiang/CoralSCOP) (SAM fine-tuned for coral) over 1,714 reef images to produce per-coral masks, then extracts DINOv2 / DinoVdeau embeddings, t-SNE/UMAP views and clusters. |
| 2. Zero-shot classification | `notebooks/weak_classification.ipynb` | Classifies each mask crop into 20 Red Sea hard-coral genera with CLIP-family models (fine-tuned BioCLIP by default; BioCLIP 2 and SigLIP 2 also supported). |
| 3. Genus audit | `notebooks/Millepora_Analysis.ipynb` | Precision audit and false-positive/negative analysis for *Millepora* (fire coral). |

Quality filters applied to masks: minimum area 25,000 px², aspect ratio at most 3, containment threshold 0.80, Laplacian sharpness at least 80. Source images are resized to 2560 x 1920.

## Finding

The *Millepora* audit found precision of only 58.9% on sharp, high-confidence (at least 0.85) crops: 129 reviewed, 66 correct, 46 wrong, 17 uncertain.

## Repository layout

```
notebooks/            Colab notebooks
third_party/CoralSCOP Submodule: upstream CoralSCOP segmentation model
docs/                 Onboarding guide with the full project map
```

## Getting started

```bash
git clone --recurse-submodules https://github.com/CoralsINPA/RedSeaCoralIdentifier
# already cloned without submodules?
git submodule update --init
```

## Data and model weights (not in this repo)

The images, mask JSONs, embeddings, label CSVs and the CoralSCOP checkpoint are stored in a shared Google Drive. See [`docs/Coral_Pipeline_Onboarding_Guide.pdf`](docs/Coral_Pipeline_Onboarding_Guide.pdf) for the folder structure.

The quality-filter helpers are copy-pasted across notebooks rather than shared.
