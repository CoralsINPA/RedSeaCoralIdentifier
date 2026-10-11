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

The images, mask JSONs, embeddings, label CSVs and the CoralSCOP checkpoint are stored in a shared Google Drive (access is by invitation). See [`docs/Coral_Pipeline_Onboarding_Guide.pdf`](docs/Coral_Pipeline_Onboarding_Guide.pdf) for the pipeline walkthrough.

Root folder: [CoralsMasking](https://drive.google.com/drive/folders/1LtqWRm2vieTbUGgb1ISfsqVD-g-urPRY)

### Start here

| What | Link |
|---|---|
| Active run folder (`results_all/iou60_sta55_pts32`): masks, crops, labels, embeddings | [folder](https://drive.google.com/drive/folders/1up9-VQJEPyv5p8YbPCPgmO5yVx-SCz3r) |
| Expert test set (78 hand-labeled crops) | [`assaf_expert_test_set.csv`](https://drive.google.com/file/d/1PIBGXIduvOnRn3lvsTT2ZyFtjKiUVRRD/view) |
| Ground-truth genus counts from the expert survey | [`ground_truth_genus_counts.csv`](https://drive.google.com/file/d/1DbQaZa-3lTehQPP-7PmjAkiCkbvuadwd/view) |
| Best zero-shot labels (fine-tuned BioCLIP, taxonomy prompts, masked) | [`coral_weak_labels_finetuned-bioclip_taxonomy_masked_best.csv`](https://drive.google.com/file/d/1oB15CLt1FEQfbp4Xai1xlWs-hsiJRYqC/view) |
| CoralSCOP checkpoint | [`vit_b_coralscop.pth`](https://drive.google.com/file/d/1tiN32L4h4A5cIZAf1jeP9q8Euf-9U0Eo/view) (382 MB) |
| Filtered mask JSONs | [`cleaned_jsons_filtered`](https://drive.google.com/drive/folders/1ATUrW-DLZMxGYio6N-9bqVuonFKkQZK4) |
| Onboarding guide (Drive copy) | [PDF](https://drive.google.com/file/d/1gr2tnBhclsnh8s2oM49YKLWSYD9-VyGg/view) |

Colab notebooks: [CoralMasking](https://colab.research.google.com/drive/1sS9aUIWFBk6EP54biw_vGc3n3oyw51H7), [WeakClassification](https://colab.research.google.com/drive/1hCcbbjEHMtZQEm0TXiYwU0q-E_ZnIX4K), [Millepora_Analysis](https://colab.research.google.com/drive/18TE7K8VihAIzcgaQ1ULCMBTlHpqtPACS).

### Inside the active run folder

| Item | Contents |
|---|---|
| `json/` | Mask JSONs (contents not inspected) |
| `cleaned_images_crops/` | Cleaned crops (contents not inspected) |
| `cluster_previews/`, `visualizations/` | Embedding and cluster plots |
| `coral_dinoV2_encodings.csv` (181 MB), `coral_dinoV2_encodings_MASKS.csv` (489 MB) | DINOv2 embeddings, whole crop and masked crop |
| [`v2_pipeline/`](https://drive.google.com/drive/folders/1pJq-dvipNiprBiQRXLAqH_iXTbEcjuNG) | Expanded weak labels (`v2_weak_labels_expanded.csv`, `v2_embeddings.npz`) and the Millepora audit: `millepora_sharp_audit` sheet, review PDFs, `expert_review_for_assaf.csv`, and [`expert_review_images/`](https://drive.google.com/drive/folders/1QejgDnzmMVtoacZ5agqIEwe39UctKeG2) |

### Other folders (lower priority)

| Folder | Contents |
|---|---|
| [`data_all`](https://drive.google.com/drive/folders/1oj6WQ9jNW6vUGe18PyvT5eniaASBGh68) | Full set of source reef photos |
| [`data_50`](https://drive.google.com/drive/folders/1s7ZONaUeHH0BAhnH0UBH8l67X8HrBUq5) | 50-image subset for quick tests |
| [`results_50`](https://drive.google.com/drive/folders/11ogV1vfd9m-28fq3fHNiZjws_GkBwYoN) | Early SAM parameter sweeps on the 50-image subset (iou55, iou60, iou65, iou72). Superseded by the active run |
| [`CoralSCOP`](https://drive.google.com/drive/folders/1rTIFnYg52cQ0jsMd4gHYOp_o69Vg5_rV) | Copy of the upstream repo plus the checkpoint. Prefer the `third_party/CoralSCOP` submodule |

`external_libs/` holds vendored pip packages and `cropped_masks/` was empty when checked. Ignore both.

The quality-filter helpers are copy-pasted across notebooks rather than shared.
