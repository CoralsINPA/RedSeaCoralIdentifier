# CLAUDE.md

Genus-level coral identification for Red Sea reef photos: segment corals (CoralSCOP/SAM), then classify each crop with zero-shot CLIP-family models. See `README.md` for the overview and `docs/Coral_Pipeline_Onboarding_Guide.pdf` for the full Drive/notebook map.

## Layout
- `notebooks/CoralMasking.ipynb`: segmentation + DINOv2/DinoVdeau embeddings + t-SNE/UMAP/K-Means.
- `notebooks/weak_classification.ipynb`: zero-shot genus labels (20 hard-coral genera). The "breakthrough" notebook. Cells 25-34 are scratch/debug.
- `notebooks/Millepora_Analysis.ipynb`: Millepora precision audit.
- `third_party/CoralSCOP`: git submodule (upstream zhengziqiang/CoralSCOP, pinned).

## Key facts
- Active run name: `iou60_sta55_pts32` (SAM points_per_side 32, pred_iou 0.60, stability 0.55, min_mask_region_area 100).
- Filters: MIN_AREA 25,000, MAX_ASPECT_RATIO 3.0, CONTAINMENT_THRESH 0.80, SHARPNESS_THRESHOLD 80. Images are resized to 2560x1920.
- Default model: ReefNet fine-tuned BioCLIP (ViT-B/16); alternatives are BioCLIP 2 and SigLIP 2.
- Millepora precision is 58.9% on sharp crops at confidence >= 0.85.
- Expert test set: 78 hand-labeled crops (`assaf_expert_test_set.csv`), joined on `(image_name, coral_id)`.

## Not in the repo
Images, mask JSONs, embeddings, label CSVs, the CoralSCOP checkpoint, `ground_truth_genus_counts.csv` and the expert test set live in a shared Google Drive. Use the Google Drive MCP tools if connected (search by `parentId`), otherwise ask the user for any file you need.

## Drive map
Root `CoralsMasking`: https://drive.google.com/drive/folders/1LtqWRm2vieTbUGgb1ISfsqVD-g-urPRY
Folder IDs below are for `parentId = '...'` queries.

Relevant:
- Active run `results_all/iou60_sta55_pts32` (id `1up9-VQJEPyv5p8YbPCPgmO5yVx-SCz3r`). Contains `json/`, `cleaned_images_crops/`, `cluster_previews/`, `visualizations/` (subfolder contents not inspected), `coral_dinoV2_encodings.csv` (181 MB), `coral_dinoV2_encodings_MASKS.csv` (489 MB), and `v2_pipeline/`.
- `assaf_expert_test_set.csv` (id `1PIBGXIduvOnRn3lvsTT2ZyFtjKiUVRRD`), in the active run folder. Columns: `image_name, coral_id, bbox, area`, 20 genus probabilities, `predicted_genus, confidence, expert_genus, is_correct, review_source`. `expert_genus` can be `unidentifiable` or `MISSING_LABEL`, or a soft coral (Cladiella, Sinularia, Litophyton) that is outside the 20 target genera. Exclude these when computing accuracy.
- `ground_truth_genus_counts.csv` (id `1DbQaZa-3lTehQPP-7PmjAkiCkbvuadwd`), in the root. Columns: `Category, genus, count, Pi`. It includes soft corals and genera outside the 20 targets, so filter to the 20 before comparing.
- `coral_weak_labels_finetuned-bioclip_taxonomy_masked_best.csv` (id `1oB15CLt1FEQfbp4Xai1xlWs-hsiJRYqC`): best zero-shot labels.
- `v2_pipeline/` (id `1pJq-dvipNiprBiQRXLAqH_iXTbEcjuNG`): `v2_weak_labels_expanded.csv`, `v2_embeddings.npz`, Millepora audit files (`millepora_sharp_audit`, review PDFs), `expert_review_for_assaf.csv`, `expert_review_images/`.
- Checkpoint: `CoralSCOP/checkpoints/vit_b_coralscop.pth` (id `1tiN32L4h4A5cIZAf1jeP9q8Euf-9U0Eo`, 382 MB).
- `cleaned_jsons_filtered` (id `1ATUrW-DLZMxGYio6N-9bqVuonFKkQZK4`): filtered mask JSONs.
- `data_all` (id `1oj6WQ9jNW6vUGe18PyvT5eniaASBGh68`): all source photos. `data_50` (id `1s7ZONaUeHH0BAhnH0UBH8l67X8HrBUq5`): 50-image subset.
- Colab: CoralMasking https://colab.research.google.com/drive/1sS9aUIWFBk6EP54biw_vGc3n3oyw51H7, WeakClassification https://colab.research.google.com/drive/1hCcbbjEHMtZQEm0TXiYwU0q-E_ZnIX4K, Millepora_Analysis https://colab.research.google.com/drive/18TE7K8VihAIzcgaQ1ULCMBTlHpqtPACS.

Ignore:
- `external_libs/`: vendored pip packages (tqdm, pyparsing, tabulate, ...).
- `CoralSCOP/` in Drive (including `.git`, `segment_anything`, `figs`, `demo_imgs`): duplicate of the submodule. Only the checkpoint is needed.
- `results_50/`: early parameter sweeps on 50 images, superseded by `iou60_sta55_pts32`.
- `cropped_masks/`: empty when checked.
- Root-level `coral_dino_encodings.csv` (1.2 MB) and the 2026-01/03 `coral_dino_*` CSVs elsewhere in My Drive: older, smaller runs.
- The embedding CSVs are 180-490 MB. Don't read them through the Drive tool, since downloading is the only practical way.

Notes: the Drive onboarding guide calls the notebook `WeakClassification.ipynb`; in the repo it is `weak_classification.ipynb`.
