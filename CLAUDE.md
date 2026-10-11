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
Images, mask JSONs, embeddings, label CSVs, the CoralSCOP checkpoint, `ground_truth_genus_counts.csv` and the expert test set live in a shared Google Drive. Ask the user for any file you need.
