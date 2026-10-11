# CLAUDE.md

Genus-level coral identification for Red Sea reef photos: segment corals (CoralSCOP/SAM), then classify each crop (zero-shot CLIP-family, then a supervised classifier on expert-corrected labels). See `README.md` for the overview and `docs/Coral_Pipeline_Onboarding_Guide.pdf` for the full Drive/notebook map.

## Layout
- `notebooks/CoralMasking.ipynb`: segmentation + DINOv2/DinoVdeau embeddings + t-SNE/UMAP/K-Means.
- `notebooks/weak_classification.ipynb`: zero-shot genus labels (20 hard-coral genera). The "breakthrough" notebook. Cells 25-34 are scratch/debug.
- `notebooks/Millepora_Analysis.ipynb`: Millepora precision audit.
- `CoralClassification_v2_Pipeline.ipynb` (24 genera, tiered labeling, LogisticRegression) is NOT in the repo yet.
- `third_party/CoralSCOP`: git submodule (upstream zhengziqiang/CoralSCOP, pinned). Do not edit; bump the pin deliberately.

## Key facts
- Active run name: `iou60_sta55_pts32` (SAM points_per_side 32, pred_iou 0.60, stability 0.55, min_mask_region_area 100).
- Filters: MIN_AREA 25,000, MAX_ASPECT_RATIO 3.0, CONTAINMENT_THRESH 0.80, SHARPNESS_THRESHOLD 80. Images are resized to 2560x1920.
- Default model: ReefNet fine-tuned BioCLIP (ViT-B/16, 512-dim embeddings); alternatives are BioCLIP 2 and SigLIP 2.
- v2 tiers: Tier 1 (10 reliable genera, confidence >= 0.85) is auto-accepted; Tier 2 (10 confused hard-coral genera + 4 soft corals) goes to expert review.
- Numbers: 44,211 annotations, 16,197 after filtering, 14,224 after sharpness; 3,058 auto-accepted, 11,166 for review.
- Millepora precision is 58.9% on sharp crops at confidence >= 0.85, so it may need demoting to Tier 2.
- Phase 2 of v2 runs after the expert's `expert_review_completed.csv` arrives: set `PHASE = 2`.
- Expert test set: 78 hand-labeled crops (`assaf_expert_test_set.csv`), joined on `(image_name, coral_id)`.

## Not in the repo
Images, mask JSONs, embeddings, label CSVs, the CoralSCOP checkpoint, `ground_truth_genus_counts.csv` and the expert test set live in a shared Google Drive. The Drive connector has been returning Forbidden, so ask the user for any file you need. Never commit large data.

## Conventions
- Notebooks are committed with outputs (14-17 MB). Prefer stripping outputs (`nbstripout`) before committing new ones.
- Filter helpers (`decode_rle`, `filter_annotations`, `get_crop`, `get_masked_crop`, `is_sharp_enough`) are duplicated across notebooks; keep them in sync or factor them out.
- Develop on the branch the session assigns; commit and push there. Open PRs only when asked.
- The repo may be public: don't add Drive IDs or personal details to committed files.
