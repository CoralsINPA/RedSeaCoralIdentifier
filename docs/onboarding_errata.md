# Onboarding guide: errata

`docs/legacy/Coral_Pipeline_Onboarding_Guide.pdf` (dated 2026-09-18) can't be edited here, so this file lists where it disagrees with the code and the data. Each item says how it was checked. Cell numbers are 0-based positions in the notebooks (`notebooks/legacy/`): **CM** = `CoralMasking.ipynb`, **WC** = `weak_classification.ipynb`, **MA** = `Millepora_Analysis.ipynb`. (The V2 notebook was excluded and removed on 2026-10-13; git history keeps it, and this file no longer cites its cells.)

## 1. Containment removes the larger mask, not the smaller one

The guide (Notebook 1, quality filters) says that if one mask is 80% or more contained in another, "the smaller one is considered a duplicate and removed".

The code does the opposite: when a small mask lies at least 80% inside a larger one, the **larger** mask is dropped (`to_remove.add(a_large['id'])` in WC 8). WC 7 says this is intended: it "keeps the finer-grained segment". The ratio is `|small ∩ large| / small["area"]`, using the annotation's `area` field.

A side effect worth knowing: the larger mask is removed before the sharpness check. If the smaller mask that remains is then rejected as blurry, the colony is lost altogether (plan section 6).

## 2. The folder structure section does not match the Drive

The guide's "Data and folder structure" lists `Coral_Segmentation_Outputs/` (`reef_masks/`, `coral_crops/`, `features/` with `.npy` files, `metadata/`) and `Coral_Classification/` (`genera_taxonomy.json`, `checkpoints/`, ...). None of these showed up in the Drive listings checked on 2026-10-11. The accurate map is the Drive map in `CLAUDE.md`. In short:

- Everything sits under `CoralsMasking/`, with photos in `data_all/` and each segmentation run in `results_all/<run name>/` (active run: `iou60_sta55_pts32`).
- Mask JSONs are in `json/`. There is no `reef_masks/` folder.
- DINOv2 features are CSV files (for example `coral_dinoV2_encodings_MASKS.csv`), not `.npy`.
- The genus prompts are Python dicts inside the notebooks. There is no `genera_taxonomy.json`.
- The classification notebook does not read a folder of crops (guide, Step 2, point 4). It reads the mask JSONs and the original photos, and cuts each crop in memory (WC 14).
- `cleaned_images_crops/` and `cleaned_jsons_filtered/` exist but are defective, so don't use them. The harvester reads the unfiltered JSONs, and several cells crop the full-size photo with bboxes from the smaller mask frame (plan section 2.1, #1-2).

## 3. The recommended prompt format is not what the best runs use

The guide says "start with `ensemble` for best results" (Step 2, point 3). The default in the code and in `configs/` is `taxonomy` (`taxonomy_v1`). `ensemble` is still available as an option.

## 4. Where the headline numbers come from

The guide's "What's been done" says zero-shot classification ran with `WeakClassification.ipynb` on 14,224 crops "across 20 hard coral genera", with a mean confidence of 0.485. Those numbers do not come from `WeakClassification.ipynb`. They come from a run of a different pipeline with **24 genera** (the 20 hard corals plus 4 soft corals) that has since been excluded from the project. They are not a reference. The 20-genus baseline of 2026-10-12 has 16,080 crops on 1,714 photos with a mean confidence of 0.494.

The runs saved from `weak_classification.ipynb` itself use a sample of 50 images and 842 crops (WC 14 and 23 outputs).

## 5. The guide's Millepora precision (58.9%) is discarded

The guide says "among 129 sharp, high-confidence Millepora predictions, only 58.9% were correct". Do not use it. It was a non-expert analyst's agreement rate with the model's answer visible, on the predictions of the excluded 24-genus pipeline, on a slice chosen by confidence (0.85 or more) and by a sharpness measure that is not the pipeline's. It is not a precision, and it says nothing about the 20-genus baseline. **There is no valid Millepora precision figure yet.** The blind review sheet (notebook `03_expert_review`, on hold) redoes the check on the 20-genus baseline.

## 6. Smaller points

- **Notebook name.** The guide says `WeakClassification.ipynb`. In the repo it is `weak_classification.ipynb`.
- **Where the notebooks are.** The guide says the four notebooks are on Drive. The legacy notebooks are frozen reference copies in `notebooks/legacy/`. The Drive copies can be newer, so compare modified times before trusting either.
- **`best` is not a configuration.** In `coral_weak_labels_finetuned-bioclip_taxonomy_masked_best.csv`, "best" is `SAMPLE_MODE = "best"`: the 50 images with the most masks after filtering. The file is a sample (842 crops), not the best model or prompt setup.
