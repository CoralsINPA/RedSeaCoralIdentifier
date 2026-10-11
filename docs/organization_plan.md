# Code organization plan

Status: proposal only. Nothing has been changed yet.

Sources read in full: `docs/Coral_Pipeline_Onboarding_Guide.pdf`, the three notebooks in `notebooks/`, and `CoralClassification_v2_Pipeline.ipynb`. The v2 notebook is on Drive only, and it produced both the 14,224-crop run and the tiered labels that the Millepora audit uses. I also checked the CoralSCOP submodule, one source photo and the file listings in the Drive folders.

Cell numbers are 0-based positions in the `.ipynb` files. Abbreviations:
**CM** = `CoralMasking.ipynb`, **WC** = `weak_classification.ipynb`, **MA** = `Millepora_Analysis.ipynb`, **V2** = `CoralClassification_v2_Pipeline.ipynb` (Drive).

## TL;DR

1. Move the logic into an installable package (`src/coralid/`) made of about 18 small modules, one concern each. Keep a few thin notebooks as the Colab front end and add a CLI for the multi-hour batch steps. A single `utils.py` would grow into a 1,500-line file that mixes GPU model loading, filtering and plotting.
2. Three conventions would remove most of the bugs found below:
   - **One coordinate frame:** always the mask JSON's frame.
   - **One coral key:** `image_name` plus `coral_id`, never `coral_id` alone.
   - **Filter once:** run the filters a single time, store the results with their reasons, and reuse them everywhere.
3. Migrate in three steps: extract the code with no behavior change, prove parity against existing outputs on Colab, then fix bugs one commit at a time and report the before/after numbers for each.
4. Some files on Drive are already wrong: `cleaned_images_crops/`, `cleaned_jsons_filtered/` and the bbox DINOv2 features. The zero-shot labels, the masked features and the crops the expert reviewed are fine (§2.1).
5. Notebooks stop being the source of truth, and their outputs get stripped before commit. The two large repo notebooks are 14 MB and 17 MB today, almost all of it embedded images.

## 1. What the code actually does today

```
data_all/  1,714 photos (sample checked: IMG_4033.JPG = 3648 x 2736)
 │
 │ CM 6-8   CoralSCOP automatic mask generator on the photo downscaled to max side 2560 (→ 2560 x 1920)
 │          points 32 · pred_iou .60 · stability .55 · crop_n_layers 1 · min region 100
 │          + CoralSCOP's built-in coral/background head drops "background" masks inside generate()
 ▼
results_all/iou60_sta55_pts32/json/<stem>.json        44,211 masks (RLE, bbox, iou, stability)
 ├─ CM 9     DINOv2-S/14, masked square crops, all 44,211        → coral_dino_encodings_MASKS.csv
 ├─ CM 10    DINOv2-S/14, bbox crops, all 44,211 (WRONG REGION)   → coral_dino_encodings.csv
 ├─ CM 11    DinoVdeau-L, masked; area + aspect + sharpness (no containment) → 18,630 rows
 ├─ CM 12-15, 19-22  PCA / t-SNE / UMAP / KMeans / HDBSCAN exploration
 ├─ CM 17    → cleaned_jsons_filtered/   (sharpness measured on the wrong region)
 ├─ CM 18    → cleaned_images_crops/     (unfiltered, and wrong region)
 ├─ WC 14    zero-shot, 20 hard-coral genera, 3 backbones x 5 prompt formats
 │           all six compared runs use the same 842 crops from the 50 densest images
 └─ V2 10-13 (Drive) zero-shot, 24 genera (20 hard + 4 soft), BioCLIP-FT, taxonomy prompts, all images:
             44,211 → 16,197 (area, aspect, containment) → 14,224 (sharpness, visibility)
             + 512-d embeddings (v2_embeddings.npz)
             → tiers: 3,058 auto-accept / 11,166 expert review → expert_review_for_assaf.csv
             Phase 2 (logistic regression on embeddings): written, never run
              └─ MA  790 predicted Millepora → 372 at confidence ≥ 0.85 → 129 with bbox sharpness ≥ 150
                     → audit 66 yes / 46 no / 17 unsure = 58.9% precision
                     → 96 sharp false-negative candidates exported to PDF
```

Things worth knowing that the docs don't say:

- **The headline numbers come from V2.** The run the guide cites (14,224 crops, mean confidence 0.485) is V2, with 24 genera including soft corals, not WC.
- **The `_best` file is a sample.** In `coral_weak_labels_finetuned-bioclip_taxonomy_masked_best.csv`, "best" means `SAMPLE_MODE = "best"`: the 50 images with the most masks after filtering. It is not the best configuration, although CLAUDE.md describes it as "best zero-shot labels".
- **No notebook scores against the 78-crop expert set.** `evaluate_against_expert` (WC 10) is commented out. The `build_expert_test_set.py` it refers to isn't in the repo, and a title search on Drive didn't find it.
- **CoralSCOP is not stock SAM.** Its vendored `segment_anything` adds a 2-class coral/background head (`cate_num=2`), which drops background masks inside `generate()`. It also returns a per-mask feature vector (`fc_features`) and a pairwise similarity matrix. CM 8 throws both away.
- **The checkpoint fallback can fail silently.** If pip's `segment_anything` gets imported instead of CoralSCOP's (CM 6 even pip-installs it when the import fails), strict loading fails, and CM 7's `strict=False` fallback then silently drops the coral head.

## 2. Problems found

### 2.1 Bugs that affect results (verified in the code)

| # | Problem | Where | Effect |
|---|---|---|---|
| 1 | **Coordinate-frame mismatch.** The mask JSONs are in the 2560×1920 frame, but the photos are larger (`IMG_4033.JPG` is 3648×2736). Several cells crop the original photo with bboxes from the smaller frame, without resizing it first. | CM 10, 14, 17, 18, 22 | They crop the wrong part of the image in four outputs: the bbox DINOv2 features, the cluster preview galleries, the sharpness filter behind `cleaned_jsons_filtered/`, and every file in `cleaned_images_crops/`. |
| 2 | **`cleaned_images_crops/` is not cleaned.** The harvester reads `json/`, not `cleaned_jsons_filtered/`. | CM 18 | The folder holds 44,211 unfiltered crops, on top of #1. |
| 3 | **`coral_id` is treated as globally unique,** but it restarts at 0 in every image. | V2 16-18, 20; WC 14 `RESUME` | V2 Phase 2 would join embeddings and expert labels to the wrong corals, so the classifier would train on garbage. `RESUME` skips corals in every image that share a done ID. Nothing is corrupted yet, because Phase 2 never ran. |
| 4 | **V2 Phase 2 methodology.** Blank expert cells count as "confirmed". The reported classifier accuracy is measured on its own training set. Tier-1 auto-accepts are used as training labels, and they include Millepora, which audited at 58.9% precision. | V2 16-20 | Inflated accuracy and noisy training labels. |
| 5 | **The model comparison mixes label sets.** The `hybrid` and `siglip2` runs came from an older 10-genus version. Today `genera_hybrid` has 10 keys against 20 genera, so that cell would raise a KeyError. | WC 14, 23 | The comparison isn't apples to apples, and all six runs score the same 842 crops from the densest images, which are not representative. |
| 6 | **There are two different "taxonomy" prompt sets.** WC files Favites, Echinopora, Cyphastrea, Goniastrea and Leptastrea under Faviidae. V2 puts them under Merulinidae or incertae sedis. | WC 14 vs V2 9 | "Taxonomy" results from WC and V2 aren't comparable. |
| 7 | **Every segmentation error is swallowed,** including CUDA out-of-memory. The cell writes an empty JSON, and smart-skip then treats that image as done. | CM 8 | A transient failure permanently becomes an image with 0 corals. You can count the affected images: these JSONs have a `note` field. |
| 8 | **There are three sharpness definitions.** The pipeline uses the masked crop on black, ≥ 80. MA uses the unmasked bbox crop, ≥ 150. Laplacian variance on a masked crop also picks up the artificial coral-to-black edge. | WC/V2/CM 11/17 vs MA 5, 9 | "Sharp" means different things in different places, and the 58.9% figure uses MA's definition. |
| 9 | **The image stats are misleading.** `coral_results.csv` counts raw, overlapping masks. Its coverage % adds up overlapping areas, so it can exceed 100%. Its size bins (1k / 10k px) have nothing to do with `MIN_AREA` 25k. | CM 8 | Misleading if anyone uses the stats. |
| 10 | **bbox strings are parsed unsafely.** 10 places use `eval()`, and CM 22 splits the string on whitespace. | WC, MA, V2, CM 22 | Fragile, and `eval()` executes arbitrary code from a CSV. |
| 11 | **Feature files of unknown origin.** CM 20 reads `coral_dinoV3_encodings.csv` (44,211 × 1024), which no notebook I read writes. CLAUDE.md lists `coral_dinoV2_encodings*.csv`, but CM writes `coral_dino_encodings*.csv`. | CM 20, Drive | I can't tell which features these files hold or which crops they used. |

**Not affected:** the V2 labels and embeddings, the WC runs, the masked DINOv2 and DinoVdeau features, the expert-review grids and the Millepora PDFs and audit. All of these resize to 2560×1920 before cropping. This holds as long as every photo is 4:3 landscape, because WC, V2 and MA hard-code `resize(2560, 1920)` instead of reading the size from the JSON. Phase 1 adds a one-off check over all 1,714 JSONs to confirm.

### 2.2 Structural problems

- **Copy-paste instead of shared code.** `decode_rle` is defined 6 times and `get_masked_crop` 6 times, plus 7 other crop variants (13 ways to cut out a coral). `filter_annotations` exists 3 times and `is_sharp_enough` 4 times. `(2560, 1920)` is hard-coded 13 times, `RUN_NAME` and `base_path` are redefined 6-7 times, and 8 places find a mask by rescanning JSON files.
- **No single source of truth.** Thresholds, genera, prompts and tiers live in notebook cells, and some defaults disagree: `is_sharp_enough(threshold=50)` against `SHARPNESS_THRESHOLD = 80`.
- **Four different "filtered" sets:** 44,211 masks, 18,630, `cleaned_jsons_filtered/`, and 14,224. Every step re-filters on the fly.
- **Slow by construction:**
  - The containment check decodes two full 2560×1920 masks for every overlapping pair. CM 17 took 2.5 h, and the counts cache in WC 12 and V2 7 exists only to avoid that cost.
  - CM 18 re-reads the photo from Drive for every single crop (2.2 h).
  - WC 20 can scan up to 1,714 JSON files for each crop it displays.
- **Outputs don't record how they were made.** Filenames encode some settings, but nothing records the label set, prompt version, sample definition or git commit.
- **Embeddings are stored as CSV text,** 180-490 MB per file for 44k rows. `.npz` or parquet would be about 5-10× smaller and load in seconds.
- **Notebook hygiene:**
  - Notebooks are committed with their outputs.
  - The Colab badges point at a stale branch and at the old root paths.
  - Scratch cells are mixed in with pipeline cells.
  - Cells depend on hidden state left by earlier cells, e.g. CM 14's `if 'final_df' in locals()`.
- **Environment:**
  - CM puts Drive's copy of CoralSCOP on `sys.path` instead of the repo submodule.
  - No versions are pinned. WC installs whatever Colab currently ships (torch 2.11, transformers 5.16, open_clip 3.3).

### 2.3 Documentation corrections (small, part of Phase 0)

- **Containment.** The guide says the smaller mask is removed, but the code removes the larger one. WC 7 says that's intended ("keeps the finer-grained segment").
- **Folder structure.** The guide's "Data and folder structure" section doesn't match Drive: it lists `Coral_Segmentation_Outputs/`, `reef_masks/`, `.npy` features, `genera_taxonomy.json` and `checkpoints/`. The Drive map in CLAUDE.md is the accurate one.
- **Prompt format.** The guide recommends `ensemble`, but V2 says taxonomy did best in v1 and uses taxonomy.
- **Where the headline numbers come from.** The 14,224-crop and 58.9% numbers come from V2: 24 genera including soft corals, with sharpness measured as the bbox Laplacian ≥ 150. The 58.9% judgments were made by a non-expert analyst (confirmed by the user on 2026-10-11).
- **CLAUDE.md fixes.** The `_best` file is the 50-image sample, and the V2 notebook exists only on Drive.

## 3. Target design

### 3.1 Decision: package + CLI + thin notebooks

| Option | Verdict |
|---|---|
| One `utils.py` imported by the notebooks | Better than today, but it grows into a grab bag. Every import pulls in torch, open_clip, umap and hdbscan, and the pieces are hard to test separately. Rejected. |
| Scripts only, no notebooks | Throws away what notebooks are good at here: Colab GPU next to Drive, looking at crops, and building expert-review grids. Rejected. |
| **Package (`src/coralid`) + CLI for batch steps + thin notebooks** | The logic is tested and code-reviewed as `.py` files. Notebooks shrink to short drivers. Long jobs run as `!python -m coralid ...` from Colab, are resumable, and survive cell re-runs. **Recommended.** |

### 3.2 Repository layout

```
RedSeaCoralIdentifier/
├── pyproject.toml          metadata, pinned deps, extras: [segment] [clip] [cluster] [dev]
├── configs/
│   ├── iou60_sta55_pts32.yaml   active run: SAM params, filters, crop mode, paths under DATA_ROOT
│   └── genera.yaml              label sets; per-genus taxonomy / descriptive / common prompts;
│                                hard vs soft; tier
├── src/coralid/            flat package (18 modules is small enough to keep flat)
│   ├── config.py   masks.py   images.py   crops.py   filters.py   manifest.py
│   ├── segmentation.py   encoders.py   genera.py   zero_shot.py   batch.py
│   ├── labeling.py   probe.py   evaluation.py   clustering.py   viz.py
│   └── colab.py   cli.py
├── notebooks/
│   ├── 01_segment_and_filter.ipynb
│   ├── 02_embeddings_explore.ipynb
│   ├── 03_zero_shot.ipynb
│   ├── 04_expert_review_and_audits.ipynb
│   ├── 05_supervised_probe.ipynb
│   └── legacy/             today's 3 notebooks + V2, frozen; deleted after parity
├── tests/                  CPU-only, synthetic masks and images; no data or GPU needed
├── third_party/CoralSCOP   submodule, unchanged
└── docs/
```

### 3.3 Modules

| Module | Responsibility | Replaces |
|---|---|---|
| `config.py` | Dataclasses `Paths`, `SamConfig`, `FilterConfig`, `CropConfig` and `ZeroShotConfig`. Loads and saves YAML, derives the run name, and takes `DATA_ROOT` from `paths.data_root` or the `CORALID_DATA_ROOT` env var. The Colab default path lives only in `colab.py`. | Path and parameter blocks in CM 2, 6, 7, 11, 18; WC 6, 8; V2 4; MA 2, 4-6, 9 |
| `masks.py` | RLE encode/decode and mask-JSON read/write. Defines `CoralKey(image_name, coral_id)` and its string `uid`. Parses bboxes without `eval`. `AnnotationStore` gives cached per-image lookup by key. | 6× `decode_rle`, the RLE block in CM 8, 10× `eval(bbox)`, 8× JSON rescans |
| `images.py` | `load_rgb` and `load_in_mask_frame(image_path, mask_json)`. This is the **only** code that resizes, and it uses the JSON's width and height. | 13× hard-coded `(2560, 1920)`, and the missing resizes in CM 10, 14, 17, 18, 22 |
| `crops.py` | `bbox_crop`, `masked_crop(background, padding)` and `square_pad` | 13 crop helpers |
| `filters.py` | Area, aspect ratio, containment computed on the RLE directly (pycocotools intersect and area, no full decode), and one documented sharpness measure. Returns a reason for every rejected mask. | 3× `filter_annotations`, 4× `is_sharp_enough`, `passes_aspect_ratio` |
| `manifest.py` | Builds and reads the per-run coral table (§3.4 #4) | The counts caches (WC 12, V2 7), `cleaned_jsons_filtered/`, on-the-fly filtering |
| `segmentation.py` | Loads CoralSCOP from the submodule, failing loudly with no `strict=False` fallback. Builds the generator and runs `segment_image`. Can optionally keep `cate_preds` and `fc_features`. | CM 6-8 |
| `encoders.py` | One interface for DINOv2, DinoVdeau, BioCLIP-FT, BioCLIP-2 and SigLIP-2: `encode_images`, plus `encode_texts` for the vision-language models. Heavy imports stay inside each class. | CM 9-11, WC 14 §1/§3, V2 9 |
| `genera.py` | Loads `genera.yaml`. Provides the label set (`hard20`; the 4 soft corals stay in the file only as optional reject prompts) and builds the prompts for each format, including the ensemble. Both current taxonomy variants are kept as named, versioned prompt sets. | Prompt dicts in WC 14 and V2 9, tier sets in V2 4 |
| `zero_shot.py` | Builds text embeddings per prompt format, scores crops into probabilities, and returns top-k, runner-up and margin as a prediction table | WC 14 §3, §8, §9; V2 10 |
| `batch.py` | Resumable per-image runner: a done-manifest, part files, a final merge, and a failure log instead of empty JSONs | V2 10's checkpointing (the best pattern in the codebase), CM 8 smart-skip |
| `labeling.py` | Tier rules, review-sheet export, ingesting completed reviews by key, and a per-genus audit sheet with summary that works for any genus | V2 12-13, 16; MA 7-8 |
| `probe.py` | Linear probe on stored embeddings, with cross-validation grouped by image and a held-out report | V2 17-18, 20 |
| `evaluation.py` | Scores predictions against the expert set. Normalizes the special labels (`unidentifiable`, `MISSING_LABEL`, and soft corals outside the target set). Reports per-genus precision/recall, confusion pairs, and the genus distribution compared with `ground_truth_genus_counts.csv`. | WC 10 (commented out), WC 23, V2 19-20, MA 2 |
| `clustering.py` | PCA, UMAP, t-SNE, KMeans and HDBSCAN, with silhouette, Davies-Bouldin and compactness scores | CM 12-15, 19-21 |
| `viz.py` | Crop grids, per-genus galleries, multi-page PDFs and mask overlays. Each takes a table plus an `AnnotationStore`. | CM 8 §4, 14, 16, 22; WC 16, 18, 20, 25; MA 3-4, 6, 9; V2 13-14, 22 |
| `colab.py` | Mounts Drive, sets `DATA_ROOT`, and installs the package from GitHub (no token needed while the repo is public). The only Colab-specific module. | The 6 `drive.mount` cells |
| `cli.py` | `python -m coralid {segment, build-manifest, embed, zero-shot, tier, export-review, train-probe, evaluate} --config ...` | "Run all cells" for multi-hour jobs |

### 3.4 Conventions (the rules that prevent the bugs in §2.1)

1. **One coordinate frame:** the mask JSON's `width × height`. Photos are only ever loaded through `images.load_in_mask_frame`. Fixes #1.
2. **One coral key:** `(image_name, coral_id)`, written as `IMG_4033.JPG#14`. Every table, embedding file and review sheet uses it. Fixes #3.
3. **Labels are tied to a segmentation run.** `coral_id`s change whenever SAM is re-run, so expert labels also store the run name and the bbox. That way they can be re-matched to new masks by IoU.
4. **Filter once and keep the reasons.** Each run gets one `manifest.parquet` with a row per mask: key, bbox, area, predicted_iou, stability, aspect, `contained_by`, sharpness, visible_px, `keep` and `reject_reason`. Every downstream step reads `keep == True` from it. This replaces the four filtered sets.
5. **Constants live in `configs/`, not in notebooks.** A notebook may override a field for an experiment, and the resolved config is saved next to the output.
6. **Every output has a `*.meta.json` sidecar** recording the config, label set, prompt-set version, sample definition, model id, git commit, row count and date.
7. **Formats:** embeddings go in `.npz` (keys plus matrix) and tables in parquet. CSV is kept only for files people edit by hand, such as review sheets.
8. **No `eval`, and no bare `except Exception: continue` around model calls.** Failures go to a log and are retried on the next resume.
9. **Plotting and computing stay separate:** plotting functions don't compute, and computing functions don't plot.

### 3.5 Run folder on Drive (additive; `json/` stays where it is)

```
results_all/iou60_sta55_pts32/
├── json/                                 unchanged (44,211 masks)
├── manifest.parquet  (+ .meta.json)
├── features/<encoder>__<crop_mode>.npz
├── predictions/<model>__<prompts>__<labelset>__<subset>.parquet  (+ .meta.json)
├── reviews/{exports,completed}/
└── reports/                              eval reports and figures
```

Existing outputs (`v2_pipeline/`, `coral_weak_labels_*.csv`) stay where they are. `evaluation.py` gets a small adapter for them so old runs can be scored with the new harness.

### 3.6 Notebooks after migration

Each notebook follows the same shape: a setup cell (`coralid.colab.setup()`), a config cell, a few library calls, then plots. The target is fewer than 15 cells per notebook and no function definitions.

- `01_segment_and_filter`: run or resume segmentation, build the manifest, and check quality (overlay samples, counts per reject reason).
- `02_embeddings_explore`: extract features and explore clusters (today's CM 12-22).
- `03_zero_shot`: run one or more configs on a named subset, compare them, and evaluate.
- `04_expert_review_and_audits`: export review sheets and PDFs, ingest the results, and run per-genus audits (MA generalized to any genus).
- `05_supervised_probe`: the linear-probe idea from V2 Phase 2, on the 20 hard genera, rewritten on keys, with honest held-out evaluation.

### 3.7 Tooling

- `pyproject.toml` with pinned versions, plus extras so CPU-only CI doesn't install SAM or CLIP.
- `ruff` for lint and formatting, `pytest`, and `pre-commit` with `nbstripout`, so outputs are never committed. Figures go to `reports/` on Drive.
- A GitHub Actions workflow running ruff and pytest on CPU.
- The tests that matter most:
  - RLE round trip.
  - RLE-native containment gives the same answer as the decode-based version.
  - A crop lands on the same pixels after a frame resize (regression test for #1).
  - Keys are unique across images (#3).
  - Expert-label normalization.
  - Zero-shot scoring with a fake encoder.
  - Resume skips exactly the images that are done.

## 4. Migration plan

The principle: extract first, prove parity, then fix. If we fix bugs while extracting, we can't tell refactoring mistakes from intended changes.

| Phase | Work | Done when | Size |
|---|---|---|---|
| **0. Hygiene** | Add `pyproject`, ruff, pre-commit with nbstripout, and `.gitignore`. Pull V2 from Drive into `notebooks/legacy/` and move today's notebooks there unchanged. Make the doc corrections from §2.3. | `pip install -e .[dev]` works and CI passes | S |
| **1. Core library, same behavior** | Write `config`, `masks`, `images`, `crops`, `filters`, `manifest`, `genera`, `zero_shot`, `evaluation` and `batch`, with unit tests. Reproduce today's behavior exactly, quirks included. Add a one-off audit listing JSONs that aren't 2560×1920 and empty JSONs that have a `note` field. | Unit tests pass on CPU | M |
| **2. Parity on Colab** | Run the new code on the Drive data. (a) Filter counts must come out as 44,211 → 16,197 → 14,224. (b) Zero-shot on the 50-image sample must match `..._masked_best.csv`, 842 rows. That file was rewritten on 2026-10-11, so confirm the row count first. | Same keys, ≥ 99.5% identical top-1, and probabilities within 1e-3 | S (2 cells for you to run on a Colab GPU) |
| **3. Fixes, one per commit** | Fix the frame bug in feature and crop code, and delete `cleaned_images_crops/`, `cleaned_jsons_filtered/` and the bbox DINOv2 CSV (approved). Switch to key-based joins. Fix segmentation error handling. Settle on one sharpness definition. Move containment onto RLE. Wire up the expert evaluation. Then run the 20-genus zero-shot on all images, saving embeddings, as the new baseline. Re-run the Millepora analysis on that baseline, reusing the 129 existing yes/no judgments, so only crops not already judged need review. | Each commit states which numbers changed and by how much | M |
| **4. Rest of the package, notebooks, CLI** | Write `segmentation`, the DINO part of `encoders`, `labeling`, `probe`, `clustering`, `viz`, `colab` and `cli`, plus the five notebooks. Delete `legacy/` after one full end-to-end run. | A new person can run notebooks 01→04 from the README without editing code | M |

Phase 1 goes in the order that unblocks the algorithm work: `masks → images → crops → filters → manifest → genera → zero_shot → evaluation`. Segmentation and DINO/clustering come last because they're stable and slow to re-run.

Who does what: all code, tests and refactoring can happen in this repo on CPU. Anything that touches the photos, the masks or the GPU models has to run in Colab against Drive. That covers the Phase 2 parity runs and regenerating features, and I'll prepare those as ready-to-run cells.

## 5. Decisions

Decided on 2026-10-11:

1. **Label set: 20 hard genera.** V2 is not carried forward as a pipeline. The package absorbs the parts of V2 that are worth keeping: saving embeddings, the runner-up column, checkpointing, and the review export. Soft corals still show up in the photos, so how to reject them is an algorithm question (§6).
2. **Drive cleanup approved.** Delete `cleaned_images_crops/`, `cleaned_jsons_filtered/` and the bbox DINOv2 CSV in Phase 3; all three are defective. Regenerate them only if something needs them.
3. **Unknown provenance doesn't block anything.**
   - `coral_dinoV3_encodings.csv` only feeds the abandoned HDBSCAN exploration (CM 20-22), so it isn't carried forward.
   - `build_expert_test_set.py` would only matter for rebuilding the expert set. Its output, `assaf_expert_test_set.csv`, exists.
4. **The repo is public for now.** Colab installs the package with plain `pip install git+https://...`, with no token. Nothing private goes into the repo.
5. **Runtime: Colab for now, though not ideal.** So nothing may depend on Colab beyond `colab.py`. Paths come from `DATA_ROOT` and long jobs run through the CLI, which lets the code move to a GPU machine later without changes.
6. **Package name: `coralid`.**

## 6. Parked for the algorithm discussion

These came up while reading but are algorithm questions, not organization ones:

- **Free CoralSCOP outputs.** CoralSCOP already produces a coral/background decision and a per-mask feature vector, and the pipeline discards both.
- **Confidence isn't comparable across models.** Confidence is `softmax(100 · cosine)` for every backbone, and SigLIP's own logit scale and bias are ignored. So a single threshold such as 0.85 means something different for each model.
- **Colonies can disappear.** Containment drops the larger mask before the sharpness filter runs. If the remaining fragment is then rejected as blurry, the colony is lost entirely.
- **Favia vs Dipsastraea.** After the recent revisions, most Indo-Pacific *Favia* now sit in *Dipsastraea*, so keeping both as separate classes may be asking the model to tell synonyms apart. Worth checking with Assaf.
- **Thin evidence.** The model comparisons and the Tier-1 genus list rest on 842 non-random crops and about 80 expert labels.
- **V2 Phase 2 needs fixing before it runs.** It's the natural next step, but it needs keyed joins, held-out evaluation, and probably excluding auto-accepts for genera with known low precision.
- **Soft corals still need somewhere to go.** With 20 hard genera as the output, soft corals in the photos have no class: 5 of the 62 identifiable crops in Assaf's review, and 11.7% of V2's predictions. A reject outcome is needed, such as an "other" threshold or soft-coral prompts used only as distractors. Otherwise these colonies get forced into a hard genus, which inflates false positives (Millepora's top runner-ups in V2 included Sinularia).
- **Millepora precision is unknown for the 20-genus setup.** The 58.9% was measured on V2's 24-genus predictions. Without the soft-coral classes, some of those crops will move to Millepora.
