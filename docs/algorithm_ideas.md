# Algorithm improvement ideas

Collected on 2026-10-11, during the code organization, for the algorithm-planning step that follows it.
- **Nothing here is decided or started.** Each idea gives its evidence (with how solid it is), how to try it, what it costs, and what it needs: expert time, analyst time, GPU, or code only.
- **Baseline:** the 20-genus zero-shot run over all images, with correct crops and saved embeddings, from refactor Stage B. It ran on 2026-10-12 (`docs/agent_memory/sessions/2026-10-12_baseline_run.md`). Ideas are compared against it.
- **Background:** `docs/agent_memory/README.md` ("Current understanding"), `docs/agent_memory/assaf_review_confusion_matrix.md`, `docs/agent_memory/sessions/2026-10-11_parity_run_1.md`, `docs/agent_memory/sessions/2026-10-12_baseline_run.md`, and plan §6 of `docs/organization_plan.md`.

## Where we stand

- **Pipeline:**
  - CoralSCOP masks (run `iou60_sta55_pts32`): 44,211 masks.
  - Filters (area, aspect, containment, blur): **16,080** crops kept.
  - BioCLIP-FT zero-shot over the **20 hard genera** with `taxonomy_v1` prompts; confidence = softmax(100 · cosine).
- **Baseline files:** `<run>/predictions/finetuned-bioclip__taxonomy_v1__hard20__all.parquet` and `<run>/features/finetuned-bioclip__masked.npz` (16,080 × 512).
  - Mean confidence 0.494; 25.8% of crops at 0.85 or more.
- **Expert test set** (`assaf_expert_test_set.csv`): 77 rows, 67 on landscape photos and 10 on portrait photos.
  - The v1 run got **36/54 = 66.7%** on crops whose genus is in the label set (95% interval 53-78%).
  - **The baseline gets 34/51 = 66.7% on the landscape rows**, identical to v1 (no prediction changed).
  - The 10 portrait rows can't score it: Assaf labeled the legacy crops, which were cut from the wrong place.
  - Error structure (from the review's matrix):
    - 13 of 23 errors are the massive / merulinid complex (Dipsastraea alone 6 of 7 wrong);
    - 5 are soft corals with no class;
    - 5 are other.
- **Strengths:** the model is reliable on distinctive genera (Acropora, Platygyra, Porites, Millepora, Montipora, Stylophora, Hydnophora, Echinopora, Pocillopora). They are 67% of colonies in the survey counts (2,463 of 3,659).
- **Biases** (baseline genus mix against the survey, the 20 genera only):
  - **over-predicted:** Pavona 15.1% vs 0.44% (34.5×), Leptastrea 5.5×, Montipora 3.7×, Porites 1.8×, Favia 1.6×;
  - **under-predicted:** Dipsastraea 0.14×, Galaxea 0.13×, Cyphastrea 0.28×, Platygyra 0.33×;
  - soft corals get forced into hard genera (Acropora precision 5/9 on the landscape expert rows, mostly because of them).
- **Known data defects, now fixed in code:**
  - the legacy code mis-cropped the 221 portrait photos (13%);
  - some expert labels were therefore given on mis-cut crops.

## Suggested priority

| # | Idea | Why first | Needs |
|---|---|---|---|
| 1 | A bigger, better evaluation set | Today's set can't show a gain under about 10 points | Assaf's time |
| 2 | A real crop-quality gate (blur) | Unidentifiable crops looked blurred and passed (7 of the 16 were mis-cuts; check the other 9) | Code + existing labels |
| 3 | A reject / "other" outcome | Soft corals and junk inflate false positives | Code |
| 4 | Linear probe on the embeddings | The standard big jump over zero-shot once labels exist | Labels from #1 |
| 5 | Calibration and prior correction | Fixes Pavona over-prediction; makes thresholds mean something | Labels from #1 |
| 6 | Taxonomy decisions with Assaf | Favia vs Dipsastraea, grouping the massive complex | 30 min with Assaf |
| 7 | Resolution: patches for massive corals | Corallite detail is lost at 224 px | GPU, code |
| 8 | Prompt and backbone comparison, done properly | Earlier comparisons used 842 non-representative crops | GPU |
| 9 | Segmentation: audit, then cheap knobs, then sweeps | Not the visible error source yet; audit first | Analyst time, then GPU |

## 1. Measurement first

**1.1 A larger, stratified expert set.**
- **Size:** about 300 crops from the new baseline.
  - About 12-15 per *predicted* genus, random within each genus rather than the most confident. This measures precision.
  - Plus a purely random sample. This measures recall and the real genus mix.
- **Review conditions:** blind (no prediction shown), with correct crops.
- **Record a reason when a crop is unidentifiable:** blurry, partial colony, several colonies, not coral, other. Those reasons feed ideas 2 and 9 directly.
- Already on the project's horizon ("~300 crops, 12-15 per genus" in `docs/agent_memory/project_context.md`).

**1.2 Re-review the expert crops from portrait photos.**
- Assaf saw mis-cut crops for these: 10 rows, listed in `<run>/reports/portrait_impact_2026-10-12/expert_portrait_rows.csv`. Re-show them, correctly cut.
- **7 of the 10 are "unidentifiable"**, against 9 of 67 on landscape photos (one-sided Fisher p = 0.0004). Two of them got the same legacy probability vector, most likely from an empty crop. The bad cut, not blur, explains these 7.

**1.3 Calibrate the non-expert analyst against Assaf.**
- About 40 crops labeled by both people.
- If they agree at roughly 90% or better on a genus or task, the analyst can do that labeling and save Assaf's time.
- The blind Millepora review (idea 10) is the first instance.

**1.4 Standard report for every experiment.**
- Accuracy and per-genus precision and recall, with intervals.
- Paired comparison on the same crops (McNemar), which is more sensitive than separate intervals.
- Broken down by photo orientation, crop size and confidence.
- The genus mix against `ground_truth_genus_counts.csv`.

**1.5 Settle the end goal with the ecologists.**
- **Option A, a label per colony:** per-crop accuracy matters.
- **Option B, genus proportions or counts per site:**
  - per-crop errors can be corrected in aggregate (adjusted classify-and-count with the confusion matrix);
  - segmentation (merged or split colonies) and the size filter matter much more.
- The answer changes the priorities of ideas 7 and 9.

## 2. Crop-quality gate (blur and size)

**Evidence.**
- With correct crops, the blur filter (Laplacian variance ≥ 80 on the masked crop) rejects only **117 of 16,197** candidates (0.7%). The threshold sits below the 1st percentile of kept crops (108; median 558).
- The mask's edge against the black background reads as sharpness, so blurry corals pass.
- The user reports that **most crops Assaf couldn't identify were simply blurred.**
- **But 7 of the 16 unidentifiable expert crops were mis-cut portrait crops** (see 1.2), and their correct cuts aren't unusually blurry by the current measure (10th-87th percentile).
  - That leaves **9 landscape crops** as the evidence.
  - First step, minutes: join them to the manifest's `sharpness` and look at them next to the 51 identified landscape crops.
  - If they are blurred, build the gate. If not, find out what else makes a crop unidentifiable (partial colony, several colonies, distance) using the reasons recorded in 1.1.

**Ideas.**
- **Measure focus only on the coral:** inside an eroded mask, so the outline doesn't count; or on the unmasked crop.
- **Measure at the model's resolution.** Every crop is resized to 224×224:
  - large crops are shrunk (up to several times), which hides mild blur;
  - crops near the minimum size are enlarged, which exposes it.
  
  So measure blur after resizing, or make the threshold depend on crop size.
- **Compare focus measures:** Laplacian variance, Tenengrad (gradient energy), high-frequency energy (FFT or wavelet).
- **Photo-level blur.** Some whole photos are out of focus, and a per-photo score could gate all their crops.
- **Choose the measure and threshold from data:**
  - Assaf's unidentifiable crops are the positive class and his identified crops the negative class;
  - exclude the mis-cut portrait crops;
  - pick the threshold on the ROC curve;
  - then check the cost (crops dropped out of 16,080) and the benefit (accuracy on what remains).
- **A learned quality gate.** A small classifier on the BioCLIP embeddings (and/or focus features) predicting "identifiable vs not", trained on the expert's unidentifiable flags. It could also catch other junk: partial colonies, sand, shadows.

**Size filter interplay.**
- `MIN_AREA` 25,000 px (in the 2560 frame, about 0.5% of the photo, roughly 158×158 px) removes **24,754 of 44,211 masks (56%)**. It is the biggest filter.
- It excludes most small colonies. That's fine for per-colony labels, but a bias if counts per genus matter (1.5).
- Tune size and blur together. A small, sharp colony may be more identifiable than a large, blurry one.
- Minor: a pure-black photo pixel inside a mask counts as "not visible" in `visible_px`.

## 3. A reject / "other" outcome

**Evidence.**
- 5 of 59 identifiable expert crops are soft corals (Cladiella 3, Litophyton 1, Sinularia 1).
- With only 20 classes they land on Acropora, Porites or Millepora. Acropora precision would rise from 56% to about 83% if they were rejected.

**Ideas.**
- **Confidence or margin threshold:** below it, output "other / uncertain". Needs calibration first (idea 5).
- **Soft-coral prompts as distractors (prompts to write and compare; no such prompt set exists yet).** Score against the 20 hard genera plus prompts for the soft corals seen in the expert set (Cladiella, Sinularia, Litophyton). If a soft coral wins, output "other". The output stays 20 hard genera.
- **"Other" as a class in the linear probe** (idea 4), trained on soft-coral and non-coral crops.
- **Open-set check:** distance to the nearest genus prototype in embedding space.

## 4. Supervised linear probe on the embeddings

**What:** a logistic regression (or similar) on the saved 512-d BioCLIP-FT embeddings. `coralid.probe` implements it (notebook `04_supervised_probe`).

**How (Stage C's `probe.py` provides it):**
- labels joined by `(image_name, coral_id)`;
- cross-validation grouped by image, because crops from one photo are correlated;
- held-out evaluation only;
- class weights;
- no auto-accepted zero-shot labels for genera whose precision is unknown or low (Millepora).

**Expected:** usually a large gain over zero-shot once there are a few hundred labeled crops. Needs idea 1's labels.

**Variants:**
- **Few-shot prototypes:** class means of labeled embeddings, which work with very few labels.
- **Combine with zero-shot:** use zero-shot text embeddings as a prior for classes with few labels.
- **Active learning:** send Assaf the crops with the smallest margin, or with probe/zero-shot disagreement, next.

## 5. Calibration and prior correction

**Evidence.**
- Confidence is softmax(100 · cosine) for every backbone, so a threshold such as 0.85 is arbitrary and not comparable across models.
- Several wrong predictions had confidence above 0.9.
- Pavona is 15.1% of the baseline's crops (2,425) vs 0.44% of the survey's colonies. The expert set has a single Pavona row, so its precision is unknown.
- At the other end, Dipsastraea is 0.14× and Platygyra 0.33× its survey share. Platygyra is right 5 of 5 times on the expert set, so its shortfall is missed colonies.

**Ideas.**
- **Temperature scaling** (one parameter) fitted on expert labels, which makes confidence meaningful.
- **Prior / bias correction:** subtract a per-genus bias from the logits so the predicted mix matches a prior (survey counts or the labeled sample). This directly targets Pavona over-prediction.
- **Per-genus thresholds** for auto-accepting labels, chosen from measured precision instead of one global 0.85.
- **Look at the Pavona predictions:** what the model calls Pavona (probably flat or encrusting textures), to understand the bias before correcting it.

## 6. Taxonomy decisions (with Assaf)

- **Favia vs Dipsastraea.** Most Indo-Pacific *Favia* were reclassified to *Dipsastraea* (2012-2016 revisions), yet the label set and the survey keep both. The model may be asked to separate what are now synonyms, and Dipsastraea is the worst genus (1 of 7 correct, errors spread across Favia, Leptastrea, Favites, Acanthastrea).
  - **Ask Assaf:** merge them? Which name for Red Sea specimens?
- **Group the massive / merulinid complex.** Dipsastraea, Favites, Goniastrea, Favia, Leptastrea, Cyphastrea and Acanthastrea/Lobophyllia cause 13 of 23 errors.
  - Option: hierarchical output. A group label when the genus is uncertain, the genus when confident. Ecologists may accept the group level.
- **Taxonomy prompt version (a prompt set to write and compare; it does not exist yet).** `taxonomy_v1` files the genera under older families (e.g. Faviidae). A version that follows the current classification (Merulinidae, incertae sedis) would be a new, versioned prompt set in `configs/genera.yaml`. The best choice depends on BioCLIP's training text and must be measured (idea 8).

## 7. Resolution and crop design

**Evidence.**
- Massive corals are told apart by corallite structure (walls, size, spacing).
- A whole colony crop is shrunk to 224 px, which erases exactly that detail.

**Ideas.**
- **Patch / tile classification:** classify several patches inside the mask at native resolution and aggregate (mean of probabilities, or of embeddings). Likely most useful for the massive complex.
- **Multi-scale input:** whole-colony crop plus zoomed texture patches.
- **Background treatment:** compare masked-on-black (today), the bbox crop with context, and a blurred or grey background. The black outline may also bias the model.
- **Test-time augmentation:** flips and small rotations, averaged.

## 8. Prompt and backbone comparison, done properly

- **Earlier comparisons can't be trusted:**
  - all used the same 842 crops from the 50 densest images (not representative);
  - two of the six used an older 10-genus label set;
  - none was scored against the expert set.
- **Redo the comparison on the new baseline's crops, scored with the standard report (1.4):**
  - prompt formats: `taxonomy_v1` (and any new taxonomy set from idea 6), descriptive, common, ensemble;
  - backbones: BioCLIP-FT, BioCLIP-2, SigLIP-2 (with its own logit scale and bias, not ×100 softmax).
- Every run records its configuration in a `*.meta.json` sidecar, so results stay comparable.

## 9. Segmentation

**Why not first.** The visible errors are mostly classification errors, and there is no segmentation ground truth. A SAM rerun also costs hours of GPU and changes `coral_id`s, which breaks the link to existing labels until they are re-matched by mask IoU.

**Order:**
1. **Mask-quality audit** (analyst, 1-2 h). About 100 random kept masks, each tagged: one whole colony / fragment / several colonies merged / not coral. If roughly 85% are whole colonies, segmentation isn't the bottleneck. Otherwise this becomes the yardstick for comparing settings.
2. **Cheap knobs on the existing masks** (minutes each, no SAM rerun):
   - **Containment rule:** today it keeps the *smaller* mask, so a colony can lose to its own fragment, and the fragment may then be rejected as blurry, losing the colony entirely. Try keeping the larger mask, or IoU-based de-duplication.
   - **`MIN_AREA`:** see idea 2.
   - **CoralSCOP's own coral/background score** (`cate_preds`) and its per-mask feature vector (`fc_features`). Both are computed and then discarded. Use the score as a filter and the features as extra classifier input. Stage C's segmentation module can keep them behind a config flag.
3. **SAM sweeps, only if the audit shows real problems.** On `data_50`, scored with the audit tags:
   - `points_per_side` 48-64 for small colonies;
   - `pred_iou_thresh` / `stability_score_thresh` if merges or fragments dominate.
   
   Re-match existing labels by mask IoU before adopting a new run.

Minor: SAM's bbox is one pixel short (`w = right - left` on inclusive indices); the 5% padding hides it.

## 10. Millepora

- **There is no valid Millepora precision figure.** An earlier check was made on the predictions of a pipeline that is no longer used, by a non-expert who saw the model's answer; it is discarded and no number from it is evidence.
- **The blind review redoes it on the 20-genus baseline** (notebook `03_expert_review`, on hold, `coralid.genus_audit`):
  - drawn only from `<run>/predictions/finetuned-bioclip__taxonomy_v1__hard20__all.parquet`; the export refuses unless its meta file says label set `hard20` and the config's model and prompt set, and the source is recorded in the answer key (scoring stops if the baseline was made again);
  - blind and mixed with non-Millepora crops;
  - sampled across confidence levels;
  - `actual_genus` recorded for every "no";
  - about 40 crops double-labeled by Assaf, to measure how well the analyst agrees with him.
- **Likely confusers to watch:** Echinopora, Seriatopora, Pavona, and the soft corals of the expert set (Cladiella, Sinularia, Litophyton). With the 20-genus label set and no reject option, soft corals may land on Millepora.

## 11. Exploring embeddings

Embeddings and clustering (DINO features, K-Means, HDBSCAN, maps) are a separate direction that the pipeline never depends on: see [`exploration_embeddings.md`](exploration_embeddings.md).
