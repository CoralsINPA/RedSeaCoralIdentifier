# Agent memory

This folder holds project memory and conclusions that should outlive a single agent session: who is involved, what has been tried, what the expert reviews showed, and what each session concluded. It doesn't replace the code docs. `CLAUDE.md` covers the repo and Drive layout, and `docs/organization_plan.md` covers the refactor.

## Read order

1. **Current understanding** (below): the short version, updated every session.
2. `sessions/`, newest first: what each session did, verified and concluded.
   - `sessions/2026-10-13_detach_exploration_remove_v2.md`: V2 and the old Millepora audit excluded; embeddings and clustering detached into `coralid.explore`; the notebooks renumbered 01-04 plus `notebooks/explore/embeddings.ipynb`; the blind review now draws only from the 20-genus baseline and records its source; the provenance table of what `coralid` took from V2; the end-to-end run to ask the user for.
   - `sessions/2026-10-12_stage_C2.md`: Stage C2 (the five thin notebooks, the segmentation check of the real CoralSCOP, the README quickstart, the stale-clone fix of the bootstrap cell); dry-run on a synthetic Drive only; what each dry run covered and could not; the end-to-end run to do before `notebooks/legacy/` is deleted.
   - `sessions/2026-10-12_stage_C1.md`: Stage C1 (segmentation, DINO embeddings, probe, clustering, plots, the Colab setup and the full CLI); library code with synthetic tests, nothing has run on real data or a GPU yet; the Stage C2 plan.
   - `sessions/2026-10-12_baseline_run.md`: the 20-genus baseline on real data (expert set by orientation, genus mix against the survey, the portrait rows behind 7 of the 16 unidentifiable calls, the old Millepora audit against the baseline, which is discarded since 2026-10-13).
   - `sessions/2026-10-11_stage_B.md`: Stage B (orientation-aware parity, the 20-genus baseline job, portrait-impact report, review sheets and the Millepora re-analysis tools, early notebooks 03 and 04); nothing has run on real data yet.
   - `sessions/2026-10-11_stage_A2.md`: Stage A2 (zero-shot path, evaluation, resumable runner, the Colab parity notebook), what was verified, the real-data check that was skipped, and what to run next.
   - `sessions/2026-10-11_stage_A1.md`: Stage A1 of the refactor (packaging, tooling, core data modules), what was verified, and the bugs found but not fixed.
   - `sessions/2026-10-11_parity_run_1.md`: first Colab parity run (portrait photos, expert CSV vs matrix).
   - `sessions/2026-10-11_code_review_and_org_plan.md`: full code read, organization plan, user decisions.
3. The snapshots, for background:

| File | What it is | Snapshot date |
|---|---|---|
| `project_context.md` | Goal, people, current state, plans | 2026-09-29 |
| `pipeline_history.md` | The three pipeline iterations and what was learned | 2026-09-07 |
| `assaf_review_confusion_matrix.md` | Results of Assaf's 78-crop review of the v1 pipeline | 2026-09-16 |

The snapshots were imported from the claude.ai project memory on 2026-10-11.

## Current understanding

Last updated 2026-10-13 (V2 excluded, exploration detached).

**Pipeline**
- **Decided 2026-10-11:** the final label set is the **20 hard genera**. The main line is the WC path: `weak_classification.ipynb`, BioCLIP-FT zero-shot over CoralSCOP masks from run `iou60_sta55_pts32` (44,211 masks, about 14,224 crops after filtering).
- **V2 is excluded (decided 2026-10-11, enforced 2026-10-13).** The 24-genus V2 pipeline, its outputs (Drive `v2_pipeline/`) and the Millepora audit made on its predictions are no longer a reference. Why: V2 scored 24 genera and was never scored against the expert set, and the audit was made on its predictions, so using either would mix pipelines. The code and the current docs refer only to `coralid` and, for provenance, to `weak_classification.ipynb` and `CoralMasking.ipynb` (`tests/test_no_v2.py` enforces it). The generic engineering that V2 also had (resumable runs, saving embeddings with the predictions, runner-up and margin columns, review sheets) is implemented independently in `coralid`; the table is in the 2026-10-13 session file.
- **Soft corals** still appear in the photos (about 8-12% of crops). The 20-genus classifier needs a reject or "other" outcome, and that is the first algorithm question.
- **The pipeline** is segment -> manifest (filters) -> zero-shot baseline -> evaluation and expert review -> linear probe (on the BioCLIP embeddings the baseline saves). Notebooks `01_segment_and_filter`, `02_zero_shot`, `03_expert_review`, `04_supervised_probe`. The probe (`coralid.probe`) joins labels to features on `(image_name, coral_id)`, cross-validates grouped by photo, reports held-out numbers only, and waits for labels: it never makes its own.
- **Embeddings and clustering are a separate exploration direction**, not part of the pipeline: the `coralid.explore` subpackage, `notebooks/explore/embeddings.ipynb`, the `explore-embed` command (`docs/exploration_embeddings.md`). No pipeline module imports it (`tests/test_structure.py`).

**Accuracy evidence**
- **The new baseline (2026-10-12)** is the reference from now on: 16,080 crops on 1,709 photos, `<run>/predictions/finetuned-bioclip__taxonomy_v1__hard20__all.parquet` and `<run>/features/finetuned-bioclip__masked.npz`.
  - **Expert set, landscape rows:** 34/51 = 66.7% (95% 53-78%), identical to legacy, with no prediction changed.
  - **Portrait rows (10):** Assaf labeled mis-cut crops, so they can't score the baseline. 7 of them are among the 16 unidentifiable calls, so those 7 were bad cuts, not blur.
  - **Genus mix against the survey:** Pavona 15.1% vs 0.44% (34.5×), Leptastrea 5.5×, Montipora 3.7×; Dipsastraea 0.14×, Platygyra 0.33×.
  - Details in `sessions/2026-10-12_baseline_run.md`.
- **v1 on Assaf's review set:** 39/57 = 68.4% on crops whose true genus is in v1's 20 classes (95% interval 55-79%). That drops to 39/62 = 62.9% if the 5 soft corals are counted as errors, and the original 80-crop count was 38/64 = 59.4%. The authoritative matrix is in `assaf_review_confusion_matrix.md`.
- **Where the errors come from:**
  - 13 of 23 are the massive / merulinid complex; Dipsastraea alone has 6 errors from 7 crops.
  - 5 are soft corals. They pull Acropora precision down to 56%.
  - 5 are other.
- **The expert set is too small to show modest improvements.** A larger stratified review of the new baseline is needed.
- **Millepora: there is no valid precision figure.** An earlier check on the excluded V2 pipeline's predictions (a non-expert analyst, the model's answer visible, only the most confident and sharpest slice) is discarded. It is redone only through the blind review sheet built from the 20-genus baseline (notebook `03_expert_review`, on hold behind `RUN_EXPORT = False`): blind, mixed with non-Millepora crops, sampled across confidence levels, `actual_genus` recorded, Assaf labeling a subset to measure how well the analyst agrees with him. The export draws only from `<run>/predictions/finetuned-bioclip__taxonomy_v1__hard20__all.parquet`, refuses unless its meta file says label set `hard20` and the config's model and prompt set, records the source in the answer key, and scoring stops if the baseline was made again.

**Known failure modes**
- The massive-coral complex: Dipsastraea, Favia, Favites, Goniastrea.
- Acanthastrea vs Lobophyllia.
- Soft corals (Cladiella 3, Litophyton 1, Sinularia 1 in the expert set): no class in the 20-genus label set (algorithm idea 3).
- Pavona over-prediction: 15.1% of the new baseline's crops (13.7% of v1 predictions), against 0.44% in the survey counts.
- **Taxonomy drift.** Most Indo-Pacific *Favia* were moved to *Dipsastraea*, yet `ground_truth_genus_counts.csv` lists both, so the survey itself mixes old and new names.

**Code**
- **State (2026-10-13):** Stages A1, A2, B, C1 and C2 of the refactor are written, and on 2026-10-13 V2 was removed and exploration detached. The `coralid` package (`src/coralid/`) has the **pipeline** modules:
  - the data path: `config`, `masks`, `images`, `crops`, `filters` (with a `sharpness_crop` option: `masked` or `bbox`), `manifest`, `orientation` (landscape / portrait, and a way to split any metric by it), `funnel`;
  - segmentation: `segmentation` creates only the missing mask JSONs and never rewrites one; errors go to a failure log, never to an empty JSON; CoralSCOP is loaded from the submodule and checked (`parity_segmentation`, `mask_compare`);
  - the zero-shot path: `genera` (with `configs/genera.yaml`), `encoders` (CLIP-family models only), `zero_shot`, `batch` (resumable runner), `baseline` (the 20-genus job);
  - scoring and reports: `evaluation`, `predictions`, `reports` (also the probe report), `portrait_impact`, and the parity checks (`parity`, `parity_manifest`, `parity_expert`, `parity_report`);
  - review: `labeling`, `genus_audit` (the blind review sheet), `review_source` (the baseline source check), `review_scoring`, `review_export`;
  - supervised learning: `probe`, `probe_labels`; plots: `viz`, `viz_plots`; `colab`, `jobs` and `cli` (`python -m coralid {segment, build-manifest, zero-shot, evaluate, export-review, train-probe, explore-embed}`).
  The **exploration** subpackage `coralid.explore` (`dino`, `embed`, `clustering`, `plots`, `analysis`) is outside the pipeline.
  **The notebooks:** pipeline `01_segment_and_filter`, `02_zero_shot`, `03_expert_review`, `04_supervised_probe`; exploration `notebooks/explore/embeddings.ipynb`. All are thin, start with the bootstrap cell and are written. It has 581 CPU tests (six need torch, umap or hdbscan), CI, ruff and pre-commit. The legacy notebooks `CoralMasking` and `weak_classification` (and `Millepora_Analysis`, kept as history only) are frozen in `notebooks/legacy/`. It reproduces the legacy behavior on landscape photos (real data, see below). **Nothing from C1, C2 or the 2026-10-13 restructuring has run on real data yet** (the notebooks were dry-run on a synthetic Drive with fake generators and encoders).
- **What has run on real data:**
  - the parity notebook (`sessions/2026-10-11_parity_run_1.md`);
  - the baseline notebook (then `03_zero_shot`, now `02_zero_shot`) in full and the first cells of the expert-review notebook (then `04`, now `03_expert_review`: the portrait-impact report) (`sessions/2026-10-12_baseline_run.md`). The 16 / 2 split of `unidentifiable` / `MISSING_LABEL` is confirmed.
  
  The blind review export and its scoring (in `03_expert_review`) haven't run on real data.
- **Found, not fixed (C1, see `sessions/2026-10-12_stage_C1.md`):** CoralSCOP's loader has no `map_location`; it is unknown whether the generator can raise for a photo with no masks. (The submodule's `__pycache__` no longer dirties `git status`: `ignore = untracked` in `.gitmodules`.) **C2:** the segmentation check's PASS / WARN / FAIL limits are guesses, and `data_50` is assumed to hold the same files as `data_all`.
- **Found, not fixed:** SAM's bbox is one pixel short (`w = right - left`), a black photo pixel inside a mask counts as not visible, running WC cell 14 overwrites a CSV and a PNG on Drive, `configs/` is not part of a non-editable install (see the two Stage sessions).
- **Outputs not to trust:** `cleaned_images_crops/`, `cleaned_jsons_filtered/` and the bbox DINOv2 features. All three cut crops from the wrong part of the photo.

**Next**
- The full organization, Stages A1 -> C of `docs/agent_prompts/implement_organization_plan.md` is written, and the baseline has run (2026-10-12). **Next: one end-to-end run in Colab, to ask the user for:** (1) `01_segment_and_filter`, all cells, GPU runtime, about 12 min (paste back the status line, the segmentation check's verdict and table, the funnel); (2) `02_zero_shot`, all cells, about 10 min (it loads the existing baseline; the landscape expert line must read 34/51 = 66.7%); (3) optionally `notebooks/explore/embeddings.ipynb` on the BioCLIP features (about 15-25 min on a CPU). Exact steps and what to paste back are in `sessions/2026-10-13_detach_exploration_remove_v2.md`. After the user confirms it, `notebooks/legacy/` can go. The blind review export of `03_expert_review` is on hold (`RUN_EXPORT = False`), to be combined with the larger expert set of algorithm idea 1.1 so Assaf reviews once.
- The 20-genus run on all images is the **new baseline** (`<run>/predictions/` and `<run>/features/`). It builds on the manifest that Check C saved to `<run>/manifest/`. Report its expert results for landscape and portrait rows apart: Assaf labeled crops cut by the legacy code, which were mis-cut on portrait photos.
- **Millepora, redone:** the blind sheet (drawn only from the 20-genus baseline, source recorded and checked at scoring), the scoring (precision, miss rate and recall with intervals, the analyst against Assaf, what the false positives really are) and the portrait re-review are built. Nothing from the old audit is reused.
- **Parity established (2026-10-11).** On landscape photos, `coralid` reproduces the legacy zero-shot results exactly (773/773 crops, 100% top-1, max probability difference 9e-6).
- **Portrait photos were broken in the legacy code.** 221 of 1,714 photos (13%) are portrait. The legacy classification, review and audit code squashed them to 2560×1920 without moving the boxes:
  - it kept only about 25% of their crops;
  - only 32% of those got the same label as with correct crops.
- **The new image manifest keeps 16,080 crops** (legacy 14,224). Only 117 were dropped for blur, so the blur filter is nearly inactive on masked crops.
- See `sessions/2026-10-11_parity_run_1.md`.
- **The expert CSV is the reference:** 77 rows, **36/54 = 66.7%** in-set. The memory matrix (39/57) is a slightly different version of the same review.
- Then the algorithm plan. **All ideas so far are collected in `docs/algorithm_ideas.md`.** It covers blur filtering, a bigger blind expert set, a reject option, the linear probe, calibration, taxonomy, resolution and segmentation.
  - Blur is a priority because the user saw that most of Assaf's unidentifiable crops were blurred.
  - But 7 of the 16 were mis-cut portrait crops, so check the 9 landscape ones first.

**Constraints**
- The repo is **public**, so nothing private goes in this folder.
- The runtime is Colab for now.

## Conventions for adding to this folder

- **One file per session:** `sessions/YYYY-MM-DD_<topic>.md`, with sections for *What was done*, *Verified* (with the evidence), *Conclusions*, *Open questions* and *Next*. Record what was checked, not what was assumed, and say how it was checked: code cell, file, row count.
- **Keep the snapshots as they are.** When later work corrects one, add the correction to the file's "Notes" block at the top, not to the original text.
- **Update "Current understanding"** above at the end of every session. Keep it short; details go in the session file.
- **No secrets.** The repo is public. Never store credentials, tokens, budgets, personal details or private links here.
