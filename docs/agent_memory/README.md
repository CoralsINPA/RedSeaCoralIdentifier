# Agent memory

This folder holds project memory and conclusions that should outlive a single agent session: who is involved, what has been tried, what the expert reviews showed, and what each session concluded. It doesn't replace the code docs. `CLAUDE.md` covers the repo and Drive layout, and `docs/organization_plan.md` covers the refactor.

## Read order

1. **Current understanding** (below): the short version, updated every session.
2. `sessions/`, newest first: what each session did, verified and concluded.
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

Last updated 2026-10-11.

**Pipeline**
- **Decided 2026-10-11:** the final label set is the **20 hard genera**. The main line is the WC path: `weak_classification.ipynb`, BioCLIP-FT zero-shot over CoralSCOP masks from run `iou60_sta55_pts32` (44,211 masks, about 14,224 crops after filtering).
- **V2** (24 genera, Drive only) is not carried forward as a pipeline. Its useful engineering (embedding saving, runner-up, checkpointing, review export) moves into the package. Its classifier core is identical to WC, and it was never scored against the expert set.
- **Soft corals** still appear in the photos (about 8-12% of crops). The 20-genus classifier needs a reject or "other" outcome, and that is the first algorithm question.
- **Phase 2** (a linear probe on the embeddings) is still the planned next step. Its V2 code joins on `coral_id` alone, which repeats across images, so it must be rewritten on `(image_name, coral_id)`.

**Accuracy evidence**
- **v1 on Assaf's review set:** 39/57 = 68.4% on crops whose true genus is in v1's 20 classes (95% interval 55-79%). That drops to 39/62 = 62.9% if the 5 soft corals are counted as errors, and the original 80-crop count was 38/64 = 59.4%. The authoritative matrix is in `assaf_review_confusion_matrix.md`.
- **Where the errors come from:**
  - 13 of 23 are the massive / merulinid complex; Dipsastraea alone has 6 errors from 7 crops.
  - 5 are soft corals. They pull Acropora precision down to 56%.
  - 5 are other.
- **The expert set is too small to show modest improvements.** A larger stratified review of the new baseline is needed.
- **Millepora:** a **non-expert analyst** (not Assaf) judged V2's sharp, confidence ≥ 0.85 Millepora predictions: 66 yes / 46 no / 17 unsure, so 58.9% of the decided ones, or 51-64% counting the unsure. It's a useful warning sign, not a measured precision:
  - the reviewer wasn't an expert;
  - the reviewer saw the model's answer;
  - only the most confident, sharpest slice was checked;
  - it was the 24-genus setup.

  Redo it on the 20-genus baseline: blind, mixed with non-Millepora crops, sampled across confidence levels, with `actual_genus` recorded, and with Assaf labeling a subset to measure how well the analyst agrees with him.

**Known failure modes**
- The massive-coral complex: Dipsastraea, Favia, Favites, Goniastrea.
- Acanthastrea vs Lobophyllia.
- Soft corals: addressed by V2's label set.
- Pavona over-prediction: 13.7% of v1 predictions and 17.6% of V2 hard-coral predictions, against 0.4% in the survey counts.
- **Taxonomy drift.** Most Indo-Pacific *Favia* were moved to *Dipsastraea*, yet `ground_truth_genus_counts.csv` lists both, so the survey itself mixes old and new names.

**Code**
- **State (2026-10-11):** Stages A1, A2 and B of the refactor are written. The `coralid` package (`src/coralid/`) has:
  - the data path: `config`, `masks`, `images`, `crops`, `filters` (with a `sharpness_crop` option: `masked` or `bbox`), `manifest`, `orientation` (landscape / portrait, and a way to split any metric by it);
  - the zero-shot path: `genera` (with `configs/genera.yaml`), `encoders`, `zero_shot`, `batch` (resumable runner), `baseline` (the 20-genus job) and `cli` (`python -m coralid zero-shot`);
  - scoring and reports: `evaluation`, `predictions`, `reports` (expert evaluation and genus mix by orientation), `portrait_impact`, and the parity checks (`parity`, `parity_manifest`, `parity_expert`, `parity_report`);
  - review and audits: `labeling`, `genus_audit` (the old Millepora audit against the baseline and the blind review sheet), `review_scoring`, `review_export`, `viz` (a crop-grid PDF so far).
  It has 342 CPU tests (two need torch), CI, ruff and pre-commit. The legacy notebooks and V2 are frozen in `notebooks/legacy/`. It reproduces the legacy behavior on landscape photos (real data, see below). Containment runs on the RLE.
- **What has run on real data:** only the parity notebook (`sessions/2026-10-11_parity_run_1.md`). **Nothing from Stage B has**: the baseline job, the portrait-impact report, the reports and the Millepora tools were tested on synthetic data and dry-run on a synthetic Drive. `notebooks/03_zero_shot.ipynb` (about 25 minutes on a T4) and `notebooks/04_expert_review_and_audits.ipynb` do the real runs. The expected 16 / 2 split of the 18 expert rows without a genus (`unidentifiable` / `MISSING_LABEL` read from `is_correct`) is still unconfirmed.
- **Found, not fixed:** SAM's bbox is one pixel short (`w = right - left`), a black photo pixel inside a mask counts as not visible, running WC cell 14 overwrites a CSV and a PNG on Drive, `configs/` is not part of a non-editable install, and the V2 notebook carries Colab account metadata (see the two Stage sessions).
- **Outputs not to trust:** `cleaned_images_crops/`, `cleaned_jsons_filtered/` and the bbox DINOv2 features. All three cut crops from the wrong part of the photo.

**Next**
- The full organization, Stages A1 → C of `docs/agent_prompts/implement_organization_plan.md`. All decisions are made, and the package is named `coralid`. **A1, A2 and B are written. The user now runs, in order:** `03_zero_shot.ipynb` (the baseline), `04_expert_review_and_audits.ipynb` (portrait impact, the old Millepora audit against the baseline, and the blind review sheets), then the analyst and Assaf fill the sheets and the last cell of 04 scores them. Then Stage C.
- The 20-genus run on all images is the **new baseline** (`<run>/predictions/` and `<run>/features/`). It builds on the manifest that Check C saved to `<run>/manifest/`. Report its expert results for landscape and portrait rows apart: Assaf labeled crops cut by the legacy code, which were mis-cut on portrait photos.
- **Millepora, redone:** the blind sheet, the scoring (precision, miss rate and recall with intervals, the analyst against Assaf, what the false positives really are) and the portrait re-review are built; the old audit's portrait rows are never reused.
- **Parity established (2026-10-11).** On landscape photos, `coralid` reproduces the legacy zero-shot results exactly (773/773 crops, 100% top-1, max probability difference 9e-6).
- **Portrait photos were broken in the legacy code.** 221 of 1,714 photos (13%) are portrait. The legacy classification, review and audit code squashed them to 2560×1920 without moving the boxes:
  - it kept only about 25% of their crops;
  - only 32% of those got the same label as with correct crops.
- **The new image manifest keeps 16,080 crops** (legacy 14,224). Only 117 were dropped for blur, so the blur filter is nearly inactive on masked crops.
- See `sessions/2026-10-11_parity_run_1.md`.
- **The expert CSV is the reference:** 77 rows, **36/54 = 66.7%** in-set. The memory matrix (39/57) is a slightly different version of the same review.
- Then the algorithm plan.

**Constraints**
- The repo is **public**, so nothing private goes in this folder.
- The runtime is Colab for now.

## Conventions for adding to this folder

- **One file per session:** `sessions/YYYY-MM-DD_<topic>.md`, with sections for *What was done*, *Verified* (with the evidence), *Conclusions*, *Open questions* and *Next*. Record what was checked, not what was assumed, and say how it was checked: code cell, file, row count.
- **Keep the snapshots as they are.** When later work corrects one, add the correction to the file's "Notes" block at the top, not to the original text.
- **Update "Current understanding"** above at the end of every session. Keep it short; details go in the session file.
- **No secrets.** The repo is public. Never store credentials, tokens, budgets, personal details or private links here.
