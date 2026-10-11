# Agent memory

This folder holds project memory and conclusions that should outlive a single agent session: who is involved, what has been tried, what the expert reviews showed, and what each session concluded. It doesn't replace the code docs. `CLAUDE.md` covers the repo and Drive layout, and `docs/organization_plan.md` covers the refactor.

## Read order

1. **Current understanding** (below): the short version, updated every session.
2. `sessions/`, newest first: what each session did, verified and concluded.
   - `sessions/2026-10-11_stage_A1.md`: Stage A1 of the refactor (packaging, tooling, core data modules), what was verified, and the bugs found but not fixed.
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
- **Millepora:** 58.9% precision on sharp, confidence ≥ 0.85 V2 predictions (66 yes / 46 no / 17 unsure). So Millepora is *not* reliable, even though older memory lists it as reliable. The crops and judgments are valid, but the figure is for the 24-genus setup. Re-run the analysis on the 20-genus full run, reusing the 129 judgments.

**Known failure modes**
- The massive-coral complex: Dipsastraea, Favia, Favites, Goniastrea.
- Acanthastrea vs Lobophyllia.
- Soft corals: addressed by V2's label set.
- Pavona over-prediction: 13.7% of v1 predictions and 17.6% of V2 hard-coral predictions, against 0.4% in the survey counts.
- **Taxonomy drift.** Most Indo-Pacific *Favia* were moved to *Dipsastraea*, yet `ground_truth_genus_counts.csv` lists both, so the survey itself mixes old and new names.

**Code**
- **State (2026-10-11):** Stage A1 of the refactor is done. The `coralid` package exists (`src/coralid/`: `config`, `masks`, `images`, `crops`, `filters`, `manifest`) with 117 CPU tests, CI, ruff and pre-commit. The legacy notebooks and V2 are frozen in `notebooks/legacy/`. It reproduces the legacy behavior exactly (checked against verbatim copies of the notebook code), and containment now runs on the RLE. Nothing has been run on real data yet: that is the Stage A2 parity check in Colab.
- **Found, not fixed:** SAM's bbox is one pixel short (`w = right - left`), a black photo pixel inside a mask counts as not visible, and the V2 notebook carries Colab account metadata (see `sessions/2026-10-11_stage_A1.md`).
- **Outputs not to trust:** `cleaned_images_crops/`, `cleaned_jsons_filtered/` and the bbox DINOv2 features. All three cut crops from the wrong part of the photo.

**Next**
- The full organization, Stages A1 → C of `docs/agent_prompts/implement_organization_plan.md`. All decisions are made, and the package is named `coralid`. **A1 is done; A2 is next** (zero-shot path, evaluation, batch runner, parity notebook), and it needs the user to run the parity notebook in Colab.
- Stage B's 20-genus run on all images is the **new baseline**.
- Don't expect the refactor alone to raise accuracy much: the bugs were mostly in side outputs, not in the zero-shot path.
- Then the algorithm plan.

**Constraints**
- The repo is **public**, so nothing private goes in this folder.
- The runtime is Colab for now.

## Conventions for adding to this folder

- **One file per session:** `sessions/YYYY-MM-DD_<topic>.md`, with sections for *What was done*, *Verified* (with the evidence), *Conclusions*, *Open questions* and *Next*. Record what was checked, not what was assumed, and say how it was checked: code cell, file, row count.
- **Keep the snapshots as they are.** When later work corrects one, add the correction to the file's "Notes" block at the top, not to the original text.
- **Update "Current understanding"** above at the end of every session. Keep it short; details go in the session file.
- **No secrets.** The repo is public. Never store credentials, tokens, budgets, personal details or private links here.
