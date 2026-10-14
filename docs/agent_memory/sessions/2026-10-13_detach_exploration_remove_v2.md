# 2026-10-13: Exploration detached, V2 and the old Millepora audit removed, the blind review tied to the 20-genus baseline

Claude Code cloud session on branch `claude/awesome-archimedes-kvh4c7`. It followed the user's decisions of 2026-10-13 and three amendments. **It had no Drive access, so nothing here has run on the real data, on a GPU or in Colab.** The notebooks were dry-run on a synthetic Drive (see "Dry runs"). Nothing was written to, moved on or deleted from Drive.

## The decisions

1. **V2 is excluded entirely.** The V2 pipeline (24 genera, tiers, its outputs in Drive's `v2_pipeline/`) is not a reference for anything. The code and the current docs refer only to `coralid` and, for provenance, to `weak_classification.ipynb` and `CoralMasking.ipynb`.
2. **The old Millepora audit is discarded.** It was made on V2's predictions. There is currently **no valid Millepora precision figure**; the blind review sheet built from the 20-genus baseline is the only way to redo it.
3. **Embeddings and clustering are a separate direction**, not part of the pipeline. The pipeline is segment, manifest (filters), zero-shot baseline, evaluation and expert review, linear probe (on the BioCLIP embeddings the baseline saves).
4. **Amendments:** `notebooks/legacy/Millepora_Analysis.ipynb` is kept (as history only; `CoralClassification_v2_Pipeline.ipynb` is deleted, git history keeps it); a provenance audit of V2's actual code, not only the word; and the blind review must use the 20-genus baseline, checked and recorded.

## What was done (one commit per piece)

| Piece | What it is |
|---|---|
| `coralid.explore` | `embed.py`, `clustering.py`, today's `explore.py` (now `analysis.py`), the DINO encoders out of `encoders.py` (`dino.py`: `DINO_MODELS`, `DinoV2Encoder`, `DinoVdeauEncoder`, `load_dino_encoder`, `ImageEncoder`) and `embedding_scatter` / `category_palette` out of `viz_plots.py` (`plots.py`) moved into a subpackage. `encoders.py` keeps only the CLIP-family models. Their tests moved to `tests/explore/`. `crop_gallery_pdf`, `mask_overlay`, `confusion_heatmap` and `funnel_chart` stay in `viz` / `viz_plots` (generic, used by the pipeline notebooks). |
| Independence | `tests/test_structure.py`: an AST scan of every module outside `coralid/explore/` fails if one imports it (any form, relative imports included); `cli.py` may, only inside the `explore-embed` command. A fresh interpreter imports the whole pipeline and `coralid.explore` is not in `sys.modules`. The `probe` extra needs none of the exploration extras. |
| CLI and config | `embed` -> `explore-embed` ("EXPLORATION, not part of the pipeline" in its help; it handles its own errors, since `cli.py` no longer imports the embed module at the top). The config section `embed:` -> `explore:` (a config with the old name is refused with a message that says so). `pyproject.toml` labels the `dino` and `cluster` extras as exploration extras. |
| Notebooks | `git mv`: `03_zero_shot` -> `02_zero_shot`, `04_expert_review_and_audits` -> `03_expert_review`, `05_supervised_probe` -> `04_supervised_probe`, `02_embeddings_explore` -> `notebooks/explore/embeddings.ipynb`. Badges, intros and cross-references updated; `tests/test_notebooks.py` holds the new layout (four numbered pipeline notebooks, none using `coralid.explore`; the explore notebook says it is not a step of the pipeline). |
| Old audit removed | `load_legacy_audit`, `audit_agreement`, the forced re-review rows (`FORCED_UNSURE`, `FORCED_PORTRAIT`) and the exclusion of judged crops (`PRIOR_DECIDED`) in `genus_audit`; `audit_portrait_impact` and the audit part of the portrait report; `jobs.LEGACY_AUDIT`; `cli --audit-csv`; the census and prior-answer logic of `review_scoring` (`pooled_rate(sample, pool_size)` now); `labeling.audit_sheet`, `summarize_audit` and the key-less fallback of `ingest_review`. The 58.9% and 60.0% numbers no longer appear in code or tests. The key column `new_pred` is now `baseline_pred`. The sampler (no audit) gives the same sheets and key as before for the same seed (checked against the previous module on 4 settings; only the string dtype of `key` differs). |
| The baseline source (amendment 3) | New `coralid.review_source`. `export_blind_review` calls `baseline_source(cfg)` before sampling: it resolves the file through `baseline_paths(cfg)`, reads the `.meta.json`, and **refuses** unless the label set is `hard20` and the model and prompt set are the config's (no meta, no file, other label set, model or prompt set are all refused, and nothing is written). `build_blind_review` refuses a sampled crop that is not a `keep == True` manifest row, and predictions whose row count is not the source file's. The source (file, path, git commit and creation time of the baseline job, model, prompt set, label set, rows) is stored in the review's `meta["source"]` and so in `<name>_answer_key.meta.json` in `keys/`. `score_blind_review(..., current_source=baseline_source(cfg))` raises `ReviewSourceError` naming the differing fields (creation time, commit, model, ...) if the baseline was made again or is another one, or if the key records no source. The export prints the source; notebook `03_expert_review` prints it before sampling and again at scoring. |
| V2 removed | `taxonomy_v2` and `soft4_taxonomy` prompt sets; Sarcophyton (V2's) from `soft_genera`, which is now the soft corals of the expert set (Cladiella 3 crops, Sinularia 1, Litophyton 1); the legacy-CSV adapter keeps only label-set genera as probability columns; Check C of the parity checks lost its comparison with V2's kept keys (`load_v2_keys`, `compare_kept_with_v2`, C3) and its rows are now C1 to C6; the parity notebook was updated the same way; every V2 mention in code, configs, tests and notebooks; `notebooks/legacy/CoralClassification_v2_Pipeline.ipynb` (`git rm`). `tests/test_no_v2.py` fails if `src/`, `configs/`, `tests/` (except itself) or the non-legacy notebooks mention V2 (capital `V2` as a word, or the names of its files and notebook); `notebooks/legacy/` is not scanned. |
| Docs | `README.md`, `CLAUDE.md`, `docs/algorithm_ideas.md` (our own evidence only; the soft-coral distractor prompts and the Merulinidae prompt version are "prompts to write and compare"), new `docs/exploration_embeddings.md`, `docs/onboarding_errata.md`, a note at the top of `docs/organization_plan.md`, a line in the Notes of the three snapshots, `docs/agent_memory/README.md`. Files in `sessions/` and `docs/agent_prompts/` are untouched (history). |

## Provenance audit: what `coralid` took from V2

V2 = `CoralClassification_v2_Pipeline.ipynb` (read from git for this audit only), MA = `Millepora_Analysis.ipynb` (which analyzed V2 output). "WC" and "CM" are `weak_classification.ipynb` and `CoralMasking.ipynb`.

| Item | `coralid` location | V2 cell | Decision | Reason |
|---|---|---|---|---|
| 24 genera, 4 soft-coral classes, their prompts | `configs/genera.yaml` `soft4_taxonomy`; Sarcophyton in `soft_genera` | 4, 9 | **removed** | V2-only. `soft_genera` now lists the soft corals of the expert set (Cladiella, Sinularia, Litophyton); it only fixes spelling and silences "unknown genus" notes, never a status (the real expert set still gives 54 in_set, 5 out_of_set, 16 unidentifiable, 2 missing, checked by the tests built on its shape). Names in the survey counts file could not be checked (no Drive access) |
| `taxonomy_v2` prompt set (Merulinidae variant) | `configs/genera.yaml` | 9 | **removed** | V2-only prompt set. A current-classification variant is "a prompt set to write and compare" (algorithm idea 6) |
| Tiers (`TIER1_GENERA`, `TIER2_GENERA_CONFUSED`, `assign_tier`, `auto_accept` / `expert_review`) | none in `coralid` | 4, 11-12 | never carried | verified by search: no tier code exists |
| Soft-coral probability columns / V2 CSV layout in the legacy adapter | `predictions.load_legacy_predictions`, `compare_predictions` | 10 (24 genus columns) | **removed** | existed only to read V2's CSV; a model predicts only label-set genera. New `GeneraConfig.label_set_genera()`. `known_genera()` (with soft corals) remains for spelling only |
| Comparison with V2's kept keys | `parity_manifest.load_v2_keys`, `compare_kept_with_v2`, row C3, parity notebook | 10 (`v2_weak_labels_expanded.csv`) | **removed** | reads V2 output. Check C keeps its completeness rows |
| V2 file paths | `jobs.LEGACY_AUDIT` (`v2_pipeline/millepora_sharp_audit.csv`); parity notebook "V2 labels" path | 4 (`v2_dir`) | **removed** | V2 file layout |
| Old audit machinery, all reading V2 output | `genus_audit.load_legacy_audit`, `audit_agreement`, forced rows; `portrait_impact.audit_portrait_impact`; `review_scoring` census; `cli --audit-csv`; `labeling.audit_sheet`, `summarize_audit`; `ingest_review` fallback for sheets without `key` | (MA 2, 7-8 on V2's `v2_tiered_labels.csv`) | **removed** | decision 2. The fallback existed only to read MA's key-less sheet |
| V2 Phase 2 join on `coral_id` alone, pickle, in-sample accuracy | none (`probe.py` joins on the key) | 16-18 | never carried | the bug the probe was written to avoid |
| Resumable runner: done-set file, numbered part files, resume | `batch.ResumableRun` | 10 (`CHECKPOINT_EVERY`, `checkpoint_processed_files.json`, `*_part_NNNN.csv/npz`) | **kept** | generic engineering. The idea is V2's; the implementation is independent (done-set written only after the parts are on disk, atomic writes, a failure log, parquet/npz parts keyed by image, input fingerprints) and defined by our tests. No code copied |
| Embeddings saved with the predictions | `baseline.run_baseline` -> `features/<model>__masked.npz` | 9-10 (`v2_embeddings.npz`) | **kept** | generic. V2 stored `embeddings, coral_ids, image_names` (joined on `coral_id`); ours stores `uids` (`image#id`) and `X`. No code copied |
| `runner_up` column (and `margin`) | `zero_shot.prediction_table` | 10 (`get_runner_up`, a row-wise `apply`) | **kept** | generic; WC cell ~25 derives the runner-up on the fly from the scores for an analysis. Ours is an `argsort`; `margin` is new. No code copied |
| Review sheet and PDF export | `labeling.export_review_sheet`, `review_export.write_blind_review`, `viz.crop_grid_pdf` | 13 (CSV with `expert_label`, PNG grids per genus); MA 4, 6, 9 (PDFs) | **kept** | generic. Our sheet is blind (key plus empty answer columns, a hidden key). The PDF layout (4 x 3, 12 x 16 in) and the column names `actual_genus` and `notes` follow MA; the code is independent (object-oriented matplotlib, numbered captions). No code copied |
| Linear probe | `probe.py`, `jobs.train_probe` | 17-18, 20 | **kept** | generic, rewritten (grouped cross-validation by photo, held-out report only, class weights, key join, no model pickle). Defaults `C=1.0`, `class_weight=balanced`, `max_iter=2000` equal V2 17's values. No code copied |
| Genus mix against the survey | `predictions.distribution_vs_ground_truth` | 19 | **kept** | also in WC (the `Observation` / `#` renames and the ratio table); not V2-only |
| Filters, masked crop (`min_visible` 500, padding 0.05), Laplacian sharpness at 80, batch size 64, BioCLIP-FT loading, softmax(100 cos) | `filters`, `crops`, `manifest`, `baseline`, `zero_shot` | 6, 9, 10 | **kept** | identical in WC 8 and 14; not V2-specific |
| `sharpness_crop = "bbox"`, threshold 150 | `config.FilterConfig`, `manifest._sharpness` | none (V2 measures the masked crop at 80); MA 5, 9 | **kept, for the user to decide** | an alternative measure for blur work (algorithm idea 2), default unchanged. Its origin is the old audit's sampling rule, not V2's code. Docstrings no longer cite it. Removing it is the two config fields, one branch and three tests |
| Blind review sampling (random calls, random others, expert subset, seed) | `genus_audit.build_blind_review` | none | **kept** | designed in Stage B; not in V2 or MA |

## Verified (with evidence)

| Fact | How it was checked |
|---|---|
| Checks | `ruff check .`, `ruff format --check .`, `pytest -q`: all pass; `pytest -q`: 581 passed and 6 skipped (the six need torch or umap) without torch, `ruff check .` and `ruff format --check .` clean; it was 566 and 6 before this session. `import coralid` and `python -m coralid --help` work without torch; `nbstripout --dry-run` changes no notebook |
| The pipeline does not import the exploration subpackage | `tests/test_structure.py` (static scan, 7 import forms checked on synthetic modules; `cli.py` only inside a function; a fresh interpreter) |
| Wrong label set, model or prompt set is refused, nothing written | `tests/test_review_source.py` (edits the baseline's meta file) |
| The key records the source | `tests/test_jobs.py::test_the_key_records_the_baseline_the_review_was_drawn_from` compares the key's meta with the baseline's meta file |
| Scoring against another baseline fails | `tests/test_jobs.py::test_scoring_against_a_baseline_that_was_made_again_fails`, `tests/test_review_scoring.py`, and the `03_expert_review` dry run below |
| V2 is gone from the code | `tests/test_no_v2.py`; `grep -rniE "\bV2\b|v2_pipeline|millepora_sharp_audit" src configs tests notebooks/*.ipynb notebooks/explore README.md CLAUDE.md docs/algorithm_ideas.md docs/exploration_embeddings.md` |

## Dry runs (synthetic Drive, in the scratchpad, not committed)

The harness of the Stage C2 session, with the new notebook names and module paths (real `coralid`, fake CoralSCOP, a random-weight ViT-B/16 for BioCLIP, a colour-histogram stand-in for DINO; Colab, git and pip stubbed). 44 photos (36 landscape, 8 portrait), 267 masks, 223 kept.

| Notebook / scenario | What ran | Result |
|---|---|---|
| `01_segment_and_filter`, faithful fake CoralSCOP (masks shifted one pixel) | all cells | PASS: 5 of 5 photos compared; the active `json/` byte-for-byte unchanged; the only new files in the active run were `reports/segcheck_<date>/` |
| 01, a fake that drops the largest mask | all cells | WARN |
| 01, a fake with unrelated masks | all cells | FAIL |
| 01, a fake that raises on portrait photos | all cells | FAIL: 3 of 5 compared, 2 "not segmented"; no JSON written for them |
| 01, no saved manifest | all cells | `build_manifest` built it: 267 masks, 223 kept |
| 01, two JSONs deleted and `SEGMENT_MISSING = True` | all cells | "42 have a mask JSON, 2 need one"; the 2 JSONs written; the check PASS |
| `02_zero_shot` | all cells | 223 crops with a random-weight ViT-B/16, then the whole report (the expert line is 0/46 because the weights are random) |
| `03_expert_review`, on hold (default) | all cells | the portrait-impact files only; the export and scoring cells print their on-hold messages |
| 03 with `RUN_EXPORT = True` | all cells | printed "The review is drawn only from:" with the file, the model `finetuned-bioclip`, prompt set `taxonomy_v1`, label set `hard20`, 223 crops and the commit and time of the baseline job; wrote the sheets, PDFs and key; the key's meta file holds the same source |
| 03, simulated returned sheets, then scoring | the scoring cell | ran, printed "The key was drawn from ... made 2026-10-13T04:23:15Z / The baseline now is ... made 2026-10-13T04:23:15Z", scored, reported a bad answer as a problem |
| 03, the baseline's meta file changed to a later creation time (the baseline "made again"), then scoring | the scoring cell | **stopped** with `ReviewSourceError: ... created_utc: the key says '2026-10-13T04:23:15Z', the baseline now says '2031-01-01T00:00:00Z'` |
| `04_supervised_probe`, expert labels | all cells | 36 labels on 23 photos from 60 synthetic expert rows, held-out report, the tables and the heat map |
| 04 with `LABELS_CSV` and `LABEL_COLUMN` | all cells | 46 labels from a custom table |
| `notebooks/explore/embeddings.ipynb`, BioCLIP features | all cells | 223 x 512, PCA keeps 98.8%, 12 K-Means clusters (cosine silhouette 0.240), HDBSCAN found 3 clusters, the report folder and the gallery PDF |
| explore notebook with `RUN_EMBED = True` (colour-histogram stand-in for DINO) | all cells | `explore-embed`'s job wrote `features/dinov2-s14__masked_square.npz` (223 x 24), then the rest ran |
| `parity_check.ipynb` (Check B skipped: it needs the Drive CSV that the synthetic Drive lacks) | all other cells | ran to the end; Check C now has rows C1 to C6 and no V2 row; A, C1 and D report FAIL against the real numbers, as expected for 44 photos |

**What the dry runs could not cover:** the real CoralSCOP weights and a GPU; the real BioCLIP and DINO weights; the real data's size and speed; Colab itself; `umap` and `hdbscan` at 16,000 crops. The synthetic expert set is scored by a random-weight model, so its accuracy line is 0/46, not 34/51.

## The end-to-end run to ask the user for (in Colab)

Cheap: about 25-35 minutes on a T4, two notebooks and an optional third, no overwrites, nothing long-running (estimates, apart from the baseline load).

1. **`01_segment_and_filter`, all cells, GPU runtime** (about 12 min). Leave `SEGMENT_MISSING = False`. Paste back: the status line ("1714 photos: 1714 have a mask JSON, 0 need one"), the segmentation check's lines (verdict, reasons, masks, found again, mean IoU) with the per-photo table, the `already_done: ... masks on ... photos, ... kept` line and the funnel table. If the verdict is WARN or FAIL, that is the result to bring back: its limits are guesses.
2. **`02_zero_shot`, all cells, GPU runtime** (about 10 min; the baseline files exist, so it loads them). Paste back: the printed expert tables. **The landscape expert line must read 34/51 = 66.7%.**
3. Optional: **`notebooks/explore/embeddings.ipynb`** on the BioCLIP features, any runtime (about 15-25 min on a CPU; set `MAKE_GALLERY = False` to save the last 3-8). Paste back: the PCA and K-Means lines, the first rows of the two cluster tables, the HDBSCAN line, and whether it finished.

After the first run, in `03_expert_review` the export is on hold; when it is switched on it prints the source it draws from (check it names `finetuned-bioclip__taxonomy_v1__hard20__all.parquet`, the model, `taxonomy_v1` and `hard20`).

## Conclusions

- **The pipeline is `01` to `04`; exploration is outside it** and enforced to stay there by tests.
- **No number from V2 or the old audit is evidence anywhere in the code or the current docs.**
- **The blind review can only be about the 20-genus baseline**, and its score can only be computed against the baseline it was drawn from.

## Open questions and found, not fixed

- Decision for the user: keep or drop `FilterConfig.sharpness_crop = "bbox"` (see the table).
- The survey counts file's soft-coral names could not be read (no Drive access). `soft_genera` is the expert set's three; add others there if a review or the survey uses them. It changes no status and no number.
- The module is still called `genus_audit` although it now only builds the blind review (kept because the user's instructions name it); `review_scoring` and `review_export` are named for the review.
- The parity notebook's Check B/C numbering changed: Check C rows are C1 to C6 (C3, the V2 comparison, is gone).
- The parity notebook's verbatim legacy cell (copied from `weak_classification.ipynb`) says "BioCLIP v2" for the BioCLIP 2 model; that is a model name, not V2, and the case-insensitive grep from the task statement finds it once in `notebooks/parity_check.ipynb`.
- Found before and still open: see the Stage C1 and C2 sessions (CoralSCOP's loader has no `map_location`; whether the generator can raise for a photo with no masks; the segmentation check's limits are guesses; `data_50` is assumed to hold the same files as `data_all`).

## Next

1. The user runs the end-to-end run above and pastes back what it asks for.
2. After confirmation: delete `notebooks/legacy/` (CoralMasking, weak_classification, and Millepora_Analysis if no longer wanted as history).
3. The larger blind expert set (algorithm idea 1.1) together with the Millepora blind review.
