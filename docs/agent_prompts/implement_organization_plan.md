# Implementation prompt: code organization plan

Written 2026-10-11. This file implements `docs/organization_plan.md` in stages, and each stage runs in its own agent session.

## For the human: how to run it

The stages have hard stops between them because some steps need you to run something in Colab on a GPU. The agent can't do that.

| Session | Message to send the agent | Then you… |
|---|---|---|
| 1 | `Read docs/agent_prompts/implement_organization_plan.md and do Stage A1. Stop when A1 is done.` | Skim the report. |
| 2 | `Read docs/agent_prompts/implement_organization_plan.md and do Stage A2. Stop when A2 is done.` | Have the diff reviewed, then run `notebooks/parity_check.ipynb` in Colab and copy its summary cell. |
| 3 | `Read docs/agent_prompts/implement_organization_plan.md and do Stage B. Parity results: <paste summary>` | Run the full 20-genus job in Colab when the agent asks. |
| 4 | `Read docs/agent_prompts/implement_organization_plan.md and do Stage C.` | Run one end-to-end pass in Colab before `legacy/` is deleted. |

Start each session on a branch that already contains this file. Until `claude/awesome-archimedes-kvh4c7` is merged into `main`, that means that branch.

---

## The prompt (everything below is for the agent)

You are implementing a planned refactor of a research codebase. The design work is done: your job is to carry it out exactly, with tests, without changing behavior unless the stage says so.

### Project in brief

- **What it does.** Genus-level coral identification in Red Sea reef photos, for the Israel Nature and Parks Authority. CoralSCOP (a coral-tuned SAM) segments each photo into coral masks. A CLIP-family model (BioCLIP-FT, from ReefNet) then labels each mask crop zero-shot with one of **20 hard-coral genera**.
- **Where things live.** All logic currently sits in Colab notebooks. The data (about 1,714 photos, mask JSONs, label CSVs) lives on Google Drive, never in the repo.
- **Users.** A small research team that works in Colab, plus an expert reviewer who looks at crop grids and CSVs.

### Read first, in this order

1. `CLAUDE.md` (it also loads automatically): repo layout, Drive map, data access rules.
2. `docs/agent_memory/README.md`, then the newest file in `docs/agent_memory/sessions/`: decisions and verified facts.
3. `docs/organization_plan.md`. This is the spec. §2 lists the bugs, §3 is the target design (modules, conventions, layout), §4 the phases and §5 the decisions.
4. The legacy notebooks, which are the reference behavior. Read them as JSON or extract their cells; don't trust summaries. Cell numbers in the plan are 0-based.
   - `notebooks/CoralMasking.ipynb` (CM)
   - `notebooks/weak_classification.ipynb` (WC)
   - `notebooks/Millepora_Analysis.ipynb` (MA)
   - after Stage A1 they move to `notebooks/legacy/`, and V2 joins them there.

**If `docs/organization_plan.md` is missing,** you are on a branch without the plan. Run `git fetch origin claude/awesome-archimedes-kvh4c7`, merge it into your working branch, and continue.

### Decisions already made (don't reopen them)

- **Label set:** the 20 hard genera. The 24-genus V2 pipeline is not carried forward; only its engineering ideas are.
- **Package:** `coralid`, src layout, flat modules (plan §3.2-§3.3).
- **Repo:** public. Nothing private, no data files, no credentials.
- **Runtime:** Colab for now. Only `colab.py` may know about Colab; everything else takes paths from config, with `DATA_ROOT` from the `CORALID_DATA_ROOT` env var.
- **Defective Drive outputs** may be deleted in Stage B, with confirmation (§Stage B).

### Ground rules (all stages)

1. **No behavior changes in Stage A.** Reproduce the legacy logic exactly, quirks included; the reference section below spells out each one. If you find a bug, don't fix it. Add it under "Found during refactor" in your session memory file.
   - The one allowed exception: images are loaded in the mask JSON's frame (convention #1 in the plan). For 4:3 photos this gives exactly the same pixels as the legacy `resize(2560, 1920)`.
2. **Don't touch Drive except to read.** No uploads, edits, moves or deletes through any Drive tool unless a stage explicitly says so. Notebooks you write may create *new* files under the run folder's `reports/`, `features/` or `predictions/`. They must never overwrite existing files.
3. **Public repo hygiene.**
   - Never commit CSV/JSON/NPZ/parquet/image/checkpoint data from Drive, not even small files.
   - Never commit personal details, tokens or budgets.
   - Downloaded data goes to your scratchpad and stays there.
4. **Tests and CI.** Tests run on CPU with no network and no data. Mock models with a fake encoder. Every commit must pass `ruff check`, `ruff format --check` and `pytest`.
5. **Code style.**
   - Type hints, and a one-line docstring on public functions.
   - Library code uses `logging`, not `print`.
   - Heavy imports (torch, open_clip, transformers, umap, hdbscan, sklearn) go inside the functions or classes that need them, so `import coralid` stays light.
   - Keep modules under about 300 lines.
   - No `eval`. No bare `except Exception: continue` around model calls (use the batch runner's failure log).
6. **Small commits.** One logical change per commit, with clear messages, and push regularly. Don't open a pull request unless the user asks.
7. **Stop at the end of your stage,** report, and don't start the next one.
8. **At the end of every stage:**
   - Add `docs/agent_memory/sessions/YYYY-MM-DD_<stage>.md` (format in the folder README).
   - Update "Current understanding" in `docs/agent_memory/README.md`.
   - Update `CLAUDE.md` if any paths or commands changed.
9. **When something in the spec conflicts with the legacy code,** the legacy code wins in Stage A. Note the conflict in the memory file.
10. **When you're blocked on a decision that is genuinely the user's,** ask. For everything else, use the spec and sensible defaults.

---

### Stage A1: repo hygiene + core data modules

**1. Preconditions**
- Run `git submodule update --init` to get `third_party/CoralSCOP`.
- Confirm the plan file exists.

**2. Packaging** (`pyproject.toml`)
- Name `coralid`, `src/` layout, `requires-python >= 3.10`.
- **Core deps:** `numpy`, `pandas`, `pyarrow`, `opencv-python-headless`, `pycocotools`, `pyyaml`, `Pillow`, `tqdm`.
- **Extras:**
  - `clip`: `torch`, `torchvision`, `open_clip_torch>=3.3,<4`, `transformers>=4.50`
  - `dino`: `torch`, `torchvision`, `transformers`
  - `segment`: `torch`, `torchvision`
  - `cluster`: `scikit-learn`, `umap-learn`, `hdbscan`
  - `viz`: `matplotlib`
  - `dev`: `pytest`, `ruff`, `pre-commit`, `nbstripout`, `scikit-learn`
- **Pinning:** don't pin `torch` (use whatever the runtime ships); give other deps sensible lower bounds.

**3. Tooling**
- `ruff` config in `pyproject.toml` (line length 100, rules E, F, I, B, UP).
- pytest config.
- `.pre-commit-config.yaml` with ruff, ruff-format and nbstripout. nbstripout must **exclude** `notebooks/legacy/`, whose outputs stay as reference evidence.
- `.gitignore`: Python artifacts, `.venv`, `*.npz`, `*.parquet`, `*.pth`, a local `data/` folder, Colab checkpoints.

**4. CI:** `.github/workflows/ci.yml` runs on push and pull request. On Python 3.11 and 3.12 it runs `pip install -e .[dev]`, then ruff check, ruff format --check and pytest.

**5. Move the legacy notebooks**
- Use `git mv` to move the three notebooks into `notebooks/legacy/` byte-for-byte unchanged. Git should show 100% renames.
- Download `CoralClassification_v2_Pipeline.ipynb` (Drive file id `1XpY-hkSy-Zm6RY8M3PJpXSJm2FWCIdzZ`) into `notebooks/legacy/` with its outputs. If you have no Drive access, skip this and tell the user.
- Before committing V2, scan its outputs for anything private (emails, tokens). Strip any you find and say so in the report.

**6. Documentation fixes**
- Create `docs/onboarding_errata.md` with the items in plan §2.3 (the PDF itself can't be edited), and link it from `README.md` next to the PDF.
- In `CLAUDE.md`: point the notebook paths at `notebooks/legacy/`. Say that `..._masked_best.csv` is the 50-densest-image sample, not the best configuration, and that the V2 notebook is in `notebooks/legacy/`.
- In `README.md`: update the repository layout.

**7. Modules** (`src/coralid/`): follow plan §3.3 and §3.4. The exact semantics are in "Reference: exact behaviors" below.

| Module | What to build |
|---|---|
| `config.py` | Dataclasses `Paths`, `SamConfig`, `FilterConfig`, `CropConfig`, `ZeroShotConfig`, with defaults equal to today's values. YAML load/save. `run_name()` must use `round()`, not `int()` (`int(0.57*100)` is 56), and has to produce `iou60_sta55_pts32` for the current values. `write_meta(path, **info)` writes the `*.meta.json` sidecar: resolved config, git commit if available, package version, UTC timestamp, row count. Add `configs/iou60_sta55_pts32.yaml`. |
| `masks.py` | `decode_rle` and `encode_mask`, mask-JSON load/save, `CoralKey(image_name, coral_id)` with `uid = f"{image_name}#{coral_id}"` (e.g. `IMG_4033.JPG#14`), and `parse_bbox` (no `eval`). `parse_bbox` accepts lists, `"[x, y, w, h]"`, numpy-style `"[x y w h]"` and float strings. `AnnotationStore(json_dir)` gives per-image lazy loading with an LRU cache, `get(key)` and `annotations(image_name)`. JSON file name = image stem + `.json`. `audit_json_dir(json_dir)` returns the count of each frame size, the JSONs with a `note` field (failed images saved as empty), the total masks and the number of files. |
| `images.py` | `load_rgb(path)` raises `ImageReadError` on failure. `load_in_mask_frame(path, width, height)` is the only function that resizes (`cv2.resize`, default interpolation). |
| `crops.py` | `masked_crop`, `bbox_crop` and `masked_square_crop` (the CM 9 variant, for DINO later). Reproduce the legacy semantics exactly. |
| `filters.py` | `aspect_ratio`, `bbox_intersection` and `filter_annotations` with legacy semantics, but containment computed on RLE: `pycocotools.mask.area(pycocotools.mask.merge([a, b], intersect=True))`, with no full decode. `filter_annotations_with_reasons` returns the kept list (identical to `filter_annotations`) plus a per-annotation reason (`area`, `aspect`, `containment`) and, for `containment`, the id of the smaller mask that caused the removal. `laplacian_variance(crop_rgb)`. `is_sharp_enough(crop, threshold)` with **no default threshold**. |
| `manifest.py` | `geometric_manifest(json_dir)` has no images and is fast: one row per mask (image_name, coral_id, uid, bbox_x/y/w/h, area, predicted_iou, stability_score, aspect, frame_w, frame_h, stage1_reason, contained_by_smaller). `image_manifest(...)` adds visible_px, sharpness, `keep` and `reject_reason` (`area`, `aspect`, `containment`, `image_missing`, `not_visible`, `blurry`, or null) using the crop config. `image_missing` means the photo is absent or unreadable; the legacy code skips those silently. Parquet read/write with a meta sidecar. |

**8. Tests** (`tests/`, synthetic data only)
- RLE round trip, including str vs bytes counts.
- The RLE-based containment equals a decode-based reference on at least 200 random synthetic mask sets. Keep the reference implementation inside the test.
- `filter_annotations` reproduces the legacy loop on hand-built cases:
  - a small mask inside a large one, where the large one is removed;
  - equal areas;
  - non-overlapping bboxes;
  - chains (A ⊂ B ⊂ C).
- `masked_crop`: padding, clamping at image edges, the `min_visible` cutoff, and mask resize when the shapes differ.
- `parse_bbox` on every supported format.
- `run_name()`.
- A frame-consistency regression test: a crop taken from an image loaded in the mask frame lands on the mask pixels.
- `audit_json_dir` on a temp directory.

**A1 done when**
- `pip install -e .[dev]` works in a clean venv, and ruff and pytest pass.
- The legacy notebooks show as 100% renames.
- Session memory file written. Then stop and report.

---

### Stage A2: zero-shot path, evaluation, batch runner, parity notebook

**1. `configs/genera.yaml` + `genera.py`**
- **The label set:** `hard20`, in exactly this order (the column order and the tie-breaking depend on it): Platygyra, Acropora, Stylophora, Millepora, Favites, Dipsastraea, Favia, Echinopora, Hydnophora, Porites, Acanthastrea, Cyphastrea, Goniastrea, Seriatopora, Pocillopora, Montipora, Leptastrea, Galaxea, Lobophyllia, Pavona.
- **Prompt sets, each copied character-for-character from the legacy code:**
  - `taxonomy_v1`, `descriptive` and `common`, from WC cell 14;
  - `taxonomy_v2`, from V2 cell 9 (the 20 hard genera only);
  - `soft4_taxonomy`, also from V2 cell 9: the 4 soft corals, stored but unused;
  - `hybrid`, from WC: only 10 entries. Mark it incomplete; using it with `hard20` must raise a clear error.
- **Verification:** extract the dicts from the legacy notebook source with a throwaway script and diff them against the YAML, so the copy is exact. Don't commit the script.
- **Prompt formats:**
  - `taxonomy`, `descriptive`, `common`, `hybrid`.
  - `ensemble`, for open_clip models: the mean of the L2-normalized taxonomy, descriptive and common embeddings, renormalized.
  - For SigLIP: `ensemble` is the mean of descriptive and common, and `taxonomy` is silently replaced by `descriptive`, as WC does.

**2. `encoders.py`**
- A small protocol: `encode_images(batch_tensor) -> np.ndarray`, `preprocess(pil_image) -> tensor`, `encode_texts(list[str]) -> np.ndarray` for the vision-language models.
- `OpenClipEncoder(model_id)` and `SiglipEncoder(model_id)`, with model IDs as in the reference.
- Lazy imports, device auto-selected, fp32, `torch.no_grad()`.
- Put a `FakeEncoder` in `tests/` for the tests.

**3. `zero_shot.py`**
- `build_text_features(encoder, label_set, prompt_format, prompt_set)`.
- `score(image_feats, text_feats)`: softmax(100 · cos).
- `predictions_table(rows, probs, genera)`, giving the legacy columns in legacy order plus `runner_up` and `margin` (top-1 minus top-2).
- `run_zero_shot(...)`, which iterates a manifest subset image by image: load in frame, masked crop, the existing keep decision, batches of 64.
- Optionally returns the embeddings, saved as `.npz` with `uids` and `X`.

**4. `evaluation.py`**
- `load_expert_set(path)`.
  - Normalize each row to a status:
    - `in_set`;
    - `out_of_set` (e.g. Cladiella, Sinularia, Litophyton, Sarcophyton);
    - `unidentifiable`;
    - `missing` (`MISSING_LABEL`).
  - Inspect the real file to confirm the column values; see `CLAUDE.md`.
- `evaluate(predictions, expert, label_set)`.
  - Joins on `(image_name, coral_id)`, with `coral_id` cast to int on both sides.
  - Reports coverage and counts per status, plus accuracy on `in_set` and on `in_set + out_of_set`.
  - Also per-genus recall and precision, the full confusion matrix (true × predicted) and the top confusion pairs.
- `distribution_vs_ground_truth(predictions, gt_csv, label_set)`. Accept both the `genus`/`count` and the `Observation`/`#` column names.
- `compare_predictions(a, b)`: key-set equality, top-1 agreement, max and mean absolute probability difference. The parity notebook uses it.
- `load_legacy_predictions(path)`: an adapter for `coral_weak_labels_*.csv` and the V2 CSVs. Genus columns = the intersection with known genera; parse bboxes.

**5. `batch.py`.** `ResumableRun(out_dir, name)`:
- A done-set saved in a JSON manifest, keyed by `image_name`.
- Numbered part files (parquet + npz) and a `merge()` step.
- `failures.jsonl` (image, error, traceback, UTC time). Failed images are **not** marked done, so the next resume retries them.
- Atomic writes: temp file, then rename.

**6. Tests**
- Prompt formats and ensembling, using `FakeEncoder`.
- `score` against a hand-computed softmax.
- Argmax tie → first in label order.
- `predictions_table` columns.
- `evaluate` on a synthetic expert set covering every status.
- `compare_predictions`.
- `ResumableRun`: resume skips done images, retries failed ones, and `merge` is correct.

**7. Real-data check, if you have Drive access**
- Download `assaf_expert_test_set.csv` (id `1PIBGXIduvOnRn3lvsTT2ZyFtjKiUVRRD`) to your scratchpad. Don't commit it.
- Evaluate the predictions stored *in that file* against its own expert labels.
- **Expected:** 78 rows; 62 identifiable and 16 unidentifiable; 39 correct; 57 in-set. Your confusion matrix should equal the "authoritative" one in `docs/agent_memory/assaf_review_confusion_matrix.md`, cell for cell.
- Report what you actually get. The CSV is the truth.
- Add a dated note to that memory file confirming the match, or listing the cells that differ.

**8. `notebooks/parity_check.ipynb`**
Commit it with outputs stripped. The user runs it in Colab on a GPU.

- **Setup cell:**
  - mount Drive;
  - `git clone --recurse-submodules` this repo at your working branch into `/content/`;
  - `pip install -e "/content/RedSeaCoralIdentifier[clip]"`;
  - set `CORALID_DATA_ROOT`. The default is `/content/drive/MyDrive/colab notebooks/CoralsMasking`, and it must be editable in one place.
- **Check A, geometric (minutes):**
  - run `audit_json_dir`, and print the frame sizes and the failed (note) JSONs;
  - run `geometric_manifest` and compare with 1,714 files, 44,211 masks and 16,197 after area/aspect/containment;
  - also time the RLE containment.
- **Check B, zero-shot parity (about 15 min).**
  - **Sample:** the 50 images from the cached counts file `annotation_counts_area25000_cont0.8_ar3.0.csv` in the run folder, `head(50)` in file order.
  - **Legacy reference:** in one clearly labeled cell, paste the legacy WC cell 14 functions and loop **verbatim**, set to finetuned-bioclip / taxonomy / masked / best. This cell is the only place legacy code is allowed. Run it to produce reference predictions in memory.
  - **Compare:** run the same configuration through `coralid` and call `compare_predictions`. Pass = identical key sets, top-1 agreement ≥ 99.5%, and max absolute probability difference ≤ 1e-3.
  - **Informational:** also compare with Drive's `coral_weak_labels_finetuned-bioclip_taxonomy_masked_best.csv`. It was rewritten on 2026-10-11 by someone, so a mismatch there is not a failure.
  - **Legacy counts to reproduce:** 2,337 masks before filtering, 1,054 after, and 842 crops saved.
- **Check C, image manifest (about 2 h, resumable through `ResumableRun`):**
  - run `image_manifest` on all images and compare with **14,224** kept;
  - save `manifest.parquet` and its meta to `<run>/manifest/` (a new folder).
  - Stage B builds on this file.
- **Summary cell:** one PASS/FAIL table for A, B and C. Write it to `<run>/reports/parity_<date>.json` (a new file). Print it so the user can paste it into the next session.

**A2 done when**
- Tests pass, and the notebook's comparison logic is unit-tested in the package.
- The notebook has been read through cell by cell for path and import errors. You can't run it, so be careful.
- Memory updated. Then stop, and tell the user exactly what to run and what to paste back.

---

### Stage B: fixes (only after the parity summary shows A and B PASS)

If parity failed, don't start fixing. Diagnose the cause from the pasted results and the code, and propose a fix to the user first.

Each item below is its own commit. Each commit message and the session memory file state which numbers change, from what to what.

1. **Frame audit follow-up.** Check A found **221 portrait frames (1920×2560)** besides 1,493 landscape ones. The legacy code squashed portrait photos to 2560×1920 without moving the bboxes, so their crops were wrong or dropped (`docs/agent_memory/sessions/2026-10-11_parity_run_1.md`).
   - **Results of the first run are already in** (`docs/agent_memory/sessions/2026-10-11_parity_run_1.md`):
     - Check B on landscape photos is exact: 773/773 crops, 100% top-1, max difference 9e-6.
     - Every difference is on the 13 portrait photos in the sample.
     - Check C kept 16,080 crops, with only 117 dropped for blur.
   - **Make Checks B and C orientation-aware** in `parity.py` and the notebook, so a rerun reports PASS when the code is right:
     - Check B: landscape images must match exactly (keys, top-1 ≥ 0.995, max difference ≤ 1e-3). Portrait images are INFO with their counts.
     - Check C: the new manifest's kept keys on landscape images must equal the keys in `v2_pipeline/v2_weak_labels_expanded.csv` for those images. That file was made with the same filters, at 24 genera, but the keep decision doesn't depend on the label set. Report portrait counts as INFO, and drop the 14,224 total as a pass criterion.
     - Check D: expectations from the real CSV (item 2).
   - **Make the orientation split a reusable function** (frame from the manifest: `frame_w > frame_h` is landscape), so later reports can break any metric down by orientation.
   - **Portrait-impact report:** list which rows of `assaf_expert_test_set.csv` and of `v2_pipeline/millepora_sharp_audit.csv` come from portrait images, and how many of the expert "unidentifiable" rows are among them.
2. **Expert status from `is_correct`.** When `expert_genus` is empty, derive the status from `is_correct`: `unidentifiable`, or `MISSING_LABEL` → `missing`. Today all 18 such rows become `missing`. Update the Check D expectations to the real CSV: 77 rows, 54 in-set, 5 soft corals, 36 correct, and the unidentifiable / missing split once confirmed. Keep the memory matrix only as a reference.
3. **One sharpness definition.**
   - Keep the pipeline definition as the default: masked crop, black background, Laplacian variance ≥ 80. This changes no numbers.
   - Add a `sharpness_crop: masked | bbox` option, with MA's definition (unmasked bbox crop, ≥ 150) available under the `bbox` setting.
   - Document in `filters.py` that masked-crop sharpness also responds to the mask outline.
   - Changing the default is an algorithm decision: don't make it.
4. **`labeling.py`.** Per plan §3.3:
   - review-sheet export;
   - ingesting completed reviews by key;
   - a generic per-genus audit sheet and summary. The audit sheet **must** have an `actual_genus` column, required when the answer is "no".
   - Tier logic only if trivial; the 20-genus workflow doesn't need V2's tiers yet.
5. **The 20-genus baseline job**, as a CLI command (see `cli.py` below; start the module now with just this command) and a notebook cell:
   - finetuned-bioclip, `taxonomy_v1`, masked crops;
   - every `keep == True` crop in the manifest;
   - predictions as parquet plus embeddings as `.npz`, both with meta.
   - **Ask the user to run it.** It takes about 2 hours on a T4.
6. **Millepora re-analysis** (after step 5's output exists). The old judgments were made by a **non-expert analyst** who saw the model's answer, so treat them as provisional labels, not ground truth.
   - Read `v2_pipeline/millepora_sharp_audit.csv` from Drive. Its columns: image_name, coral_id, bbox, area, confidence, runner_up, sharpness, is_true_millepora, actual_genus, notes. 129 rows, judged yes/no/?.
   - Join it to the new predictions by key. Report the agreement rate on the overlap as a range: unsure counted as wrong, then unsure counted as right. Label it "non-expert agreement", not precision.
   - Build a **blind, mixed review sheet** with `labeling.py`:
     - Sample new 20-genus Millepora predictions randomly across all confidence levels. Leave out crops already judged.
     - Add about as many randomly chosen crops that the model did *not* call Millepora.
     - Shuffle. Show no prediction, confidence or runner-up.
     - Columns: key, `is_millepora` (yes/no/unsure), `actual_genus` (required when "no"), notes.
     - Export a hidden answer key separately.
   - Mark a subset of about 40 crops for Assaf, including every crop the old audit marked unsure. This measures how often the analyst agrees with the expert.
   - Add the scoring function: precision and recall estimates with intervals, analyst-vs-expert agreement, and the confusion list from `actual_genus`.
7. **Expert evaluation wired in.** Running `evaluate` on the baseline against the expert set and the ground-truth distribution gives a report in `<run>/reports/`.
8. **Drive cleanup.** These are defective (plan §2.1 #1-2) and their deletion is approved:
   - `results_all/iou60_sta55_pts32/cleaned_images_crops/`
   - `cleaned_jsons_filtered/` (in the CoralsMasking root)
   - `results_all/iou60_sta55_pts32/coral_dino_encodings.csv`

   First list exactly what you'd trash and ask the user to confirm in this session. Only then trash it, or let the user do it. Don't delete anything else.

**Don't, in this stage:** change prompts, thresholds, the label set or the temperature, or add classes. Those are for the algorithm plan.

---

### Stage C: remaining modules, thin notebooks, CLI

1. **`segmentation.py`.**
   - Load CoralSCOP's **vendored** `segment_anything` from `third_party/CoralSCOP` (put it on `sys.path` in one place). Assert that `segment_anything.__file__` is under that folder: pip's `segment-anything` lacks the coral/background head. No `strict=False` fallback.
   - Reproduce the legacy generator parameters and the JSON writer exactly (reference below).
   - Errors go to the `ResumableRun` failure log, **never** to an empty JSON. An image with genuinely zero masks gets an empty JSON with `"note": "no masks"`.
   - Optionally keep the per-mask `cate_preds` and `fc_features` behind a config flag, off by default.
   - Unit-test the JSON writer with a fake generator.
2. **`encoders.py`, DINO part:** DINOv2-S/14 and DinoVdeau, as in the reference.
3. **`probe.py`:**
   - a linear probe (logistic regression) on stored embeddings, keyed by uid;
   - cross-validation grouped by image (`GroupKFold`);
   - a held-out report, never accuracy on the training set.
4. **`clustering.py`, `viz.py` and `colab.py`** per plan §3.3. Each `viz` function takes a table plus an `AnnotationStore`, and none of them computes anything.
   - **`colab.setup()` must install the way `notebooks/parity_check.ipynb` does:** clone the repo with `--recurse-submodules`, `pip install -e`, then make the editable install importable in the running interpreter.
   - Plain `pip install git+https://...` won't work: it would miss `configs/genera.yaml` and the CoralSCOP submodule.
   - Alternatively, ship `configs/` as package data, but the submodule still needs the clone.
5. **`cli.py`:** `python -m coralid {segment, build-manifest, embed, zero-shot, evaluate, export-review, train-probe} --config ...`
6. **Notebooks** (plan §3.6): `01_segment_and_filter`, `02_embeddings_explore`, `03_zero_shot`, `04_expert_review_and_audits`, `05_supervised_probe`.
   - Each is under 15 cells, with no function definitions; it calls the library.
   - Outputs stripped.
   - They must pass `ruff check` and `ruff format --check`, because ruff 0.17 lints notebooks too.
7. **`README.md`:** a quickstart with the Colab setup, then running 01 → 04.
8. **Deleting `notebooks/legacy/`:** only after the user confirms one end-to-end run worked. Ask.

---

### Reference: exact behaviors to reproduce

These come from the legacy notebooks. Verify them there; if anything here disagrees with the code, the code wins in Stage A.

1. **Mask frame.** CM 8 computes `s = 2560 / max(w, h)` when `max(w, h) > 2560`, and the frame is `(int(w*s), int(h*s))`; otherwise the original size. 4:3 photos → 2560×1920; 3:2 → 2560×1706; portrait 4:3 → 1920×2560. The JSON `image.width` / `image.height` hold the frame, and annotation `id`s run 0..N-1 per image, in generator order.
2. **RLE.**
   - `segmentation = {"size": [h, w], "counts": <str>}`. To decode, encode `counts` to UTF-8 bytes and call `pycocotools.mask.decode` → bool.
   - CM 9 also accepts uncompressed RLE (list `counts`) and polygons through `frPyObjects` (+ `merge`).
3. **`filter_annotations(annos, min_area=25000, containment_thresh=0.80, max_aspect_ratio=3.0)`.**
   - **Step 1:** keep `area >= min_area`.
   - **Step 2:** keep `aspect_ratio <= max_aspect_ratio`, where `aspect_ratio = max(w, h) / max(min(w, h), 1)`.
   - **Step 3:** if 1 or fewer remain, return them.
   - **Step 4, the pairwise loop:**
     - `for i in range(n): for j in range(n):`, skipping when `i == j` or `cand[j].id in to_remove`;
     - `small = cand[i]`, `large = cand[j]`; skip when `large.area <= small.area`, or when the bboxes don't intersect;
     - `containment = |small ∧ large| / small["area"]`. The denominator is the annotation's **`area` field**, not a pixel count;
     - if `containment >= thresh`, add **`large.id`** to `to_remove`.
     - `i` is **not** skipped when `cand[i]` is already removed.
   - **Step 5:** return the candidates whose id is not in `to_remove`, in their original order.
4. **`masked_crop(img_rgb, bbox, rle, padding_pct=0.05, background="black", min_visible=500)`.**
   - Decode the mask. If its shape differs from the image's, resize it with `INTER_NEAREST`.
   - Background value: black = 0, gray = 128, white = 255. Set `img[~mask] = bg` on a copy.
   - `x, y, w, h = map(int, bbox)`; `pad_w = int(w * pct)`, `pad_h = int(h * pct)`.
   - Clamp: `x1 = max(0, x - pad_w)`, `y1 = max(0, y - pad_h)`, `x2 = min(W, x + w + pad_w)`, `y2 = min(H, y + h + pad_h)`.
   - Return `None` if the crop is empty, or if `np.any(crop != bg, axis=2).sum() < min_visible`.
5. **`bbox_crop(img, bbox, padding_pct=0.15)`:** the same clamping, without a mask.
6. **`masked_square_crop` (CM 9).**
   - `cv2.bitwise_and` with mask × 255.
   - Clamp as `x = max(0, x)`, `y = max(0, y)`, `w = min(w, W - x)`, `h = min(h, H - y)`, then crop.
   - Return `None` if the crop is empty.
   - Pad to a square with black: `top = (side - h) // 2`, remainder at the bottom; the same left/right.
7. **Sharpness (the pipeline's):** `cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() >= 80` on the masked crop from item 4.
8. **Per-crop order (WC 14 / V2 10):**
   1. load the image;
   2. resize it to the frame;
   3. convert BGR→RGB;
   4. `filter_annotations`;
   5. for each kept mask, `masked_crop`, skipping on `None`;
   6. the sharpness check, skipping if blurry;
   7. preprocess → batch of 64 → encode.
9. **Zero-shot scoring.**
   - Image and text features are each L2-normalized, and `probs = softmax(100.0 * img @ text.T)` over the label set.
   - `predicted_genus` = argmax (the first on ties, as in pandas `idxmax`), and `confidence` = the max probability.
   - open_clip: `model, _, preprocess_val = open_clip.create_model_and_transforms(model_id)`, applied to `PIL.Image.fromarray(rgb_crop)`. fp32, no autocast.
   - SigLIP: `AutoProcessor` / `AutoModel`. Texts use `padding="max_length"`. `get_text_features` and `get_image_features` may return an object; if it isn't a tensor, use `.pooler_output`.
10. **Model IDs:**
    - `finetuned-bioclip` → open_clip `hf-hub:ReefNet/finetuned-bioclip`
    - `bioclip2` → open_clip `hf-hub:imageomics/bioclip-2`
    - `siglip2` → transformers `google/siglip2-so400m-patch14-384`
11. **Legacy output.** Columns: `image_name, coral_id, bbox, area, <20 genus probabilities in label order>, predicted_genus, confidence`. File name: `coral_weak_labels_{model}_{prompt}_{crop_mode}_{sample_tag}.csv`.
12. **The "best" sample.** WC 12's counts cache (`file`, `raw`, `filtered`), sorted by `filtered` descending, `head(50)`. Use the cached CSV from Drive; don't recompute the order (ties).
13. **SAM, for Stage C.**
    - Model `vit_b`, with checkpoint `CoralSCOP/checkpoints/vit_b_coralscop.pth` on Drive.
    - `SamAutomaticMaskGenerator(points_per_side=32, pred_iou_thresh=0.60, stability_score_thresh=0.55, crop_n_layers=1, crop_n_points_downscale_factor=2, min_mask_region_area=100)`.
    - The input is RGB in the frame.
    - Each JSON annotation has: `segmentation` (RLE, str counts), `bbox`, `area`, `predicted_iou` (float), `crop_box`, `stability_score` (float), `point_coords` (list) and `id`.
14. **DINO, for Stage C.**
    - **DINOv2:** `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`. Input is the masked square crop, then `Resize((224, 224))`, ImageNet mean/std, fp16 autocast, batch 64. The output is the model's forward output (384-d).
    - **DinoVdeau:** HF `lombardata/DinoVdeau-large-2024_04_03-with_data_aug_batch-size32_epochs150_freeze`. `Resize((518, 518))`, ImageNet normalization, batch 16, `pooler_output` (1024-d).

### Reference: numbers to check against

| What | Value | Source |
|---|---|---|
| Mask JSON files | 1,714 | WC 8, V2 6 |
| Masks, total | 44,211 | V2 10, CM 9 |
| After area/aspect/containment | 16,197 | V2 7 / V2 10 |
| After visibility and sharpness (kept) | 14,224 | V2 10 |
| Files with 0 masks after filtering | 4 | WC 12 |
| Mean filtered masks per file | 9.4 | WC 12 |
| `best` sample: masks before / after filter / crops saved | 2,337 / 1,054 / 842 | WC 14 |
| `best` taxonomy run: mean confidence; top predicted | 0.478; Acropora 116, Pavona 115, Stylophora 107 | WC 14 |
| Expert set: rows; identifiable / unidentifiable; correct; in-set | 78; 62 / 16; 39; 57 (from memory, verify) | memory |
| Photo size, sample | `IMG_4033.JPG` = 3648×2736 | verified 2026-10-11 |

### End-of-stage report (in chat)

- **What was built:** files, plus test count and status.
- **Deviations:** anything that differs from this spec, and why.
- **Found during refactor:** bugs or oddities you noticed but didn't fix.
- **What the user must do next:** exact cells or commands, and what to paste back.
- **Memory:** confirm that the session file and "Current understanding" were updated.
