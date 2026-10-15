# CLAUDE.md

Genus-level coral identification for Red Sea reef photos: segment corals (CoralSCOP/SAM), then classify each crop with zero-shot CLIP-family models. See `README.md` for the overview and `ONBOARDING.md` for the newcomer guide. The original PDF guide is kept in `docs/legacy/` (outdated; `docs/onboarding_errata.md` lists where).

## Layout
- `src/coralid/`: the installable package that is replacing the notebooks (plan: `docs/organization_plan.md`, staged prompts: `docs/agent_prompts/implement_organization_plan.md`). The **pipeline** is segment -> manifest (filters) -> zero-shot baseline -> evaluation and expert review -> linear probe. Embeddings and clustering are a **separate exploration direction** (`coralid.explore`); no pipeline module imports it. Pipeline modules:
  - data path: `config`, `masks`, `images`, `crops`, `filters` (area, aspect, containment, sharpness; `sharpness_crop` is `masked` or `bbox`), `manifest`, `orientation` (landscape / portrait from the manifest's frame, and `by_orientation` to split any metric), `funnel` (the masks left after each filter, by orientation);
  - segmentation: `segmentation` (CoralSCOP from the `third_party` submodule; `run_segmentation` creates only the missing mask JSONs and never rewrites one, errors go to a failure log, never to an empty JSON) `parity_segmentation` (the segmentation check: segments a few photos into a scratch run, `paths.run_prefix` = `segcheck_`) and `mask_compare` (compares two folders of mask JSONs by IoU, with a PASS / WARN / FAIL verdict);
  - zero-shot path: `genera`, `encoders` (the CLIP-family models only), `zero_shot`, `batch` (resumable runner), `baseline` (the 20-genus job: `run_baseline`, which also saves the BioCLIP embeddings the probe uses);
  - scoring and reports: `evaluation` and `predictions` (expert set, legacy CSV adapter, prediction comparison), `reports` (expert evaluation and genus mix by orientation, probe report), `portrait_impact`, and `parity`, `parity_manifest`, `parity_expert`, `parity_report` (the pass/fail logic of the parity notebook);
  - review: `labeling` (review sheets, ingest by key), `genus_audit` (the blind review sheet, drawn from the 20-genus baseline), `review_source` (checks that the sheet is drawn from that baseline and that scoring uses the same one), `review_scoring`, `review_export`;
  - supervised learning: `probe` (logistic regression on stored embeddings, grouped cross-validation, held-out report; the labels are an input, it never makes them), `probe_labels` (probe labels from the expert set: in-set rows on landscape photos only);
  - plots: `viz` and `viz_plots` (crop-grid and gallery PDFs, mask overlays, confusion heat maps, the funnel chart; each takes a table plus an `AnnotationStore` and computes nothing);
  - the command line and Colab: `jobs` (the config wiring behind the commands), `cli` (`python -m coralid ...`), `colab` (the only module that knows about Colab: `setup()` and the `BOOTSTRAP_CELL` that every notebook starts with).
  - Exploration, outside the pipeline (`coralid.explore`, extras `dino` and `cluster`; see `docs/exploration_embeddings.md`): `dino` (the DINOv2 and DinoVdeau encoders), `embed` (`run_embeddings`: DINO embeddings of the kept masks, the `explore-embed` command), `clustering` (PCA, t-SNE, UMAP, K-Means, HDBSCAN with the legacy settings), `plots` (2-D maps colored by a number or a category), `analysis` (a features file joined to the baseline's predicted genus and the manifest's orientation, cluster summaries, the saved exploration report). `tests/test_structure.py` fails if a pipeline module imports it (only `cli.py` may, inside the `explore-embed` command).
  - Tests are in `tests/` (synthetic data only). Configs: `configs/iou60_sta55_pts32.yaml` (the run) and `configs/genera.yaml` (the `hard20` label set and the prompt sets, copied verbatim from the legacy notebooks).
- `notebooks/legacy/`: the frozen reference notebooks, outputs included. Kept as history (the end-to-end run of the new notebooks was confirmed 2026-10-14):
  - `CoralMasking.ipynb`: segmentation + DINOv2/DinoVdeau embeddings + t-SNE/UMAP/K-Means.
  - `weak_classification.ipynb`: zero-shot genus labels (20 hard-coral genera). The "breakthrough" notebook. Cells 25-34 are scratch/debug.
  - `Millepora_Analysis.ipynb`: the old Millepora audit. It analyzed V2 output and is kept only as history; nothing in `coralid` uses it.
- `notebooks/parity_check.ipynb`: the Colab notebook that checks `coralid` against the legacy notebooks and the recorded numbers (Stage A2; made orientation-aware in Stage B). It reads Drive and creates only new files. Its summary cell prints what to paste back to the next session.
- `notebooks/01_...` to `04_...`: the four pipeline notebooks. Each is thin (library calls, tables and plots; under 15 cells; no function definitions), starts with `coralid.colab.BOOTSTRAP_CELL` (the settings block with `EXTRAS` is the only edit), reads Drive and creates only new files; `tests/test_notebooks.py` holds them to this.
  - `01_segment_and_filter`: which photos lack a mask JSON (creates them only with `SEGMENT_MISSING = True`), the segmentation check on 5 photos from `data_50` into the scratch run `results_all/segcheck_iou60_sta55_pts32/`, the saved manifest and its filter funnel by orientation. Nothing destructive on the active run.
  - `02_zero_shot`: the baseline job (about 25 min on a T4) and its reports.
  - `03_expert_review`: the portrait-impact report on the expert set, the blind review sheet export (**on hold**, `RUN_EXPORT = False`; it prints which predictions it draws from) and its scoring.
  - `04_supervised_probe`: the probe on the expert set's 51 landscape rows (a template: too small to mean anything until the larger expert set of `docs/algorithm_ideas.md` idea 1.1 exists, set `LABELS_CSV`).
- `notebooks/05_millepora.ipynb`: the Millepora analysis on the 20-genus baseline (it checks the source with `review_source.baseline_source` and stops otherwise): counts by orientation and confidence band, runner-ups, a confidence histogram, and crop PDFs of the predictions per band and of the likely misses (Millepora as runner-up), into a new `reports/millepora_<date>/`. It replaces `legacy/Millepora_Analysis.ipynb`; precision still needs the blind review of notebook 03.
- `notebooks/explore/embeddings.ipynb`: the exploration notebook, outside the pipeline: the optional DINO job (`RUN_EMBED`, GPU), then maps, K-Means, HDBSCAN and a cluster gallery colored by the baseline's predicted genus; `FEATURES` (first setting) is `finetuned-bioclip__masked` by default, so it runs on a CPU.
- `third_party/CoralSCOP`: git submodule (upstream zhengziqiang/CoralSCOP, pinned).
- `docs/onboarding_errata.md`: where the PDF guide disagrees with the code and the data.

## Commands
- Install: `pip install -e ".[dev]"` (extras: `clip`, `segment`, `probe`, `viz`; `dino` and `cluster` are for the exploration direction only).
- Command line, every command with `--config configs/iou60_sta55_pts32.yaml` and `CORALID_DATA_ROOT` (or `--data-root`). None overwrites a file; the long jobs resume. Exit code 2 means it could not finish (the message says why):
  - `python -m coralid segment`: create the mask JSONs of photos that have none (CoralSCOP checkpoint and a GPU). An existing JSON is never touched, so it is safe on the active run; `--dry-run` only counts.
  - `python -m coralid build-manifest`: `manifest/manifest.parquet` (reads every photo; resumable). `--geometric` needs no photos.
  - `python -m coralid zero-shot`: the 20-genus baseline job (GPU, needs the manifest). About 25 minutes on a T4.
  - `python -m coralid evaluate`: the baseline against the expert set and the survey counts, into a new folder under `reports/`.
  - `python -m coralid export-review --genus Millepora`: a blind review sheet drawn only from the 20-genus baseline (it stops unless the baseline's meta file says label set `hard20` and the config's model and prompt set; the source is recorded in the key's meta file); hand out `reviews/exports/`, keep `reviews/keys/` private.
  - `python -m coralid train-probe --labels <csv>`: held-out report of a linear probe, trained only on labels you give it.
  - `python -m coralid explore-embed [--model dinov2-s14|dinovdeau-l]` (exploration, not part of the pipeline): `features/<model>__masked_square.npz` for the kept masks (GPU).
- Check: `ruff check .`, `ruff format --check .`, `pytest -q`. All three must pass before a commit. `notebooks/legacy/` and `third_party/` are excluded from ruff, and from nbstripout (their outputs stay).
- Data root: set `CORALID_DATA_ROOT` (or `paths.data_root` in the config). Only `colab.py` may know about Colab: `coralid.colab.setup()` mounts Drive, clones and installs the package; notebooks 01-04 and the exploration notebook start with its `BOOTSTRAP_CELL`, which brings an existing clone to the branch head before importing from it (the parity notebook still carries its own setup cell).
- Segmentation needs the vendored CoralSCOP, never pip's `segment-anything` (no coral head): `git submodule update --init`, and the `segment` extra.

## Key facts
- Active run name: `iou60_sta55_pts32` (SAM points_per_side 32, pred_iou 0.60, stability 0.55, min_mask_region_area 100).
- Filters: MIN_AREA 25,000, MAX_ASPECT_RATIO 3.0, CONTAINMENT_THRESH 0.80, SHARPNESS_THRESHOLD 80.
- Mask frame: the long side is 2560. 1,493 photos are 2560x1920 (landscape) and 221 are 1920x2560 (portrait). Always load photos in the JSON's frame (`coralid.images.load_in_mask_frame`). The legacy code forced 2560x1920, which mis-cropped every portrait photo.
- Expert test set results: the CSV has 77 rows; v1 accuracy on it is 36/54 = 66.7% in-set. The 20-genus baseline (run 2026-10-12, the reference from now on) gets 34/51 = 66.7% on the 67 landscape rows, identical to v1. The 10 portrait rows can't score it (Assaf labeled mis-cut crops), and 7 of them are among the 16 "unidentifiable" rows.
- Default model: ReefNet fine-tuned BioCLIP (ViT-B/16); alternatives are BioCLIP 2 and SigLIP 2.
- Millepora: no valid precision figure exists. An earlier check on the excluded V2 pipeline's predictions is discarded; the blind review sheet (notebook 03) redoes it on the 20-genus baseline.
- V2 is excluded (decided 2026-10-11, enforced 2026-10-13): the 24-genus pipeline, its outputs in Drive's `v2_pipeline/` and the Millepora audit made on them are not a reference for anything. The code and the current docs refer only to `coralid` and, for provenance, to `weak_classification.ipynb` and `CoralMasking.ipynb`; `tests/test_no_v2.py` enforces it.
- Expert test set: `assaf_expert_test_set.csv`, joined on `(image_name, coral_id)`. The CSV has 77 rows (the 78-crop matrix in the memory notes is a slightly different version of the same review). Orientation matters: the expert labeled crops cut by the legacy code, which were mis-cut on portrait photos, so report expert results for landscape and portrait rows separately.
- Outputs the notebooks create under the run folder on Drive: `manifest/manifest.parquet` (16,080 kept crops, from the parity notebook), `predictions/<model>__<prompts>__<labels>__all.parquet` and `features/<model>__<crop mode>.npz` (the baseline, each with a `.meta.json`; progress in `predictions/parts/`), `reports/` (one new folder per report, probe reports included), `reviews/{exports,keys,completed}/` (blind review sheets; `keys/` holds the hidden answer key and its meta file with the predictions' source, do not share it). The new jobs add `segmentation/` (the `segment` failure log, run records and optional extras; `json/` holds mask JSONs only) and `manifest/manifest_geometric.parquet` (`build-manifest --geometric`). Notebook 01's segmentation check adds a scratch run `results_all/segcheck_iou60_sta55_pts32/` (mask JSONs of 5 photos; never the active run's `json/`) and `reports/segcheck_<date>/`. Notebooks 01 and 02 save every picture they draw into a new `reports/figures_01_<date>/` or `reports/figures_02_<date>/` (`reports.save_figure`, never overwrites). The exploration direction adds `features/<dino model>__masked_square.npz` (`explore-embed`, with progress in `features/parts/`) and, from `notebooks/explore/embeddings.ipynb`, `reports/embeddings_<features>_<date>/`.
- Label set: the 20 hard-coral genera (decided 2026-10-11). The 24-genus pipeline of the excluded V2 is not carried forward.

## Agent memory
`docs/agent_memory/` holds project history, expert-review results and conclusions from earlier sessions. Read its `README.md` before analysis or planning work. At the end of a session, add `sessions/YYYY-MM-DD_<topic>.md` and update "Current understanding" there. The repo is public, so don't put anything private in it.

Algorithm ideas (evidence, cost and priority for each) are collected in `docs/algorithm_ideas.md`. Start algorithm planning there.

## Not in the repo
Images, mask JSONs, embeddings, label CSVs, the CoralSCOP checkpoint, `ground_truth_genus_counts.csv` and the expert test set live in a shared Google Drive. Use the Google Drive MCP tools if connected (search by `parentId`), otherwise ask the user for any file you need.

## Accessing code and data
Code:
1. The legacy notebooks are in `notebooks/legacy/`. Read them there first. `third_party/CoralSCOP` needs `git submodule update --init` if it is empty.
2. The working copies run in Colab on the Drive (links under "Drive map"). They can be newer than the repo, so compare modified times before trusting a repo copy. If a notebook, helper or cell is missing from the repo, fetch the Colab file from Drive.
3. Don't open the Colab links with WebFetch; they need a login. Use the Drive tools instead.

Data and other files not in the repo:
1. Look up the ID in the "Drive map" below. If it isn't listed, search with `mcp__Google_Drive__search_files` (e.g. `title contains 'name'`, or `parentId = '<folder id>'` to list a folder).
2. Small text files (CSVs under a few MB, notebooks, PDFs) can be read with `mcp__Google_Drive__read_file_content`. For anything larger, use `mcp__Google_Drive__download_file_content`.
3. The embedding CSVs and `.npz` files are too big to read through the tool, and images and the checkpoint are binary. Ask the user to upload them or run the step in Colab.
4. If the Drive tools return "Forbidden", the connector needs reconnecting in the user's claude.ai connector settings. Many files are owned by the team account (`corals_inpa@menasim.org`) or by a collaborator's personal account, and are only visible if shared with the user's account.
5. If a file can't be found at all, ask the user for it. Don't recreate it.

## Drive map
Root `CoralsMasking`: https://drive.google.com/drive/folders/1LtqWRm2vieTbUGgb1ISfsqVD-g-urPRY
Folder IDs below are for `parentId = '...'` queries.

Relevant:
- Active run `results_all/iou60_sta55_pts32` (id `1up9-VQJEPyv5p8YbPCPgmO5yVx-SCz3r`). Contains `json/`, `cleaned_images_crops/`, `cluster_previews/`, `visualizations/` (subfolder contents not inspected), `coral_dinoV2_encodings.csv` (181 MB) and `coral_dinoV2_encodings_MASKS.csv` (489 MB).
  - **Don't use the DINO CSVs.** Their provenance is unclear:
    - `coral_dinoV2_encodings.csv` is most likely the bbox-crop file, cut from the wrong part of each photo;
    - `coral_dinoV2_encodings_MASKS.csv` and `coral_dinoV3_encodings_MASKS.csv` have exactly the same size (489,150,549 bytes), probably copies;
    - `coral_dinoV3_encodings.csv` (1024-d) was made by a notebook not in the repo.
    
    They're kept on Drive but nothing uses them; regenerate features with `coralid` if needed.
- `assaf_expert_test_set.csv` (id `1PIBGXIduvOnRn3lvsTT2ZyFtjKiUVRRD`), in the active run folder. Columns: `image_name, coral_id, bbox, area`, 20 genus probabilities, `predicted_genus, confidence, expert_genus, is_correct, review_source`. `expert_genus` can be a soft coral (Cladiella, Sinularia, Litophyton) that is outside the 20 target genera. For crops with no genus it is empty and `is_correct` holds the word `unidentifiable` or `MISSING_LABEL` (`load_expert_set` reads it from there). Exclude these when computing accuracy.
- `ground_truth_genus_counts.csv` (id `1DbQaZa-3lTehQPP-7PmjAkiCkbvuadwd`), in the root. Columns: `Category, genus, count, Pi`. It includes soft corals and genera outside the 20 targets, so filter to the 20 before comparing.
- `coral_weak_labels_finetuned-bioclip_taxonomy_masked_best.csv` (id `1oB15CLt1FEQfbp4Xai1xlWs-hsiJRYqC`): zero-shot labels for the **50-densest-image sample** (842 crops). "best" is the sample mode (`SAMPLE_MODE = "best"`), not the best configuration.
- Checkpoint: `CoralSCOP/checkpoints/vit_b_coralscop.pth` (id `1tiN32L4h4A5cIZAf1jeP9q8Euf-9U0Eo`, 382 MB).
- `cleaned_jsons_filtered` (id `1ATUrW-DLZMxGYio6N-9bqVuonFKkQZK4`): filtered mask JSONs.
- `data_all` (id `1oj6WQ9jNW6vUGe18PyvT5eniaASBGh68`): all source photos. `data_50` (id `1s7ZONaUeHH0BAhnH0UBH8l67X8HrBUq5`): 50-image subset.
- Colab: CoralMasking https://colab.research.google.com/drive/1sS9aUIWFBk6EP54biw_vGc3n3oyw51H7, WeakClassification https://colab.research.google.com/drive/1hCcbbjEHMtZQEm0TXiYwU0q-E_ZnIX4K, Millepora_Analysis https://colab.research.google.com/drive/18TE7K8VihAIzcgaQ1ULCMBTlHpqtPACS.

Archive (2026-10-14): outdated files were moved, not deleted, into `results_all/iou60_sta55_pts32/archive/` (`embedding_csvs_old/`, `zero_shot_experiments_old/`, `clustering_old/`, `defective_wrong_crops/` = `cleaned_images_crops`, `excluded_v2_pipeline/` = `v2_pipeline`) and the root `archive/` (`cleaned_jsons_filtered`). Each has a `README_archive.txt`. Items owned by the original personal account could not be moved: the two `coral_dinoV2_encodings*.csv`, `cluster_summary*`, `coral_results.csv` and `visualizations/` in the run folder; `results_50/`, `cropped_masks/`, `coral_dino_encodings.csv` and `Coral_Segmentation_Comparison.jpg` in the root. That account also owns `data_all/`, `data_50/`, `results_all/` and `json/`; transferring ownership to the team account is advised.

Ignore:
- `v2_pipeline/` (id `1pJq-dvipNiprBiQRXLAqH_iXTbEcjuNG`) in the active run folder: the excluded V2 pipeline's outputs (its predictions, embeddings, review files and the old Millepora audit). Nothing reads them; do not use them as a reference.
- `external_libs/`: vendored pip packages (tqdm, pyparsing, tabulate, ...).
- `CoralSCOP/` in Drive (including `.git`, `segment_anything`, `figs`, `demo_imgs`): duplicate of the submodule. Only the checkpoint is needed.
- `results_50/`: early parameter sweeps on 50 images, superseded by `iou60_sta55_pts32`.
- `cropped_masks/`: empty when checked.
- Root-level `coral_dino_encodings.csv` (1.2 MB) and the 2026-01/03 `coral_dino_*` CSVs elsewhere in My Drive: older, smaller runs.
- The embedding CSVs are 180-490 MB. Don't read them through the Drive tool, since downloading is the only practical way.

Notes: the Drive onboarding guide calls the notebook `WeakClassification.ipynb`; in the repo it is `weak_classification.ipynb` (in `notebooks/legacy/`).
