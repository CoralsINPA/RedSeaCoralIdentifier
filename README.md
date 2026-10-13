# Red Sea Coral Identifier

Identify the genus of individual corals in Red Sea reef photos. The pipeline first cuts each coral out of a reef image (segmentation), then labels it with a genus (classification).

## Pipeline

The logic lives in the tested `coralid` package (`src/coralid/`, built from the original Colab notebooks; see [`docs/organization_plan.md`](docs/organization_plan.md)). Four thin Colab notebooks drive the pipeline, and the same jobs run from a terminal as `python -m coralid ...`. The original notebooks are kept as frozen reference copies in `notebooks/legacy/` until one end-to-end run of the new ones has been confirmed.

| Step | Notebook | What it does |
|---|---|---|
| 1. Segmentation and filters | `notebooks/01_segment_and_filter.ipynb` | Reports which photos still need a [CoralSCOP](https://github.com/zhengziqiang/CoralSCOP) mask JSON, checks the real CoralSCOP on a few photos against the saved masks, and loads the manifest (one row per mask, with the reason each rejected mask was dropped) and its filter funnel. |
| 2. Zero-shot classification | `notebooks/02_zero_shot.ipynb` | Classifies every kept crop into 20 Red Sea hard-coral genera with the fine-tuned BioCLIP (BioCLIP 2 and SigLIP 2 also supported), saves predictions and embeddings (the baseline), and reports on it: the expert test set, the genus mix against the survey, both by photo orientation. |
| 3. Expert review | `notebooks/03_expert_review.ipynb` | The portrait-impact report on the expert test set, and the blind review sheet of one genus (Millepora), drawn only from the 20-genus baseline, with its scoring (the sheet export is on hold). |
| 4. Supervised probe | `notebooks/04_supervised_probe.ipynb` | A linear probe on the baseline's BioCLIP embeddings, trained on expert labels and reported on held-out photos only. A template until the larger expert set exists. |

**Exploration (a separate direction, not part of the pipeline).** `notebooks/explore/embeddings.ipynb` embeds the kept crops with DINO (optional, GPU) and explores embeddings: t-SNE / UMAP maps colored by predicted genus, K-Means and HDBSCAN clusters, cluster galleries. It lives in the `coralid.explore` subpackage, which no pipeline module imports; see [`docs/exploration_embeddings.md`](docs/exploration_embeddings.md).

Quality filters applied to masks: minimum area 25,000 px², aspect ratio at most 3, containment threshold 0.80, Laplacian sharpness at least 80. Source images are resized to the mask frame, whose long side is 2560 (2560 x 1920 for landscape photos, 1920 x 2560 for the 221 portrait ones).

## Quickstart for the team (Colab)

1. **Get the data.** The shared Drive folder `CoralsMasking` must be in your My Drive (add a shortcut if it was shared with you), and Colab must be mounted with the Google account it is shared with.
2. **Open a notebook.** Click its "Open in Colab" badge, at the top of the notebook on GitHub. Use a **GPU runtime** (Runtime > Change runtime type > T4) for 01 (the segmentation check) and 02. 03 and 04 need no GPU.
3. **Run the first cell.** It clones this repository, installs `coralid` and mounts Drive. Its settings block (`DATA_ROOT`, `BRANCH`, `EXTRAS`) is the only place to edit. If it says to restart the runtime, do that and run the cell again.
4. **Run the rest in order.** Each notebook opens with a table of what every step reads, what it writes and how long it takes. Nothing is overwritten: a result that exists is loaded, a long job resumes where it stopped, and every report goes into a new folder, so re-running is safe.
5. **Go through the notebooks in this order: 01, 02, 03, and 04 when labels exist.**

| Notebook | Needs | Produces on Drive (under `results_all/iou60_sta55_pts32/`) |
|---|---|---|
| 01 segment and filter | the photos, the mask JSONs, the CoralSCOP checkpoint | on the active run nothing but a scratch run `results_all/segcheck_*/` and `reports/segcheck_<date>/`; `manifest/manifest.parquet` for a new run |
| 02 zero shot | the manifest, the photos and mask JSONs | `predictions/finetuned-bioclip__taxonomy_v1__hard20__all.parquet`, `features/finetuned-bioclip__masked.npz`, `reports/baseline_*_<date>/` |
| 03 expert review | the baseline, the expert set | `reports/portrait_impact_<date>/`; with the export on: `reviews/exports/` (hand out) and `reviews/keys/` (keep private); `reports/blind_score_*/` |
| 04 supervised probe (when labels exist) | the baseline's embeddings and predictions, a labels table | `reports/probe_*_<date>/` |

On the active run the baseline files already exist, so 02 loads them. For a new run, 02 writes them, and 04 needs them. Paste each notebook's printed tables back to whoever asked for the run.

There is **no valid Millepora precision figure** yet. An earlier check, made on the predictions of a pipeline that is no longer used, is discarded; the blind review sheet (notebook 03) redoes it on the 20-genus baseline.

## Repository layout

```
src/coralid/          Installable package: segmentation (CoralSCOP) and its check, masks, crops, filters, manifest and filter funnel, zero-shot scoring and the 20-genus baseline job, a linear probe, expert-set evaluation, review sheets, plots, the Colab setup, and the `python -m coralid` command line. The subpackage `explore/` holds the exploration direction (DINO embeddings, clustering) that the pipeline does not use
configs/              Run configuration (iou60_sta55_pts32.yaml) and the genus label set and prompts (genera.yaml)
tests/                CPU-only tests on synthetic data (no data or GPU needed)
notebooks/01_segment_and_filter.ipynb         Colab: segmentation status, the segmentation check, the manifest and its filter funnel
notebooks/02_zero_shot.ipynb                  Colab: the baseline job and its reports (expert set, genus mix, by photo orientation)
notebooks/03_expert_review.ipynb              Colab: portrait impact and the Millepora blind review (sheets on hold)
notebooks/04_supervised_probe.ipynb           Colab: a linear probe on the baseline's embeddings, held-out report only
notebooks/explore/embeddings.ipynb            Colab: DINO embeddings and the exploration of embeddings (not part of the pipeline)
notebooks/parity_check.ipynb                  Colab: checks the package against the legacy notebooks (a one-time check)
notebooks/legacy/     The original Colab notebooks, frozen as reference (to be deleted after one confirmed end-to-end run)
third_party/CoralSCOP Submodule: upstream CoralSCOP segmentation model
docs/                 Onboarding guide, its errata, the organization plan, algorithm ideas, agent memory
```

## Getting started

```bash
git clone --recurse-submodules https://github.com/CoralsINPA/RedSeaCoralIdentifier
# already cloned without submodules?
git submodule update --init
```

To work on the package (Python 3.10 or newer):

```bash
pip install -e ".[dev]"      # add extras as needed: clip, segment, probe, viz (dino, cluster: exploration only)
ruff check . && ruff format --check . && pytest -q
```

Set `CORALID_DATA_ROOT` to the data folder (the Drive's `CoralsMasking` folder) before running anything that reads data.

Everything the notebooks do also runs from a terminal. For example the 20-genus baseline (fine-tuned BioCLIP over every kept crop of the run's manifest), on a machine with a GPU:

```bash
python -m coralid zero-shot --config configs/iou60_sta55_pts32.yaml
```

It is resumable, never overwrites a file, and takes about 25 minutes for the 1,714 photos on a T4. The other commands (`segment`, `build-manifest`, `evaluate`, `export-review`, `train-probe`, and `explore-embed` for the exploration direction) work the same way: see `python -m coralid --help`. `notebooks/` holds the four pipeline notebooks, `explore/embeddings.ipynb` and the one-time `parity_check.ipynb`; outputs are stripped before every commit (pre-commit with nbstripout).

## Data and model weights (not in this repo)

The images, mask JSONs, embeddings, label CSVs and the CoralSCOP checkpoint are stored in a shared Google Drive (access is by invitation). See [`docs/Coral_Pipeline_Onboarding_Guide.pdf`](docs/Coral_Pipeline_Onboarding_Guide.pdf) for the pipeline walkthrough, and [`docs/onboarding_errata.md`](docs/onboarding_errata.md) for where the guide disagrees with the code and the data.

Root folder: [CoralsMasking](https://drive.google.com/drive/folders/1LtqWRm2vieTbUGgb1ISfsqVD-g-urPRY)

### Start here

| What | Link |
|---|---|
| Active run folder (`results_all/iou60_sta55_pts32`): masks, crops, labels, embeddings | [folder](https://drive.google.com/drive/folders/1up9-VQJEPyv5p8YbPCPgmO5yVx-SCz3r) |
| Expert test set (77 hand-labeled crops) | [`assaf_expert_test_set.csv`](https://drive.google.com/file/d/1PIBGXIduvOnRn3lvsTT2ZyFtjKiUVRRD/view) |
| Ground-truth genus counts from the expert survey | [`ground_truth_genus_counts.csv`](https://drive.google.com/file/d/1DbQaZa-3lTehQPP-7PmjAkiCkbvuadwd/view) |
| Zero-shot labels for the 50-densest-image sample, 842 crops (fine-tuned BioCLIP, taxonomy prompts, masked). "best" names the sample, not the best configuration | [`coral_weak_labels_finetuned-bioclip_taxonomy_masked_best.csv`](https://drive.google.com/file/d/1oB15CLt1FEQfbp4Xai1xlWs-hsiJRYqC/view) |
| CoralSCOP checkpoint | [`vit_b_coralscop.pth`](https://drive.google.com/file/d/1tiN32L4h4A5cIZAf1jeP9q8Euf-9U0Eo/view) (382 MB) |
| Filtered mask JSONs | [`cleaned_jsons_filtered`](https://drive.google.com/drive/folders/1ATUrW-DLZMxGYio6N-9bqVuonFKkQZK4) |
| Onboarding guide (Drive copy) | [PDF](https://drive.google.com/file/d/1gr2tnBhclsnh8s2oM49YKLWSYD9-VyGg/view) |

Colab notebooks: [CoralMasking](https://colab.research.google.com/drive/1sS9aUIWFBk6EP54biw_vGc3n3oyw51H7), [WeakClassification](https://colab.research.google.com/drive/1hCcbbjEHMtZQEm0TXiYwU0q-E_ZnIX4K).

### Inside the active run folder

| Item | Contents |
|---|---|
| `json/` | Mask JSONs (contents not inspected) |
| `cleaned_images_crops/` | Cleaned crops (contents not inspected) |
| `cluster_previews/`, `visualizations/` | Embedding and cluster plots |
| `coral_dinoV2_encodings.csv` (181 MB), `coral_dinoV2_encodings_MASKS.csv` (489 MB) | DINOv2 embeddings, whole crop and masked crop |
| `v2_pipeline/` | Not used. The outputs of a pipeline that has been excluded; nothing reads them. |

### Other folders (lower priority)

| Folder | Contents |
|---|---|
| [`data_all`](https://drive.google.com/drive/folders/1oj6WQ9jNW6vUGe18PyvT5eniaASBGh68) | Full set of source reef photos |
| [`data_50`](https://drive.google.com/drive/folders/1s7ZONaUeHH0BAhnH0UBH8l67X8HrBUq5) | 50-image subset for quick tests |
| [`results_50`](https://drive.google.com/drive/folders/11ogV1vfd9m-28fq3fHNiZjws_GkBwYoN) | Early SAM parameter sweeps on the 50-image subset (iou55, iou60, iou65, iou72). Superseded by the active run |
| [`CoralSCOP`](https://drive.google.com/drive/folders/1rTIFnYg52cQ0jsMd4gHYOp_o69Vg5_rV) | Copy of the upstream repo plus the checkpoint. Prefer the `third_party/CoralSCOP` submodule |

`external_libs/` holds vendored pip packages and `cropped_masks/` was empty when checked. Ignore both.

The legacy notebooks copy-paste the quality-filter helpers; the package keeps one shared copy in `coralid.filters`.
