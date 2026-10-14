# Red Sea Coral Identifier

Identify the genus of individual corals in Red Sea reef photos. The pipeline cuts each coral out of a reef photo (segmentation), drops bad cut-outs (filters), names its genus (classification) and checks the result against an expert (evaluation).

**New here? Start with [`ONBOARDING.md`](ONBOARDING.md).** It assumes no computer-vision background and walks you from Drive access to your first notebook run.

## Status (October 2026)

- **The original Colab notebooks are now one tested Python package,** `coralid` (`src/coralid/`), driven by thin Colab notebooks and a command line. It was checked end to end on the real data (2026-10-14):
  - segmentation reproduces the original masks exactly;
  - the filters keep the same 16,080 crops from 1,714 photos;
  - the baseline reproduces the expert-set results.
- **A bug that mis-cut all 221 portrait photos (13%) is fixed.** Every crop is now cut from the right place.
- **Baseline:** zero-shot fine-tuned BioCLIP, 20 hard-coral genera, all 16,080 crops. On the expert test set it scores **34/51 = 66.7%** (landscape photos).
- **Clustering with an independent reef model (DinoVdeau) found the main error sources:**
  - bare substrate is predicted as "Pavona" (34× its survey share);
  - giant clams are predicted as "Lobophyllia";
  - brain corals (*Platygyra*) are under-called.

  See [`docs/clustering_results.md`](docs/clustering_results.md).
- **Next** ([`docs/algorithm_ideas.md`](docs/algorithm_ideas.md), "Proposed plan"):
  1. expert-confirmed clusters;
  2. a gate that rejects non-coral;
  3. a linear probe for the genus;
  4. a flag where the models disagree;
  5. a blind expert test set of about 300 crops.
- **Millepora:** there is no valid precision figure yet. An earlier check was made on a pipeline that is no longer used; `notebooks/03_expert_review.ipynb` redoes it blind on the baseline.

## Notebooks

Each notebook runs in Google Colab. Its first cell mounts Drive, downloads this repository and installs `coralid`. Notebooks only read the shared data and write new, dated outputs; they never overwrite anything.

| Notebook | What it does | Runtime |
|---|---|---|
| [01 segment and filter](https://colab.research.google.com/github/CoralsINPA/RedSeaCoralIdentifier/blob/main/notebooks/01_segment_and_filter.ipynb) | Which photos lack masks; a check of the real [CoralSCOP](https://github.com/zhengziqiang/CoralSCOP) segmentation on 5 photos; the manifest (one row per mask, with the reason each rejected mask was dropped) and its filter funnel | T4 GPU |
| [02 zero-shot](https://colab.research.google.com/github/CoralsINPA/RedSeaCoralIdentifier/blob/main/notebooks/02_zero_shot.ipynb) | The 20-genus baseline (predictions and BioCLIP embeddings of every crop) and its reports: the expert set, the genus mix against the survey, both by photo orientation | T4 GPU |
| [03 expert review](https://colab.research.google.com/github/CoralsINPA/RedSeaCoralIdentifier/blob/main/notebooks/03_expert_review.ipynb) | The portrait-impact report on the expert set; the blind review sheet (on hold) and its scoring | any |
| [04 supervised probe](https://colab.research.google.com/github/CoralsINPA/RedSeaCoralIdentifier/blob/main/notebooks/04_supervised_probe.ipynb) | A linear probe on the baseline's embeddings, held-out report only (a template until more labels exist) | any |
| [05 Millepora](https://colab.research.google.com/github/CoralsINPA/RedSeaCoralIdentifier/blob/main/notebooks/05_millepora.ipynb) | One genus up close on the baseline: counts, confidence bands, crop PDFs, likely misses | any |
| [explore/embeddings](https://colab.research.google.com/github/CoralsINPA/RedSeaCoralIdentifier/blob/main/notebooks/explore/embeddings.ipynb) | **Exploration, separate from the pipeline:** DINO embeddings, t-SNE/UMAP maps, K-Means and HDBSCAN clusters, sample crops of chosen clusters | any (GPU to compute new embeddings) |

`notebooks/parity_check.ipynb` is the one-time check of the package against the original notebooks. `notebooks/legacy/` keeps the original notebooks as history.

## Repository layout

```
ONBOARDING.md         start here
src/coralid/          the package: segmentation, filters and manifest, zero-shot baseline, evaluation,
                      review sheets, linear probe, plots, Colab setup, command line (python -m coralid)
  explore/            the exploration direction (DINO embeddings, clustering); the pipeline never imports it
configs/              iou60_sta55_pts32.yaml (all run settings), genera.yaml (the 20 genera and prompts)
notebooks/            the notebooks above; legacy/ = the original notebooks
tests/                automatic checks on synthetic data (no Drive or GPU needed)
third_party/CoralSCOP the segmentation model (git submodule)
docs/                 algorithm_ideas.md, clustering_results.md, exploration_embeddings.md,
                      organization_plan.md, onboarding_errata.md, agent_memory/ (project history),
                      legacy/ (the original PDF onboarding guide)
CLAUDE.md             a dense technical map of the code and the Drive (written for AI assistants)
```

## Working on the code

```bash
git clone --recurse-submodules https://github.com/CoralsINPA/RedSeaCoralIdentifier
cd RedSeaCoralIdentifier
pip install -e ".[dev]"      # extras: clip, segment, probe, viz; dino and cluster for exploration
ruff check . && ruff format --check . && pytest -q    # all three must pass before every commit
```

Work on a branch and open a pull request; never push to `main` directly. Notebooks stay thin: logic goes into `src/coralid/` with a test, and outputs are stripped before committing.

Every notebook job also runs from a terminal with a GPU and the data (set `CORALID_DATA_ROOT` to the Drive's `CoralsMasking` folder): `python -m coralid --help`. For example, `python -m coralid zero-shot --config configs/iou60_sta55_pts32.yaml`. Long jobs resume, and nothing is ever overwritten.

## Data (not in this repository)

Photos, masks, results and model weights live in a shared Google Drive (by invitation): [CoralsMasking](https://drive.google.com/drive/folders/1LtqWRm2vieTbUGgb1ISfsqVD-g-urPRY).

| What | Where |
|---|---|
| All 1,714 photos / a 50-photo subset | [`data_all/`](https://drive.google.com/drive/folders/1oj6WQ9jNW6vUGe18PyvT5eniaASBGh68) / [`data_50/`](https://drive.google.com/drive/folders/1s7ZONaUeHH0BAhnH0UBH8l67X8HrBUq5) |
| The active run | [`results_all/iou60_sta55_pts32/`](https://drive.google.com/drive/folders/1up9-VQJEPyv5p8YbPCPgmO5yVx-SCz3r): `json/` (masks), `manifest/`, `predictions/`, `features/`, `reports/` (one dated folder per report), `reviews/` |
| Expert test set (77 crops) | [`assaf_expert_test_set.csv`](https://drive.google.com/file/d/1PIBGXIduvOnRn3lvsTT2ZyFtjKiUVRRD/view), in the active run |
| Survey counts per genus | [`ground_truth_genus_counts.csv`](https://drive.google.com/file/d/1DbQaZa-3lTehQPP-7PmjAkiCkbvuadwd/view) |
| CoralSCOP checkpoint | [`vit_b_coralscop.pth`](https://drive.google.com/file/d/1tiN32L4h4A5cIZAf1jeP9q8Euf-9U0Eo/view) (382 MB) |
| Outdated files (kept, not used) | `archive/` folders in the root and in the active run, each with a `README_archive.txt` |
