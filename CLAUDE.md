# CLAUDE.md

Genus-level coral identification for Red Sea reef photos: segment corals (CoralSCOP/SAM), then classify each crop with zero-shot CLIP-family models. See `README.md` for the overview and `docs/Coral_Pipeline_Onboarding_Guide.pdf` for the full Drive/notebook map.

## Layout
- `src/coralid/`: the installable package that is replacing the notebooks (plan: `docs/organization_plan.md`, staged prompts: `docs/agent_prompts/implement_organization_plan.md`). Built so far: `config`, `masks`, `images`, `crops`, `filters`, `manifest`. Tests are in `tests/` (synthetic data only), the run config in `configs/iou60_sta55_pts32.yaml`.
- `notebooks/legacy/`: the frozen reference notebooks, outputs included. The three Colab notebooks and V2 are here:
  - `CoralMasking.ipynb`: segmentation + DINOv2/DinoVdeau embeddings + t-SNE/UMAP/K-Means.
  - `weak_classification.ipynb`: zero-shot genus labels (20 hard-coral genera). The "breakthrough" notebook. Cells 25-34 are scratch/debug.
  - `Millepora_Analysis.ipynb`: Millepora precision audit.
  - `CoralClassification_v2_Pipeline.ipynb`: V2 (24 genera, tiers, embeddings saved). It produced the 14,224-crop run and the Millepora audit inputs.
- `third_party/CoralSCOP`: git submodule (upstream zhengziqiang/CoralSCOP, pinned).
- `docs/onboarding_errata.md`: where the PDF guide disagrees with the code and the data.

## Commands
- Install: `pip install -e ".[dev]"` (extras: `clip`, `dino`, `segment`, `cluster`, `viz`).
- Check: `ruff check .`, `ruff format --check .`, `pytest -q`. All three must pass before a commit. `notebooks/legacy/` and `third_party/` are excluded from ruff, and from nbstripout (their outputs stay).
- Data root: set `CORALID_DATA_ROOT` (or `paths.data_root` in the config). Only `colab.py` (not written yet) may know about Colab.

## Key facts
- Active run name: `iou60_sta55_pts32` (SAM points_per_side 32, pred_iou 0.60, stability 0.55, min_mask_region_area 100).
- Filters: MIN_AREA 25,000, MAX_ASPECT_RATIO 3.0, CONTAINMENT_THRESH 0.80, SHARPNESS_THRESHOLD 80. Images are resized to 2560x1920.
- Default model: ReefNet fine-tuned BioCLIP (ViT-B/16); alternatives are BioCLIP 2 and SigLIP 2.
- Millepora check: a non-expert analyst (not Assaf) judged 66 of 112 decided crops correct (58.9%; 51-64% counting the 17 unsure). The crops were V2's sharp, confidence >= 0.85 Millepora predictions, and the reviewer saw the model's answer. Treat the figure as provisional, and not as the 20-genus precision.
- Expert test set: 78 hand-labeled crops (`assaf_expert_test_set.csv`), joined on `(image_name, coral_id)`.
- Label set: the 20 hard-coral genera (decided 2026-10-11). The 24-genus v2 pipeline is not carried forward.

## Agent memory
`docs/agent_memory/` holds project history, expert-review results and conclusions from earlier sessions. Read its `README.md` before analysis or planning work. At the end of a session, add `sessions/YYYY-MM-DD_<topic>.md` and update "Current understanding" there. The repo is public, so don't put anything private in it.

## Not in the repo
Images, mask JSONs, embeddings, label CSVs, the CoralSCOP checkpoint, `ground_truth_genus_counts.csv` and the expert test set live in a shared Google Drive. Use the Google Drive MCP tools if connected (search by `parentId`), otherwise ask the user for any file you need.

## Accessing code and data
Code:
1. The three notebooks and V2 are in `notebooks/legacy/`. Read them there first. `third_party/CoralSCOP` needs `git submodule update --init` if it is empty.
2. The working copies run in Colab on the Drive (links under "Drive map"). They can be newer than the repo, so compare modified times before trusting a repo copy. If a notebook, helper or cell is missing from the repo, fetch the Colab file from Drive.
3. Don't open the Colab links with WebFetch; they need a login. Use the Drive tools instead.

Data and other files not in the repo:
1. Look up the ID in the "Drive map" below. If it isn't listed, search with `mcp__Google_Drive__search_files` (e.g. `title contains 'name'`, or `parentId = '<folder id>'` to list a folder).
2. Small text files (CSVs under a few MB, notebooks, PDFs) can be read with `mcp__Google_Drive__read_file_content`. For anything larger, use `mcp__Google_Drive__download_file_content`.
3. The embedding CSVs and `.npz` files are too big to read through the tool, and images and the checkpoint are binary. Ask the user to upload them or run the step in Colab.
4. If the Drive tools return "Forbidden", the connector needs reconnecting in the user's claude.ai connector settings. Many files are owned by `corals_inpa@menasim.org` or `ataragold@gmail.com` and only visible if shared with the user's account.
5. If a file can't be found at all, ask the user for it. Don't recreate it.

## Drive map
Root `CoralsMasking`: https://drive.google.com/drive/folders/1LtqWRm2vieTbUGgb1ISfsqVD-g-urPRY
Folder IDs below are for `parentId = '...'` queries.

Relevant:
- Active run `results_all/iou60_sta55_pts32` (id `1up9-VQJEPyv5p8YbPCPgmO5yVx-SCz3r`). Contains `json/`, `cleaned_images_crops/`, `cluster_previews/`, `visualizations/` (subfolder contents not inspected), `coral_dinoV2_encodings.csv` (181 MB), `coral_dinoV2_encodings_MASKS.csv` (489 MB), and `v2_pipeline/`.
- `assaf_expert_test_set.csv` (id `1PIBGXIduvOnRn3lvsTT2ZyFtjKiUVRRD`), in the active run folder. Columns: `image_name, coral_id, bbox, area`, 20 genus probabilities, `predicted_genus, confidence, expert_genus, is_correct, review_source`. `expert_genus` can be `unidentifiable` or `MISSING_LABEL`, or a soft coral (Cladiella, Sinularia, Litophyton) that is outside the 20 target genera. Exclude these when computing accuracy.
- `ground_truth_genus_counts.csv` (id `1DbQaZa-3lTehQPP-7PmjAkiCkbvuadwd`), in the root. Columns: `Category, genus, count, Pi`. It includes soft corals and genera outside the 20 targets, so filter to the 20 before comparing.
- `coral_weak_labels_finetuned-bioclip_taxonomy_masked_best.csv` (id `1oB15CLt1FEQfbp4Xai1xlWs-hsiJRYqC`): zero-shot labels for the **50-densest-image sample** (842 crops). "best" is the sample mode (`SAMPLE_MODE = "best"`), not the best configuration.
- `v2_pipeline/` (id `1pJq-dvipNiprBiQRXLAqH_iXTbEcjuNG`): `v2_weak_labels_expanded.csv`, `v2_embeddings.npz`, Millepora audit files (`millepora_sharp_audit`, review PDFs), `expert_review_for_assaf.csv`, `expert_review_images/`.
- Checkpoint: `CoralSCOP/checkpoints/vit_b_coralscop.pth` (id `1tiN32L4h4A5cIZAf1jeP9q8Euf-9U0Eo`, 382 MB).
- `cleaned_jsons_filtered` (id `1ATUrW-DLZMxGYio6N-9bqVuonFKkQZK4`): filtered mask JSONs.
- `data_all` (id `1oj6WQ9jNW6vUGe18PyvT5eniaASBGh68`): all source photos. `data_50` (id `1s7ZONaUeHH0BAhnH0UBH8l67X8HrBUq5`): 50-image subset.
- Colab: CoralMasking https://colab.research.google.com/drive/1sS9aUIWFBk6EP54biw_vGc3n3oyw51H7, WeakClassification https://colab.research.google.com/drive/1hCcbbjEHMtZQEm0TXiYwU0q-E_ZnIX4K, Millepora_Analysis https://colab.research.google.com/drive/18TE7K8VihAIzcgaQ1ULCMBTlHpqtPACS.

Ignore:
- `external_libs/`: vendored pip packages (tqdm, pyparsing, tabulate, ...).
- `CoralSCOP/` in Drive (including `.git`, `segment_anything`, `figs`, `demo_imgs`): duplicate of the submodule. Only the checkpoint is needed.
- `results_50/`: early parameter sweeps on 50 images, superseded by `iou60_sta55_pts32`.
- `cropped_masks/`: empty when checked.
- Root-level `coral_dino_encodings.csv` (1.2 MB) and the 2026-01/03 `coral_dino_*` CSVs elsewhere in My Drive: older, smaller runs.
- The embedding CSVs are 180-490 MB. Don't read them through the Drive tool, since downloading is the only practical way.

Notes: the Drive onboarding guide calls the notebook `WeakClassification.ipynb`; in the repo it is `weak_classification.ipynb` (in `notebooks/legacy/`). The V2 notebook, which used to exist only on Drive, is now in `notebooks/legacy/` too.
