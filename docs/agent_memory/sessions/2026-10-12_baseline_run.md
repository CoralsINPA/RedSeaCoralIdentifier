# 2026-10-12: The 20-genus baseline and the portrait-impact report

The user ran `notebooks/03_zero_shot.ipynb` and the first cells of `notebooks/04_expert_review_and_audits.ipynb` (up to the old Millepora audit against the baseline) on a Colab T4. The numbers below were read from the files these notebooks wrote on Drive, through the Drive connector:
- `<run>/reports/baseline_finetuned-bioclip__taxonomy_v1__hard20__all_2026-10-12/` (13 files: `summary.json`, the expert, distribution and orientation CSVs);
- `<run>/reports/portrait_impact_2026-10-12/` (`summary.json`, `expert_portrait_rows.csv`, `audit_rows.csv`);
- `<run>/manifest/manifest.parquet`.

## What was done

- **The "manifest missing" error.** On the first try every input printed "not there yet". Drive was mounted with a Google account other than the one the data is shared with, and that account had its own folder at the same path. Mounting with the right account fixed it. The setup cells of 03, 04 and the parity notebook now stop with that explanation when `DATA_ROOT` has no `results_all/` folder.
- **The baseline ran.** The run resumed from earlier progress, and the remaining images took 771 s. It wrote:
  - `<run>/predictions/finetuned-bioclip__taxonomy_v1__hard20__all.parquet`
  - `<run>/features/finetuned-bioclip__masked.npz`, embeddings of shape (16080, 512).

## Verified

**Coverage and confidence** (`orientation_summary` in the baseline `summary.json`)

| Photos | Photos (n) | Masks | Kept crops = predictions | Mean confidence | Share ≥ 0.85 | Mean margin |
|---|---|---|---|---|---|---|
| All | 1,714 | 44,211 | 16,080 | 0.494 | 25.8% | 0.382 |
| Landscape | 1,493 | 38,130 | 13,607 | 0.491 | 25.5% | 0.380 |
| Portrait | 221 | 6,081 | 2,473 | 0.506 | 27.5% | 0.396 |

- 1,709 photos have at least one kept crop, so 5 have none.

**Expert set** (`expert_paired.csv` and `summary.json`; Wilson 95% intervals)

| Expert rows | in_set | out_of_set | unidentifiable | missing | Legacy correct | New correct | Predictions changed |
|---|---|---|---|---|---|---|---|
| Landscape, 67 | 51 | 5 | 9 | 2 | 34/51 = 66.7% | **34/51 = 66.7% (53-78%)** | 0 of 67 |
| Portrait, 10 | 3 | 0 | 7 | 0 | 2/3 | 0/3 | 7 of 10 |
| All, 77 | 54 | 5 | 16 | 2 | 36/54 = 66.7% | 34/54 = 63.0% | 7 of 77 |

- The 16 `unidentifiable` / 2 `MISSING_LABEL` split read from `is_correct` is now confirmed.
- **On landscape rows the baseline equals the legacy result exactly,** as parity predicted. Counting the 5 soft corals as errors, it is 34/56 = 60.7%.
- **The lower "all" figure is not a regression.** On the 10 portrait rows Assaf labeled the legacy crop, which was cut from the wrong place. The baseline classifies the right region, so his label doesn't apply to it.
- **Per genus, landscape rows** (support / predicted / correct):

  | Genus | Support | Predicted | Correct |
  |---|---|---|---|
  | Platygyra | 6 | 5 | 5 |
  | Acropora | 5 | 9 | 5 |
  | Porites | 6 | 6 | 4 |
  | Millepora | 4 | 5 | 4 |
  | Dipsastraea | 7 | 2 | 1 |
  | Favites | 4 | 3 | 2 |
  | Goniastrea | 4 | 3 | 2 |
  | Echinopora | 3 | 2 | 2 |
  | Hydnophora | 2 | 3 | 2 |
  | Montipora | 2 | 1 | 1 |
  | Seriatopora | 2 | 2 | 2 |
  | Pocillopora | 2 | 2 | 2 |
  | Acanthastrea | 2 | 1 | 0 |
  | Favia | 0 | 4 | 0 |
  | Leptastrea | 0 | 2 | 0 |
  | Lobophyllia | 0 | 2 | 0 |
  | Cyphastrea, Galaxea | 0 | 1 each | 0 |
  | Stylophora, Pavona | 1 each | 1 each | 1 each |
  | Cladiella, Sinularia, Litophyton (soft) | 3, 1, 1 | n/a | n/a |

- **Top landscape confusions:** Dipsastraea → Favia 2, Dipsastraea → Leptastrea 2, Acanthastrea → Lobophyllia 2, then single cases (Platygyra → Galaxea, Favites → Favia, Favites → Goniastrea, …).

**The portrait rows explain 7 of the 16 "unidentifiable" calls** (`expert_portrait_rows.csv`)
- 7 of the 10 portrait rows are unidentifiable, against 9 of 67 landscape rows (70% vs 13%; one-sided Fisher exact p = 0.0004).
- Two of them, `IMG_4470.JPG#28` and `IMG_7282.JPG#43`, have the same legacy probability vector (Porites 0.9960259). Two different corals gave the model the same input, most likely an empty (black) crop.
- The correctly cut versions of the 7 are not unusually blurry by the current measure. Their masked Laplacian variance sits between the 10th and 87th percentile of kept crops; three are at the 79th-87th.
- So these 7 were judged on bad cuts, not on blurred corals. The evidence that blur causes unidentifiable crops now rests on the **9 landscape rows**, which haven't been looked at yet.

**Old Millepora audit against the baseline** (portrait-impact `summary.json`)
- 129 rows: 126 on landscape photos and 3 on portrait photos.
- The 3 portrait rows (0 yes, 2 no, 1 unsure) were judged on mis-cut crops and need re-review.
- Landscape: 66 yes, 44 no, 16 unsure, so **60.0% yes of the decided (95% 51-69%)**. 110 judgments are reusable.
- This is still a non-expert opinion, given with the model's answer visible, on the most confident slice. It is not a precision figure.

**Genus mix against the survey** (`distribution_all.csv`, the 20 genera only)

| Genus | Predicted | Share | Survey share | Ratio |
|---|---|---|---|---|
| Pavona | 2,425 | 15.1% | 0.44% | **34.5×** |
| Leptastrea | 1,094 | 6.8% | 1.2% | 5.5× |
| Montipora | 979 | 6.1% | 1.7% | 3.7× |
| Porites | 1,108 | 6.9% | 3.9% | 1.8× |
| Favia | 1,412 | 8.8% | 5.6% | 1.6× |
| Echinopora | 1,239 | 7.7% | 5.1% | 1.5× |
| Acropora | 2,257 | 14.0% | 13.5% | 1.0× |
| Millepora | 953 | 5.9% | 6.1% | 1.0× |
| Stylophora | 1,174 | 7.3% | 12.3% | 0.6× |
| Favites | 480 | 3.0% | 6.1% | 0.5× |
| Platygyra | 952 | 5.9% | 18.1% | **0.33×** |
| Cyphastrea | 143 | 0.9% | 3.2% | 0.28× |
| Dipsastraea | 129 | 0.8% | 5.6% | **0.14×** |
| Galaxea | 20 | 0.1% | 1.0% | 0.13× |

The other six genera are between 0.48× and 1.07× of the survey (Goniastrea, Acanthastrea and Hydnophora about 0.5×; Seriatopora, Pocillopora and Lobophyllia about 1×).
- The survey counts colonies and the baseline counts kept crops, so the ratios are rough. But nothing in that difference explains 34×.
- Platygyra is precise on the expert set (5 of 5), so its shortfall is missed colonies (recall), probably going to other massive genera.

**Portrait vs landscape mix** (`genus_by_orientation.csv`): similar overall. The largest gaps are Porites 11.0% vs 6.2%, Echinopora 9.5% vs 7.4%, Montipora 4.5% vs 6.4% and Leptastrea 5.3% vs 7.1%. It's unknown whether the portrait photos come from different sites.

**Blur filter** (`manifest.parquet`)
- The kept crops' sharpness (masked Laplacian variance) has its 1st percentile at 108, 5th at 172 and median at 558.
- The threshold of 80 sits below the 1st percentile, which is why only 117 masks were dropped for blur.
- Kept crop area: median 59,398 px, 5th percentile 26,797 px (the minimum is 25,000).

## Conclusions

1. **The baseline is sound and is now the reference.**
   - It equals the legacy result on landscape photos.
   - It fixes the 2,473 portrait crops.
   - It has 16,080 crops instead of 14,224.
2. **The expert set can't show its improvement.** The landscape rows are unchanged by construction, and the portrait labels don't apply. Measuring the baseline needs new labels (idea 1.1) and a re-review of the 10 portrait rows (idea 1.2).
3. **Pavona is the largest systematic error:** 15.1% of crops against 0.44% of colonies. The expert set has a single Pavona row, so its precision is unknown. Look at a random sample of Pavona predictions before correcting the prior (idea 5).
4. **The massive complex is under-called and confused.**
   - Dipsastraea is 0.14× the survey, and 1 of 7 is right.
   - Platygyra is 0.33× the survey.
   - Favia is 1.6× the survey and is never right on the expert set.
   - This supports the taxonomy question (idea 6) and the patch idea (idea 7).
5. **Blur is still a priority, but its evidence is thinner than thought.** Half of the unidentifiable calls (7 of 16) were mis-cuts. Next, check the 9 landscape unidentifiable crops: their sharpness and the crops themselves.

## Open

- **The 9 landscape unidentifiable crops:** join the expert CSV with the manifest's `sharpness` and look at them. A few lines in Colab, or a download of the expert CSV.
- **The blind review sheet (04, cell 9) is on hold,** to be combined with the larger expert set of idea 1.1 so Assaf reviews once.
- **Drive cleanup** of `cleaned_images_crops/` and `cleaned_jsons_filtered/`: waiting for the user's confirmation.

## Next

1. Stage C of `docs/agent_prompts/implement_organization_plan.md`.
2. The algorithm plan, starting from `docs/algorithm_ideas.md` (updated with these numbers).
