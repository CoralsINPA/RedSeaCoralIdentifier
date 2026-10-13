# Assaf review: confusion matrix

Results of Assaf's expert review of 78 crops from the v1 pipeline (20 hard genera, BioCLIP-FT, zero-shot).

- **Original source:** the claude.ai project memory entry "Assaf Review Confusion Matrix" (last updated 2026-09-16), imported on 2026-10-11.
- **Authoritative matrix:** the version the user provided on 2026-10-11 (below).
- The original snapshot text is kept unchanged at the end of the file.

## Notes (updated 2026-10-11)

- **V2 and its Millepora audit are excluded (2026-10-13).** The V2 pipeline (24 genera) and the Millepora audit made on its predictions are no longer a reference; numbers below that come from them (for example the 58.9% Millepora figure, V2's Pavona share or its soft-coral share) are history, not evidence. The original text is kept as it was. See `docs/agent_memory/sessions/2026-10-13_detach_exploration_remove_v2.md`.
- **Check against the real file (2026-10-11, Stage A2): not done yet.** The agent could not read `assaf_expert_test_set.csv` (the Drive tool only returns contents inline). The matrix below is now data in the tests (`tests/assaf_matrix.py`): a file shaped like the real one, built from it, goes through `coralid.evaluation` and gives 78 rows, 57 in set, 5 soft corals, 16 unidentifiable, 39 correct and every printed recall and precision. Whether the **real file** equals the matrix is Check D of `notebooks/parity_check.ipynb`, which prints PASS or FAIL for each cell. Add the result here when the user pastes it back.

- **The user's matrix is internally consistent; the snapshot's table was not.** Recomputing the matrix below from its cells reproduces every printed recall and precision value. The table in the original snapshot had three transcription errors, so it is kept only for history (collapsed at the end):
  - **Dipsastraea row:** Favites 1, Favia 2, Acanthastrea 1, Leptastrea 2. The snapshot table had Acropora 1 and Lobophyllia 2 instead.
  - **Platygyra's one miss** went to Galaxea, not Lobophyllia.
- **What this means for the snapshot's prose:**
  - The Dipsastraea scatter in "Key conclusions" is **right**. An earlier note in this file said otherwise; that note relied on the faulty table and has been withdrawn.
  - Still wrong: "every Lobophyllia (2/2) was predicted as Acanthastrea". It's the reverse: the 2 true *Acanthastrea* were predicted as *Lobophyllia*. No true Lobophyllia are in the set.
  - The calibration examples ("Dipsastraea 0.96 → Porites", "Acropora 0.94 → Cladiella") are written predicted → true. The matrix records them as true Porites → predicted Dipsastraea, and true Cladiella → predicted Acropora.
- **Accuracy.** 39 correct out of 62 identifiable crops = 62.9%. Five of those are soft corals that v1 could never predict. On the 57 crops whose genus is in the label set, accuracy is 39/57 = **68.4%**, with a Wilson 95% interval of 55.5-79.0%. So this set can only show large differences between models. Comparisons on the same crops are paired, which helps a little.
- **Where the 23 errors come from, by true genus:**

  | Group | Errors | Genera |
  |---|---|---|
  | Massive / merulinid complex | 13 | Dipsastraea 6, Favites 2, Goniastrea 2, Acanthastrea 2, Cyphastrea 1 |
  | Soft corals with no class | 5 | Cladiella 3, Litophyton 1, Sinularia 1 |
  | Everything else | 5 | Porites 2, Echinopora 1, Montipora 1, Platygyra 1 |

- **Precision is where the soft corals hurt.**
  - Acropora precision is 56% because 3 of its 4 wrong predictions are soft corals.
  - If soft corals were rejected as "other", Acropora would reach 5/6 = 83%, Millepora 4/4 and Porites 5/6.
  - Favia is a sink for the massive complex: 5 predictions, 1 correct (20%).
- **Counts don't fully reconcile.** `CLAUDE.md` says `expert_genus` can be `MISSING_LABEL`, but 62 identifiable + 16 unidentifiable already adds up to the 78 rows. Check how `MISSING_LABEL` rows are counted when the evaluation is wired up.
- **Tier-1 evidence is thin.** The set has no Seriatopora (the 2 missing crops were "likely Seriatopora"). Every other per-genus figure rests on 1-7 crops.
- **Millepora was audited later.** The 4/4 Millepora recall comes from this random review sample. The later Millepora precision audit (129 sharp crops at confidence ≥ 0.85, V2 24-genus setup) found 58.9% precision.

## Confusion matrix (authoritative, provided by the user 2026-10-11)

Rows are Assaf's genus (true), columns the model's prediction. Blank = 0.

| True \ Predicted | Acropora | Platygyra | Porites | Millepora | Montipora | Favites | Stylophora | Goniastrea | Hydnophora | Echinopora | Pocillopora | Dipsastraea | Pavona | Favia | Acanthastrea | Cyphastrea | Cladiella | Litophyton | Sinularia | Lobophyllia | Leptastrea | Galaxea | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Acropora | 5 | | | | | | | | | | | | | | | | | | | | | | 100% |
| Platygyra | | 5 | | | | | | | | | | | | | | | | | | | | 1 | 83% |
| Porites | | | 5 | | | | | | | | | 1 | | | | 1 | | | | | | | 71% |
| Millepora | | | | 4 | | | | | | | | | | | | | | | | | | | 100% |
| Montipora | 1 | | | | 3 | | | | | | | | | | | | | | | | | | 75% |
| Favites | | | | | | 3 | | 1 | | | | | | 1 | | | | | | | | | 60% |
| Stylophora | | | | | | | 3 | | | | | | | | | | | | | | | | 100% |
| Goniastrea | | | | | | | | 2 | 1 | | | | | 1 | | | | | | | | | 50% |
| Hydnophora | | | | | | | | | 2 | | | | | | | | | | | | | | 100% |
| Echinopora | | | 1 | | | | | | | 2 | | | | | | | | | | | | | 67% |
| Pocillopora | | | | | | | | | | | 2 | | | | | | | | | | | | 100% |
| Dipsastraea | | | | | | 1 | | | | | | 1 | | 2 | 1 | | | | | | 2 | | 14% |
| Pavona | | | | | | | | | | | | | 1 | | | | | | | | | | 100% |
| Favia | | | | | | | | | | | | | | 1 | | | | | | | | | 100% |
| Acanthastrea | | | | | | | | | | | | | | | | | | | | 2 | | | 0% |
| Cyphastrea | | | | | | | | | | | | | 1 | | | | | | | | | | 0% |
| Cladiella | 1 | | 1 | 1 | | | | | | | | | | | | | | | | | | | 0% |
| Litophyton | 1 | | | | | | | | | | | | | | | | | | | | | | 0% |
| Sinularia | 1 | | | | | | | | | | | | | | | | | | | | | | 0% |
| Lobophyllia | | | | | | | | | | | | | | | | | | | | | | | - |
| Leptastrea | | | | | | | | | | | | | | | | | | | | | | | - |
| Galaxea | | | | | | | | | | | | | | | | | | | | | | | - |
| **Precision** | 56% | 100% | 71% | 80% | 100% | 75% | 100% | 67% | 67% | 100% | 100% | 50% | 50% | 20% | 0% | 0% | - | - | - | 0% | 0% | 0% | |

---

# Original snapshot (2026-09-16, unchanged)

## Summary

Confusion matrix and accuracy analysis from Assaf's 78-crop expert review of the v1 BioCLIP pipeline.

## Dataset

- 78 crops transcribed from Assaf's review (2 likely Seriatopora crops missing from page 1)
- 62 identifiable crops (expert gave a genus), 16 unidentifiable ("?")
- Overall accuracy on identifiable crops: 62.9% (39/62)

## Per-genus recall

- 100% recall: Acropora (5/5), Millepora (4/4), Stylophora (3/3), Hydnophora (2/2), Pocillopora (2/2), Pavona (1/1), Favia (1/1)
- High recall: Platygyra 83% (5/6), Porites 71% (5/7)
- Moderate recall: Favites 60% (3/5), Montipora 75% (3/4), Goniastrea 50% (2/4), Echinopora 67% (2/3)
- Low recall: Dipsastraea 14% (1/7)
- Zero recall: Acanthastrea (0/2), Lobophyllia (0/0 true), Leptastrea (0/0 true), Galaxea (0/0 true), Cyphastrea (0/1), Cladiella (0/3), Litophyton (0/1), Sinularia (0/1)

## Key conclusions

- Dipsastraea is the worst-performing genus: 7 true samples but only 1 correctly predicted; the model scatters it across Favia (2×), Leptastrea (2×), Acanthastrea (1×), Favites (1×) — all related to the Merulinidae/Faviidae taxonomic reclassification
- Soft corals are a blind spot: Cladiella (3), Litophyton (1), Sinularia (1) have 0% recall because they were absent from the v1 20-genus class set; the model defaults to hard coral genera like Acropora
- Lobophyllia↔Acanthastrea confusion: every Lobophyllia (2/2) was predicted as Acanthastrea — a known massive coral confusion pair
- The massive coral complex (Favia/Favites/Goniastrea/Dipsastraea) is the primary source of errors, driven by morphological similarity and outdated taxonomy
- High-confidence genera (Acropora, Millepora, Pocillopora, Stylophora, Hydnophora) are reliably correct — these have distinctive morphologies the model learned well
- Confidence is not always calibrated: some misclassifications occur at high confidence (e.g. Dipsastraea 0.96 → Porites, Acropora 0.94 → Cladiella)
- The v2 expansion targets these gaps: adding soft coral genera and improving massive coral discrimination

<details>
<summary>Snapshot's original matrix table (superseded: three transcription errors, see Notes)</summary>

Abbreviations: Acr Acropora, Pla Platygyra, Por Porites, Mil Millepora, Mon Montipora, Fav Favites, Sty Stylophora, Gon Goniastrea, Hyd Hydnophora, Ech Echinopora, Poc Pocillopora, Dip Dipsastraea, Pav Pavona, Favia Favia, Aca Acanthastrea, Cyp Cyphastrea, Gal Galaxea, Lob Lobophyllia, Lep Leptastrea.

| True \ Predicted | Acr | Pla | Por | Mil | Mon | Fav | Sty | Gon | Hyd | Ech | Poc | Dip | Pav | Favia | Aca | Cyp | Gal | Lob | Lep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Acropora | 5 | | | | | | | | | | | | | | | | | | |
| Platygyra | | 5 | | | | | | | | | | | | | | | | 1 | |
| Porites | | | 5 | | | | | | | | | 1 | | | | 1 | | | |
| Millepora | | | | 4 | | | | | | | | | | | | | | | |
| Montipora | 1 | | | | 3 | | | | | | | | | | | | | | |
| Favites | | | | | | 3 | | 1 | | | | | | 1 | | | | | |
| Stylophora | | | | | | | 3 | | | | | | | | | | | | |
| Goniastrea | | | | | | | | 2 | 1 | | | | | 1 | | | | | |
| Hydnophora | | | | | | | | | 2 | | | | | | | | | | |
| Echinopora | | | 1 | | | | | | | 2 | | | | | | | | | |
| Pocillopora | | | | | | | | | | | 2 | | | | | | | | |
| Dipsastraea | 1 | | | | | 1 | | | | | | 1 | | 2 | | | | 2 | |
| Pavona | | | | | | | | | | | | | 1 | | | | | | |
| Favia | | | | | | | | | | | | | | 1 | | | | | |
| Acanthastrea | | | | | | | | | | | | | | | | | | 2 | |
| Cyphastrea | | | | | | | | | | | | | 1 | | | | | | |
| Cladiella | 1 | | 1 | 1 | | | | | | | | | | | | | | | |
| Litophyton | 1 | | | | | | | | | | | | | | | | | | |
| Sinularia | 1 | | | | | | | | | | | | | | | | | | |

</details>
