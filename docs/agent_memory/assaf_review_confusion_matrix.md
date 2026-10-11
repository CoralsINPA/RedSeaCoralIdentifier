# Assaf review: confusion matrix

Snapshot of the claude.ai project memory entry "Assaf Review Confusion Matrix" (last updated 2026-09-16), imported on 2026-10-11. The original text is unchanged below the Notes. The matrix was a tab-separated table and is transcribed here as markdown.

## Notes (added 2026-10-11)

- **The matrix and the prose disagree.** Rows are the expert (true) genus and columns the predicted genus. The matrix agrees with the per-genus recall list, but three statements in "Key conclusions" contradict it. Re-derive all of these from `assaf_expert_test_set.csv` once the evaluation code exists.
  1. **Dipsastraea.** The matrix row has 1 correct, plus Favia 2, Lobophyllia 2, Favites 1 and Acropora 1. The prose says Favia 2, Leptastrea 2, Acanthastrea 1 and Favites 1. The prose was probably read one column off (Lob vs Lep) or confused the abbreviations (Acr vs Aca).
  2. **Lobophyllia ↔ Acanthastrea.** The matrix says the 2 true *Acanthastrea* were predicted as *Lobophyllia*, and the recall list agrees (0 true Lobophyllia). The prose has it the other way round.
  3. **The miscalibration examples are written backwards.** "Dipsastraea 0.96 → Porites" and "Acropora 0.94 → Cladiella" read as predicted → true, the opposite of the matrix. In the matrix they appear as true Porites → predicted Dipsastraea and true Cladiella → predicted Acropora.
- **Which accuracy number to use.** The 62 identifiable crops include 5 soft corals (Cladiella 3, Litophyton 1, Sinularia 1) that v1 could never predict. Accuracy on in-set crops is 39/57 = 68.4%, the number to use under the rule in `CLAUDE.md`. The original 80-crop count gave 38/64 = 59.4%.
- **Counts don't fully reconcile.** `CLAUDE.md` says `expert_genus` can be `MISSING_LABEL`, but 62 identifiable + 16 unidentifiable already adds up to the 78 rows. Check how `MISSING_LABEL` rows are counted when the evaluation is wired up.
- **Tier-1 evidence is thin.** There are no Seriatopora rows; the 2 missing crops were "likely Seriatopora". So Seriatopora's Tier-1 (auto-accept) status in V2 isn't supported by this review. The other Tier-1 genera rest on 2-7 crops each.
- **Millepora was audited later.** The 4/4 Millepora recall here comes from the random review sample. The later Millepora precision audit (129 sharp crops at confidence ≥ 0.85) found 58.9% precision.

## Summary

Confusion matrix and accuracy analysis from Assaf's 78-crop expert review of the v1 BioCLIP pipeline.

## Dataset

- 78 crops transcribed from Assaf's review (2 likely Seriatopora crops missing from page 1)
- 62 identifiable crops (expert gave a genus), 16 unidentifiable ("?")
- Overall accuracy on identifiable crops: 62.9% (39/62)

## Confusion matrix (true genus → predicted genus, counts)

Blank = 0. Abbreviations: Acr Acropora, Pla Platygyra, Por Porites, Mil Millepora, Mon Montipora, **Fav Favites**, Sty Stylophora, Gon Goniastrea, Hyd Hydnophora, Ech Echinopora, Poc Pocillopora, Dip Dipsastraea, Pav Pavona, **Favia Favia**, Aca Acanthastrea, Cyp Cyphastrea, Gal Galaxea, Lob Lobophyllia, Lep Leptastrea.

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
