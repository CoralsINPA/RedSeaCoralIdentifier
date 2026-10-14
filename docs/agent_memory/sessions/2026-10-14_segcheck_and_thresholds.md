# 2026-10-14: Segmentation check, overlay review, and the filter thresholds

## Verified
- **Notebook `01_segment_and_filter` ran in Colab.** The segmentation check verdict is **PASS** (report: `reports/segcheck_2026-10-14/` on Drive).
  - The new `coralid` segmentation re-segmented 5 photos into the scratch run `results_all/segcheck_iou60_sta55_pts32/`, one of them portrait (IMG_4484).
  - All 109 saved masks were found again, 51 of 51 large ones. Mask counts and frames are identical, and the mean IoU is 1.000.
  - So the new code reproduces the original CoralSCOP masks.
- **The filter funnel matches the earlier record:** 44,211 masks → area 19,457 → aspect 18,945 → containment 16,197 → blurry 16,080 kept. Landscape and portrait photos lose similar shares to each filter.

## Observations from the overlays (user's screenshots)
- **Masks sit exactly on the corals**, including on the portrait photo IMG_4484, so the portrait fix is confirmed visually.
- **The size filter (MIN_AREA 25,000 px) is the dominant filter (about 56% of masks).** In IMG_8909, dozens of small, distinct colonies are rejected for size; only about 6 are kept.
- **Segmentation misses large colonies entirely.** In IMG_4484, a large tan encrusting or plate colony covering about half the photo has no mask, and neither do the large branching colonies on the left. This is a recall problem upstream of the filters (algorithm idea 9). A mask audit should measure it.
- **The blur filter does nothing,** and whole photos can be hazy. IMG_8909 has a cyan cast and low contrast throughout, yet its crops pass. This supports a per-photo quality score.
- **Some kept masks are probably not one of the 20 genera.** In IMG_4067, three flat orange-tan encrusting patches are kept (sponge, encrusting coral or coralline algae?). The model forces each into a genus. They are examples for a reject / "other" outcome (idea 3).
- **Cosmetic bug, not fixed:** the overlay legend colors depend on which categories appear in a photo. "Kept" is green in IMG_8909 and IMG_4484 but orange in IMG_4067. Use a fixed color per category.

## Conclusions: the thresholds
Fix the thresholds from data, not by guessing again:
1. **Blur (Laplacian variance ≥ 80 on the masked crop):** the measure is wrong, so changing only the threshold won't help. The mask outline against the black background reads as sharpness.
   - Measure focus inside an eroded mask, at the 224 px size the model sees. Consider a per-photo score.
   - Pick the threshold that best separates the crops Assaf could not identify from those he could.
   - First, free step: the sharpness snippet on the 9 landscape unidentifiable expert crops (7 of the 16 were portrait mis-cuts, not blur).
2. **Size (MIN_AREA 25,000 px):** depends on the end goal.
   - **A label per colony:** small crops are probably less accurate, and the current cut may be acceptable.
   - **Genus composition per site:** dropping 56% of masks biases toward large colonies, and the threshold should come down.
   - Set it from accuracy by size.
3. **Containment (0.80):** keep it. The overlays show it removing fragments correctly.
4. **Don't change the thresholds before there are labels to judge them.** A change means new crops and a baseline re-run (about 25 min on a GPU), and the current expert set only labels crops that passed today's filters.

## Next
- **The next blind expert sample (about 300 crops, idea 1.1)** should deliberately span crop sizes, including below 25,000 px, and sharpness levels. Then both thresholds can be set from Assaf's labels in one pass.
- **Decision for the user / ecologists:** a label per colony, or genus composition per site? It sets the size threshold and the weight of the segmentation-recall problem.
- **Run `02_zero_shot`.** The landscape expert line should read 34/51 = 66.7%. Then `notebooks/legacy/` can be cleaned up.
