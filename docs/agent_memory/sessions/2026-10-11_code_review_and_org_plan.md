# 2026-10-11: Full code read and organization plan

Claude Code cloud session on branch `claude/awesome-archimedes-kvh4c7`. Main output: `docs/organization_plan.md`.

## What was done

**Read in full**
- The onboarding guide and all three repo notebooks.
- `CoralClassification_v2_Pipeline.ipynb`, downloaded from Drive.

**Checked**
- The CoralSCOP submodule.
- One source photo.
- The Drive listings of `v2_pipeline/` and the active run folder.

**Wrote**
- The organization plan: no code changes yet.
- This memory folder.

## Verified (with evidence)

| Fact | How it was checked |
|---|---|
| Source photos are larger than the mask frame: `IMG_4033.JPG` is 3648×2736, and its masks are in the 2560×1920 frame | Read the JPEG header after downloading the file from Drive. CM 8 downscales to max side 2560. |
| CM 10, 14, 17, 18 and 22 crop the full-size photo using bboxes from the smaller frame, so they cut the wrong region | Code read. Affected: the bbox DINOv2 features, the cluster preview galleries, `cleaned_jsons_filtered/`, and `cleaned_images_crops/` (which is also unfiltered) |
| WC, V2 and MA resize to 2560×1920 first, so their crops are correct for 4:3 photos | Code read. Not yet checked for all 1,714 photos. |
| `coral_id` restarts at 0 in every image, but V2 Phase 2 and WC `RESUME` treat it as unique | Code read: CM 8 resets `anno_id` per image |
| The 14,224-crop run (mean confidence 0.485) is V2 with 24 genera, not WC | V2 cell 10 output |
| `..._masked_best.csv` is the 50-densest-image sample (842 crops), and all six WC comparison runs use the same 842 crops | WC cells 14 and 23 outputs |
| The CoralSCOP SAM has a 2-class coral/background head and returns per-mask `fc_features`, which the pipeline discards | Diffed the submodule's `segment_anything` against pip `segment-anything==1.0` |
| The Millepora audit's crops were cut correctly, and its yes/no labels are keyed `(image_name, coral_id)` on run `iou60_sta55_pts32` | MA cells 4 and 6 resize to 2560×1920 and look up the RLE in that image's own JSON |
| v1 accuracy on Assaf's set is 39/57 = 68.4% on in-set crops (39/62 counting soft corals as errors); 95% interval 55.5-79.0% | Recomputed from the user's confusion matrix (2026-10-11), whose printed recall and precision rows all match its cells. Not yet re-derived from the CSV. |

## Decisions made by the user

- **20 hard genera** is the final label set, and V2 is not carried forward as a pipeline.
- `cleaned_images_crops/`, `cleaned_jsons_filtered/` and the bbox DINOv2 features may be deleted, since they're defective.
- Where `coral_dinoV3_encodings.csv` and `build_expert_test_set.py` came from is not crucial. Neither is needed going forward.
- The repo is public for now, so no private details go into it.
- The runtime is Colab for now, though not ideal, so the code must not depend on Colab.
- The package name is `coralid`.

## Conclusions

**Does V2 classify better than WC? No; it's an engineering upgrade.**
- The classification core is identical: same BioCLIP-FT weights, same masks, filters and masked crop, same `softmax(100·cos)`.
- The real differences are the 4 soft-coral classes, updated family names in the taxonomy prompts, and running on all images instead of a 50-image sample.
- V2 was never scored against the expert set. That set holds v1 (20-genus) probabilities.
- Both runs grossly over-predict Pavona: 13.7% (WC sample) and 17.6% (V2, hard-coral predictions), against 0.4% in the survey.
- What V2 does better is engineering: it saves embeddings, adds runner-up and checkpointing, runs the full set, and exports reviews. The package absorbs those.

**The Millepora audit isn't wrong data, but its number doesn't carry over to the 20-genus setup.**
- The 129 human judgments (66 yes / 46 no / 17 unsure) are valid labels for those crops.
- The 58.9% precision, though, was measured on V2's 24-genus predictions. Without the soft-coral classes, crops V2 called Sinularia or Sarcophyton will move to the hard genera. Sinularia was Millepora's #3 runner-up, so the 20-genus precision could be lower.
- **Plan:** run the 20-genus model on all images, reuse the existing 129 judgments where crops overlap, and review only the new Millepora predictions.
- **Next time:** fill in `actual_genus` for the "no" crops. It was left empty, so we don't know what Millepora gets confused with.

**Going 20-hard still needs a reject option.**
- Soft corals are in the photos: 5 of 62 identifiable crops in Assaf's review, and 11.7% of V2's predictions.
- Without an "other" outcome they get forced into a hard genus.
- This is the first algorithm question to settle.

**Other conclusions**
- **The DINOv2 clustering failure stands, but its visual evidence is confounded.** The galleries showed misaligned crops, while the silhouette scores came from correctly aligned DinoVdeau features.
- **The confusion matrix: correction.**
  - Earlier in this session I flagged three contradictions between the memory's matrix and its prose.
  - The user then supplied the authoritative matrix. It shows the memory's *table* had transcription errors, and the prose was right about Dipsastraea.
  - Still wrong in the prose: the Lobophyllia ↔ Acanthastrea direction, and the predicted → true wording of the calibration examples.
  - See `../assaf_review_confusion_matrix.md`.

## Later in the session: sequencing and expectations

- **Sequencing (user decision):** do the full organization (Stages A1-C of `docs/agent_prompts/implement_organization_plan.md`) before any algorithm work. Stage B's 20-genus run on all images becomes the new baseline.
- **Expectation: the refactor alone probably won't raise accuracy much.**
  - The bugs found mostly affect side outputs: cleaned crops, bbox DINOv2 features, cluster previews, V2 Phase 2 joins, and WC's resume option. The zero-shot classification path (resize → filter → masked crop → BioCLIP-FT) was already correct for 4:3 photos.
  - Two things could still move the numbers:
    - photos that aren't 4:3, which legacy code stretched (the Stage A frame audit will show these);
    - images whose segmentation error was silently saved as "0 corals" (this affects coverage, not accuracy).
  - What the refactor does deliver:
    - a full-dataset baseline instead of the 50-image sample;
    - correct keys everywhere;
    - a repeatable evaluation, so later algorithm changes can actually be measured.
- **The measurement limit.** With 57 in-set expert crops, the 95% interval on 68.4% runs from 55.5% to 79.0%, so only large changes will show. A larger stratified expert review drawn from the new baseline (the ~300-crop idea in `project_context.md`) is what makes improvements measurable.
- **Error budget from the matrix (23 errors):**
  - massive / merulinid complex: 13;
  - soft corals with no class: 5;
  - everything else: 5.
- **Soft corals drag down precision.** They cause most of the precision loss on Acropora (56% → 83% if they were rejected), and on Millepora and Porites. A reject option is likely the cheapest early gain once the baseline exists.

## Open questions

- Do all 1,714 photos share the 4:3 landscape frame? A Phase 1 audit script will check.
- How are `MISSING_LABEL` rows counted in the expert CSV, given that 62 + 16 = 78 already?

## Next

1. Refactor Phase 0 (hygiene) and Phase 1 (core library with no behavior change). See `docs/organization_plan.md` §4.
2. Algorithm planning for the 20-genus classifier:
   - reject option for soft corals;
   - calibration and thresholds;
   - a representative evaluation sample;
   - the linear probe on embeddings;
   - Favia/Dipsastraea naming with Assaf.
