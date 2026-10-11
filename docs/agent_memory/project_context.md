# Project context

Snapshot of the claude.ai project memory entry "Coral genus classification for the Israel Nature and Parks Authority" (last updated 2026-09-29), imported on 2026-10-11. The original text is unchanged below the Notes.

## Notes (added 2026-10-11)

- **Overall accuracy.** "~59% accuracy (38 correct, 26 wrong, 16 unidentifiable)" counts the original 80-crop review. The transcribed 78-crop CSV gives 39/62 = 62.9%. If you leave out the 5 soft corals, which weren't among v1's 20 classes (the rule in `CLAUDE.md`), it's 39/57 = 68.4%. See `assaf_review_confusion_matrix.md`.
- **Millepora may not be reliable.** It is listed below as handled reliably. But the Millepora check finished on 2026-09-15 (`Millepora_Analysis.ipynb`) judged only 58.9% of the sharp, confidence ≥ 0.85 V2 predictions correct. The judging was done by a non-expert analyst, not Assaf, so it is provisional. See `README.md`.
- **Pavona over-prediction.** "Pavona overprediction worsened significantly" fits what I measured: 13.7% of v1 predictions (842-crop sample) and 17.6% of V2 hard-coral predictions (all 14,224 crops), against 0.4% in the survey. The two runs used different crop sets, though, so the size of the change isn't pinned down.
- **The GitHub repo exists** at `CoralsINPA/RedSeaCoralIdentifier`.
- **Phase 2 has a bug.** The planned Phase 2 code joins on `coral_id` alone, and `coral_id` repeats across images. It needs fixing before the classifier is trained (`docs/organization_plan.md` §2.1 #3).
- **Superseded on 2026-10-11:** the project goes with the **20 hard genera**, and V2 is not carried forward as a pipeline. See `sessions/2026-10-11_code_review_and_org_plan.md`.
- **Redacted for the public repo:** a GitHub account handle, the Claude credit budget, and a personal detail about the new team member. Removed spots are marked `[…]`.

## Summary

Coral genus classification for the Israel Nature and Parks Authority: goal, collaborators, current pipeline state, and what's next.

## Details

- Arielle is building an automated coral genus classifier for the Israel Nature and Parks Authority, covering ~1,700 underwater Red Sea images
- Goal: a reliable automated classifier that identifies coral genera at scale, using a human-in-the-loop workflow where expert review feeds supervised training
- Collaboration with Atara
- Noam — computer vision expert advisor
- Assaf Zvuloni — domain expert at the Israel Nature and Parks Authority, responsible for expert review of model predictions

## Current state

- v2 pipeline is the active iteration: BioCLIP-FT (ReefNet fine-tuned ViT-B/16) over CoralSCOP masks, expanded to 24 genera and saving raw 512-dim embeddings alongside predictions
- Expansion to 24 genera added four soft corals: Cladiella, Sinularia, Litophyton, Sarcophyton
- v2 is the first iteration to persist coral-aware embeddings for downstream supervised training
- v2 runs a two-phase workflow — Phase 1: tiered auto-labeling (high-confidence crops auto-accepted, remainder flagged for expert review); Phase 2: LogisticRegression trained on BioCLIP-FT embeddings once expert corrections are incorporated
- Assaf reviewed 80 crops from v1 output: ~59% accuracy on identifiable corals (38 correct, 26 wrong, 16 unidentifiable)
- Assaf's review has been digitized into a structured CSV test set (`assaf_expert_test_set.csv`) with a reusable `evaluate_against_expert()` function for benchmarking future models; 78 of 80 crops transcribed (2 likely Seriatopora missing from page 1 of the review PDF)
- Genera the model handles reliably: Acropora, Stylophora, Pocillopora, Platygyra, Millepora, Hydnophora, Montipora, Porites, Echinopora
- Known failure patterns: Dipsastraea confusion (training data uses outdated "Favia" labels predating the 2012 Merulinidae reclassification); soft coral blind spot (addressed in v2); massive coral complex confusions (Lobophyllia↔Acanthastrea, Goniastrea↔Favites)
- Open v2 tradeoff: expanding to 24 classes caused softmax dilution — some previously well-calibrated genera degraded, and Pavona overprediction worsened significantly

## On the horizon

- Generate a stratified review set of ~300 crops (12–15 per genus) for Assaf rather than sending the full flagged set
- Incorporate Assaf's corrections and run Phase 2 to train the LogisticRegression classifier on BioCLIP-FT embeddings
- Monitor whether softmax dilution from the 24-class expansion needs mitigation strategies
- A new team member ([…], name not yet mentioned) is joining the coral project; entirely new and needs onboarding on the codebase
- Looking to recruit another volunteer for the project
- (Sep 2026) Not yet sure about the v2 expert-review classification track; onboarding is based on CoralMasking + "WeakClassification copy" (the copy is the canonical WeakClassification notebook)
- Planning to move the Colab code to a private GitHub repo she creates ([…]): the 4 notebooks plus lightweight data (embeddings etc.); [Claude cloud-session credit details: […]]

## Earlier claude.ai conversations (titles only)

| Date | Title |
|---|---|
| 2026-09-29 | Drive structure documentation |
| 2026-09-23 | Volunteer candidate requirements |
| 2026-09-18 | Colab notebook review |
| 2026-09-18 | Fable1 project compatibility |
| 2026-09-16 | Saving Asaf's classification as CSV |
| 2026-09-15 | Notebook functionality summary |
| 2026-08-09 | Google Colab access verification |
