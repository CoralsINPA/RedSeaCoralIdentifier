# Pipeline history and learnings

Snapshot of the claude.ai project memory entry "Pipeline History And Learnings" (last updated 2026-09-07), imported on 2026-10-11. The original text is unchanged below the Notes.

## Notes (added 2026-10-11)

- **V2 and its Millepora audit are excluded (2026-10-13).** The V2 pipeline (24 genera) and the Millepora audit made on its predictions are no longer a reference; numbers below that come from them (for example the 58.9% Millepora figure, V2's Pavona share or its soft-coral share) are history, not evidence. The original text is kept as it was. See `docs/agent_memory/sessions/2026-10-13_detach_exploration_remove_v2.md`.
- **The evidence that Iteration 1 "failed entirely" is partly confounded.**
  - The cluster preview galleries (`CoralMasking.ipynb` cells 14 and 22) display crops cut from the wrong part of each photo. The masks are in a 2560×1920 frame, but those cells crop the full-size photo (e.g. 3648×2736) without resizing it first.
  - The HDBSCAN run used `coral_dinoV3_encodings.csv`, and no notebook I read writes that file.
  - The features behind the k-means gallery (masked DINOv2) and the DinoVdeau metrics were cropped correctly. On DinoVdeau, the cosine silhouette scores (0.27, 0.24 and 0.19 for k = 2, 4, 6) do show weak cluster structure, so the conclusion probably holds.
  - Still, the visual judgement was made on misaligned crops.
- **The DINOv2 features weren't all aligned.** "DINOv2 embeddings" covers three files. Only the bbox-crop file (`coral_dino_encodings.csv`) is misaligned; the masked-crop file is fine.
- **Iteration 2 compared runs unevenly.** All its model and prompt comparisons used the same 842 crops from the 50 densest images, and two of the six runs used an older 10-genus label set. The file named `..._masked_best.csv` comes from that 50-image sample, not from the best configuration.
- **Iteration 3 has a join bug.** Embeddings and predictions are keyed by `(image_name, coral_id)`, but the Phase 2 code looks them up by `coral_id` alone. That has to be fixed before training.

## Summary

The three pipeline iterations, why each approach worked or failed, and the architectural insights and tools behind the current design.

## Pipeline iterations

- Iteration 1 — CoralMasking: CoralSCOP (SAM-based, coral-specific weights) for instance segmentation + DINOv2 embeddings → unsupervised clustering (K-Means/HDBSCAN); failed entirely, no genus separation
- Iteration 2 — WeakClassification v1: same CoralSCOP masks + BioCLIP-FT (ReefNet fine-tuned ViT-B/16) for zero-shot classification across 20 hard coral genera; saved softmax scores only, not raw embeddings
- Iteration 3 — v2 pipeline (current): same BioCLIP-FT model, expanded to 24 genera, saving raw 512-dim BioCLIP-FT embeddings alongside predictions

## Key learnings

- Core architectural insight: LogisticRegression operates on 512-dim BioCLIP-FT embedding vectors, not images directly — BioCLIP-FT does the visual heavy lifting, and the classifier learns decision boundaries in an embedding space where genera are already partially separated
- Unsupervised clustering on generic embeddings (DINOv2) is insufficient for fine-grained coral taxonomy; domain-aware embeddings are essential
- The confusion structure is biologically interpretable rather than random noise, which validates BioCLIP-FT as a viable foundation for a supervised classifier
- Taxonomic label drift (e.g. Favia→Dipsastraea reclassification) is a real data quality issue that affects model predictions in predictable ways
- Persisting raw embeddings, not just softmax scores, is the critical enabler for downstream supervised training — the key contribution of v2

## Tools & resources

- CoralSCOP — SAM-based segmentation model with coral-specific weights
- BioCLIP-FT (ReefNet) — fine-tuned ViT-B/16; current embedding and zero-shot classification backbone, 512-dim embeddings
- DINOv2 — used in the early pipeline, abandoned after the clustering failure
- Google Colab — development environment for the notebooks
- Google Drive — connected but requires session-level authentication to access files via MCP tools; direct Colab URL fetching via web tools fails at authentication
