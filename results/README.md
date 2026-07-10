# Results

Curated, paper-relevant result artifacts (small CSVs + figures). The heavy raw generations (per-cell
source/target/disguised text) live on the HuggingFace dataset
`dementor-research/dementor-matrix-responses`, not here. Reproduce any of these from the cached text
with the scripts under `experiments/`.

> **Numbers pending.** Specific effect sizes / statistics are being finalized against the completed
> runs and are omitted from these descriptions rather than shipped stale — each CSV still stores its
> computed values, and `docs/RESULTS.md` / `METHODS.md` carry the authoritative framing.

## The disguise ladder (the experimental scaffold)
- `matrix_ladder/<dataset>_matrix_ladder.{csv,png}` — per-rung persistence across all source→target
  pairs, per dataset. Shows the monotone collapse: naming → prompting → SFT → DPO.

## The headline result — model-dependent DPO erasure
- `d2_multiseed_ci.csv` — **D2**: per-cell persistence, mean ± 95% CI over adapter seeds
  (de-confounded); the two-tier **retain** vs **launder** source split.
- `fig1_source_fingerprint.png` — DPO persistence per source model (two-tier split).

## Robustness / corrections
- `d1_decontam_before_after.csv` — **D1**: before/after the gpt-oss chat-template CoT-leak fix.
- `d3_encoder_swap_bge.csv` — **D3**: persistence re-scored under a different encoder (bge-small);
  the rung ordering replicates.
- `source_distinctiveness.csv` — baseline distinctiveness per model under several encoders vs DPO
  persistence. **Caveat:** the distinctiveness↔persistence link is *encoder-sensitive* (holds under
  MiniLM/bge, not under mpnet/structural), so "distinctive models resist DPO" is suggestive, not the
  load-bearing mechanism; the two-tier result itself is robust.

## Which behavioural directions carry the fingerprint
- `style_directions.csv` / `big5_personality_directions.csv` — style-axis vs Big-Five projections.
  Big-Five barely separates the models; style axes (verbosity, structure) separate most — the
  fingerprint is structural-style, not personality.

See `docs/RESULTS.md` for the authoritative framing and `METHODS.md` for the metric definition.
