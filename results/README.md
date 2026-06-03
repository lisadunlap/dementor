# Results

Curated, paper-relevant result artifacts (small CSVs + figures). The heavy raw
generations (per-cell source/target/disguised text, 36 cells × rungs × seeds) live on
the HuggingFace dataset `dementor-research/dementor-matrix-responses`, not here.
Reproduce any of these from the cached text with the scripts in `scripts/analysis/`.

## The disguise ladder (the experimental scaffold)
- `matrix_ladder/<dataset>_matrix_ladder.{csv,png}` — per-rung persistence across all
  12 source→target pairs, per dataset. Monotone collapse: naming 0.92 → prompting
  ~0.45 → SFT 0.37 → DPO 0.15.

## The headline result — model-dependent DPO erasure
- `d2_multiseed_ci.csv` — **D2**: per-cell persistence, mean ± 95% CI over 3 adapter
  seeds (de-confounded). Seed-sd median 0.018. The two-tier source split: nemotron
  0.211 / gpt-oss 0.190 **retain**, qwen 0.077 / llama 0.012 **launder**. Survivors
  (6–7/36, all gpt-oss/nemotron-sourced, Fisher p≈0.008).
- `fig1_source_fingerprint.png` — DPO persistence per source model (two-tier split).

## Robustness / corrections
- `d1_decontam_before_after.csv` — **D1**: before/after the gpt-oss chat-template
  CoT-leak fix. corr(leak-rate, Δpersistence) = −0.99; clean cells untouched
  (mean|Δ|=0.002). The leak inflated the raw arena survivors (0.84 → ~0.46).
- `d3_encoder_swap_bge.csv` — **D3**: persistence re-scored under a different encoder
  (bge-small). Rung ordering replicates (Spearman ~0.70 vs MiniLM).
- `source_distinctiveness.csv` — baseline distinctiveness of each model under 4
  measures (MiniLM, bge, mpnet, 32 structural features) vs DPO persistence.
  **Caveat:** the distinctiveness↔persistence link is *encoder-sensitive* — it holds
  under MiniLM and bge (ρ=1.0) but NOT under mpnet (ρ=−0.2) or structural features
  (ρ=−0.3). So "distinctive models resist DPO" is suggestive, not the load-bearing
  mechanism; the two-tier result itself is robust.

## Which behavioural directions carry the fingerprint
- `style_directions.csv` / `big5_personality_directions.csv` — gpt-4.1-mini logprob
  projection onto named style vs Big-Five axes. Big-Five personality barely separates
  the models (|d| 0.12–0.17); **style axes verbosity (0.26) and structure (0.21)**
  separate most. Fingerprint is structural-style, not personality.

See `docs/strategy.md` for the authoritative framing and `docs/evaluation_framework.md`
for the metric definition.
