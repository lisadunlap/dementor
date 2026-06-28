# D2 — Reasoning-Structure Transfer

**Status:** plan only (no generation/training authorized here).
**Branch:** ethan. **Runtime:** `./.venv/bin/python` (editable install; no `PYTHONPATH` needed).

## Claim

Imitation does not transfer a single monolithic "fingerprint." It transfers at
least three separable layers, at **different rates**:

- (a) **surface STYLE** — markdown, bullets, greeting/signoff, sentence length
  (the existing 32-D `_style_*` features and the headline `st_axis` persistence);
- (b) **REASONING-TRACE STRUCTURE** — the *shape* of the chain-of-thought:
  number of reasoning steps, tokens-per-step, enumeration / "Step N" / "Let me"
  scaffolding, equation density, answer-restatement, self-correction markers,
  branching/case-split markers;
- (c) **final-answer CAPABILITY** — gsm8k exact-match correctness.

The gpt-oss chat-template CoT-leak we previously de-confounded (the `decontam/`
generations) is itself evidence that reasoning structure is *highly* transferable:
the imitator picked up the source's scaffolding so faithfully it leaked the raw
template. D2 turns that bug into a measured, falsifiable signal by defining a
**reasoning-structure persistence** that reuses the exact same Fisher-LDA /
projection-persistence machinery the style headline uses, applied to a disjoint
reasoning-feature vector — measured strictly on the **de-confounded** generations.

## Hypotheses

- **H1 (different rate).** Reasoning-structure persistence diverges from style
  persistence along the ladder (`just_name_it → random_sampling → stylistic →
  sft → dpo`). Specifically reasoning structure either (i) transfers *earlier*
  (lower persistence at SFT) or (ii) *persists after style erases* at DPO
  (higher residue). Test: per-rung paired Δ = `persist_reasoning − persist_style`
  is non-zero and rung-ordered.
- **H2 (capability coupling).** Across cells, reasoning-structure transfer
  predicts capability transfer better than style transfer does:
  `|corr(Δreasoning, Δcapability)| > |corr(Δstyle, Δcapability)|` (gsm8k only,
  where capability is gradeable).
- **H3 (two-tier explanation).** The source-dependent DPO survivor split
  (gpt-oss/nemotron RETAIN ~0.19–0.21; qwen/llama LAUNDER ~0.01–0.08; 7/3
  survivors) is explained better by **reasoning-structure durability** of the
  source than by style **distinctiveness**. Test: per-source DPO
  reasoning-persistence rank-correlates with the survivor split more strongly
  (and with the *correct* sign) than baseline style distinctiveness, which is
  known to *invert* (r = −0.31, `distinctiveness_structural.py`).

## Data — cached vs needs-generation

### Cached (no spend) — confirmed by inspection

- 36 de-confounded cells: `data/results/decontam/<dataset>/<src>_to_<tgt>/gen/`
  with `dataset ∈ {gsm8k, writingprompts, chatbot_arena}`, all 12 ordered
  cross-pairs of `{llama-3.1-8b, qwen3.6-27b, gpt-oss-20b, nemotron-nano-30b-a3b}`.
- Each `gen/` dir has, **all 36/36 present**:
  `source_seed1.csv`, `source_seed2.csv`, `target_seed1.csv`, `target_seed2.csv`,
  `rung_just_name_it.csv`, `rung_random_sampling.csv`, `rung_stylistic.csv`,
  `rung_sft.csv`, `rung_dpo.csv`. Columns: `prompt`, `model_response`.
  (`staged/<rung>.csv` adds `source_response`, `target_response` aligned by prompt.)
- **Full CoT is present** in `model_response` (verified: gpt-oss writes compact
  LaTeX derivations; nemotron writes explicit "**Step-by-step reasoning** / 1. /
  2. ..." enumerations — exactly the structural contrast D2 measures).
- Existing **style** persistence per (dataset, source, target, rung, seed-aggregated):
  `data/results/multiseed_ci_s3.csv` (180 rows = 3 datasets × 12 pairs × 5 rungs;
  cols `dataset, source, target, base_rung, mean, sd, n, ci95`).
- Existing per-feature structural decomposition template:
  `dementor/steering/structural_decomp.py` (DPO-only, 32-D style features, 36 cells).
- gsm8k gold: decontam gsm8k uses the **test** split (verified: 200/200 prompts
  match gsm8k test, 0 match the local train CSV that has answers). Gold final
  answers are NOT in `data/datasets/gsm8k/gsm8k_test.csv` (cols `prompt, split`
  only) but ARE recoverable.

### Needs a one-time network fetch (NOT generation/compute spend, ~free)

- **gsm8k gold answers for the test split** (needed only for H2 capability).
  `datasets==4.8.5` is installed. Fetch once:
  `load_dataset("openai/gsm8k","main",split="test")`, join on normalized
  `prompt`, extract gold via regex `####\s*([-0-9.,]+)`. This is a small dataset
  download (no API key, no model inference). If offline-only is required, this
  part (H2) is deferred; H1 and H3 are fully cache-runnable.

### Genuinely needs generation (NOT in scope; cost-gated)

- **None for H1/H3.** No new sampling or LoRA training is required — all rungs
  and both endpoints already exist for all 36 cells. D2 is a pure CPU re-analysis.
- Optional robustness extension (seed2 stability, see Risks) also uses only
  cached `*_seed2.csv` — still no spend.

## Method — step by step

### Step 0 — reasoning-feature extractor (new module)

New file: `experiments/analysis/reasoning_structure.py`. Deterministic, regex/counting
only, CPU, no MiniLM, no API. Mirror the signature style of
`structural_matrix()` in `structural_decomp.py`:
`reasoning_matrix(texts) -> (X: np.ndarray[n, F], names: list[str])`.

Features (kept **disjoint** from the existing 32 style features; reuse nothing
that is pure markdown/greeting). Each is computed per response:

- `rs_step_count` — # explicit enumerated reasoning steps: count lines matching
  `^\s*(\d+[.)]|[-*•]|Step\s*\d+|First|Second|Then|Next|Finally)\b` (de-dup so a
  numbered list of length k = k steps).
- `rs_tokens_per_step` — `len(words) / max(step_count, 1)` (granularity of CoT).
- `rs_step_markers` — count of `Step N` / `Step-by-step` literal markers.
- `rs_letme_markers` — count of first-person planning openers
  (`let me`, `let's`, `we have`, `we need`, `we get`, `note that`, `observe that`).
- `rs_equation_density` — (`# of "="`, `# of LaTeX \[...\] / \(...\)` blocks,
  `# of <<...>>` calc markers) per 100 tokens; captures formal-derivation shape.
- `rs_restatement` — answer-restatement: does a final line re-state the answer
  with a cue (`answer is`, `therefore`, `so the`, `**Answer`, `= \d+ \w+\.$`)?
  binary + count.
- `rs_selfcorrection` — self-correction / backtracking markers
  (`wait`, `actually`, `re-?check`, `let me reconsider`, `that's wrong`,
  `on second thought`, `correction`). count.
- `rs_branching` — case-split / branching markers
  (`case 1`, `if .* then`, `otherwise`, `either`, `alternatively`, `on the other hand`).
  count.
- `rs_connective_density` — logical connectives per 100 tokens
  (`therefore`, `thus`, `hence`, `because`, `since`, `so that`, `which means`).
- `rs_n_equations_total`, `rs_n_numbers` — raw equation and numeric-token counts
  (length-correlated; included raw, residualization handled by z-score + the
  separation gate, exactly as the style basis does).

Determinism + audit: write the feature table for a small sample to
`data/results/reasoning/feature_smoke.csv` and eyeball 3 source/target pairs to
confirm step-count etc. behave (e.g. nemotron > gpt-oss on `rs_step_count`).

### Step 1 — per-cell, per-rung reasoning persistence (H1, H3)

New driver: `experiments/analysis/reasoning_transfer.py`. Reuse the projection-
persistence recipe verbatim from `structural_decomp.per_feature_persistence`,
but (a) over the reasoning matrix, (b) for **every rung**, not just DPO, and
(c) collapsed to a single per-cell scalar via the **Fisher-LDA projection axis**
(the headline metric), not a per-feature mean.

For each of 36 cells and each rung r:

1. Build matrices `src = reasoning_matrix(source_seed1)`,
   `tgt = reasoning_matrix(target_seed1)`, `dis = reasoning_matrix(rung_r)`.
2. Z-score every column on the **pooled source+target** rows only (the basis
   rule from `fit_behavioral_axis_basis` / `structural_decomp`; disguised rows
   never define the measurement).
3. Fit the supervised axis with the existing
   `latent_behavior_axes._fit_supervised_basis(src_z, tgt_z, k=1,
   shrinkage=0.15)` → unit vector `w1` (Fisher-LDA discriminant, shrinkage-
   regularized, identical to the style headline). Using the same `k`/`shrinkage`
   as the cell evaluator keeps reasoning- and style-persistence on the same
   estimator.
4. Headline scalar persistence via
   `behavioral_inertia_metrics.projection_persistence(src_z@w1[:,None],
   dis_z@w1[:,None], tgt_z@w1[:,None])` → `1 − clip(movement,0,1)`. This is the
   rotation-invariant metric used for style, so reasoning vs style are
   apples-to-apples.
5. Also keep the per-feature persistence table (Step-0 features × cells) for the
   decomposition narrative, reusing `structural_decomp` logic unchanged on the
   reasoning matrix.

Output `data/results/reasoning/reasoning_persistence_percell.csv`:
`dataset, source, target, rung, persist_reasoning, movement_reasoning,
axis_separation, n_active_features`.

### Step 2 — align style persistence for paired comparison (H1)

Join Step-1 output to the existing style persistence in
`data/results/multiseed_ci_s3.csv` on `(dataset, source, target, rung)`
(rename `base_rung→rung`, `mean→persist_style`). Compute
`delta_rs_minus_style = persist_reasoning − persist_style` per cell-rung.

To keep the comparison on identical footing (single seed, single estimator),
also recompute a **style** projection-persistence with the same Step-1 pipeline
but the 32-D `_style_*` matrix, and report both the join-to-multiseed version and
the recomputed-identical-pipeline version. The recomputed version is the primary
H1 statistic (removes the seed-count / pipeline mismatch confound:
`multiseed_ci_s3` aggregates 3 seeds for sft/dpo but 1 for the early rungs).

### Step 3 — capability transfer (H2, gsm8k only)

New helper `experiments/analysis/gsm8k_capability.py`:

1. Fetch gold once via `datasets` (cache-gated; see cost gate G2). Build
   `{normalized_prompt: gold_int}`.
2. `extract_pred(text)` — last `\\boxed{}`, else last `####`, else last number in
   the final line; normalize (strip `$`, `,`, trailing `.0`).
3. For each gsm8k cell, grade `source_seed1`, `target_seed1`, and each
   `rung_*` by exact-match → accuracy per (cell, rung, role).
4. Define **capability transfer** at rung r:
   `cap_transfer = (acc_disguised − acc_source) / (acc_target − acc_source)`
   when `acc_target ≠ acc_source` (signed, like persistence movement; clip to
   [−0.5, 1.5] for reporting, flag denominators with `|acc_t−acc_s| < 0.05` as
   low-power and exclude from the headline correlation).

Output `data/results/reasoning/gsm8k_capability_percell.csv`:
`source, target, rung, acc_source, acc_target, acc_disguised, cap_transfer`.

### Step 4 — statistics & hypothesis tests

- **H1:** per-rung mean `delta_rs_minus_style` with bootstrap 95% CI over the 36
  cells; Wilcoxon signed-rank (paired, per cell) at each rung. Headline figure:
  two persistence-vs-rung curves (reasoning, style) with CI bands. Claim direction
  = sign and rung-position of the gap.
- **H2:** at the SFT and DPO rungs (gsm8k cells), Pearson + Spearman of
  `cap_transfer` vs `(1 − persist_reasoning)` and vs `(1 − persist_style)`.
  Report both rs and style correlations + a bootstrap CI on their difference.
  H2 supported iff reasoning correlation is larger in magnitude with non-
  overlapping/one-sided CI.
- **H3:** per-source DPO `persist_reasoning` (mean over targets×datasets, mirror
  `structural_decomp` per-source aggregation). Rank the 4 sources; compare to the
  survivor split (gpt-oss/nemotron high, qwen/llama low). Report Spearman vs the
  per-source DPO style persistence in `multiseed_ci_s3` AND vs the inverting
  baseline distinctiveness from `distinctiveness_structural.py` (r=−0.31). H3
  supported iff reasoning-persistence orders the survivors correctly with a
  stronger, correctly-signed rho than distinctiveness.

All correlations use n=4 (sources) for H3 — explicitly under-powered; report as
**rank-consistency / direction**, not significance (same epistemic standard the
existing `distinctiveness_structural.py` uses). H1/H2 use n=36 / n≈12 cells.

## Metrics / statistics

- Primary metric: Fisher-LDA **projection persistence** on the reasoning matrix,
  `1 − clip(<dis−src, w>/<tgt−src, w>, 0, 1)`, identical estimator to style.
- Separation gate: drop cells/features where `|axis_separation| < 0.3` (Cohen's d
  on z-scored features) — same `SEP_GATE` as `structural_decomp`. A reasoning
  axis that doesn't separate source from target carries no signal.
- Paired Δ statistics with bootstrap CIs (n=36) and Wilcoxon signed-rank.
- Capability: gsm8k exact-match accuracy + signed `cap_transfer` fraction.
- Report Spearman (rank) everywhere n is small; Pearson where n≥12.

## Deliverables

CSVs (under `data/results/reasoning/`):
- `reasoning_persistence_percell.csv` — 36 cells × 5 rungs, reasoning persistence.
- `reasoning_vs_style_percell.csv` — joined reasoning/style + `delta_rs_minus_style`.
- `reasoning_feature_decomp.csv` — per-feature persistence (which CoT-shape
  features carry the survivor residue), mirroring `structural_decomp` output.
- `gsm8k_capability_percell.csv` — accuracy + capability transfer (gated on G2).
- `reasoning_hypothesis_summary.csv` — one row per hypothesis with the headline
  stat + CI + verdict.

Figures (under `figures/reasoning/`):
- `fig_rs_vs_style_ladder.png` — persistence-vs-rung curves (reasoning vs style)
  with CI bands (H1).
- `fig_cap_vs_transfer_scatter.png` — capability transfer vs reasoning-transfer
  and vs style-transfer, gsm8k SFT/DPO (H2).
- `fig_persource_reasoning_survivors.png` — per-source DPO reasoning persistence
  bars, survivors highlighted, vs the inverting distinctiveness baseline (H3).

## Reuse map

- `dementor/steering/structural_decomp.py` — `structural_matrix`, the z-score-on-
  source+target basis rule, `per_feature_persistence`, `load_all_cells`,
  `SEP_GATE`, `SHORT`, the per-source aggregation. **Primary template**; copy its
  cell-loop and per-feature logic, swap the feature matrix for the reasoning one.
- `dementor/metric/latent_behavior_axes.py` — `_fit_supervised_basis(...,
  k=1, shrinkage=0.15)` for the Fisher-LDA axis; `_style_scalar_features`,
  `_style_binary_features` for the recomputed-identical-pipeline style baseline.
- `dementor/metric/behavioral_inertia_metrics.py` — `projection_movement`,
  `projection_persistence`, `projection_movement_per_row`, `axis_separation`.
- `experiments/analysis/distinctiveness_structural.py` — pattern for the per-source
  distinctiveness baseline H3 compares against (and its r=−0.31 inversion).
- `data/results/multiseed_ci_s3.csv` — cached style persistence to join (H1) and
  the per-source DPO style numbers (H3).
- `data/datasets/gsm8k/gsm8k_test.csv` + `datasets` lib — gsm8k prompts + gold.

## Cost gates

- **G0 (no spend, default-allowed):** H1 and H3 are pure CPU re-analysis of
  cached decontam generations. No API, no LoRA, no GPU, no Tinker. Runtime is
  minutes. **Runnable now from cache.**
- **G1 (no spend):** reasoning-feature extraction, projection persistence,
  per-source aggregation, all figures except H2 scatter.
- **G2 (tiny, ~free, network only — confirm before running):** one-time
  `datasets.load_dataset("openai/gsm8k","main",split="test")` download (a few MB,
  no API key, no model inference) to obtain gold answers for H2 capability. This
  is NOT model-generation spend, but it touches the network — gate it. If denied,
  ship H1+H3 and mark H2 deferred.
- **NO cost gate is needed for any generation/training** — none is required.
  This direction explicitly does not request inference or training spend.

## Risks

- **R1 — reasoning features correlate with length/style.** Step-count and
  equation-density scale with response length, which the style word-count feature
  also captures, so "different rate" could be a length artifact. Mitigation: the
  z-score + separation gate + Fisher-LDA *within-class whitening* already
  down-weights pure length; additionally report a length-residualized variant
  (regress each reasoning feature on `log1p(word_count)` before the axis, reusing
  `latent_behavior_axes._residualize_against_length`) and require H1 to survive it.
- **R2 — overlap with existing style features** (`style_numbered_count`,
  `style_reasoning_markers`, `contains_numbered_steps`). Mitigation: keep the
  reasoning matrix **disjoint** (deeper counts: tokens-per-step, self-correction,
  branching, restatement — none of which exist in `_style_*`), and report the
  feature-set correlation matrix to document residual overlap.
- **R3 — CoT-leak reintroduction.** Working only on `decontam/` (the de-
  confounded gens) avoids the chat-template leak. Mitigation: assert no row
  contains raw template tokens (`<|channel|>`, `<|message|>`, `analysis<|`),
  abort if found; never read from non-decontam `data/results/<dataset>/`.
- **R4 — gsm8k answer extraction noise.** Models format answers differently
  (`\boxed`, `####`, bold). Mitigation: multi-pattern extractor + manual audit of
  20 mismatches per source; report extraction-failure rate; exclude unparseable
  rows symmetrically across roles.
- **R5 — single-seed estimate.** Persistence uses `*_seed1` only. Mitigation:
  recompute on `*_seed2` (cached) and report seed-pair agreement; the headline
  style metric this compares to is multi-seed for sft/dpo, so flag the asymmetry.
- **R6 — n=4 for H3.** Under-powered. Mitigation: present as rank-consistency vs
  the *known-inverting* distinctiveness baseline (a direction claim, not p-value),
  exactly as `distinctiveness_structural.py` frames its r=0.97/−0.31 story.

## Open questions

- Should reasoning persistence be a single Fisher-LDA scalar (headline,
  comparable to style) or a feature-vector profile (richer, but not a single
  number)? Plan computes both; the scalar is the headline.
- For non-gsm8k datasets there is no capability ground truth (writingprompts has
  none; chatbot_arena has human-preference votes, not correctness). H2 is gsm8k-
  only; is a preference-based capability proxy for arena worth adding later?
- Does "reasoning structure transfers earlier" hold per-dataset, or only pooled?
  (writingprompts CoT is sparse — reasoning features may be near-constant there;
  the separation gate will auto-exclude those cells, which is the right behavior
  but shrinks n for H1 on writingprompts.)
- For H3, is per-source reasoning durability better measured at DPO (residue) or
  as the *rung at which reasoning persistence crosses 0.5* (a survival "half-life"
  along the ladder)? The half-life framing may separate the two tiers more cleanly.
