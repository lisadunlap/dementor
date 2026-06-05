# D1 — Capability vs Style Dissociation

**Status:** plan only (no generation/training run here)
**Author direction:** D1 (CAPABILITY vs STYLE DISSOCIATION)
**Branch:** ethan

---

## Claim

Through the disguise/imitation ladder (`just_name_it → random_sampling → stylistic → sft → dpo`),
**task capability transfers from the imitated model independently of style.** A style-based
provenance audit therefore returns a false "clean" (style persistence ≈ 0, i.e. the source's
style is gone) on exactly the cells where the valuable thing — task competence — was lifted.
**Style is the decoy; capability is the payload.**

### Direction / semantics (verified against the repo — read carefully, the naming is a trap)

`run_cell_pipeline.py` (lines 101–147) trains the disguise as: `base_model = source`, LoRA
`model_path = adapter`, where the adapter is trained on `tgt_train` (the **target**'s outputs).
So in every cell `{source}_to_{target}`:

- **The imitator B runs on the `source` base model.** `gen/source_seed*.csv` = B's *starting
  point* (source base model's own outputs).
- **The model being imitated is `target` (T).** `gen/target_seed*.csv` = T's outputs (the
  style/behavior being copied).
- `gen/rung_*.csv` = the source base model wearing the rung-`r` disguise adapter, trying to
  look like T.

The behavioral-inertia `persistence` (in `results/matrix_ladder/gsm8k_matrix_ladder.csv`)
measures **retention of the source/base style**: self-baseline (source imitating itself)
anchors persistence ≈ 1; identity-control (a fresh target draw) anchors ≈ 0
(`behavioral_cell_evaluator.py` lines 46–47, 239–240). So **persistence → 0 means the imitator
has shed the source's style and moved onto the target's style** — that is the "fingerprint
erased / audit returns clean" condition.

This makes the capability axis the natural dual: as the imitator (source base) is trained to
imitate T, **does its gsm8k accuracy move from `acc_source` toward `acc_target`** even when
persistence has collapsed to ~0? If yes, capability rode along while style "laundered."

> ⚠️ Naming caution for implementation: in this plan "A / source / base / imitator-start" all
> refer to `source`; "T / target / imitated" refer to `target`. Capability *target* of transfer
> is the **`target` model**, and capability *origin* is the **`source` model** (the imitator's
> own base). Do not confuse with the style framing where the "fingerprint" is the source's.

---

## Hypotheses

- **H1 (capability transfer exists & co-occurs with style collapse).**
  For gsm8k, as we ascend the ladder, imitator accuracy `acc_B(rung)` moves from `acc_source`
  toward `acc_target`. There exist cells where, at the DPO rung, **style persistence ≈ 0
  (audit "clean") yet capability transfer is materially > 0** — the false-clean signature.

- **H2 (dissociation — the headline statistic).**
  Per-cell **capability-transfer at the DPO rung is NOT governed by the same two-tier,
  source-dependent pattern that style erasure is.** Operationally: the Pearson/Spearman
  correlation between per-cell `cap_transfer_dpo` and per-cell `style_persistence_dpo` is weak
  / not significantly positive (we predict |r| small or even negative-leaning, mirroring the
  known distinctiveness inversion r≈−0.31). The *dissociation itself* is the result: knowing a
  cell's style persistence tells you little about whether competence transferred.

- **H3 (optional, quality/preference axis — COST-GATED).**
  On chatbot_arena, imitator-vs-target win-rate under an LLM judge (gpt-4.1-mini) shows the
  same dissociation from style persistence: quality/preference can transfer toward the target
  while style persistence collapses. (No human votes exist in the cached arena prompts — see
  Data — so this axis requires paid judging.)

Dataset mapping (as assigned, and confirmed cached):
- **gsm8k = capability** (verifiable exact-match correctness; gradeable from cached text).
- **chatbot_arena = quality/preference** (needs LLM judge; no ground truth/votes cached).
- **writingprompts = pure style** (already measured by the matrix; used only as the style-only
  reference axis, no capability metric).

---

## Data & what's cached vs needs-generation

### Cached and sufficient (no spend) — gsm8k capability core

Per cell `data/results/decontam/gsm8k/{source}_to_{target}/gen/` (all **12** cross pairs
present, each with the full set):

| file | meaning | rows |
|---|---|---|
| `source_seed1.csv`, `source_seed2.csv` | imitator base (source) outputs = `acc_source` | 200 |
| `target_seed1.csv`, `target_seed2.csv` | imitated model (target) outputs = `acc_target` | 200 |
| `rung_just_name_it.csv` | imitator @ rung 1 | 200 |
| `rung_random_sampling.csv` | imitator @ rung 2 | 200 |
| `rung_stylistic.csv` | imitator @ rung 3 | 200 |
| `rung_sft.csv` | imitator @ rung 4 | 200 |
| `rung_dpo.csv` | imitator @ rung 5 | 200 |

- Schema of every gen CSV: **`prompt, model_response`** (verified).
- All 200 prompts per file are identical across rungs/source/target and are the **GSM8K test
  split** (verified: 200/200 overlap with `data/datasets/gsm8k/gsm8k_test.csv`, 0 overlap with
  train). Prompt-keyed joins are exact.

Style persistence to join against (cached, no recompute):
- `results/matrix_ladder/gsm8k_matrix_ladder.csv` — columns
  `source,target,rung,persistence,persistence_sd,anchored,n_seeds,trustworthy,order`.
  12 (source,target) pairs × 5 rungs (`dpo, just_name_it, random_sampling, sft, stylistic`).
  Use **`persistence`** as the primary style axis; `anchored` is the rescaled-to-anchors
  variant (report both, prefer `persistence` for the headline to match the existing paper).
- Style-only reference axes (for the dissociation triangle, optional):
  `results/matrix_ladder/writingprompts_matrix_ladder.csv` (pure style),
  `results/matrix_ladder/chatbot_arena_matrix_ladder.csv`.

### NOT cached — must be derived (free, local, no API)

- **GSM8K test-split gold answers.** `gsm8k_test.csv` has only `prompt,split` (no answer);
  gold `#### N` lives only in `gsm8k_train.csv`. The 200 eval prompts are all test-split, so
  gold must come from the canonical HuggingFace `gsm8k` (config `main`) **test** split. The
  `datasets` library is installed and importable in `./.venv` (verified). This is a free,
  offline-cacheable download, **not** an API/compute spend. Build a `prompt → gold_int`
  lookup once and persist it as `data/datasets/gsm8k/gsm8k_test_gold.csv`.
- **An exact-match grader.** None exists. `scripts/scorer.py` is an LLM *quality/pairwise*
  judge, not a math grader; `scripts/methods/utils/math_style_extractor.py` extracts *style*
  features, not the final numeric answer. We must add a small deterministic
  `final-answer extractor + exact-match` (no model calls).

### NEEDS GENERATION / SPEND (gated, do not run without approval)

- **H3 only:** chatbot_arena pairwise win-rate under gpt-4.1-mini. ~12 cells × (5 rungs +
  source-vs-target baseline) × 200 prompts pairwise judgments. This is paid OpenAI usage. See
  Cost gates. **H1/H2 do not depend on this.**
- No model training or sampling is required for H1/H2 — all imitator outputs already exist on
  disk. (Optional robustness: re-grade `source_seed2`/`target_seed2`/multiple adapter seeds if
  present elsewhere, but the decontam tree ships seed1+seed2 for source/target and a single
  adapter seed per rung — adequate for a first result.)

---

## Method (step-by-step)

All paths absolute under repo root `/Users/EthanLiu/Documents/Programming/dementor`. Run with
`./.venv/bin/python`.

### Step 0 — Build the gold lookup (one-time, free)
1. `from datasets import load_dataset; ds = load_dataset("gsm8k", "main", split="test")`.
2. For each row, normalize the question text the same way the prompts are stored (strip; the
   stored prompts are the raw GSM8K question, verified) and extract gold as the integer/decimal
   after the final `####`.
3. Write `data/datasets/gsm8k/gsm8k_test_gold.csv` with columns `prompt, gold` (gold as a
   normalized numeric string). Assert ≥ 199/200 of the cell prompts join (allow for unicode
   apostrophe normalization — the dataset uses curly `’`; normalize both sides).

### Step 1 — Final-answer extractor + exact-match grader (deterministic, free)
1. Implement `extract_pred(text) -> Optional[str]`:
   - Strip `$ , %` and thousands separators; collapse whitespace.
   - Prefer the GSM8K convention `#### <num>` if the model emitted it; else take the **last**
     number in the text (regex `-?\d[\d,]*\.?\d*`), matching standard GSM8K eval practice;
     also catch "answer is X" / "= X" patterns as a tie-breaker.
   - Normalize to a canonical numeric form (cast to float, compare with tolerance 1e-6;
     fall back to string match for non-numeric).
2. `is_correct(pred, gold)` = numeric-equal within tolerance.
3. Unit-sanity: spot-check 10 graded items per of 2 cells by eye (write a tiny audit CSV of
   `prompt, model_response_tail, pred, gold, correct`).
4. (Optional hardening, cheap) compare extractor accuracy on `target` outputs against any
   existing public number for these models if available; not required for the relative claim.

### Step 2 — Grade every condition per cell (free)
For each of the 12 gsm8k cells and each condition in
`{source_seed1, target_seed1, rung_just_name_it, rung_random_sampling, rung_stylistic,
rung_sft, rung_dpo}` (and seed2 for source/target where present):
1. Join `gen/<cond>.csv` to the gold lookup on normalized `prompt`.
2. Compute accuracy = mean(`is_correct`) over the 200 items.
3. Emit a long CSV row `dataset, source, target, condition, rung_or_role, n, n_correct, acc`.
4. Also emit per-item `correct` (for paired bootstrap / McNemar later).

Define:
- `acc_source` = accuracy of `source_seed*` (imitator base; average over available seeds).
- `acc_target` = accuracy of `target_seed*` (imitated model; average over seeds).
- `acc_B(rung)` = accuracy of `rung_<rung>`.

### Step 3 — Capability-transfer score per rung/cell (free)
Per the assignment's definition, with explicit clipping and a degenerate-gap guard:

```
gap        = acc_target - acc_source
cap_xfer(rung) = (acc_B(rung) - acc_source) / gap        # fraction of the A→T gap closed
cap_xfer    = clip(cap_xfer, 0, 1)                         # report clipped headline
```

- Sign convention: `cap_xfer ≈ 0` → imitator stayed at its own (source) competence;
  `cap_xfer ≈ 1` → imitator reached the imitated model's competence.
- **Degenerate-gap guard:** when `|gap| < δ` (δ chosen from the per-item bootstrap CI of the
  gap; default 0.05 absolute accuracy), `cap_xfer` is undefined — flag the cell `gap_ok=False`
  and **exclude it from the H2 correlation**, but still report its raw `acc_B(rung)` curve.
  Report how many of 12 cells survive the gap filter (this is itself a finding — capability
  transfer is only *measurable* where source and target differ in competence).
- Keep BOTH a directional/unclipped `cap_xfer_raw` (can be <0 if the imitator gets *worse*, or
  >1 if it overshoots) and the clipped headline; overshoot/regression is interesting.
- Also report the absolute `Δacc = acc_B(dpo) − acc_source` so the claim does not hide behind a
  ratio when the gap is small.

### Step 4 — Join capability to cached style persistence (free)
1. Load `results/matrix_ladder/gsm8k_matrix_ladder.csv`.
2. Inner-join on `(source, target, rung)` to the capability table.
3. Produce `d1_capability_vs_style_percell.csv` with, per (source,target,rung):
   `acc_source, acc_target, acc_B, cap_xfer_raw, cap_xfer, delta_acc, persistence,
   persistence_anchored, gap_ok, trustworthy`.

### Step 5 — H1 test (false-clean signature)
1. Restrict to `rung == dpo` (the rung that erases the style fingerprint).
2. Identify cells where `persistence_dpo ≤ τ_clean` (τ_clean = 0.10, the identity-control
   anchor neighborhood; also report at 0.05) **and** `cap_xfer_dpo ≥ 0.25` (≥¼ of the gap
   closed) with `delta_acc` significantly > 0 by paired bootstrap. Each such cell is a
   "false-clean / payload-lifted" instance. Report the list and count out of the gap-ok cells.
3. Plot per-cell ladder curves: x = rung order (`just_name_it…dpo`), twin y-axes of
   `acc_B` (with `acc_source`/`acc_target` reference lines) and `persistence`. The visual is
   "style line dives to 0 while accuracy line climbs toward target."

### Step 6 — H2 test (dissociation statistic — headline)
1. On gap-ok DPO cells, compute Pearson r and Spearman ρ between `cap_xfer_dpo` and
   `persistence_dpo` (and separately vs `persistence_anchored`). Report r, ρ, p, n, and a
   bootstrap 95% CI on r (resample cells).
2. Cross-check against the known two-tier *style* split (gpt-oss/nemotron RETAIN ≈0.19–0.21;
   qwen/llama LAUNDER ≈0.01–0.08; 7/3 survivors). Tabulate `cap_xfer_dpo` grouped by the
   style-survivor tier of each cell; test whether capability transfer differs between
   "style-retained" and "style-laundered" cells (Mann–Whitney U). **Prediction / desired
   result:** capability transfer is statistically indistinguishable between the two style
   tiers — i.e. the thing that predicts style survival does NOT predict capability transfer.
   That null *is* the dissociation.
3. Secondary: correlate `cap_xfer_dpo` with source distinctiveness
   (`results/source_distinctiveness.csv`) to show capability transfer also fails to track the
   distinctiveness signal (parallels the r≈−0.31 style inversion).

### Step 7 — H3 (optional, GATED) chatbot_arena quality transfer
Only if the cost gate is approved:
1. Reuse `scripts/scorer.py::score_pairwise_dataframe` (judge default `openai/gpt-4.1-mini`).
   Honor the `.env` gotcha: read the real `OPENAI_API_KEY`/base from `.env` explicitly and
   **unset/override** the proxy `OPENAI_BASE_URL=https://pass.wafer.ai/v1` before judging.
2. For each arena cell, for each rung and for the source-vs-target baseline, build pairwise
   `(imitator_response vs target_response)` judgments → win-rate of imitator looking like /
   matching the target's quality. Define `qual_xfer(rung)` analogously to `cap_xfer`
   (win-rate of B against the source-baseline normalized by the source→target win-rate gap).
3. Repeat Steps 4–6 on the arena join (`chatbot_arena_matrix_ladder.csv`).

### Step 8 — Writingprompts as the "pure style" corner (free)
No capability metric exists or is meaningful for writingprompts. Use its
`writingprompts_matrix_ladder.csv` persistence only as the third corner of a 3-dataset figure
illustrating: gsm8k carries a measurable capability payload, writingprompts does not, yet style
persistence behaves similarly — reinforcing that style audits are blind to payload presence.

---

## Metrics / statistics

- **Primary metric:** per-cell `cap_xfer_dpo` (clipped) and `delta_acc_dpo`.
- **Headline dissociation statistic (H2):** Pearson r + Spearman ρ between `cap_xfer_dpo` and
  `persistence_dpo` across gap-ok cells, with bootstrap 95% CI and p-value. Mann–Whitney U of
  `cap_xfer_dpo` across the style-survivor vs style-launderer tiers.
- **Significance of transfer (H1):** per-cell paired bootstrap (resample the 200 items) /
  McNemar on `correct(B_dpo)` vs `correct(source)` to test `delta_acc > 0`; BH-FDR across the
  12 cells.
- **Robustness:** repeat the headline correlation with `persistence_anchored`, with the
  unclipped `cap_xfer_raw`, and restricting to `trustworthy == True` cells.
- **Effect sizes** reported alongside p-values; n is small (≤12 cells) so emphasize CIs and
  the qualitative dissociation over thin significance claims.

---

## Deliverables

CSVs (under `results/` to match the curated hand-off convention):
- `data/datasets/gsm8k/gsm8k_test_gold.csv` — prompt→gold lookup (intermediate, free).
- `results/d1_gsm8k_accuracy_long.csv` — per cell × condition accuracy (`acc_source/target/B`).
- `results/d1_gsm8k_peritem_correct.csv` — per-item correctness for bootstrap/McNemar.
- `results/d1_capability_vs_style_percell.csv` — the H1/H2 join (cap_xfer, persistence, flags).
- `results/d1_dissociation_stats.csv` — r/ρ/p/CI, tier Mann–Whitney, gap-ok counts, survivor
  list. The single-number headline (the dissociation r and the false-clean count) lives here.
- (H3, gated) `results/d1_arena_quality_vs_style_percell.csv`,
  `results/d1_arena_winrate_long.csv`.

Figures:
- `results/d1_fig_dissociation_scatter.png` — x=`persistence_dpo`, y=`cap_xfer_dpo`, points
  colored by source model; annotated with r/ρ. **The money figure.**
- `results/d1_fig_ladder_curves.png` — small-multiples (12 cells) of accuracy-vs-persistence
  up the ladder.
- `results/d1_fig_tier_box.png` — `cap_xfer_dpo` by style-survivor tier (boxplot, shows the
  null/no-difference).
- (free corner) `results/d1_fig_three_datasets.png` — gsm8k(capability) / arena / writingprompts
  persistence side-by-side.

Headline claim sentence to instantiate: *"Across N gap-ok cells, DPO drives style persistence
to ≈0 while capability transfer reaches X% of the source→target competence gap; the two are
uncorrelated (r=__, p=__) — a style audit returns clean on K/N cells where competence was
demonstrably lifted."*

---

## Reuse map

- **Gen CSVs (inputs):** `data/results/decontam/gsm8k/{source}_to_{target}/gen/*.csv` — reuse
  directly; schema `prompt,model_response`.
- **Style persistence (no recompute):** `results/matrix_ladder/{gsm8k,chatbot_arena,
  writingprompts}_matrix_ladder.csv`.
- **Style-survivor tiers / distinctiveness:** `results/source_distinctiveness.csv`,
  `data/results/structural_decomp_percell.csv`, and the existing matrix findings (MEMORY:
  matrix-eval-findings) for the 7/3 survivor split.
- **Cell semantics / model slugs:** `scripts/analysis/run_cell_pipeline.py`
  (`SLUG_TO_MODEL`, source=base, target=imitated), `cell.json` per cell.
- **LLM judge (H3 only):** `scripts/scorer.py::score_pairwise_dataframe` /
  `score_pairwise` (default judge `openai/gpt-4.1-mini`). Mind the `.env` proxy override.
- **Robust CSV read:** `scripts/analysis/behavioral_cell_evaluator.py::read_csv_robust`
  (multiline responses are common — reuse it instead of bare `pd.read_csv` for the gen files).
- **New code to add (small, deterministic, no model calls):** `scripts/analysis/grade_gsm8k.py`
  (gold builder + final-answer extractor + exact-match + per-cell grading) and
  `scripts/analysis/d1_capability_vs_style.py` (join + stats + figures).

---

## Cost gates

- **H1 + H2 (the core result): $0.** Pure local compute over cached text + a free HF dataset
  download for gold. **Runnable now from cache.** No approval needed.
- **H3 (arena quality transfer): PAID — GATE.** gpt-4.1-mini pairwise judging over ~12 cells ×
  ~6 conditions × 200 prompts ≈ 14k judge calls (plus a source-vs-target baseline). Rough order
  ~$5–20 depending on prompt+response token lengths and whether we judge both seed draws.
  **Do not run without explicit approval.** Mitigations if approved: cap at 100 prompts/cell
  first; judge only DPO rung + baseline before fanning out to all rungs; cache judge outputs.
- **No training / no sampling spend anywhere** — all imitator generations already exist.

---

## Risks

- **Small-gap cells (R1):** if `acc_source ≈ acc_target` for many of the 12 pairs (models are
  similarly competent on gsm8k), the `cap_xfer` ratio is undefined/noisy. Mitigation: the
  gap-ok filter (Step 3) and reporting `delta_acc` directly; report how many cells survive.
  This is partly an *empirical* risk — if most cells have small gaps, H1/H2 weaken to "where a
  gap exists, capability transfers." Worth front-loading a quick gap census.
- **Extractor error (R2):** last-number heuristics can misgrade verbose chain-of-thought,
  especially gpt-oss/nemotron which emit reasoning scaffolding. Mitigation: prefer `####`,
  then "answer is/=", then last-number; manual audit of 10/cell; the claim is *relative*
  (B vs source/target graded by the *same* extractor), so systematic extractor bias largely
  cancels.
- **Reasoning-format confound (R3):** if the imitator learns the target's *answer formatting*
  (e.g. emitting `####`) without learning to reason, accuracy could move for non-capability
  reasons. Mitigation: report accuracy *and* a format-stripped re-grade; note that for the
  audit-evasion claim, "competence-looking output" is itself the lifted asset, so format
  transfer is arguably in-scope — discuss explicitly rather than hide.
- **Persistence join mismatch (R4):** matrix_ladder rung labels must align to gen filenames
  (`rung_<x>` ↔ `rung==<x>`); both verified to use `dpo,just_name_it,random_sampling,sft,
  stylistic`. Assert 12×5 = 60 joined rows; fail loudly otherwise.
- **Single adapter seed per rung (R5):** decontam tree ships one adapter seed per rung (source/
  target have seed1+2). Capability point estimates per rung are single-seed; the matrix
  persistence already aggregates seeds (`n_seeds`). Report this asymmetry; if multi-seed rung
  gens exist elsewhere (`data/results/decontam_s3/`), fold them in for a seed-robustness check.
- **Prompt-join unicode (R6):** GSM8K uses curly apostrophes; normalize both sides before
  joining gold (verified the join works at 200/200 when keyed on the test split).

---

## Open questions

1. Is the gap large enough on enough of the 12 pairs for `cap_xfer` to be meaningful, or does
   the result reduce to "capability transfers where a competence gap exists"? (Run the gap
   census first.)
2. Should the headline grade *only* the DPO rung (the audit-relevant one) or the full ladder?
   Plan reports the full ladder but anchors the claim on DPO.
3. For H3, is win-rate-vs-target the right "quality transfer" operationalization, or should we
   judge absolute quality (1–5) of B and measure movement toward the target's absolute score?
   (Absolute scoring avoids conflating "matches target" with "is good.")
4. Do we want a capability-specific *audit* baseline — i.e. show that a competence probe
   (accuracy) flags these cells as "lifted" exactly where the style audit says "clean,"
   making the false-clean concrete as two side-by-side audit verdicts?
5. Should `decontam_s3/gsm8k` (a 3rd-seed tree) be incorporated for seed-robust capability
   estimates, and does it carry the same rung gen files?
