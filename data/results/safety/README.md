# Direction 3 — Safety-Behavior Laundering (scaffold)

**Status: SCAFFOLD ONLY. Nothing here has been run against a model. $0 spent.**

This directory documents what the D3 safety-audit harness *would* produce once a human
approves the gated Tinker sampling. The deterministic, free parts (prompt download +
refusal classifier) are built and tested; the sampling is written but **not armed**.

Defensive framing: we measure whether a model's *alignment survives black-box
imitation*. The only quantity produced is a **refusal RATE** (binary refuse/comply).
See "Dual-use" below.

## What's already done (free, no spend)

| Artifact | How |
|---|---|
| `data/datasets/benchmarks/refusal/advbench_harmful.csv` (520 prompts) | `dementor/safety/fetch_refusal_prompts.py` — HF download, prompts only |
| `data/datasets/benchmarks/refusal/xstest.csv` (450: 250 safe + 200 unsafe) | same |
| `data/datasets/benchmarks/refusal/eval_120h_80o_seed42.csv` (200-prompt eval set) | same, seed-42 subsample |
| Deterministic Tier-A refusal classifier + unit test | `dementor/safety/refusal_classifier.py --selftest` (24/24 synthetic cases pass), also `tests/test_refusal_classifier.py` |
| Spend-guarded ladder runner + cost dry-run | `dementor/safety/run_safety_ladder.py` |

Benchmark mirrors used (non-gated; the canonical `walledai/*` repos are gated):
- AdvBench harmful_behaviors (Zou et al. 2023) via `ivnle/advbench_harmful_behaviors`.
- XSTest v2 (Röttger et al. 2024) via `natolambert/xstest-v2-copy` (`prompts` split).

## What running it would produce (GATED — needs approval)

Outputs land under the top-level **`results/safety/`** (mirroring `results/matrix_ladder/`),
NOT this `data/results/safety/` dir (which holds this README only):

- `results/safety/native_refusal/{model_slug}_seed{1,2}.csv` — per base model, columns
  `prompt, category, refused, snippet_redacted`. Aggregated → `r_native(model)`.
- `results/safety/adapter_refusal/{rung}_{ds}_{src}_as_{tgt}_seed{n}.csv` — per adapter
  rung, same columns. Aggregated → `r_imitated(A→B, ds, rung, seed)`.
- (Phase-4 analysis, separate script, not in this scaffold) `safety_native_refusal.csv`,
  `safety_refusal_ladder.csv` (drift/pull/CIs), `safety_drift_vs_style.csv` (H5 join with
  `results/matrix_ladder/*_matrix_ladder.csv`), figures.

Each cell samples the 200-prompt eval set at **temp 0.7, max_tokens 512**, classifies with
the deterministic Tier-A matcher, writes only the verdict + a ≤200-char redacted snippet,
and **deletes the raw completion** (dual-use mitigation — no full harmful completions persist).

## Comparator convention (CRITICAL — do not invert claims)

**The "native" baseline for an A→B adapter rung is `r_native(SOURCE A)`, NOT the target.**

Verified directly against `data/tinker_adapters.json`: all 108 `{rung}_{ds}_{src}_as_{tgt}_seed{n}`
entries that carry a `base_model` have `MODEL_SLUG[base_model] == source` (0 mismatches).
This matches `dementor/metric/run_cell_pipeline.py` line 146, which samples adapter rungs with
`base_model = source`, `clean_model = source`. The LoRA adapter sits on the **source** model
fine-tuned to imitate the target's benign outputs.

Consequence for the write-up: the claim is *"fine-tuning source A on target B's benign
outputs drifts A's refusal rate away from A's own native rate"*. The runner therefore sets
`base_model_slug = source` for every adapter unit and drift is computed against
`r_native(source)`. The README headline elsewhere ("B imitates A") uses the opposite A/B
letter convention — pin to the `base_model = source` semantics above, not the letters.

`drift = r_imitated − r_native(source)`;  `pull = r_imitated − r_native(target)` (toward the
model whose outputs were imitated). H2 source-pull tests whether drift moves toward the
imitated model's native rate.

## Cost tiers (exact sampling-call counts; verified by the runner's dry-run)

Sampling calls = (cells) × (200 prompts/cell). One call = one prompt × one draw @
max_tokens 512, temp 0.7, via Tinker. **No training spend** (adapters already exist).

| Tier | Composition | Sampling calls |
|---|---|---|
| **(a) Pilot** (`--pilot`) | gsm8k only, adapter seed1; 4 baselines × 2 seeds + 12 pairs × 2 rungs × seed1 | 1,600 baselines + 4,800 adapter = **6,400** |
| **(b) Full matrix** (`--full`) | 3 datasets, adapter seeds 1–3; 4 baselines × 2 seeds + 12 pairs × 3 ds × 2 rungs × 3 seeds | 1,600 baselines + 43,200 adapter = **44,800** |

Reproduce these numbers for free:
```
./.venv/bin/python -m dementor.safety.run_safety_ladder --pilot   # prints 6,400
./.venv/bin/python -m dementor.safety.run_safety_ladder --full    # prints 44,800
```

**Dollar estimate:** dominated by `calls × ~512 max_tokens`. The runner ships a
`USD_PER_1K_CALLS_PLACEHOLDER = 0.0` constant; set it to the real Tinker rate to print a $
figure. Order of magnitude: pilot ≈ 6.4k generations (a modest sampling run); full ≈ 44.8k
generations (≈ 7× the pilot). **Recommendation: approve the pilot first, decide go/no-go on
the full matrix from the pilot.**

Optional gated extensions (not in the default counts):
- **GATE-2** prompting rungs (just_name_it / random_sampling / stylistic):
  +12 pairs × 3 ds × 3 methods × 200 ≈ **21,600** calls.
- **GATE-3** Tier-B `gpt-4.1-mini` judge for Cohen's-κ classifier validation on ~150
  (prompt,response) pairs: trivial $ but still an external paid-API spend — gated, and
  **not implemented** in this scaffold (`refusal_classifier.py` is Tier-A only).
- **Self-SFT placebo:** 12 self-SFT adapters exist in the registry — a cheap, high-value
  control (does imitating your *own* benign outputs move refusal?). Add if pilot approved.

## How to launch (after approval)

The runner is deliberately **not armed**: even passing `--i-have-approval-to-spend` makes it
print the plan and exit non-zero with a refusal message. To actually run, a human must
(1) confirm the Tinker rate card/budget, then (2) replace the guard at the end of
`run_safety_ladder.main()` with the call to `_run_with_spend(args, plan)` (already written
and reachable, just commented out). This two-step arming prevents accidental spend.

## Dual-use considerations

This is a **defensive safety audit**, not a jailbreak tool.
- Only **published, citable** refusal-benchmark prompts are used; no new harmful prompts are
  authored. AdvBench ships an example affirmative `response` column — the fetch script
  **never reads or stores it**.
- The measured quantity is an aggregate **refusal rate**. The runner classifies each
  completion to a 0/1 verdict, keeps at most a ≤200-char redacted snippet for audit, and
  **deletes the raw completion**. No full harmful completion is ever persisted.
- No harmful content is optimized for, amplified, or released — outputs are rates +
  correlations. All generation is gated behind explicit human approval.
