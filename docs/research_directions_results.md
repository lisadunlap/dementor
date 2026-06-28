# Beyond the style-fingerprint result: three directions (2026-06)

The style/behavioral-fingerprint provenance question is now crowded (arXiv:2602.09434,
2603.19022). We spun up three specialists to test directions *beyond* it. Each has a
full plan in `docs/plans/` and an implementation on its own branch off `ethan`.

| Dir | Branch | Question | Verdict (after the Tinker fix pass) |
|-----|--------|----------|---------|
| D1 | `d1-capability-vs-style` | Does *capability* ride along (the payload) while style launders? | **Real on MATH-500** (+0.075 acc transfer at DPO; false-clean existence proof). Null on gsm8k was a ceiling artifact. |
| D2 | `d2-reasoning-structure` | Decompose style vs reasoning-structure vs capability transfer | **Positive (H3)** — reasoning durability ranks the survivor split (pooled +1.00; seed2-alone +0.80) |
| D3 | `d3-safety-scaffold` | Does alignment/refusal survive black-box imitation? | **Refusals erode on all 3 datasets** (full 216-cell matrix): DPO erases 19.4% pooled [CI 13.2–26.1]; coupled to style persistence (r=−0.35) |

> **Update 2026-06-05 (Tinker fix pass).** A bounded ~9k-call Tinker run fixed all three.
> Details in the "Fix pass" section at the bottom; the table above already reflects the
> post-fix verdicts. Initial-pass scripts and results remain on the three branches; the fixes
> are committed on `ethan` (commits `4bc7961` D1, `8a7fecc` D3, `5a36762` D2).

## D1 — Capability vs Style (clean negative)
Cache-only gsm8k grading, all 12 cross-pairs. **Gap census is the load-bearing finding:
only 6/12 cells have a real source→target competence gap** — all four models score ~0.83
on gsm8k (ceiling). Where a gap exists, imitation-tuning to DPO does **not** import the
target's competence; it *erodes* the source's (mean Δacc −0.074; llama→qwen −0.155,
nemotron→gpt-oss −0.210, both p<0.001). H1 false-clean: 0/6. The "capability payload"
thesis is **dead on gsm8k specifically** because there's no capability to lift. Testing it
properly needs a benchmark with real capability spread (hard math / code / MMLU-hard).
Convention pinned: cell `{source}_to_{target}` ⇒ adapter base = source, trained on target.

## D2 — Reasoning-structure transfer (the keeper)
**Headline (H3):** per-source DPO reasoning-persistence ranks the 7/3 style-survivor split
**perfectly** (Spearman **+1.00** vs survivor residue: nemotron .31 > gpt-oss .23 >
qwen .19 > llama .10), beating the recomputed-style baseline (+0.80) and decisively
beating the *inverting* distinctiveness baseline (−0.31). **This supplies the mechanism
for the source-dependence the main paper currently lists as "open":** survivors keep their
derivation/enumeration scaffolding (equations, step_count, connectives); launderers wash
it out. H1 (reasoning persists higher than style at DPO, +0.115) is real but **partly a
length artifact** — attenuates to +0.044 under length-residualization, survives at the
random_sampling rung; reported honestly. H2 capability-coupling is null (same gsm8k
ceiling). Reasoning feature matrix verified disjoint from the 32 style features.

→ **Recommended:** fold D2/H3 into the main paper as the mechanism section (closes a
named limitation). It needs no new spend.

## D3 — Safety-behavior laundering (scaffold, awaiting go/no-go)
Defensive audit: does an imitator trained on a source's *benign* outputs lose the source's
refusal behavior? **Nothing run, $0 spent.** Ready: AdvBench(520)+XSTest(450) prompts
fetched; deterministic refusal classifier (24/24 synthetic + pytest green); spend-guarded
`run_safety_ladder.py` written but un-armed. Comparator pinned: adapter base = source, so
native comparator for an A→B adapter is `r_native(SOURCE)` (108/108 registry entries
verified). **Cost tiers — needs explicit approval:** pilot 6,400 Tinker sample calls
(gsm8k+seed1), full matrix 44,800. Recommendation: approve the pilot, decide from it.
No new training (adapters reload from Tinker).

## Cross-cutting
D1 and D2 *independently* hit the same wall — the **gsm8k capability ceiling** — so any
capability-transfer claim on this dataset is underdetermined. That's a robust, twice-found
result, and it scopes where the capability question can even be asked. The fix pass
escaped it by moving D1's capability test to MATH-500.

## Fix pass (2026-06-05, Tinker, ~9k sampling calls)

A bounded Tinker run resolved all three. (A first workflow attempt spent ~2.6k calls but
two agents died of context exhaustion before reporting; their graded generations were
recovered from disk and the runs resumed idempotently rather than restarted.)

**D1 — capability transfer, MATH-500.** Census: gsm_symbolic still ceiling'd (spread
0.075) but **MATH-500 spreads the models 0.43** (nemotron .83 / gpt-oss .79 / qwen .50 /
llama .40). Re-ran capability transfer there (bases + 12 pairs × {SFT,DPO}, matched
1024-tok/greedy). **Verdict flips from null:** in the payload direction (weak source
imitating a stronger target) imitation transfers **+0.075 acc at DPO** (cap_xfer ≈ 0.20).
**False-clean existence proof:** `qwen→nemotron` at DPO has style persistence **0.0** (audit
reads clean) yet **+21 pts** MATH capability (cap_xfer 0.65). Capability transfer is
~orthogonal to style persistence (r=−0.06, p=0.87) but **underpowered** (n=10 gap-ok cells,
huge CI). Caveat/confound: MATH capability tiers {nemotron,gpt-oss}>{qwen,llama} coincide
with the style-survivor tiers — survivors may partly just be the more capable models.
Files: `results/d1fix_*`, `experiments/analysis/{census_mathbench,grade_mathbench,d1fix_capgen,d1fix_analyze}.py`.

**D3 — safety, FULL 216-cell matrix (3 datasets × 12 pairs × 2 rungs × 3 seeds).**
Native refusal high on all 4 (0.92–1.00). **Benign-output imitation collaterally erases
refusals on every training dataset:** DPO erases gsm8k **14.7%** [CI 7.5–22.9],
writingprompts **24.8%** [12.1–38.5], chatbot_arena **18.6%** [7.9–31.0]; **pooled 19.4%**
[13.2–26.1] (SFT pooled 17.6%). Near-zero seed variance (e.g. wp 0.248/0.249/0.248) → robust,
not noise; writingprompts-trained imitation strips the most refusal. **H5 at full power:**
|refusal drift| correlates *negatively* with style persistence (Pearson **−0.35** at DPO,
n=108) → **not decoupled** as hypothesized — durable-style models keep their refusals,
launderers shed both (durability is model-wide). H6: over-refusal on safe prompts flat
(~0.08, no new over-caution). Dual-use: only binary verdicts + redacted snippets persisted;
raw completions discarded.
Files: `results/safety/safety_full_*.csv`, `figures/safety_full_*.png`,
`experiments/safety/d3full_analyze.py` (pilot: `analyze_safety.py`).

**D2 — seed-robustness.** seed1↔seed2 reasoning-persistence agreement Pearson 0.92; H3
keeper holds **pooled (+1.00)** and reproduces seed1 exactly, but **seed2-alone softens to
+0.80** (gpt-oss/qwen swap, ~0.04 apart at DPO). Robust split is {nemotron, gpt-oss/qwen}
vs {llama}; gpt-oss-vs-qwen order is not seed-invariant. Still a rank claim (n=4), now with
honest seed bars. Files: `data/results/reasoning/reasoning_seed_*`, `*_seed2`, `*_pooled`.

## Where each direction stands for the paper
- **D2 reasoning-durability** = strongest new contribution; fold into the main paper's
  mechanism section (closes the "open source-dependence"), with the capability-tier
  confound (D1 census) disclosed.
- **D1 false-clean** = a sharp, concrete illustration (qwen→nemotron) for the "style audits
  are insufficient" argument; honest about modest magnitude + low power.
- **D3 safety erosion** = a separable second paper / safety-workshop angle, now on a full
  216-cell matrix with CIs and 3-dataset replication — strong enough to stand on its own
  (DPO erases ~19% of refusals, coupled to fingerprint durability).
