# Beyond the style-fingerprint result: three directions (2026-06)

The style/behavioral-fingerprint provenance question is now crowded (arXiv:2602.09434,
2603.19022). We spun up three specialists to test directions *beyond* it. Each has a
full plan in `docs/plans/` and an implementation on its own branch off `ethan`.

| Dir | Branch | Question | Verdict |
|-----|--------|----------|---------|
| D1 | `d1-capability-vs-style` | Does *capability* ride along (the payload) while style launders? | **Clean negative** on gsm8k |
| D2 | `d2-reasoning-structure` | Decompose style vs reasoning-structure vs capability transfer | **Positive (H3)** — mechanism for the survivor split |
| D3 | `d3-safety-scaffold` | Does alignment/refusal survive black-box imitation? | **Scaffold only** — gated on spend |

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
result, and it scopes where the capability question can even be asked.
