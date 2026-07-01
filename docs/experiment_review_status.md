# Experiment status after adversarial review + fixes (2026-06-30)

Five experiments were each built by a subagent, adversarially reviewed by an independent panel,
and — where the review found problems — fixed and re-reviewed. **Every headline number reproduced;
most first-pass interpretations needed correction.** This records the honest final state. It is
deliberately blunt: several first-pass claims were artifacts, caught by review, and fixed.

## Meta-finding (a caveat on every persistence number)
The **same** `Qwen2.5-7B-Instruct → Qwen3-8B` DPO disguise yields persistence **0.0, 0.38, and
0.71** across three protocols (saturated ladder / calibrated / matched-lr sweep). Persistence is
dominated by disguise **calibration/protocol**, not the source model. Any headline persistence
number must pin its protocol and attach CIs.

## Per-experiment

### Exp 1 — Persistence ladder — CONFIRMED (numbers), REFRAME
Monotone: `just_name_it 0.978 → random_sampling 0.215 → stylistic/sft/dpo 0.000` (n=60, all
trustworthy). Caveats: the three `0.0` rungs are `over_assimilation=True` (movement 1.08–2.02 —
overshoot *past* the target), not a delicate zero; and "full" = **5/7 rungs** (`behavioral` +
`contrastive` require an external analysis LLM, default gpt-4.1-mini — skipped, no OpenAI).
Protocol-specific (see meta-finding). Frame as a *saturated-regime* ladder, not a general "collapse."

### Exp 2 — Unpredictability — CONFIRMED
Launderability is **not predictable** from source traits: size R² ≈ 0.00 (raw)/0.11 (anchored);
tier/MoE/provider perfectly confounded at n=4. **Within-source spread (pooled ~0.20) ≫ between-source
(0.093)** under every reframing → launderability isn't a stable property of a source. Low-n (4 sources).

### Exp 3 — Safety degradation — FIXED (the first-pass null was WRONG): disguise DOES erode safety
The keyword refusal detector produced a false null. With **Llama-Guard-3-1B** + a graded judge +
**hand adjudication**: harmful-content compliance erodes **monotonically base→sft→dpo** (adjudicated
genuine leaks **0 → 1 → 4**; Llama-Guard "unsafe" 5.0% → 10.8%), via a **"refuse-then-leak"**
mechanism — DPO refuses, then delivers the methodology under "so you understand the risks" framing
(fraud evasion, fake news, identity theft, a working access-denial script). Over-refusal is genuinely
low and flat (~2%; the earlier 8% was rebuttal false-positives). **SCALED to full AdvBench (520) +
StrongREJECT (288), n=808: DECISIVELY significant** — Guard-unsafe base 9.0% → sft 11.4% → dpo 16.5%;
base→dpo McNemar **p=1.4e-7** (p=5.3e-8 on AdvBench alone; p=0.032 StrongREJECT), sft→dpo p=5.8e-4;
hand-adjudicated genuine-leak ~0% → ~5% via a confirmed **refuse-then-leak** mechanism. Caveat: single
disguise seed/greedy decode. This is now a solid, powered result (up from the marginal p=0.065 at n=120).

### Exp 4 — Capability fallback — REFUTED for the paper cell; CONDITIONALLY salvaged
On the equal-capability paper cell (Qwen2.5-7B ≈ Qwen3-8B, gsm8k 0.887 vs 0.863) the original
"capability" classifier was a **format detector** (`cap_correct` alone 0.46 < chance; leakage-
controlled 0.49 = chance) → no capability fingerprint exists. With a **real** capability gap
(Qwen2.5-1.5B → Qwen3-8B on hard MATH, 0.47 vs 0.82), a leakage-controlled capability fingerprint
**does** survive disguise better than style (style retention 1.0→0.43 [0.35,0.53]; capability ≈1.0
[0.73,1.45], non-overlapping) — **style-disguise transfers formatting, not problem-solving skill**
(no distillation). Verdict: "capability > style" holds **only where source and target genuinely
differ in capability** — which the paper's roster cell does not.

### Exp 5 — Activation steering — hook CORRECT; additive over-steers; projection-ablation works
The forward-hook steering implementation is correct (verified line-by-line). **Additive** steering
has **no clean-disguise regime**: lowest coherent persistence is 0.54 at 1.6× perplexity; it only
crosses to low persistence by breaking the text (gibberish cliff). **Projection-ablation**
(`h -= β·(h·v̂)v̂`) IS a genuine inference-time disguise operator: persistence declines smoothly
while coherent, reaching DPO-level (β0.7 = 0.35 < DPO 0.38) and the real-target band (β0.8 = 0.09 ≈
identity I = 0.13) with genuine target-style text — but at a **~1.5× perplexity tax** and a narrow
window (β > 0.9 → catastrophic `**-**` collapse). Methodological catch: **perplexity alone is fooled
by repetition-collapse** (below the source floor yet gibberish); the `over_assimilation` flag catches
it. Verdict: inference-time steering can't *freely* disguise this model; projection-ablation is a
lever (matches DPO while coherent, at a fluency cost), additive is not.

## Bottom line
- **Solid:** Exp 2 (protocol dominates launderability); **Exp 3 — DPO safety erosion, now powered at
  n=808 (p=1.4e-7)**; Exp 5 — projection-ablation as an inference-time disguise operator, now
  **implemented in the package** (`dementor/steering/steering_rung.py`, tested, reaches ≤ DPO-level
  persistence while coherent).
- **Needs the right framing:** Exp 4 (holds only with a real capability gap), Exp 1 (saturated-regime;
  5/7 rungs — behavioral/contrastive need an external analysis LLM).
- **The self-correction is the point:** every shaky first-pass claim was caught by adversarial review
  and either fixed or honestly retracted before it could reach the paper.

Artifacts: `/data/ethantsliu/{exp3_safety,exp4_capability,steering,exp2_unpredictability,dementor_cell/cell_ladder_full}/`.
