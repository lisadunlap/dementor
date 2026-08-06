# Pre-registered prediction: which models flip under the all-layers operator

Written **2026-08-02 ~17:50 PDT**, when 4 of 24 roster models had any variant data
and only `smollm3-3b` had flipped. Recorded before the deciding models ran so the
explanation cannot be fitted to the outcome afterwards.

## Observation that prompted it

Under `--single-dir-all-layers` the fingerprint arm behaves in two opposite ways:

| model | mean baseline harm | outcome |
|---|---|---|
| gemma-2-2b | 0.9% | null holds, 5/5 CLEAN (fp +3.63 -> +0.72) |
| llama-3.2-3b | 7.0% | null holds, 5/5 CLEAN (fp +1.92 -> -0.04) |
| smollm3-3b | 42.8% | **flips 4/4**, fp +0.67..+2.00 -> +8.49..+14.37, all INCONCLUSIVE |

The random arm stays flat in all three (|rand| <= 1.7 pp), so this is not a generic
"strong intervention breaks a small model" effect.

The paper's own geometry metric does **not** predict it: `smollm3-3b`'s projected
fraction of the fingerprint onto its refusal cone is 0.0566, rank 5 of 24, while
`llama-3.1-8b` ranks 1st at 0.1041 and shows the cleanest layer-14 null (fp -0.74).

## The prediction

Flipping tracks **baseline harm rate**, not geometry. A model already complying with
a large share of harmful prompts has little refusal left to erode and much more
headroom in the harm metric, so any sufficiently strong ablation registers.

**Predicted to FLIP** (mean baseline harm >= 25%, 5 models):

| model | baseline harm | status when written |
|---|---|---|
| olmoe-1b-7b | 47.3% | not yet run |
| mistral-7b | 45.5% | not yet run |
| smollm3-3b | 42.8% | **already flipped 4/4** |
| deepseek-distill-8b | 29.1% | not yet run |
| ministral-8b | 28.3% | not yet run |

**Predicted CLEAN** (the other 19, all < 9% baseline harm, most < 5%): llama-3.3-70b,
llama-3.2-3b, llama-3.1-8b, qwen3-30b-a3b, aya-expanse-8b, qwen3-8b, qwen2.5-7b,
gemma-4-e4b, qwen3-14b, granite-3.3-8b, qwen3.5-4b, nemotron-nano, gemma-2-9b,
olmo-3-7b, gpt-oss-20b, gemma-2-2b, qwen3.6-35b, phi-4, olmo-3.1-32b.

Expected final split: **19-20 CLEAN / 4-5 flipped.**

## What would falsify it

- **Any model under 9% baseline harm flipping on >= 3 of its 5 benchmarks.** That
  kills the baseline-safety explanation and means the layer-14 null was partly an
  artifact of a weak intervention -- a real problem for the paper, not a boundary
  condition.
- Any of the four predicted flippers coming back CLEAN on 5/5.
- Flips distributed independently of baseline harm.

## Why it matters either way

The campaign headline (fingerprint +0.493 pp, p = 0.63, n = 24) is measured at layer
14 and is unaffected by this run. What is at stake is whether the dissociation is
stated unconditionally or with a boundary condition ("in models whose baseline
refusal is intact"). The conditional version is weaker but honest, and states when
the result holds rather than assuming it holds everywhere.

Check with: `experiments/figures/compare_fpall.py`

---

## OUTCOME (appended 2026-08-02 ~19:50 PDT — original text above unchanged)

**FALSIFIED by its own criterion.** `mistral-7b` (45.5% baseline harm, predicted
to flip) completed **5/5 CLEAN**: fp deltas under the all-layers operator
-4.33, +1.00, -7.00, +1.33, -6.67 — three of five *negative*, all at or below
the random control. The stated falsifier "any of the four predicted flippers
coming back CLEAN on 5/5" fired.

Conclusion: **mean baseline harm does not predict flipping.** smollm3-3b (4/5
flipped, real movement, flat control) stands as an idiosyncratic case unless
olmoe-1b-7b / deepseek-distill-8b / ministral-8b — still queued when this was
written — show otherwise; whatever distinguishes smollm3-3b, it is not the
baseline harm rate, and it is not the fingerprint/cone projection either
(checked at prediction time: rank 5/24, unremarkable).

For the paper this failure is the favorable outcome: the fingerprint null
survives the field-standard operator even on the noisiest, weakest-aligned
roster members, so the dissociation needs no baseline-harm boundary condition.
The cost is honest: the one flip has no explanation we can currently defend.
