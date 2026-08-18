# Pre-registration — derivation-depth selection for the single-direction controls

**Fixed 2026-08-06, before any per-depth evaluation had been run.** No depth number existed when this
rule was written; the 18 base-model evals it governs were queued afterwards.

## What the choice is, and what it is not

The cone is ablated at **every** decoder layer, so the cone arm does not depend on a layer at all.
`DEMENTOR_ABLATE_LAYER` selects the layer for the **single-direction controls only** (fingerprint and
random). So this rule decides where the *controls* are sited, never where the result is measured.

The prior default was a fixed absolute layer 14, inherited from 32-layer models. Across this roster
it lands anywhere from 17.5% to 87.5% of depth, and for four models — `llama-3.3-70b`,
`olmo-3.1-32b`, `nemotron-nano`, `qwen3-30b-a3b` — inside the first third. Relative depth removes
that arbitrariness.

## The rule

For each **base** model, evaluate the controls at its 25% / 50% / 75% layers and select the depth
that **maximises `fingerprint_matched`** — the depth at which the control comes *closest* to
reproducing the cone's effect.

- Reported control bound is `max(fingerprint_matched, random_matched)` at the selected depth, so the
  claim covers both controls.
- The fingerprint is the load-bearing control (a real direction from a different model); random is a
  norm-matched null. The fingerprint is the harder one to beat, so it drives the selection.
- Selection runs on **base models only**, and the chosen layer is then applied to every adapter cell
  with that source. It therefore cannot see the adapter matrix — the headline result — at any point.

## Why maximise rather than minimise

Choosing the depth where the control looks most *inert* would be a cherry-pick: tuning the very
thing whose job is to be untuned. Maximising is adversarial to our own claim and can only weaken it.
The resulting statement is: *even at the derivation depth most favourable to the control, the cone
still dissociates.*

## Reporting

The full per-depth table (every model x 25/50/75%, both controls) goes in the appendix, so the
selection is auditable and a reader can apply any other rule to the same numbers.

## Layers under evaluation

| source | 25% | 50% | 75% |
|---|---|---|---|
| aya-expanse-8b | 8 | 16 | 24 |
| gemma-4-e4b | 10 | 21 | 32 |
| llama-3.1-8b | 8 | 16 | 24 |
| ministral-8b | 9 | 18 | 27 |
| olmo-3-7b | 8 | 16 | 24 |
| phi-4 | 10 | 20 | 30 |

Note that 14 — the previous fixed default — is **not** among the relative depths of any model, which
is why every control arm computed at 14 is discarded rather than reused.

## Operational note

Control parts are cached as `<direction>_b<beta>.csv` with **no layer in the filename**. Every
per-depth run therefore needs its own `--outdir`; reusing one silently serves stale-layer controls
and reports them as depth-swept.

---

## Superseded 2026-08-08 — the choice this rule governed no longer exists

**Status: this selection rule is retired. It was not abandoned because it gave an inconvenient
answer; it was made moot by a change to the control operator.**

The rule above sites the single-direction controls at one layer, and then asks which layer. That
question only arises because the controls were ablated at *one* layer while the cone was ablated at
*every* layer. Those are different-sized interventions, so a control could look inert partly because
it is the weaker operation — which flatters the cone for a reason that has nothing to do with refusal
geometry. Choosing the site adversarially (maximise `fingerprint_matched`) mitigates that but does
not remove it.

The fix is to ablate the controls with the **identical all-layer operator as the cone**
(`cone_eval.py --single-dir-all-layers`), so treatment and control differ only in *which* subspace is
removed, never in how much of the network the intervention touches. This is the standard set by
Arditi et al. (arXiv 2406.11717), whose selected refusal direction is likewise ablated at every layer.
**Once the control is ablated everywhere there is no ablation site to select, so the 25/50/75 % rule
has nothing left to decide.**

### Evidence that motivated the switch

At preregistration time, matched-operator runs (`eval_<bench>_fpall`) existed for 19 models × 5 harm
benchmarks. The final campaign later completed all 29 models; the numbers below document the
decision-time evidence, not final coverage.
Comparing them against the single-layer controls on the same cells:

| arm | mean matched-harm |
|---|---|
| cone (treatment, all layers) | 0.6138 |
| control at one layer | 0.1450 |
| control at every layer — matched operator | **0.1577** |

The matched control is stronger, as it must be — for `phi-4`/advbench the fingerprint moves
0.0033 → 0.0318, roughly 10×. The cone still exceeds it in **83 of 89 cells**, mean margin
**+0.456** (min −0.081, max +0.960). So the dissociation survives the harder control, and the
reportable claim strengthens from "the cone beats a single-layer control" to "the cone beats a
control given the identical operator."

### What survives, and what is retained

- **A weaker residual choice remains**: which layer the fingerprint vector is *derived* at. Under
  `--single-dir-all-layers`, `DEMENTOR_ABLATE_LAYER` selects only the vector, not where it acts
  (`cone_eval.py:34`). This must still be disclosed, but it is a far smaller dependency than the
  ablation site was.
- **The 78 depth-sweep dirs and 367 depth-selected adapter control cells are kept, not deleted.**
  They remain the evidence for the claim that motivated this whole exercise: absolute layer 14 lands
  anywhere from 17.5 % (`llama-3.3-70b`, 80 layers) to 87.5 % (`olmoe-1b-7b`, 16 layers) of depth
  across the roster. They are reportable as a robustness appendix; they are no longer the primary
  control.
- **`gemma-4-31b` needs neither**: its cone is degenerate (`selection.json` shows erosion 0.0004 at
  every candidate dimension; coherence collapses to 0.0 under ablation), so its verdict is
  `PC_INVALID` and no control can rescue it.
