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
