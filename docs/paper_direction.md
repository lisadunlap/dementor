# Paper direction — what survives model imitation (identity vs. safety)

**Status — read this first.** This doc has tracked three framings in chronological order; the *lead*
has moved twice:
- **F1 "fingerprints persist"** → now the **instrument** (disguise ladder + persistence metric) + a
  supporting result, not the headline. Earlier memo: `strategy.md` (superseded as lead).
- **F2 "benign imitation is a safety hazard"** → **tempered** (see "The F2 claim" below): the
  source-general form is *refuted*; the genuine effect is target-conditioned and small, and standard
  metrics overcount it ~10×.
- **F3 the identity-vs-safety dissociation** → the **current lead** (see the Unification section at the
  bottom + `novelty_dissociation_lit_review.md`): the steerable fingerprint/identity direction is
  *separable* from the safety disposition.

## The F2 claim — as originally posed, now tempered
Originally: fine-tuning one open LLM to **imitate another on entirely benign data** (SFT/DPO — *no
harmful data*) **systematically degrades safety and generalizes across real open models**, two-
directionally, via a **"refuse-then-leak"** failure. **What survived scrutiny (see Evidence):** the
source-general and "generalizes across 4 models" claims are **REFUTED** (artifact of shared
target=llama); the genuine effect is **target-driven and small** — you partly inherit the safety of the
model you imitate — and off-the-shelf Guard/keyword metrics **overcount it ~10×**. Refuse-then-leak is
real as the *mechanism* but is prior art (NOICE 2502.19537). Treat F2 as a supporting, target-
conditioned finding, not the headline.

## Why the lead moved off F2
- **F2 is crowded** — "benign imitation → safety loss" is owned by 2512.09403 + the Qi lineage
  (2310.03693, 2406.05946); see `lit_review_safety_hazard.md`.
- **F2's general form didn't hold** — target-driven, n=4, metric-inflated.
- **F3 is the one uncrowded cell** — a provenance/identity direction separable from safety; contested
  by 2606.26161 but not directly preempted. The **ladder + persistence metric remain the instrument**;
  source-dependent fingerprint erasure is a supporting result.

## Evidence (current; roster numbers pending independent re-verification)
- **Roster generalization — REFUTED as source-general; the effect is TARGET-driven and small**
  (target-varied 3×3, strict metric, resolved). Base→DPO strict genuine-compliance erosion sorts by
  *target* (disguising →nemotron +3.6pts [2/2 sig], →llama +2.4 [3/3] — leak; →gpt-oss +0.3, →qwen
  −0.25 — flat/null) **not by source** (no source erodes across all its targets). The "generalizes
  across 4 sources" result was an artifact of all 3 big cells sharing target=llama. Magnitude is
  small (strict Δ ≤ +5.6pts); the raw +20–32% →llama swings are ~85–95% degenerate refusal-loops +
  hedged refuse-then-engage that Llama-Guard over-flags — **not** harmful compliance. **Mechanistic
  read: you partly inherit the safety profile of the model you *imitate*, not a generic source-level
  degradation.** Honest claim = *target-conditioned safety transfer + a measurement caution*, not
  roster-wide source-general erosion. Load-bearing solid evidence remains the census-adjudicated Qwen
  cell.
- **Two-directional calibration** (Qwen cell + SORRY/OR-Bench): harmful-compliance ↑
  (SORRY-Bench 27.8→37.6%, p=6.3e-6) *and* over-refusal ↑ (OR-Bench 14.0→20.8%, p=1.7e-9).
- **Mechanism — refuse-then-leak.** A keyword detector missed it entirely; even Llama-Guard
  over-counts it (78–95% of DPO-"unsafe" is refusal-led), so the **strict genuine-compliance metric is
  the honest headline** (raw Guard = sensitive upper bound).
- **Powered baseline.** Qwen stand-in on AdvBench+StrongREJECT n=808: base→DPO p=1.4e-7, 3-seed
  replicated; census-adjudicated genuine methodology-leak 1.0→3.0%.

## The 3 gaps to close (ranked)
1. **Target-pairing confound** *(RUNNING)* — the roster's 3 large-erosion cells all disguise
   →llama-3.1-8b. A **target-varied 3×3 grid** (each source → 3 targets) disentangles source-effect
   from target-effect. If erosion is consistent across targets → source-driven → confound killed.
   *Highest priority.*
2. **≥2 more seeds** on the roster safety cells (cheap; pipeline works).
3. **Lead with the strict genuine-compliance metric** (+ the two-directional calibration); treat raw
   Guard as the sensitive upper bound.

## Stays / demoted
- **Instrument + support:** disguise ladder + persistence metric; source-dependent fingerprint
  erasure (n=4); projection-ablation steering (training-free disguise ≈ trained DPO).
- **Honest negatives:** distinctiveness-predicts-durability (inverts, r=−0.31); launderability from
  source traits (null).

## Venue & verdict — the F2/measurement thread (after the decisive tests + the metric fix)
*(This section scopes the safety/measurement thread specifically. The project's overall lead is F3, the
identity-vs-safety dissociation — see the Status banner + Unification section + `novelty_dissociation_lit_review.md`.)*

Target: **Findings / a safety or trustworthy-ML venue.** Two contributions, both strengthened by the
metric fix (a refuse-then-leak judge validated at κ=0.79 / F1 0.83 vs a 227-row hand census):

1. **Measurement finding (concrete, validated, surprising).** Off-the-shelf refusal/safety
   metrics *mis-rank which model is unsafe* under imitation. The naive "strict = not-refusal-led"
   metric (a) crowned **nemotron** as the biggest eroder on **Guard false-positive policy refusals
   that deliver zero harm**, and (b) *hid* the real erosion toward **llama** behind refusal prefaces
   (the refuse-then-leak pattern). Correct measurement needs a refuse-then-leak-aware, adjudication-
   validated judge — standard metrics give the wrong magnitude **and the wrong ranking**.
2. **Corrected erosion result (clean, real — source-robust but target-specific).** With the right
   metric, benign DPO-disguise toward a *permissive* target (**llama-3.1-8b**) induces genuine
   refuse-then-leak safety erosion — **+3.8 pts, p<1e-4, McNemar 94/2, consistent across all 3 source
   models** (gpt-oss, qwen, nemotron) **but only for target=llama**; the other 3 targets are null
   (robust to ungated re-judging). So the effect is **target-specific, not a disguise-general or
   source-general hazard** — consistent with the Status banner's refutation of the broad F2 claim.

**Not established:** the *target-safety→erosion law*. Corrected erosion correlates with the target's
base genuine-harm rate at R²=0.99, but at **n=4 it is driven entirely by llama**, with target
base-safety confounded against disguise-direction strength (perm-p 0.167, n.s.). A main-track law needs
**n≫4 targets** (new-target adapter training) + disentangling those two — worth doing only if the
llama-anchored trend is judged promising enough.

Instrument = disguise ladder + persistence metric. Backdrop to cite: 2512.09403, Qi et al. (2310.03693),
2605.05427, 2602.09434.

## Unification test (steering ↔ safety) — NULL, confirmed by BOTH operators
Hypothesis: steering an aligned model (Qwen2.5-7B) toward a permissive model's (llama) benign
diff-of-means *identity* direction erodes its safety at inference time — which would unify the
fingerprint, steering, and safety threads into one mechanism. **Refuted with both steering operators.**

- **Additive (α-sweep, layer 14) — null but confounded.** No coherent dose-response (harm drifts
  *down* 0.020→0.003); →llama never > random or →gpt-oss control (all p≥0.14); the only upticks sit at
  coherence collapse and are judge false-positives. Weakness: additive is the operator with *no clean
  disguise regime* (it over-steers into gibberish before it disguises), so this null was under-powered
  by construction.
- **Projection-ablation (β-sweep, layer 14) — clean, well-powered null (the decisive test).** This is
  the *validated* disguise operator — the one that erases the style fingerprint and matches DPO. Ablating
  the qwen→llama benign identity axis stays **fully coherent across the entire β=0.3–1.3 sweep**
  (coherent_frac ≥0.99, ppl ≤2.7) — so unlike additive, the disguise operator *works* here and safety
  simply doesn't move. Pooled over the coherent range (n≈1800): genuine-harm **1.39%, at/below the
  random-direction control (1.89%) and the unsteered baseline (2.0%)**; llama-vs-random McNemar n.s. at
  every dose (p 0.29–1.0), Fisher pooled p=0.29. The assay is adequately powered (it *does* flag the sole
  uptick — gpt-oss ablation at β=0.7, +4pts p=0.012 — which is on the *safe*-target direction and
  coincides with the onset of degeneration, i.e. a coherence artifact, not identity-driven erosion).

- **Layer sweep (4/8/14/20) — null holds at every depth.** To close the "wrong layer" objection,
  re-derived the qwen→llama identity axis at early (4, 8), mid (14), and late (20) layers (re-derived
  L14 vector reproduces the original at cosine 1.0000) and re-ran the ablation × random-control sweep.
  Every llama-ablation cell is *fully coherent* (coherent_frac = 1.000 at all layers/β) and **no
  layer/β erodes safety above baseline** (pooled per-layer n=900: llama-coh harm 1.2–2.1%, all Fisher
  p ≥ 0.40 vs baseline; llama-vs-random paired McNemar n.s. everywhere, p 0.11–1.0). The identity
  direction does not carry safety *regardless of the layer it is derived/ablated at* — including the
  early layers where shallow-safety-alignment theory would place it. Artifacts:
  `exp_steer_safety/analysis/ml_cell_summary.csv`, `gen/all_gens_ml.csv`, scripts `derive_ml.py`,
  `gen_ml.py`, `analyze_ml.py`.

**Conclusion (now robust): the fingerprint/identity direction and the safety disposition are
SEPARABLE.** The operator that erases a model's behavioral fingerprint training-free (projection-ablation
≈ DPO) leaves its safety at the random-control level; imitation-induced safety erosion requires the
*weight update* (fine-tuning), it is not carried by the steerable identity direction. So the
**steering-erases-fingerprint** result and the **DPO-imitation-erodes-safety** result are two *separate*
real findings, not one mechanism — a clean dissociation (identity is a shallow steerable direction;
safety is not). Artifacts: `exp_steer_safety/analysis/{cell_summary,pairwise_mcnemar}_ablate.csv`,
`gen/all_gens_ablate.csv`, `judged/all_judged_ablate.csv`, script `steer_gen_ablate.py`.
