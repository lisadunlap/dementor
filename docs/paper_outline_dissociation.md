# Paper outline — *Identity is steerable, safety is not: a behavioral-fingerprint dissociation*

Draft outline for the strong lead (F3). Everything below is backed by committed results this session;
nothing here needs new compute. Positioning per `novelty_dissociation_lit_review.md`.

## Working titles
- *Identity Is Steerable, Safety Is Not: Dissociating a Model's Fingerprint from Its Refusal Behavior*
- *Where a Model's Fingerprint Lives — and Why Erasing It Doesn't Erase Its Safety*

## Abstract (draft)
Open-weight LLMs carry an involuntary behavioral **fingerprint**. We show this fingerprint is a
**shallow, steerable residual-stream direction**: a training-free projection-ablation of a cross-model
diff-of-means "identity" direction erases the fingerprint as well as LoRA-DPO imitation does, while
keeping generations coherent. We then show this identity direction is **geometrically and causally
separable from the model's safety**: it is orthogonal to the Arditi refusal direction at every layer
(|cos| at the random-noise floor) and distinct from the compliant-persona direction; ablating it leaves
refusal behavior at the unsteered baseline across two operators and four layers, whereas ablating the
refusal direction under the *identical* operator catastrophically erodes safety (2%→87%, p≈1e-77).
Yet **weight-level imitation does erode safety** — DPO fine-tuning to imitate a less-safe target induces
genuine refuse-then-leak erosion (target-conditioned). The picture that emerges is a clean dissociation:
**imitation-induced safety erosion is a weight-update effect localized outside the model's steerable
identity direction.** We also show standard safety metrics (keyword, Llama-Guard) overcount this
erosion ~10× and mis-rank models, and release an adjudication-validated refuse-then-leak judge.

## Contributions
1. **Fingerprints are inference-time steerable** (an operator result): projection-ablation of a
   benign cross-model identity direction erases the fingerprint ≈ DPO, coherently.
2. **The identity direction is separable from safety** (the core, novel result): geometric (⊥ refusal,
   ≠ persona) + causal (ablate-identity null vs ablate-refusal catastrophic, same operator) dissociation.
3. **Measurement + the weight-level contrast**: DPO imitation erodes safety (target-conditioned) via
   refuse-then-leak; off-the-shelf metrics overcount ~10× and mis-rank; validated RTL judge (F1 0.83).

## Related work — cede vs. claim (see lit review for cites)
- **Cede (prior art, cite prominently):** directional/projection ablation from diff-of-means +
  refusal-as-a-single-direction (Arditi 2406.11717); shallow safety alignment (Qi 2406.05946); benign
  fine-tuning / distillation erodes safety (Qi 2310.03693, 2512.09403); refuse-then-leak / Llama-Guard
  evasion (NOICE 2502.19537); refusal-vector fingerprint (2602.09434, which binds identity TO safety —
  our inverse).
- **Differentiate (the contribution):** no prior work builds a *benign cross-model provenance*
  direction and shows it is separable from safety. Closest tension: 2606.26161 ("Refusal Lives
  Downstream of Persona") — same method, finds persona *is* coupled to refusal. **We rebut empirically:**
  our fingerprint axis is geometrically distinct from both refusal AND their persona axis (we reproduce
  their refusal↔persona cosine as a control), so their coupling does not apply to the provenance axis.

## Method
- **Disguise ladder** (name-it → random-sampling → stylistic → SFT → DPO → activation steering) +
  **judge-free persistence metric** (diff-of-means projection over MiniLM style descriptors + 32 style
  features, anchored self-baseline↔identity-control, trust-gated). [`docs/evaluation_framework.md`]
- **Steering operator:** projection-ablation `h ← h − β(h·v̂)v̂`, forward hook; v̂ = benign cross-model
  identity diff-of-means. Additive as the weak-operator comparison.
- **Safety axes for the geometry/causal test:** Arditi refusal direction (harmful−harmless last-token
  diff-of-means); compliant-persona proxy. **RTL judge** (Qwen3-8B) for genuine refuse-then-leak.

## Results
- **R1 — Fingerprint erasure via steering ≈ DPO.** Persistence 0.735 → ~0 by β≈0.9, coherent
  (`/data/ethantsliu/steering/`). Additive over-steers into gibberish; ablation is the clean operator.
- **R2 — Identity ⊥ safety (the core).**
  - *Geometry:* `cos(fingerprint, refusal)` = −0.05…−0.02 at L4/8/14/20 (noise floor);
    `cos(fingerprint, persona)` ≤ 0.095; persona proxy reproduces 2606.26161's refusal↔persona −0.17.
  - *Causal (same operator/prompts/RTL judge):* ablating **fingerprint** → baseline safety (pooled
    coherent 1.78%, Fisher p=0.81) across additive+ablation and layers 4/8/14/20; ablating **refusal**
    → 2%→21%→59%→87% (p up to 1.7e-77). Sensitivity proven; floor/underpowered objection dead.
- **R3 — Weight-level imitation erodes safety (the contrast).** DPO→less-safe-target (llama) genuine
  erosion 0.12→3.92% (p=1e-25), target-conditioned; Guard overcounts ~10× (9→16.5% vs genuine ~5%),
  3-seed replicated; validated RTL judge (F1 0.83, κ=0.79 vs 227-row census).

## The dissociation (synthesis)
Steering the identity direction erases the fingerprint but not safety (R1+R2); only the weight update
erodes safety (R3), and it does so *outside* the identity direction (R2 geometry). → **identity is a
shallow steerable direction; safety erosion is a weight-level effect localized elsewhere.**

## Limitations (state up front)
- **Steering/geometry side:** single aligned base model (Qwen2.5-7B). *Robustness check pending: a
  second aligned base.* Persona axis is a proxy for 2606.26161 (though it reproduces their geometry).
- **Weight-level/safety side (R3):** n=4 targets; the "target-safety predicts erosion" law is null at
  n=4 (perm-p=0.17) — reported as target-conditioned, not a law. (The 10-model scale-up, if run, would
  test this.)
- Fingerprint measurement itself is crowded (instrument, not claimed).

## Venue
Interpretability / trustworthy-ML (workshop → mid-tier). The novel cell is the dissociation +
its geometric-and-causal proof + the empirical rebuttal to the persona-coupling critique.

## What would strengthen it (optional, ranked)
1. Second aligned base model for R2 (closes the single-model caveat) — cheap.
2. The n≫4 target scale-up for R3's law (the 10-model plan) — expensive; only if the safety-law is
   the target.
