# Rung 5 — Activation Steering (disguise via inference-time representation edits)

Plan to add **activation steering** as an inference-time disguise rung in the persistence
ladder, alongside the prompt rungs (name/stylistic/...) and the trained rungs (SFT/DPO).
Grounded in a verified literature synthesis (citations below) and this codebase's existing
difference-of-means persistence machinery.

## Why it fits Dementor exactly
The dominant, label-cheap recipe for a behavior/persona/style steering direction is
**difference-of-means of residual-stream activations** — `v = mean(D+) − mean(D−)` per layer
[CAA 2312.06681; persona vectors 2507.21509; ITI mass-mean-shift 2306.03341]. That is the
**same construction as our metric's source→target axis** (`latent_behavior_axes.py:671`
`d_feat = target_mean − source_mean`). And the standard way steering effectiveness is measured —
**projecting activations onto the steering vector** — is mathematically the same projection the
metric already computes. So Rung 5 reuses our existing math, in activation space.

## Method (what the literature converged on)
- **Direction**: residual-stream activations on contrastive examples, **averaged over response
  tokens** (response > prompt tokens), differenced. Alternatives: RepE-LAT (PCA on difference
  vectors, 1st PC = reading vector) [2310.01405]; probes are used only to *localize* layers/heads,
  not as the direction itself [ITI 2306.03341].
- **Application**: `h_l ← h_l + α·v` at a **single early-mid layer (~40–60% depth**, e.g. layer 15
  of Llama-3-8B / ~layer 20 at 50% depth), added to **post-prompt/response** tokens and re-applied
  **autoregressively** at every generated token [CAA; persona vectors].
- **Operators**: additive `R±v` (impose target persona) · **projection/ablation
  `R − (Rᵀv/‖v‖²)v`** (erase the *source* fingerprint direction) · piece-wise [RepE 2310.01405].
- **Strength α**: inverted-U tradeoff — small α steers; over-steering raises KL/CE and breaks
  fluency (gibberish) [CAE 2505.03189; persona vectors]. Tune α against a capability guardrail.
- **vs SFT/DPO**: training-free, **complementary** (stacks on finetuning; can be injected during
  finetuning as "preventative steering"); nearly matched a LoRA-SFT ceiling in one benchmark
  (9.60 vs 9.61).
- **Key persistence risk**: steering is reliably effective **in-distribution** and can fail or
  reverse **out-of-distribution** — and our persistence eval is on *held-out* prompts, i.e. OOD
  relative to the steering-vector derivation set. This must be measured, not assumed.

## Implementation in this repo
1. **Derive the steering vector** (offline, reuse existing data): run source and target models over
   the *train* prompts (`gen/source_train_greedy.csv`, `gen/target_train_greedy.csv`), capture
   residual activations averaged over response tokens per layer, `v_l = mean(target) − mean(source)`.
2. **Steered generation path** — steering edits the forward pass, so it is NOT a `dementor/methods/*`
   prompt method. Add `generate_steered_responses(...)` next to `generate_local_responses` in
   `local_backend.py`: a forward hook on the chosen decoder layer adds `α·v_l` to the residual stream
   per token during `.generate()`. Support additive and projection-ablation modes.
3. **Tune (layer, α)** on a small dev slice: layer ∈ {~40–60% of depth}, α swept; pick the point that
   maximizes movement toward target persistence while MMLU/perplexity stay within tolerance.
4. **Score** the steered eval responses with the existing `run_behavioral_cell` — add an
   `activation_steering` rung to the ladder; report persistence + `movement_raw` + the
   **capability-cost curve** (persistence vs MMLU/fluency), and compare its Pareto front to SFT/DPO.
5. **Two variants to run**: additive-toward-target (disguise) and project-out-source (fingerprint
   removal).

## Reference implementations
- **`safety-research/persona_vectors`** — near drop-in: automated NL-trait-description → diff-of-means
  extraction, single-layer residual-add steering with token-scope control, training+inference modes,
  and a projection monitor.
- **StyleRemix** — handcrafted style-axis set + a diff-of-means→coefficient mapping for choosing which
  style axes to steer and how hard (and a weight-based LoRA-disguise comparison point).

## Sources (verified, 3-0 adversarial)
CAA 2312.06681 · RepE/LAT 2310.01405 · ITI 2306.03341 · Persona vectors 2507.21509 ·
CAE 2505.03189 · `github.com/safety-research/persona_vectors` · StyleRemix.
