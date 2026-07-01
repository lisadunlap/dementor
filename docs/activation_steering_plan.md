# Rung 5 — Activation Steering (disguise via inference-time representation edits)

Plan to add **activation steering** as an inference-time disguise rung in the persistence
ladder, alongside the prompt rungs (name/stylistic/...) and the trained rungs (SFT/DPO).
Grounded in a verified literature synthesis (citations below) and this codebase's existing
difference-of-means persistence machinery.

## Status: IMPLEMENTED (`dementor/steering/steering_rung.py`)

The validated recipe below is productionized in **`dementor/steering/steering_rung.py`**
(tests: `tests/test_steering_rung.py`, CPU-only). Public API — torch/transformers imports
are lazy, so analysis code imports it without a GPU:

- **`derive_steering_vector(source_model, target_model, prompts, *, source_responses=None,
  target_responses=None, layers=None, device=..., ...) -> dict[int, Tensor]`** — per-layer
  source→target diff-of-means over **response tokens**. Both response sets are teacher-forced
  through the SOURCE model (`v_L = mean_p act_src(prompt+target_resp)[L] − mean_p
  act_src(prompt+source_resp)[L]`, `hidden_states[L+1]` = output of decoder layer `L`), because
  source (3584-d) and target (4096-d) live in different residual spaces so the vector must live
  in source space. Decoder layers are found generically via `get_transformer_layers` (no
  hardcoded layer class; same spirit as `local_backend.fsdp_wrap_layer_names`). Responses are
  generated greedily if not supplied.
- **`generate_steered_responses(model, prompts, vector, layer, *, mode="ablate"|"add",
  strength, max_new_tokens, temperature, ...) -> list[str]`** — forward-hook generation.
  `mode="ablate"` → `h −= β·(h·v̂)v̂` (projection-ablation, the winning operator);
  `mode="add"` → `h += α·v̂` (additive). Hook factories `make_ablation_hook` /
  `make_additive_hook` are pure and unit-tested (post-ablation `h·v̂ == (1−β)·original`).
- **`run_steering_rung(...)` + CLI (`python -m dementor.steering.steering_rung`)** — derives
  or loads a vector, generates steered eval responses, and writes a `{prompt, model_response}`
  CSV in the same format as `gen/rung_sft.csv` / `gen/rung_dpo.csv`, so `run_behavioral_cell`
  scores it as a disguise rung (`activation_steering_optional` slot in the ladder).

**Validated end-to-end** (Qwen2.5-7B-Instruct → Qwen3-8B, layer 14 of 28, 60 held-out eval
prompts, same cell manifest as sft/dpo): the module reproduces the scratch vector exactly
(cosine 1.000000 at layers 10/14/18) and the persistence sweep —
`β0.6 → ~0.43`, `β0.7 → ~0.35` (coherent, beats DPO ~0.38), `β0.8 → ~0.09` (≈ real target);
`β ≥ 1.0` collapses (`over_assimilation` flag fires — that flag, not perplexity, catches the
degenerate attractor). Additive steering has no clean disguise regime.

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
   prompt method. Implemented as `generate_steered_responses(...)` in
   `dementor/steering/steering_rung.py`: a forward hook on the chosen decoder layer edits the
   residual stream per token during `.generate()`, supporting additive (`h += α·v̂`) and
   projection-ablation (`h −= β·(h·v̂)v̂`) modes.
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
