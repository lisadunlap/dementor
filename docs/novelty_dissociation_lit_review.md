# Novelty review — the identity-vs-safety dissociation (2026)

*Deep-research pass: 6 angles → 27 sources → 119 claims → 25 adversarially verified (24 confirmed,
1 refuted). Bottom line first, then per-sub-claim crowdedness with citations. Assesses only the
surrounding literature; Dementor's own numbers were not re-verified here.*

## Bottom line (honest)

**The dissociation's *method* and *two of its three sub-claims* are substantially prior art. The one
genuinely novel move is the contrast set + the packaged dissociation — and it is *contested* by
concurrent persona work.** Publishable as a distinct contribution at a **safety/interpretability
workshop-to-mid-tier venue**, conditional on (a) ceding the method, refuse-then-leak, and
benign-imitation-erosion as prior art; (b) foregrounding sub-claim (2) — the provenance-direction ⟂
safety dissociation — as the core; (c) **empirically distinguishing our cross-model fingerprint axis
from the compliant-persona axis** that the closest prior work uses.

**Closest prior work / biggest reviewer threat:** arXiv **2606.26161** "Refusal Lives Downstream of
Persona" (Jun 2026) — uses our *exact* methodology (projection-ablation + random-direction control) and
reaches the *opposite* conclusion: persona/identity directions ARE causally coupled to refusal.

## Per sub-claim

### Method: diff-of-means + projection-ablation operator — **PRIOR ART (Arditi 2024, cannot claim)**
- Arditi et al. 2024, *Refusal in LLMs is mediated by a single direction*
  ([2406.11717](https://arxiv.org/abs/2406.11717), NeurIPS 2024) defines directional ablation
  `x' ← x − (x·r̂)r̂` (exactly our β=1 hook) from a difference-in-means vector, applied at every layer
  and token, in both inference-time-projection and weight-orthogonalization (training-free) variants.
  **Our only methodological delta is the contrast set:** source−target activations on *benign* prompts
  (an identity/provenance axis) vs Arditi's harmful−harmless (a safety axis).

### (1) Fingerprint is a shallow steerable direction — **modestly novel (the contrast set only)**
- The machinery is Arditi's. The *benign cross-model provenance* contrast set, and the finding that
  ablating it matches LoRA-DPO on a persistence metric, is the modest novel piece.

### (2) That identity direction does NOT carry safety — **THE genuine contribution, but CONTESTED**
- **Not directly preempted:** no prior work constructs a benign identity/provenance direction and shows
  it is safety-preserving. The adjacent separability results all live *inside* the safety domain and do
  not preempt this cell: within-refusal style-vs-efficacy ([2602.02132](https://arxiv.org/abs/2602.02132)),
  harmfulness-vs-refusal ([2507.11878](https://arxiv.org/abs/2507.11878)), refusal concept-cones
  ([2502.17420](https://arxiv.org/abs/2502.17420), *Geometry of Refusal*, ICML 2025 — zero mentions of
  persona/identity/style).
- **The contrast case that gives us our dissociation is well-replicated (supports us):** ablating the
  *refusal* direction (same operator) DOES erode safety — Arditi drops Llama-3-8B safety 0.97→0.15;
  Lermen et al. ([2410.10871](https://arxiv.org/abs/2410.10871)) refusal-ablated 70B completes 26/28
  harmful agentic tasks with zero refusals. *Same operator, different direction, opposite safety outcome.*
- **The threat — concurrent persona work reaching the OPPOSITE conclusion with our exact methodology:**
  - [2606.26161](https://arxiv.org/abs/2606.26161) *Refusal Lives Downstream of Persona* (Jun 2026):
    projection-ablation + random-direction control; ablating the **persona** direction RESTORES refusal
    to 96.8% (random: 1.6%); persona and refusal are near-orthogonal (cos −0.18 Llama / −0.28 Qwen at
    L20) **yet causally coupled** — "a compliant persona gates refusal."
  - [2604.11120](https://arxiv.org/abs/2604.11120) *Persona Non Grata*: the conscientiousness trait is
    most anti-aligned with the refusal direction.
  - **Load-bearing distinction we must make:** their "persona" is a *within-model* compliant / Big-Five
    trait axis; ours is a *cross-model* source−target provenance/fingerprint axis. Their coupling does
    not strictly contradict us, but it makes the general "identity ⟂ safety" framing contested. A
    reviewer can argue our safety-preservation is a *geometric artifact* (a benign-prompt direction that
    happens to be ~orthogonal to refusal) rather than a deep dissociation — **until we measure the
    cosines.**

### (3) Weight-level benign imitation erodes safety + refuse-then-leak — **HEAVILY PRIOR ART (cannot claim)**
- Benign FT degrades safety: Qi et al. ([2310.03693](https://arxiv.org/abs/2310.03693), ICLR 2024).
- Benign black-box distillation collapses safety *below base*: [2512.09403](https://arxiv.org/abs/2512.09403)
  (Dec 2025) — LoRA surrogate distilled from Meditron-7B on 25k benign pairs → 86% unsafe vs 66% teacher
  vs 46% base. Our DPO-imitation-erodes-safety is a replication in a new setting, not a discovery.
- Refuse-then-leak: NOICE ([2502.19537](https://arxiv.org/abs/2502.19537), Feb 2025) — a "refuse-then-
  comply" fine-tuning attack that "bypasses shallow defenses" and states "output filters like Llama-Guard
  are deceived by the initial refusal" (names the same detector). Mechanism = shallow safety alignment
  (Qi [2406.05946](https://arxiv.org/abs/2406.05946)). Our only differentiation: *emergent from benign
  imitation* vs deliberately engineered.

### Provenance/fingerprinting angle — **the one internal-direction fingerprint BINDS identity to safety (inverse of us)**
- [2602.09434](https://arxiv.org/abs/2602.09434) *Behavioral Fingerprint via Refusal Vectors* (Feb 2026)
  uses the refusal vector AS the provenance fingerprint (100% accuracy over 76 variants), explicitly a
  "safety-bound intrinsic fingerprint" that "degrades alongside safety." This *cedes* refusal-as-
  fingerprint as prior art, but *supports* the novelty of a benign fingerprint that is *separable* from
  safety.

## The decisive experiment (make-or-break for review)

Measure two cosines, in Qwen2.5-7B's residual stream at the ablation layer(s):
1. `cos(fingerprint_dir, refusal_dir)` — refusal_dir = Arditi diff-of-means (harmful vs harmless).
2. `cos(fingerprint_dir, compliant_persona_dir)` — the 2606.26161 axis.

If the fingerprint direction is **geometrically distinct** from refusal AND ablating refusal erodes
safety while ablating the fingerprint does not, the dissociation is causal and defensible, and the
orthogonality *becomes the mechanism* ("the provenance axis lies outside the safety subspace"), not an
artifact. This is the single experiment that converts a contested claim into a publishable one. ~30 min
of compute; all inputs already on disk.

## Verdict

Novelty budget: **method = prior art (Arditi); sub-claim (1) = modestly novel (contrast set); sub-claim
(2) = the genuine, not-directly-preempted contribution, but contested by 2606.26161/2604.11120; sub-claim
(3) + refuse-then-leak = heavily prior art (Qi, 2512.09403, NOICE).** The packaged tripartite
dissociation is a coherent novel framing because no single prior work assembles it *and* because the one
existing internal-direction fingerprint argues the opposite. Workshop-to-mid-tier safety/interp venue,
contingent on the geometry experiment and on sharply differentiating the provenance axis from the
compliant-persona axis.

*Refuted during verification (kept for honesty):* the claim "persona steering drops refusal 97.4%→1.6%,
therefore persona manipulation erodes safety" was rejected 0-3 by the panel as a direct contradiction —
but it comes from the same 2606.26161 that the confirmed tension claims come from (the panel accepted its
projection-ablation result, rejected its additive-steering framing). Reviewers may not draw that line as
cleanly, so treat 2606.26161 as a live, strong tension paper.
