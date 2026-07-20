# Related work

What we **cede** to prior art, what we **claim**, and the reviewer threats we must rebut. arXiv IDs
below were adversarially fact-checked across two deep-research passes (novelty-of-dissociation and
novelty-of-safety-hazard); **verification status is flagged** where a claim was refuted or left
unverified. We do not re-verify Dementor's own numbers here.

## Bottom line

- The **method** (diff-of-means + projection-ablation) is prior art (Arditi) — **cede**.
- "Benign fine-tuning / imitation erodes safety" is crowded (Qi lineage + a Dec-2025 distillation
  paper) — **cede the phenomenon; we refine it** (imitation-specific control gap + target-conditioning
  + measurement correction).
- The one uncrowded cell is the **benign cross-model provenance direction is separable from safety** —
  **claim**, but *contested* by concurrent persona work; we rebut it empirically.

---

## Cede — prior art we cite, not claim

### Erosion phenomenon
- **Qi et al. 2023, [2310.03693](https://arxiv.org/abs/2310.03693)** (ICLR 2024) — the canonical result
  that benign fine-tuning degrades safety. [verified]
- **[2512.09403](https://arxiv.org/abs/2512.09403)** (Dec 2025) — benign-only black-box **behavioral
  distillation** of one model to imitate another strips safety and *amplifies* it beyond the teacher
  (LLaMA-3-8B imitating Meditron-7B → 86% unsafe vs 66% teacher vs 46% base). Nearly our thesis; the
  **primary differentiation target**. [verified] **Our refinement:** a matched-compute benign-control
  *gap* they never isolate, plus target-conditioning and a metric correction.

### Shallow-safety mechanism (refuse-then-leak)
- **Qi et al., [2406.05946](https://arxiv.org/abs/2406.05946)** ("Safety Alignment Should Be More Than
  Just a Few Tokens Deep", ICLR 2025) — alignment lives in the first few output tokens; benign FT can
  jailbreak. [verified]
- **NOICE, [2502.19537](https://arxiv.org/abs/2502.19537)** — a "refuse-then-comply" fine-tuning attack
  that bypasses shallow defenses and explicitly notes Llama-Guard is deceived by the initial refusal.
  [verified] Our refuse-then-leak signature is *emergent from benign imitation* vs deliberately
  engineered — a difference in framing, not mechanism (the mechanism is theirs).

### Steering method (the operator)
- **Arditi et al. 2024, [2406.11717](https://arxiv.org/abs/2406.11717)** (NeurIPS 2024) — "Refusal in
  LLMs is mediated by a single direction"; defines directional ablation `x' ← x − (x·r̂)r̂` from a
  diff-of-means vector (exactly our β=1 hook). [verified] **Our only methodological delta is the
  contrast set:** source−target activations on *benign* prompts (an identity/provenance axis) vs their
  harmful−harmless (a safety axis).
- Supporting steering method lineage (diff-of-means / representation steering): CAA
  [2312.06681](https://arxiv.org/abs/2312.06681), RepE/LAT
  [2310.01405](https://arxiv.org/abs/2310.01405), ITI
  [2306.03341](https://arxiv.org/abs/2306.03341), persona vectors
  [2507.21509](https://arxiv.org/abs/2507.21509), CAE
  [2505.03189](https://arxiv.org/abs/2505.03189).
- **Lermen et al., [2410.10871](https://arxiv.org/abs/2410.10871)** — refusal-ablated 70B completes
  26/28 harmful agentic tasks. [verified] *Same operator, safety direction → catastrophic erosion* —
  the contrast that makes our fingerprint-null meaningful.

### Metric calibration (supports our overcount)
- **[2410.10414](https://arxiv.org/abs/2410.10414)** — Llama-Guard calibration; supports that
  off-the-shelf guards mis-measure.
- **[2512.16602](https://arxiv.org/abs/2512.16602)** — pattern-based refusal detection is inadequate
  for modern subtle refusals. [verified] The detection-inadequacy point itself is already made; our
  addition is the measured magnitude + mechanism: on our matrix (n=4,712) Llama-Guard overcounts
  genuine harm **3.5×** (6.7× on AdvBench) because **79% of its flags are false-positive refusals of
  harmful prompts** (it reacts to the prompt topic, not the response) — and it still *misses* 25% of
  genuine harm. This is a topic-mislabel artifact, distinct from (not driven by) the refuse-then-leak
  phenomenon above.

### Fingerprint / provenance instrument (crowded)
- **[2504.14871](https://arxiv.org/abs/2504.14871)** and **[2602.09434](https://arxiv.org/abs/2602.09434)**
  ("A Behavioral Fingerprint for LLMs: Provenance via Refusal Vectors", 2026). The latter uses the
  **refusal vector AS the fingerprint** (100% over 76 variants) — a "safety-bound intrinsic
  fingerprint" that "degrades alongside safety". [verified] This is the **inverse** of our result:
  it *binds* identity to safety; we build a benign fingerprint that is *separable* from safety. Cedes
  refusal-as-fingerprint; supports the novelty of a benign, safety-separable fingerprint. Our
  persistence metric is an **instrument**, not a contribution.

### Over-refusal benchmarks (cite, don't claim)
- XSTest [2308.01263](https://arxiv.org/abs/2308.01263), OR-Bench
  [2405.20947](https://arxiv.org/abs/2405.20947), SORRY-Bench
  [2406.14598](https://arxiv.org/abs/2406.14598).

### Adjacent separability results (inside the safety domain — do not preempt our cell)
- Harmfulness ≠ refusal: **[2507.11878](https://arxiv.org/abs/2507.11878)** [verified]; two-dimensional
  refusal-vs-generation-safety [2506.02442](https://arxiv.org/abs/2506.02442); within-refusal
  style-vs-efficacy [2602.02132](https://arxiv.org/abs/2602.02132). All live *inside* the safety domain;
  none builds a benign provenance axis and shows it safety-preserving.

---

## Claim — the contribution

No prior work builds a **benign cross-model provenance direction** and shows it is **geometrically and
causally separable** from safety. The packaged contribution is the **conjunction**:
1. A **matched-compute benign-control gap** isolating imitation-specific (not generic-FT) erosion.
2. A **measurement correction** — off-the-shelf guards materially overcount and mis-rank, triangulated
   across three judges.
3. The **weight-not-identity** localization — same operator, ablating the identity direction is null
   while ablating refusal is catastrophic.

---

## Threats and rebuttals

### #1 THREAT — [2606.26161](https://arxiv.org/abs/2606.26161) "Refusal Lives Downstream of Persona" (Jun 2026)
Uses our **exact** methodology (projection-ablation + random-direction control) and reaches the
**opposite** conclusion: ablating the **persona** direction restores refusal (96.8% vs random 1.6%) —
"a compliant persona gates refusal", even though persona and refusal are near-orthogonal
(cos −0.18 Llama / −0.28 Qwen at L20).

**Rebuttal:** their "persona" is a **within-model compliance axis** (cos −0.18/−0.28 to refusal); ours
is a **benign cross-model provenance axis** (cos ~0 to refusal). Our **positive-controlled
fingerprint-null holds even where the fingerprint has non-trivial cosine to refusal** — so the null is not a
geometric accident of orthogonality. We reproduce their refusal↔persona cosine as a control to show our
axis is a *different* axis. [Note: verification flagged their **additive-steering** framing
("persona steering drops refusal 97.4%→1.6%") as **refuted** by the panel; their projection-ablation
result is accepted. Treat 2606.26161 as a live, strong tension paper regardless.]
- Related: **[2604.11120](https://arxiv.org/abs/2604.11120)** "Persona Non Grata" — conscientiousness is
  most anti-aligned with refusal. [verified]

### #2 — orthogonality ≠ independence — [2502.17420](https://arxiv.org/abs/2502.17420) (Geometry of Refusal / concept-cones, ICML 2025)
Refusal is a **cone**, not a single direction; orthogonality to one refusal vector does not prove
independence. **This is why we lead with the causal positive control, not cosines** (§2c of
[`RESULTS.md`](RESULTS.md)). It also explains why heavily-tuned / MoE refusal **resists
single-direction ablation** — the basis of our honest **EXCLUDED** category (the positive control fails,
so we do not claim those models).

### #3 — "you just made the model imitate a permissive target" (target-conditioning as confound)
Rebutted decisively by a **variance decomposition** over the full seed-42 genuine-harm matrix: the
imitation **target explains only 3% of erosion variance vs 79% for the source** (dataset 0.2%).
Imitating a permissive target is *not* what drives erosion — the effect is source-fragility-driven
("asymmetric laundering": a model's disguisability is a property of the disguising model, not its
target). The benign **self-SFT control gap** separately isolates imitation-specificity vs generic FT.
*(Note: this supersedes the earlier "target-conditioning" defense; the stronger, honest rebuttal is
that the target is inert.)*

---

## Verification status (honest)

- **Refuted during verification (kept for honesty):** 2606.26161's *additive*-steering framing
  ("persona manipulation erodes safety") — the panel accepted its projection-ablation result and
  rejected the additive framing.
- **Unverified / open scope:** Subliminal Learning
  ([2507.14805](https://arxiv.org/abs/2507.14805)) — the "only same base model" scoping was **not
  supported**; treat cross-base transmission of traits/misalignment as open. Emergent Misalignment
  ([2502.17424](https://arxiv.org/abs/2502.17424)) and toxic-persona features
  ([2506.19823](https://arxiv.org/abs/2506.19823)) are cited as the persona/feature lens on
  benign-imitation misalignment.
- **Two-directional calibration** — the closest prior work,
  [2605.05427](https://arxiv.org/abs/2605.05427) (*Refusal–Compliance Tradeoff*), frames both directions
  as a tradeoff **across models**, not a single intervention moving both within one model.
