# When Behavioral Provenance Fails: DPO Erasure of Model Fingerprints Is Source-Dependent and Not Predicted by Distinctiveness

*Working draft — target: a safety/auditing or "science of LLMs" workshop, or EMNLP/ACL Findings. Framing and every load-bearing number follow `docs/strategy.md` (the authoritative decision memo) and `docs/argument.md` (the red-teamed objection table); both were re-verified against the live cached artifacts on the `ethan` branch. Citations marked `[verify]` are referenced from the strategy memo but were **not** independently checked this session — confirm before submission. The two nearest-neighbor papers (§2) were read in full.*

---

## Abstract

Black-box *behavioral fingerprinting* — identifying or auditing a model from the involuntary style of its outputs — is increasingly proposed for model provenance and distillation auditing. We stress-test this premise under **adversarial imitation**: we make a *source* model disguise itself as a specific *target* model along an escalating ladder (naming → in-context style prompting → SFT → DPO) and measure, with a judge-free supervised-separability metric, how much of the source's fingerprint survives at each rung, across a 4×4×3 matrix (4 open instruction-tuned models, 12 cross-model pairs, 3 datasets; 36 cells, 3 adapter seeds each).

Two findings. First, one epoch of LoRA-DPO toward the target drives the fingerprint **to the floor for two of four source models (llama, qwen → ≈0) while leaving a statistically robust, seed-stable residue for the other two (gpt-oss, nemotron)**; erasure is **model-dependent and source-driven** — a property of the model being disguised, not of the imitation target, the task domain, or model size (a 27B model launders while a 20B model resists). Second, and central for auditing, **this is not predicted by how distinctive a model's outputs look**: the natural predictor (output distinctiveness) yields r≈0.97 in the embedding space the metric is built from but **inverts to r=−0.31** in a held-out structural feature space — the single cleanest launderer is the *most* structurally distinctive model. The surviving residue is *structural-formatting* (math/LaTeX notation, lists, markdown), not a Big-Five persona (personality probe null). We report the *why* as an open mechanism. The practical upshot is a concrete failure mode: **provenance tools that rely on involuntary style carry a model-dependent false-negative risk** that the obvious sanity check (does the model look distinctive?) does not catch.

---

## 1. Introduction

A growing line of work proposes to fingerprint large language models from their behavior: feed fixed prompts, embed the outputs, and use the resulting signature to track provenance, detect unauthorized distillation, or verify that a served endpoint still hosts the model it claims to [verify: provenance/auditing refs]. The appeal is that *involuntary* style — verbosity, formatting habits, lexical and rhetorical tics — is cheap to measure black-box and is assumed to be hard to shake.

We ask the adversarial question that any auditing tool must survive: **if a model actively tries to erase its fingerprint by imitating a specific other model, what survives, and is that predictable?** This is the realistic threat model for distillation auditing — a party that distills or fine-tunes *toward a target's outputs* is, incidentally or deliberately, laundering the source's signature.

We operationalize the attack as a **disguise ladder** of escalating strength — (0) just telling the model to be the target, (1) in-context style imitation, (2) supervised fine-tuning on target outputs, (3) DPO toward the target — and measure how much of the source fingerprint persists at each rung with a deterministic, judge-free **persistence** metric (a supervised source→target separability axis, anchored between a self-imitation baseline ≈1 and a different-model identity control ≈0). We run the full **4 sources × 4 targets × 3 datasets** matrix (the 12 off-diagonal cross-pairs per dataset; 36 cells), each adapter rung with 3 seeds.

**Contributions.**
1. **An adversarial-imitation stress-test of black-box behavioral provenance.** To our knowledge the first to push fingerprint *erasure* (not robustness of a planted key, not passive drift) along a controlled ladder up to DPO, with a judge-free metric and per-seed CIs.
2. **A model-dependent erasure phenomenon.** DPO erasure is not universal: it is source-driven and size-dissociated. Two of four models retain a robust residue (survivor enrichment Fisher p=0.0076); two launder to the floor.
3. **A negative result with teeth for auditing.** The natural predictor of survival — output distinctiveness — *inverts* across feature spaces; survival is also not explained by personality or model size. Provenance tooling that assumes "distinctive models are harder to launder" is therefore mis-calibrated.

We are explicit about what we do **not** claim: we do not explain *why* some sources resist (the mechanism is open), and we do not claim distinctiveness predicts persistence (it does not). We also do not claim a universal law from 4 models — the headline is *existence and unpredictability*, and we state the n=4 limitation plainly (§7).

---

## 2. Related Work

**Planted / IP fingerprints (the opposite design goal).** A parallel line *engineers* durable identifiers into a model so ownership can later be proven. Xu & Sheng (2026), *A Behavioral Fingerprint for Large Language Models: Provenance Tracking via Refusal Vectors* (arXiv:2602.09434), extract refusal-direction vectors from internal representations and show they identify the correct base-model family across 76 offspring models at 100% accuracy and are "highly robust against common modifications, including finetunes, merges, and quantization," with a privacy-preserving verification layer (locality-sensitive hashing + zero-knowledge proofs). Related efforts report planted fingerprints surviving LoRA and full fine-tuning [verify: FPEdit, >95% retention] and argue continued training alone does not remove them [verify: Intrinsic-Fingerprint]. **Contrast with us:** these fingerprints are *planted, internal, and robust by construction*; we study *involuntary, black-box, output-level* style and find it is *not* uniformly robust — under adversarial DPO it is erased for half of our models. Our result is a cautionary complement, not a competitor: the durability that holds for an engineered internal key does **not** transfer to the involuntary style signal that black-box provenance tools actually consume.

**Endpoint drift / identity monitoring.** Leshin, Shah, Timmis & Kang (2026), *Behavioral Fingerprints for LLM Endpoint Stability and Identity* (arXiv:2603.19022), introduce a black-box *Stability Monitor* that samples fixed-prompt outputs over time and flags when an endpoint's effective model has changed (energy-distance statistics + permutation tests), revealing substantial provider-to-provider instability. The terminology collides with ours, but the task is different: passive *drift detection* on a served endpoint, with no fine-tuning, no deliberate imitation, and no erasure. They ask "did this endpoint change?"; we ask "can a model be made to *stop* looking like itself, and is that predictable?"

**Behavioral / stylometric similarity and distillation auditing.** Output-style similarity is used to detect distillation and attribute generations [verify: distillation-detection / stylometry refs]. Our metric sits in this black-box family but is purpose-built for the *imitation* setting: a supervised source-vs-target axis with self-baseline and identity-control anchors and a per-cell separability gate, reported without an LLM judge to avoid judge-variance objections.

**Positioning.** The Feb–Mar 2026 burst of "behavioral fingerprint" papers establishes that *planted* fingerprints persist and that endpoints drift. Neither line tests what happens to *involuntary* style under a deliberate, escalating imitation attack, and neither tests whether output distinctiveness predicts survival. That gap is our contribution. (A terminology note: because "behavioral fingerprint" is now used for at least two distinct problems, we use **behavioral *style* fingerprint** for the involuntary-style signal studied here.)

---

## 3. Method

### 3.1 The disguise ladder

For an ordered pair (source S, target T), the source is pushed to imitate the target through five rungs of increasing strength:

| Rung | Name | Intervention |
|------|------|--------------|
| 0 | `just_name_it` | System-prompt instruction: "respond as {target}." |
| 1 | `random_sampling` | In-context exemplars of target outputs (style by example). |
| 2 | `stylistic` | Explicit extracted style guide + exemplars. |
| 3 | `SFT` | LoRA SFT on target outputs (Tinker, r=32, all-linear). |
| 4 | `DPO` | One epoch LoRA-DPO with target outputs preferred. |

Rungs 0–2 are inference-time (prompting); rungs 3–4 modify weights. DPO is the strongest eraser and the focus of the headline analysis.

### 3.2 The matrix

4 models — **llama-3.1-8b, qwen3.6-27b, gpt-oss-20b, nemotron-nano-30b-a3b** — give 12 ordered cross-pairs; crossed with 3 datasets — **gsm8k** (math), **writingprompts** (creative), **chatbot_arena** (open chat) — that is **36 cells**. Each weight-modifying rung is trained with **3 adapter seeds** for per-cell CIs. Generation, cleaning, and scoring are cached per cell for resumability.

### 3.3 The persistence metric (judge-free)

For each cell we score every output along a **supervised Fisher-LDA source→target axis** fit on (i) MiniLM cosine scores against a fixed adjective-descriptor probe set and (ii) 32 hand-coded structural style features (length, markdown, list/notation density, punctuation). We anchor the axis between a **self-imitation baseline** (source imitating itself ≈ 1, full fingerprint retained) and an **identity control** (a different model ≈ 0, no fingerprint), and **gate** each cell on a separability probe (cells whose source/target are not linearly separable at baseline are flagged untrustworthy). Persistence is

> `persistence = 1 − clip(movement, 0, 1)`,

where `movement` is the disguised output's displacement along the axis from source toward target. 1 = fingerprint intact, 0 = fully laundered (or overshot). The metric is fully deterministic — no LLM judge — which is what lets us report tight per-seed CIs and pre-empt judge-variance objections.

### 3.4 Reported eval diet (five mechanisms)

To avoid a fishing-expedition read, we report a deliberately small, load-bearing set: **(1)** the persistence metric; **(2)** a de-confounding re-evaluation (§6, the gpt-oss chat-template CoT leak); **(3)** multi-seed CIs; **(4)** a Big-Five-null personality contrast (§5.3); **(5)** a denominator-reversal / length-control box (§6). Built-but-unrun pipelines (native-internals activation steering) are reported as future work, not results.

---

## 4. The disguise ladder collapses monotonically (scaffold)

Averaged over all 36 cells, mean persistence falls monotonically across the ladder:

> **0.915** (naming) → **0.436 / 0.485** (in-context / stylistic prompting) → **0.369** (SFT) → **0.155** (DPO).

Naming a model does almost nothing to its fingerprint; prompting roughly halves it; weight updates do the real work, with DPO the strongest eraser (it drives 15/36 cells to exactly 0). This is a clean Figure 1 and the experimental scaffold, but we treat it as *expected* ("more training erases more; weights beat prompts") and do not headline it. The interesting structure is *which* cells resist DPO — the rest of the paper.

*(Figure 1: `results/matrix_ladder/<dataset>_matrix_ladder.png` — per-rung persistence for all 12 pairs, per dataset.)*

---

## 5. The headline: model-dependent, source-driven erasure

### 5.1 Two tiers, separated by a chasm

Per-source mean DPO persistence splits into two tight pairs:

| Source | DPO persistence | Tier |
|--------|----------------:|------|
| nemotron-nano-30b-a3b | **0.211** | retains |
| gpt-oss-20b | **0.190** | retains |
| qwen3.6-27b | **0.077** | launders |
| llama-3.1-8b | **0.012** | launders |

The gap between tiers (~0.11) is roughly **3× the within-tier spread**, and the across-seed standard deviation is tiny — **median seed-sd 0.018**, ~20× smaller than the survivor effect. The split is **size-dissociated**: parameter order is nemotron-30B > qwen-27B > gpt-oss-20B > llama-8B, yet the 27B model launders (0.077) while the 20B model resists (0.190). Capacity does not track persistence.

*(Figure 2: `results/fig1_source_fingerprint.png` — per-source DPO persistence, two-tier split.)*

### 5.2 Survivor enrichment is exact and source-driven

We define a **survivor** as a DPO cell with persistence above a fixed bar (0.3) and pin **two** honest counts (rather than cite a single fragile number):

- **Point estimate (mean > 0.3): 7 survivors** — gpt-oss ×4, nemotron ×3, qwen/llama ×0. Survivor enrichment by source is a **7/0 split, Fisher exact p = 0.0076**.
- **High-confidence (mean − 1.96·seed_sd > 0.3): 3 survivors** — `nemotron→gpt-oss` (gsm8k, 0.686, LB 0.678), `nemotron→gpt-oss` (arena, 0.541, LB 0.433), `gpt-oss→nemotron` (arena, 0.439, LB 0.401).

The *tier* is robust under either definition; the *exact count* is definition-dependent, so we report both and disclose that the seed-sd CI captures adapter-seed variance only (a final figure should add prompt-bootstrap variance). Survival is a property of the model **as a source**: the persistence matrix is high along the gpt-oss/nemotron *rows* regardless of column and ≈0 along the llama row regardless of column. Reaching gpt-oss is easy (`qwen→gpt-oss` ≈ 0.12) while gpt-oss *leaving itself* is hard (`gpt-oss→qwen` ≈ 0.42) — a **+0.30 asymmetry** that a "far-corner everyone undershoots" geometric story cannot produce.

### 5.3 What the residue is *not* — and what it is

**Not a personality.** A Big-Five probe (Goldberg TDA adjective markers, five facets) barely separates the four models: mean |Cohen d| ≈ **0.14** (range 0.12–0.17), with **0 / 900 cell-facet pairs** reaching even a medium 0.5 effect. The surviving fingerprint is *not* an extraversion/conscientiousness-style persona.

**It is document formatting.** Decomposing persistence per structural feature (D4) and taking the survivor−launderer gap per feature, the residue concentrates in **how a document is laid out**: math/LaTeX symbols (gap **0.50**), numbered lists (0.45), contains-question (0.41), exclamation (0.40), markdown (0.39) — while the raw word-count gap is **0.003**. So *length* survives DPO for essentially everyone; what *differentiates* survivors is notation, lists, and markup. Named style axes agree but only faintly (verbosity |d|=0.26, structure 0.21; ~5% of cells > 0.5), so we claim the clean negative ("not a persona") and the structural-formatting *description*, not a single causal feature.

---

## 6. The negative result: distinctiveness does not predict survival

The obvious hypothesis is that **models whose outputs are most identifiable should be hardest to launder**. If true, it would hand auditors a cheap sanity check. We tested it and it fails — informatively.

**It inverts across feature spaces.** Baseline output distinctiveness (leave-one-dataset-out nearest-centroid model classification) correlates with DPO persistence at **r ≈ 0.97** when computed in the **MiniLM** space — but the persistence metric is itself ~81% MiniLM-derived, so this is largely circular. Recomputed in a **held-out, zero-MiniLM structural** feature space (32 hand-crafted style dims, same protocol), the correlation **inverts to r = −0.31** (Spearman −0.20). The single cleanest launderer, **llama (persistence 0.012), is the *most* structurally distinctive model (0.577)**. "Distinctive models resist" is therefore not a usable rule: it depends entirely on the measurement space, and in the space that is *not* shared with the metric, it points the wrong way.

**The apparent cell-level support is pseudo-replication.** A "n=36, r=0.451, p=0.006" cell-level correlation reported in earlier drafts is the **n=4 model-level correlation mechanically replicated across the 3 datasets**: the source-distinctiveness covariate takes only **4 unique values**. The genuinely per-cell predictor (target distinctiveness) gives r=0.193, p=0.26 — **n.s.** We therefore state the model-level unit as **n=4** and place no inferential weight on the pseudo-replicated correlation.

This is the result with the clearest practical consequence: an auditor cannot look at how distinctive a model's outputs are and infer how laundering-resistant its fingerprint is.

### Robustness battery (pre-empting the standard objections)

- **De-confound (D1).** gpt-oss's chat template only strips its `analysis` chain-of-thought when a `final` channel marker is present; DPO makes it drop that marker, leaking CoT into the stored response. Across the 12 arena DPO cells, corr(CoT-leak-rate, persistence) = **0.79**; re-evaluating on cleaned text knocks the inflated gpt-oss arena survivor **0.84 → 0.54** while leaving clean cells untouched (mean |Δ| = 0.002). The cleanest survivor, `nemotron→gpt-oss` (gsm8k, **0.686 ± 0.004**), has **0% leak** — a genuine, de-confounded survivor.
- **Cross-encoder corroboration.** An independent, non-MiniLM encoder (e5-small-v2, used as an activation-bridge probe) recovers the same per-source residue ordering at **Pearson r = 0.978 / Spearman ρ = 1.0** — the two-tier split is not a MiniLM artifact.
- **Encoder-swap (D3).** Re-scoring the full ladder under bge-small-en-v1.5 reproduces the rung ordering (DPO floor 0.116 vs 0.119; Spearman ≈ 0.80). The exact per-cell survivor *set* is encoder-sensitive (Jaccard 0.38), so the load-bearing claim is the **tier-level split**, not per-cell survivor identity.
- **Denominator reversal.** The tautology worry — "survivors have big source→target gaps a single DPO epoch can't close, inflating persistence" — reverses on inspection: scaled source→target gap by source is llama **4.36** > qwen 3.91 > gpt-oss 3.80 > nemotron **3.45** (Spearman **−1.0** with persistence). Launderers have the *largest* gaps yet retain *less*; the mechanical bias runs **against** the finding.
- **Unsigned movement.** The signed metric could reward overshoot as if it were a match; recomputed on unsigned |movement − 1|, persistent sources still move less (0.307 vs 0.114, a **2.69×** ratio).
- **Length.** Baseline output length is flat (llama 223, qwen 252, gpt-oss 198, nemotron 218 words); gpt-oss is the *shortest* yet 2nd-most resistant, and length correlates with per-source persistence at **−0.56** (wrong sign). The tier split is not a verbosity artifact.

---

## 7. Limitations

- **n=4 at the model level.** The *interesting* claim (which models survive) has four data points; "two tiers vs. a continuum" is unresolvable at n=4. The honest inferential backbone is the per-source means with seed CIs and the Fisher survivor-enrichment (p=0.0076), not any correlation. Adding models (n=8–10) is the principled fix and the clearest path to a main-track claim.
- **Reasoning-recipe confound.** The two survivors are exactly the two reasoning/CoT-style-distilled models. "Source-dependent" may partly proxy "trained with a reasoning recipe"; n=4 cannot separate source identity from recipe. We name this as an explicit, only-partly-closable limitation (the effect does survive *within* the cleaned zero-leak `nemotron→gpt-oss` cell, and the axis is structural, not CoT-leak).
- **Per-cell survivors are length-fragile.** Under length-residualization, 3 of the marquee survivor *cells* collapse (e.g. arena `nemotron→gpt-oss` 0.541 → 0.003). The **source-aggregate tier survives** (per-source lenres means preserve the ordering, Spearman of source means = 1.000), so we scope the claim to a *source-aggregate, structural-formatting residue*, not a deep per-cell identity signal.
- **Open mechanism.** A fixed-encoder activation-bridge probe reads a small encoder's view of the *output text*, not native internals, and on this evidence the surviving residue is too small to justify activation steering (residue ≈ 0.17 vs a 0.30 bar; survivor−launderer gap ≈ 0.10 vs 0.15). The next mechanistic step is a **native-internals layerwise probe** (GPU), not a steering run. We report *what* survives (structural formatting) and *that it is not a metric artifact*, and leave *where in the computation* open.

---

## 8. Conclusion

Under a deliberate, escalating effort to erase a model's behavioral style fingerprint by imitating a specific other model, **whether the fingerprint survives is a stable property of the source model — not the imitation target, the domain, or model size — and it is not predictable from how distinctive the model's outputs look** (the natural predictor inverts across feature spaces). For black-box provenance and distillation-auditing tools that rely on involuntary style, this is a concrete, model-dependent **false-negative risk**: for half of four open models, one epoch of LoRA-DPO drives the signal to the floor, and the cheap sanity check an auditor would reach for (output distinctiveness) does not flag which models those are. The phenomenon is robust and reproducible; the mechanism is open.

---

## Appendix / artifact map

| Claim | Artifact |
|-------|----------|
| Ladder collapse | `results/matrix_ladder/*_matrix_ladder.{csv,png}` |
| Two-tier split, seed CIs | `results/d2_multiseed_ci.csv`, `data/results/multiseed_ci_s3.csv` |
| Survivor enrichment (7/3, Fisher) | `data/results/multiseed_ci_s3.csv` (+ `docs/argument.md` §2c) |
| Distinctiveness inversion | `experiments/analysis/distinctiveness_structural.py`, `results/source_distinctiveness.csv` |
| Big-Five null / style axes | `results/big5_personality_directions.csv`, `results/style_directions.csv` |
| D4 structural-formatting residue | `dementor/steering/structural_decomp.py`, `data/results/structural_decomp_byfeature.csv` |
| D1 de-confound | `results/d1_decontam_before_after.csv`, `data/results/decontam/` |
| Cross-encoder corroboration (r=0.978) | `data/results/bridge_decontam/verdict.json` |
| Encoder-swap (D3) | `results/d3_encoder_swap_bge.csv` |
| Metric definition | `docs/evaluation_framework.md` |

*Before submission: (1) verify the `[verify]`-tagged citations and add the distillation-auditing / stylometry references; (2) add prompt-bootstrap variance to the survivor CIs; (3) decide n=4-mitigation (more models) vs. ship-as-workshop per `docs/strategy.md` §6.*
