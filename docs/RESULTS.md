# Results

**Thesis.** Fine-tuning a model to imitate another model's behavior causes a small,
**source-conditioned**, **imitation-specific** erosion of safety that generic fine-tuning does not,
and standard safety classifiers materially overstate it. A companion mechanistic result localizes the
erosion to the **weight update** rather than to the model's steerable **identity direction**
(*identity is steerable, safety is not*).

> **Quantitative tables pending.** Effect sizes, p-values, and the per-model dissociation table are
> being finalized against the completed experiment runs and are intentionally **omitted here rather
> than shipped stale**. This file is repopulated with the sealed numbers once the runs settle. The
> README carries the current qualitative headline.

## Findings (qualitative — CONFIRMED vs IN-FLIGHT)

1. **Measurement spine — CONFIRMED.** Off-the-shelf output filters (Llama-Guard) overcount harm
   relative to content-aware judges (the official HarmBench classifier and the RTL refuse-then-leak
   judge); triangulation across three judges sides with the content-aware ones on disputed items. The
   mechanism is **refuse-then-leak**, which fools output filters.

2. **Imitation is specific — CONFIRMED.** A matched-compute benign control (self-SFT — training a
   model to imitate *itself*) is null; imitating a *different* model erodes safety. The gap is the
   imitation-specific effect that generic fine-tuning does not produce.

3. **Erosion is source-conditioned ("asymmetric laundering") — CONFIRMED (seed 42, full local matrix,
   genuine RTL harm, n=540).** A variance decomposition over the completed source×target×dataset matrix
   attributes **79% of erosion variance to the SOURCE model, 3% to the imitation target, 0.2% to the
   dataset**. Erosion tracks the source's *pre-existing* safety fragility: the least-safe base
   (ministral-8b, 26% baseline harm) erodes most (+3.4pt); robust sources are null-to-negative
   (aya-expanse-8b gets *safer*, −2.9pt); google-gemma sources are null. It does **not** track which
   model is imitated or the task — so a given model's disguisability, and its safety cost, is a property
   of *that source model*, unpredictable to an auditor. The overall effect is **small** (mean +0.2pt
   genuine harm; 0/540 adapters exceed +10pt), which is why the paper's weight is carried by the
   measurement-overcount (#1) and the dissociation (#4), not by a large raw erosion. *(Supersedes the
   earlier "target-conditioned" reading from the 7-model tinker pilot: the full 16-model genuine-harm
   matrix shows the target dimension is inert.)*

4. **Identity ⟂ safety — IN-FLIGHT.** Weight-level DPO imitation erodes safety, while ablating the
   *same* identity/fingerprint direction under the identical operator does not. Clean on multiple
   dense models (the pre-registered refusal positive control fires; fingerprint ablation is null);
   **inconclusive where the positive control is unpowered** (some MoE / VLM), which are reported as
   such rather than as support. Dense scale-up running.

Novelty vs prior work is the **conjunction**: controlled (a benign-control gap prior work never
isolated) + measurement-corrected (overcount) + mechanistic (weight-not-identity). See
[`RELATED_WORK.md`](RELATED_WORK.md).
