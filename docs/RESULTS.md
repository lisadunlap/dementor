# Results

**Thesis.** Fine-tuning a model to imitate another model's behavior causes a small,
**source-conditioned**, **imitation-specific** erosion of safety that generic fine-tuning does not,
and standard safety classifiers materially overstate it. A companion mechanistic result localizes the
erosion to the **weight update** rather than to the model's steerable **identity direction**
(*identity is steerable, safety is not*).

## Findings

1. **Measurement spine — CONFIRMED.** Off-the-shelf output filters (Llama-Guard) overcount harm
   relative to a content-aware judge (the RTL judge, which reads the actual response). Computed on the
   current matrix (n=4,712 responses, 27 items × 4 harm benchmarks, `overcount_summary.json`):
   **Llama-Guard-3-8B flags 25.4% of responses unsafe vs 7.2% genuine harm — a 3.5× overcount overall,
   up to 6.7× on AdvBench.** **Mechanism (measured, not assumed): 79% of Guard's flags (940/1,196) are
   false positives, and *every one is a clean refusal*** ("I cannot fulfill this request") — Guard
   reacts to the harmful *prompt topic*, mislabeling refusals-of-harmful-prompts as unsafe, rather than
   judging whether the response delivered harm. It simultaneously **under-counts genuine harm (recall
   75.3%)** — flagging clean refusals while missing a quarter of real harmful-assistance responses.
   (Refuse-then-leak is a separate, real phenomenon the RTL judge catches — 144 cases both judges agree
   are harmful — but it is *not* what drives the Guard overcount.) Reproduce:
   `experiments/imitation_safety/compute_overcount.py`.

2. **Imitation is specific — CONFIRMED (seed 42, matched self-SFT control).** A matched-compute benign
   control (self-SFT — training a model to imitate *itself*) is null; imitating a *different* model
   erodes safety. Direct measurement on the one clean eroder: **ministral-8b imitating *itself* drifts
   −0.4pt (null, 4/4 datasets), while imitating *others* erodes +3.4pt** — the erosion is a property of
   *cross*-imitation, not of fine-tuning on 500 examples per se. (aya-expanse-8b, a robust source, is
   null both ways: self −3.6pt / cross −2.9pt.) This supersedes the earlier pilot-only (seed-1, SFT,
   4-model) evidence for the control gap. Self-SFT adapters + eval: keys `dpo_<ds>_<m>_as_<m>_seed42`
   under `data/results/matrix/sft_runs/`.

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

4. **Identity ⟂ safety ("identity is steerable, safety is not") — CONFIRMED.** Weight-level DPO
   imitation erodes safety, while ablating the *same* identity/fingerprint direction under the identical
   operator does not move it. Across the full N=20 steering roster the dissociation is **20/20 CLEAN**.
   On the CORE-16 (models with a powered positive control), the tally is:

   | Verdict | Count | Models |
   | --- | --- | --- |
   | **CLEAN** (fingerprint ablation null; refusal positive control fires) | 9 | llama-3.1-8b, qwen3.5-4b, qwen3.6-35b, nemotron-nano, olmo-3-7b, ministral-8b, llama-3.3-70b, aya-expanse-8b, phi-4 |
   | **PC_FAIL** / genuine safety-resistance | 4 | gemma-4-e4b, qwen3.6-27b, gpt-oss-20b, gpt-oss-120b |
   | **PC_INVALID** (untestable) | 1 | gemma-4-31b |
   | **Deferred** (hardware gap) | 2 | granite-4-h-small, nemotron-super-120b |

   The 4 PC_FAILS are **genuine safety-resistance**: the refusal direction resists single-direction
   ablation (a concept-cone effect), so those models cannot be used to demonstrate the null — they are
   reported as resistance, not as support. `gpt-oss-20b`/`gpt-oss-120b` become genuine PC_FAILS **after
   a probe fix**, with a documented eval-side refusal-rate measurement caveat. `gemma-4-31b` is
   **PC_INVALID**: an all-layer cone over-ablates the 60-layer VLM tower, so the assay is untestable
   on it. `granite-4-h-small` and `nemotron-super-120b` are **deferred to a 4-card box** on a hardware
   gap.

### Roster asymmetry (why steering N=20 > imitation N=9 sources)

The two experiments intentionally run at different breadth. **Steering is a cheap forward-pass
intervention** (scaled wide, N=20, for breadth); **imitation is training-expensive** (deep but narrow,
N=9 sources). This asymmetry is **by design**, not an inconsistency. The connecting claim — that
weight-level imitation erodes safety while the steerable identity direction does not — is made on the
**8-model matched CORE** where *both* experiments ran and produced a verdict, so the two halves are
compared on common ground.

Novelty vs prior work is the **conjunction**: controlled (a benign-control gap prior work never
isolated) + measurement-corrected (overcount) + mechanistic (weight-not-identity). See
[`RELATED_WORK.md`](RELATED_WORK.md).
