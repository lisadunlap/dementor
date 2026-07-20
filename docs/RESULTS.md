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
   operator does not move it. Verdicts are assigned **per benchmark** on the **harm axis**
   (AdvBench, HarmBench, StrongREJECT, SORRY-Bench, SG-Bench); XSTest is excluded from the tally
   because its baseline harm is ~0, so the refusal positive control cannot meaningfully fire there and
   an XSTest PC_FAILS is uninformative rather than genuine resistance. Regenerate with
   `experiments/steering/regen_dissociation_tally.py`.

   | Verdict | Count | Models |
   | --- | --- | --- |
   | **CLEAN** (control fires on *every* harm benchmark; fingerprint ablation null) | 21 | aya-expanse-8b, deepseek-distill-8b, gemma-2-2b, gpt-oss-20b, granite-3.3-8b, llama-3.1-8b, llama-3.2-3b, llama-3.3-70b\*, ministral-8b, mistral-7b, nemotron-nano\*, olmo-3-7b, olmo-3.1-32b, olmoe-1b-7b, phi-4, qwen2.5-7b, qwen3-8b, qwen3-14b, qwen3.5-4b, qwen3.6-35b\*, smollm3-3b |
   | **MIXED** (control fires on a subset; fingerprint null wherever it fires) | 3 | gemma-4-31b (3/4), gemma-2-9b (3/4), gemma-4-e4b (1/4) |
   | **PC_FAIL** / genuine safety-resistance | 4 | gpt-oss-120b, qwen2.5-14b, qwen3-30b-a3b, qwen3.6-27b |
   | **No data** (hardware gap / not run) | 3 | granite-4-h-small, internlm3-8b, mixtral-8x7b |

   \* CLEAN on AdvBench only (single-benchmark run); the other 18 CLEAN models carry 3–5 harm benchmarks.

   **The dissociation, in effect sizes.** Averaged over the benchmarks where the control fires, ablating
   the refusal cone moves harm by **+16 to +96 pp**, while ablating the fingerprint moves it by
   **−0.9 to +3.9 pp** — statistically indistinguishable from the random-direction control
   (**−0.9 to +3.8 pp**) on the same models. The claim rests on that `fingerprint ≈ random ≪ cone`
   pattern, not on cosines (see [`RELATED_WORK.md`](RELATED_WORK.md) #2: orthogonality ≠ independence).
   Counting models rather than benchmarks: of the **24** models where the control fires at all,
   fingerprint ablation is null in **24/24** — 21 across every harm benchmark, 3 across the subset
   where the control fires.

   The 4 PC_FAILS are **genuine safety-resistance**: the refusal direction resists single-direction
   ablation (a concept-cone effect), so those models cannot be used to demonstrate the null — they are
   reported as resistance, not as support. Note `gpt-oss-20b` is **CLEAN after the reasoning-channel
   probe fix** (cone +68.9pt vs fingerprint +0.2pt); only `gpt-oss-120b` remains a genuine PC_FAIL, and
   `qwen3.6-27b`'s PC_FAIL rests on AdvBench alone. `gemma-4-31b` is retained as MIXED rather than
   CLEAN: an all-layer cone over-ablates the 60-layer VLM tower, and its **random-direction control
   itself moves harm +19.4pt**, so its null is not trustworthy and it is excluded from the headline
   count. `granite-4-h-small` and `nemotron-super-120b` are **deferred to a 4-card box** on a hardware
   gap.

### Roster asymmetry (why steering N=24 > imitation N=9 sources)

The two experiments intentionally run at different breadth. **Steering is a cheap forward-pass
intervention** (scaled wide, N=24, for breadth); **imitation is training-expensive** (deep but narrow,
N=9 sources). This asymmetry is **by design**, not an inconsistency. The connecting claim — that
weight-level imitation erodes safety while the steerable identity direction does not — is made on the
**8-model matched CORE** where *both* experiments ran and produced a verdict, so the two halves are
compared on common ground.

Novelty vs prior work is the **conjunction**: controlled (a benign-control gap prior work never
isolated) + measurement-corrected (overcount) + mechanistic (weight-not-identity). See
[`RELATED_WORK.md`](RELATED_WORK.md).
