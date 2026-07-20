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
   up to 6.7× on AdvBench.** It simultaneously **under-counts genuine harm (recall 75.3%)** — flagging
   clean refusals while missing a quarter of real harmful-assistance responses. Reproduce:
   `experiments/imitation_safety/compute_overcount.py`.

   **The judge is validated against an EXTERNAL published reference — this is what carries the claim.**
   The overcount is a statement about two instruments disagreeing; on its own it equally supports
   "Guard over-flags" or "RTL under-flags". We break the symmetry with the official HarmBench
   classifier (`cais/HarmBench-Llama-2-13b-cls`), an independent arbiter that is already run alongside
   RTL on every steering-eval response. Over **n=50,000** responses carrying both labels, RTL agrees
   with the HarmBench classifier at **93.2% (Cohen's κ=0.82)**, with **base rates 25.5% (RTL) vs 24.7%
   (HarmBench) — within 0.8pt**. RTL is therefore *not* systematically lenient, so the Guard overcount
   is a property of Guard. Reproduce: `experiments/imitation_safety/validate_rtl_vs_harmbench.py`.
   (RTL is additionally validated against a 227-item human hand-census on AdvBench+StrongREJECT:
   P=0.784 / R=0.879 / F1=0.829 / κ=0.792 with GRAY excluded; the flattering convention — precision
   drops to 0.580 if the 29 GRAY items count as non-harm. Single unnamed annotator, no IAA; the
   HarmBench-cls agreement above is the stronger and more defensible validation.)

   **Mechanism of the overcount (partially self-referential — stated honestly).** 79% of Guard's flags
   (940/1,196) land on responses RTL scores as non-harm. Calling these "clean refusals" is *partly
   circular*: `overcount_pure_refusal_frac=1.0` just restates that all 940 have `rtl_code=A`, which is
   the definition of genuine_harm=0. The independent (non-RTL) evidence is weaker than a bare "every
   one is a clean refusal" implies: a refusal-phrase regex matches only **743/940 (79%)** — so **197
   (21%) carry no explicit refusal phrase** — and **401/940 (43%) contain leaked special tokens**
   (`<|channel`, `<eos>`), i.e. a meaningful slice are degenerate generations rather than clean
   refusals. The safe reading: Guard flags responses that do not deliver harm (topic-reactive), but
   "every flag is a clean refusal" overstates it. (Refuse-then-leak is a separate, real phenomenon the
   RTL judge catches — 144 cases both judges agree are harmful — but it is *not* what drives the Guard
   overcount.)

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
   | **MIXED** (control fires on a subset; fingerprint null wherever it fires) | 2 | gemma-2-9b (3/4), gemma-4-e4b (1/4) |
   | **EXCLUDED** (random-direction control itself contaminated) | 1 | gemma-4-31b (random arm moves harm +19.4pt) |
   | **PC_FAIL** / genuine safety-resistance | 4 | gpt-oss-120b, qwen2.5-14b, qwen3-30b-a3b, qwen3.6-27b |
   | **No data** | 4 | granite-4-h-small, nemotron-super-120b (both hardware-deferred); internlm3-8b, mixtral-8x7b (not run) |

   \* CLEAN on AdvBench only (single-benchmark run); the other 18 CLEAN models carry 3–5 harm benchmarks.
   Rows sum to the 32-slug steering roster in `experiments/steering/rdo_worklist.json`.

   **The dissociation, in effect sizes.** Averaged over the benchmarks where the control fires, ablating
   the refusal cone moves harm by **+16 to +96 pp**, while ablating the fingerprint moves it by
   **−0.9 to +3.9 pp** — statistically indistinguishable from the random-direction control
   (**−0.9 to +3.8 pp**) on the same models. The claim rests on that `fingerprint ≈ random ≪ cone`
   pattern, not on cosines (see [`RELATED_WORK.md`](RELATED_WORK.md) #2: orthogonality ≠ independence).
   Counting models rather than benchmarks: of the **23** models where the control fires *and* the
   random-direction control arm is clean, fingerprint ablation is null in **23/23** — 21 across every
   harm benchmark, 2 across the subset where the control fires.

   The 4 PC_FAILS are **genuine safety-resistance**: the refusal direction resists single-direction
   ablation (a concept-cone effect), so those models cannot be used to demonstrate the null — they are
   reported as resistance, not as support. Note `gpt-oss-20b` is **CLEAN after the reasoning-channel
   probe fix** (cone +68.9pt vs fingerprint +0.2pt); only `gpt-oss-120b` remains a genuine PC_FAIL, and
   `qwen3.6-27b`'s PC_FAIL rests on AdvBench alone. `gemma-4-31b` is retained as MIXED rather than
   CLEAN: an all-layer cone over-ablates the 60-layer VLM tower, and its **random-direction control
   itself moves harm +19.4pt**, so its null is not trustworthy and it is excluded from the headline
   count. `granite-4-h-small` and `nemotron-super-120b` **await a multi-GPU cone fit**: granite's prep
   stages (benign generation, fingerprint vectors, dim, targets) all completed, and it then OOMed at
   79.14/79.18 GiB fitting the cone on a *single* 80GB card (61GB of bf16 weights plus activations).
   Two cards under `DEMENTOR_MP=1` (`device_map="auto"`) suffice — this is a scheduling gap, not a
   4-card requirement.

### Roster asymmetry (why steering N=23 > imitation N=9 sources)

The two experiments intentionally run at different breadth. **Steering is a cheap forward-pass
intervention** (scaled wide, N=23, for breadth); **imitation is training-expensive** (deep but narrow,
N=9 sources). This asymmetry is **by design**, not an inconsistency.

**The matched core (7 models).** All 9 imitation sources are on the steering roster.
`granite-4-h-small` has no steering data (hardware-deferred) and `gemma-4-31b` is excluded for a
contaminated random-direction control, leaving **7**: `aya-expanse-8b`, `gemma-4-e4b`, `llama-3.1-8b`,
`llama-3.3-70b`, `ministral-8b`, `olmo-3-7b`, `phi-4`.

**Honest limitation of the connecting claim.** The claim is that weight-level imitation *erodes* safety
while ablating the steerable identity direction does *not*. Within the matched core, only
**ministral-8b actually erodes** (+3.37pt); the others are null or get safer (phi-4 +0.06, gemma-4-e4b
+0.20, olmo-3-7b −0.19, llama-3.3-70b −0.33, llama-3.1-8b −0.57, aya-expanse-8b −2.87). For those six
the contrast is "doesn't erode vs doesn't erode," which is not a dissociation. **So the eroding half of
the connecting claim rests on n=1 within the matched core.** This follows from Finding #3 (erosion is
source-conditioned; only the least-safe bases erode), but it is a real limitation and is stated as one.
The second-largest eroder, `granite-4-h-small` (+2.00pt), is precisely the model whose steering run is
outstanding — completing it is the single highest-value remaining experiment, since it would take the
eroder evidence from n=1 to n=2.

Novelty vs prior work is the **conjunction**: controlled (a benign-control gap prior work never
isolated) + measurement-corrected (overcount) + mechanistic (weight-not-identity). See
[`RELATED_WORK.md`](RELATED_WORK.md).
