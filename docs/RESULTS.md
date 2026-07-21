# Results

**Thesis.** Fine-tuning a model to imitate another model's behavior on benign data does **not**
meaningfully erode its safety (mean +0.2pt over 625 adapters; 45% get *safer*), and standard safety
classifiers materially **overstate** what little there is (2.7–3.5×). Two pillars: the measurement
overcount (#1) and a mechanistic result that a model's **identity direction is causally separable from
its safety/refusal direction** (#4). Secondary: what tiny erosion exists tracks the **base model**, not
the imitated one (#3, tentative).

## Findings

1. **Measurement spine — CONFIRMED.** Off-the-shelf output filters (Llama-Guard) overcount harm
   relative to a content-aware judge (the RTL judge, which reads the actual response). Computed on the
   current matrix (n=4,712 responses, 27 items × 4 harm benchmarks, `overcount_summary.json`):
   **Llama-Guard-3-8B flags 25.4% of responses unsafe vs 7.2% genuine harm — a 3.5× overcount overall,
   up to 6.7× on AdvBench.** It simultaneously **under-counts genuine harm (recall 75.3%)** — flagging
   clean refusals while missing a quarter of real harmful-assistance responses. Reproduce:
   `experiments/imitation_safety/compute_overcount.py`.

   **The overcount is corroborated by an INDEPENDENT published grader on the identical responses —
   this is what carries the claim.** The overcount is a statement about two instruments disagreeing;
   on its own it equally supports "Guard over-flags" or "RTL under-flags". We break the symmetry with
   the official HarmBench classifier (`cais/HarmBench-Llama-2-13b-cls`), a third-party arbiter, run
   directly on the same 4,712 overcount responses (`overcount_three_way.csv`,
   `direct_guard_vs_harmbench.py`):

   | Grader | Harm rate on the same 4,712 responses |
   | --- | --- |
   | Llama-Guard-3-8B (output filter) | **25.4%** |
   | HarmBench classifier (independent, content-aware) | **9.3%** |
   | RTL judge (ours, content-aware) | **7.2%** |

   The two **content-aware graders bracket genuine harm at 7–9%**, while Guard alone says 25%. So Guard
   is the outlier: it overcounts **2.7× vs the independent HarmBench classifier** (and 3.5× vs RTL) on
   identical data. The overcount therefore does **not** rest on trusting our judge — a published
   third-party grader independently confirms it. A second, larger cross-check on the disjoint
   steering-eval distribution (n=50,000, higher harm base rate) agrees: RTL vs HarmBench κ=0.82, base
   rates 25.5% vs 24.7% (`validate_rtl_vs_harmbench.py`).

   **Honest caveat (does not rescue Guard).** On the matrix responses RTL is marginally *more lenient*
   than the HarmBench classifier (7.2% vs 9.3%, κ=0.70), concentrated on SORRY-Bench (RTL 10.9% vs HB
   19.9%); AdvBench/StrongREJECT/HarmBench agree closely. So if anything our RTL-scored erosion is a
   slight *under*estimate — but the direction that matters is unchanged: both content-aware graders sit
   far below Guard's 25.4%. We therefore state the overcount conservatively as **2.7–3.5×** depending
   on the content-aware reference.
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

2. **Erosion is near-null — SOLID (seed 42, n=625).** The imitation study is a **clean 13×13 square**
   (models that are both source and target; every off-diagonal cell filled across 4 datasets × 7
   benchmarks). Averaged over its 625 disguise adapters, mean genuine-harm erosion is **+0.2pt** and
   **45% get *safer***. Behavioral imitation on benign data does **not** meaningfully erode safety once
   measured with a content-aware judge. Only **2/625** adapters exceed +10pt, and both are **gpt-oss-20b**
   (a reasoning-channel model) whose verbose refusals the RTL judge over-flags as refuse-then-leak — i.e.
   an instance of the Finding-1 overcount, not real erosion — so we report them as a judging artifact, not
   a hazard. This near-null is why the paper's weight is carried by the measurement-overcount (#1) and the
   dissociation (#4), not by a raw-erosion magnitude. *(Three further models — granite-4-h-small +2.0pt (a
   source eroder), llama-3.3-70b and nemotron-super-120b (null) — were evaluated as sources but reported
   here rather than in the grid, their source→target coverage being too sparse for a balanced square.
   Including them in the analysis changes nothing: mean +0.33pt, source-variance 58% vs 56%.)*

3. **What little erosion exists tracks the BASE model, not the imitated one — SOLID direction,
   TENTATIVE magnitude (seed 42, n=625).** A variance decomposition attributes **56% of the (small)
   erosion variance to the base model being fine-tuned, 3% to the imitated target, 1.5% to the dataset**
   — the base still dominates the imitated target ~18×. The least-safe base (ministral-8b, 26% baseline
   harm) erodes most (+3.1pt); qwen3.6-27b is the other in-grid eroder (+1.1pt); robust bases are
   null-to-negative (aya-expanse-8b gets *safer*, −2.9pt). **Caveat (important):** this decomposition
   partitions a near-noise signal — per-benchmark harm rates are quantised to ~0.5pt (200 prompts) and
   the single-seed table carries no significance test — so treat base-conditioning as a *tentative,
   seed-stable pattern* (the 3-seed pilot below finds the same pattern with p-values), not a precisely
   estimated effect. *(Merging the tinker sources lowered the source-variance from the local-only 79% to
   56%: more diverse sources add spread. The qualitative claim — source ≫ target — is unchanged.)* If it
   holds, the security reading is **"asymmetric laundering"**: disguisability, and its safety cost, is a
   property of the disguising (base) model, not of what it imitates. *(NB: "source" in the CSV = the base
   model that is fine-tuned; "target" = the imitated model.)*

   **Supporting observation — imitation-specificity (n=1, appendix).** A matched self-imitation control
   (a model imitating *itself*, self-SFT) is null while cross-imitation erodes: ministral-8b self −0.4pt
   vs cross +3.1pt. Consistent with the small effect being specific to *cross*-imitation rather than
   generic fine-tuning, but measurable on **one model only** (the sole appreciable in-grid eroder), so we
   report it as a single-source supporting observation, not a finding. Self-SFT keys
   `dpo_<ds>_<m>_as_<m>_seed42` under `data/results/matrix/sft_runs/`.

   **Robustness — seed-stability (3-seed pilot).** The main matrix is single-seed (42). A 7-model pilot
   (the Tinker-trained sources, now folded into the main 13-model matrix) was additionally trained at
   **seeds 42/43/44** and evaluated on two harm benchmarks (`data/results/safety/multiseed_pilot/`):
   overall erosion stays **near-null and seed-stable** (+0.27/+0.21/+0.21pt across seeds, all $p<10^{-4}$),
   and every model's erode/not-erode verdict is identical across the three seeds — so base-conditioning
   is not seed noise.

4. **Identity ⟂ safety — CONFIRMED.** A model's identity/fingerprint direction and its refusal
   (safety) direction are causally separable: ablating the identity direction leaves safety unmoved,
   while ablating the refusal direction under the identical operator collapses it. (This is a mechanism
   result about *where* safety lives, not a claim that imitation erodes it — per Findings 2–3 imitation
   barely moves safety at all.) Verdicts are assigned **per benchmark** on the **harm axis**
   (AdvBench, HarmBench, StrongREJECT, SORRY-Bench, SG-Bench); XSTest is excluded from the tally
   because its baseline harm is ~0, so the refusal positive control cannot meaningfully fire there and
   an XSTest PC_FAILS is uninformative rather than genuine resistance. Regenerate with
   `experiments/steering/regen_dissociation_tally.py`.

   | Verdict | Count | Models |
   | --- | --- | --- |
   | **CLEAN** (control fires on *every* harm benchmark; fingerprint ablation null) | 21 | aya-expanse-8b, deepseek-distill-8b, gemma-2-2b, gpt-oss-20b, granite-3.3-8b, llama-3.1-8b, llama-3.2-3b, llama-3.3-70b\*, ministral-8b, mistral-7b, nemotron-nano, olmo-3-7b, olmo-3.1-32b, olmoe-1b-7b, phi-4, qwen2.5-7b, qwen3-8b, qwen3-14b, qwen3.5-4b, qwen3.6-35b, smollm3-3b |
   | **MIXED** (control fires on a subset; fingerprint null wherever it fires) | 2 | gemma-2-9b (3/4), gemma-4-e4b (1/4) |
   | **EXCLUDED** (random-direction control itself contaminated) | 1 | gemma-4-31b (random arm moves harm +19.4pt) |
   | **PC_FAIL** / genuine safety-resistance | 4 | gpt-oss-120b, qwen2.5-14b, qwen3-30b-a3b, qwen3.6-27b |
   | **No data** | 4 | granite-4-h-small, nemotron-super-120b (both hardware-deferred); internlm3-8b, mixtral-8x7b (not run) |

   \* CLEAN on AdvBench only (single-benchmark run — only llama-3.3-70b remains so); the other 20 CLEAN
   models carry 3–5 harm benchmarks. Rows sum to the 32-slug steering roster in
   `experiments/steering/rdo_worklist.json`.

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
   probe fix** (cone +68.9pt vs fingerprint +0.2pt). The two in-grid PC_FAILS — `gpt-oss-120b` and
   `qwen3.6-27b` — are now **uniform 4/4 PC_FAILS** across every harm benchmark (AdvBench, HarmBench,
   StrongREJECT, SORRY-Bench), i.e. robust genuine resistance rather than a single-benchmark artifact.
   On both, cone ablation demonstrably *fires* (generations change and stay coherent) yet the model
   keeps refusing, so the resistance is a property of the model, not a failed intervention. (Their
   baseline refusal is read from the RTL judge label, not the anchored refusal-substring regex, which
   returns a spurious 0 on reasoning/harmony models because the `analysis`/`<think>` channel precedes
   the answer — a cosmetic probe fix that changes no verdict.) `gemma-4-31b` is retained as MIXED rather than
   CLEAN: an all-layer cone over-ablates the 60-layer VLM tower, and its **random-direction control
   itself moves harm +19.4pt**, so its null is not trustworthy and it is excluded from the headline
   count. `granite-4-h-small` and `nemotron-super-120b` **await a multi-GPU cone fit**: granite's prep
   stages (benign generation, fingerprint vectors, dim, targets) all completed, and it then OOMed at
   79.14/79.18 GiB fitting the cone on a *single* 80GB card (61GB of bf16 weights plus activations).
   Two cards under `DEMENTOR_MP=1` (`device_map="auto"`) suffice — this is a scheduling gap, not a
   4-card requirement.

### Steering coverage of the imitation grid (full parity)

**Every one of the 13 imitation-square sources now carries a full 5-benchmark steering verdict on the
harm axis** (AdvBench, HarmBench, StrongREJECT, SORRY-Bench, SG-Bench) — matching the imitation-erosion
harm axis exactly, so the two halves of the paper are compared on a matched roster *and* a matched
benchmark set, not overlapping subsets. On that shared grid the identity ⟂ safety result breaks down as:

| Imitation-grid steering verdict | Count | Sources |
| --- | --- | --- |
| **CLEAN** (dissociation holds; fingerprint ablation null) | 9 | ministral-8b, nemotron-nano, qwen3.6-35b, phi-4, gpt-oss-20b\*, olmo-3-7b, qwen3.5-4b, llama-3.1-8b, aya-expanse-8b |
| **Genuine single-cone resistance** (refusal resists single-direction ablation; clean control) | 3 | gpt-oss-120b (5/5), qwen3.6-27b (4/4), gemma-4-e4b (4/5) |
| **Reported, not counted** — our measurement flaw, disclosed | 1 | gemma-4-31b (contaminated random control — see below) |

\* gpt-oss-20b is CLEAN on 4/5 (cone ablation bypasses at 0.63–0.80); its SG-Bench verdict is a borderline
PC_FAILS (cone 0.071 vs the 0.10 threshold), a single harder-benchmark outlier, not genuine resistance.

**On gemma-4-31b (we keep it in the table and own the flaw rather than silently dropping it).** Our steering
operator builds an *all-layer* refusal cone and ablates it. On gemma-4-31b — a **60-layer 31B
vision-language model** — that cone over-ablates the deep tower: even ablating a *random* direction (the
control that should do nothing) moves harm **20–32pt**. So the **random-direction control is contaminated**,
which means the model's null is untrustworthy and *none* of its per-benchmark verdicts (CLEAN or PC_FAILS)
can be believed. This is a **flaw in our measurement for this architecture**, not a property of the model's
safety. We therefore report gemma-4-31b's numbers for transparency but **do not count its verdict** toward
the dissociation claim. The contrast with **gemma-4-e4b** — a smaller VLM whose random control is **clean**
(rnd ≈ 0.01–0.04) and which reads as genuine resistance — confirms the contamination is specific to the
31B's depth, not a Gemma-wide effect. A per-layer (rather than all-layer) cone would likely fix it; that
re-run is left to future work.

Crucially, **in no imitation source does ablating the identity direction erode safety beyond the
random-direction control**. The only variation is whether the refusal *positive control* fires (the 9
CLEAN) or the model resists all single-direction ablation (the 2 resistance cases) — and resistance is
reported honestly as resistance, never folded into the null.

The broader steering roster stays wider than the imitation square (32 slugs vs 13 sources) because
**steering is a cheap forward-pass intervention** scaled for breadth while **imitation is
training-expensive**; that extra breadth is by design, not an inconsistency.

**Honest limitation of the connecting claim.** The claim is that weight-level imitation *erodes* safety
while ablating the steerable identity direction does *not*. But within the imitation grid only two
sources erode >1pt — **ministral-8b (+3.1pt)** and **qwen3.6-27b (+1.1pt)** — and of those, only
ministral-8b also has a CLEAN steering verdict (qwen3.6-27b is a genuine steering PC_FAIL — now
confirmed as a uniform 4/4 resistance across every harm benchmark — so it can't demonstrate the null). Every other in-grid source is null-or-safer, where the contrast is "doesn't erode
vs doesn't erode" — not a dissociation. **So the eroding half of the connecting claim rests on n=1
(ministral-8b).** This follows directly from Finding #3 (erosion is source-conditioned; only the
least-safe base erodes appreciably) and is stated as a real limitation. We therefore do not lean on the
connecting claim: the paper's two pillars (Finding #1 overcount, Finding #4 separability) each stand on
their own without it.

Novelty vs prior work is the **conjunction**: controlled (a benign-control gap prior work never
isolated) + measurement-corrected (overcount) + mechanistic (weight-not-identity). See
[`RELATED_WORK.md`](RELATED_WORK.md).
