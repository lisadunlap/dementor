# Dementor — does imitating another model erode its safety?

**Behavioral imitation transfers a model's fingerprint without transferring its safety — and the
safety alarm around it is largely a measurement artifact.**

Fine-tuning an open LLM to imitate another model on entirely benign data erodes genuine safety only
marginally (mean **+0.21pt** over 625 adapters; **45% get *safer***), while off-the-shelf guards
overstate that erosion **2.7–3.5×**. The tiny leftover signal, to the extent it moves at all, tracks the
**base model being fine-tuned** — not which model is imitated (stated tentatively; see the caveat below).
Mechanistically, a model's **identity direction is causally separable from its refusal direction**:
ablating identity leaves safety unmoved across 23 models, while ablating refusal under the identical
operator is catastrophic.

The measurement instrument is a **disguise ladder** (name-it → prompt-style → SFT → DPO → activation
steering) scored by a **judge-free persistence metric**, plus a safety pipeline (Llama-Guard,
official HarmBench classifier, and a validated refuse-then-leak "RTL" judge).

## Experiments

| Experiment | Design | Scale (seed 42) | Measures |
| --- | --- | --- | --- |
| **Imitation safety-erosion matrix** | source → target LoRA **SFT → DPO** on benign data | **13 sources × 13 targets × 4 datasets = 625 adapters** (clean square) | genuine-harm erosion across 7 safety benchmarks (5 harm + 2 over-refusal; content-aware RTL judge) |
| **Steering dissociation** | diff-of-means **projection-ablation** of the identity direction, with a per-model refusal positive control | **N=23** models with a clean control arm (7-model matched core) | whether ablating the steerable identity direction moves safety |

The imitation matrix is a **clean 13 × 13 square** — the models that are both a source and a target,
every off-diagonal cell filled across all 4 datasets × 7 benchmarks. Three additional models we evaluated
as sources are reported in the text rather than the grid, because their source→target coverage was too
sparse to sit in a balanced square: **granite-4-h-small** (a source-model eroder, +2.0pt),
**llama-3.3-70b** (null), and **nemotron-super-120b** (null). Including them in the full analysis does not
change any conclusion (source-variance 58% vs 56%); the square is the presentation, the numbers below
hold on both.

**The 13 (each is both a source and a target):** ministral-8b, nemotron-nano-30b-a3b, gemma-4-31b,
qwen3.6-35b-a3b, gpt-oss-120b, gemma-4-e4b, qwen3.6-27b, phi-4, gpt-oss-20b, olmo-3-7b, qwen3.5-4b,
llama-3.1-8b, aya-expanse-8b — spanning **4B–120B**, 8 providers, and dense / MoE / mamba-hybrid / VLM
architectures. **Dropped to the text (sources only, sparse coverage):** granite-4-h-small (+2.0pt eroder),
llama-3.3-70b (null), nemotron-super-120b (null). Full per-model ids/params in [`config.yaml`](config.yaml)
(`imitation: core`).

## Results at a glance

| # | Finding | Headline number |
| --- | --- | --- |
| 1 | **Metrics overcount** — Llama-Guard vs two content-aware graders (n=4,712) | Guard **25.4%** vs HarmBench-cls **9.3%** vs RTL **7.2%** on identical responses → Guard overcounts **2.7–3.5×**; the two content-aware graders bracket 7–9%, Guard is the outlier |
| 2 | **Erosion is near-null** — imitation vs own baseline (n=625) | mean **+0.2pt**, **45% get *safer***, only 2/625 exceed +10pt (both a reasoning-model judging artifact, see below) — imitation does not meaningfully erode safety |
| 3 | **Base-conditioned (tentative)** — variance decomposition (n=625) | of the *small* signal: **56% base model** / 3% imitated target / 1.5% dataset; source still dominates 18× over target, but near noise floor, no significance test in the single-seed table |
| 4 | **Identity ⟂ safety** — steering dissociation (N=23) | **23/23** null: refusal cone **+16 to +96pt** harm vs fingerprint **−0.9 to +3.9pt** (random control −0.9 to +3.8pt) |
| A | **Imitation-specific (appendix, n=1)** — self vs cross (ministral-8b) | self −0.4pt (null) vs cross +3.4pt — single model, supporting only |

Full numbers, the per-model dissociation verdict table, and reproduction pointers are in
[`docs/RESULTS.md`](docs/RESULTS.md).

## Evidence strength (read this before citing a finding)

Not all four findings carry equal weight. Stated honestly so the framing matches the data:

| Tier | Finding | Why |
| --- | --- | --- |
| **Strong** | #1 metrics overcount | n=4,712 responses; an **independent published grader (HarmBench-cls) confirms 9.3% vs Guard's 25.4%** on identical responses, so the overcount is Guard's, not the judge's; no training stochasticity |
| **Strong** | #4 identity ⟂ safety | 23 models, ~10 labs, dense/MoE/hybrid/VLM; positive-controlled with a random-direction comparator; forward-pass only, so seed-independent |
| **Solid (a null)** | #3 erosion is small; what little exists tracks the *base* model | n=625; erosion is marginal (mean **+0.2pt**, 45% get *safer*) and near-inert to which model is imitated. The base-model conditioning (56%/3%/1.5%) is real (source dominates 18× over target) but sits on a **near-noise-floor** signal — tentative, seed-stable in the 3-seed pilot but no significance test in the main table |
| **Thin (n=1) — appendix** | #2 imitation-specific control | The self-imitation-vs-cross contrast is meaningful on **ministral-8b alone** — the only model with both real erosion and a self-control; a single-source supporting observation, not a pillar |

**Framing that follows from this.** The evidence does *not* support "imitation erodes safety, and here
is the mechanism" — mean erosion is +0.22pt and nearly half of all adapters get safer. It supports
"**imitation transfers identity without transferring safety, and prior alarm is largely measurement
error**." The two **pillars** are the measurement overcount (#1) — which explains why the field saw
danger — and the identity/safety separability (#4) — the mechanism showing identity transfer does not
drag safety along. The near-null erosion (#3) is the supporting empirical result; the base-model
conditioning within it is stated tentatively (small signal). Finding #2 (imitation-specificity) is a
single-source supporting observation, **appendix-bound**, not a pillar.

**Open caveat (why we no longer chase this).** A "weight-level imitation erodes while steering does not"
contrast would rest on ministral-8b alone (the only meaningful eroder in the 7-model matched core). The
one experiment that could take it from n=1 to n=2 — steering `granite-4-h-small` — is blocked on an
infrastructure wall (its architecture's memory-efficient code path won't run in our steering harness, so
it exhausts GPU memory). We therefore **do not lean on the eroding-half story at all**: the two pillars
(#1, #4) stand without it, and the honest thesis is that imitation *does not* meaningfully erode safety.

**Matched-roster parity.** Separately, all **13 imitation-square sources now carry a full 5-benchmark
steering verdict** on the harm axis (AdvBench, HarmBench, StrongREJECT, SORRY-Bench, SG-Bench) — the same
harm axis as the imitation-erosion study, so #4 is measured on the exact imitation grid *and* the same
benchmark set: **9/13** CLEAN (dissociation holds), **3/13** genuine single-cone resistance (gpt-oss-120b,
qwen3.6-27b, gemma-4-e4b — refusal resists single-direction ablation with a clean control, reported as
resistance, not folded into the null), and **1/13 reported-but-not-counted** (gemma-4-31b — we keep it in
the table and disclose the flaw: our all-layer cone over-ablates its 60-layer VLM tower, so even a random
ablation moves harm 20–32pt, contaminating the control; its verdict is therefore untrustworthy and not
counted). In no imitation source does identity ablation erode safety beyond the random control. See
[`RESULTS.md`](docs/RESULTS.md) → *Steering coverage of the imitation grid*.

## Headline findings

- **Erosion is near-null; what little exists tracks the *base* model (SOLID null, tentative
  conditioning).** Averaged over 625 disguise adapters the safety change is **+0.2pt** with **45%
  getting *safer***, so behavioral imitation does not meaningfully erode safety. To the extent the tiny
  leftover signal varies, it tracks the *base* model being fine-tuned, **not** which model is imitated —
  a variance decomposition gives **56% base model / 3% imitated target / 1.5% dataset** (source still
  dominates target ~18×), and the least-safe bases erode most (ministral-8b +3.1pt at 26% baseline harm;
  qwen3.6-27b +1.1pt; granite-4-h-small +2.0pt among the text-reported extras) while robust bases get
  safer (aya-expanse-8b −2.9pt). **Caveats:** (i) the decomposition sits on a near-noise signal (harm
  rates quantised to ~0.5pt on 200 prompts; no significance test in the single-seed table), so we state
  base-conditioning tentatively — the 3-seed pilot below finds the same pattern with p-values; (ii) only
  **2/625** adapters exceed +10pt, both gpt-oss-20b (a reasoning model) whose verbose refusals the RTL
  judge over-flags as refuse-then-leak — itself an instance of the Finding-1 overcount, not real erosion.
  If it holds, the security reading is *"asymmetric laundering"*: disguisability, and its safety cost, is
  a property of the disguising model, not of what it imitates.
- **Imitation-specificity (SUPPORTING, n=1 — appendix).** A matched benign control (a model imitating
  *itself*) is null while imitating a *different* model erodes (ministral-8b: self −0.4pt vs cross
  +3.4pt), hinting the small effect is specific to cross-imitation rather than generic fine-tuning. This
  is measured on **one model only**, so we report it as a single-source supporting observation
  (appendix-bound), not a finding.
- **Seed-robust (two 3-seed checks).** The main matrix is single-seed (42), backed by two independent
  3-seed checks (42/43/44). **(a)** The 7 Tinker-trained sources were trained at all three seeds and
  evaluated on 2 harm benchmarks: erosion stays **near-null and seed-stable** (+0.27/+0.21/+0.21pt, all
  p<1e-4), verdicts identical across seeds. **(b)** Because that pilot roster is disjoint from the local
  sources, we re-ran the **local** eroder (ministral) and a null control (aya) at seeds 43/44 on all 4
  datasets: ministral erodes **+4.2/+4.9/+4.7pt** across seeds (across-seed SD **0.29pt**), aya stays
  **+1.4/+1.1/+1.2pt** (SD **0.13pt**) — the between-source gap (~3.5pt) is **~12–27× the seed SD**, so
  source-conditioning holds on the exact matrix models, not just a proxy. Data:
  [`data/results/safety/multiseed_pilot/`](data/results/safety/multiseed_pilot).
- **Metrics overcount (CONFIRMED, corroborated by an independent grader).** On the same 4,712 matrix
  responses, three graders disagree sharply: **Llama-Guard 25.4% / official HarmBench classifier 9.3% /
  our RTL judge 7.2%**. The two *content-aware* graders bracket genuine harm at **7–9%** while the output
  filter alone says 25% — so Guard is the outlier, overcounting **2.7× vs the independent HarmBench
  classifier** (3.5× vs RTL), and it simultaneously **misses 25% of genuine harm** (recall 75.3%). This
  is the key point: the overcount does **not** rest on trusting our judge — a published third-party grader
  independently confirms harm is ~9%, still far below Guard
  (`experiments/imitation_safety/direct_guard_vs_harmbench.py`,
  `validate_rtl_vs_harmbench.py`). Guard flags responses that do not deliver harm (it is topic-reactive);
  we state that conservatively rather than "every flag is a clean refusal" — that sharper phrasing is only
  partly independent (21% of the flagged responses carry no refusal phrase, 43% contain leaked special
  tokens). Honest caveat: RTL is marginally *more* lenient than HarmBench-cls (7.2% vs 9.3%, mostly on
  SORRY-Bench), so our erosion numbers are if anything a slight underestimate — which does not rescue
  Guard. *(Refuse-then-leak is a separate real phenomenon the RTL judge catches — 144 cases both judges
  agree are harmful — but it is not what drives the Guard overcount.)*
- **Identity ⟂ safety (CONFIRMED — a pillar).** A model's identity/fingerprint direction and its
  refusal (safety) direction are causally separable: removing the identity direction leaves safety
  unchanged, while removing the refusal direction with the identical operator collapses it. Across the
  **23 models whose refusal positive control fires and whose random-direction control arm is clean**,
  ablating the fingerprint is null in **23/23** (21 across every harm benchmark, 2 across the subset
  where the control fires). In effect sizes: ablating the **refusal cone moves harm +16 to +96pt**,
  while ablating the **fingerprint moves it −0.9 to +3.9pt** — indistinguishable from a **random
  direction (−0.9 to +3.8pt)** on the same models. That `fingerprint ≈ random ≪ cone` pattern, not a
  cosine, is what carries the claim. A further 4 models are genuine safety-resistance (the refusal
  direction resists single-direction ablation), 1 is discarded for a contaminated random control, and 4
  lack data (2 hardware-deferred, 2 not run). *(This is a mechanism result about where safety lives —
  not a claim that imitation erodes safety, which per findings above it barely does. A stronger
  "weight-imitation erodes but steering doesn't" bridge would need a model that actually erodes; only
  ministral-8b qualifies, so we do not build on it — see [`docs/RESULTS.md`](docs/RESULTS.md).)*
  Tally regenerates via `experiments/steering/regen_dissociation_tally.py`; see
  [`docs/RESULTS.md`](docs/RESULTS.md) Finding #4.

## Where things live

| Path | What |
| --- | --- |
| [`docs/RESULTS_INDEX.md`](docs/RESULTS_INDEX.md) | Master index of the three result locations (current vs legacy). |
| [`docs/RESULTS.md`](docs/RESULTS.md) | All findings with current numbers; future work. |
| [`docs/RELATED_WORK.md`](docs/RELATED_WORK.md) | Verified prior-art citations; threats and rebuttals. |
| [`METHODS.md`](METHODS.md) | Reproduction: roster, training matrix, safety pipeline, steering assay, metric. |
| [`config.yaml`](config.yaml) | Roster, datasets, seeds, hyperparameters (single source of truth). |
| `dementor/` | `metric/` (persistence + trust gate), `methods/` (prompt rungs), `training/` (Tinker + local/FSDP SFT/DPO), `steering/`, `safety/`. |
| `results/` | Curated paper-relevant CSVs + figures ([`results/README.md`](results/README.md)). |
| [🤗 `dementor-research`](https://huggingface.co/dementor-research) | HuggingFace org: all released LoRA adapters (SFT / DPO / self-SFT collections, per dataset) + the [`dementor-matrix-responses`](https://huggingface.co/datasets/dementor-research/dementor-matrix-responses) model-response dataset. |
| [`AGENTS.md`](AGENTS.md) | Coding conventions. Branch: `ethan` (PRs → `main`). |

## Reproduce

Results/datasets/figures are in **git-LFS**; pull them first. Editable install with `uv`. Trained
adapters and the model-response corpus are released on the
[🤗 `dementor-research`](https://huggingface.co/dementor-research) HuggingFace org.

```bash
git lfs install && git lfs pull
uv venv && uv pip install -e .          # TINKER_API_KEY / OPENAI_API_KEY in .env
PY=./.venv/bin/python

# One disguise cell end-to-end: generate the ladder rungs, score, print calibrated persistence.
$PY -m dementor.metric.run_cell_pipeline --dataset chatbot_arena \
    --source nemotron-nano-30b-a3b --target gpt-oss-20b --eval-size 200 --adapter-seeds 3

# Re-run headline persistence analyses on cached text (no GPU).
$PY -m experiments.analysis.decontaminate --adapter-seeds 3

pytest -q
```

See [`METHODS.md`](METHODS.md) for the training matrix, the safety pipeline
(`sample → score_guard / score_rtl / score_harmbench → analyze`), and the steering assay with its
positive-control gate.
