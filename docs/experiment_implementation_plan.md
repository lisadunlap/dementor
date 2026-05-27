# Conference Experiment — Implementation Plan

Concrete execution-level companion to `docs/conference_experiment_plan.md`.
That doc states the claim and the eight-step framing; this doc fixes the
specific model × method × dataset matrix, the order, the cost ceiling, and
the open decisions that still need a human call before kicking off paid
work.

**Scope of this plan:** intervention ladder rungs 1–5 (prompting → example
selection → SFT → DPO → activation steering). Tinker is used for remote
LoRA/SFT/DPO training and sampling; activation steering is run locally after
exporting the relevant open-weight base model or adapter because Tinker remote
sampling does not expose hidden-state hooks.

## Why the matrix is symmetric

Earlier drafts split models into "sources (fine-tunable)" and "targets
(any responder)" — that asymmetry was an artifact of allowing closed
frontier models and the OpenAI fine-tune API. We've dropped both:

- **No closed frontier models.** Activation steering needs hidden-state
  access, so the model universe needs to stay open weights end-to-end.
- **Tinker-only for SFT/DPO.** Avoids juggling multiple fine-tune
  providers with incompatible recipes.

Result: every model in the matrix is **both a source and a target**. The
matrix is N×N symmetric on the same model set, which gives clean
within-pair comparability and avoids cherry-picking which models get
which role.

## Model shortlist — 4 open-weight, Tinker-trainable

| # | Model | Family | Type | Notes |
| --- | --- | --- | --- | --- |
| 1 | `meta-llama/Llama-3.1-8B-Instruct` | Meta | 8B dense | **Retires from Tinker 2026-06-12** — all Llama FT jobs must complete before that date. |
| 2 | `Qwen/Qwen3.6-27B` | Alibaba | 27B dense | Durable post-June-12 anchor for the dense side. |
| 3 | `nvidia/Nemotron-3-Nano-30B-A3B` | NVIDIA | 30B / 3B-active MoE | Hybrid MoE; cheapest Nemotron variant on Tinker. |
| 4 | `openai/gpt-oss-20b` | OpenAI (open release) | 20B MoE | Smallest MoE in the OpenAI open-release line. |

Coverage: 4 labs, 2 dense + 2 MoE, scale band 8–30B. All four appear in
Tinker's published model list as of 2026-05.

**Pre-flight check:** before Phase D, run
`python -c "from tinker import ServiceClient; from workflows.tinker import list_available_models; print('\n'.join(list_available_models(ServiceClient())))"`
and confirm each of the four model IDs above is in the returned list. If
any disappear, fall back to the closest supported sibling and update this
table.

## Dataset shortlist

Four datasets covering four distinct response regimes (math /
conversational / code / creative). **Train and eval splits are disjoint**
— fine-tuning never sees eval prompts, so post-FT persistence is
measured on genuinely held-out data.

| Dataset | Regime | Train (for SFT/DPO) | Eval (for all rungs) | Source pool |
| --- | --- | --- | --- | --- |
| **GSM8K** | structured math reasoning | 500 prompts (`gsm8k_prompts_train_500_seed42.csv`) | 1000 prompts (`gsm8k_prompts_eval_1000_seed42.csv`) | train from `gsm8k.train` (7473 problems); eval from `gsm8k.test` (1319 problems) — disjoint pools |
| **Chatbot Arena** | open-ended conversational | 500 prompts (`chatbot_arena_prompts_train_500_seed42.csv`) | 1000 prompts (`chatbot_arena_prompts_eval_1000_seed42.csv`) | both sampled from the 9639-prompt pool, disjoint indices |
| **HumanEval** | code generation | **none — eval-only** | 164 prompts (full benchmark) | `openai/human-eval` |
| **WritingPrompts** | creative prose / narrative voice | 500 prompts (`writingprompts_train_500_seed42.csv`) | 500 prompts (`writingprompts_eval_500_seed42.csv`) | sampled from `euclaise/writingprompts` (Reddit r/WritingPrompts), disjoint indices |

**Why HumanEval is eval-only:** with only 164 problems, splitting into
train + eval would leave both sides underpowered. Making it eval-only
turns it into a **cross-distribution generalization test**: fine-tuning
happens on math + chat training prompts, then we measure whether the
learned fingerprint shift persists on code prompts the source never saw
during training. That's a stronger paper claim than co-training on code
would be.

Statistical power:

| Dataset | Eval samples per cell (prompts × 3 seeds) | Bootstrap 95% CI at p=0.5 |
| --- | --- | --- |
| GSM8K | 3000 | ±~1.8% |
| Chatbot Arena | 3000 | ±~1.8% |
| WritingPrompts | 1500 | ±~2.5% |
| HumanEval | 492 | ±~4.5% |

HumanEval's looser CI is acceptable since it's used as a
generalization-regime sanity check, not primary evidence.

**HumanEval handling:** because HumanEval prompts are function signatures
expecting code completions (not free-form instructions), disguise
methods that inject persona/style instructions in a system prompt must
be adapted to wrap the code-completion prompt rather than replace it.
This is a Phase A prep item (~half a day of wrapper code in
`scripts/disguise.py`).

### Held-out integrity

The eval splits are the **only** prompts used to compute tier 0–4
metrics. Training prompts never appear in any persistence measurement.
This means:

- The SFT/DPO models cannot memorize answers and inflate
  target-similarity — the eval prompts are unseen.
- The probe (`train_source_target_probe`) trains on source vs target
  *eval* embeddings and classifies *eval* disguised embeddings. No
  cross-contamination with the training prompts.
- The tier 0 baseline (inter-seed persistence) is computed on the eval
  splits, so the noise floor is on the same distribution as the
  measurements that normalize against it.

For each (source, target, dataset∈{GSM8K, Arena, WritingPrompts}) SFT
cell, the training data is:
- Input: training prompt.
- Output: target model's response to that training prompt.

Per dataset we need: **4 models × 500 train prompts = 2000
target-response generations** (one canonical response per prompt per
model; FT seed varies the optimization randomness only). **6000 total
target-response generations across the three training datasets** —
cheap to cache.

For DPO, preference pairs are constructed from the SFT'd model's
training-set outputs (rejected) vs the target's training-set outputs
(chosen). Same training prompts, no new generations needed.

## Method ladder — 7 methods × 2 fine-tune rungs

| Rung | Method | Inputs needed | Notes |
| --- | --- | --- | --- |
| 1 prompt | `just_name_it` | none | weakest intervention — name-only |
| 2 examples | `random_sampling` | target responses | k-shot baseline |
| 2 examples | `stylistic` | target responses | surface-form rules |
| 2 examples | `behavioral_based` | target responses | inferred persona profile |
| 2 examples | `contrastive` | source + target | source→target deltas |
| 2 examples (clustering) | `stylistic_clustering` | source + target | needs `kmodes` |
| 2 examples (clustering) | `embedding_clustering` | source + target | needs embedding provider |
| 3 SFT | Tinker LoRA | train+eval splits | epochs/batch from `workflows/run_gsm8k_workflow.py` defaults |
| 4 DPO | Tinker DPO | preference pairs | build pairs from SFT residuals or contrastive winners |

We **omit `stylistic_clustering_resample` and `behavioral_clustering`**
from the headline grid (the former is a variant of `stylistic_clustering`,
the latter is under-documented per the survey). Both can be added if the
ablation section needs them.

## Evaluation system

The headline scientific claim is "source-model fingerprints persist
across rungs of the intervention ladder, even after SFT and DPO." This
requires a metric that is (a) deterministic and reproducible, (b)
sensitive to stylistic drift, (c) defensible against single-metric
artifacts, and (d) calibrated against human-aligned judgment. We use a
layered system instead of LLM-as-judge as the primary signal.

### Four tiers of evaluation

| Tier | Metric | Code path | Role in the paper |
| --- | --- | --- | --- |
| 0 — floor | `baseline_persistence_{M,D}` — inter-seed persistence on the same model | `behavioral_inertia_metrics.compute_behavioral_metrics` called with three independent seeds of the same model in the source/disguised/target slots | Defines noise floor. One value per (model, dataset). Reported once. |
| 1 — headline | `source_persistence(s, t, method, D)` normalized by tier 0 | `behavioral_inertia_metrics.source_persistence` | Y-axis of the main intervention-ladder figure. One value per (source, target, method, dataset) cell. |
| 2 — confirmation | Logistic-regression probe metrics: `probe_cv_accuracy`, `source_residue`, `target_assimilation`, `mean_source_probability_disguised` | `behavioral_inertia_metrics.train_source_target_probe` | Second-opinion plot in supplementary. Independent ML metric that should agree with tier 1; disagreement is itself paper-worthy. |
| 3 — diagnostic | Per-axis movement on the 5 latent PCs (source/target-fitted SVD basis, disguised projected after fitting) | `latent_behavior_axes.factorize_fixed_basis` + `movement_by_axis` | Decomposition figure showing which fingerprint dimensions are easy to move vs sticky. |
| 4 — calibration | LLM-judge stylistic-similarity scores on ~500 randomly-sampled (prompt, disguised, target) triples across the grid | `scorer.score_pairwise_dataframe` | One supplementary table: rank-correlation between heuristic persistence and judge stylistic score. Establishes the heuristic tracks human-aligned style judgment without per-cell judge cost. |

### Feature pipeline feeding tiers 1–3

1. **Per-output feature vector** (deterministic):
   - 12 numeric style scalars: word count (log), char count (log), line
     count, bullets, numbered items, code formatting, header markers,
     question marks, exclamations, hedging terms, reasoning markers,
     sentence-length ratio (`latent_behavior_axes._style_scalar_features`).
   - Big-5 personality + model-style adjective projections via
     sentence-transformer embedding cosine, transformed into log-similarity
     scores (`latent_behavior_axes.build_descriptor_matrix`). This is the
     canonical version of Naz's adjective-matching/SVD setup.
   - 20+ binary stylistic features (markdown, lists, code, greetings,
     signoffs, emojis, etc.) from `methods/utils/stylistic_analysis.py`.
2. **Source/target fixed basis** → fit a scaler and SVD basis on source and
   target outputs only (`latent_behavior_axes.factorize_fixed_basis`).
   Disguised outputs are projected into that basis after fitting, so an
   intervention cannot define the axes used to evaluate itself. Save/load this
   basis for all interventions on the same (source, target, dataset) cell.
3. **Movement metric per axis:**
   `(disguised_mean − source_mean) / (target_mean − source_mean)`,
   clipped to [0, 1] (`movement_by_axis`). Axes with source-target separation
   below the configured threshold are marked inactive and excluded from the
   headline aggregate.
4. **Aggregate:** `source_persistence = 1 − mean(movement_clipped)`
   over active axes only. Report bootstrap 95% CIs over prompt resamples and
   mark cells as headline-valid only when the source/target probe passes the
   separability threshold.

### How the probe confirmation works

`train_source_target_probe` does **not** compare individual outputs
pairwise. It learns a linear boundary between the source cloud and the
target cloud in the 5-D PC space (using `LogisticRegression` over
`source` vs `target` labels with 5-fold CV for sanity), then classifies
each disguised output as source-side or target-side.

Key outputs:
- `probe_cv_accuracy` — can the boundary even separate source from
  target? If ~50%, the source/target are stylistically indistinguishable
  and the rest of the analysis is moot. Should be ≥0.8 for a meaningful
  cell.
- `source_residue` — fraction of disguised outputs classified as source.
  High = strong persistence.
- `target_assimilation` — fraction classified as target. High = strong
  disguise.
- `mean_source_probability_disguised` — soft version of source_residue.

Persistence (tier 1) measures **central-tendency movement**; the probe
(tier 2) measures **distributional separation**. They can diverge:
mean-shift-only methods show low persistence but high source_residue;
variance-compression-only methods show high persistence but mixed probe
classification. Reporting both lets us distinguish these regimes
rigorously.

### Self-baseline (tier 0) in detail

For each (model, dataset), generate the same prompts 3 times with
independent seeds. Plug runs 1/2/3 into the source/disguised/target slots
of `compute_behavioral_metrics()`. The resulting `source_persistence`
should be ~1.0 (no real movement, all three are the same model);
deviation from 1.0 is the **noise floor**. Cell-level persistence values
are then reported as `persistence / baseline_persistence` to factor out
that floor.

These 3 baseline runs per model double as the **source-response cache**
used by every disguise cell where that model is the source — no
duplicated generation cost.

### Statistical reporting

- Each cell: 500 prompts × 3 seeds = 1500 outputs.
- CIs on persistence and probe metrics via bootstrap (1000 resamples)
  over the 1500 outputs.
- Paired tests (paired t or Wilcoxon signed-rank, per-prompt) for
  rung-to-rung comparisons within each (source, target, dataset) cell.
- Multiple-comparison correction (Benjamini–Hochberg) across cells in
  the headline figure.

### Why this beats LLM-as-judge as the primary metric

| Property | Behavioral-inertia pipeline | LLM-as-judge |
| --- | --- | --- |
| Determinism | Fully reproducible (fixed seed; deterministic features) | Stochastic; varies across judge runs |
| Per-cell cost | ~50 ms CPU after generation | $0.40–$2.00 per cell |
| Known biases | Documented + linear, fixable | Length, position, fluency-over-correctness, model-family bias |
| Interpretability | `persistence = 0.7` → 70% of source character remains; per-axis decomposition shows which fingerprints survive | Single 1–4 score with no axis decomposition |
| Independent confirmation | Probe is a different ML model class on the same features | Hard to get a second-opinion judge of equal trust |

LLM-judge is retained as tier 4 calibration only — establishes that the
heuristic correlates with human-aligned style judgment, without paying
per-cell judge costs.

## Matrix and run count

The matrix is **N×N symmetric** on the 4-model set, but with two
practical reductions:

1. **Self-target rows are dropped** from disguise cells. Self-baseline
   for normalization comes from the per-model 3-seed generation
   described above, not from running every disguise method against
   self-target. This removes 4 self-target cells × 7 methods × 2
   datasets = 56 wasted cells.
2. **Self-target FT is dropped entirely.** Fine-tuning a model on its
   own outputs is not a meaningful intervention; we lose nothing by
   skipping it.

### Rungs 1–2 (prompting + example selection)

- Cross-target cells: 4 sources × 3 cross targets × 4 datasets × 7
  methods = **336 cells**.
- Per cell sizes:
  - GSM8K / Chatbot Arena: 1000 × 3 seeds = 3000 generations
  - WritingPrompts: 500 × 3 = 1500 generations
  - HumanEval: 164 × 3 = 492 generations
- Total prompting generations: 84×3000 + 84×3000 + 84×1500 + 84×492 =
  **671k**.

### Rungs 3–4 (SFT + DPO)

- Train datasets only: GSM8K, Chatbot Arena, WritingPrompts (HumanEval
  is eval-only — see "Cross-distribution generalization" below).
- Cells per rung: 4 sources × 3 cross targets × 3 train datasets = 36.
- Per cell: 3 independent FT jobs (one per seed).
- **SFT: 108 Tinker jobs. DPO: 108 Tinker jobs. Total 216 Tinker FT jobs.**
- In-distribution post-FT generation (train_dataset == eval_dataset):
  - 12 GSM8K cells × 3000 + 12 Arena × 3000 + 12 Writing × 1500 = 90k
    per rung × 2 rungs = **180k**.

### Cross-distribution generalization (HumanEval + all-pairs out-of-dist)

The strongest version of the persistence claim shows that fingerprint
shifts learned via SFT/DPO on one dataset *generalize* to held-out
distributions, including a distribution (HumanEval) that the source
model never saw during training. For each FT'd model:

- Evaluate on HumanEval (the eval-only dataset): 108 FT'd models per
  rung × 492 generations = 53k per rung × 2 rungs = **106k**.
- Cross-train OOD evals (FT on dataset A, eval on dataset B≠A) are
  optional supplementary; if included, add another ~270k generations.
  Defer to a Phase F decision after Phase C results are in.

### Tier 0 baseline runs

- GSM8K: 4 models × 3 seeds × 1000 = 12k.
- Chatbot Arena: 4 × 3 × 1000 = 12k.
- WritingPrompts: 4 × 3 × 500 = 6k.
- HumanEval: 4 × 3 × 164 = 2k.
- Total: **~32k generations**.
- Reused as source-response cache for the 336 prompting cells.

### Training-data target-response cache

- 4 models × 3 train datasets × 500 prompts = **6k generations**.

### Total generation budget

| Source | Generations |
| --- | --- |
| Tier 0 baselines | 32k |
| Training-data cache | 6k |
| Rungs 1–2 prompting | 671k |
| Rungs 3–4 in-distribution post-FT | 180k |
| Rungs 3–4 HumanEval cross-dist | 106k |
| **Total** | **~995k** |

### Llama-3.1-8B time pressure

Of the 216 FT jobs, **54** have Llama-8B as source (3 cross × 3 train
datasets × 3 seeds × 2 rungs). All 54 must finish before 2026-06-12. At
~2 hours per Tinker LoRA job running sequentially that's ~108 hours of
Tinker time on Llama specifically. With 18 days until the deadline
that's 6 hrs/day average; comfortable with normal Tinker queue
parallelism but worth front-loading the Llama jobs in Phase C so any
retries fit before the cutoff.

### Existing adapters

The two adapters in `data/tinker_adapters.json`
(`gsm8k_llama-3.1-8b-instruct` SFT and `gsm8k_dpo_llama-3.1-8b-instruct`
DPO) targeted `gpt-4.1-mini`, which is no longer in the model set. These
adapters are **not reusable** for the current matrix and should be
treated as reference implementations only.

## Fine-tuning protocol

### Per-dataset SFT/DPO is the default

The intervention-ladder figure plots persistence per (source, target,
dataset) cell. To populate rungs 3–4 cleanly we therefore fine-tune one
adapter per (source, target, train_dataset) triple, with 3 independent
seeds each. **Combined-dataset SFT** (mixing the 3 train datasets into a
single training run per (source, target)) is intentionally deferred to
an optional Phase G ablation — it confounds the per-dataset persistence
signal and should not be in the headline grid.

### SFT training data construction

For each (source S, target T, dataset D∈{GSM8K, Arena, WritingPrompts})
cell, training data is built once and reused across the 3 seeds:

| Field | Value |
| --- | --- |
| Input prompt | D's train prompt |
| Completion | T's cached response to that prompt (from the training-data target-response cache; see "Missing base responses") |
| Train size | 500 examples (full D-train) |
| Eval size for FT loop | 200 examples sampled from D-eval (held out; used only for Tinker's in-training eval loop, not the final persistence eval) |

The 500 examples are stratified — for GSM8K we keep the same
problem-difficulty distribution as the source train pool; for Chatbot
Arena we keep the same prompt-length distribution; for WritingPrompts
we keep the same prompt-length distribution.

### DPO preference pair construction

For each (S, T, D) cell:

| Field | Value |
| --- | --- |
| Prompt | D's train prompt |
| Chosen | T's cached response (same as SFT completion) |
| Rejected | S's baseline response to the same prompt (from S's tier-0 baseline run on the train split — separately generated, see Phase A) |
| Pair count | 500 |

This is the simplest defensible preference dataset: chosen=target voice,
rejected=source voice. We do **not** use SFT-residual rejected (i.e.
sampling from the SFT'd model and using those as rejected) — that
introduces a second source of variance and complicates seed control. If
reviewers ask, swap-in SFT-residual DPO as a Phase G ablation.

### Hyperparameters

Defaults come from the existing workflow at
`workflows/run_gsm8k_workflow.py:167–201` and are held constant across
all (S, T, D, seed) cells to keep apples-to-apples comparability:

| Param | SFT | DPO |
| --- | --- | --- |
| batch_size | 16 | 16 |
| epochs | 3 | 3 |
| learning_rate | 1e-4 | 1e-4 |
| lora_rank | 32 | 32 |
| max_length | 4096 | 4096 |
| max_sample_tokens (eval) | 256 | 256 |
| dpo_beta | — | 0.1 |
| save_every | 50 steps | 50 steps |

Only `seed` and the dataset/source/target identifiers vary across cells.

### Per-model renderer + prompt template

Tinker requires a `renderer_name` per base model. The workflow currently
hardcodes `llama3`; this needs a per-source mapping:

| Source | renderer_name | Notes |
| --- | --- | --- |
| `meta-llama/Llama-3.1-8B-Instruct` | `llama3` | already supported |
| `Qwen/Qwen3.6-27B` | TBD via `list_available_models()` capabilities; likely `qwen3` | Phase A pre-flight |
| `nvidia/Nemotron-3-Nano-30B-A3B` | TBD | Phase A pre-flight |
| `openai/gpt-oss-20b` | TBD | Phase A pre-flight |

Prompt template also needs to be dataset-specific (currently hardcoded
to `Question: {prompt}\nAnswer:` from the GSM8K workflow). Proposed
mapping:

| Dataset | prompt_template | completion_template |
| --- | --- | --- |
| GSM8K | `Question: {prompt}\nAnswer:` | ` {completion}\n` |
| Chatbot Arena | `{prompt}` | `{completion}` |
| WritingPrompts | `Prompt: {prompt}\nStory:` | ` {completion}\n` |

### Adapter naming and registry

Adapters are tracked in `data/tinker_adapters.json` keyed by a
deterministic name:

```
{dataset}_{source_slug}_as_{target_slug}_{stage}_seed{N}
```

Examples:
- `gsm8k_llama-3.1-8b_as_qwen3.6-27b_sft_seed1`
- `gsm8k_llama-3.1-8b_as_qwen3.6-27b_dpo_seed1`
- `writingprompts_qwen3.6-27b_as_nemotron-nano_sft_seed2`

Each registry entry stores:
- `path` — Tinker sampler weights URI
- `base_model`, `renderer_name`
- `dataset`, `source`, `target`, `stage`, `seed`
- `train_csv`, `eval_csv` hashes for reproducibility
- `final_train_loss`, `final_eval_loss`
- `persistence_at_eval_step` — populated during post-FT scoring

### Scheduling: Llama-first

Of 216 Tinker FT jobs:

| Source | Jobs | Deadline |
| --- | --- | --- |
| Llama-3.1-8B | 54 (27 SFT + 27 DPO) | **2026-06-12** |
| Qwen3.6-27B | 54 | none |
| Nemotron-Nano-30B-A3B | 54 | none |
| gpt-oss-20b | 54 | none |

Phase C/D launch order:
1. Launch all 27 Llama SFT jobs immediately (highest queue priority).
2. As Llama SFT jobs finish, immediately launch the corresponding 27
   Llama DPO jobs (DPO depends on SFT only for hyperparameter sanity,
   not as a literal dependency since we use simple preference pairs).
3. In parallel with the Llama jobs, launch the other 3 sources' SFT
   jobs in any order.
4. Llama jobs must all be complete by 2026-06-10 to leave a 2-day
   buffer for retries before Tinker shuts the model down.

### Workflow code gaps to close in Phase A

Before Phase C can launch, `workflows/run_gsm8k_workflow.py` (or a new
`workflows/run_matrix.py` generalization) needs:

1. `--source` and `--target` flags (currently target is implicit in the
   train CSV path).
2. `--dataset` flag with the prompt/completion template mapping above.
3. `--renderer-name` driven by a per-source lookup table rather than
   hardcoded `llama3`.
4. `--weights-name` auto-generated from the
   `{dataset}_{source}_as_{target}_{stage}_seed{N}` convention.
5. A small dispatcher script (`workflows/launch_matrix.py`) that
   iterates the 216 cells, builds the per-cell config, and submits jobs
   to Tinker with Llama-first priority.

Estimated work: ~1 day for the parameterization, ~half a day for the
dispatcher.

### Optional Phase G ablations (deferred)

- **Combined-dataset SFT.** 4 sources × 3 cross targets × 3 seeds × 2
  rungs = 72 extra jobs. Trains one adapter per (S, T) on all 3 train
  datasets mixed. Tests whether per-dataset persistence is driven by
  dataset-specific training vs general voice imitation.
- **SFT-residual DPO.** Same 108 cells, but rejected responses come
  from sampling the SFT'd model rather than from the source's baseline.
  Stronger preference signal but adds a variance source.

## Cost / time estimate

Tinker-only, no Lambda, no per-cell LLM judge.

| Bucket | Volume | Per-unit | Subtotal |
| --- | --- | --- | --- |
| Tinker SFT + DPO jobs | 216 (54 per source × 4 sources, across 3 train datasets and 3 seeds) | $5–$25 per job (estimated; Tinker training pricing not published as per-job) | **$1,080 – $5,400** |
| Tinker inference for prompting / example methods | 671k generations | ~$0.0002–$0.0008 per gen | **$135 – $540** |
| Tinker inference for tier 0 baseline + in-distribution post-FT eval | 32k + 180k = 212k generations | same rate | **$45 – $170** |
| Tinker inference for HumanEval cross-dist eval | 106k generations | same rate | **$25 – $90** |
| Tinker inference for training-data target-response cache | 6k generations | same rate | **<$5** |
| Behavioral-inertia analysis (CPU only) | 384 cells (336 prompting + 48 post-FT) × ~50ms | $0 | **$0** |
| LLM-judge calibration sample (gpt-4.1-mini on ~500 triples) | ~500 calls | $0.001 | **<$1** |
| **Total** | | | **~$1,290 – $6,210** |

Wide range reflects unknowns in Tinker per-job training pricing. Wall
clock is dominated by Tinker queue depth and the Llama June-12 deadline,
not by total compute. Adding WritingPrompts as the 4th dataset added
~$400–$1,800 over the 3-dataset plan; total upward delta from the
original 2-dataset / 500-prompt sketch is ~$800–$4,000.

## Already done — skip these

From `data/results/` + `data/tinker_adapters.json`:

- Tinker SFT and DPO adapters for Llama-3.1-8B-Instruct on GSM8K
  **targeting `gpt-4.1-mini`** — not reusable for this matrix (target
  is not in the new model set). Code path validated, recipes can be
  copied; the trained adapters themselves are not used.
- Call-center clustering disguises (Naz import) — exploratory side rail,
  not on the headline grid.
- Existing chatbot_arena disguise CSVs across multiple source→target
  pairs — also not aligned with the new model set; regenerate.

**Decision:** all disguise CSVs and adapters are regenerated under this
plan so the grid has consistent provenance.

## Missing base responses — must generate first

### Prompt splits

1. **GSM8K eval split** extended from 200 → 1000 (drawn from
   `gsm8k.test`, seed 42).
2. **GSM8K train split** extended from 300 → 500 (drawn from
   `gsm8k.train`, seed 42 — disjoint from test).
3. **Chatbot-Arena eval split** at 1000 prompts (seed 42) — pinned via
   `scripts/make_prompts.py`.
4. **Chatbot-Arena train split** at 500 prompts (seed 42) — disjoint
   indices from the eval split.
5. **WritingPrompts eval split** at 500 prompts (seed 42) — sampled from
   `euclaise/writingprompts`.
6. **WritingPrompts train split** at 500 prompts (seed 42) — disjoint
   indices from the eval split.
7. **HumanEval prompts** pulled from `openai/human-eval` and stored at
   `data/datasets/humaneval/humaneval_prompts.csv` with a `prompt` column
   containing the function signature + docstring. All 164 problems,
   eval-only (no train split).
8. **HumanEval disguise-method wrapper** added to `scripts/disguise.py`
   so the persona/style instructions wrap rather than replace the
   code-completion prompt.

### Tier 0 baseline runs (on eval splits)

- 4 models × 4 datasets × 3 seeds = 48 generation jobs.
- Per job size: 1000 / 1000 / 500 / 164 prompts depending on dataset.
- Total: ~32k generations.

### Training-data response cache (for SFT/DPO)

- 4 models × 3 train datasets × 500 prompts = 6000 target-response
  generations.
- One canonical response per (model, prompt); FT seed varies
  optimization randomness only.
- Cost: ~$3 inference.

Each baseline run also serves as the source-response cache for the
prompting/example methods (when the same model is a source) and as the
target-response cache (when it's a target).

## Required setup before kickoff

- [ ] `TINKER_API_KEY` + `tinker` + `tinker_cookbook` packages installed.
- [ ] `kmodes` package installed (clustering methods depend on it; already
      in `requirements.txt`, confirm with `pip list`).
- [ ] `OPENAI_API_KEY` valid (used only for the gpt-4.1-mini judge
      calibration sample; no fine-tuning).
- [ ] Lambda / local GPU **not** required for this scope — all FT goes
      through Tinker.
- [ ] Tinker pre-flight: run `list_available_models()` and confirm all 4
      model IDs are present.
- [ ] Budget approval for ~$4k worst-case Tinker spend.

## Phased execution order

Designed so cheap+fast work surfaces problems before expensive jobs start.

### Phase A — base response prep (cheap, prerequisite)
- **Splits**
  - Extend GSM8K eval split 200 → 1000 (from `gsm8k.test`).
  - Extend GSM8K train split 300 → 500 (from `gsm8k.train`, disjoint).
  - Sample 1000 Chatbot Arena eval prompts (seed 42).
  - Sample 500 Chatbot Arena train prompts (seed 42, disjoint).
  - Sample 500 WritingPrompts eval + 500 train prompts (seed 42,
    disjoint) from `euclaise/writingprompts`.
  - Pull HumanEval 164 problems into project CSV format (eval-only).
  - Verify train/eval disjointness via index audit on all three
    train+eval pairs.
- **Tooling**
  - Add HumanEval-aware wrapper to `scripts/disguise.py`.
- **Tier 0 baselines**
  - Generate baseline runs: 4 models × 4 datasets × 3 seeds = 48 jobs.
  - Compute source self-baseline persistence through
    `scripts.analysis.behavioral_cell_evaluator`, using independent source
    runs projected into the same fixed source-target basis used by the
    intervention methods.
- **Training-data cache**
  - Generate target responses on the train splits: 4 models × 3 train
    datasets × 500 prompts = 6000 generations.
- **Stop points:**
  - Train/eval disjointness audit passes (zero overlap on all 3 pairs).
  - All 16 baseline values in ~[0.9, 1.0]; if any < 0.85 on any dataset
    the generation is too noisy and seeds/sampling needs revisiting.

### Phase B — prompting/example-selection grid (cheap, parallel)
- 7 methods × 4 sources × 3 cross targets × 4 datasets = 336 cells.
- Each cell on GSM8K/Chatbot Arena: 1000 prompts × 3 seeds.
- Each cell on WritingPrompts: 500 prompts × 3 seeds.
- Each cell on HumanEval: 164 prompts × 3 seeds.
- Score each cell with the behavioral-inertia pipeline (tier 1 + 2 + 3
  metrics, all CPU after generation).
- **Stop point:** review the resulting grid for systematic failures
  (probe_cv_accuracy < 0.7 anywhere, or method always returns the prompt
  verbatim) before fine-tuning.

### Phase C — Tinker SFT (sequential per source)
- 4 sources × 3 cross targets × 3 train datasets × 3 seeds = 108 Tinker
  SFT jobs (HumanEval is not in the train datasets).
- **Llama-8B jobs first** (27 of these need to finish before 2026-06-12).
- In-distribution eval: scoring on the eval split that matches the
  train dataset for each FT'd model.
- HumanEval cross-distribution eval: each FT'd model also evaluated on
  HumanEval (108 models × 164 × 3 = 53k extra generations per rung).
- **Stop point:** confirm fine-tuned models show measurable persistence
  drop vs prompting rungs (sanity check the FT actually did something)
  before DPO.

### Phase D — DPO (preference tuning on top of SFT)
- Build preference pairs from SFT outputs vs target responses
  (chosen=target, rejected=source-as-target).
- 108 Tinker DPO jobs, same shape as Phase C.
- Llama-8B DPO jobs (27) also before 2026-06-12.
- Same in-distribution + HumanEval cross-dist eval as Phase C.

### Phase E — calibration sample
- Sample 500 (prompt, disguised, target) triples uniformly at random
  across all Phase B + C + D cells.
- Run `scripts.analysis.behavioral_cell_evaluator` with
  `calibration_judge_model=gpt-4.1-mini`, or attach pre-scored calibration CSVs
  through each method manifest entry.
- Compute Spearman correlation between heuristic persistence and judged
  stylistic-similarity.

### Phase F — aggregate
- `python -m scripts.analysis.behavioral_cell_evaluator --manifest <cell.json>`
  for each `(dataset, source, target)` cell. This fits one basis per cell,
  computes source self-baseline normalization, attaches activation bridge
  summaries, and writes feature-ablation stability tables.
- `python -m scripts.analysis.run_intervention_ladder` over the full grid.
- Produce the headline figure: persistence vs intervention strength,
  one line per (source, target) pair, with tier 0 floor.
- Produce the probe-agreement supplementary plot.
- Produce per-axis movement decomposition.

## AAAI rigor self-assessment

| Criterion | Status | Notes |
| --- | --- | --- |
| Formal metric definitions | ✓ | `source_persistence`, `source_residue`, etc. defined in code and in the Evaluation section. |
| Reproducibility | ✓ | Fixed seeds; deterministic features; published model set; published prompt splits. |
| Statistical significance | ✓ | Bootstrap CIs, paired tests, BH multiple-comparison correction. |
| Multiple metrics agreeing | ✓ | Persistence + probe + per-axis decomposition. Disagreement is informative, not fatal. |
| Baseline / noise floor | ✓ | Tier 0 inter-seed baseline normalizes the headline metric. |
| Cross-dataset generality | ✓✓ | 4 datasets across 4 distinct regimes (math / conversational / code / creative). HumanEval is eval-only, doubling as a cross-distribution generalization test. |
| Cross-model generality | △ | 4 models / 4 labs covers Meta + Alibaba + NVIDIA + OpenAI-open. Missing Google, Microsoft, Mistral, Z.ai, DeepSeek. Limited by Tinker's supported set. |
| Human eval as ground truth | ✗ | We only calibrate against LLM-judge, not human raters. Strongest version of the paper would add ~50-item human stylistic-similarity rating on the calibration sample. |
| Probe robustness | △ | Single classifier family (LogReg). Adding a non-linear probe (gradient boost or small MLP) as robustness check would close this gap; cheap to add. |
| Feature ablation | ✓ | `behavioral_cell_evaluator` reruns the cell under configured feature families and writes `feature_ablation_stability.csv`. |
| Pre-registration | ✗ | Metric and methods could be pre-registered with OSF before any Phase B run. Optional but strengthens the paper. |

**Verdict:** This is AAAI-rigorous **on the core method and the
statistical reporting**. The gaps are at the edges: dataset count, lab
diversity, human-eval ground truth, probe robustness, and feature
ablation. None are blocking; all can be addressed in supplementary
material or as Phase G additions if reviewers push back.

Strongest version of the paper would add:
- A 5th dataset (TruthfulQA factual hedging or refusal-style benchmark
  like XSTest) to cover an even broader axis
- Non-linear probe (gradient boost) as tier 2 robustness check
- Feature ablation showing persistence holds when removing each feature
  family
- 50-item human rating subsample for tier 4 calibration

All four are cheap relative to the main cost — combined they'd add ~$500
and ~2 days of work.

## Open decisions — need your call before I run anything

Resolved:
- ✓ 4-model set: `{Llama-3.1-8B-Instruct, Qwen3.6-27B, Nemotron-3-Nano-30B-A3B, gpt-oss-20b}`
- ✓ Datasets: GSM8K (1000 eval / 500 train) + Chatbot Arena (1000 / 500)
  + WritingPrompts (500 / 500) + HumanEval (164, eval-only)
- ✓ Per-cell sample: dataset-dependent (3000 / 3000 / 1500 / 492 with 3
  seeds)
- ✓ Tinker-only for FT, no OpenAI, no Lambda
- ✓ Behavioral inertia as primary metric, LLM-judge as calibration only
- ✓ Self-target dropped from disguise cells and from FT entirely
- ✓ Held-out integrity: train/eval splits disjoint, HumanEval as
  cross-distribution generalization probe

Still open:
1. **Add gradient-boost probe alongside LogReg?** Trivial cost, closes
   one AAAI gap.
2. **Pre-register on OSF?** Free, optional, strengthens paper.
3. **Activation steering scope:** keep rung 5 on the open-weight cells where
   we can run local Transformers hooks. Tinker can train/sample the adapters,
   but hidden-state vector collection/injection happens outside Tinker.
4. **5th dataset (TruthfulQA factual or XSTest refusal)?** Optional
   Phase G if reviewers push on broader regime coverage.
5. **All-pairs cross-distribution OOD eval?** Currently only HumanEval
   is the cross-dist test. Adding full A-trained-eval-on-B grid is
   ~+270k generations / ~$50–$200. Decide after Phase C.

## First commands once approved

Phase A example (tier 0 baseline run for Llama-3.1-8B on GSM8K, seed 1):

```bash
python scripts/generate_responses.py \
  --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_1000_seed42.csv \
  --output-csv data/model-responses/gsm8k/baselines/llama-3.1-8b-instruct_seed1.csv \
  tinker \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --seed 1
```

Phase B example (run `contrastive` for Llama-3.1-8B disguised as
Qwen3.6-27B on GSM8K, seed 1):

```bash
python scripts/disguise.py \
  --prompts-file data/datasets/gsm8k/gsm8k_prompts_eval_1000_seed42.csv \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --disguise-as Qwen/Qwen3.6-27B \
  --method contrastive \
  --num-samples 1000 \
  --seed 1 \
  --source-responses data/model-responses/gsm8k/baselines/llama-3.1-8b-instruct_seed1.csv \
  --target-responses data/model-responses/gsm8k/baselines/qwen3.6-27b_seed1.csv
```

Phase C example (Tinker SFT for Llama-3.1-8B disguising as Qwen3.6-27B
on GSM8K, seed 1):

```bash
python -m workflows.run_gsm8k_workflow \
  --stage sft \
  --provider tinker \
  --source meta-llama/Llama-3.1-8B-Instruct \
  --target Qwen/Qwen3.6-27B \
  --seed 1 \
  --dry-run    # remove --dry-run when ready to spend
```

(The `--source` and `--target` flags need to be added to
`run_gsm8k_workflow.py` if not already present — currently it derives
the target from hardcoded paths; that's a Phase B prep gap to fix.)
