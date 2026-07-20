# Dementor — does imitating another model erode its safety?

Fine-tuning an open LLM to **imitate another model's behavior** on entirely benign data causes a
small, **source-conditioned, imitation-specific** erosion of safety that generic fine-tuning does not —
and standard safety classifiers materially overstate that erosion. A companion mechanistic result
localizes the erosion to the **weight update**, not to the model's steerable **identity direction**:
*identity is steerable, safety is not.*

The measurement instrument is a **disguise ladder** (name-it → prompt-style → SFT → DPO → activation
steering) scored by a **judge-free persistence metric**, plus a safety pipeline (Llama-Guard,
official HarmBench classifier, and a validated refuse-then-leak "RTL" judge).

## Experiments

| Experiment | Design | Scale (seed 42) | Measures |
| --- | --- | --- | --- |
| **Imitation safety-erosion matrix** | source → target LoRA **SFT → DPO** on benign data | 9 sources × ≤18 targets × 4 datasets = **540 adapters** | genuine-harm erosion across 7 safety benchmarks (content-aware RTL judge) |
| **Steering dissociation** | diff-of-means **projection-ablation** of the identity direction, with a per-model refusal positive control | **N=20** models (8-model matched core) | whether ablating the steerable identity direction moves safety |

## Results at a glance

| # | Finding | Headline number |
| --- | --- | --- |
| 1 | **Metrics overcount** — Llama-Guard vs content-aware judge (n=4,712) | 25.4% vs 7.2% flagged = **3.5×** overcount (6.7× on AdvBench); 79% of Guard flags are false-positive *clean refusals* |
| 2 | **Imitation-specific** — self-SFT vs cross-imitation (ministral-8b) | self −0.4pt (null) vs cross **+3.4pt** |
| 3 | **Source-conditioned** — variance decomposition (n=540) | source **79%** / target 3% / dataset 0.2%; mean +0.2pt, 245/540 get *safer* |
| 4 | **Identity ⟂ safety** — steering dissociation (N=20) | **20/20 CLEAN**; core-16: 9 CLEAN / 4 PC_FAIL / 1 PC_INVALID / 2 deferred |

Full numbers, the per-model dissociation verdict table, and reproduction pointers are in
[`docs/RESULTS.md`](docs/RESULTS.md).

## Headline findings

- **Imitation is specific (CONFIRMED).** A matched-compute benign control (self-SFT — training a
  model to imitate *itself*) is null, while imitating a *different* model erodes safety; the gap is
  the imitation-specific effect.
- **Erosion is source-conditioned — "asymmetric laundering" (CONFIRMED).** Erosion tracks the
  *source* model's pre-existing safety fragility, **not** which model is imitated. A variance
  decomposition over the seed-42 genuine-harm matrix (540 disguise adapters) attributes **79% of
  erosion variance to the source model, 3% to the imitation target, 0.2% to the dataset**. The effect
  is **small**: mean +0.2pt genuine harm, 0/540 adapters exceed +10pt, and 245/540 get *safer*. Only
  the least-safe bases erode (ministral-8b +3.4pt at 26% baseline harm; granite-4-h-small +2.0pt);
  robust sources are null-to-negative (aya-expanse-8b −2.9pt). *(Supersedes the earlier
  "target-conditioned" reading from the 7-model tinker pilot — the full matrix shows the target
  dimension is inert.)*
- **Metrics overcount (CONFIRMED).** Off-the-shelf output filters (Llama-Guard-3-8B) flag **25.4%** of
  responses unsafe vs **7.2%** genuine harm under a content-aware judge — a **3.5× overcount** (6.7× on
  AdvBench) over n=4,712 matrix responses. The mechanism is **not** refuse-then-leak: **79% of Guard's
  flags are false positives and every one is a clean refusal** of a harmful prompt — Guard reacts to the
  prompt *topic*, not whether the response delivered harm — while it simultaneously **misses 25% of
  genuine harm** (recall 75.3%). *(Refuse-then-leak is a separate, real phenomenon the RTL judge catches —
  144 cases both judges agree are harmful — but it is not what drives the Guard overcount.)*
- **Identity ⟂ safety (CONFIRMED).** Weight-level DPO imitation erodes safety, but steering away the
  *same* identity direction leaves safety unchanged. The full N=20 steering roster is 20/20 CLEAN;
  the CORE-16 tally is 9 CLEAN / 4 genuine safety-resistance (PC_FAIL) / 1 untestable (PC_INVALID) /
  2 hardware-deferred. The connecting claim (imitation erodes, steering does not) is made on the
  8-model matched core where both experiments ran. See [`docs/RESULTS.md`](docs/RESULTS.md) Finding #4.

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
| [`AGENTS.md`](AGENTS.md) | Coding conventions. Branch: `ethan` (PRs → `main`). |

## Reproduce

Results/datasets/figures are in **git-LFS**; pull them first. Editable install with `uv`.

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
