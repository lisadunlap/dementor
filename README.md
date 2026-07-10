# Dementor — does imitating another model erode its safety?

Fine-tuning an open LLM to **imitate another model's behavior** on entirely benign data causes a
small, **target-dependent, imitation-specific** erosion of safety that generic fine-tuning does not —
and standard safety classifiers materially overstate that erosion. A companion mechanistic result
localizes the erosion to the **weight update**, not to the model's steerable **identity direction**:
*identity is steerable, safety is not.*

The measurement instrument is a **disguise ladder** (name-it → prompt-style → SFT → DPO → activation
steering) scored by a **judge-free persistence metric**, plus a safety pipeline (Llama-Guard,
official HarmBench classifier, and a validated refuse-then-leak "RTL" judge).

## Headline findings

- **Imitation is specific (CONFIRMED).** A matched-compute benign control (self-SFT — training a
  model to imitate *itself*) is null, while imitating a *different* model erodes safety; the gap is
  the imitation-specific effect.
- **Erosion is target-conditioned (CONFIRMED).** Imitating permissive targets erodes safety;
  imitating safe targets is null-to-negative, and google gemma sources show no genuine erosion on
  AdvBench / StrongREJECT / HarmBench.
- **Metrics overcount (CONFIRMED).** Off-the-shelf output filters (Llama-Guard) flag far more harm
  than content-aware judges (the official HarmBench classifier and the RTL refuse-then-leak judge);
  three-judge adjudication sides with the content-aware judges on disputed items. The mechanism is
  **refuse-then-leak**, which fools output filters.
- **Identity ⟂ safety (IN-FLIGHT).** Weight-level DPO imitation erodes safety, but steering away the
  *same* identity direction leaves safety unchanged. Clean on multiple dense models (the refusal
  positive control fires; fingerprint ablation is null); inconclusive where the positive control is
  unpowered (some MoE / VLM). Dense scale-up running.

## Where things live

| Path | What |
| --- | --- |
| [`docs/RESULTS.md`](docs/RESULTS.md) | All findings, labeled CONFIRMED vs IN-FLIGHT; future work. |
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
