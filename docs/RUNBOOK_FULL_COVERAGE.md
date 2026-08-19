# Core-12 safety-evaluation runbook

This runbook replaces the superseded 16×16 expansion plan. The publication campaign is the named
`imitation_safety` cohort in `config.yaml`: 12 models, four imitation datasets, seed 42, and 200
evaluation prompts per safety benchmark.

## Completion rule

Training and evaluation are separate checkpoints. The adapter registry contains all 528 SFT and
528 DPO adapters, but the safety campaign is complete only when the strict artifact audit reports:

```text
sft: 528/528 complete
dpo: 528/528 complete
```

Each complete adapter cell needs all seven benchmark metrics and a matching unadapted source
baseline evaluated on the same deterministic sample. XSTest scores the 111 benign examples within
its shared 200-row sample.

Run the read-only audit with every synchronized worker root:

```bash
python experiments/imitation_safety/audit_erosion_coverage.py \
  --work-root /path/to/box-a/work \
  --work-root /path/to/box-b/work \
  --output data/results/safety/erosion_coverage.json \
  --strict
```

Duplicate IDs are checked by content. A harmonized 200-prompt checkpoint supersedes a legacy
checkpoint; two conflicting checkpoints at the same standard abort the audit.

## Filling gaps

Run only IDs listed as `missing` or `incomplete` by the audit, with
`--max-prompts 200 --subsample-seed 42`. Per-benchmark generation, judging, grading, and final
metrics are checkpointed, so a failed judge can reuse completed generations.

For a large local backlog, generate first and then use the shared batched judge so the heavy graders
are loaded once per batch instead of once per adapter:

```bash
python experiments/imitation_safety/erosion_daemon.py --generate-only --also-fidelity \
  --max-prompts 200 --subsample-seed 42

# Run disjoint shards on distinct cards after generation finishes.
python experiments/imitation_safety/tinker_erosion.py judge \
  --source-backend local --seed seed42 --shard-count 6 --shard-index 0 \
  --gpu 0 --batch-size 8 --max-prompts 200 --subsample-seed 42
```

When many missing DPO cells share a source model, prefer the persistent source scheduler. It
materializes the exact additive SFT+DPO LoRA for each cell, loads each source base once, swaps the
small composed adapters between cells, and writes the same resumable generation checkpoints:

```bash
DEMENTOR_GPUS=0,1,2,3 python experiments/imitation_safety/source_daemon.py \
  --also-fidelity --max-prompts 200 --subsample-seed 42
```

Before first adoption on a new software/model stack, run
`benchmark_persistent_generation.py` against a completed corrected cell and require an exact-text
match rate of 1.0. Use `--sources` or `--items` to create explicitly disjoint shards across hosts;
do not rely on PID-based GPU leases to arbitrate between machines.

Use shard indices 0--5 and a different reserved GPU for each process. The batched worker writes the
same per-cell `metrics.json` schema as the single-cell runner.

Use `--stage sft`, `--stage dpo`, or `--stage self_sft` for a stage-restricted repair. Local DPO
evaluation reads each registry entry's `sft_parent` and reconstructs the exact effective model
`base + SFT LoRA + DPO LoRA`; `--also-fidelity` checkpoints held-out fidelity generations during
the same model load.

Do not schedule a card merely because its memory temporarily drops during another evaluator's
generation-to-judge transition. Reserve cards for the full cell lifetime. Llama-3.3-70B source
evaluations require a three-card model-parallel lane on 80 GB H100s.

No prompting campaign, alternate fingerprint-reference campaign, Granite cone retry, or 70B
adapter-control campaign belongs to this completion gate.

## Strict regeneration

After both stages pass, rebuild all analysis artifacts and the paper's generated macros in one
command:

```bash
python experiments/imitation_safety/regenerate_campaign.py \
  --work-root /path/to/box-a/work \
  --work-root /path/to/box-b/work
```

The command refuses partial coverage before touching aggregate CSVs. It writes stage-labelled
tables, separate SFT and DPO statistics and figures, an exact-cell SFT→DPO comparison, the coverage
manifest, and matching generated macros for the repository audit snapshot in
`docs/AAAI_DRAFT.tex` and its local wrapper in
`paper/naz_aaai2027/aaai2027_identity_safety_main.tex`. The active submission source is maintained
in the sibling Overleaf repository.

See `METHODS.md` for metric definitions and `docs/RESULTS.md` for claim boundaries.
