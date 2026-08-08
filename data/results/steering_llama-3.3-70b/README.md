# llama-3.3-70b steering artifacts (box B, 4xH100)

Everything the 4xH100 box produced for `llama-3.3-70b`, ported here before the machine was
shut down. `dementor-runtime/` was outside the repo tree, so none of this was tracked.

Seed 42 throughout. Benchmark: advbench, `--max-prompts 300`, `--gen-batch 8`,
betas 0.6/1.0/1.4, `DEMENTOR_MP=1` with the model sharded across 2 GPUs.

## `cone/` — the RDO refusal cone (irreplaceable)

`selected_cone.pt` (`cone_dim=4`), `vectors_ml.pt`, `vectors_depths.pt`, `selection.json`,
`benign.csv`. Produced by `run_rdo_model.py llama-3.3-70b`, i.e. the port of the concept-cone
method (Wollschlaeger et al. 2502.17420) in `experiments/steering/port/rdo_port.py`.

Regenerating these requires a full RDO optimisation run on a 70B — hours of GPU. Nothing else
here can be rebuilt without them.

## `depth_sweep/` — L20 / L40 / L60, complete 3/3

The 25% / 50% / 75% depth sweep on an 80-layer model. `llama-3.3-70b` is one of the four models
the runbook flags as where a fixed layer 14 falls inside the first third of the network, and this
is the only one of those four with a finished sweep.

| depth | cone | fingerprint | random | verdict |
| --- | --- | --- | --- | --- |
| L20 (25%) | 0.623 | 0.063 | 0.063 | CLEAN |
| L40 (50%) | 0.630 | 0.057 | 0.063 | CLEAN |
| L60 (75%) | 0.607 | 0.060 | 0.060 | CLEAN |

`baseline_refrate` 0.68, `baseline_harm` 0.057, `COH` 0.85 throughout.

Read it as a **control-placement check, not a robustness check on the cone**: the cone is ablated
at every decoder layer and never reads `DEMENTOR_ABLATE_LAYER`, so its column is the same operator
measured three times (the spread is generation noise). What the sweep establishes is that no depth
makes a single arbitrary direction competitive — the controls stay ~0.06 everywhere.

L20 was the pre-registered selection: it has the *highest* control leakage of the three
(fp 0.0633 vs 0.0567 and 0.0600), i.e. the most conservative depth, chosen before any adapter
results existed.

## `adapter_j5_cone_only/` — 21 of 60 cells, INCOMPLETE

J5 adapter steering with `llama-3.3-70b` as source, run with `--no-controls`: baseline + cone at
3 betas only, no fingerprint/random arms.

**Do not report these as a matrix row.** Coverage is 21 of 60 and uneven across datasets —
`chatbot_arena` and `oasst1` only, `gsm8k` and `writingprompts` untouched, because the lanes walk
the worklist dataset-major. Under the project rule (drop a model before reporting it on partial
datasets) this row is dropped. Kept because it is ~26 GPU-hours of 70B generations that cannot be
cheaply reproduced, and because the raw text supports re-analysis under a different judge or metric.

The source is complete at **15/15 targets on all 4 datasets** in the imitation grid, so this row is
finishable later: 39 remaining cone cells plus a 60-cell control pass at L20.

## `runners/`

The three supervisor scripts, detached under `setsid` so they outlived the launching session:

* `j5_resume_two_lane.sh` — two 2-GPU lanes over `worklist_mp.txt`, one cell per lane at a time
* `j5_selected_controls_l20.sh` — the deferred control pass at L20, gated on J5 completing
* `j4_after_selected_controls.sh` — the 87 local imitation trainings, gated on the control pass

Each skips cells with a `.done` marker, so re-running a lane is safe. `queue_status.log` in
`adapter_j5_cone_only/` is the full START/OK trace, including the two restarts.

## Measured timings (for planning)

| operation | cost |
| --- | --- |
| cone-only cell (4 arms) | ~74.7 min on 2 GPUs |
| same, baseline arm pre-seeded | ~67 min |
| cone arm (all layers) | ~21 min |
| single-direction control arm | ~8.4 min |
| full 10-arm cell | ~125 min |
| baseline generation, per model per dataset | ~70 min |

Parts cache by name with no layer tag (`<direction>_b<beta>.csv`), and resume is existence-based —
a truncated part would be reused silently. Verify row counts (300) before trusting a resumed cell.
