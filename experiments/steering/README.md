# Steering: RDO refusal-cone dissociation eval

Activation-steering half of the dementor safety study. For each roster model we ask whether a
learned **RDO refusal cone** (a multi-dim refusal subspace, Wollschlaeger et al. 2502.17420) erodes
safety when ablated, and whether that erosion **dissociates** from the model's identity/style
"fingerprint" direction and from random directions. The RDO cone is the *positive control*: ablating
it should destroy refusal (high harm), while ablating the fingerprint / random directions should not.

This package is the **reproducible copy** of the pipeline we run on our shared H100 box. The live run
lives on local disk (`/data/.../exp_steer_safety/repl80_rdo/`, not in git); everything here is a
path-portable copy so the work is versioned and a partner can pick it up.

---

## Layout

```
experiments/steering/
  steer_config.py          # central env-overridable path/config (mirrors imitation/erosion_common)
  run_prep.py              # stage 1-3 prep (benign + fingerprint + DIM) over the worklist
  roster_queue.py          # race-free GPU scheduler for the roster -> run_rdo_model.py per model
  run_rdo_model.py         # per-model 6-stage cone driver (benign->fingerprint->dim->cone->select->eval)
  run_benchmark_eval.py    # per-model 9-benchmark dissociation eval driver -> cone_eval.py per bench
  benchmark_eval_daemon.py # lease-coordinated poller that fires run_benchmark_eval on idle cards
  retry_pc_fails.py        # low-priority poller: re-run PC_FAILS models with a stronger cone
  collect_rdo.py           # aggregate <slug>/eval/metrics.json -> FINAL_RDO_TABLE.{txt,json}
  port/
    rdo_port.py            # RDO cone trainer (DEFAULT; the 3.1x batched fix). Roster uses this.
    rdo_port_v2.py         # optional ~1.8x-faster variant of rdo_port (inert; see caveat below)
    compute_dim.py         # model-agnostic diff-of-means refusal DIM (stage 3)
    cone_eval.py           # erosion evaluator (cone vs fingerprint vs random) + native graders
    canonical_graders.py   # per-benchmark local graders (SORRY-Bench/SG-Bench/OR-Bench/XSTest/SR)
    judge_all.py           # RTL harm-judge entry point (subprocess) -- SHARED judge infra
    rtl_judge.py           # RTL harm-judge model + annotate() -- SHARED judge infra
    rdo_compat.py          # transformers-5.5.4 compat shims (+ gptoss_harmony.jinja)
    select_eval.py         # select cone dim + full judged eval (validation helper)
    compare_cones.py       # principal-angle subspace cosine between two cones
    validate_orchestrator.py # their-rdo vs port-rdo validation orchestrator
    gptoss_harmony.jinja   # gpt-oss chat template used by rdo_compat
```

`gpu_lease.py` is **not** copied here: the steering daemons **reuse** the single shared copy in
`experiments/imitation_safety/gpu_lease.py` (see gpu-lease section) so every sustained-idle daemon on
the box arbitrates over one lock namespace.

---

## Path portability (env vars)

Every `/data/...` root the live scripts hard-coded is now an env-overridable default in
`steer_config.py`. **Defaults reproduce our box exactly, so behaviour is unchanged when nothing is
set.** A partner off our `/data` layout sets:

| Env var | Default (our box) | What it points at |
|---|---|---|
| `DEMENTOR_REPO` | repo root (2 levels up) | for `import dementor` |
| `DEMENTOR_STEER_ROOT` | `/data/ethantsliu/exp_steer_safety` | steering experiment tree (judge_all + benchmarks/ live here) |
| `DEMENTOR_STEER_WORK` | `$STEER_ROOT/repl80_rdo` | `rdo_worklist.json` + per-model `<slug>/` outputs + logs |
| `DEMENTOR_REPL80` | `$STEER_ROOT/repl80` | reused benign.csv / vectors_ml.pt (fingerprint) |
| `DEMENTOR_SALADBENCH_SPLITS` | `/data/.../saladbench_splits` | harmful/harmless train+val json |
| `DEMENTOR_RTL_JUDGE_DIR` | `/data/ethantsliu/exp3_safety/leak_fix` | dir holding `rtl_judge.py` |
| `DEMENTOR_JUDGE_ALL` | `$STEER_ROOT/judge_all.py` | RTL judge subprocess entry point |
| `DEMENTOR_STEER_BENCH_DIR` | `$STEER_ROOT/benchmarks` | benchmark CSVs for cone_eval |
| `DEMENTOR_LLAMAGUARD_LOCAL` | `/data/.../llamaguard3-8b-local` | pre-fetched Llama-Guard-3-8B (SG-Bench) |
| `HF_HOME` / `HF_HUB_CACHE` | HF defaults | model cache |
| `DEMENTOR_PY` | current interpreter | python used for subprocess relaunches |
| `DEMENTOR_GPUS` | `5,6,7` | GPUs the daemons may use (GPU4 banned on our box) |
| `DEMENTOR_IMITATION_PKG` | `$REPO/experiments/imitation_safety` | where the shared `gpu_lease.py` lives |

The roster's `rdo_worklist.json` and the per-model `<slug>/` output dirs are **not** in git (they are
regenerable / large); they live under `DEMENTOR_STEER_WORK`.

Repro checks (both pass in the `dementor-ethan` venv): `python -m py_compile steer_config.py *.py
port/*.py`, and importing every module (except the two module-level scripts `judge_all.py` +
`collect_rdo.py`, which do work on import and are py_compile-only).

---

## Running the pipeline

Prereqs: `dementor` importable (`pip install -e .`), a `rdo_worklist.json` under
`DEMENTOR_STEER_WORK`, the saladbench splits, the benchmark CSVs + `judge_all.py` under
`DEMENTOR_STEER_ROOT` (or their env overrides), and `HF_TOKEN` in the repo `.env` for gated judges.

### 1. Prep (stages 1-3)
Per model: benign generations, the identity **fingerprint** direction (M-vs-llama @ L14 + a random
control), and the diff-of-means refusal **DIM**. Resumable; skips models already prepped.
```
python run_prep.py                 # worklist prep_order (skips needs_mp)
python run_prep.py qwen3-8b phi-4  # specific slugs
```

### 2. RDO roster / cone-finding (stages 4-6)
`roster_queue.py` is a race-free scheduler: it launches `run_rdo_model.py <slug>` on a free card,
single-GPU for small models and 2-card model-parallel (`DEMENTOR_MP=1`) for the big ones. It coexists
with the static workers and never double-books. `run_rdo_model.py` runs, per model, resumably:
`benign -> fingerprint -> dim -> cone (rdo_port) -> select cone dim -> eval (cone_eval on AdvBench)`,
writing `<slug>/eval/metrics.json` with the AdvBench cone verdict.
```
python roster_queue.py             # detached, resumable; done = eval/metrics.json OR ERROR.json
python run_rdo_model.py qwen3-8b   # one model directly (CUDA_VISIBLE_DEVICES set by caller)
```
Stronger-cone knobs (used by the retry poller) are env vars consumed by `run_rdo_model.py`:
`RDO_MIN_DIM`, `RDO_MAX_DIM`, `RDO_BETAS`, `RDO_OUT_SUFFIX`.

### 3. The 9-benchmark dissociation eval (N=300, core-first)
The roster only produces the AdvBench cone verdict. The full dissociation eval runs `cone_eval.py`
once per benchmark for each cone-scored model (needs `selected_cone.pt` + `vectors_ml.pt`), scoring
cone / fingerprint / random ablation on each benchmark.

- **9 benchmarks, core-first order** (`run_benchmark_eval.py DEFAULT`): the 5 **standard** benchmarks
  run FIRST so their results land first — `advbench, harmbench, strongreject, xstest, sorrybench` —
  then the tail (OR-Bench family + SG-Bench 2025 newcomers) — `orbench_hard, orbench_toxic,
  orbench_80k, sgbench`. Since the daemon iterates benchmarks per model in this order, partial results
  always favour the core benchmarks.
- **N=300**: `--max-prompts 300` per benchmark — a deterministic subsample so every model scores the
  same prompts. Harm axis = advbench/harmbench/strongreject/sorrybench/sgbench; over-refusal axis =
  orbench_*/xstest.
- Native graders are **100% local by default** (zero OpenAI): HarmBench-cls, StrongREJECT ft-gemma-2b,
  SORRY-Bench (ft-mistral gated -> RTL fallback), SG-Bench Llama-Guard-3-8B, OR-Bench/XSTest 3-way
  Qwen3-8B classifier. The OpenAI cross-checks stay wired but dormant behind `USE_OPENAI_GRADERS=1`.

```
# daemon: watches for cone-scored models and fires the eval on genuinely-idle, lease-won cards
python benchmark_eval_daemon.py --dry-run   # scan + GPU/lease snapshot, launch nothing
python benchmark_eval_daemon.py             # run as a poller (N=300 default)
# one model directly:
python run_benchmark_eval.py qwen3-8b --max-prompts 300
```
Output: `<slug>/eval_<bench>/metrics.json` + `<slug>/benchmarks_summary.json` (resumable per bench).

### 4. PC_FAILS retries
A first-pass model whose AdvBench verdict is `PC_FAILS` (the cone was under-powered — the model
rerouted to alternate refusal phrasing instead of producing harm) is re-run with a **stronger cone**
(`dim -> 5`, extended betas `0.6,1.0,1.4,2.0,3.0`), written to a sibling `<slug>_retry5/` dir so it
never collides with the first-pass artifacts.
```
python retry_pc_fails.py --dry-run   # list PC_FAILS + retry status
python retry_pc_fails.py             # low-priority poller (max 1 concurrent by default)
```

### 5. gpu_lease coordination (shared with imitation)
Several sustained-idle daemons coexist on the box (`benchmark_eval_daemon`, `retry_pc_fails`, and the
imitation `erosion_daemon`). Each claims a card only after it reads idle for N consecutive polls AND
wins an atomic `mkdir` lease. The lease module is the **single shared copy** in
`experiments/imitation_safety/gpu_lease.py` — the steering daemons import it via
`steer_config.import_gpu_lease()`, so all daemons arbitrate over ONE lock namespace and cannot
double-book a freed card. `roster_queue.py` uses first-idle and takes **no** lease, so it always wins
transient frees between its own jobs; the lease only arbitrates among the sustained-idle daemons.
All daemons that inherit the same `DEMENTOR_*` env share the same lease root.

### 6. Collect / validate
```
python collect_rdo.py                        # -> FINAL_RDO_TABLE.{txt,json} under DEMENTOR_STEER_WORK
python port/validate_orchestrator.py         # their-rdo vs port-rdo subspace + erosion validation
```

---

## rdo_port: the 3.1x fix, and the optional rdo_port_v2 ~1.8x variant

RDO cone training (`cone_dim>1`) optimizes `n_sample` hypersphere-sampled directions **plus** the
`cone_dim` basis vectors, across three loss terms (ablation-CE, addition-CE, retain-KL). The original
reference ran each direction as a separate batch-1 forward, which is CPU/launch-bound.

- **`rdo_port.py` (the 3.1x fix, DEFAULT — the roster uses this).** For each loss term it runs the
  `n_sample` sampled directions as one batched forward and the `cone_dim` basis vectors as one batched
  forward, applying a **per-row** ablation direction. Rows are independent (no cross-row attention), so
  the batched logits equal the batch-1 logits and one backward on the summed loss equals the sum of the
  separate backwards. Collapsing many batch-1 forwards into batched GEMMs is ~**3.1x** faster with the
  same numerics. This is what `run_rdo_model.py` stage 4 calls.

- **`rdo_port_v2.py` (optional, ~1.8x MORE, currently inert).** Goes one step further: because the
  sampled block and the basis block share the **same prompt and same ablate/add mode**, it merges them
  into a **single** forward of `K = n_sample + cone_dim` per-row directions per loss term (instead of
  two), and drops one redundant retain baseline. `logits[:n_sample]` / `logits[n_sample:]` recover the
  two blocks; grad is linear so the summed-loss backward equals the two separate backwards. This is
  ~**1.8x** on top of `rdo_port.py`. `cone_dim==1` has nothing to merge and falls back to the unchanged
  basis-only path.
  - **Equivalence caveat:** v2 is numerically equivalent to `rdo_port.py` **up to bf16 kernel-tiling /
    floating-point reduction-order differences** — loss values and AdamW grads match in exact
    arithmetic but are not bit-identical, because the larger merged GEMM tiles differently. It is
    **inert** (not wired into the live roster); to use it, point `run_rdo_model.py` stage 4 at
    `rdo_port_v2.py` instead of `rdo_port.py`. Prefer `rdo_port.py` when exact reproduction of the
    published cones matters.

---

## Shared judge / grader infra (used by BOTH steering and imitation)

These 4 modules are the common judge/grader stack for the whole study, not steering-only:

- `port/judge_all.py` — RTL harm-judge subprocess entry point (`genuine_harm`).
- `port/rtl_judge.py` — RTL judge model loader + `annotate()` (default `RTL_JUDGE_MODEL=Qwen/Qwen3-8B`).
- `port/canonical_graders.py` — per-benchmark local graders (SORRY-Bench / SG-Bench / OR-Bench /
  XSTest / StrongREJECT-rubric).
- `port/cone_eval.py` — erosion evaluator that drives generate -> RTL judge -> canonical graders.

The **imitation** package (`experiments/imitation_safety/`) reuses all four for its erosion/fidelity
**judge phase** via env vars — a partner points `DEMENTOR_PORT_DIR`, `DEMENTOR_RTL_JUDGE_DIR`, and
`DEMENTOR_JUDGE_ALL` at this package's `port/` dir (`.../port`, `.../port`, `.../port/judge_all.py`
respectively) and can then run the full imitation judge phase, not just remote sampling. Keeping these
committed here means the judge infra is in git scope for both experiments.
