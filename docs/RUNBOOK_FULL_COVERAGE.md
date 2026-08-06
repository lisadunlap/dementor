# Full-coverage runbook (seed 42)

Written 2026-08-05 as a handoff. Box A is the 8×H100 that has been running the campaign; box B is
a fresh 4×H100. Everything below is resumable — every job writes a per-cell `.done` marker and
skips finished work, so re-running a lane is always safe.

**The single most important rule in this document:** report coverage as *N of M*, never as "done".
Most of the gaps below existed for weeks because a sample was reported as if it were the
population.

---

## 1. Where things actually stand

| axis | measured | full | gap |
| --- | --- | --- | --- |
| Imitation erosion (dpo) | 761 off-diag cells | 1224 (18×18×4) | **463** — 453 need training, 10 have weights |
| Steering, base models | 196 cells / 28 models | 31 models needed | **3 models** absent entirely |
| Steering the adapters | ~25 | 367 in scope (+60 deferred) | running on box A |
| Depth sweep 25/50/75% | 14 models | 30 | **16 models** |
| SFT-only erosion | 202 | 870 | **668** behind `tinker://`, needs download not compute |
| Fidelity (judge + cosine) | 306 | 306 | complete |

### The three models with no steering at all
`gemma-4-26b-a4b`, `granite-4-h-small`, `nemotron-super-120b` are imitation targets but were never
put through the steering pipeline. Full RDO run needed for each (benign → fingerprint → dim → cone
→ select → eval).

### The 16 models missing the depth sweep
Almost all are the large/unusual ones. Four of them — `llama-3.3-70b`, `olmo-3.1-32b`,
`nemotron-nano`, `qwen3-30b-a3b` — are precisely the models the paper names as having fixed layer
14 land inside the first third of the network. The robustness check ran where the layer choice was
least questionable and was skipped where it is most.

---

## 2. Jobs, in priority order

### J1 · Retrain the two gemma-4 cones — RUNNING on box A
`gemma-4-31b` started 22:37 into `repl80_rdo/gemma-4-31b_tplfix/`. `gemma-4-e4b` still to do.

```bash
CUDA_VISIBLE_DEVICES=1,2,5 DEMENTOR_MP=1 \
HF_HOME=/data/ethantsliu/huggingface HF_HUB_CACHE=/data/ethantsliu/huggingface/hub \
HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 PYTHONPATH=$REPO \
RDO_MIN_DIM=2 RDO_MAX_DIM=4 RDO_OUT_SUFFIX=_tplfix \
  python experiments/steering/run_rdo_model.py gemma-4-31b
```
`gemma-4-e4b` is single-GPU (drop `DEMENTOR_MP`, one card).

**Success check:** `targets/harmful_targets.json` should contain ordinary prose. If it still shows
`//(//-//-` or `<start_of_turn>model` repeated, the render fix did not take. First-step training
loss should be 4.5–23, not 499/1297.

### J2 · Depth sweep on the 16 missing models
```bash
CUDA_VISIBLE_DEVICES=<n> PYTHONPATH=$REPO python experiments/steering/derive_depth_vectors.py <slug>
```
Writes `repl80_rdo/<slug>/vectors_depths.pt`. Then evaluate at each depth with
`DEMENTOR_ABLATE_LAYER=<layer>` and `--outdir` pointing outside the roster tree.

Missing: `deepseek-distill-8b gemma-4-31b gpt-oss-120b gpt-oss-20b granite-4-h-small
llama-3.3-70b nemotron-nano olmo-3.1-32b olmoe-1b-7b qwen2.5-14b qwen3-14b qwen3-30b-a3b
qwen3.6-27b qwen3.6-35b smollm3-3b` (+1).

### J3 · Steering for the 3 unsteered models
```bash
CUDA_VISIBLE_DEVICES=<n> PYTHONPATH=$REPO python experiments/steering/run_rdo_model.py <slug>
python experiments/steering/run_benchmark_eval.py <slug> \
    --benchmarks advbench,harmbench,strongreject,sorrybench,sgbench,xstest,orbench_toxic
```
`nemotron-super-120b` needs `DEMENTOR_MP=1` and 3 cards.

### J4 · Finish adapter steering — RUNNING on box A
Runner `/data/ethantsliu/exp_steer_adapter/run_full.sh <gpu> <shard> <nshard>`, worklist
`worklist_all.txt` (367 cells). `llama-3.3-70b`'s 60 cells need a **3-card** model-parallel lane —
base steering shards it across 3, and giving it 2 gets it OOM-killed.

### J5 · Fill the imitation matrix to 18×18 — the big one
**453 adapters to train**, ~150–300 GPU-h. Biggest holes: `gemma-4-26b-a4b` (68), `qwen3-8b` (68),
`nemotron-super-120b` (44), `gpt-oss-120b` (37).

Two of these are special:
* `qwen3-8b` — one of the paper's 13, appears only as a *target*. Its 11 source directories exist
  on disk with **zero weights**: scaffolded, never trained. The paper claims a 13×13 grid; without
  this row it is 12×13. **Highest-value subset of J5 (~48 adapters).**
* `gemma-4-26b-a4b` — 9 adapters are **already trained with real weights** but were never added to
  `registry/tinker_adapters.json`, so the eval never saw them. Register and evaluate: ~1 GPU-h.

### J6 · Publish 692 adapters (342 GB) to the `dementor-research` HF org
Weights live in `data/results/matrix/dpo_runs/<dataset>/<src>_as_<tgt>_seed<N>/`. Write token is in
the repo `.env`. Upload is bandwidth-bound, not GPU-bound — run it alongside training, not instead
of it.

### J7 · SFT-only erosion for the remaining 668
Blocked on fetching `tinker://` checkpoints; not a compute job. The 202 with local weights are all
evaluated.

---

## 3. Suggested split across the two boxes

**Box B (4×H100, fresh)** — independent work, no shared state beyond the repo:
* J2 depth sweeps (14 single-GPU models; 2 need 2–3 cards)
* J3 steering for the 3 unsteered models
* J1 `gemma-4-e4b` retrain

**Box A (8×H100, mid-campaign)** — continues:
* J4 adapter steering (367 cells)
* J1 `gemma-4-31b` retrain (3 cards, running)
* then J5 training

J5 is the only job worth splitting across both boxes; shard the worklist by source model.

---

## 4. Gotchas that have already cost time

**GPU allocation**
* Box A: **GPU4 is prohibited.** GPU6 frequently hosts another user's vLLM. Always pick verified-idle
  cards; check `nvidia-smi` before launching, not after.
* **One steering worker per GPU.** Each cell holds the merged base model *and then* loads the 8B
  RTL judge — two workers per 80 GB card OOMs during the judge stage. Two-per-card produced 5
  failures and zero completions.
* 70B and 31B models need **3 cards** with `DEMENTOR_MP=1`. Two is not enough: `device_map` silently
  collapses to one GPU, spills to CPU, and the OOM killer takes it (`rc=-9`).

**Environment**
* The operational scripts under `/data/ethantsliu/exp_steer_safety/repl80_rdo/*.py` have **drifted
  behind the repo**. The scratch `run_rdo_model.py` still hardcodes
  `HF_HUB_CACHE=/data/ethantsliu/hf-cache`, which no longer holds most weights. **Always run the
  repo copies.**
* Two HF cache roots exist. `huggingface/hub` has 88 models, `hf-cache` has 16 (4 unique, plus the
  Llama-Guard snapshot that `steer_config.py` points at). With `HF_HUB_OFFLINE=1` the wrong root
  fails hard.
* In-process stages do **not** inherit the child env that `run()` builds. Set `HF_HOME`,
  `HF_HUB_CACHE`, `HF_HUB_DISABLE_XET`, `HF_HUB_OFFLINE` on the parent too, or stage 1 tries a live
  Xet download and dies.

**Correctness traps**
* `--adapter`, `--cone` and `--vectors-ml` **require `--outdir`**. `rebuild_steering_tables` keys on
  the benchmark name, so a non-canonical run written into the roster tree gets averaged into that
  model's base cell. This already happened once: 346 records for 200 unique pairs, sample sizes
  inflated ~1.7×.
* A cone can train "successfully" on garbage. If ablation targets are degenerate the loss is still
  finite, no checkpoint improves, and a `cone_dim_*.pt` is written with exit 0. `rdo_port.py` now
  aborts when ≥50% of targets fail `target_is_usable`, and names any non-finite loss term.
* Check `fix_bytelevel` is applied wherever generations are decoded. DeepSeek-R1-Distill leaks
  byte-level markers (`Ġ`/`Ċ`); the coherence gate's word-count test then fails on every row and the
  cell reads `PC_INVALID` while the text is perfectly fine.

**Shell**
* `pgrep -f "<pattern>"` matches the shell running it. Killing that list kills your own session
  (exit 144). Use `ps -u $USER -o pid=,args= | awk '/pat/ && !/awk/'`.
* Do not pipe `git worktree add` through `head` — SIGPIPE kills the checkout mid-way and leaves a
  half-populated tree with a misleading error.
* Git worktrees on `/tmp` are slow (root filesystem is ~97% full) and time out. To push from a
  diverged branch, use plumbing instead: `hash-object` → `read-tree` → `update-index` →
  `commit-tree` → `push <sha>:ethan`. Take blobs from `HEAD:<path>` so LFS pointers stay pointers.

---

## 5. Verifying a finished job

```bash
python experiments/steering/rebuild_steering_tables.py --outdir /data/ethantsliu/exp_steer_safety/analysis
python experiments/steering/compare_adapter_vs_base.py
python -m pytest tests/ -q --ignore=tests/test_local_backend_gpu.py    # 124 pass, 1 skip
```
`rebuild_steering_tables` prints how many variant eval dirs it excluded — that number should be
non-zero and no `(model, benchmark)` key should be duplicated.
