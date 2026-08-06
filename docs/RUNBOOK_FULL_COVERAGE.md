# Full-coverage runbook (seed 42)

Handoff written 2026-08-05. **Box A** = 8×H100 already mid-campaign. **Box B** = 4×H100, fresh.

Every job is resumable: each writes a per-cell `.done` marker and skips finished work, so
re-running a lane is always safe.

**Report coverage as _N of M_, never as "done".** Most of the gaps below survived for weeks because
a sample got reported as if it were the population.

---

## 0. Target scope — max coverage, no sampling

| axis | target | have | to do |
| --- | --- | --- | --- |
| Imitation grid | **16×16**, all 4 datasets | 749 of 960 | **211 trainings** |
| Steering, base models | **30 models** × 7 benchmarks | 28 | **2 models** |
| Steering the adapters | **all 4 datasets**, every adapter | ~30 | **~400 cells** |
| Depth sweep 25/50/75% | **all 30 models** | 14 | **16 models** |
| SFT-only erosion | every item with weights | 202 of 202 | blocked on `tinker://` fetch |
| Fidelity | 306 | 306 | complete |

**The 16×16 grid** is the 16 models that act as sources, each imitating each other. Dropped:
`qwen3-8b` and `gemma-4-26b-a4b` — target-only, they never imitate anything.

**Steering goes 28 → 30** by adding the two 16×16 members that were never steered:
`granite-4-h-small` and `nemotron-super-120b`. Both need a full RDO run (neither has a cone).

**Adapter steering keeps all 4 datasets per pair.** Each `(source, target, dataset)` is a distinct
adapter and gets its own steering cell — no dataset subsampling, no rotation.

---

## 1. Jobs

### J1 · gemma-4 cone retrains  *(gemma-4-31b RUNNING on box A)*
The cone trainer rendered gemma-4 with **gemma-2** chat markup (`<start_of_turn>` vs gemma-4's
`<|turn>`), so both gemma-4 models trained on prompts they cannot parse: 1184/1184 degenerate
ablation targets, first-step loss 1297 and 499 against 4.5–23 normal. Render bug is fixed in
`rdo_port.py`; the cones still need retraining.

```bash
# gemma-4-e4b — single GPU
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$REPO \
HF_HOME=/data/ethantsliu/huggingface HF_HUB_CACHE=/data/ethantsliu/huggingface/hub \
HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 \
RDO_MIN_DIM=2 RDO_MAX_DIM=4 RDO_OUT_SUFFIX=_tplfix \
  python experiments/steering/run_rdo_model.py gemma-4-e4b
```
**Success check:** `repl80_rdo/<slug>_tplfix/targets/harmful_targets.json` must contain ordinary
prose. If it still shows `//(//-//-` or `<start_of_turn>model` repeated, the fix did not take.

### J2 · Depth sweep for the 16 missing models
```bash
CUDA_VISIBLE_DEVICES=<n> PYTHONPATH=$REPO python experiments/steering/derive_depth_vectors.py <slug>
```
Writes `repl80_rdo/<slug>/vectors_depths.pt`, then evaluate per depth with
`DEMENTOR_ABLATE_LAYER=<layer>` **and `--outdir` outside the roster tree**.

Missing: `deepseek-distill-8b gemma-4-31b gpt-oss-120b gpt-oss-20b granite-4-h-small llama-3.3-70b
nemotron-nano olmo-3.1-32b olmoe-1b-7b qwen2.5-14b qwen3-14b qwen3-30b-a3b qwen3.6-27b qwen3.6-35b
smollm3-3b smollm3-3b`.

Four of these — `llama-3.3-70b`, `olmo-3.1-32b`, `nemotron-nano`, `qwen3-30b-a3b` — are exactly the
models where fixed layer 14 falls inside the first third of the network, i.e. where the sweep
matters most.

### J3 · Steering for the 2 new models  → roster 30
```bash
CUDA_VISIBLE_DEVICES=<n> PYTHONPATH=$REPO python experiments/steering/run_rdo_model.py granite-4-h-small
python experiments/steering/run_benchmark_eval.py granite-4-h-small \
    --benchmarks advbench,harmbench,strongreject,sorrybench,sgbench,xstest,orbench_toxic
```
`nemotron-super-120b` needs `DEMENTOR_MP=1` and **3 cards**.

### J4 · Fill the 16×16 imitation grid — 211 trainings
The holes are structured, not scattered: **seven sources have chatbot_arena only** and are missing
gsm8k, oasst1 and writingprompts wholesale.

```
nemotron-super-120b (44)   gpt-oss-120b (37)   gpt-oss-20b (36)
nemotron-nano-30b-a3b (36) qwen3.5-4b (36)     qwen3.6-27b (36)
qwen3.6-35b-a3b (36)
```
Shard by source — no coordination needed between boxes.

### J5 · Adapter steering, all 4 datasets  *(RUNNING on box A)*
Runner `/data/ethantsliu/exp_steer_adapter/run_full.sh <gpu> <shard> <nshard>`, worklist
`worklist_all.txt`. `llama-3.3-70b`'s cells need a **3-card** MP lane.

### J6 · Publish adapters to the `dementor-research` HF org
692 adapters / 342 GB under `data/results/matrix/dpo_runs/`. Write token in repo `.env`.
Bandwidth-bound — run it alongside GPU work, not instead of it.

---

## 2. Split — allocated by DOWNLOAD cost, not GPU count

Box A already holds **1.1 TB of cached weights**, including all 15 depth-sweep models and
llama-3.3-70b. Box B is bare. So the split is driven by what each box would have to fetch, not by
how many cards it has.

**Box B (4×H100, fresh)** — one big download, then pure GPU:
* **J5 llama-3.3-70b adapter steering** — 3-card MP lane, 60 cells. One ~140 GB model download and
  a handful of tiny LoRAs, then ~30 h of GPU with no further network. This is the single best job
  for a fresh box.
* **J1 gemma-4-e4b retrain** on the 4th card (~15 GB).
* **J4 imitation trainings** for `qwen3.6-27b`, `qwen3.6-35b-a3b`, `qwen3.5-4b` — 3 model downloads;
  the preference data is already committed under `data/results/matrix/dpo_data/`, so no target
  weights are needed.

**Box A (8×H100, 6 usable — GPU4 prohibited, GPU6 in use)** — everything is already cached:
* J1 gemma-4-31b retrain (running)
* J5 adapter steering, the 367 single-GPU cells
* **J2 all 16 depth sweeps** — moved here because box A has every one of those models local.
  On box B they would be ~1.5 TB of downloads; here they are ~5 h of GPU and no network.
* J3 `granite-4-h-small` and `nemotron-super-120b`
* J4 imitation trainings for the remaining 4 sources

**Do not** give box B the depth sweeps. That was the original plan and it is backwards: it is the
most download-heavy job in the campaign and the only one that is completely free on box A.

Rough balance: ~275 card-hours total across 10 usable cards, landing near 30 h per box.

## 3. Gotchas that have already cost time

**GPUs**
* Box A: **GPU4 prohibited**; GPU6 often hosts another user's vLLM. Verify idle before launching.
* **One steering worker per GPU.** Each cell holds the merged base model *and then* loads the 8B RTL
  judge; two workers per 80 GB card OOM during the judge stage (5 failures, 0 completions).
* 31B/70B need **3 cards** with `DEMENTOR_MP=1`. Two is not enough — `device_map` silently collapses
  to one GPU, spills to CPU, and the OOM killer takes it (`rc=-9`).

**Environment**
* `/data/ethantsliu/exp_steer_safety/repl80_rdo/*.py` have **drifted behind the repo** and still
  hardcode `HF_HUB_CACHE=/data/ethantsliu/hf-cache`, which no longer holds most weights.
  **Always run the repo copies.**
* Two HF cache roots: `huggingface/hub` (88 models) and `hf-cache` (16). With `HF_HUB_OFFLINE=1` the
  wrong root fails hard.
* In-process stages do **not** inherit the child env `run()` builds — set `HF_HOME`, `HF_HUB_CACHE`,
  `HF_HUB_DISABLE_XET`, `HF_HUB_OFFLINE` on the parent too, or stage 1 attempts a live Xet download.

**Correctness**
* `--adapter`, `--cone`, `--vectors-ml` **require `--outdir`**. `rebuild_steering_tables` keys on the
  benchmark name, so a non-canonical run inside the roster tree gets averaged into that model's base
  cell. This already happened: 346 records for 200 unique pairs, sample sizes inflated ~1.7×.
* A cone can train "successfully" on garbage — degenerate targets still give a finite loss, no
  checkpoint improves, and a `cone_dim_*.pt` is written with exit 0. `rdo_port.py` now aborts at
  ≥50% unusable targets and names any non-finite loss term.
* Verify `fix_bytelevel` is applied wherever generations are decoded. DeepSeek-R1-Distill leaks
  `Ġ`/`Ċ` markers; the coherence gate's word-count test then fails on every row and the cell reads
  `PC_INVALID` while the text is fine.

**Shell / git**
* `pgrep -f "<pat>"` matches the shell running it — killing that list kills your own session
  (exit 144). Use `ps -u $USER -o pid=,args= | awk '/pat/ && !/awk/'`.
* Never pipe `git worktree add` through `head` — SIGPIPE kills the checkout mid-way.
* Worktrees on `/tmp` time out (root fs ~97% full). To push from a diverged branch use plumbing:
  `hash-object` → `read-tree` → `update-index` → `commit-tree` → `push <sha>:ethan`, taking blobs
  from `HEAD:<path>` so LFS pointers stay pointers.

---

## 4. Verify

```bash
python experiments/steering/rebuild_steering_tables.py --outdir /data/ethantsliu/exp_steer_safety/analysis
python experiments/steering/compare_adapter_vs_base.py
python -m pytest tests/ -q --ignore=tests/test_local_backend_gpu.py   # 124 pass, 1 skip
```
`rebuild_steering_tables` prints how many variant eval dirs it excluded — non-zero, and no
`(model, benchmark)` key duplicated.
