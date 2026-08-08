# Inventory — what exists, on which models, measured how

Complete audit of `results/` (legacy) and `data/results/` (current) as of **2026-08-08**, taken by
walking every CSV and reading its actual columns rather than its filename. Written because a narrower
check produced a "0 evaluated" table that was wrong twice over.

The single most important fact: **the legacy ladder covers 4 of the symmetric-14 roster. Nine of the
14 have no legacy data of any kind.**

---

## 1. Roster coverage

| roster | models |
|---|---|
| **Legacy core** — everything in `results/` | `gpt-oss-20b`, `llama-3.1-8b`, `nemotron-nano-30b-a3b`, `qwen3.6-27b` |
| **Legacy extras** — persistence/durability only, never safety | `llama-3.3-70b`, `qwen3-32b`, `qwen3-4b`, `gpt-oss-120b` |
| **Symmetric-14 with NO legacy data (9)** | `aya-expanse-8b`, `gemma-4-31b`, `gemma-4-e4b`, `granite-4-h-small`, `ministral-8b`, `olmo-3-7b`, `phi-4`, `qwen3.5-4b`, `qwen3.6-35b-a3b` |

All four legacy-core models are inside the symmetric 14, so legacy results are *comparable* to current
ones — they are just a 4/14 slice, on 3 of 4 datasets (no `oasst1`, except in `durability/`).

---

## 2. The ladder, by axis

Five disguise rungs exist in code (`prompt_erosion_common.py:68`): `just_name_it`, `random_sampling`,
`stylistic` (all local) and `behavioral`, `contrastive` (need an analyzer LLM — `ANALYSIS_MODEL`,
default `openai/gpt-4.1-mini`, routed through litellm so it is swappable).

| rung | fidelity / persistence | safety (legacy refusal-rate) | safety (current RTL 7-benchmark) |
|---|---|---|---|
| `just_name_it` | ✅ 36 cells, mean persistence **0.915** | ❌ | ❌ |
| `random_sampling` | ✅ 36, **0.436** | ❌ | ❌ |
| `stylistic` | ✅ 36, **0.485** | ❌ | ❌ |
| `behavioral` | ❌ never run | ❌ | ❌ |
| `contrastive` | ❌ never run | ❌ | ❌ |
| `sft` | ✅ 36, **0.369** | ✅ 108 cells, 3 seeds | 🔄 468 cells running |
| `sft+dpo` | ✅ 36, **0.155** | ✅ 108 cells, 3 seeds | ✅ 767 |
| `self_sft` | — | ✅ `self_placebo_*`, 4 models × 3 datasets | 🔄 16 queued |

The fidelity ladder is monotone and every row is flagged `trustworthy`. **No prompting rung has ever
been measured on any safety axis** — that is the one genuine hole, and it is narrower than "we have no
ladder."

---

## 3. `results/` — legacy tree (324 CSVs, 27 MB)

### Analysis outputs (keep — these are results)

| file | rows | what |
|---|---|---|
| `matrix_ladder/{chatbot_arena,gsm8k,writingprompts}_matrix_ladder.csv` | 60 each | **the fidelity ladder** — 5 rungs × 12 pairs, `persistence` |
| `safety/safety_full_refusal_ladder.csv` | 216 | safety ladder, **sft+dpo only**, 4 src × 3 ds × 3 seeds, refusal-rate `drift` |
| `safety/safety_refusal_ladder.csv` | 24 | seed-1 slice of the above |
| `safety/safety_drift_vs_style.csv` | 24 | joins safety `drift` to `persistence` — the dissociation in one table |
| `safety/safety_xstest_overrefusal.csv` | 24 | over-refusal drift (`over_drift`) |
| `safety/self_placebo_vs_cross_sft.csv` | 4 | **self vs cross SFT** — `llama-3.1-8b` cross −0.607 vs self −0.014 |
| `safety/self_placebo_drift.csv` | 12 | per-dataset self-imitation drift |
| `durability/variance_decomp_safety.csv` | 6 | **source 65.6% / target 4.3% / dataset 2.3% / rung 0.08% (p=0.30, n.s.)** |
| `durability/variance_decomp_{style,reasoning}.csv` | 5 each | same decomposition on the other axes |
| `durability/b2c_cross_dataset.csv` | 48 | dpo persistence over **4 datasets incl. oasst1**, adds `llama-3.3-70b`/`qwen3-4b` |
| `durability/cap_vs_durability_n7.csv` | 7 | durability vs capability vs size; `launders` / `retains` tiers |
| `d1_dissociation_stats.csv`, `d1fix_*` | 25 / var | capability-vs-style dissociation + the corrected rerun |
| `d2_multiseed_ci.csv` | 180 | multi-seed CIs, 3 datasets |
| `d3_encoder_swap_bge.csv` | 175 | robustness: persistence recomputed with a different encoder |
| `big5_personality_directions.csv` | 900 | personality-direction projections per rung |
| `findings/*.csv` | 4–20 | style richness, verbosity bias, native stability |

### Raw generation stores (keep, but they are inputs not results)

`safety/adapter_refusal/` 216 files · `safety/native_refusal/` 8 · `safety/self_adapter_refusal/` 12 ·
`d1fix_capgen/` 32 · `d1fix_census_gen/` 8 · `durability/extra_census_gen/` 4. All 80–200 rows each.

### Known defect

`durability/per_source_durability.csv` has a multi-line CAVEAT comment **as its header row**, so pandas
reads the real header as data and every column name is prose. Read it with `skiprows=1` or fix the file.

---

## 4. `data/results/` — current tree

| path | what | status |
|---|---|---|
| `safety/erosion_seed42_{summary,long}.csv` | the 769-adapter erosion matrix | **current, load-bearing** |
| `safety/erosion_variance_stats.json` | source/target/dataset decomposition | current |
| `safety/multiseed_{pilot,local_check}/` | 3-seed robustness | current |
| `fidelity/fidelity_{seed42,all}_{embed,judge}_{long,summary}.csv` | 212 rows, 8 src × 13 tgt × 4 ds — **no `rung` column**, scores finished adapters only | current but **thin**: 33 judge-scored rows, `gsm8k` count **0** |
| `matrix/{sft,dpo}_runs/` | the adapter weights themselves | current |
| `matrix/{sft,dpo,self_sft}_data/` | training data (HF snapshots, not in git) | current |
| `chatbot_arena/`, `gsm8k/`, `call_center/`, `openai_finetune/`, `tinker_*/`, `workflows/`, `generic/`, `reasoning/`, `bridge_decontam/` | pre-safety-pivot style-transfer era | **superseded** |

Off-repo: `/data/ethantsliu/exp_steer_safety/repl80_rdo/` (base steering, 7 benchmarks × ~30 models)
and `/data/ethantsliu/exp_steer_adapter/` (adapter steering, 367 + 70B cells).

---

## 5. What this changes

1. **The prompting ladder is not missing — its safety measurement is.** Replicating the legacy ladder's
   own shape on the current pipeline is 3 local rungs × 12 pairs × 3 datasets = **108 cells ≈ 33 card-h**,
   with the legacy sft/dpo rungs available as a cross-check on the same models.
2. **`rung` explains 0.08% of safety variance (p=0.30)** in the legacy decomposition — but only across
   `sft` vs `dpo`. It says nothing about prompting-vs-weights, which is the contrast a ladder would test.
3. **`self_placebo_vs_cross_sft.csv` already contains a strong self-imitation control** on 4 models:
   `llama-3.1-8b` drifts −0.607 imitating others vs −0.014 imitating itself.
4. **Fidelity on the current roster is thin**, not absent — 212 rows but only 33 judge-scored and zero
   on `gsm8k`. If fidelity appears in the paper it needs a fill, not a rebuild.

Nothing was moved or deleted in producing this inventory.
