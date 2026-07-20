# Results index

Master map of where Dementor's result artifacts live. Three locations, each tied to a distinct
experiment. Read this first, then follow the pointer to the location you need. The authoritative
framing for every number is [`RESULTS.md`](RESULTS.md).

| # | Location | Experiment | Status |
|---|---|---|---|
| (a) | [`data/results/safety/`](../data/results/safety/) | Imitation safety-erosion matrix (Exp 1) | **CURRENT** |
| (b) | [`results/`](../results/) | Fingerprint / persistence "disguise ladder" (Exp 0) | **LEGACY** |
| (c) | `/data/ethantsliu/exp_steer_safety/repl80_rdo/` | Steering dissociation (Exp 2) | **CURRENT — OFF-REPO** |

---

## (a) `data/results/safety/` — imitation safety-erosion matrix — CURRENT

The current genuine-harm (RTL) erosion results for the seed-42 disguise-adapter sweep (540
adapters). Small, **source-conditioned** erosion (variance 79% source / 3% target / 0.2% dataset).
Contains `erosion_seed42_summary.csv` (540 adapters), `erosion_seed42_long.csv` (3,780 rows), and
`figures/` (`erosion_heatmap.png`, `erosion_by_source.png`, `erosion_by_dataset.png`,
`erosion_distribution.png`). Built by `experiments/imitation_safety/build_erosion_csv.py` +
`plot_erosion.py`. See [`data/results/safety/README.md`](../data/results/safety/README.md) and
[`RESULTS.md`](RESULTS.md) Finding #3.

## (b) `results/` — fingerprint / persistence "disguise ladder" — LEGACY

The **older** persistence experiment: the disguise ladder (name-it → prompt-style → SFT → DPO)
scored by the judge-free persistence metric, plus the model-dependent DPO-erasure fingerprint
analyses. This is the provenance/fingerprint instrument, **not** the current genuine-harm safety
matrix. Includes `matrix_ladder/`, `d2_multiseed_ci.csv`, `fig1_source_fingerprint.png`, and a
`results/safety/` subtree of the earlier refusal-rate ladder (`safety_refusal_ladder.csv`,
`safety_native_refusal.csv`, etc.). See [`results/README.md`](../results/README.md). Treat as
**legacy** context for the persistence instrument; the load-bearing safety numbers now come from
(a) and (c).

## (c) `/data/ethantsliu/exp_steer_safety/repl80_rdo/` — steering dissociation — CURRENT, OFF-REPO

The steering-dissociation experiment ("identity is steerable, safety is not"), Exp 2. **Off-repo**
(too large for git): per-model eval outputs live under `<model>/…/metrics.json`, with the summary
in `REGEN_DISSOCIATION_TABLE.txt` (and `FINAL_RDO_TABLE.{txt,json}`). Full roster N=20 CLEAN;
CORE-16 tally = 9 CLEAN / 4 PC_FAIL / 1 PC_INVALID / 2 deferred. See [`RESULTS.md`](RESULTS.md)
Finding #4 for the per-model breakdown and the roster-asymmetry rationale (steering N=20 vs
imitation N=9 sources; connecting claim on the 8-model matched core).
