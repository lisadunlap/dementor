# Dementor — where this goes next

*Decision memo for Ethan (PI). One synthesis across five assessments (value/positioning, methodological consolidation, results-strength, story, mechanism-route). Where the assessments disagreed, I re-ran the numbers against the live CSVs this session and adjudicated with data, not opinion. Every load-bearing figure below was reproduced from `multiseed_ci_s3.csv`, `/tmp/highn_percell.csv`, and `distinctiveness_structural.py` on the current `ethan` branch. This memo is decisive on purpose: it tells you the one paper to write, the four numbers to cut, and the next three moves.*

---

## The one thing to internalize before anything else

You have **one** publishable empirical result and **one** dead explanatory claim, and your own docs (`argument.md` §5) are still recommending you lead with the dead one. Resolve that contradiction in favor of the data:

- **Alive (verified, defensible):** DPO-erasure of behavioral fingerprint is **model-dependent and source-driven** — two of four open models retain a robust residue through one epoch of LoRA-DPO, two launder to the floor, and which is which is a property of the *source* model, not the target, the domain, or model size.
- **Dead (verified to *invert*, not merely weaken):** "distinctiveness predicts persistence." In zero-MiniLM structural feature space r = **−0.31**, and llama — the cleanest launderer (0.012) — is the **most** structurally distinctive model (0.577). The r=0.97 lives only in the MiniLM space that the persistence metric is ~81% built from, i.e. exactly the circularity (O1) you were trying to kill. A2 was nominated in `argument.md` as "the one result a reviewer cannot call circular." It was run. It fails. That is not "needs work" — that is the load-bearing claim of the current framing failing its own test.

Three of the five assessments independently reached the same verdict (cut distinctiveness, lead with model-dependent erasure). The value and consolidation lenses add the harder truth that the *framing itself* — not just the mechanism — needs to change to survive review. I agree with the harder version.

---

## 1. IS IT VALUABLE

**Verdict: CONDITIONAL yes.** There is a real, defensible, modestly-novel result here, but **not at AAAI main track as currently framed**, and **not** with "distinctiveness" as the thesis. As-is, honest acceptance odds at a top venue are ~15-25% (concur with the strength lens). Reframed as a provenance/auditing negative result, it is a credible workshop or *EMNLP/ACL Findings*-tier contribution, and a plausible main-track paper only if you add a mechanism or more models.

**Why not "no":** the phenomenon is real and reproduces exactly. Two tight tiers (per-source DPO persistence nemotron 0.211, gpt-oss 0.190 | qwen 0.077, llama 0.012), separated by a gap ~3× the within-tier spread, seed-sd median 0.018, source-driven, size-dissociated (27B launders, 20B resists). That is not nothing.

**Why not unconditional "yes":** the area is crowded and moving fast. "Behavioral fingerprints survive fine-tuning/DPO" is already established (FPEdit >95% retention under LoRA *and* full FT; Intrinsic-Fingerprint "continue-training is not all you need"; refusal-vector fingerprints survive FT/quant/merge, Feb 2026). "Erasure is model-dependent" is also already observed in that literature. And the name/vocabulary collides with concurrent work (*Behavioral Fingerprints for LLM Endpoint Stability and Identity*, Mar 2026 — different task, but a reviewer will know the terminology is taken). You are not first to "fingerprints persist." You must be first to something narrower.

**The one sentence of genuine novelty (the strongest defensible version):**

> Under a *deliberate, escalating effort to erase a model's fingerprint by imitating a specific other model* (a disguise ladder up to DPO), whether the source's behavioral fingerprint survives is a stable property of the **source model** — not the imitation target, the domain, or model size — and, critically, it is **not predictable from how distinctive the model's outputs look**: the natural predictor inverts across feature spaces.

Note what that does *not* claim: it does not say *why* (mechanism open), and it does not say distinctiveness explains it (it doesn't).

**Right framing + venue (highest-leverage repositioning):** sell it to the **model-provenance / distillation-auditing** community as a **negative result about their tooling**, not to the watermarking/IP-fingerprinting crowd (who engineer durable *planted* keys and will yawn at "involuntary style survives"). The pitch that lands:

> Black-box behavioral fingerprinting is proposed for provenance/distillation auditing. We stress-test it under adversarial imitation and show its reliability is **source-model-dependent and unpredictable from output distinctiveness**: for half of four open models, one epoch of LoRA-DPO drives the signal to the floor. Provenance tools relying on involuntary style therefore carry a **model-dependent false-negative risk**.

That reframe (a) is genuinely novel (adversarial-imitation stress test of provenance tooling), (b) survives the n=4 problem because the headline is *existence + unpredictability*, not a law, (c) converts the failed-distinctiveness result from a liability into a finding ("the obvious predictor doesn't work"), and (d) has a clear customer. **Venue: target a safety/auditing or "science of LLMs" workshop, or EMNLP/ACL Findings.** AAAI main track with n=4, a confounded predictor, and a same-month terminology collision is a likely reject *unless* you land the mechanism (§5) or more models (§6).

**Before committing, one lit task:** someone must read the *full texts* of arXiv:2602.09434 and 2603.19022 — they are the nearest neighbors and the area produced three behavioral-fingerprint papers in Feb–Mar 2026 alone. The adversarial-imitation-ladder angle looks unclaimed, but verify it.

---

## 2. THE STORY

**Recommended narrative: model-dependent erasure as an observed phenomenon with an explicitly open mechanism, wrapped in the calibrated metric for credibility, framed for the auditing audience.** This is the one story where every load-bearing number reproduced this session and held.

**Title (phenomenon-forward):**
> *Some Models Won't Take Off the Mask: Source-Dependent Survival of Behavioral Fingerprints Under DPO Imitation*

**Title (contribution-forward alternative, better for a venue that rewards methods/negative results):**
> *When Behavioral Provenance Fails: DPO Erasure of Model Fingerprints Is Source-Dependent and Not Predicted by Distinctiveness*

**Thesis (2 sentences):**
> Using a calibrated, judge-free persistence metric over a 4×4×3 imitation matrix, we show that one epoch of LoRA-DPO toward a target model completely erases the behavioral fingerprint of some open models (llama, qwen → ~0) while leaving others statistically intact (gpt-oss, nemotron retain a residue robust across seeds and datasets; survivor enrichment Fisher p<0.01). The split is a property of the *source* model being disguised — not the algorithm, dataset, or model size — and it is **not** explained by personality or by any embedding-space notion of distinctiveness we tested (the natural predictor inverts across feature spaces), which we report as an open mechanism rather than a solved one.

**What gets cut from the spine (this is the decisive part):**

1. **The entire distinctiveness-explains-it edifice** — r=0.97, the n=36 "law," A2/A3/A7 as a causal argument. Keep at most one honesty paragraph: "a MiniLM-space correlation that does not survive a feature-space change; we flag it as unexplained." Leading with it invites the exact circularity reject you fear.
2. **The n=4 model-level correlation as anything inferential.** ρ=1.0 on 4 points is ~1 bit; permutation p≈0.33 for the 2-vs-2 split. State the n=4 limitation in one sentence and move on.
3. **The disguise-ladder taxonomy as the headline.** The monotone collapse (0.915 → 0.436/0.485 → 0.369 → 0.155, verified) is a clean Figure 1 and the experimental scaffold — but as a *thesis* it's close to expected ("more training erases more; weights beat prompts"). Demote to scaffold + one figure.
4. **"Structural, not personality" as a co-headline.** It's a real *negative control* (Big-Five mean |d|=0.14, 0/900 reach 0.5) and a *faint* positive (style VERB |d|=0.26, only 5% of cells > 0.5). Frame it as "the residue is **not** a Big-Five persona" (clean) — not "the residue **is** verbosity/structure" (the data only weakly support that, and it flirts with the length artifacts in O4–O6). One supporting figure-pair, not a contribution.

---

## 3. EVAL DIET

**You have ~11 eval mechanisms. Report 5. The fishing-expedition smell (which a reviewer *will* flag) comes almost entirely from (a) counting metric-parameters as "mechanisms" and (b) keeping a built-but-never-run activation pipeline and a *failed* distinctiveness explanation in the active inventory.** The fix is mostly subtraction.

**The minimal load-bearing set (report these 5):**

| # | Mechanism | Why it stays |
|---|-----------|--------------|
| 1 | **Persistence metric** (supervised LDA, anchored, separability-gated) — *absorb "anchored rescaling" and the separability gate into this; they are parameters, not mechanisms* | The whole instrument. Non-negotiable. |
| 2 | **D1 de-confound re-eval** | *Earned its place by changing a conclusion*: knocked the gpt-oss arena survivor 0.84→0.54; corr(CoT-leak, persistence)=0.79 is a fatal gotcha a reviewer would grep. Establishes the clean zero-leak survivor `nemotron→gpt-oss` (gsm8k 0.686±0.004). |
| 3 | **D2 multi-seed CIs** | Converts "survivors" from anecdote to result; seed-sd 0.018 is ~20× smaller than the survivor effect. Without it you have no defensible per-cell claim. |
| 4 | **Big-Five-null contrast** (one figure; style probe folded in as the faint positive) | Clean negative control that makes "structural not personality" honest. |
| 5 | **`sep_full` denominator-reversal + length partial-r** (a two-line "it's not an artifact" box) | *Earned its place by reversing the most dangerous objection*: launderers have the *largest* source→target gaps (llama 4.36), survivors the smallest (nemotron 3.45), Spearman −1.0. Pre-empts the killer "large-denominator" reviewer in one paragraph. |

**CUT from the methods narrative (move to future work or delete):**

- **`activation_steering` + `activation_bridge`** — *as reported mechanisms*. Both built, never run, zero output. Mentioning a built-but-unrun pipeline invites "so what did it show?" → nothing. (They are still the right *next move* — see §5 — but they belong in future work until they produce a number.)
- **The feature-ablation suite** as a headline panel. Keep `_lenres` alive as a single length-control robustness row; the full suite is over-instrumentation no reviewer asked for. Appendix or cut.
- **All three distinctiveness variants as a causal claim.** Keep MiniLM-distinctiveness at most as a one-line descriptive annotation. The structural and mpnet variants *disagree with it* — that is a disagreement you are currently hiding, and a reviewer will run exactly this check.

**DEMOTE to one robustness line:**

- **Encoder-swap (D3)** → "rung ordering replicates across encoders, Spearman ~0.80–0.89; llama is 0.000 under both encoders in all four of its cells." **But do not ship the magnitude claim until you regenerate the artifact** — see §4 and the next move. The on-disk `encoder_swap_all-mpnet-base-v2.csv` is still the buggy 12-cells-triplicated version (all 180 rows `dataset='decontam'`); the code fix in commit 6ad3d74 was never re-run. Shipping "36-cell encoder robustness" off that file is currently false.

That is a 5-mechanism paper with a clean spine: **calibrated metric → reproducible across seeds → not a CoT-leak artifact**, plus two pre-emption boxes (Big-Five-null, denominator-reversal).

---

## 4. STRENGTH

I verified every disputed number this session. The assessments split on two facts; the data resolved both **against** the optimistic reading.

### Solid vs shaky

| Status | Claim | Verified value | Note |
|---|---|---|---|
| **SOLID** | Monotone ladder collapse | 0.915 → 0.436/0.485 → 0.369 → 0.155 | Clean, de-confounded — but least surprising. |
| **SOLID** | Two-tier source split | nemotron 0.211, gpt-oss 0.190 \| qwen 0.077, llama 0.012 | The one result I'd defend. Seed-sd 0.018. |
| **SOLID** | Survivor enrichment | Fisher p=**0.0076**, 7/0 split (gpt-oss×4, nemotron×3, qwen/llama×0) | But it is a 2-vs-2 *source* comparison with cell replication (see below). |
| **SOLID** | Big-Five null | mean \|d\|=0.14, 0/900 reach 0.5 | Genuine negative control. |
| **SOLID** | Size dissociation | qwen-27B launders, gpt-oss-20B resists | Kills the obvious nuisance variable. |
| **SOLID** | Ladder encoder-robustness (rank) | Spearman(MiniLM, mpnet) ~0.89 on the real cells | *Rank* only; magnitude artifact unfixed on disk. |
| **BROKEN** | "Distinctiveness explains the tiers" | structural r=**−0.31**; llama most distinctive (0.577) yet lowest persistence | **Inverts.** Dead, not shaky. |
| **BROKEN** | "n=36 cell-level backbone" | `src_distinct` has exactly **4 unique values** (0.13/0.145/0.44/0.447) | The "n=36 r=0.451 p=0.006" is the n=4 r=0.966 replicated 9×. The genuinely per-cell predictor `tgt_distinct` gives **r=0.193, p=0.26 — n.s.** Effective n is 4. |
| **SHAKY** | "6 robust survivors (CI LB > 0.3)" | point-estimate>0.3 → **7**; mean−1.96·seed_sd>0.3 → **3** | The "6" reproduces from *neither* clean definition. Pin one CI. |
| **SHAKY** | Survivors are a deep identity signal | 3/6 marquee survivors collapse under length-residualization (arena nemotron→gpt-oss 0.541→0.003) | Residue is concentrated in *formatting/length*; defensible as "structural-formatting residue," not "deep identity." |
| **BUG** | D3 "36-cell robustness" | CSV still 12 unique pairs × 3, all `dataset='decontam'` | Code fixed, never re-run. False as written. |

**The two adjudications that matter most** (both confirm the strength lens, both contradict `argument.md` §5):

1. **Pseudo-replication is real.** I opened `/tmp/highn_percell.csv`: `src_distinct` is constant within source, 4 unique values total. The headline "n=36" inferential backbone is the n=4 model-level correlation mechanically tripled across datasets. A competent reviewer who opens this CSV sees it immediately. **Stop calling it n=36.** The honest backbone is the *Fisher survivor-enrichment* (p=0.0076) and the *per-source means with seed CIs* — but even those are, at root, a 2-vs-2 source split with cell-level replication. Effective n at the model level is 4. Say so.

2. **Survivor count is definition-fragile.** 7 by point estimate, 3 by seed-sd CI lower bound, "6" by neither. The *tier* is robust regardless; the *exact count* is not. Pin one CI definition (and note seed-sd alone excludes prompt-sampling variance — a reviewer will catch error bars that don't include it).

### The single strongest defensible claim

> One epoch of LoRA-DPO toward a target's outputs drives behavioral source-vs-target separability **to the floor for two of four open models (llama, qwen → ~0)** while leaving a **statistically robust, seed-stable structural residue for the other two (gpt-oss, nemotron)** — erasure is **model-dependent and source-driven, not universal**, and is **not** predicted by output distinctiveness, personality, or model size.

True, verified, reproducible (seed-sd 0.018), de-confounded, size-dissociated. Everything past it (the *why*) is an open mechanism — and the paper should say so.

### Honest acceptance odds

- **As currently framed (distinctiveness thesis, "n=36," buggy D3): ~15-25% at a top venue.** The reject reasons are concrete and *findable in the artifacts*: n=4 dressed as n=36, the predictor inverts under feature-space change, the robustness CSV is the buggy 12-cell version, survivors are length-fragile.
- **Reframed (auditing negative result + source-asymmetry, distinctiveness dropped, D3 re-run, one CI pinned): ~35-50%** at Findings / a strong workshop.
- **+ a real mechanism (steering or layerwise probe) OR n=8-10 models: plausibly main-track.** No reframing substitutes for more model points on the *interesting* claim.

---

## 5. MECHANISM PLAN

**Go/no-go on activation steering: conditional GO — but only as a scoped capstone, run *after* a cheap probe gates it, and NOT the version the docs imagine.** The mechanism lens corrected three stale blockers and I'm adopting its corrected picture:

- The "hard-blocked at line 427" claim is **false** — that's a `note` string, not an error. `export_adapter_to_peft` (`workflows/run_matrix.py:662`) already downloads the Tinker checkpoint and unpacks a real PEFT dir, and is already wired (lines 921, 1045). **The Tinker→PEFT bridge is solved.**
- Survivor DPO adapters are already in local HF cache (verified PEFT, LoRA r=32, all-linear).
- **The genuine bind:** the survivor *base models* (gpt-oss-20B MXFP4, nemotron-30B) do not fit in 16 GB with hidden-state capture. The only locally-runnable model is **llama-8B — which is the launderable one with no fingerprint to steer.** So the cheap-local version is a null that proves nothing; the informative version needs a rented GPU.

**The sequence that actually changes the paper (do exactly this, in order):**

1. **Step 1 — `activation_bridge.py`, fixed-encoder mode, all 36 cached cells (built, NEVER run; CPU/MPS, hours, zero new generation).** This is the highest-value-per-dollar move in the whole project. It produces a **layerwise separability curve**: do survivors (gpt-oss/nemotron) stay linearly decodable from target at deep layers post-DPO while launderable models (qwen/llama) collapse into the target cluster? If yes, *that is the mechanism story* — "DPO moves launderable models off the source manifold; survivors retain a decodable residual" — and it's publishable **without any 80 GB GPU**. If survivors are *not* decodable post-DPO, **stop — there is no direction to steer**, and you've saved the GPU spend. Honesty caveat: fixed-encoder mode reads a *small encoder's representation of the output text*, not the source model's own residual stream; it's a strong embedding-probe, not true internals. Be precise about that claim. Native-probe mode is the real internals read and re-incurs the GPU bind.

2. **Step 2 (parallel, cheap) — D4 structural decomposition** (cached `latent_scores.csv`, pure pandas, no GPU/API). Regress per-cell DPO persistence on per-feature persistence → *name* the features carrying the residue (length, markdown, lists, LaTeX). This is the most reviewer-proof move and converts the O8 recipe-confound into a feature-level claim. **But it is descriptive, not mechanistic** — it says *what* the fingerprint is, not *where in the computation* or that it's causally addable. It tells you what the steering vector should move (and warns you it's heavily length/markdown).

3. **Step 3 — ONE survivor-cell steering run, only if Step 1 is positive and you can rent 1×80 GB GPU for a day (~$30-80 spot + ~half a day of confound engineering).** Diff-of-means steering vector at the bridge-identified layer: (a) base + vector → persistence falls; (b) DPO-adapter − vector → persistence rises; (c) **random-direction control flat** (non-negotiable — without it a reviewer says "you pushed activations toward target text and the target classifier noticed"); (d) score on the **length-residualized** metric using the **hardened `clean_response`** (or you re-import the exact CoT leak you just removed, and you "prove" you can make a model more verbose). That single cell is the entire causal claim — it upgrades the paper from correlational to "the fingerprint is a linear residual-stream direction you can add to erase and subtract to restore."

**Three validity traps that will sink steering if ignored** (all verified): (1) the metric is ~81% MiniLM — build the steering vector from a *held-out* dataset and report the random-direction control, or it's the same circularity the rest of the project fights; (2) local generation does not replicate the gpt-oss `final`-channel extraction — steer through the hardened path or re-contaminate; (3) the marquee survivor cell is **length-fragile** (`full` 0.478 → `full_lenres` 0.003) — pick a `→qwen` writingprompts cell that survives lenres better, and report the effect on the length-residualized metric.

**Bottom line:** **do NOT gate the paper on steering.** The honest auditing/negative-result paper ships *now* without it. The bridge probe (Step 1) is nearly free, has never been run, directly attacks the open *why*, and de-risks everything — **if you do only one mechanism thing, do that.** Steering is the capstone that *could* lift this to main-track; it is not a prerequisite for a real paper.

---

## 6. THE NEXT 3 MOVES

Ranked by expected paper-value per unit effort. The first two are nearly free and *must* happen before anyone external sees the artifacts; the third is the upside swing.

**MOVE 1 — Re-run the de-confounded analyses under the *corrected* statistical story, and excise the dead claims. (Effort: S, ~1 day, zero new compute.)**
Concretely: (a) **regenerate the encoder-swap CSV** with the committed path fix so D3 is a real 36-cell grid, not 12 triplicated (it is currently *false on disk*; this is the single most embarrassing greppable artifact); (b) **pin one survivor-CI definition** (recommend point-estimate>0.3 = 7 survivors, *plus* the 3 that clear a seed-sd lower bound as "high-confidence," and add prompt-bootstrap variance so the bars are honest); (c) **delete the distinctiveness causal claim and the "n=36" language** from `argument.md` and the draft — replace with the model-dependent-erasure phenomenon + open mechanism. This move alone moves acceptance odds the most because it removes the four concrete reject-triggers a reviewer can find in the CSVs. Highest leverage, lowest cost.

**MOVE 2 — Run the activation-bridge layerwise probe (Step 1 above) + D4 structural decomposition. (Effort: S–M, ~1-2 days, CPU/MPS, zero new generation.)**
Two cheap, never-run analyses that *both* attack the open "why." The bridge tells you whether the surviving fingerprint is linearly decodable post-DPO and at which layer (a real internals figure, and the go/no-go gate for any GPU spend). D4 names the structural features that carry the residue (turns the O8 recipe-confound into a feature-level claim). Either one alone strengthens the paper; together they convert "4 models differ, we don't know why" into "the residue is decodable at layer L and lives in named structural features." This is the cheapest path from *negative result* toward *mechanism*.

**MOVE 3 — Buy down the n=4 problem: add models OR land one steering cell. (Effort: M–L; pick based on budget.)**
This is the only move that touches the paper's deepest weakness (the *interesting* claim has 4 data points and an unclosable recipe confound — the two survivors are exactly the two reasoning/CoT-distilled models). Two routes, choose one:
- **(3a) More models (n=8-10).** What the model-level claim actually needs; no reframing substitutes for it. Reuses the entire pipeline; cost is generation + adapter training for 4-6 new sources. This is the *honest* fix and the one a main-track reviewer implicitly demands. It also helps separate "distinctive" from "reasoning-recipe" from "trained-with-DPO" — currently inseparable at n=4.
- **(3b) One survivor-cell steering capstone (Step 3 above), gated on Move 2's probe.** ~$30-80 GPU + half a day of confound handling. Converts correlational → causal in one cell. Higher ceiling, lower certainty than 3a; a *negative* steering result (no single steerable direction) is arguably more interesting but harder to publish cleanly.

*Recommendation:* if the goal is a **defensible Findings/workshop paper soon**, do Moves 1-2 and submit the honest negative result — you have it today. If the goal is **AAAI main track**, Moves 1-2 are still prerequisites, and then **3a (more models)** is the surer bet than 3b, because it fixes the weakness reviewers actually cite (n=4) rather than adding a familiar technique (diff-of-means steering) whose novelty is only the application. Do **not** attempt 3b before Move 2's probe says there is a direction to steer.

---

### Files that matter

- `/Users/EthanLiu/Documents/Programming/dementor/data/results/multiseed_ci_s3.csv` — the real spine: per-source tiers, seed-sd 0.018, survivor cells.
- `/Users/EthanLiu/Documents/Programming/dementor/scripts/analysis/distinctiveness_structural.py` — reproduces the inversion (structural r=−0.31, llama most distinctive yet lowest persistence). The dead thesis.
- `/tmp/highn_percell.csv` — the pseudo-replication: `src_distinct` has 4 unique values; `tgt_distinct` vs dpo is r=0.193, p=0.26 (n.s.).
- `/Users/EthanLiu/Documents/Programming/dementor/data/results/encoder_swap_all-mpnet-base-v2.csv` — still the buggy 12-cells-triplicated version (all 180 rows `dataset='decontam'`); regenerate before anyone greps it.
- `/Users/EthanLiu/Documents/Programming/dementor/scripts/analysis/activation_bridge.py` — built, never run; the cheap mechanism win (Move 2).
- `/Users/EthanLiu/Documents/Programming/dementor/scripts/analysis/activation_steering.py` — steering capstone; the L425 "note" is not a blocker.
- `/Users/EthanLiu/Documents/Programming/dementor/workflows/run_matrix.py:662` — `export_adapter_to_peft`, the Tinker→PEFT bridge (already solved/wired).
- `/Users/EthanLiu/Documents/Programming/dementor/docs/argument.md` — §5 over-claims A2 and "n=36"; this memo supersedes its recommendation.
