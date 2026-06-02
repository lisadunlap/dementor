# Dementor — Reviewable Plan (AAAI submission)

*For Ethan (PI). One synthesized plan across five analyst lenses (methodology, results, related work, next-directions, paper-draft). Numbers below were re-verified against the live CSVs this session, not taken from the brief. Where lenses disagreed, the data was used to adjudicate; the most consequential adjudication is that a **chat-template CoT leak partially explains the headline "survivor" result**, which re-orders the priorities.*

---

## 1. Where the project stands

We have a hardened, deterministic **behavioral-persistence** metric (supervised Fisher-LDA source→target axis on MiniLM-descriptor + hand-coded style features, anchored between a self-baseline ≈1 and an identity control ≈0, with a per-cell separability gate) and a full **4×4×3 = 36-cell disguise ladder** (naming → few-shot/style prompting → SFT → DPO), all 180/180 cell-rungs passing the trust gate. The central result reproduces exactly against the live `matrix_ladder.csv` files: mean persistence collapses monotonically **0.915 → 0.436/0.485 → 0.369 → 0.155** across rungs (naming never disguises; DPO is the strongest eraser), DPO drives 15/36 cells to exactly 0 yet **6/36 survive >0.3** (all gpt-oss/nemotron-sourced, arena-heavy, max 0.84), and a Big-Five probe shows the four models barely differ on personality (mean |Cohen d| ≈ 0.14–0.20 per facet, **0/180 cell-facet pairs reach even a medium 0.5 effect**) — i.e. the surviving fingerprint is structural/stylistic, not personological. **Two facts gate everything below**: (a) every one of the 36 cells is **single-seed** (`n_seeds=1`, `persistence_sd` empty in all 180 rows), so no point estimate has a CI; and (b) I verified a **chat-template confound** — `clean_response` (`workflows/run_matrix.py:79-81`) only strips gpt-oss's `analysis` chain-of-thought when a `<|channel|>final<|message|>` marker is present, but DPO makes gpt-oss stop emitting that marker, so raw CoT leaks into the stored "response"; across the 12 arena DPO cells **corr(persistence, analysis-leak-rate) = 0.79**, and the three gpt-oss-source arena survivors carry 0.6–0.7 leak rates — *partly* explaining them — while `nemotron→gpt-oss` (0.54) has **0% leak** and is a genuine survivor.

---

## 2. Next directions (prioritized, top 6, de-duplicated across lenses)

Ranking principle: an empirical-measurement paper lives or dies on whether the headline numbers are (i) **clean** (not artifacts), (ii) **stable** (have CIs), and (iii) **robust** (not encoder-specific). The first three directions secure exactly that, cheaply, mostly by re-scoring text already on disk. Flashier "more interventions / more models / activation steering" come after, because they multiply an unvalidated number rather than validate it.

> **Reuse substrate (why most of this is cheap, verified this session):** all 36 cells cache raw per-rung text (`…/cells/<src>_to_<tgt>/staged/*.csv`, `…/gen/*.csv`) and per-row `st_axis` projections (`methods/0X_*/latent_scores.csv`); **3 SFT + 3 DPO adapter seeds exist for every cell** (`tinker_adapters.json`: seed1×288, seed2×252, seed3×252 leaf keys); `run_cell_pipeline.py`/`run_matrix_cells.py` already expose `--adapter-seeds 3` and `--bootstrap`; `BehavioralCellSpec.encoder_model` is plumbed through the evaluator (`behavioral_cell_evaluator.py:68,350,397,477,532`) but **no `--encoder` CLI flag is wired**; a `humaneval` prompt set already exists; the LLM-judge calibration path (`_calibration_summary`) is built but never run. So D1/D2/D3 need **zero new generation or training**.

### D1 — Decontaminate the chat-template leak, then re-attribute "survival" (RANK 1) — Effort: **S**
- **Question.** How much of the DPO-survivor signal — especially the gpt-oss-source arena cluster (0.84/0.78/0.30) — is the `analysis`-channel CoT leak vs a genuine surviving fingerprint? Does `nemotron→gpt-oss` (0.54, 0% leak) remain the one robust survivor after a correct re-clean?
- **Why it matters.** This is the single biggest threat to the headline and I have direct evidence it bites (corr 0.79; leaked `<|channel|>analysis<|message|>` strings are sitting in `chatbot_arena/.../gpt-oss-20b_to_*/staged/dpo.csv`, ~130/12k rows each but concentrated in exactly the survivor cells). A reviewer who greps the artifacts finds it. Fixing it yields a *stronger, more honest* claim either way: corrected gpt-oss effect + a genuine nemotron survivor, or a collapse of the gpt-oss survivors leaving nemotron — both publishable.
- **Experiment.** Reuse: re-run the evaluator on the **cached** `staged/*.csv` (no regen, all 228 adapters untouched). New: ~30-line fix to `clean_response` (when no `final` marker, drop the `analysis` block up to the answer; strip residual `<|channel|>/<|message|>/<|start|>` for all models) + a regression test. Emit a 36-cell before/after persistence table with per-cell leak-rate as a covariate; recompute persistence on the **zero-leak subset** of survivor rows to isolate the genuine component.
- **Payoff / risk.** Payoff: de-confounds the paper's twist. Risk: low technical; the "risk" is the gpt-oss survivor effect shrinks — which is the point.

### D2 — Multi-seed CIs across all 36 cells; re-classify survivors & zeros under uncertainty (RANK 2) — Effort: **M**
- **Question.** Are the (de-confounded) survivors and the 15 exact-zeros statistically real across the 3 already-trained adapter seeds, or one-seed draws? Is `nemotron→gpt-oss` gsm8k = 0.68 a model property or noise?
- **Why it matters.** Every per-cell and per-dataset claim currently rests on a single seed (verified: 180/180 rows `n_seeds=1`). The only multi-seed evidence is a **1-cell** pilot on disk (`gsm8k/llama→gpt-oss` has `rung_{sft,dpo}_seed{2,3}.csv` → the "DPO 0.002±0.003, SFT 0.16±0.046" check) — and that cell has the *easy* gpt-oss target, not a survivor. CIs convert "6 survivors" into "k survivors significant at the cell level" and put error bars on Figure 1.
- **Experiment.** Reuse: the seed2/seed3 SFT+DPO adapters already exist for all cells; baselines/basis/anchors are seed-independent and cached. New (sampling only, **no training**): `run_matrix_cells … --adapter-seeds 3` to generate eval text from the existing seed2/3 adapters, then aggregate per-cell mean ± 95% CI and report seed-variance vs prompt-bootstrap-variance ("which dominates?"). Run **after D1** so CIs are placed on de-confounded text. If compute-bound, scope to the ~12 cells that matter (survivors + a matched set of zeros + a few mid-range).
- **Payoff / risk.** Payoff: the credibility ask a reviewer will demand. Risk: low (idempotent, infra proven by the pilot); scientific risk that a survivor flips — better found now.

### D3 — Encoder-swap external validity, with an encoder-vs-style variance decomposition (RANK 3) — Effort: **S–M**
- **Question.** Does the rung ordering and DPO-collapse replicate under a structurally different (and stronger) text encoder, and **how much of the headline even moves** given that the metric mixes encoder-derived descriptor scores with encoder-*independent* hand-coded style features?
- **Why it matters.** The brief names encoder-swap as *the* intended deterministic external-validity test, and it has **never been run** (verified: all scoring is MiniLM). Without it, "persistence" = "persistence as MiniLM sees it," and the most likely methodological objection is "embedding artifact." The subtle higher-value version: report (a) replication under a stronger encoder *and* (b) a decomposition of how much movement the encoder block vs the hand-coded style block carries — if conclusions hold *and* style features carry most of it, you have "not an embedding artifact, it's interpretable style."
- **Experiment.** Reuse: all cached text (zero regen); the existing `feature_ablations/` machinery already isolates `full` vs `style_*`. New: wire a `--encoder` passthrough to the existing `encoder_model` spec field (the only code change) and re-score all 36×5 with 2 contrasting encoders (e.g. `all-mpnet-base-v2` and a `bge`/`gte` asymmetric model). Report 36-cell Spearman across encoders + survivor-set Jaccard; keep the LDA basis re-fit *within* each encoder (code already fits per run). CPU-only, no API.
- **Payoff / risk.** Payoff: cheapest insurance for the whole paper. Risk: low; worst case is a publishable "encoder-sensitivity" finding.

### D4 — Name the surviving structure: which features carry the fingerprint, and is it ~1–2D? (RANK 4) — Effort: **M**
- **Question.** "Structural not personality" is currently a *negative* result. Of {length, sentence length, markdown/tables, headers, bold, bullet/numbered lists, LaTeX `\[…\]`/`\boxed`, emoji, hedging/reasoning markers}, which few features (i) separate each source→target on held-out prompts and (ii) **persist through DPO** in the survivor cells? Is the source→target axis essentially 1–2D (verbosity/structure), and is DPO-erasure concentrated along those same axes?
- **Why it matters.** Turns the title's promise ("structural") into a falsifiable, positive mechanism: e.g. "≥X% of separation and of DPO erasure lives in length + markdown-structure + LaTeX; personality axes are inert." It also directly rules out the deflationary "it's just length" alternative (gpt-oss survivors are 1500+ chars) via the existing `*_lenres` length-residualized feature sets.
- **Experiment.** Reuse heavily: `latent_behavior_axes.py` `_style_scalar_features` (12) + `_style_binary_features` (20) already exist as a `style` feature set but are **not reported per-cell as a decomposition**; `axis_loadings.csv` and `feature_ablations/` already emitted per cell. New: a script computing per-feature source↔target separation and per-feature univariate persistence (same `1−movement` formula) → a 36×~30 matrix; regress MiniLM persistence on the structural-feature persistence vector (target R²>0.7 = mechanism found); length-residualize to separate "longer" from "structured-at-fixed-length."
- **Payoff / risk.** Payoff: the highest-upside *scientific* finding; pairs with the in-progress style-axes run. Risk: medium — the axis may not be cleanly low-D, in which case you can only bound it ("≥X% interpretable, remainder sub-lexical"), still useful and consistent with the Big-Five blunt-probe null.

### D5 — Finish the style-axes logprob run (verbosity/formality/certainty/structure/warmth) (RANK 5) — Effort: **M**
- **Question.** Do interpretable *style* directions separate the models where Big-Five personality did not, and is that where the surviving fingerprint lives?
- **Why it matters.** Completes the central narrative arc (personality null → style positive) and gives a clean contrast figure to the Big-Five negative control. It is the natural positive complement the paper needs so "structural" isn't only inferred from the absence of personality signal.
- **Experiment.** Reuse: the `big5_directions … --axes style --backend logprob` driver and STYLE axis set exist. New: only **27 of 36 cells remain** — verified the cache has **9/36 (arena only)** in `big5_cache_style_logprob/` and **no aggregated CSV**. Generate the remaining gsm8k+wp cells (gpt-4.1-mini logprob, ~40 prompts × 5 axes × 27 cells) and aggregate into a matrix `|d|` table mirroring the Big-Five one.
- **Payoff / risk.** Payoff: closes the story. Risk: medium — bounded API cost; if style separation is also weak, that's an informative (if less exciting) result that pushes D4's "sub-lexical" interpretation.

### D6 — Cross-distribution generalization: do dataset-trained disguises transfer off-distribution? (RANK 6) — Effort: **M (HumanEval) / L (MMLU)**
- **Question.** An adapter SFT/DPO-trained to imitate the target *on gsm8k* — does it still suppress the source fingerprint on **HumanEval (code)**, a distribution it never trained on? Is DPO-erasure a property of the fingerprint or of the training distribution?
- **Why it matters.** Every persistence number so far is **in-distribution**, confounding "fingerprint erased" with "learned to match target on these prompts." If erasure snaps back off-distribution, the honest reframing is "disguise is shallow and distribution-bound" — the most novel result reachable from existing adapters, and the obvious "did you test OOD?" reviewer question.
- **Experiment.** Reuse: distribution-agnostic evaluator (needs only `prompt`+`model_response`), all 228 adapters as-is (**no retraining**), and the **already-staged** `humaneval_prompts.csv`. New: generate base+adapter+prompting-rung responses on HumanEval (re-anchor the LDA on the new distribution, which the pipeline builds). Start with the **gsm8k-trained adapters on HumanEval** (math→code, the sharpest transfer) — one 12-pair slice lands the claim. Let the `separable` gate, not hope, decide whether to add MMLU (short constrained answers compress style variance).
- **Payoff / risk.** Payoff: reframes the "behavioral inertia" thesis as shallow-vs-deep. Risk: medium — code-gen may trip gpt-oss channel/`clean_response` edge cases (mitigated once D1 lands); MMLU may gate out (report it).

**Explicitly deferred / NOT now:** (a) **activation steering / layerwise probes** (`activation_steering.py` is built but hard-blocked — line 427 errors demanding a non-existent `tinker://`→PEFT export, and needs local GPU for 27–30B weights; highest ceiling, lowest certainty, L effort) — only after D1–D2 confirm a real survivor to explain; (b) **more models / closed-model open-set attribution** — multiplies the same confound; future work; (c) **further Big-Five adjective probing** — already null, blunt by the brief's own admission. One **free** analysis worth pulling forward into the paper: the **imitation-asymmetry table** (persistence(A→B) − persistence(B→A)) is computable today from the existing 36 cells, zero compute (e.g. gpt-oss→nemotron 0.84 vs nemotron→gpt-oss 0.54 on arena).

---

## 3. AAAI paper plan

### Title (recommended + alternatives)
1. **The Disguise Ladder: Measuring How Much of a Language Model's Behavioral Fingerprint Survives Imitation** *(recommended — contribution-forward, neutral)*
2. DPO Erases the Fingerprint, Mostly: A Calibrated Measure of Behavioral Inertia Under Escalating Imitation *(sharper, but must immediately qualify with survivors)*
3. What Survives When a Model Pretends to Be Another? An Intervention-Ladder Account of LLM Behavioral Fingerprints

### Contributions (3, each tied to verified artifacts)
- **C1 (method).** A deterministic, judge-free, **calibrated persistence metric**: held-out responses projected onto a supervised Fisher-LDA source→target axis, anchored between a self-baseline (≈1) and identity control (≈0), with a separability gate that refuses untrustworthy cells (180/180 passed). Basis- and k-independent by construction. *The reusable contribution.*
- **C2 (measurement).** A **disguise ladder** across the full 4×4×3 matrix showing **monotone fingerprint collapse with intervention strength** — naming 0.915, prompting 0.44/0.49, SFT 0.369, DPO 0.155 — i.e. naming alone never disguises; only weight-level interventions move the fingerprint, DPO most.
- **C3 (the twist).** **DPO erasure is domain- and model-dependent, not universal**: ~0.11 on structured tasks vs 0.24 on open chat, 15/36 fully erased yet 6/36 survivors (the most structurally distinctive sources), and the surviving signal is **structural/stylistic, not Big-Five personality** (mean |d| ≈ 0.14–0.20, 0/180 medium effects). *Honest caveat folded in: part of the gpt-oss survivor signal is a chat-template artifact we identify and correct (D1); the corrected survivors + the zero-leak `nemotron→gpt-oss` cell remain.*

### Abstract (~150 words)
> Open-weight language models carry a *behavioral fingerprint* — stylistic and structural regularities that distinguish one model's outputs from another's. We ask how much of that fingerprint survives when a source model is pushed to imitate a target, and introduce a **disguise ladder**: interventions of increasing strength (naming, few-shot style prompting, SFT, DPO). We score each rung with a **calibrated, judge-free persistence metric** — held-out responses projected onto a supervised source-versus-target axis, anchored between a self-baseline and an identity control, with a separability gate. Across four open models imitating one another on three task domains (36 cells, 180/180 trustworthy), persistence falls monotonically with intervention strength: naming never disguises (0.92), while DPO is the strongest eraser (0.15). Crucially, erasure is *not* universal — structurally distinctive models retain their fingerprint through DPO on open-ended chat. A Big-Five probe shows the surviving signal is structural, not personality.

### Section-by-section outline
1. **Introduction.** Hook: open models routinely imitate each other (distillation, "act-as", provenance laundering); "behavioral fingerprint" is asserted, rarely measured. Contributions C1–C3.
2. **Related work** (positioning below).
3. **The disguise ladder.** The four rungs as an ordered intervention-strength axis; 4×4×3 matrix; LoRA SFT + DPO toward target outputs on held-out eval prompts.
4. **Persistence metric.** Features (MiniLM descriptor block + style scalars/binaries; `*_lenres` length control); supervised LDA basis fit on source+target only (disguised never defines its own axes); self/identity anchors and `anchored`; separability/trustworthiness gate; enforced shared endpoints; anchored bootstrap. Emphasize determinism and basis/k-independence.
5. **Results.** Monotone collapse (C2); DPO erasure + non-universality + survivors (C3); Big-Five negative control (structural-not-personality); style-axes positive control (D5). Figure 1 + the Big-Five bar.
6. **Robustness & threats.** Multi-seed CIs (D2), encoder-swap (D3), the chat-template de-confound (D1), length residualization, refusal confound.
7. **Limitations & scope** (verbatim from §below).
8. **Conclusion / future work** (cross-distribution D6, activation-level mechanism).

### Figure / experiment table (done vs todo)

| ID | Figure / Table | Supports | Status | Effort |
|----|----------------|----------|--------|--------|
| **F1** | Ladder-collapse, 3 panels (persistence vs rung) — redesign as bold mean+band per dataset with **survivor trajectories overlaid/labeled** (replace 12-line spaghetti) | C2 + C3 in one image | data DONE (`matrix_ladder.png`); **redesign TODO** | S |
| **F2** | Single cross-dataset headline (rung means overlaid, or 36-cell DPO heatmap) | C2 | TODO (no combined fig exists) | S |
| **F3** | DPO-survivor highlight (6 cells >0.3) | C3 | TODO (numbers verified) | S |
| **F4** | Big-Five |d| bars per facet w/ 0.5/0.8 lines far above every bar | structural-not-personality | data DONE (`big5_directions_matrix_logprob.csv`); fig TODO | S |
| **F5** | Style-axes |d| matrix (contrast to F4) | structural positive | **27/36 cells + aggregation TODO** (D5) | M |
| **T1** | 36-cell persistence matrix (src×tgt × dataset) at DPO + per-rung means | C2/C3 | TODO assembly (data present) | S |
| **R1** | **De-confound before/after** persistence table + leak-rate covariate | C3 honesty | **TODO (D1)** | S |
| **R2** | **Multi-seed CIs / error bars on F1** | credibility | **TODO (D2)** — adapters exist, sampling only | M |
| **R3** | **Encoder-swap** scatter + Spearman vs MiniLM | external validity | **TODO (D3)** — needs `--encoder` flag | S–M |
| **R4** | Structural-feature decomposition (which features carry/persist) | "structural how?" | TODO (D4) | M |
| App. | Per-cell trust roll-up (sep_ratio, probe_cv, anchored CI, z_vs_baseline) | rigor | data DONE in every `cell_summary.csv`; roll-up TODO | S |

### Related-work positioning (one paragraph + the four contrasts)
> Existing work treats a model's signature as something to **attribute** (LLM stylometry, e.g. arXiv:2507.00838, 2506.17323), **inject and protect** (instructional/backdoor fingerprinting, Xu et al. NAACL 2024 arXiv:2401.12255; intrinsic output-space fingerprints arXiv:2407.01235), or **trace back to a teacher** (imitation/distillation detection — Gudibande et al. ICLR 2024 "False Promise"; Lee et al. ACL 2025 distillation quantification arXiv:2501.12619), and studies persona change mostly as a within-model, named-trait phenomenon (Persona Vectors arXiv:2507.21509; sycophancy; "The Personality Illusion" arXiv:2509.03730). We instead ask a **robustness question that cuts across these**: under escalating adversarial pressure to make a source imitate a target, how much of the source's *involuntary* fingerprint survives, and does it depend on domain and model pair? Distinct from CKA/Platonic-Representation convergence (arXiv:1905.00414, 2405.07987), which measure white-box representational similarity on neutral inputs, we measure **black-box behavioral discriminability under adversarial imitation** on held-out task prompts, with self/identity-anchored normalization.
- **vs fingerprinting/watermarking:** they engineer a durable *planted key*; we measure the *involuntary* fingerprint's durability and document when it *fails* to survive (DPO erasure).
- **vs distillation/imitation detection:** they detect the *teacher* in the student; we measure the *self* remaining in the imitator (retention, not trace-back), plus a domain/model-conditional erasure law.
- **vs RLHF/persona work:** they track *named* personality directions within one model; we use a *supervised discriminative* axis toward a *specific other* model and show personality axes are the wrong probe — the residual is structural.
- **vs CKA/Platonic:** they argue convergence in white-box representations; our DPO-erasure is one *mechanism* of behavioral convergence, but the arena survivors show it's incomplete and domain-gated.

### Top 4 likely reviewer objections + rebuttals
1. **"Persistence is a MiniLM artifact / the feature space never validated against true behavioral identity."** → The trustworthiness gate proves the space *is* model-discriminative per cell (180/180); `*_lenres` shows it isn't pure length; feature-block ablations isolate each part. **And we run the encoder swap (D3)** showing the rung ordering and survivor set replicate under a stronger, different-family encoder, with a decomposition showing interpretable style features (not just embeddings) carry the signal. *(D3 must be done — this is the most reject-prone lever.)*
2. **"Per-cell claims (the 6 survivors, arena=0.84) are single-seed with no CIs."** → We add **3-seed CIs across all cells (D2)**; the monotone collapse and SFT→DPO drop are already robust (DPO<SFT in 32/36; pilot DPO 0.002±0.003, SFT 0.16±0.046), and survivors are re-reported with CI lower bounds. *(D2 must be done.)*
3. **"'DPO erases the fingerprint' is circular — DPO optimizes toward target outputs and you measure distance to target."** → Held-out *eval* prompts mitigate prompt overfitting; the metric space (adjective-style embeddings) is **not** the DPO loss; and the *interesting* claim is the **non-universality** (arena survivors, structural residue), which no target-pushing objective trivially predicts.
4. **"Did you check the raw outputs? gpt-oss survivors look like leaked reasoning."** → **Yes — we identify and fix it (D1).** We disclose the `clean_response` chain-of-thought leak, report corrected before/after persistence with leak-rate as a covariate, and show the genuine survivor (`nemotron→gpt-oss`, 0% leak) is unaffected. *(Pre-empting this in the paper converts a fatal "gotcha" into a credibility signal — D1 is non-negotiable.)*

---

## 4. Critical path to submission-ready (sequenced)

> **AAAI deadline is UNCONFIRMED** — no source in the repo or brief states one. Do not plan against an invented date; confirm the venue/deadline first (decision below). Sequence assumes weeks, not days.

1. **D1 — de-confound (S).** Fix `clean_response` + regression test; re-score all 36 cells on cached text; produce R1 before/after table. *Blocks D2/D3/F1/F3/T1 — they must run on clean text.* **Do first, this week.**
2. **In parallel, S-effort assembly on de-confounded outputs:** F2, F3, F4, T1, the asymmetry table, and the trust roll-up appendix — all data-present, pure aggregation.
3. **D3 — encoder swap (S–M).** Wire `--encoder`; re-score with 2 encoders; R3 figure. *No generation — can start the moment D1 lands; highest insurance/cost.*
4. **D2 — multi-seed CIs (M).** Sample seed2/3 adapters across cells (or the ~12-cell priority subset), aggregate CIs, add error bars to F1. *Needs Tinker generation; start its runs early since they take wall-clock.*
5. **D5 — style-axes (M).** Generate the 27 missing cells + aggregate → F5, the positive control.
6. **D4 — structural decomposition (M).** Name the surviving features; R4. Turns C3 from negative to positive mechanism.
7. **First complete draft** once F1–F5 + R1–R3 exist (i.e. after steps 1–5). **D6 (cross-distribution)** and activation-level work are **post-first-draft / future-work** unless reviewers/time allow.

Minimal bar to submit a defensible paper: **D1 + D2 + D3** (clean, CI'd, encoder-robust headline) + the S-effort figure/table assembly. D4/D5 strengthen the story; D6 is future work.

---

## 5. Decide now (3 decisions for Ethan)

1. **Confirm the target venue and deadline.** No deadline exists in the repo/brief; the critical path is weeks long. Pick AAAI (and which deadline/track) vs an alternative, so D2/D5 generation can be scheduled against real wall-clock. *(Blocks all sequencing.)*
2. **Authorize the D1 de-confound as a hard gate.** Agree that the `clean_response` CoT-leak fix lands **before** any CI/encoder/figure work, and that the paper will *disclose* it (corrected survivors + the leak-rate covariate) rather than quietly re-running. This reframes a likely fatal reviewer "gotcha" into a strength — but it may shrink the gpt-oss survivor effect. Confirm we accept that.
3. **Set the multi-seed scope: all 36 cells vs the ~12-cell priority subset.** Full coverage gives every cell a CI (cleanest for reviewers) but ~28k extra generations; the subset (survivors + matched zeros + mid-range) is far cheaper and still answers "are the survivors real." Choose based on the deadline from (1).
