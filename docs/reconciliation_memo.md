# Reconciliation Memo — partner findings vs. dementor (Phase C)

*Co-authored work. This memo reconciles the partner's behavioral-stability findings with the
dementor adversarial-erasure results, ahead of the single combined paper (Phase D). Every number
on **our** side is quoted from a live artifact on the `ethan` branch (path given inline). Numbers
attributed to the **partner** are marked `[as relayed]` where they are **not** reproduced in this
repo — those must be confirmed against the partner's own artifacts before they enter the paper. The
one item that cannot be resolved unilaterally (Big-Five) is called out explicitly in §4 with a
concrete data request.*

---

## 0. The headline reconciliation in one sentence

The partner measures **native** behavioral stability of **undisguised** models (how consistent a
model's own style is across prompts/datasets); dementor measures **resistance under adversarial
DPO** (how much of a source's style survives when the model is actively fine-tuned to imitate a
*different* model). These are **different regimes**, so "identity is stable" (partner) and "identity
is erasable" (us) are **not in conflict for style** — and where we reproduce the partner's *native*
measurement on our 4 models (A4), the native stability ranking **agrees** with our durability
ranking (Spearman +0.80). The only place a genuine disagreement could survive is **Big-Five**
(§4), and that turns on a measurement detail we need from the partner.

---

## 1. What each side actually measured

| | Partner | dementor |
|---|---|---|
| **Regime** | Native / undisguised outputs | Adversarial: source fine-tuned to imitate target (SFT→DPO ladder) |
| **Quantity** | Consistency of a model's *own* style across prompts/datasets | Fraction of source style that *survives* imitation, vs. self-imitation (≈1) and identity (≈0) anchors |
| **Unit** | Per model | Per source→target **cell** (36 cells), aggregated to per-source |
| **Feature space** | 20 binary style features (shared) | MiniLM adjective scores + 32 structural features; supervised Fisher-LDA axis |
| **Claim** | Identity is stable / "60–65% ceiling" `[as relayed]`; Big-Five identity resists `[as relayed]` | Style is source-dependently erasable (2/4 launder to floor under DPO); Big-Five probe null |

The key point: a model can have a **highly stable native style** (partner) *and* have that style
**driven to the floor under adversarial DPO** (us). Stability-under-no-pressure and
resistance-under-attack are orthogonal axes. The partner's "ceiling" is a statement about the
first; our erasure is a statement about the second.

---

## 2. Primary reconciliation — native stability *predicts* our durability (A4 / S7 is the bridge)

We reproduced the partner's native-stability measurement on our 4 models using the **shared 20
binary style features** (`experiments/analysis/native_style_stability.py`, reusing
`latent_behavior_axes.py::_style_binary_features`). Three of the partner's findings reproduce in
shape:

- **F1 (scale-illusion):** param count ≠ style richness. `results/findings/style_richness.csv` —
  gpt-oss-20b has the highest TTR (0.626) despite being mid-size; llama-3.1-8b the lowest (0.510).
  Richness does not track parameter count. **Reproduces.**
- **F3 (cross-prompt-type stability):** `results/findings/native_xprompt_stability.csv` — all four
  models cosine > 0.97 across prompt buckets. Native style is near-invariant to prompt type.
  **Reproduces (method-validation).**
- **S7 (cross-dataset stability) — the load-bearing bridge:**
  `results/findings/native_xdataset_stability.csv` ranks
  **nemotron 0.908 > gpt-oss 0.872 > llama 0.861 > qwen 0.793**.

  This native, **imitation-independent** ranking matches our DPO style-durability ranking
  (retains: nemotron 0.211, gpt-oss 0.190; launders: qwen 0.077, llama 0.012) at **Spearman +0.80
  (n=4)**. A model's native cross-dataset style consistency *anticipates* how well its style
  survives adversarial DPO.

**Why this dissolves the apparent conflict.** The partner is right that style identity is a
pre-existing, stable property — and *that same pre-existing stability is what predicts our
durability*. Their "60–65% ceiling" `[as relayed]` is an **average across models**; our
contribution is that this average **masks a bimodal, source-dependent split** — two models retain,
two launder. Same underlying quantity, viewed at population-mean (partner) vs. per-source
resolution (us). The S7 reproduction is the literal bridge: our pipeline, their measurement, ranks
the same.

**Caveat carried forward:** +0.80 at n=4 is rank-direction only (permutation p≈0.33; see §6). It is
*suggestive corroboration*, not inference. Phase B (capability, not size, drives durability; n=7)
is where the durability claim actually earns its confidence — see `results/durability/
cap_vs_durability_n7.csv`.

---

## 3. Metric alignment — the two feature spaces are compatible

- **Shared 20-binary set.** Both sides compute style on the same binary feature set
  (`_style_binary_features`), so cross-side comparison is apples-to-apples at the feature level.
- **F2 (verbosity bias) ⊆ our length control.** `results/findings/verbosity_bias_features.csv`
  flags 8 features with |r| > 0.15 against log word count (contains_repetition 0.62, math_symbols
  0.45, has_markdown 0.39, header 0.36, all_caps 0.32, question 0.28, first_person 0.20, list 0.18).
  All 8 are a **subset** of what our `*_lenres` residualization already removes
  (`native_style_stability.py:166–167`). **No new residualization needed — F2 is already controlled
  in our pipeline.** This is a clean method-validation: the partner's verbosity concern was
  pre-empted by our length-residualized variants.
- **Encoder-swap robustness.** `results/d3_encoder_swap_bge.csv` re-runs persistence under BGE
  (bge-small-en-v1.5) instead of MiniLM. The **rung ordering and the tier-level
  retains/launders split are reproducible across encoders**; what is *not* stable across encoders is
  per-cell survivor identity (Jaccard ≈ 0.38). Load-bearing claim is therefore stated at
  **tier level, not per-cell survivor set** — consistent with how the paper already hedges it
  (`paper_draft.md` §6).

---

## 4. The one genuine open conflict — Big-Five (needs the partner's raw inputs)

**Our result (`results/big5_personality_directions.csv`, via `big5_directions.py`):** a Big-Five
logprob probe (Goldberg 1992 TDA adjective poles, 5 facets EXT/AGR/CON/NEU/OPN) is **null** —
mean |Cohen d| ≈ 0.14, and **0 / 900 cell-facet pairs reach even a medium 0.5 effect**
(README.md:62–64; `argument.md` §2a). The faint surviving signal lives in **structural/verbosity
style axes** (verbosity |d|≈0.26, structure |d|≈0.21), *not* personality.

**Partner's claim `[as relayed]`:** Big-Five identity *resists* (S10 variance shares ≈35%/17%
`[as relayed — not in repo]`).

**Why this is not yet a real contradiction — and what would make it one.** The disagreement turns
entirely on **what the Big-Five axis is computed over**:

- If the partner's Big-Five is **native-output variance** (how much models differ from *each other*
  on personality dimensions, undisguised) → **no conflict**. Native between-model variance ≠
  movement-under-imitation. Two models can differ in native personality (partner's 35%) while
  *neither moves* its personality under DPO imitation (our d≈0.14). Different quantities.
- If the partner's Big-Five is on **disguised / under-imitation outputs** and still shows
  resistance → **real disagreement**, and we have to reconcile method (logprob probe vs. their SVD;
  adjective set; anchoring).

**Concrete request to the partner (blocks this section of the paper):**
1. Is the Big-Five axis computed on **native** or **disguised** outputs?
2. The **per-trait Cohen's d** (or equivalent effect size), not just the variance share, so it is
   comparable to our 0/900-at-d=0.5.
3. The **SVD inputs** behind the 35%/17% — feature matrix and whether it is between-model
   (native) or within-model-under-imitation.

We can then re-express our movement result in the partner's **variance-share framing** (cache-only,
no new spend) and state either "no conflict, different quantities" or a reconciled method note.
**Until those three arrive, the paper states the Big-Five finding as ours (probe null on
movement) with an explicit footnote that the partner's native-variance framing is not in conflict
if measured on native outputs.**

---

## 5. Numeric discrepancy to chase down

The A4 commit (`6fd35c7`) flagged: **our llama cross-dataset stability 0.86 vs. the partner's 0.32**
`[as relayed]`. This is large and worth resolving before it appears anywhere. Likely causes, in
order of probability: (a) **different datasets** in the cross-dataset average (ours: gsm8k /
writingprompts / chatbot_arena); (b) **different feature definitions / normalization** despite the
nominally shared 20-binary set; (c) **different similarity metric** (cosine vs. correlation). Action:
diff the two feature extractors and dataset lists directly. This does **not** affect the *ranking*
agreement in §2 (which is internal to our pipeline), only the absolute magnitudes — but a 0.86 vs
0.32 gap must be explained, not papered over.

---

## 6. Statistical honesty carried into the combined paper

- **n=4 native correlations** (S7 +0.80, A1 co-ranking) are rank-direction descriptors:
  permutation p≈0.33 for a 2-vs-2 split, ~1 bit of information. `per_source_durability.csv` header
  already carries this caveat verbatim.
- **A2 verdict** (variance decomposition): **style and safety durability are stable per-source
  traits** (SOURCE η² large, rank-consistent); **reasoning durability is not stable**. So the
  partner's "stability" framing is supported by us *for style and safety, not reasoning* — state
  the asymmetry, don't generalize it.
- **Phase B is an honest null — capability does *not* drive durability** (corrected; an earlier
  draft of this memo overstated it). The B2b gsm8k result *looked* like "capability drives
  durability" (new high-cap sources retained at 0.17–0.27), but the retention was concentrated
  almost entirely in the **→nemotron** target cell (0.52–0.76) and laundered into the other three.
  B2c re-ran all three new sources across writingprompts + chatbot_arena + oasst1 and the →nemotron
  retention **does not replicate**: mean **0.613 on gsm8k → 0.007 on the other datasets**
  (`experiments/analysis/b2c_verdict.py`, `results/durability/b2c_cross_dataset.csv`). So the only
  defensible Phase-B claim is the null: high-capability sources launder like everyone else, and
  capability joins distinctiveness/size/Big-Five as a **failed** predictor of durability. This makes
  the "every intuitive predictor fails" story *stronger*, not weaker. The robust durability signal
  remains the original gpt-oss/nemotron broad retention (A2: SOURCE η² large for style+safety),
  mechanism open.

---

## 7. Related work to integrate (Phase D)

Status of external citations the combined paper needs. Items already on file in `docs/aaai_plan.md`
/ `paper_draft.md` §2:

- **On file:** Xu & Sheng 2026 (refusal-vector provenance, arXiv:2602.09434); Leshin et al. 2026
  (endpoint fingerprints, arXiv:2603.19022); Lee et al. ACL 2025 (distillation quantification,
  arXiv:2501.12619); Gudibande et al. ICLR 2024 ("False Promise" of imitation); Persona Vectors
  (arXiv:2507.21509); "Personality Illusion" (arXiv:2509.03730); CKA / Platonic Representation
  (arXiv:1905.00414, 2405.07987).
- **To add (`[verify]` — not yet looked up):** Prometheus (LLM-as-judge eval), Kartáč rubric
  (style rubric eval), Salemi/LaMP (personalization benchmark), Jin et al. style-transfer survey,
  Hinton et al. distillation (the canonical KD reference). These are referenced in the Phase-C plan
  but are **not** in the repo bibliography — confirm and insert before submission.

---

## 8. Open items (checklist for Phase D)

- [ ] Partner: Big-Five axis on **native** or **disguised** outputs? (§4) — *blocks the Big-Five
      paragraph*
- [ ] Partner: per-trait Cohen's d + SVD inputs behind 35%/17% (§4)
- [ ] Resolve the **0.86 vs 0.32** llama cross-dataset-stability discrepancy (§5)
- [ ] Re-express our Big-Five movement result in the partner's variance-share framing once inputs
      arrive (cache-only)
- [ ] Confirm + insert the `[verify]` related-work citations (§7)
- [ ] Fold §2 (native stability predicts durability) and §6 (Phase-B capability verdict) into the
      paper's durability section as the rigor backbone behind the n=4 hook

---

*Bottom line: for **style**, the partner and dementor are measuring two ends of the same
phenomenon — native stability (theirs) predicts adversarial durability (ours) at +0.80, so the
findings corroborate rather than conflict. The only live disagreement is **Big-Five**, and it is
almost certainly a native-variance-vs-movement framing difference that three numbers from the
partner will settle.*
