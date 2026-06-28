# Plan: synthesize the dementor paper into one coherent two-author narrative

**Target file:** `/Users/EthanLiu/Documents/Programming/dementor-overleaf/neurips_2024.tex`
**Branch:** `reframe-security-thesis` (already checked out; `main` untouched for revert).

## Goal
The reframe removed the thesis-A/thesis-B contradictions and added cross-references,
but the paper still reads as two interleaved contributions with no unifying
conclusion and an abstract that under-credits Naz's prompt-based method. Three
changes turn "two non-contradictory threads" into "one coherent paper":
- **A.** Rebalanced abstract crediting both the disguise *method* (Naz) and the
  audit-failure *finding* (Ethan).
- **B.** A synthesis Discussion/Conclusion that unifies the benign and adversarial
  regimes.
- **C.** (Careful, do last) Reorder sections so the narrative builds benign →
  adversarial.

## Hard constraints (do NOT violate)
1. **Preserve every one of Naz's numbers and method descriptions verbatim.** No
   edits to her data in Killer Findings, Cross-Dataset Stability, Intervention
   Ladder, SFT/DPO Transfer, SVD, Methods, Related Work, Qualitative Examples.
2. **Stay on branch `reframe-security-thesis`.** Do not touch `main`.
3. **Must compile** (`pdflatex -interaction=nonstopmode` exit 0) after every change.
4. **Equal-contribution authorship**: the abstract and conclusion must represent
   BOTH the method (Naz) and the audit finding (Ethan) as co-equal.
5. Keep all existing `\label`s so cross-refs keep resolving; do not introduce new
   undefined refs.

---

## Change A — Rebalanced abstract
**Problem:** current abstract headlines the audit-failure thesis; mentions prompting
only as a ladder rung; omits Naz's clustering/contrastive "LLM certificate" method.

**Replace the current `\begin{abstract}...\end{abstract}` with (draft — implementer
may polish wording but must keep both halves):**

> Every public language model carries a distinctive behavioral fingerprint—tone,
> formatting habits, failure modes. We introduce **Dementor**, a framework for
> transferring this fingerprint between models and, with it, a judge-free
> separability metric for how much of a source model's fingerprint survives an
> attempt to disguise it as a target. As a *constructive* method, Dementor builds a
> compact "LLM certificate" by clustering a target's responses and selecting
> contrastive exemplars used as in-context demonstrations. With no weight changes,
> contrastive prompting transfers stylistic markers and outperforms random
> few-shot; escalating the full ladder—prompting, then SFT, then DPO toward the
> target—pushes a source up to a 60–65% match-rate ceiling. We then turn the same
> ladder into an *adversarial* stress-test of behavioral provenance across a 4×4
> model matrix on four datasets, asking what of the source survives. One epoch of
> LoRA-DPO drives the fingerprint to the floor for some source models while leaving
> a statistically robust residue for others: erasure is **source-dependent**—a
> property of the model being disguised, not the target, the task, or model size.
> Strikingly, *which* models launder is **not predicted by any intuitive signal**:
> output distinctiveness inverts as a predictor (r≈0.97 in-embedding → r=−0.31 in a
> held-out structural feature space), and neither size, task capability, nor a
> Big-Five persona probe predicts durability. Laundering also erodes safety
> (refusals degrade specifically under cross-model imitation), while a model's
> capability profile shifts toward its target even when style is fully erased—an
> independent provenance channel. Behavioral identity is thus, at once, a faithful
> and transferable signal that supports model-migration continuity, and—under
> adversarial pressure—an erasable one whose removal is model-dependent and
> unpredictable, carrying a silent false-negative risk for provenance auditing.

**Acceptance:** abstract gives the constructive method TWO sentences and names the
clustering/contrastive LLM-certificate method [Naz], correctly attributes the
**60–65% ceiling to the STACKED ladder** (not prompt-only; prompt-only contrastive
is +18–28% — do NOT say the no-weight-change method reaches 60–65%), AND covers
source-dependent erasure + predictors-fail + safety + capability channel [Ethan].
Closing sentence must be symmetric (neither regime a mere foil for the other). No
number contradicts the body; any restated Naz number copied verbatim from her
sections.

---

## Change A2 — Rebalance the Introduction contribution list (REQUIRED for fairness)
**Problem:** the Intro's explicit "**Contributions.**" enumeration (4 items) is
currently 100% Ethan's (adversarial stress-test, model-dependent erasure,
predictors-fail, safety+capability). Naz's prompt-method and native-fingerprint
findings appear nowhere in it. Abstract-only rebalancing is cosmetic; reviewers
attribute work from the Intro contribution list.

**Action:** insert a NEW contribution (1), renumbering the existing four to (2)–(5):
> (1) **Dementor, a prompt-based behavioral-transfer method**: a clustering +
> contrastive-exemplar "LLM certificate" that, with no weight changes, transfers a
> target's stylistic markers and outperforms random few-shot, plus native
> "fingerprint" analyses (scale-illusion, verbosity-bias, cross-prompt and
> cross-dataset stability) establishing that behavioral signatures are genuine,
> measurable per-model properties.

Keep the existing four contributions intact, just renumbered. Use Naz's verbatim
numbers if any are cited (+18–28% contrastive vs +8–12% random).

---

## Change B — Synthesis Discussion/Conclusion
**Add a new `\section{Discussion: Two Regimes of Behavioral Identity}`** immediately
**before** `\section{Ethical Considerations and Model Integrity}`. Draft content
(implementer may tighten; must hit all four beats):

1. **The benign regime [Naz].** Behavioral fingerprints are real, measurable,
   per-model properties: stable across prompt types (Killer Finding 3, cosine
   >0.91) and across datasets (native per-source consistency), and they can be
   deliberately transferred by prompting up to a 60–65% match-rate ceiling. In this
   regime fingerprinting *works*—it supports the continuity/model-migration use case
   that motivates Dementor.
2. **The adversarial regime [Ethan].** Under weight-level DPO imitation, the same
   fingerprints erase source-dependently: two of four sources launder to the floor,
   two retain; the split tracks the source, not the target/task/size; and no
   intuitive predictor (distinctiveness, size, capability, persona) says which is
   which. So as an *audit* tool, fingerprinting fails silently.
3. **The bridge.** The two regimes measure the same underlying quantity at
   different resolution and pressure. The connection is *directional and to be
   stated as such*: among the four base sources, the least natively cross-dataset-
   stable model (Llama, 0.319) is also the cleanest launderer, and the most stable
   (Nemotron, 0.706) retains. **Do NOT write that native stability "predicts"
   durability**—with n=4 this co-ranks with capability (ρ=1, permutation p≈0.33),
   which §capability_null explicitly rejects as a tested predictor; the Discussion
   must carry that same honest-bound caveat or it contradicts the body. Separately,
   the 60–65% "ceiling" is a limit on pushing a source *toward* a target (match
   rate); it is orthogonal to *erasing* the source's own fingerprint—DPO does the
   latter to the floor for the laundering sources. Frame the population-average
   ceiling and the bimodal split as two views of one quantity (symmetric), NOT as
   "Naz's average was hiding the real story."
4. **Unified takeaway.** Behavioral identity is simultaneously *transferable and
   measurable* (benign) and *erasable and unpredictable* (adversarial). For
   practitioners: fingerprinting is fine for continuity/UX, unsafe as sole proof of
   provenance; capability-testing is the more robust audit fallback.

**Acceptance:** the section explicitly names both contributions as co-equal, states
the bridge as a *directional, n=4-caveated* observation (NOT "predicts"), and
reconciles ceiling vs erasure symmetrically. ~1 paragraph per beat. New
`\label{sec:discussion}`.

**DUPLICATION WARNING (reviewer-flagged):** two "Implications for model auditing"
passages already exist — one in §source_dependent (the "Implications for model
auditing" subsection) and one as item 4 of §phase_f (Behavioral Erasure). The new
Discussion must be **synthetic only** (the benign↔adversarial bridge + the
ceiling-vs-erasure reconciliation); do **not** re-list the auditing implications
already stated in those two places, or the paper will say the same thing three
times. Do not edit those existing passages.

---

## Change C — Reorder (CAREFUL; do last, separate commit)
Move sections so the arc is benign → adversarial. Target order:

1. Introduction
2. Related Work
3. Methods
4. Evaluation and Results (match-rate)
5. Killer Findings (F1–F3)                    ← native: fingerprints are real
6. Extended Evaluation: Cross-Dataset Stability ← native stability
7. Prompt-Based Transfer: Intervention Ladder  ← promptable…
8. Fine-Tuning Transfer: SFT and DPO           ← …to a ceiling
9. Behavioral Axis Analysis (SVD)
10. **Source-Dependent Erasure** (move here, currently right after Eval) ← THE TURN
11. **Cross-Dataset Replication and the Capability Null**
12. **Behavioral Erasure: Model Laundering** (security)
13. **Discussion: Two Regimes** (new, Change B)
14. Ethical Considerations
15. Limitations and Future Work (move to end)
16. Appendix (Qualitative Examples, Reproducibility)

**Mechanics:** move whole `\section{...}...` blocks only; do not edit their bodies.
LaTeX resolves `\label`/`\ref` globally, so reordering will NOT break cross-refs
(`sec:source_dependent`, `sec:capability_null`, `sec:phase_f`, `sec:limitations`
all resolve regardless of order), and each figure lives inside its owning section
so none get orphaned. After moving, two-pass `pdflatex` must be clean.

**Pre-existing warnings you WILL see and must NOT "fix"** (they are in Naz's
sections / pre-date this work): `fig:finding5` is a mislocated label buried
mid-sentence in Killer Findings, and `fig:finding6`, `fig:oasst_disc_features`,
`fig:oasst_intermodel_corr`, `fig:oasst_style_heatmap` are undefined refs to Naz
figures not in this file. Do NOT touch the `fig:finding5` label or any of these —
editing them means editing Naz's prose. They are out of scope.

If any move risks breaking flow, STOP and leave it out (Change A+A2+B are the
high-value core; C is optional polish, separate commit).

---

## Verification (run after each change, report results)
- `pdflatex -interaction=nonstopmode -halt-on-error neurips_2024.tex` twice → exit 0,
  PDF builds.
- `git diff main --stat` — confirm only the intended sections changed; Naz's
  data-bearing lines show **no** numeric diffs (grep her numbers: 0.674, 0.590,
  7.92, 0.946, 0.936, 0.913, 0.706, 0.319, 35\%, 17\%, 60–65, +18–28, +8–12,
  +15–22, +12–18, +28–35, +32–42, 58\%). **Any Naz number restated in the new
  abstract/contribution/Discussion prose must be copied verbatim from her sections
  — never approximated** (e.g. write "60–65%", not "60%"; "+18–28%", not "~20%").
- No new "undefined reference" warnings beyond the pre-existing Naz figure refs
  (fig:finding6, fig:oasst_disc_features, fig:oasst_intermodel_corr,
  fig:oasst_style_heatmap).
- Commit A+B together; commit C separately so it can be reverted alone.
