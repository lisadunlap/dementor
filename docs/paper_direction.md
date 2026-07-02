# Paper direction — *Benign model imitation is a safety hazard*

The spine of the paper. Supersedes the "fingerprints persist" framing as the lead contribution
(that work becomes the instrument + a supporting finding). Prior strategic analysis: `strategy.md`.

## The claim
Fine-tuning one open LLM to **imitate another on entirely benign data** (SFT/DPO on innocuous
imitation targets — *no harmful data*) systematically **degrades the disguised model's safety**, and
this **generalizes across real open models**. The degradation is **two-directional — a calibration
collapse**: the model both complies more with harmful requests *and* over-refuses more on
borderline-benign ones, via a subtle **"refuse-then-leak"** failure (refuse, then hedge/pivot to
adjacent harmful content) that evades naive refusal detectors.

## Why this is the paper (not "fingerprints persist")
- **Novel** — distinct from the crowded "behavioral fingerprints survive fine-tuning" literature
  (three such papers Feb–Mar 2026). Nobody has shown *benign imitation → safety degradation, in both
  directions, via refuse-then-leak*.
- **General** — demonstrated on **4 real roster models**, not one stand-in.
- **Consequential** — a safety hazard of ordinary distillation/personalization, with a clear audience
  (safety/alignment; model-provenance/distillation auditing).
- The fingerprint-disguise **ladder + persistence metric are the *instrument***; source-dependent
  fingerprint erasure is a supporting result, not the headline.

## Evidence (current; roster numbers pending independent re-verification)
- **Roster generalization — REFUTED as source-general; the effect is TARGET-driven and small**
  (target-varied 3×3, strict metric, resolved). Base→DPO strict genuine-compliance erosion sorts by
  *target* (disguising →nemotron +3.6pts [2/2 sig], →llama +2.4 [3/3] — leak; →gpt-oss +0.3, →qwen
  −0.25 — flat/null) **not by source** (no source erodes across all its targets). The "generalizes
  across 4 sources" result was an artifact of all 3 big cells sharing target=llama. Magnitude is
  small (strict Δ ≤ +5.6pts); the raw +20–32% →llama swings are ~85–95% degenerate refusal-loops +
  hedged refuse-then-engage that Llama-Guard over-flags — **not** harmful compliance. **Mechanistic
  read: you partly inherit the safety profile of the model you *imitate*, not a generic source-level
  degradation.** Honest claim = *target-conditioned safety transfer + a measurement caution*, not
  roster-wide source-general erosion. Load-bearing solid evidence remains the census-adjudicated Qwen
  cell.
- **Two-directional calibration** (Qwen cell + SORRY/OR-Bench): harmful-compliance ↑
  (SORRY-Bench 27.8→37.6%, p=6.3e-6) *and* over-refusal ↑ (OR-Bench 14.0→20.8%, p=1.7e-9).
- **Mechanism — refuse-then-leak.** A keyword detector missed it entirely; even Llama-Guard
  over-counts it (78–95% of DPO-"unsafe" is refusal-led), so the **strict genuine-compliance metric is
  the honest headline** (raw Guard = sensitive upper bound).
- **Powered baseline.** Qwen stand-in on AdvBench+StrongREJECT n=808: base→DPO p=1.4e-7, 3-seed
  replicated; census-adjudicated genuine methodology-leak 1.0→3.0%.

## The 3 gaps to close (ranked)
1. **Target-pairing confound** *(RUNNING)* — the roster's 3 large-erosion cells all disguise
   →llama-3.1-8b. A **target-varied 3×3 grid** (each source → 3 targets) disentangles source-effect
   from target-effect. If erosion is consistent across targets → source-driven → confound killed.
   *Highest priority.*
2. **≥2 more seeds** on the roster safety cells (cheap; pipeline works).
3. **Lead with the strict genuine-compliance metric** (+ the two-directional calibration); treat raw
   Guard as the sensitive upper bound.

## Stays / demoted
- **Instrument + support:** disguise ladder + persistence metric; source-dependent fingerprint
  erasure (n=4); projection-ablation steering (training-free disguise ≈ trained DPO).
- **Honest negatives:** distinctiveness-predicts-durability (inverts, r=−0.31); launderability from
  source traits (null).

## Venue & verdict (after the decisive tests + the metric fix)
Target: **Findings / a safety or trustworthy-ML venue.** Two contributions, both strengthened by the
metric fix (a refuse-then-leak judge validated at κ=0.79 / F1 0.83 vs a 227-row hand census):

1. **Measurement finding (the lead — concrete, validated, surprising).** Off-the-shelf refusal/safety
   metrics *mis-rank which model is unsafe* under imitation. The naive "strict = not-refusal-led"
   metric (a) crowned **nemotron** as the biggest eroder on **Guard false-positive policy refusals
   that deliver zero harm**, and (b) *hid* the real erosion toward **llama** behind refusal prefaces
   (the refuse-then-leak pattern). Correct measurement needs a refuse-then-leak-aware, adjudication-
   validated judge — standard metrics give the wrong magnitude **and the wrong ranking**.
2. **Corrected erosion result (clean, real, source-general).** With the right metric, benign
   DPO-disguise toward a *permissive* target (**llama-3.1-8b**) induces genuine refuse-then-leak
   safety erosion — **+3.8 pts, p<1e-4, McNemar 94/2, significant across all 3 source models**
   (gpt-oss, qwen, nemotron). The other 3 targets are null (robust to ungated re-judging). A clean
   **target-specific** effect, not a broad target law.

**Not established:** the *target-safety→erosion law*. Corrected erosion correlates with the target's
base genuine-harm rate at R²=0.99, but at **n=4 it is driven entirely by llama**, with target
base-safety confounded against disguise-direction strength (perm-p 0.167, n.s.). A main-track law needs
**n≫4 targets** (new-target adapter training) + disentangling those two — worth doing only if the
llama-anchored trend is judged promising enough.

Instrument = disguise ladder + persistence metric. Backdrop to cite: 2512.09403, Qi et al. (2310.03693),
2605.05427, 2602.09434.
