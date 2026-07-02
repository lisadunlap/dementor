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
- **Roster generalization — PRELIMINARY + target-confounded (independently reviewed).** 4 real
  sources show a base→DPO Guard-unsafe rise, but the design **aliases source with target**: all 3
  large-erosion cells disguise toward the *same* target (llama-3.1-8b), and the one different-target
  cell (llama→qwen) is small and reverses. So "generalizes across sources" is **not yet supported** —
  it is equally consistent with "disguising *toward* llama erodes safety." The raw Guard magnitude is
  also inflated: **78–95% of "unsafe" flags are refusal-led** (qwen: ~59% degenerate repeated-refusal
  *loops*). On the strict genuine-compliance metric the honest effect is a **few points**,
  two-sided-significant for only **gpt-oss + nemotron** (qwen one-sided; llama reverses). The
  **target-varied 3×3 run (Gap 1) is the decisive test** — until it lands, the load-bearing safety
  evidence is the Qwen cell below, not the roster.
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

## Venue
A safety/alignment or "science of LLMs" venue. The safety framing + roster generalization is the
main-track-caliber contribution; the n=4 fingerprint story alone is Findings/workshop tier.
