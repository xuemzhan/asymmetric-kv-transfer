# Response to Reviewer (Audit 1) — Phase-1 Fixes

**Version:** 2026-08-31 (Phase-1 revision)
**Paper:** *Asymmetric and Interaction-Dependent KV State Transfer Across Language Models*

We thank the reviewer for the detailed and constructive report. The four priority
issues identified in the summary have driven this revision. Below we respond
point-by-point. Items marked **[Phase 1]** are completed in this version; items
marked **[Phase 2]** are committed experiments in the release pipeline.

---

## 1. K×V interaction (review §1) — **[Phase 1: analysis added; Phase 2: per-sample statistics for all pairs]**

We agree this is the most important point, and we have made it the centerpiece of
the revision. The four arms are now analyzed as a formal 2×2 factorial design:

- New §4.2 "K×V Interaction Analysis" reports K main effect, V main effect, and
  the interaction Joint − K-only − V-only + Self for all six pairs
  (Table~\ref{tab:interaction} in the paper), with a dedicated statistics script
  (`scripts/factorial_analysis.py`, output `reports/factorial_analysis.json`).
- Flagship 8B→0.6B: interaction **+2.12** (mean over 3 seeds;
  per-sample CI95 [+0.96,+3.26], [+1.07,+3.40], [+1.06,+3.39]; all
  p ≤ 1.3×10⁻³). Teacher V flips from −0.23 (under student K) to +1.90
  (under teacher K).
- The interaction is **not** universal: it is positive on the two large-gap
  0.6B-student pairs (+2.12, +2.15) and negative on the remaining pairs
  (−1.98 to −3.50) and on SQuAD (−2.77/−2.51/−2.51). We therefore adopt the
  reviewer's suggested framing: **K transfers robustly; V transfer is
  interaction- and alignment-dependent**.

Per-sample significance for all six pairs requires per-sample four-arm rows,
which we are persisting in the Phase-2 rerun; the flagship pair already has
per-sample significance.

*Data note:* the reviewer's numeric example for 1.7B→0.6B (K +0.63, V −0.23)
does not match the authoritative baseline (K −0.82, V +0.63; see
`v3_data_baseline.md`). We believe the review was based on an earlier draft;
all numbers in the revision were cross-verified against the JSON reports.

## 2. Capability-gap confound (review §2) — **[Phase 2]**

Accepted. Until the decoupling experiments are run, the paper no longer claims
"capability-gated" V transfer. All occurrences are replaced by
"alignment-/pairing-dependent", and the confound is stated explicitly in
§Boundary Conditions and §Limitations. The controlled matrix
(28→28 misaligned, 36→28 learned selection, 36→36 offset; x = capability gap,
y = alignment quality) is listed as the top-priority experiment in the release
pipeline.

## 3. Transfer success criterion (review §3) — **[Phase 1]**

Accepted. §Method now pre-registers a criterion (applied retrospectively as
re-analysis): EM-preserving (retention ≥ 0.85) plus non-inferior
(EM ≥ Self − 3pp), and LL-positive (significant in all 3 seeds, CI95 excludes 0).
The main table and text use this criterion; the K-only "0.82" case (8B→1.7B,
seed 0) is now reported as a drop under the EM lens, and SQuAD K-only is
described as non-destructive rather than beneficial where appropriate.

## 4. CCA / representation similarity (review §4) — **[Phase 1 wording; Phase 2 spectrum]**

Accepted. The claim "share nearly the same linear subspace" is removed; ρ₁ is now
described as the first canonical direction only. The full canonical spectrum,
SVCCA/PWCCA, linear CKA, held-out reconstruction, and a similarity-to-function
prediction analysis are listed in the reproducibility appendix and planned as
analysis artifacts.

## 5. "V is weight-bound" as hypothesis (review §5) — **[Phase 1 wording; Phase 2 mechanism]**

Accepted. The interpretation no longer attributes V-only failure to downstream
weight-consumption alone; the positive interaction shows routing compatibility
matters. Attention-map comparisons and a W_O-aware V mapper are specified as the
decisive experiments.

## 6. "Beats both full models" overclaim (review §6) — **[Phase 1]**

Fixed. The claim is now "achieves higher gold-answer log-likelihood than both
standalone runs", with EM reported honestly (K-only 0.76 vs teacher 1.00 vs
student 0.43) and the caveat that teacher-forced LL is not a capability measure.

## 7. Direct K-vs-V test (review §7) — **[Phase 1]**

Added. Per-sample D_i = ΔLL_K − ΔLL_V: mean +2.46 on the flagship pair,
CI95 [+1.73,+3.29], [+1.98,+3.57], [+2.53,+4.31], p < 1.4×10⁻⁹ in all seeds.
The paper no longer infers K > V from two separate arm-wise tests.

## 8. Layer hotspot selection bias (review §8) — **[Phase 2]**

Accepted. L8/L12 selection moves to validation-only selection with frozen
test-set evaluation plus a second pair; the "26 of 28 layers add noise" claim is
replaced by "joint full-layer injection is harmful".

## 9. Reproducibility (review §9) — **[Phase 1]**

Added a full appendix: graph/entity generation, templates, answer normalization,
prompt/decoding settings, model checkpoints, mapper details, and the corrected
novelty-probe wording ("synthetically constructed to minimize prior familiarity,
with a no-context probe showing low recall").

## 10. SQuAD scale and beneficial vs non-destructive (review §10) — **[Phase 1 wording; Phase 2 scale]**

The distinction is now explicit, and the SQuAD interaction is reported
(negative, matching the mid-gap pairs). Long-context (1K/2K/4K) extensions are in
the Phase-2 plan.

## 11. Cost crossover (review §11) — **[Phase 1]**

The cost crossover is now explicitly labeled a heuristic arithmetic analysis
(bytes only; no latency/throughput), and its status as a contribution is
softened. A serving benchmark and fp16/amortization sensitivity analysis are in
the Phase-2 plan.

## 12. Controls (review §12) — **[Phase 2]**

Wrong-document, zero/random/moment-matched, identity/no-map, learned layer
selection, held-out hotspot discovery, and (optional) cross-family controls are
specified as the next experiment batch.

## 13. Novelty positioning (review §13) — **[Phase 1]**

The contribution is repositioned as the first systematic functional
decomposition (K-only / V-only) plus the interaction analysis, rather than
"KV cache can transfer" or "K/V have different geometry".

## 14. Title (review §14) — **[Phase 1]**

The title is now *Asymmetric and Interaction-Dependent KV State Transfer Across
Language Models*, matching the factorial results.

---

## Summary of changes

| Item | Change |
|---|---|
| Title / abstract / intro | Rewritten around interaction-dependent asymmetry |
| New analysis | 2×2 factorial (K/V main effects + interaction), direct K-vs-V test |
| New table | Interaction table (6 pairs × 3 seeds) |
| Claims | "beats both" → LL claim; "capability-gated" → alignment/pairing-dependent |
| CCA | ρ₁-only wording; no subspace-equivalence claim |
| Reproducibility | Full appendix added |
| Verification | 65+ automated assertions pass; LaTeX compiles cleanly |
