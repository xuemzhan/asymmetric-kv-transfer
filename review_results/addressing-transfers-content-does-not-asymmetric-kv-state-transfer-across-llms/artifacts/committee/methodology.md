## Methodology Transparency Review (SRQR-aware)

### MUST-FIX (submission blockers)
- No methodology blocker was surfaced by the fallback pass.

### SHOULD-FIX (quality improvements)
- (abstract) "Across 6 model pairs from the Qwen3 family (0.6B--8B, 3 seeds), we find a sharp functional asymmetry that is gated by the teacher--student capability gap: keys (the addressing substrate) transfer near-universally, recovering task performance on all 6 pairs (exact match ; positive on 5/6 pairs, to ), while values (the content substrate) transfer only when teacher and student are closely matched (B4B: EM 1.000.88; B0.6: EM 0.430.80) and fail on large-gap pairs (B0.6: , ns)---despite K and V being equally, almost perfectly linearly alignable (CCA both)." — Multiple sections contain numeric claims. Confirm that the same quantities reconcile across main text, tables, and appendix material.
- (experiment) "Baseline comparison." — Comparative evaluation language was detected. Deep review should verify that baseline tuning, data splits, and reporting conventions are described symmetrically.
- (experiment) "K-only injection outperforms V-only on all three 0.6B-student pairs (8B0.6B: vs ; 4B0.6B: vs )." — The results section reports comparative performance. Confirm whether the paper states the evaluation scope, variance, and fairness conditions tightly enough for a reviewer.

### SRQR Checklist Deltas
- Sampling rationale: clarify how the evidence base supports the paper's strongest claims.
- Data collection details (time/place/duration): add context when results depend on specific settings.
- Coding process (stages, coders, disagreement resolution): specify if qualitative or hybrid analysis is used.
- Saturation: state whether the evidence scope is exhaustive or bounded.
- Triangulation: explain whether multiple evidence sources were reconciled.
- Reflexivity: acknowledge researcher choices that shape interpretation.
