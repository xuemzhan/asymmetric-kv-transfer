# Peer Review Report

**Paper**: `/workspace/v3/paper/main.tex` | **Language**: EN | **Mode**: deep-review
**Generated**: 2026-08-30 21:05
**Artifacts**: `/workspace/v3/review_results/addressing-transfers-content-does-not-asymmetric-kv-state-transfer-across-llms`

## Summary

The manuscript examines We study what transfers when migrating key-value (KV) cache state across large language models (LLMs) and argues that Across 6 model pairs from the Qwen3 family (0.6B--8B, 3 seeds), we find a sharp functional asymmetry that is gated by the teacher--student capability gap: keys (the addressing substrate) transfer near-universally, recovering task performance on all 6 pairs (exact match ; positive on 5/6 pairs, to ), while values (the content substrate) transfer only when teacher and student are closely matched (B4B: EM 1.000.88; B0.6: EM 0.430.80) and fail on large-gap pairs (B0.6: , ns)---despite K and V being equally, almost perfectly linearly alignable (CCA both).

Deep review found 1 major, 4 moderate, 0 minor issues. The highest-priority concerns are: Abstract and conclusion claims need explicit evidence traceability; Cross-section numeric consistency should be reconciled.

In its current form, the paper would benefit most from revisions that better align the headline contribution with the presented evidence, clarify the methodological basis of the claims, and tighten the overall argumentative coherence.

## Major Issues

1. In introduction, the manuscript shows a problem with abstract and conclusion claims need explicit evidence traceability. At least one headline claim was detected. Deep review should check whether experiments and conclusion language trace back to the same bounded evidence base. This matters because it weakens the credibility or interpretability of the corresponding claim. The authors should revise this part directly and make the supporting evidence or reasoning explicit. This issue also affects results, conclusion. [LLM]

## Minor Issues

1. In abstract, the manuscript shows a problem with cross-section numeric consistency should be reconciled. Multiple sections contain numeric claims. Confirm that the same quantities reconcile across main text, tables, and appendix material. This matters because it weakens the credibility or interpretability of the corresponding claim. The authors should revise this part directly and make the supporting evidence or reasoning explicit. This issue also affects introduction, related. [LLM]

2. In experiment, the manuscript shows a problem with comparison protocol should make fairness assumptions explicit. Comparative evaluation language was detected. Deep review should verify that baseline tuning, data splits, and reporting conventions are described symmetrically. This matters because it weakens the credibility or interpretability of the corresponding claim. The authors should revise this part directly and make the supporting evidence or reasoning explicit. The quoted text ("Baseline comparison.") sharpens this concern. This issue also affects method. [LLM]

3. In experiment, the manuscript shows a problem with result claims should identify comparison scope and uncertainty. The results section reports comparative performance. Confirm whether the paper states the evaluation scope, variance, and fairness conditions tightly enough for a reviewer. This matters because it weakens the credibility or interpretability of the corresponding claim. The authors should revise this part directly and make the supporting evidence or reasoning explicit. This issue also affects methods. [LLM]

4. In introduction, the manuscript shows a problem with novelty claim should be grounded against the closest prior work. The paper positions itself against prior work, but the current wording should make the closest comparator and the real novelty delta explicit instead of relying on broad superiority language. This matters because it weakens the credibility or interpretability of the corresponding claim. The authors should revise this part directly and make the supporting evidence or reasoning explicit. This issue also affects results. [LLM]

## Recommendation

**Major Revision**. The paper may become publishable, but key issues still affect the credibility, completeness, or transparency of the claims.