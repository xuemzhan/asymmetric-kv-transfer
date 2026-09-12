# Deep Review Report

**Paper**: `/workspace/v3/paper/main.tex` | **Language**: EN | **Mode**: deep-review
**Generated**: 2026-08-30 21:05
**Artifacts**: `/workspace/v3/review_results/addressing-transfers-content-does-not-asymmetric-kv-state-transfer-across-llms`

## Overall Assessment

Deep review found 1 major, 4 moderate, 0 minor issues. The highest-priority concerns are: Abstract and conclusion claims need explicit evidence traceability; Cross-section numeric consistency should be reconciled.

- **Major**: 1
- **Moderate**: 4
- **Minor**: 0

## Academic Pre-Review Committee

### Editor (Desk Reject Screen)

## Editor Pre-Screen (1-10)

Score: 4.0/10
Verdict: Desk Reject

### Desk-Reject Triggers (if any)
- Abstract and conclusion claims need explicit evidence traceability

### Top 3 Reasons (no hedging)
1. Abstract and conclusion claims need explicit evidence traceability

### Fast Fixes (within 1-2 days)
- Clarify introduction to address abstract and conclusion claims need explicit evidence traceability.
- Clarify abstract to address cross-section numeric consistency should be reconciled.
- Clarify introduction to address novelty claim should be grounded against the closest prior work.

### Reviewer 1 (Theory Contribution)

## Theory Contribution Review

### 3 Fatal Theory Holes
1. (introduction) "Across 6 model pairs spanning 0.6B to 8B parameters, K-only injection improves log-likelihood by to on 4/6 pairs (paired , 3 seeds) and recovers EM on all pairs, while V-only injection is significant only on the two small-gap pairs (B4: ; B0.6: ) and neutral or negative elsewhere (B0.6: , ns)." — At least one headline claim was detected. Deep review should check whether experiments and conclusion language trace back to the same bounded evidence base.
2. (introduction) "Prior work transfers entire KV caches without identifying which components (keys vs.\ values) are transferable." — The paper positions itself against prior work, but the current wording should make the closest comparator and the real novelty delta explicit instead of relying on broad superiority language.

### Concrete Moves
- Tighten the paper's theoretical positioning in introduction to resolve abstract and conclusion claims need explicit evidence traceability.
- Tighten the paper's theoretical positioning in introduction to resolve novelty claim should be grounded against the closest prior work.

### Reviewer 3 (Literature Dialogue)

## Literature Dialogue Review

### Closest Prior Work Risks
- (introduction) "Prior work transfers entire KV caches without identifying which components (keys vs.\ values) are transferable." — The paper positions itself against prior work, but the current wording should make the closest comparator and the real novelty delta explicit instead of relying on broad superiority language.

### Gap Claim Risks
- The claimed gap should be defended more explicitly: Novelty claim should be grounded against the closest prior work.

### Fast Fixes
- Name the closest prior comparator in introduction and explain the real novelty delta.

### Reviewer 2 (Methodology & Transparency)

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

### Reviewer 4 (Logic Chain)

## Logic Chain Review

### Breakpoints
- (introduction) "Across 6 model pairs spanning 0.6B to 8B parameters, K-only injection improves log-likelihood by to on 4/6 pairs (paired , 3 seeds) and recovers EM on all pairs, while V-only injection is significant only on the two small-gap pairs (B4: ; B0.6: ) and neutral or negative elsewhere (B0.6: , ns)." — At least one headline claim was detected. Deep review should check whether experiments and conclusion language trace back to the same bounded evidence base.

### Structural Fix Moves
- Add one explicit bridge sentence in introduction so the argument chain closes cleanly.

### Committee Consensus

## Committee Consensus

Overall Score: 4.0/10
Editor Verdict: Desk Reject

### Score Formula
- base 9.0
- minus 1.5 * major (1)
- minus 0.7 * moderate (4)
- minus 0.2 * minor (0)
- floor 1.0
- desk reject cap 4.0

### Top 3 Issues To Fix First
1. Abstract and conclusion claims need explicit evidence traceability
2. Cross-section numeric consistency should be reconciled
3. Comparison protocol should make fairness assumptions explicit

## Paper Summary

# Paper Summary: Addressing Transfers, Content Does Not: \\ Asymmetric KV State Transfer Across LLMs

## Research Question
- We study what transfers when migrating key-value (KV) cache state across large language models (LLMs)

## Core Thesis
- Across 6 model pairs from the Qwen3 family (0.6B--8B, 3 seeds), we find a sharp functional asymmetry that is gated by the teacher--student capability gap: keys (the addressing substrate) transfer near-universally, recovering task performance on all 6 pairs (exact match ; positive on 5/6 pairs, to ), while values (the content substrate) transfer only when teacher and student are closely matched (B4B: EM 1.000.88; B0.6: EM 0.430.80) and fail on large-gap pairs (B0.6: , ns)---despite K and V being equally, almost perfectly linearly alignable (CCA both).

## Headline Claims
- Across 6 model pairs from the Qwen3 family (0.6B--8B, 3 seeds), we find a sharp functional asymmetry that is gated by the teacher--student capability gap: keys (the addressing substrate) transfer near-universally, recovering task performance on all 6 pairs (exact match ; positive on 5/6 pairs, to ), while values (the content substrate) transfer only when teacher and student are closely matched (B4B: EM 1.000.88; B0.6: EM 0.430.80) and fail on large-gap pairs (B0.6: , ns)---despite K and V being equally, almost perfectly linearly alignable (CCA both).
- Our results establish that KV state transfer is asymmetric by construction: addressing geometry is composable across models, while content is weight-bound. abstract
- No systematic analysis exists for KV cache transfer. enumerate We introduce the concept of gated asymmetric KV transferability: keys (the addressing substrate) transfer across models, while values (the content substrate) transfer only when the teacher--student capability gap is small.
- This paper makes five contributions: enumerate Gated K/V functional asymmetry.
- We show that K-state transfer recovers task performance on all 6 model pairs (EM ; positive on 5/6 pairs, to ), while V-state transfer succeeds only on small-gap pairs (B4, B0.6) and fails or destabilizes decoding on large-gap pairs.
- We show that geometric compatibility is necessary but not sufficient; the bottleneck lies in weight-parameterized content consumption.

## Section Map
- abstract (40-61): 218 words
- introduction (66-157): 598 words
- related (158-260): 533 words
- method (261-384): 603 words
- experiment (385-548): 939 words
- discussion (549-717): 773 words
- discussion_2 (718-913): 970 words

## Closure Targets
- No closure target was extracted automatically.

## Major Issues

### M1: Abstract and conclusion claims need explicit evidence traceability
- **Type**: claim_accuracy
- **Source**: [LLM] via `claims_vs_evidence`
- **Confidence**: medium
- **Section**: introduction
- **Related Sections**: introduction, results, conclusion
- **Root Cause Key**: `abstract-and-conclusion-claims-need-explicit-evidence-traceability`
- **Quote Verified**: no
- **Quote**: `Across 6 model pairs spanning 0.6B to 8B parameters, K-only injection improves log-likelihood by to on 4/6 pairs (paired , 3 seeds) and recovers EM on all pairs, while V-only injection is significant only on the two small-gap pairs (B4: ; B0.6: ) and neutral or negative elsewhere (B0.6: , ns).`
- **Explanation**: At least one headline claim was detected. Deep review should check whether experiments and conclusion language trace back to the same bounded evidence base.

## Moderate Issues

### M1: Cross-section numeric consistency should be reconciled
- **Type**: presentation
- **Source**: [LLM] via `notation_and_numeric_consistency`
- **Confidence**: medium
- **Section**: abstract
- **Related Sections**: abstract, introduction, related
- **Root Cause Key**: `cross-section-numeric-consistency-should-be-reconciled`
- **Quote Verified**: no
- **Quote**: `Across 6 model pairs from the Qwen3 family (0.6B--8B, 3 seeds), we find a sharp functional asymmetry that is gated by the teacher--student capability gap: keys (the addressing substrate) transfer near-universally, recovering task performance on all 6 pairs (exact match ; positive on 5/6 pairs, to ), while values (the content substrate) transfer only when teacher and student are closely matched (B4B: EM 1.000.88; B0.6: EM 0.430.80) and fail on large-gap pairs (B0.6: , ns)---despite K and V being equally, almost perfectly linearly alignable (CCA both).`
- **Explanation**: Multiple sections contain numeric claims. Confirm that the same quantities reconcile across main text, tables, and appendix material.

### M2: Comparison protocol should make fairness assumptions explicit
- **Type**: methodology
- **Source**: [LLM] via `evaluation_fairness_and_reproducibility`
- **Confidence**: medium
- **Section**: experiment
- **Related Sections**: method, experiment
- **Root Cause Key**: `comparison-protocol-should-make-fairness-assumptions-explicit`
- **Quote Verified**: yes
- **Quote**: `Baseline comparison.`
- **Explanation**: Comparative evaluation language was detected. Deep review should verify that baseline tuning, data splits, and reporting conventions are described symmetrically.

### M3: Result claims should identify comparison scope and uncertainty
- **Type**: methodology
- **Source**: [LLM] via `evaluation_fairness_and_reproducibility`
- **Confidence**: medium
- **Section**: experiment
- **Related Sections**: experiment, methods
- **Root Cause Key**: `result-claims-should-identify-comparison-scope-and-uncertainty`
- **Quote Verified**: no
- **Quote**: `K-only injection outperforms V-only on all three 0.6B-student pairs (8B0.6B: vs ; 4B0.6B: vs ).`
- **Explanation**: The results section reports comparative performance. Confirm whether the paper states the evaluation scope, variance, and fairness conditions tightly enough for a reviewer.

### M4: Novelty claim should be grounded against the closest prior work
- **Type**: claim_accuracy
- **Source**: [LLM] via `prior_art_and_novelty_grounding`
- **Confidence**: medium
- **Section**: introduction
- **Related Sections**: introduction, results
- **Root Cause Key**: `novelty-claim-should-be-grounded-against-the-closest-prior-work`
- **Quote Verified**: no
- **Quote**: `Prior work transfers entire KV caches without identifying which components (keys vs.\ values) are transferable.`
- **Explanation**: The paper positions itself against prior work, but the current wording should make the closest comparator and the real novelty delta explicit instead of relying on broad superiority language.

## Phase 0 Automated Findings

### [Script] BIB

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | Check: /workspace/v3/paper/main.tex |
| --- | Minor | PASS |
| --- | Minor | entries: 0 |
| --- | Minor | entries: 0 |

### [Script] CITATIONS

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | No citation stacking issues found. |

### [Script] DEAI

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | Use --analyze for full analysis |

### [Script] EXPERIMENT

| Line | Severity | Issue |
|------|----------|-------|
| 422 | Major | Performance claim lacks an explicit baseline or comparator. |
| 444 | Major | Performance claim lacks an explicit baseline or comparator. |
| 465 | Major | Performance claim lacks an explicit baseline or comparator. |
| 465 | Major | Performance claim is not tied to a concrete metric or numeric result. |
| 466 | Major | Performance claim lacks an explicit baseline or comparator. |
| 472 | Critical | Conclusion overreaches the reported evidence; avoid universal or guarantee-style claims. |
| 486 | Major | Performance claim lacks an explicit baseline or comparator. |
| 486 | Major | Performance claim is not tied to a concrete metric or numeric result. |
| 492 | Major | Performance claim lacks an explicit baseline or comparator. |
| 492 | Major | Performance claim is not tied to a concrete metric or numeric result. |
| 515 | Major | Performance claim lacks an explicit baseline or comparator. |
| 515 | Major | Performance claim is not tied to a concrete metric or numeric result. |
| 521 | Major | Performance claim lacks an explicit baseline or comparator. |
| 532 | Major | Performance claim lacks an explicit baseline or comparator. |
| 532 | Major | Performance claim is not tied to a concrete metric or numeric result. |
| 543 | Major | Performance claim lacks an explicit baseline or comparator. |
| 385 | Minor | No ablation or component-level evidence is mentioned; verify that contribution attribution is covered. |
| 385 | Minor | No efficiency comparison is mentioned; verify whether runtime, memory, or parameter cost should be reported. |
| 549 | Major | Discussion may lack depth: low ratio of explanatory/attribution language (3/130 lines). Add causal analysis explaining why results occur. |
| 549 | Major | No citations from Related Work reappear in Discussion. Compare your findings with prior work to strengthen the narrative. |

### [Script] FIGURES

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | figures in /workspace/v3/paper/main.tex... |
| --- | Minor | 3 figures. |
| --- | Minor | Line 583: figures/fig_cca.pdf |
| --- | Minor | Line 627: figures/fig_layers.pdf |
| --- | Minor | Line 674: figures/fig_cost.pdf |
| --- | Minor | All figures passed check. |

### [Script] FORMAT

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | ============================================================ |
| --- | Minor | Format Check Report |
| --- | Minor | ============================================================ |
| --- | Minor | /workspace/v3/paper/main.tex |
| --- | Minor | UNAVAILABLE |
| --- | Minor | chktex not found. Install with: apt-get install chktex (Linux) or via TeX Live/MiKTeX |
| --- | Minor | MODE] chktex not available |
| --- | Minor | chktex for detailed format checking |

### [Script] GRAMMAR

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | [Script]: goal=grammar strength=minimal |
| --- | Minor | No rule-based issues detected in selected scope. |

### [Script] LOGIC

| Line | Severity | Issue |
|------|----------|-------|
| 340 | Major | Method choice lacks explicit justification |
| 340 | Major | Method choice lacks explicit justification |

### [Script] PRESUBMISSION

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | [A1] Abstract five-element check is incomplete; missing background. |
| 40 | Minor | [G2] Long paragraph detected (219 words, 6 sentences); split or add a clearer topic sentence. |

### [Script] REFERENCES

| Line | Severity | Issue |
|------|----------|-------|
| 393 | Minor | Reference before definition: \ref{tab:main} at line 393 appears before label definition at line 405 |
| 498 | Minor | Unreferenced label: \label{tab:reassembly} is never cited in text |
| 560 | Minor | Reference before definition: \ref{tab:cca} at line 560 appears before label definition at line 569 |
| 606 | Minor | Reference before definition: \ref{tab:layers} at line 606 appears before label definition at line 612 |
| 668 | Minor | Reference before definition: \ref{fig:cost} at line 668 appears before label definition at line 679 |

### [Script] SENTENCES

| Line | Severity | Issue |
|------|----------|-------|
| --- | Minor | [Script]: goal=grammar strength=minimal |
| --- | Minor | SENTENCE (Line 109, 55 words, 5 clauses)  [Script] |
| --- | Minor | This paper makes five contributions: \begin{enumerate} \item \textbf{Gated K/V functional asymmetry.} We show that K-state transfer recovers task performance on all 6 model pairs (EM  ;   positive on 5/6 pairs,   to  ), while V-state transfer succeeds only on small-gap pairs ( B 4 ,  B 0.6 ) and fails or destabilizes decoding on large-gap pairs. |
| --- | Minor | This paper makes five contributions: \begin{enumerate} \item \textbf{Gated K/V functional asymmetry.} We show that K-state transfer recovers task performance on all 6 model pairs (EM  ;   positive on 5/6 pairs. to  ). while V-state transfer succeeds only on small-gap pairs ( B 4. B 0.6 ) and fails or destabilizes decoding on large-gap pairs.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 139, 57 words, 5 clauses)  [Script] |
| --- | Minor | \textbf{Results preview.} Across 6 model pairs spanning 0.6B to 8B parameters, K-only injection improves log-likelihood by   to   on 4/6 pairs (paired  , 3 seeds) and recovers EM   on all pairs, while V-only injection is significant only on the two small-gap pairs ( B 4 :  ;  B 0.6 :  ) and neutral or negative elsewhere ( B 0.6 :  , ns). |
| --- | Minor | \textbf{Results preview.} Across 6 model pairs spanning 0.6B to 8B parameters. K-only injection improves log-likelihood by   to   on 4/6 pairs (paired. 3 seeds) and recovers EM   on all pairs. while V-only injection is significant only on the two small-gap pairs ( B 4 :  ;  B 0.6 :  ) and neutral or negative elsewhere ( B 0.6 :. ns).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 298, 27 words, 5 clauses)  [Script] |
| --- | Minor | We study 6 pairs from the Qwen3 family: 8B 0.6B, 4B 1.7B, 4B 0.6B, 8B 1.7B, 1.7B 0.6B, and 8B 4B. |
| --- | Minor | We study 6 pairs from the Qwen3 family: 8B 0.6B. 4B 1.7B. 4B 0.6B. 8B 1.7B. 1.7B 0.6B. and 8B 4B.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 302, 55 words, 4 clauses)  [Script] |
| --- | Minor | This design covers three configurations: \begin{itemize} \item \textbf{Equal-layer}: 36 36 (8B 4B), 28 28 (1.7B 0.6B) \item \textbf{Unequal-layer} (proportional mapping): 36 28 (8B 0.6B, 8B 1.7B, 4B 0.6B, 4B 1.7B) \item \textbf{Size ratio}: 2.0  (8B 4B) to 13.3  (8B 0.6B) \end{itemize} |
| --- | Minor | This design covers three configurations: \begin{itemize} \item \textbf{Equal-layer}: 36 36 (8B 4B). 28 28 (1.7B 0.6B) \item \textbf{Unequal-layer} (proportional mapping): 36 28 (8B 0.6B. 8B 1.7B. 4B 0.6B. 4B 1.7B) \item \textbf{Size ratio}: 2.0  (8B 4B) to 13.3  (8B 0.6B) \end{itemize}. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 310, 33 words, 4 clauses)  [Script] |
| --- | Minor | \textbf{Second domain (SQuAD).} To test whether the K/V asymmetry is specific to our synthetic domain, we additionally evaluate on SQuAD (30 questions, in-document facts, answers 1--5 tokens, doc avg.\ 119 tokens). |
| --- | Minor | \textbf{Second domain (SQuAD).} To test whether the K/V asymmetry is specific to our synthetic domain. we additionally evaluate on SQuAD (30 questions. in-document facts. answers 1--5 tokens. doc avg.\ 119 tokens).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 348, 18 words, 4 clauses)  [Script] |
| --- | Minor | We also tested Whitened, Procrustes, CCA, and RidgePerHead mappers; results are invariant to mapper choice (see Robustness, \S ). |
| --- | Minor | We also tested Whitened. Procrustes. CCA. and RidgePerHead mappers; results are invariant to mapper choice (see Robustness. \S ).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 406, 87 words, 0 clauses)  [Script] |
| --- | Minor | \small \begin{tabular}{lrrrccr} \toprule Pair & Size & K-only   & V-only   & K sig  & V sig  & EM Self K V (seed 0) \\ \midrule 8B 0.6B  &   &   &   & \checkmark &   & 0.43 0.73 0.29 \\ 4B 0.6B  &    &   &   & \checkmark &   & 0.43 0.84 0.18 \\ 1.7B 0.6B &   &   &   &   & \checkmark & 0.43 1.00 0.80 \\ 8B 4B    &    &   &   & \checkmark & \checkmark & 1.00 0.91 0.88 \\ 4B 1.7B  &   &   &   & \checkmark & \checkmark & 1.00 0.95 0.12 \\ 8B 1.7B  &    &   &   &   & \checkmark & 1.00 0.82 0.07 \\ \bottomrule \end{tabular} \end{table} |
| --- | Minor | \small \begin{tabular}{lrrrccr} \toprule Pair & Size & K-only   & V-only   & K sig  & V sig  & EM Self K V (seed 0) \\ \midrule 8B 0.6B  &   &   &   & \checkmark &   & 0.43 0.73 0.29 \\ 4B 0.6B  &    &   &   & \checkmark &   & 0.43 0.84 0.18 \\ 1.7B 0.6B &   &   &   &   & \checkmark & 0.43 1.00 0.80 \\ 8B 4B    &    &   &   & \checkmark & \checkmark & 1.00 0.91 0.88 \\ 4B 1.7B  &   &   &   & \checkmark & \checkmark & 1.00 0.95 0.12 \\ 8B 1.7B  &    &   &   &   & \checkmark & 1.00 0.82 0.07 \\ \bottomrule \end{tabular} \end{table} |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 439, 28 words, 4 clauses)  [Script] |
| --- | Minor | \textbf{Model size ratio.} The K/V asymmetry strengthens with the capability gap: on 0.6B students (large gaps,  -- ), K transfers (K-only   to  ) while V fails (  to  , ns). |
| --- | Minor | \textbf{Model size ratio.} The K/V asymmetry strengthens with the capability gap: on 0.6B students (large gaps. -- ). K transfers (K-only   to  ) while V fails (  to. ns).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 477, 51 words, 4 clauses)  [Script] |
| --- | Minor | \textbf{Takeaway 2.} V transfer is gated by teacher--student capability matching: it succeeds on equal-layer small-gap pairs (8B 4B EM 0.88, 1.7B 0.6B EM 0.80), collapses greedy decoding on 1.7B students (EM 0.07--0.12), and fails entirely on large-gap 0.6B students ( , EM below Self). |
| --- | Minor | \textbf{Takeaway 2.} V transfer is gated by teacher--student capability matching: it succeeds on equal-layer small-gap pairs (8B 4B EM 0.88. 1.7B 0.6B EM 0.80). collapses greedy decoding on 1.7B students (EM 0.07--0.12). and fails entirely on large-gap 0.6B students (. EM below Self).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 484, 31 words, 4 clauses)  [Script] |
| --- | Minor | \textbf{Takeaway 3.} Three pairs (8B 4B, 4B 1.7B, 8B 1.7B) saturate at Self EM = 100\%; there, only   is diagnostic, and V's likelihood gains must be interpreted with caution. |
| --- | Minor | \textbf{Takeaway 3.} Three pairs (8B 4B. 4B 1.7B. 8B 1.7B) saturate at Self EM = 100\%; there. only   is diagnostic. and V's likelihood gains must be interpreted with caution.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 519, 19 words, 4 clauses)  [Script] |
| --- | Minor | All three produced consistent K-only gains ( ,  ,  ), while Procrustes/CCA mappers are destructive (K-only EM drops to 0.00--0.05). |
| --- | Minor | All three produced consistent K-only gains (. ). while Procrustes/CCA mappers are destructive (K-only EM drops to 0.00--0.05).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 534, 34 words, 6 clauses)  [Script] |
| --- | Minor | \textbf{Second domain (SQuAD).} On SQuAD (n=30, 3 seeds, mapper fit on OOD train only): K-only   (all  , EM 0.73--0.80), V-only   (all ns, EM 0.03---destroys decoding), Joint   (EM 0.60--0.73). |
| --- | Minor | \textbf{Second domain (SQuAD).} On SQuAD (n=30. 3 seeds. mapper fit on OOD train only): K-only   (all. EM 0.73--0.80). V-only   (all ns. EM 0.03---destroys decoding). Joint   (EM 0.60--0.73).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 730, 27 words, 4 clauses)  [Script] |
| --- | Minor | Focus instead on V/content pathways, which carry task-specific knowledge---and note that V knowledge is concentrated in a few layers, making selective layer distillation a promising target. |
| --- | Minor | Focus instead on V/content pathways. which carry task-specific knowledge---and note that V knowledge is concentrated in a few layers. making selective layer distillation a promising target.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 749, 33 words, 4 clauses)  [Script] |
| --- | Minor | \textbf{EM saturation on 3/6 pairs.} 8B 4B, 4B 1.7B, 8B 1.7B saturate at Self EM = 100\%; there, only   is diagnostic, and V's likelihood gains must be interpreted with caution. |
| --- | Minor | \textbf{EM saturation on 3/6 pairs.} 8B 4B. 4B 1.7B. 8B 1.7B saturate at Self EM = 100\%; there. only   is diagnostic. and V's likelihood gains must be interpreted with caution.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 789, 76 words, 8 clauses)  [Script] |
| --- | Minor | We have shown that KV state transfer across LLMs exhibits a sharp, capability-gated functional asymmetry: keys (addressing) transfer near-universally across 6 model pairs (EM  ;   positive in 5/6), while values (content) transfer only under equal-layer capability matching (8B 4B EM 0.88, 1.7B 0.6B EM 0.80), collapse greedy decoding on mid-gap students (EM 0.07--0.12), and fail entirely on large-gap pairs ( , ns) despite near-perfect linear alignment (CCA   for both K and V). |
| --- | Minor | We have shown that KV state transfer across LLMs exhibits a sharp. capability-gated functional asymmetry: keys (addressing) transfer near-universally across 6 model pairs (EM  ;   positive in 5/6). while values (content) transfer only under equal-layer capability matching (8B 4B EM 0.88. 1.7B 0.6B EM 0.80). collapse greedy decoding on mid-gap students (EM 0.07--0.12). and fail entirely on large-gap pairs (. ns) despite near-perfect linear alignment (CCA   for both K and V).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 828, 14 words, 4 clauses)  [Script] |
| --- | Minor | \bibitem{dao2022flashattention} Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R{\'e}. |
| --- | Minor | \bibitem{dao2022flashattention} Tri Dao. Dan Fu. Stefano Ermon. Atri Rudra. and Christopher R{\'e}.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 834, 12 words, 5 clauses)  [Script] |
| --- | Minor | Dery, Zohar Yahav, Henry Prior, Qixuan Feng, Jiajun Shen, and Arthur Szlam. |
| --- | Minor | Dery. Zohar Yahav. Henry Prior. Qixuan Feng. Jiajun Shen. and Arthur Szlam.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 851, 17 words, 6 clauses)  [Script] |
| --- | Minor | \bibitem{fu2026c2c} Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, and Yu~Wang. |
| --- | Minor | \bibitem{fu2026c2c} Tianyu Fu. Zihan Min. Hanling Zhang. Jichao Yan. Guohao Dai. Wanli Ouyang. and Yu~Wang.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 859, 24 words, 8 clauses)  [Script] |
| --- | Minor | \bibitem{heo2026crossmodel} Taekyung Heo, Rasoul Shafipour, Ritchie Zhao, Maximilian Golub, Mohammad~Mahdi Kamani, Ritika Borkar, Makesh~Tarun Chandran, Pantea Zardoshti, and Bita~Darvish Rouhani. |
| --- | Minor | \bibitem{heo2026crossmodel} Taekyung Heo. Rasoul Shafipour. Ritchie Zhao. Maximilian Golub. Mohammad~Mahdi Kamani. Ritika Borkar. Makesh~Tarun Chandran. Pantea Zardoshti. and Bita~Darvish Rouhani.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 882, 17 words, 6 clauses)  [Script] |
| --- | Minor | \bibitem{lee2026translators} Jin-woo Lee, Minkyung Song, Junghyun Oh, Seunghoon Han, Soyoung Park, Gwangseon Jang, and Sungsu Lim. |
| --- | Minor | \bibitem{lee2026translators} Jin-woo Lee. Minkyung Song. Junghyun Oh. Seunghoon Han. Soyoung Park. Gwangseon Jang. and Sungsu Lim.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 895, 15 words, 5 clauses)  [Script] |
| --- | Minor | \bibitem{su2024rope} Jianlin Su, Murtadha Ahmed, Yu~Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. |
| --- | Minor | \bibitem{su2024rope} Jianlin Su. Murtadha Ahmed. Yu~Lu. Shengfeng Pan. Wen Bo. and Yunfeng Liu.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 40, 97 words, 10 clauses)  [Script] |
| --- | Minor | Across 6 model pairs from the Qwen3 family (0.6B--8B, 3 seeds), we find a sharp functional asymmetry that is gated by the teacher--student capability gap: keys (the addressing substrate) transfer near-universally, recovering task performance on all 6 pairs (exact match  ;   positive on 5/6 pairs,   to  ), while values (the content substrate) transfer only when teacher and student are closely matched ( B 4B: EM 1.00 0.88;  B 0.6 : EM 0.43 0.80) and fail on large-gap pairs ( B 0.6 :  , ns)---despite K and V being equally, almost perfectly linearly alignable (CCA   both). |
| --- | Minor | Across 6 model pairs from the Qwen3 family (0.6B--8B. 3 seeds). we find a sharp functional asymmetry that is gated by the teacher--student capability gap: keys (the addressing substrate) transfer near-universally. recovering task performance on all 6 pairs (exact match  ;   positive on 5/6 pairs. to  ). while values (the content substrate) transfer only when teacher and student are closely matched ( B 4B: EM 1.00 0.88;  B 0.6 : EM 0.43 0.80) and fail on large-gap pairs ( B 0.6 :. ns)---despite K and V being equally. almost perfectly linearly alignable (CCA   both).. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |
| --- | Minor | SENTENCE (Line 40, 57 words, 5 clauses)  [Script] |
| --- | Minor | We localize the V advantage to layers 8/12 (selective injection   vs.\ full-layer V injection  ), replicate the asymmetry on a second domain (SQuAD, where V-only injection collapses EM to 0.03), and show that heterogeneous reassembly (teacher K + student V) beats both full models (  vs.\ teacher\_full and student\_full  ), with a cost crossover at  2050 tokens. |
| --- | Minor | We localize the V advantage to layers 8/12 (selective injection   vs.\ full-layer V injection  ). replicate the asymmetry on a second domain (SQuAD. where V-only injection collapses EM to 0.03). and show that heterogeneous reassembly (teacher K + student V) beats both full models (  vs.\ teacher\_full and student\_full  ). with a cost crossover at  2050 tokens.. |
| --- | Minor | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| --- | Minor | none (split proposal only; source not rewritten) |
| --- | Minor | none |
| --- | Minor | Check: NEEDS-LLM |
| --- | Minor | Flags:    not-assessed |

## Decision Signals

- **Committee Score**: 4.0/10
- **Editor Verdict**: Desk Reject
- **Reviewer Recommendation**: Major Revision
- **Issue Bundle**: 1 major / 4 moderate / 0 minor

## Revision Roadmap

### Priority 1 --- Must Address (Blocking)

- [ ] Abstract and conclusion claims need explicit evidence traceability ([LLM]; introduction)

### Priority 2 --- Strongly Recommended

- [ ] Cross-section numeric consistency should be reconciled ([LLM]; abstract)
- [ ] Comparison protocol should make fairness assumptions explicit ([LLM]; experiment)
- [ ] Result claims should identify comparison scope and uncertainty ([LLM]; experiment)
- [ ] Novelty claim should be grounded against the closest prior work ([LLM]; introduction)
