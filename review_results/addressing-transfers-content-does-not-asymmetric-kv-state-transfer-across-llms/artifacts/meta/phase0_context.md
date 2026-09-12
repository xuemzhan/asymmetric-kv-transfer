# Phase 0: Automated Audit Results

**File**: `/workspace/v3/paper/main.tex` | **Language**: en | **Mode**: quick-audit
**Generated**: 2026-08-30T21:05:12.706945

## Issue Summary (236 total)
- Critical: 1
- Major: 19
- Minor: 216

## Issues by Module

### BIB

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | Check: /workspace/v3/paper/main.tex |
| 2 | — | Minor | P2 | PASS |
| 3 | — | Minor | P2 | entries: 0 |
| 4 | — | Minor | P2 | entries: 0 |

### CITATIONS

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | No citation stacking issues found. |

### DEAI

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | Use --analyze for full analysis |

### EXPERIMENT

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | 422 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 2 | 444 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 3 | 465 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 4 | 465 | Major | P1 | Performance claim is not tied to a concrete metric or numeric result. |
| 5 | 466 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 6 | 472 | Critical | P0 | Conclusion overreaches the reported evidence; avoid universal or guarantee-style claims. |
| 7 | 486 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 8 | 486 | Major | P1 | Performance claim is not tied to a concrete metric or numeric result. |
| 9 | 492 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 10 | 492 | Major | P1 | Performance claim is not tied to a concrete metric or numeric result. |
| 11 | 515 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 12 | 515 | Major | P1 | Performance claim is not tied to a concrete metric or numeric result. |
| 13 | 521 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 14 | 532 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 15 | 532 | Major | P1 | Performance claim is not tied to a concrete metric or numeric result. |
| 16 | 543 | Major | P1 | Performance claim lacks an explicit baseline or comparator. |
| 17 | 385 | Minor | P2 | No ablation or component-level evidence is mentioned; verify that contribution attribution is covered. |
| 18 | 385 | Minor | P2 | No efficiency comparison is mentioned; verify whether runtime, memory, or parameter cost should be reported. |
| 19 | 549 | Major | P1 | Discussion may lack depth: low ratio of explanatory/attribution language (3/130 lines). Add causal analysis explaining why results occur. |
| 20 | 549 | Major | P1 | No citations from Related Work reappear in Discussion. Compare your findings with prior work to strengthen the narrative. |

### FIGURES

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | figures in /workspace/v3/paper/main.tex... |
| 2 | — | Minor | P2 | 3 figures. |
| 3 | — | Minor | P2 | Line 583: figures/fig_cca.pdf |
| 4 | — | Minor | P2 | Line 627: figures/fig_layers.pdf |
| 5 | — | Minor | P2 | Line 674: figures/fig_cost.pdf |
| 6 | — | Minor | P2 | All figures passed check. |

### FORMAT

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | ============================================================ |
| 2 | — | Minor | P2 | Format Check Report |
| 3 | — | Minor | P2 | ============================================================ |
| 4 | — | Minor | P2 | /workspace/v3/paper/main.tex |
| 5 | — | Minor | P2 | UNAVAILABLE |
| 6 | — | Minor | P2 | chktex not found. Install with: apt-get install chktex (Linux) or via TeX Live/MiKTeX |
| 7 | — | Minor | P2 | MODE] chktex not available |
| 8 | — | Minor | P2 | chktex for detailed format checking |

### GRAMMAR

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | [Script]: goal=grammar strength=minimal |
| 2 | — | Minor | P2 | No rule-based issues detected in selected scope. |

### LOGIC

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | 340 | Major | P1 | Method choice lacks explicit justification |
| 2 | 340 | Major | P1 | Method choice lacks explicit justification |

### PRESUBMISSION

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | [A1] Abstract five-element check is incomplete; missing background. |
| 2 | 40 | Minor | P2 | [G2] Long paragraph detected (219 words, 6 sentences); split or add a clearer topic sentence. |

### REFERENCES

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | 393 | Minor | P2 | Reference before definition: \ref{tab:main} at line 393 appears before label definition at line 405 |
| 2 | 498 | Minor | P2 | Unreferenced label: \label{tab:reassembly} is never cited in text |
| 3 | 560 | Minor | P2 | Reference before definition: \ref{tab:cca} at line 560 appears before label definition at line 569 |
| 4 | 606 | Minor | P2 | Reference before definition: \ref{tab:layers} at line 606 appears before label definition at line 612 |
| 5 | 668 | Minor | P2 | Reference before definition: \ref{fig:cost} at line 668 appears before label definition at line 679 |

### SENTENCES

| # | Line | Severity | Priority | Issue |
|---|------|----------|----------|-------|
| 1 | — | Minor | P2 | [Script]: goal=grammar strength=minimal |
| 2 | — | Minor | P2 | SENTENCE (Line 109, 55 words, 5 clauses)  [Script] |
| 3 | — | Minor | P2 | This paper makes five contributions: \begin{enumerate} \item \textbf{Gated K/V functional asymmetry.} We show that K-state transfer recovers task performance on all 6 model pairs (EM  ;   positive on 5/6 pairs,   to  ), while V-state transfer succeeds only on small-gap pairs ( B 4 ,  B 0.6 ) and fails or destabilizes decoding on large-gap pairs. |
| 4 | — | Minor | P2 | This paper makes five contributions: \begin{enumerate} \item \textbf{Gated K/V functional asymmetry.} We show that K-state transfer recovers task performance on all 6 model pairs (EM  ;   positive on 5/6 pairs. to  ). while V-state transfer succeeds only on small-gap pairs ( B 4. B 0.6 ) and fails or destabilizes decoding on large-gap pairs.. |
| 5 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 6 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 7 | — | Minor | P2 | none |
| 8 | — | Minor | P2 | Check: NEEDS-LLM |
| 9 | — | Minor | P2 | Flags:    not-assessed |
| 10 | — | Minor | P2 | SENTENCE (Line 139, 57 words, 5 clauses)  [Script] |
| 11 | — | Minor | P2 | \textbf{Results preview.} Across 6 model pairs spanning 0.6B to 8B parameters, K-only injection improves log-likelihood by   to   on 4/6 pairs (paired  , 3 seeds) and recovers EM   on all pairs, while V-only injection is significant only on the two small-gap pairs ( B 4 :  ;  B 0.6 :  ) and neutral or negative elsewhere ( B 0.6 :  , ns). |
| 12 | — | Minor | P2 | \textbf{Results preview.} Across 6 model pairs spanning 0.6B to 8B parameters. K-only injection improves log-likelihood by   to   on 4/6 pairs (paired. 3 seeds) and recovers EM   on all pairs. while V-only injection is significant only on the two small-gap pairs ( B 4 :  ;  B 0.6 :  ) and neutral or negative elsewhere ( B 0.6 :. ns).. |
| 13 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 14 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 15 | — | Minor | P2 | none |
| 16 | — | Minor | P2 | Check: NEEDS-LLM |
| 17 | — | Minor | P2 | Flags:    not-assessed |
| 18 | — | Minor | P2 | SENTENCE (Line 298, 27 words, 5 clauses)  [Script] |
| 19 | — | Minor | P2 | We study 6 pairs from the Qwen3 family: 8B 0.6B, 4B 1.7B, 4B 0.6B, 8B 1.7B, 1.7B 0.6B, and 8B 4B. |
| 20 | — | Minor | P2 | We study 6 pairs from the Qwen3 family: 8B 0.6B. 4B 1.7B. 4B 0.6B. 8B 1.7B. 1.7B 0.6B. and 8B 4B.. |
| 21 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 22 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 23 | — | Minor | P2 | none |
| 24 | — | Minor | P2 | Check: NEEDS-LLM |
| 25 | — | Minor | P2 | Flags:    not-assessed |
| 26 | — | Minor | P2 | SENTENCE (Line 302, 55 words, 4 clauses)  [Script] |
| 27 | — | Minor | P2 | This design covers three configurations: \begin{itemize} \item \textbf{Equal-layer}: 36 36 (8B 4B), 28 28 (1.7B 0.6B) \item \textbf{Unequal-layer} (proportional mapping): 36 28 (8B 0.6B, 8B 1.7B, 4B 0.6B, 4B 1.7B) \item \textbf{Size ratio}: 2.0  (8B 4B) to 13.3  (8B 0.6B) \end{itemize} |
| 28 | — | Minor | P2 | This design covers three configurations: \begin{itemize} \item \textbf{Equal-layer}: 36 36 (8B 4B). 28 28 (1.7B 0.6B) \item \textbf{Unequal-layer} (proportional mapping): 36 28 (8B 0.6B. 8B 1.7B. 4B 0.6B. 4B 1.7B) \item \textbf{Size ratio}: 2.0  (8B 4B) to 13.3  (8B 0.6B) \end{itemize}. |
| 29 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 30 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 31 | — | Minor | P2 | none |
| 32 | — | Minor | P2 | Check: NEEDS-LLM |
| 33 | — | Minor | P2 | Flags:    not-assessed |
| 34 | — | Minor | P2 | SENTENCE (Line 310, 33 words, 4 clauses)  [Script] |
| 35 | — | Minor | P2 | \textbf{Second domain (SQuAD).} To test whether the K/V asymmetry is specific to our synthetic domain, we additionally evaluate on SQuAD (30 questions, in-document facts, answers 1--5 tokens, doc avg.\ 119 tokens). |
| 36 | — | Minor | P2 | \textbf{Second domain (SQuAD).} To test whether the K/V asymmetry is specific to our synthetic domain. we additionally evaluate on SQuAD (30 questions. in-document facts. answers 1--5 tokens. doc avg.\ 119 tokens).. |
| 37 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 38 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 39 | — | Minor | P2 | none |
| 40 | — | Minor | P2 | Check: NEEDS-LLM |
| 41 | — | Minor | P2 | Flags:    not-assessed |
| 42 | — | Minor | P2 | SENTENCE (Line 348, 18 words, 4 clauses)  [Script] |
| 43 | — | Minor | P2 | We also tested Whitened, Procrustes, CCA, and RidgePerHead mappers; results are invariant to mapper choice (see Robustness, \S ). |
| 44 | — | Minor | P2 | We also tested Whitened. Procrustes. CCA. and RidgePerHead mappers; results are invariant to mapper choice (see Robustness. \S ).. |
| 45 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 46 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 47 | — | Minor | P2 | none |
| 48 | — | Minor | P2 | Check: NEEDS-LLM |
| 49 | — | Minor | P2 | Flags:    not-assessed |
| 50 | — | Minor | P2 | SENTENCE (Line 406, 87 words, 0 clauses)  [Script] |
| 51 | — | Minor | P2 | \small \begin{tabular}{lrrrccr} \toprule Pair & Size & K-only   & V-only   & K sig  & V sig  & EM Self K V (seed 0) \\ \midrule 8B 0.6B  &   &   &   & \checkmark &   & 0.43 0.73 0.29 \\ 4B 0.6B  &    &   &   & \checkmark &   & 0.43 0.84 0.18 \\ 1.7B 0.6B &   &   &   &   & \checkmark & 0.43 1.00 0.80 \\ 8B 4B    &    &   &   & \checkmark & \checkmark & 1.00 0.91 0.88 \\ 4B 1.7B  &   &   &   & \checkmark & \checkmark & 1.00 0.95 0.12 \\ 8B 1.7B  &    &   &   &   & \checkmark & 1.00 0.82 0.07 \\ \bottomrule \end{tabular} \end{table} |
| 52 | — | Minor | P2 | \small \begin{tabular}{lrrrccr} \toprule Pair & Size & K-only   & V-only   & K sig  & V sig  & EM Self K V (seed 0) \\ \midrule 8B 0.6B  &   &   &   & \checkmark &   & 0.43 0.73 0.29 \\ 4B 0.6B  &    &   &   & \checkmark &   & 0.43 0.84 0.18 \\ 1.7B 0.6B &   &   &   &   & \checkmark & 0.43 1.00 0.80 \\ 8B 4B    &    &   &   & \checkmark & \checkmark & 1.00 0.91 0.88 \\ 4B 1.7B  &   &   &   & \checkmark & \checkmark & 1.00 0.95 0.12 \\ 8B 1.7B  &    &   &   &   & \checkmark & 1.00 0.82 0.07 \\ \bottomrule \end{tabular} \end{table} |
| 53 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 54 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 55 | — | Minor | P2 | none |
| 56 | — | Minor | P2 | Check: NEEDS-LLM |
| 57 | — | Minor | P2 | Flags:    not-assessed |
| 58 | — | Minor | P2 | SENTENCE (Line 439, 28 words, 4 clauses)  [Script] |
| 59 | — | Minor | P2 | \textbf{Model size ratio.} The K/V asymmetry strengthens with the capability gap: on 0.6B students (large gaps,  -- ), K transfers (K-only   to  ) while V fails (  to  , ns). |
| 60 | — | Minor | P2 | \textbf{Model size ratio.} The K/V asymmetry strengthens with the capability gap: on 0.6B students (large gaps. -- ). K transfers (K-only   to  ) while V fails (  to. ns).. |
| 61 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 62 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 63 | — | Minor | P2 | none |
| 64 | — | Minor | P2 | Check: NEEDS-LLM |
| 65 | — | Minor | P2 | Flags:    not-assessed |
| 66 | — | Minor | P2 | SENTENCE (Line 477, 51 words, 4 clauses)  [Script] |
| 67 | — | Minor | P2 | \textbf{Takeaway 2.} V transfer is gated by teacher--student capability matching: it succeeds on equal-layer small-gap pairs (8B 4B EM 0.88, 1.7B 0.6B EM 0.80), collapses greedy decoding on 1.7B students (EM 0.07--0.12), and fails entirely on large-gap 0.6B students ( , EM below Self). |
| 68 | — | Minor | P2 | \textbf{Takeaway 2.} V transfer is gated by teacher--student capability matching: it succeeds on equal-layer small-gap pairs (8B 4B EM 0.88. 1.7B 0.6B EM 0.80). collapses greedy decoding on 1.7B students (EM 0.07--0.12). and fails entirely on large-gap 0.6B students (. EM below Self).. |
| 69 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 70 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 71 | — | Minor | P2 | none |
| 72 | — | Minor | P2 | Check: NEEDS-LLM |
| 73 | — | Minor | P2 | Flags:    not-assessed |
| 74 | — | Minor | P2 | SENTENCE (Line 484, 31 words, 4 clauses)  [Script] |
| 75 | — | Minor | P2 | \textbf{Takeaway 3.} Three pairs (8B 4B, 4B 1.7B, 8B 1.7B) saturate at Self EM = 100\%; there, only   is diagnostic, and V's likelihood gains must be interpreted with caution. |
| 76 | — | Minor | P2 | \textbf{Takeaway 3.} Three pairs (8B 4B. 4B 1.7B. 8B 1.7B) saturate at Self EM = 100\%; there. only   is diagnostic. and V's likelihood gains must be interpreted with caution.. |
| 77 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 78 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 79 | — | Minor | P2 | none |
| 80 | — | Minor | P2 | Check: NEEDS-LLM |
| 81 | — | Minor | P2 | Flags:    not-assessed |
| 82 | — | Minor | P2 | SENTENCE (Line 519, 19 words, 4 clauses)  [Script] |
| 83 | — | Minor | P2 | All three produced consistent K-only gains ( ,  ,  ), while Procrustes/CCA mappers are destructive (K-only EM drops to 0.00--0.05). |
| 84 | — | Minor | P2 | All three produced consistent K-only gains (. ). while Procrustes/CCA mappers are destructive (K-only EM drops to 0.00--0.05).. |
| 85 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 86 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 87 | — | Minor | P2 | none |
| 88 | — | Minor | P2 | Check: NEEDS-LLM |
| 89 | — | Minor | P2 | Flags:    not-assessed |
| 90 | — | Minor | P2 | SENTENCE (Line 534, 34 words, 6 clauses)  [Script] |
| 91 | — | Minor | P2 | \textbf{Second domain (SQuAD).} On SQuAD (n=30, 3 seeds, mapper fit on OOD train only): K-only   (all  , EM 0.73--0.80), V-only   (all ns, EM 0.03---destroys decoding), Joint   (EM 0.60--0.73). |
| 92 | — | Minor | P2 | \textbf{Second domain (SQuAD).} On SQuAD (n=30. 3 seeds. mapper fit on OOD train only): K-only   (all. EM 0.73--0.80). V-only   (all ns. EM 0.03---destroys decoding). Joint   (EM 0.60--0.73).. |
| 93 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 94 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 95 | — | Minor | P2 | none |
| 96 | — | Minor | P2 | Check: NEEDS-LLM |
| 97 | — | Minor | P2 | Flags:    not-assessed |
| 98 | — | Minor | P2 | SENTENCE (Line 730, 27 words, 4 clauses)  [Script] |
| 99 | — | Minor | P2 | Focus instead on V/content pathways, which carry task-specific knowledge---and note that V knowledge is concentrated in a few layers, making selective layer distillation a promising target. |
| 100 | — | Minor | P2 | Focus instead on V/content pathways. which carry task-specific knowledge---and note that V knowledge is concentrated in a few layers. making selective layer distillation a promising target.. |
| 101 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 102 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 103 | — | Minor | P2 | none |
| 104 | — | Minor | P2 | Check: NEEDS-LLM |
| 105 | — | Minor | P2 | Flags:    not-assessed |
| 106 | — | Minor | P2 | SENTENCE (Line 749, 33 words, 4 clauses)  [Script] |
| 107 | — | Minor | P2 | \textbf{EM saturation on 3/6 pairs.} 8B 4B, 4B 1.7B, 8B 1.7B saturate at Self EM = 100\%; there, only   is diagnostic, and V's likelihood gains must be interpreted with caution. |
| 108 | — | Minor | P2 | \textbf{EM saturation on 3/6 pairs.} 8B 4B. 4B 1.7B. 8B 1.7B saturate at Self EM = 100\%; there. only   is diagnostic. and V's likelihood gains must be interpreted with caution.. |
| 109 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 110 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 111 | — | Minor | P2 | none |
| 112 | — | Minor | P2 | Check: NEEDS-LLM |
| 113 | — | Minor | P2 | Flags:    not-assessed |
| 114 | — | Minor | P2 | SENTENCE (Line 789, 76 words, 8 clauses)  [Script] |
| 115 | — | Minor | P2 | We have shown that KV state transfer across LLMs exhibits a sharp, capability-gated functional asymmetry: keys (addressing) transfer near-universally across 6 model pairs (EM  ;   positive in 5/6), while values (content) transfer only under equal-layer capability matching (8B 4B EM 0.88, 1.7B 0.6B EM 0.80), collapse greedy decoding on mid-gap students (EM 0.07--0.12), and fail entirely on large-gap pairs ( , ns) despite near-perfect linear alignment (CCA   for both K and V). |
| 116 | — | Minor | P2 | We have shown that KV state transfer across LLMs exhibits a sharp. capability-gated functional asymmetry: keys (addressing) transfer near-universally across 6 model pairs (EM  ;   positive in 5/6). while values (content) transfer only under equal-layer capability matching (8B 4B EM 0.88. 1.7B 0.6B EM 0.80). collapse greedy decoding on mid-gap students (EM 0.07--0.12). and fail entirely on large-gap pairs (. ns) despite near-perfect linear alignment (CCA   for both K and V).. |
| 117 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 118 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 119 | — | Minor | P2 | none |
| 120 | — | Minor | P2 | Check: NEEDS-LLM |
| 121 | — | Minor | P2 | Flags:    not-assessed |
| 122 | — | Minor | P2 | SENTENCE (Line 828, 14 words, 4 clauses)  [Script] |
| 123 | — | Minor | P2 | \bibitem{dao2022flashattention} Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R{\'e}. |
| 124 | — | Minor | P2 | \bibitem{dao2022flashattention} Tri Dao. Dan Fu. Stefano Ermon. Atri Rudra. and Christopher R{\'e}.. |
| 125 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 126 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 127 | — | Minor | P2 | none |
| 128 | — | Minor | P2 | Check: NEEDS-LLM |
| 129 | — | Minor | P2 | Flags:    not-assessed |
| 130 | — | Minor | P2 | SENTENCE (Line 834, 12 words, 5 clauses)  [Script] |
| 131 | — | Minor | P2 | Dery, Zohar Yahav, Henry Prior, Qixuan Feng, Jiajun Shen, and Arthur Szlam. |
| 132 | — | Minor | P2 | Dery. Zohar Yahav. Henry Prior. Qixuan Feng. Jiajun Shen. and Arthur Szlam.. |
| 133 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 134 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 135 | — | Minor | P2 | none |
| 136 | — | Minor | P2 | Check: NEEDS-LLM |
| 137 | — | Minor | P2 | Flags:    not-assessed |
| 138 | — | Minor | P2 | SENTENCE (Line 851, 17 words, 6 clauses)  [Script] |
| 139 | — | Minor | P2 | \bibitem{fu2026c2c} Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, and Yu~Wang. |
| 140 | — | Minor | P2 | \bibitem{fu2026c2c} Tianyu Fu. Zihan Min. Hanling Zhang. Jichao Yan. Guohao Dai. Wanli Ouyang. and Yu~Wang.. |
| 141 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 142 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 143 | — | Minor | P2 | none |
| 144 | — | Minor | P2 | Check: NEEDS-LLM |
| 145 | — | Minor | P2 | Flags:    not-assessed |
| 146 | — | Minor | P2 | SENTENCE (Line 859, 24 words, 8 clauses)  [Script] |
| 147 | — | Minor | P2 | \bibitem{heo2026crossmodel} Taekyung Heo, Rasoul Shafipour, Ritchie Zhao, Maximilian Golub, Mohammad~Mahdi Kamani, Ritika Borkar, Makesh~Tarun Chandran, Pantea Zardoshti, and Bita~Darvish Rouhani. |
| 148 | — | Minor | P2 | \bibitem{heo2026crossmodel} Taekyung Heo. Rasoul Shafipour. Ritchie Zhao. Maximilian Golub. Mohammad~Mahdi Kamani. Ritika Borkar. Makesh~Tarun Chandran. Pantea Zardoshti. and Bita~Darvish Rouhani.. |
| 149 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 150 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 151 | — | Minor | P2 | none |
| 152 | — | Minor | P2 | Check: NEEDS-LLM |
| 153 | — | Minor | P2 | Flags:    not-assessed |
| 154 | — | Minor | P2 | SENTENCE (Line 882, 17 words, 6 clauses)  [Script] |
| 155 | — | Minor | P2 | \bibitem{lee2026translators} Jin-woo Lee, Minkyung Song, Junghyun Oh, Seunghoon Han, Soyoung Park, Gwangseon Jang, and Sungsu Lim. |
| 156 | — | Minor | P2 | \bibitem{lee2026translators} Jin-woo Lee. Minkyung Song. Junghyun Oh. Seunghoon Han. Soyoung Park. Gwangseon Jang. and Sungsu Lim.. |
| 157 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 158 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 159 | — | Minor | P2 | none |
| 160 | — | Minor | P2 | Check: NEEDS-LLM |
| 161 | — | Minor | P2 | Flags:    not-assessed |
| 162 | — | Minor | P2 | SENTENCE (Line 895, 15 words, 5 clauses)  [Script] |
| 163 | — | Minor | P2 | \bibitem{su2024rope} Jianlin Su, Murtadha Ahmed, Yu~Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. |
| 164 | — | Minor | P2 | \bibitem{su2024rope} Jianlin Su. Murtadha Ahmed. Yu~Lu. Shengfeng Pan. Wen Bo. and Yunfeng Liu.. |
| 165 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 166 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 167 | — | Minor | P2 | none |
| 168 | — | Minor | P2 | Check: NEEDS-LLM |
| 169 | — | Minor | P2 | Flags:    not-assessed |
| 170 | — | Minor | P2 | SENTENCE (Line 40, 97 words, 10 clauses)  [Script] |
| 171 | — | Minor | P2 | Across 6 model pairs from the Qwen3 family (0.6B--8B, 3 seeds), we find a sharp functional asymmetry that is gated by the teacher--student capability gap: keys (the addressing substrate) transfer near-universally, recovering task performance on all 6 pairs (exact match  ;   positive on 5/6 pairs,   to  ), while values (the content substrate) transfer only when teacher and student are closely matched ( B 4B: EM 1.00 0.88;  B 0.6 : EM 0.43 0.80) and fail on large-gap pairs ( B 0.6 :  , ns)---despite K and V being equally, almost perfectly linearly alignable (CCA   both). |
| 172 | — | Minor | P2 | Across 6 model pairs from the Qwen3 family (0.6B--8B. 3 seeds). we find a sharp functional asymmetry that is gated by the teacher--student capability gap: keys (the addressing substrate) transfer near-universally. recovering task performance on all 6 pairs (exact match  ;   positive on 5/6 pairs. to  ). while values (the content substrate) transfer only when teacher and student are closely matched ( B 4B: EM 1.00 0.88;  B 0.6 : EM 0.43 0.80) and fail on large-gap pairs ( B 0.6 :. ns)---despite K and V being equally. almost perfectly linearly alignable (CCA   both).. |
| 173 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 174 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 175 | — | Minor | P2 | none |
| 176 | — | Minor | P2 | Check: NEEDS-LLM |
| 177 | — | Minor | P2 | Flags:    not-assessed |
| 178 | — | Minor | P2 | SENTENCE (Line 40, 57 words, 5 clauses)  [Script] |
| 179 | — | Minor | P2 | We localize the V advantage to layers 8/12 (selective injection   vs.\ full-layer V injection  ), replicate the asymmetry on a second domain (SQuAD, where V-only injection collapses EM to 0.03), and show that heterogeneous reassembly (teacher K + student V) beats both full models (  vs.\ teacher\_full and student\_full  ), with a cost crossover at  2050 tokens. |
| 180 | — | Minor | P2 | We localize the V advantage to layers 8/12 (selective injection   vs.\ full-layer V injection  ). replicate the asymmetry on a second domain (SQuAD. where V-only injection collapses EM to 0.03). and show that heterogeneous reassembly (teacher K + student V) beats both full models (  vs.\ teacher\_full and student\_full  ). with a cost crossover at  2050 tokens.. |
| 181 | — | Minor | P2 | Sentence exceeds complexity threshold, split for readability. Applying the split needs --strength moderate or higher. |
| 182 | — | Minor | P2 | none (split proposal only; source not rewritten) |
| 183 | — | Minor | P2 | none |
| 184 | — | Minor | P2 | Check: NEEDS-LLM |
| 185 | — | Minor | P2 | Flags:    not-assessed |

## Pre-Submission Checklist

- [x] No placeholder text (TODO, FIXME, XXX)
- [x] All figures referenced in text
- [ ] All tables referenced in text — Unreferenced: {'tab:reassembly'}
- [ ] Anonymous submission (blind review check) — Author information detected — verify if blind review required
- [x] Consistent math notation
- [ ] Acronyms defined on first use — Potentially undefined: ['COLM', 'FILLED', 'TMLR', 'FINAL', 'ALL']
