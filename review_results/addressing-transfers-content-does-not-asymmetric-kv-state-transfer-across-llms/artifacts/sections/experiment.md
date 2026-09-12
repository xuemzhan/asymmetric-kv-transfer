## Evaluation
sec:results
Section~sec:method describes the task, models, injection protocol, and metrics.
Below we present results.
### K/V Transfer Asymmetry Across 6 Pairs
Table~tab:main presents the core result:
K-state transfer recovers task performance in 5/6 pairs,
while V-state transfer is weaker in all pairs.
Head-to-head.
K-only injection outperforms V-only on all three 0.6B-student pairs
(8B0.6B: vs ; 4B0.6B: vs ).
On 1.7B students the likelihood picture inverts---V-only exceeds
K-only on 8B1.7B ( vs )---but V-only collapses greedy EM
to 0.07--0.12, destroying task success while K-only preserves it (EM 0.82--0.95).
The K/V functional asymmetry is thus strongest on the largest capability gap
and persists wherever it is measured on task success.
Baseline comparison.
Self (doc-only KV) provides the negative baseline.
K-only injection improves over Self by to 
on 5/6 pairs (the exception, 1.7B0.6B at , still achieves
EM 1.00---perfect decoding despite negative teacher-forced likelihood),
demonstrating functional value from teacher K state.
### Deep Dive: Pair Characteristics
Model size ratio.
The K/V asymmetry strengthens with the capability gap:
on 0.6B students (large gaps, --), K transfers
(K-only to ) while V fails ( to , ns).
On 1.7B students (mid gaps, --), K transfers
(EM 0.82--0.95) and V produces likelihood gains (, )
but collapses greedy EM (0.07--0.12).
On the smallest gaps (8B4B: ), V transfers cleanly
(, EM 0.88).
Layer count.
V transfers only on equal-layer pairs:
8B4B (3636, EM 0.88) and 1.7B0.6B (2828, EM 0.80).
On every unequal-layer pair (3628), V fails or collapses.
K transfers on both equal- and unequal-layer pairs (EM ).
Layer alignment is a necessary condition for V, but not for K.
Boundary cases.
Two pairs require special attention:
itemize
 1.7B0.6B (equal-layer, negative K-only ):
 the weakest teacher produces negative teacher-forced likelihood
 yet perfect EM (1.00). V transfers here (, EM 0.80)---the only
 0.6B-student pair where V works, consistent with equal-layer alignment
 outweighing the size gap.
 8B4B (EM saturation): Self EM = 100\%, so 
 is partially calibration gain; the task is already solved, and transfer
 still improves likelihood and preserves EM (0.88).
itemize
### Takeaways
Takeaway 1.
K-state transfer succeeds near-universally: EM on all 6 pairs,
 positive in 5/6 (the 1.7B0.6B exception is negative, ;
8B1.7B is positive but unstable across seeds, ).
K is a composable addressing substrate.
Takeaway 2.
V transfer is gated by teacher--student capability matching:
it succeeds on equal-layer small-gap pairs (8B4B EM 0.88,
1.7B0.6B EM 0.80), collapses greedy decoding on 1.7B students
(EM 0.07--0.12), and fails entirely on large-gap 0.6B students
(, EM below Self).
Takeaway 3.
Three pairs (8B4B, 4B1.7B, 8B1.7B) saturate at Self EM = 100\%;
there, only is diagnostic, and V's likelihood gains must be
interpreted with caution.
### Heterogeneous Reassembly
A striking finding emerges from the 8B0.6B pair:
teacher K + student V outperforms either model running alone.
The K-only configuration achieves ,
which is better than student\_full ()
and better than teacher\_full ().
Takeaway 4.
Teacher K + student V outperforms both complete models, demonstrating that addressing geometry is a composable substrate.
### Robustness
Mapper invariance.
We tested Affine, Whitened, and RidgePerHead mappers on the 8B0.6B pair.
All three produced consistent K-only gains (, , ),
while Procrustes/CCA mappers are destructive (K-only EM drops to 0.00--0.05).
RidgePerHead rescues K-only EM to 0.78 but leaves V-only EM at 0.16.
The K/V asymmetry is a property of the representations, not the mapper choice.
Statistical significance.
Paired Wilcoxon signed-rank test on 56 evaluation samples (3 seeds):
K-only vs Self: in all 3 seeds;
V-only vs Self: (significant in 1/3 seeds, ns overall).
Pooled bootstrap 95\% CI (10000 resamples, 3 seeds):
K-only : ; V-only : .
The K-only gain is statistically robust; the V-only effect is not.
Second domain (SQuAD).
On SQuAD (n=30, 3 seeds, mapper fit on OOD train only):
K-only (all , EM 0.73--0.80),
V-only (all ns, EM 0.03---destroys decoding),
Joint (EM 0.60--0.73).
The K/V asymmetry replicates across domains without domain-specific calibration.
Limitations.
Self EM saturates at 100\% on 3/6 pairs, limiting EM diagnostics there.
V's EM collapse on 1.7B students is robust but its likelihood gains
may be calibration effects (see :analysis).