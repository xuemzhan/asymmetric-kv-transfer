## Method
sec:method
### Problem Setup
Consider a teacher model and a student model .
Given a document , the teacher produces KV states 
and the student produces .
We ask: which components of can replace 
while preserving or improving task performance?
Key insight. Keys encode attention routing (addressing);
values encode content representations (semantics).
These transfer differently across models.
### Task and Data
Synthetic OOD task.
We construct a synthetic out-of-domain (OOD) task with novel entities
that do not appear in model pretraining data.
The task requires reasoning across 1--4 hops over a knowledge graph,
with answers verifiable by exact match.
Data splits.
We split the synthetic domain by entity clusters to prevent information leakage:
entities in the evaluation set do not appear in the training set.
For each of 3 independent seeds, we draw a fresh entity-cluster split:
training 70 samples, validation 28 samples, evaluation 56 samples ().
All reported results are means over the 3 seeds with per-seed statistical tests.
Why synthetic?
Real-world tasks require ground-truth reasoning chains.
Our synthetic task provides verifiable answers while controlling
entity familiarity---critical for studying OOD transfer.
### Model Pairs
We study 6 pairs from the Qwen3 family:
8B0.6B, 4B1.7B, 4B0.6B,
8B1.7B, 1.7B0.6B, and 8B4B.
Layers: 8B/4B have ; 1.7B/0.6B have ; all heads , .
This design covers three configurations:
itemize
 Equal-layer: 3636 (8B4B), 2828 (1.7B0.6B)
 Unequal-layer (proportional mapping): 3628 (8B0.6B, 8B1.7B, 4B0.6B, 4B1.7B)
 Size ratio: 2.0 (8B4B) to 13.3 (8B0.6B)
itemize
Second domain (SQuAD).
To test whether the K/V asymmetry is specific to our synthetic domain,
we additionally evaluate on SQuAD (30 questions, in-document facts,
answers 1--5 tokens, doc avg.\ 119 tokens).
The mapper is fit on the synthetic OOD training set and applied to SQuAD
without retraining---the mapper never sees SQuAD documents.
This tests cross-domain, cross-distribution transfer.
### Injection Protocol
For each evaluation sample, we construct four KV configurations:
Self.
Student prefills document with its own KV states.
This is the baseline (doc-only KV).
K-only.
Teacher K (mapped) + Student V.
Tests whether addressing geometry transfers.
V-only.
Student K + Teacher V (mapped).
Tests whether content representations transfer.
Joint.
Teacher K (mapped) + Teacher V (mapped).
Tests full KV transfer (both components).
### Mapper
We use Affine mappers fitted on calibration data.
The mapper learns a linear transformation and bias :
For K, we apply de-RoPE before mapping and re-RoPE after.
For V, no positional alignment is needed.
We also tested Whitened, Procrustes, CCA, and RidgePerHead mappers;
results are invariant to mapper choice (see Robustness, :results).
### Metrics
Primary: Answer log-likelihood.
We compute teacher-forced log-likelihood of answer tokens.
This is continuous, sensitive to small changes,
and correlates with task performance.
Secondary: Exact match.
We compute greedy exact match on final answers.
This is binary, interpretable, but less sensitive.
Baselines.
itemize
 Self: Doc-only KV (negative baseline).
 Student full: Doc + query KV (upper bound for single model).
 Teacher full: Doc + query KV (reference for teacher capability).
itemize
### Statistical Analysis
We report 95\% bootstrap confidence intervals (1000 resamples per seed)
and paired Wilcoxon signed-rank tests ( evaluation samples per seed).
We consider statistically significant.
All significance claims require in all 3 independent seeds
(data-level replication with entity-cluster held-out splits).
Warning. Three pairs saturate at Self EM = 100\%, limiting EM
diagnostics there; remains the primary diagnostic on those pairs.
All bootstrap CIs are computed per seed and pooled across seeds
for the final estimate (3 seeds 56 samples).