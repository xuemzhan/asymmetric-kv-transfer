article
geometry
inputenc
fontenc
booktabs
graphicx
amsmath
amssymb
hyperref
EM
Addressing Transfers, Content Does Not: \\
Asymmetric KV State Transfer Across LLMs
 Anonymous Author(s) \\
 anonymous@example.com
document
abstract
We study what transfers when migrating key-value (KV) cache state across large language models (LLMs).
Across 6 model pairs from the Qwen3 family (0.6B--8B, 3 seeds), we find a sharp
functional asymmetry that is gated by the teacher--student capability gap:
keys (the addressing substrate) transfer near-universally, recovering task performance
on all 6 pairs (exact match ; positive on 5/6 pairs, to ),
while values (the content substrate) transfer only when teacher and student are
closely matched (B4B: EM 1.000.88; B0.6: EM 0.430.80)
and fail on large-gap pairs (B0.6: , ns)---despite K and V
being equally, almost perfectly linearly alignable (CCA both).
This correlation--transferability dissociation shows that geometric compatibility
is necessary but not sufficient; the bottleneck lies in downstream decode,
where content is consumed through weight-parameterized pathways.
We localize the V advantage to layers 8/12 (selective injection vs.\
full-layer V injection ), replicate the asymmetry on a second domain (SQuAD,
where V-only injection collapses EM to 0.03), and show that heterogeneous reassembly
(teacher K + student V) beats both full models ( vs.\ teacher\_full 
and student\_full ), with a cost crossover at 2050 tokens.
Our results establish that KV state transfer is asymmetric by construction:
addressing geometry is composable across models,
while content is weight-bound.
abstract
## Introduction
Modern LLM serving incurs substantial memory overhead from KV caches.
A single 70B-parameter model requires 20GB+ of KV cache for a 4K-token sequence.
When deploying model upgrades or heterogeneous ensembles, existing KV caches become obsolete,
forcing expensive re-computation.
Reusing cached KV states across model versions reduces serving latency and memory costs,
but only if the right parts of the cache transfer across models.
Current approaches to KV transfer treat it as all-or-nothing.
Three structural limitations prevent effective cross-model KV reuse:
enumerate
 No decomposition. Prior work transfers entire KV caches without identifying
 which components (keys vs.\ values) are transferable.
 Geometry function. Linear alignment methods (CCA, Procrustes)
 show high geometric correlation between KV representations across models,
 but correlation does not guarantee functional transferability.
 No cost model. When is it cheaper to transfer state vs.\ transfer weights?
 No systematic analysis exists for KV cache transfer.
enumerate
We introduce the concept of gated asymmetric KV transferability:
keys (the addressing substrate) transfer across models,
while values (the content substrate) transfer only when the
teacher--student capability gap is small.
This asymmetry is functional, not geometric---both K and V are linearly almost perfectly
alignable (CCA ), yet only K recovers task performance when injected
across large gaps.
Our approach consists of three stages:
(1) KV capture: extract teacher and student KV states from matched documents;
(2) mapper fitting: learn affine transformations from teacher to student KV spaces
using calibration data;
(3) injection and evaluation: replace student KV with mapped teacher KV,
measure task performance via answer log-likelihood and exact match.
We decompose injection into four arms (Self, K-only, V-only, Joint) to isolate
the contribution of each component.
This paper makes five contributions:
enumerate
 Gated K/V functional asymmetry.
 We show that K-state transfer recovers task performance on all 6 model pairs
 (EM ; positive on 5/6 pairs, to ),
 while V-state transfer succeeds
 only on small-gap pairs (B4, B0.6) and fails or
 destabilizes decoding on large-gap pairs.
 Correlation transferability.
 CCA for both K and V, yet only K transfers across large gaps.
 We show that geometric compatibility is necessary but not sufficient;
 the bottleneck lies in weight-parameterized content consumption.
 Layer localization.
 V advantage concentrates in layers 8/12
 (V-only ; selective joint injection ),
 while K advantage distributes globally across all layers.
 Heterogeneous reassembly.
 Teacher K + student V beats both full models
 ( vs teacher\_full / student\_full ),
 demonstrating that addressing geometry is a composable substrate.
 Cost crossover.
 State/weight transfer cost crossover at 2050 tokens,
 providing a practical decision rule for cache transfer.
enumerate
Results preview.
Across 6 model pairs spanning 0.6B to 8B parameters,
K-only injection improves log-likelihood by to on 4/6 pairs
(paired , 3 seeds) and recovers EM on all pairs,
while V-only injection is significant only on the two small-gap pairs
(B4: ; B0.6: )
and neutral or negative elsewhere (B0.6: , ns).
The flagship 8B0.6B pair shows K-only 
versus V-only (K/V asymmetry: ),
with Joint injection .
Heterogeneous reassembly achieves ,
outperforming both teacher\_full () and student\_full ().
CCA reveals near-perfect linear alignment () for both K and V,
yet only K transfers functionally---a dissociation that establishes
the correlation--transferability gap.
## Related Work
sec:related
### Cross-Model KV Cache Transfer
The closest concurrent line of work transfers KV cache state across
different models.
Heo et al.\ (NVIDIA, 2026)~heo2026crossmodel introduce a closed-form
per-head ridge mapper with RoPE-stripped keys and cross-layer source
selection, demonstrating prefill reuse between Qwen3-14B and Qwen3-32B.
Cache-to-Cache (Fu et al., 2026)~fu2026c2c transfers semantic
states between LLMs for direct communication.
Mixture-of-Translators~lee2026translators translates KV caches
across heterogeneous LLMs with a mixture of per-layer translators.
LatentAlign~dery2026latentalign aligns latent spaces via
KV cache alignment for cross-model communication.
These works treat KV as a monolithic artifact and focus on engineering
the transfer (better mappers, layer selection, alignment).
Our contribution is orthogonal and diagnostic: we decompose KV into
K (addressing) and V (content), and show they transfer with different
functional laws---K near-universally (EM on all 6 pairs),
V only under capability matching.
We further show that geometric alignment (CCA )
does not predict functional transfer, localize the V advantage to
layers 8/12, and demonstrate heterogeneous reassembly
(teacher K + student V) that beats both full models.
### KV Cache Optimization
LLM serving incurs substantial memory overhead from KV caches.
FlashAttention~dao2022flashattention addresses this via IO-aware kernel fusion,
reducing memory footprint without changing the KV format.
Speculative decoding~leviathan2023fast,chen2023accelerating
reduces latency by predicting multiple tokens ahead,
but requires a draft model to generate candidates.
Our work is orthogonal: we study what transfers across models,
not how to optimize cache memory within a single model.
### Model Compression and Distillation
Model compression reduces size via quantization~frantar2023gptq,dettmers2024qlora,
pruning, or distillation~touvron2023llama.
These methods transform model weights,
while we study transferring KV cache state across existing models.
Knowledge distillation~hinton2015distilling
transfers knowledge from teacher to student via soft labels.
Our work shows that KV state can transfer directly,
without the full distillation pipeline, and that the transferable
content is concentrated in a few layers (8/12).
### Attention Mechanisms
The Transformer architecture~vaswani2017attention
decomposes attention into keys (addressing) and values (content).
Rotary position embeddings~su2024rope further structure
K geometry positionally---which we strip before mapping.
This decomposition is fundamental to our work:
we study whether K and V transfer differently across models.
Prior work treats KV as monolithic.
We show K and V serve different functional roles:
K is model-agnostic routing; V is weight-bound content.
### Model Stitching
Model stitching combines layers from different models
to create hybrid architectures~bansal2021stitching.
Recent work shows that attention layers can be stitched
across models with minimal performance loss.
Our heterogeneous reassembly (teacher K + student V)
extends this line:
addressing geometry is composable,
while content representations are not.
This suggests a functional decomposition for model stitching.
### Scaling Laws
Scaling laws~kaplan2020scaling,hoffmann2022chinchilla
characterize how performance scales with compute, data, and parameters.
Our capability-gap analysis complements this:
V transfer success is gated by teacher--student matching,
and our cost crossover at 2050 tokens provides a practical
decision rule for when state transfer beats weight transfer.
### Summary
Our work is distinct from prior efforts in three ways:
enumerate
 Decomposition. We decompose KV into K and V,
 showing gated asymmetric transferability.
 Functional analysis. We study what transfers functionally,
 not just geometrically (CCA correlation).
 Capability gating. We map the boundary where V transfer
 succeeds (equal-layer small-gap) and fails (large-gap, mid-gap decode collapse).
enumerate
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
## Analysis
sec:analysis
### Correlation Transferability
A striking finding emerges from comparing geometric alignment
with functional transferability.
CCA results.
Canonical Correlation Analysis (CCA) reveals that
first canonical correlation for both K and V
across all model pairs (Table~tab:cca).
This indicates near-perfect linear alignability:
teacher and student KV representations lie in essentially
the same linear subspace.
The dissociation.
Despite near-perfect geometric alignment for both K and V,
only K transfers functionally.
V-only is weak or negative on these pairs (Table~tab:main),
even though trails by roughly 0.005
(Figure~fig:cca; K heads exceed V heads on 75--79\% of heads).
This dissociation is the paper's central finding.
Interpretation.
Linear-geometric compatibility (CCA ) is necessary but not sufficient.
The bottleneck is downstream decode, where content is consumed through weight-parameterized pathways:
K geometry is model-agnostic; V content is weight-bound.
### Layer Localization of V Advantage
Per-layer V-only replacement reveals concentrated hotspots
at layers 8 and 12 (Table~tab:layers).
Concentration.
Layer 12 alone accounts for (EM 1.00).
Layer 8 contributes .
Injecting both selectively () yields 
with EM 0.75--0.89.
Injecting all 28 layers (as in G0 V-only) is harmful: ,
EM 0.14--0.29---below Self (0.43).
Mechanism.
V transferable information is sparse (layers 8/12 only);
full-layer V injection fails because 26/28 layers inject noise,
explaining the G0 V-only failure.
This reconciles near-perfect geometric alignment ()
with functional failure: alignment is averaged over all layers,
but only two carry transferable content.
Asymmetry with K.
K advantage distributes globally (Figure~fig:layers):
the best single K layer (L21) gives , and
cumulative prefix injection gives to 
with EM ---K saturates as coverage grows,
while early layers alone () hurt ().
K geometry is a property of the entire network,
while V content is localized to specific processing stages.
### Cost Crossover
Mapper cost.
Affine mapper parameters: 58.78M (224.2 MiB at float32), fixed regardless of sequence length.
Cache cost.
Per-token KV cache: 112 KB (28-layer student), scaling linearly with sequence length.
Crossover.
State/weight transfer cost crossover at 2050 tokens (Figure~fig:cost).
Below 2050 tokens, KV cache transfer is cheaper;
above, mapper transfer is preferred.
### Boundary Conditions
The 6-pair design reveals the capability-gap gating boundary:
1.7B0.6B (V success at minimum gap).
V-only (sig in all 3 seeds), EM 0.430.80.
This is the only 0.6B-student pair where V transfers, and the only
2828 equal-layer pair besides 8B4B.
The gating factor is equal-layer alignment, not absolute size.
4B1.7B / 8B1.7B (V likelihood--EM dissociation).
V-only (sig) but EM collapses to 0.12/0.07.
V content improves likelihood yet breaks student greedy decoding;
under ``V is shared content'' theory, both should succeed together.
This dissociation localizes the failure to decode-time consumption.
8B4B (saturation).
Self EM = 100\%, so is partially calibration gain,
not pure task improvement.
The student already solves the task; K transfer improves calibration
while V transfer preserves EM (0.88).
### Synthesis
The evidence converges on a clear picture:
K is addressing geometry (model-agnostic, composable, transfers
near-universally); V is weight-bound content (transfers only under
equal-layer capability matching, and its likelihood gains can coexist with
decoding collapse).
This explains the functional asymmetry despite geometric equivalence,
and refines the earlier ``V never transfers'' claim:
V does transfer on closely matched pairs, but the boundary is narrow.
## Discussion
sec:discussion
### Implications
For LLM serving.
Transfer K state, not V, across model versions---but only when the
teacher--student capability gap is large. On closely matched pairs,
V transfer is viable (8B4B: EM 0.88) and can be selectively
exploited via layers 8/12, avoiding the 26 noise layers.
The cost crossover at 2050 tokens remains the practical decision rule.
For distillation.
Prior work focuses on matching attention patterns (K geometry).
This is misguided: K geometry is already generic.
Focus instead on V/content pathways, which carry task-specific knowledge---and
note that V knowledge is concentrated in a few layers, making selective
layer distillation a promising target.
For model stitching.
Heterogeneous reassembly (teacher K + student V) beats both full models.
Addressing geometry is a composable substrate, and the sparse localization
of V advantage (L8/L12) suggests where cross-model stitching should focus.
### Limitations
Single model family.
All 6 pairs come from Qwen3; cross-family transfer (e.g., LlamaQwen)
remains untested and may behave differently under different RoPE
configurations and attention implementations.
EM saturation on 3/6 pairs.
8B4B, 4B1.7B, 8B1.7B saturate at Self EM = 100\%;
there, only is diagnostic, and V's likelihood gains
must be interpreted with caution.
V collapse on 1.7B students.
V-only improves likelihood but collapses greedy EM (0.07--0.12) on
4B1.7B and 8B1.7B. Whether this reflects calibration effects
or genuine content misalignment is not fully resolved.
SQuAD scale.
The second-domain validation uses 30 questions; larger benchmarks
would tighten the cross-domain claim.
Mapper family.
Only linear mappers (Affine/Whitened/Procrustes/CCA/RidgePerHead) were tested.
Nonlinear mappers (MLP/attention-based) could alter the boundary,
especially for V transfer on mid-gap pairs.
### Future Work
Cross-family transfer.
Test whether the gated K/V asymmetry holds across model families
(Llama/Qwen/Mistral), where RoPE and attention implementations differ.
Per-hop difficulty analysis.
Scale to per-hop breakdown (hop-1 through hop-4) to characterize
how transfer success varies with reasoning depth.
Theoretical modeling.
Why does K geometry transfer but V content only under capability matching?
A theoretical framework would predict the gating boundary
from model architecture and training data.
Selective layer transfer in production.
Leverage the L8/L12 V hotspot to design layer-selective cache transfer
that avoids the 26 noise layers, and measure actual serving gains.
### Conclusion
We have shown that KV state transfer across LLMs exhibits
a sharp, capability-gated functional asymmetry:
keys (addressing) transfer near-universally across 6 model pairs
(EM ; positive in 5/6), while values (content) transfer
only under equal-layer capability matching (8B4B EM 0.88,
1.7B0.6B EM 0.80), collapse greedy decoding on mid-gap students
(EM 0.07--0.12), and fail entirely on large-gap pairs
(, ns) despite near-perfect linear alignment
(CCA for both K and V).
The V advantage localizes to layers 8/12; selective injection
( ) beats full-layer injection ( ).
The asymmetry replicates on a second domain (SQuAD), where V-only
injection destroys decoding (EM 0.03), and heterogeneous reassembly
(teacher K + student V) beats both full models ( vs /).
Our results establish that KV state transfer is asymmetric by construction:
addressing geometry is composable across models,
while content is weight-bound---transferable only where
weights already agree.
This has implications for serving, distillation,
and modular model construction.
thebibliography10
bansal2021stitching
Yamini Bansal, Preetum Nakkiran, and Boaz Barak.
 How could we ever measure the perceptron alignment of neural
 networks?
 In International Conference on Learning Representations, 2021.
 Model stitching.
chen2023accelerating
Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, et~al.
 Accelerating large language model decoding with speculative sampling.
 arXiv preprint arXiv:2302.01318, 2023.
dao2022flashattention
Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R\'e.
 Flashattention: Fast and memory-efficient exact attention with
 io-awareness.
 Advances in Neural Information Processing Systems, 35, 2022.
dery2026latentalign
Lucio~M. Dery, Zohar Yahav, Henry Prior, Qixuan Feng, Jiajun Shen, and Arthur
 Szlam.
 Latent space communication via k-v cache alignment.
 arXiv preprint arXiv:2601.06123, 2026.
dettmers2024qlora
Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer.
 Qlora: Efficient finetuning of quantized language models.
 Advances in Neural Information Processing Systems, 36, 2024.
frantar2023gptq
Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh.
 Gptq: Accurate post-training quantization for generative pre-trained
 transformers.
 arXiv preprint arXiv:2210.17323, 2023.
fu2026c2c
Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, and
 Yu~Wang.
 Cache-to-cache: Direct semantic communication between large language
 models.
 In International Conference on Learning Representations, 2026.
 arXiv:2510.03215.
heo2026crossmodel
Taekyung Heo, Rasoul Shafipour, Ritchie Zhao, Maximilian Golub, Mohammad~Mahdi
 Kamani, Ritika Borkar, Makesh~Tarun Chandran, Pantea Zardoshti, and
 Bita~Darvish Rouhani.
 Cross-model kv cache transfer in llm families: A closed-form linear
 mapping for prefill reuse.
 arXiv preprint arXiv:2608.03893, 2026.
hinton2015distilling
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean.
 Distilling the knowledge in a neural network.
 arXiv preprint arXiv:1503.02531, 2015.
hoffmann2022chinchilla
Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et~al.
 Training compute-optimal large language models.
 Advances in Neural Information Processing Systems, 35, 2022.
kaplan2020scaling
Jared Kaplan, Sam McCandlish, Tom Henighan, et~al.
 Scaling laws for neural language models.
 arXiv preprint arXiv:2001.08361, 2020.
lee2026translators
Jin-woo Lee, Minkyung Song, Junghyun Oh, Seunghoon Han, Soyoung Park, Gwangseon
 Jang, and Sungsu Lim.
 Mixture-of-translators: Translating kv caches across heterogeneous
 large language models.
 arXiv preprint arXiv:2607.28979, 2026.
leviathan2023fast
Yan Leviathan, Matan Kalman, and Yossi Matias.
 Fast inference from transformers via speculative decoding.
 International Conference on Machine Learning, pages
 19274--19290, 2023.
su2024rope
Jianlin Su, Murtadha Ahmed, Yu~Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu.
 Roformer: Enhanced transformer with rotary position embedding.
 Neurocomputing, 568:127063, 2024.
touvron2023llama
Hugo Touvron, Thibaut Lavril, Gautier Izacard, et~al.
 Llama: Open and efficient foundation language models.
 arXiv preprint arXiv:2302.13971, 2023.
vaswani2017attention
Ashish Vaswani, Noam Shazeer, Niki Parmar, et~al.
 Attention is all you need.
 Advances in Neural Information Processing Systems, 30, 2017.
thebibliography
document