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