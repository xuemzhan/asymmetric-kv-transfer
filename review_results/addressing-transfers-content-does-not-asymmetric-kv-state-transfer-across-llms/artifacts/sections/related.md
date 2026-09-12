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