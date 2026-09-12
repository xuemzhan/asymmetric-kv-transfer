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