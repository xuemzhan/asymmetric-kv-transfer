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