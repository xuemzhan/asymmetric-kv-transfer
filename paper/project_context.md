# Project Context: V3 — Addressing Transfers, Content Does Not

## Identity Sentence

We show that KV state transfer across LLMs exhibits a sharp functional asymmetry: the addressing substrate (keys) transfers across models and can even improve upon either model alone, while the content substrate (values) does not transfer functionally despite being linearly almost perfectly alignable (CCA ρ≈1).

## Venue

COLM or TMLR (mechanistic/empirical analysis paper)

## Contribution Claims

1. **K/V Functional Asymmetry**: K-state transfer recovers task performance across 6 model pairs (ΔLL = +3.26 to +7.67), while V-state transfer is at best neutral on likelihood and task-harming on EM (G0: V-only EM 28.6% vs Self 42.9%). K/V ratio ranges from 1.38 to 7.58.

2. **Correlation ≠ Transferability**: CCA ρ₁ ≈ 1.0 for both K and V across model pairs, yet only K transfers functionally. Linear-geometric compatibility is necessary but not sufficient; the bottleneck is downstream decode (weight-parameterized content consumption).

3. **Layer Localization**: V advantage concentrates in layers 8/12 (V-only ΔLL = +1.07/+3.72), while K advantage distributes globally (per-layer effects small, global transfer large).

4. **Heterogeneous Reassembly**: Teacher K + student V can beat both full models (K-only -6.74 vs teacher_full -12.73 / student_full -10.13 on 8B→0.6B), demonstrating that addressing geometry is a composable substrate.

5. **Cost Crossover**: State/weight transfer cost crossover at ~2050 tokens; below this, cache transfer is cheaper; above, mapper transfer is cheaper.

## Locked Decisions

- **Models**: Qwen3 family (8B/4B/1.7B/0.6B), 6 pairs
- **Task**: Synthetic OOD domain (entity-cluster splits, hop-1..4 reasoning)
- **Metrics**: Answer token teacher-forced log-likelihood + greedy exact match
- **Mapper**: Affine (best performer); results validated with Whitened/RidgePerHead
- **Baseline**: Self (doc-only KV) + student_full (doc+query)
- **Statistical**: Paired Wilcoxon + bootstrap CI95

## Open Questions

1. **teacher_full < student_full**: On flagship pair 8B→0.6B, teacher_full (-12.73) < student_full (-10.13). Teacher has no measured advantage, yet K transfer works. Explanation: teacher content is miscalibrated on OOD domain, but K geometry is generic.

2. **1.7B→0.6B negative**: K-only ΔLL = -0.91 on this pair. Under "K is shared geometry" theory, this should be neutral. Negative value suggests weak teacher K can actively hurt.

3. **8B→4B EM saturation**: Self EM = 100%, so ΔLL = +7.67 is pure likelihood calibration, not task improvement.

## Key Figures/Tables

- **Figure 1**: K/V transferability asymmetry across 6 pairs (bar chart: K-only ΔLL vs V-only ΔLL)
- **Figure 2**: CCA ρ vs functional transferability (scatter: ρ_K/ρ_V vs ΔLL_K/ΔLL_V)
- **Figure 3**: Layer-wise V advantage heatmap (layers × pairs)
- **Figure 4**: Cost crossover curve (cache_cost/mapper_cost vs seq_len)
- **Table 1**: 6-pair results (Self, K-only, V-only, Joint, EM, ΔLL, K/V ratio)

## Narrative Spine

**Opening**: LLMs share attention geometry but differ in content representations. Can KV cache state transfer across models?

**Problem**: Prior work assumes KV transfer is either all-or-nothing. No systematic analysis of WHAT transfers.

**Key Insight**: K (addressing) is a generic substrate; V (content) is weight-bound.

**Evidence**: 6-pair scan shows K transfer works (ΔLL +3.26 to +7.67), V doesn't (EM negative). CCA shows correlation ≠ transferability. Layer analysis localizes V hotspot.

**Implication**: For serving/caching, transfer K, not V. For distillation, focus on V/content.

**Boundary**: 1.7B→0.6B fails (weak teacher K hurts), 8B→4B saturates (EM 100%).

---

*Created: 2026-08-30*
*Status: Draft 0 stage*
