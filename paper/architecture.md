# Paper Architecture: Addressing Transfers, Content Does Not

## Section Map

| Section | Rhetorical Move | Claims | Evidence |
|---------|----------------|--------|----------|
| Abstract | Summary | 5 contributions | Headline numbers from all experiments |
| Introduction | Stakes → Gap → Abstraction → Design → Contributions → Preview | K/V asymmetry, correlation ≠ transferability | 6-pair scan, CCA results |
| Method | Protocol | Injection decomposition, metrics | Task definition, model pairs table |
| Results | Evidence | K-only works, V-only doesn't, K/V ratio | Table 1 (6-pair scan), Figure 1 |
| Analysis | Mechanism | Layer localization, cost crossover | Figure 2 (layers), Figure 3 (cost) |
| Discussion | Boundary | 1.7B→0.6B failure, 8B→4B saturation | Limitations table |

## Claim → Evidence Mapping

### Claim 1: K/V Functional Asymmetry
- **Claim**: K transfers, V does not
- **Evidence**: Table 1 (6-pair scan), K/V ratio column
- **Validation**: K-only ΔLL > 0 in 5/6 pairs; V-only ΔLL < 0 or small
- **Location**: §3 Results, §1 Introduction Move 5

### Claim 2: Correlation ≠ Transferability
- **Claim**: CCA ρ ≈ 1.0 for both K and V, but only K transfers
- **Evidence**: Table 2 (CCA ρ vs ΔLL), Figure 2
- **Validation**: ρ_K ≈ ρ_V ≈ 1.0, but K/V ratio >> 1
- **Location**: §4 Analysis, §1 Introduction Move 5

### Claim 3: Layer Localization
- **Claim**: V advantage concentrates at layers 8/12
- **Evidence**: Per-layer V-only scan (Phase 1), Figure 2
- **Validation**: Layer 12 ΔLL = +3.72, Layer 8 ΔLL = +1.07
- **Location**: §4 Analysis

### Claim 4: Heterogeneous Reassembly
- **Claim**: Teacher K + student V beats both full models
- **Evidence**: 8B→0.6B K-only (-6.74) vs teacher_full (-12.73) vs student_full (-10.13)
- **Validation**: -6.74 > -10.13 (better than student alone)
- **Location**: §3 Results

### Claim 5: Cost Crossover
- **Claim**: State/weight transfer cost crossover at ~2050 tokens
- **Evidence**: Mapper bytes (58.78M) vs cache bytes/token (112KB)
- **Validation**: 58.78M / 112KB ≈ 525 tokens (wait, need to recalculate)
- **Location**: §4 Analysis

## Open Questions (for further investigation)

1. **teacher_full < student_full**: Teacher (-12.73) worse than student (-10.13)
   - Hypothesis: Teacher content miscalibrated on OOD domain, but K geometry generic
   - Need: Verify with in-domain data

2. **1.7B→0.6B negative**: K-only ΔLL = -0.91
   - Hypothesis: Weak teacher K actively hurts
   - Need: Test with more weak→strong pairs

3. **8B→4B EM saturation**: Self EM = 100%
   - Hypothesis: ΔLL is pure calibration gain
   - Need: Harder eval set

## Figures Needed

| Figure | Content | Source | Status |
|--------|---------|--------|--------|
| Figure 1 | K/V transferability asymmetry (bar chart) | Phase 4 Table 1 | Draft |
| Figure 2 | CCA ρ vs functional transferability (scatter) | Phase 2 Table | Draft |
| Figure 3 | Layer-wise V advantage heatmap | Phase 1 Table | Draft |
| Figure 4 | Cost crossover curve | Phase 3 analysis | Draft |

## Table Needed

| Table | Content | Source | Status |
|-------|---------|--------|--------|
| Table 1 | 6-pair results (Self, K-only, V-only, Joint, EM, ΔLL, K/V ratio) | Phase 4 results | Draft |
| Table 2 | CCA ρ₁ vs functional transferability | Phase 2 results | Draft |
| Table 3 | Per-layer V-only ΔLL | Phase 1 results | Draft |

## Writing Checklist

- [ ] Draft 0 introduction covers all 6 moves
- [ ] Each contribution has matching evaluation subsection
- [ ] All figures/tables have source data
- [ ] No process descriptions in contributions ("We show that X" not "We propose X")
- [ ] Move 6 (Results Preview) left blank in Draft 0, filled in final intro

## Next Steps

1. Complete Draft 0 introduction (all 6 moves)
2. Write Evaluation section (§3 Results)
3. Write Design/Method section (§2 Method)
4. Write Related Work section
5. Rewrite Final Introduction (with real numbers)
6. Write Abstract
7. Pass through humanizer-academic-zh for polish

---

*Created: 2026-08-30*
*Status: Stage 2 complete*
