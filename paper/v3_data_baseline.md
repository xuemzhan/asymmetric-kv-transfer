# V3 Paper — Authoritative Data Baseline (v2, 3-seed)
**Date:** 2026-08-30 | **Source:** reports/*.json (all regenerable)
**Use ONLY these numbers in the paper. Do NOT invent or reuse v1 numbers (n=14).**

---

## 1. Flagship G0 (8B→0.6B, n=56, 3 seeds) — phase0_g0_v2

| Arm | Mean LL (3 seeds) | ΔLL vs Self | Wilcoxon p (3 seeds) | EM (3 seeds) |
|-----|-------------------|-------------|----------------------|--------------|
| Self | −9.89 | — | — | 0.43 |
| Affine K-only | −7.27 | **+2.62** | <1.2e-10 ×3 (all significant) | 0.71–0.82 |
| Affine V-only | −10.11 | **−0.23** | 0.024/0.125/0.144 (2/3 ns) | 0.14–0.29 (below Self) |
| Affine Joint | −5.37 | **+4.52** | ≈7.5e-11 ×3 | 0.73–0.86 |
| Teacher full (8B) | −14.79 | — | — | — |
| Student full (0.6B) | −9.89 (= Self, doc-only) | — | — | — |

- Mapper ladder (all mappers, 8B→0.6B): **none rescues V** — Procrustes/CCA destructive; RidgePerHead K EM=0.78 but V EM=0.16.
- Heterogeneous reassembly (v2): K-only (−7.27) **beats** student_full (−9.89) and teacher_full (−14.79). Δ = +2.62 vs student, +7.52 vs teacher.

## 2. W8 Scaling Law — 6 pairs × 3 seeds (n=56) — phase4_scaling_law_v2

| Pair | Param ratio | K ΔLL (3 seeds) | V ΔLL (3 seeds) | Joint ΔLL (mean) | K sig ×3 | V sig ×3 | EM Self→K→V |
|------|------------|-----------------|-----------------|------------------|----------|----------|-------------|
| 8B→0.6B | 13.3× | +2.41/+2.50/+2.95 | −0.05/−0.22/−0.42 | +4.52 | ✅ | ❌ | 0.43→0.73→0.29 |
| 4B→0.6B | 6.7× | +1.58/+1.56/+1.59 | −1.27/−1.23/−1.16 | — | ✅ | ❌ | 0.43→0.84→0.18 |
| 1.7B→0.6B | 2.8× | −0.82/−0.83/−0.80 | +0.83/+0.56/+0.51 | — | ❌ | ✅ | 0.43→1.00→0.80 |
| 8B→4B | 2.0× | +8.56/+8.77/+8.22 | +2.54/+2.44/+2.58 | — | ✅ | ✅ | 1.00→0.91→0.88 |
| 4B→1.7B | 2.35× | +2.97/+2.79/+2.93 | +3.14/+2.87/+2.74 | — | ✅ | ✅ | 1.00→0.95→0.12 |
| 8B→1.7B | 4.7× | +0.30/+0.03/+2.80 (unstable) | +5.58/+5.66/+5.64 | — | ❌ (s0/s1 ns) | ✅ | 1.00→0.82→0.07 |

**Cross-seed means ± std:**
- 8B→0.6B: K +2.62±0.29 | V −0.23±0.19
- 4B→0.6B: K +1.58±0.02 | V −1.22±0.05
- 1.7B→0.6B: K −0.82±0.02 | V +0.63±0.17
- 8B→4B: K +8.52±0.28 | V +2.52±0.07
- 4B→1.7B: K +2.90±0.09 | V +2.91±0.21
- 8B→1.7B: K +1.04±1.53 | V +5.63±0.04

**KEY NARRATIVE (capability-gap gating):**
- K (addressing): near-universally transferable — EM ≥0.73 on all 6 pairs; LL positive 5/6 (8B_1.7B unstable).
- V (content): transfer **gated by teacher–student matching**:
  - Large-gap pairs (0.6B student + 8B/4B teacher): V FAILS (LL ≤0, EM < Self).
  - 1.7B→0.6B (equal-layer 28→28): V transfers (EM 0.43→0.80) — the ONLY 0.6B-student pair where V works.
  - Small-gap pairs (8B→4B, 4B→1.7B, 8B→1.7B): V LL gains (+2.5 to +5.6) BUT EM collapses to 0.07–0.12 (except 8B→4B: EM 0.88) → V content boosts likelihood but breaks student greedy decoding.
  - 3 pairs have Self EM=1.0 (8B→4B, 4B→1.7B, 8B→1.7B) → EM saturated, LL-only diagnostic there.

## 3. W6 Causal Analysis — 3 seeds (n=56) — phase1_causal_v2

| Metric | Seed values |
|--------|-------------|
| Best V-only layer | L12 ×3 seeds (ΔLL +1.27/+1.19/+1.25, p<5e-09, EM 1.00) |
| L8 V-only | ΔLL +0.32/+0.29/+0.33 (p≈0.012) |
| Selective V L8+L12 | ΔLL **+2.33/+2.11/+2.00** (all p<1e-08), EM 0.75/0.75/0.89 |
| V_ALL (all 28 layers) | ΔLL **−0.05/−0.22/−0.42** (harmful), EM 0.29/0.29/0.14 (below Self 0.43) |
| Best K-only layer | L21 ×3 seeds (ΔLL +1.61/+1.58/+1.59, p<1.5e-10) |
| K prefix-cumulative [0..20] | ΔLL +2.34/+2.49/+2.71, EM ≈1.0 (saturates) |
| K prefix [0..4] | ΔLL −0.57/−0.47/−0.48 (early layers hurt) |

**Interpretation:** V transferable info is SPARSE (L8/L12 only); full-layer V injection fails because 26/28 layers inject noise — explains G0 V-only failure. K info is GLOBAL (prefix saturation).

## 4. W7 Second-Domain (SQuAD, cross-domain calibration) — 3 seeds (n=30) — phase7_second_domain

| Arm | ΔLL (3 seeds) | Wilcoxon p | EM (3 seeds) |
|-----|---------------|------------|--------------|
| Self | — | — | 0.77 |
| K-only | +5.22/+4.98/+5.52 | 2.8e-06/4.7e-07/2.0e-07 (all sig) | 0.73/0.80/0.80 |
| V-only | +1.05/+1.05/+0.80 | 0.21/0.21/0.34 (all ns) | **0.03 all seeds** (destroys decode) |
| Joint | +3.50/+3.52/+3.81 | <3.4e-04 | 0.70/0.73/0.60 |

- **Design: mapper fit on OOD train_v2, evaluated on SQuAD — mapper never saw SQuAD domain.**
- Novelty probe (nq_open, 30): teacher EM=0.167, student EM=0.067 (both near random → task data not in weights).

## 5. W4 CCA Per-Head (seed 0) — phase2_g_scalar_perhead

| Pair | K ρ₁ mean (std) | V ρ₁ mean (std) | Heads K>V |
|------|-----------------|-----------------|-----------|
| 8B→0.6B | 0.9944 (0.0027) | 0.9899 (0.0059) | 0.759 |
| 4B→1.7B | 0.9943 (0.0027) | 0.9890 (0.0066) | 0.790 |
| 4B→0.6B | 0.9944 | 0.9906 | 0.746 |

**K and V are both ~perfectly linearly alignable (ρ₁ ≈ 0.99+), yet only K transfers → correlation ≠ transferability.**

---

## 6. Method/Settings (v2 — update all old v1 numbers)

- Data: OOD synthetic, train 70 / val 28 / test 56 (per seed, entity-cluster splits). **OLD paper says 42/14 — WRONG, update.**
- Seeds: **3 independent seeds** (data-level replication, held-out by entity cluster). OLD paper says "3 seeds identical" — WRONG, update.
- Eval: n=56 (OOD), n=30 (SQuAD).
- SQuAD: 30 samples, doc avg 119 tokens, answer 1–5 tokens, in-document facts.
- Models: Qwen3 8B(L36)/4B(L36)/1.7B(L28)/0.6B(L28), H8 D128, proportional layer mapping for unequal layers.
- Metrics: teacher-forced answer log-likelihood (primary), greedy EM (secondary). Bootstrap CI95 (10000 resamples) + paired Wilcoxon (p<0.05).
- Cost: Affine mapper 58.78M params (224.2 MB fp32); per-token KV 112 KB (28-layer student); crossover ≈2050 tokens.

## 7. Claims Allowed (narrative guardrails)

✅ K transfers near-universally (EM ≥0.73 all pairs; LL positive 5/6)
✅ V transfer gated by capability gap: fails on large-gap 0.6B pairs; EM-collapses on small-gap 1.7B students; works only on equal-layer small-gap (8B→4B EM 0.88, 1.7B→0.6B EM 0.80)
✅ CCA ρ₁≈0.99+ both K,V; only K transfers → correlation ≠ transferability
✅ V info sparse (L8/L12 hotspots); V_ALL harmful (26/28 noise layers)
✅ Second domain (SQuAD, cross-domain mapper): K transfers, V destroys EM (0.03)
❌ DO NOT claim "V never transfers anywhere" (refuted by 1.7B→0.6B, 8B→4B)
❌ DO NOT claim "3 seeds identical/deterministic" (v2 uses real data-level replication)
❌ DO NOT use n=14 numbers (all v1 superseded)
