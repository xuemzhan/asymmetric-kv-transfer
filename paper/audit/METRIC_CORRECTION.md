# Metric Correction: Legacy Greedy EM is an Artifact

**Date:** 2026-09-12
**Trigger:** reviewer-requested control experiments (REVISION_PLAN §2, audit1 §12)
**Status:** all downstream EM claims in the v2/v3 paper are invalidated; LL results
are demoted to a non-specific calibration signal.

---

## 1. Root cause (two bugs in the legacy evaluation path)

The published evaluation helper is `phase0_g0.answer_loglik` + `greedy_answer`.
Two defects make the reported Exact Match (EM) numbers an artifact.

### Bug A --- shared, mutated `DynamicCache`

In `phase0_g0.run_g0` and in every phase4/phase7 arm:

```python
c = build_cache(eval_s_kv[i])
row["Self"]    = answer_loglik(student, tok_s, c, q, s["answer"])
row["Self_em"] = exact_match(greedy_answer(student, tok_s, c, q), s["answer"])
```

`answer_loglik` advances the **same** `c` through the query **and the gold answer
tokens** (it feeds `a_ids[:, j:j+1]` in a loop). `greedy_answer` is then called
with that already-advanced cache. Generation therefore starts from
`[doc, query, <gold answer>, query, ...]` --- the model has just seen the gold
answer, so it mostly repeats it. EM measured "can the model reproduce an answer
it was already shown", not "can it answer from the injected state".

### Bug B --- double-feed of the last query token

```python
out = model(q_ids, past_key_values=cache, use_cache=True)   # query already in past
...
model(torch.tensor([[gen[-1]]]) if gen else q_ids[:, -1:], ...)  # re-feeds last token
```

For the first generated token the code ignores the prefill logits and re-feeds
`q_ids[:, -1]`, so the model decodes after a duplicated prompt token
(`...Answer::`). This produces punctuation loops once the cache is perturbed.

`phaseB_common.greedy_answer_fixed` fixes both: fresh cache for generation and
the first token taken from the prefill logits. `phaseB_common.score_arm` now uses
it. The legacy functions are left untouched so old reports remain reproducible,
but their EM outputs must not be cited.

---

## 2. Corrected flagship four-arm result (8B→0.6B, seed 0, n=56)

| Arm | LL | ΔLL | Legacy EM | **Corrected EM** |
|---|---|---|---|---|
| Self | −9.89 | — | 0.43 | **0.911** |
| K-only | −7.48 | +2.41 | 0.71–0.82 | **0.000** |
| V-only | −9.94 | −0.05 | 0.14–0.29 | **0.000** |
| Joint | −5.48 | +4.41 | 0.73–0.86 | **0.000** |

Corrected, the student answers 91% of questions from its **own** document KV and
**0%** when the teacher K/V is injected at any combination.

## 2b. Corrected six-pair four-arm scan (3 seeds, n=56)

`reports: reports/phaseB_fourarm_seed{0,1,2}.json`

| Pair | Self EM | K EM | V EM | Joint EM | K ΔLL | V ΔLL | Joint ΔLL |
|---|---|---|---|---|---|---|---|
| 8B→0.6B | 0.899±0.010 | 0.000 | 0.000 | 0.000 | +2.62±0.29 | −0.23±0.19 | +4.52±0.15 |
| 4B→1.7B | 0.440±0.010 | 0.054±0.018 | 0.000 | 0.024±0.010 | +2.90±0.09 | +2.91±0.21 | +2.31±0.25 |
| 4B→0.6B | 0.899±0.010 | 0.000 | 0.000 | 0.006±0.010 | +1.58±0.02 | −1.22±0.06 | +2.50±0.52 |
| 8B→1.7B | 0.440±0.010 | 0.000 | 0.000 | 0.012±0.021 | +1.04±1.53 | +5.63±0.04 | +4.69±0.27 |
| 1.7B→0.6B | 0.899±0.010 | 0.030±0.010 | 0.113±0.052 | 0.000 | −0.82±0.02 | +0.63±0.17 | −2.25±0.14 |
| 8B→4B | 0.815±0.021 | 0.149±0.010 | 0.440±0.010 | 0.065±0.055 | +8.52±0.28 | +2.52±0.07 | +8.37±0.28 |

The **LL** deltas reproduce the published v2 numbers exactly (e.g. 8B→0.6B K
+2.62, V −0.23), confirming the pipeline and that only the EM path was broken.
The **EM** column is the corrected one: K-only reaches at most 0.149 (8B→4B) and
V-only at most 0.440 (8B→4B) against Self 0.44–0.90. The published EM of
0.71–0.99 (K) and 0.80/0.88 (V) does not reproduce anywhere.

## 2c. Corrected SQuAD (second domain, cross-domain mapper, seed 0)

`report: reports/phaseB_squad_seed0.json` (mapper fit on OOD train, evaluated on
SQuAD, n=30)

| Arm | Self | K-only | V-only | Joint |
|---|---|---|---|---|
| corrected EM | 0.367 | **0.000** | **0.000** | **0.000** |
| ΔLL | — | +5.223 | +1.053 | +3.497 |
| Wilcoxon p | — | 2.8e-06 | 0.205 | 1.9e-04 |

The published SQuAD claim (K-only EM 0.73–0.80, V-only 0.03) does not reproduce:
corrected K-only EM is 0.000. The large K-only LL gain (+5.22) is the same
non-specific effect seen in the synthetic domain.

## 3. Control suite (8B→0.6B, seed 0, corrected evaluator)

`report: reports/phaseB_controls_8B_0.6B_seed0_fixed.json`

| Arm | ΔLL vs Self | Clustered CI95 | Corrected EM | % of K-only LL gain |
|---|---|---|---|---|
| K-only (real) | +2.412 | [+2.21, +2.64] | 0.000 | 100% |
| Wrong-document K | +2.365 | [+2.19, +2.56] | 0.000 | **98%** |
| Random Gaussian K | +2.210 | [+1.72, +2.67] | 0.018 | **92%** |
| Random KV | +2.291 | [+1.59, +2.89] | 0.018 | 95% |
| Zero KV | +2.760 | [+2.49, +3.00] | 0.000 | **114%** |
| Identity V-only | +1.577 | [+1.09, +2.23] | 0.000 | 65% |
| Wrong-document Joint | +4.341 | [+4.03, +4.59] | 0.000 | (Joint +4.410) |
| Token-shuffled KV | −0.008 | [−0.02, +0.01] | **0.929** | — |

Two conclusions:

1. **The LL gain carries no example-specific content.** A wrong document, a random
   Gaussian tensor with the right per-layer moments, or even an all-zero KV produce
   the same (or larger) answer-LL increase. The exact same result holds for the
   equal-layer pair 1.7B→0.6B
   (`reports/phaseB_controls_1.7B_0.6B_seed0_fixed.json`): Zero_KV +2.76 and
   Rand_K +2.21 exceed the real K-only (−0.82) and V-only (+0.83).
2. **The controls are internally valid:** token-shuffling the *student* KV keeps
   corrected EM at 0.929 (content is a bag of facts), proving the corrected
   evaluator can detect usable context. Only teacher-injected states fail.

## 3b. B1 — layer-alignment decoupling (1.7B→0.6B, seed 0, corrected EM)

`report: reports/phaseB_alignment_1.7B_0.6B_seed0.json`

The pair is equal-layer (28→28), so we can arbitrarily scramble the source
layer map while holding capability fixed.

| Layer map | alignment deviation | K ΔLL | V ΔLL | Joint ΔLL | corrected V EM |
|---|---|---|---|---|---|
| proportional (= identity) | 0.00 | −0.82 | +0.83 | −2.24 | 0.054 |
| offset +3 | 2.79 | +0.24 | +0.98 | +1.67 | 0.000 |
| offset −3 | 2.79 | +2.69 | +0.89 | +3.31 | 0.000 |
| random permutation | 8.43 | +0.36 | +1.35 | +1.14 | 0.000 |

Misaligning the layer map does **not** reduce transfer — a random permutation
*increases* the LL deltas relative to the proportional map — while corrected EM
is 0 in every case. This removes the "equal-layer / alignment-gated V transfer"
explanation: layer-map quality is not the variable that controls functional
transfer.

## 3c. B3 — mechanism: output-space V mapper recovers V transfer

`reports/phaseB_mechanism_{1.7B_0.6B,8B_0.6B}_seed0.json` (seed 0, corrected EM)

Routing diagnostic (Self attention vs K-only attention, mean over layers) and
V-mapper objective comparison:

| Pair | routing TV | routing cosine | top-1 agree | Self EM | Affine V EM | **OutAware V EM** |
|---|---|---|---|---|---|---|
| 1.7B→0.6B | 0.165 | 0.971 | 0.701 | 0.911 | 0.054 | **0.911** |
| 8B→0.6B | 0.245 | 0.932 | 0.590 | 0.911 | 0.000 | **0.232** |

An attention-output-aware V mapper (fit `A_S V_T → A_S V_S` on calibration)
recovers V-only task performance almost completely when routing mismatch is
small (1.7B, equal-layer), and partially for the large-gap flagship (8B) where
routing mismatch is larger. Two implications:1. V transfer failure under the standard mapper is substantially a
   **mapper-objective / downstream-consumption** problem — it is not intrinsic
   non-transferability, and it is fixable (reviewer §5).
2. The residual gap tracks the K-induced routing mismatch (TV 0.165→0.245),
   which a V mapper cannot repair. This is consistent with the four-arm result
   that K perturbing the cache is what destroys generation.

Across seeds $1$--$2$ the 1.7B diagnostic and recovery are stable: routing
top-1 agreement $0.698$--$0.701$, OutAware V-only EM
$0.911/0.929/0.964$ (mean $0.935$), Affine $0.054/0.143/0.143$ (mean $0.113$),
shuffled-target $0.089/0.071/0.071$ (mean $0.077$).

**8B→0.6B is equally stable.** Routing top-1 agreement $0.577$--$0.590$
(cosine $0.925$--$0.934$); OutAware V-only EM $0.232/0.250/0.268$ (mean
$0.250$), shuffled $0.036/0.071/0.089$ (mean $0.065$), $W_O$-aware
$0.268/0.286/0.250$ (mean $0.268$).

**B1/B2 across seeds.** All $8$B layer maps give corrected V-only EM $\le0.11$
in every seed (proportional Joint LL $+4.41$--$+4.68$; permutations and
learned selection lower and destabilize it). Held-out layer selection is
identical in all seeds (val top-2 $[12,16]$, $L_{12}$ rank 1): test EM $0.35$
for the validation-selected pair vs $0.22$ post-hoc $8,12$ and $0.00$ for
all-layer injection.

## 3d. Optimization — proper W_O-aware V mapper + content-specificity control

`reports: reports/phaseB_outaware_{1.7B_0.6B,8B_0.6B}_seed0.json`

Extends B3 in the direction the reviewer asked for: the mapper is fit so that
its V output survives the student's `o_proj`. Per KV head `h`, `M_h` is the
least-squares solution of `M_h W_O,q ≈ N_q` for every query head `q` in the GQA
group, where `N_q` is the ridge map `A_q V_T,h → A_q V_S,h W_O,q`. A
shuffled-target mapper (targets from a *different document*) is the control.

| Pair | Self EM | V-Affine | V-OutAware | V-OutAware-shuffled | **V-W_O-aware** |
|---|---|---|---|---|---|
| 1.7B→0.6B | 0.911 | 0.054 | 0.911 | 0.089 | **0.929** |
| 8B→0.6B | 0.911 | 0.000 | 0.232 | 0.036 | **0.268** |

The W_O-aware objective is consistently the best mapper, and the shuffled-target
control stays near the floor in both pairs. This does two things: it implements
the reviewer's W_O-aware mapper, and it rules out the trivial explanation that
the output-aware recovery is just a generic "make the cache student-like"
transform — the recovery is document-content-specific.

## 3e. Innovation — consumption adapter recovers cross-model transfer

`reports: reports/phaseB_adapter_1.7B_0.6B_{o_proj_,}seed{0,1,2}.json`
(rank-8 low-rank correction on every `o_proj`, student and mapper frozen,
trained on the calibration split only, evaluated with corrected EM on held-out
test; mean ± std over 3 seeds)

| Training condition | Self | K-only | V-only | Joint |
|---|---|---|---|---|
| none (published pipeline) | 0.899±0.010 | 0.030±0.010 | 0.113±0.051 | **0.000** |
| adapter on **Joint teacher KV** | 0.940±0.041 | **0.940±0.041** | **0.940±0.041** | **0.827±0.237** |
| adapter on **student (Self) KV** | 0.899±0.045 | 0.203±0.203 | 0.345±0.536 | 0.179±0.247 |
| adapter on **shuffled-doc teacher KV** | 0.482±0.064 | 0.559±0.020 | 0.345±0.054 | 0.458±0.152 |

Honest reading:

- A ~0.7M-parameter consumer adapter trained on teacher KV raises K/V/Joint EM
  from ~0 to **0.94/0.94/0.83** (vs no-adapter 0.03/0.11/0.00).
- Training the same adapter on **student** KV is far weaker (Joint 0.18) — the
  adapter must see the teacher-state distribution.
- The **shuffled-content** control is intermediate (Joint 0.46): part of the
  recovery is generic to teacher-like states (consistent with the wrong-document
  LL controls), part is content-specific. Variance across seeds is high for the
  joint-trained condition (Joint 0.55–0.96).

Conclusion for the paper: cross-model KV transfer failure is a
**state↔consumer compatibility** problem, and a small consumer-side adaptation
substantially recovers it. The contribution is the diagnostic (state↔consumer)
plus the constructive toolkit (output-space / W_O-aware mapper, consumption
adapter), not a claim that adaptation is purely content-specific.

**8B→0.6B replication (3 seeds, corrected EM).** No adapter:
Self $0.899\pm0.010$, K/V/Joint $0.000$. Adapter on teacher KV:
K $0.792\pm0.162$, V $0.357\pm0.047$, Joint $0.470\pm0.119$.
Adapter on student KV: K $0.113\pm0.010$, V $0.107$, Joint $0.119\pm0.021$.
Adapter on shuffled teacher KV: K $0.369\pm0.109$, V $0.113\pm0.052$,
Joint $0.131\pm0.021$. The flagship pair therefore separates the
teacher-state-trained adapter from both controls as cleanly as the 1.7B pair.

**Which projection? (1.7B→0.6B, seed 0)** — adapting `o_proj` is both the most
effective and the most content-specific:

| Adapted modules | Joint-trained Joint EM | Self-trained | Shuffled |
|---|---|---|---|
| `o_proj` | **0.964** | 0.036 | 0.286 |
| `q_proj,k_proj` | 0.625 | 0.411 | 0.500 |
| `v_proj` | (running) | (running) | (running) |

Adapting the output projection (`o_proj`) recovers most and shows the largest
gap over controls, locating the bottleneck at the attention-output consumption
stage rather than at the state projections.

## 4. Impact on the paper

Invalidated by the corrected metric:

- "K transfers near-universally (EM ≥ 0.71 on all 6 pairs)".
- "V transfers only on equal-layer pairs (EM 0.80 / 0.88)".
- "V-only collapses greedy EM on small-gap pairs" and the LL-vs-EM dissociation
  narrative (the reported EM side of the dissociation was the bug).
- All EM columns in `v3_data_baseline.md`, `main.tex` and `factorial_analysis.json`.

Still valid but re-interpreted:

- Teacher-forced LL numbers (the cache is fresh per arm there), **but** the
  controls show they measure a content-agnostic confidence/calibration shift,
  not information transfer. They cannot support transfer claims.

## 5. Follow-up status

Completed this session:

1. ✅ phase4 corrected (6 pairs × 3 seeds) → §2b; phase7 SQuAD corrected → §2c.
2. ✅ Re-derived claims from corrected EM; the transfer claims do not survive →
   reframe needed (negative/methodological + constructive mapper result).
3. ✅ B1 decoupling (§3b), B2 held-out hotspot, B3 mechanism (§3c), B4 controls
   (§3), optimizer W_O-aware mapper + shuffled control (§3d).
4. ⬜ Re-check every figure/table and `scripts/verify_paper_numbers.py` against
   the corrected reports (the old checker still validates the buggy numbers).
5. ⬜ Keep both legacy and corrected numbers in an appendix for auditability.
6. ⬜ B experiments on 3 seeds (currently seed 0 except the four-arm table).

## 6. Reproduce

```bash
cd /workspace/v3
# corrected four-arm main table (3 seeds)
python3 phaseB_fourarm.py --seed 0 --n-calib 70 --n-eval 56
# controls (wrong-doc / zero / random / shuffled / identity)
python3 phaseB_controls.py --pair 8B_0.6B --seed 0 --n-calib 70 --n-eval 56
# layer-alignment decoupling (B1)
python3 phaseB_alignment.py --pair 1.7B_0.6B --seed 0 --n-calib 70 --n-eval 56
# held-out layer hotspot (B2)
python3 phaseB_layers_heldout.py --pair 8B_0.6B --seed 0 --n-calib 70 --n-eval 56
# mechanism + output-aware / W_O-aware V mappers (B3 + optimization)
python3 phaseB_mechanism.py --pair 8B_0.6B --seed 0 --n-calib 70 --n-eval 56
python3 phaseB_outaware.py --pair 1.7B_0.6B --seed 0 --n-calib 70 --n-eval 56
# second domain (cross-domain calibration)
python3 phaseB_squad.py --seed 0 --n-calib 70 --n-eval 30
```

Modules: `phaseB_common.py` (fixed evaluator + harness, learned layer selection),
`phaseB_controls.py` (B4), `phaseB_alignment.py` (B1),
`phaseB_layers_heldout.py` (B2), `phaseB_mechanism.py` (B3 + output-aware),
`phaseB_outaware.py` (W_O-aware + control), `phaseB_fourarm.py` (main table),
`phaseB_squad.py` (second domain).

---

## 7. Revision-2 GPU results (audit2 / REVISION_PLAN2)

### B1 — Evaluator gold-standard validation (`reports/phaseB_evalcheck_seed0.json`)

Full Self test set (56), injected cache vs full doc+query prefill:

| Student | token agreement | normalized-EM agreement | first-logit mean/max abs err | Self EM injected/full |
|---|---|---|---|---|
| 0.6B | 0.786 | 0.964 | 0.595 / 1.344 | 0.911 / 0.875 |
| 1.7B | 0.804 | 0.929 | 0.455 / 0.875 | 0.429 / 0.393 |
| 4B | 0.929 | 0.982 | 0.570 / 0.969 | 0.804 / 0.786 |

The two paths are functionally equivalent (normalized EM agrees on 93–98% of
samples; first-token logits differ only by bf16 cache-vs-recompute noise), but
not bit-identical, so token-string agreement is 79–93%. Premise holds within
bf16 tolerance; the paper should report these full numbers instead of a 6-item
probe and state the residual explicitly.

### B4 — Valid content-breaking control (`phaseB_controls_8B_0.6B_seed{0,1,2}.json`)

The old `Shuf_KV` (same permutation applied to K and V) is permutation-invariant
and therefore invalid. The correct controls, `Shuf_K` (shuffle K, keep V) and
`Shuf_V` (keep K, shuffle V), give corrected EM **0.000** in all three seeds
(vs Self 0.899), while `Shuf_KV` gives 0.86/0.88/0.93. The evaluator is
therefore sensitive to genuine content corruption.

### B3 — 8B→0.6B adapted Self (`phaseB_adapter_8B_0.6B_o_proj_seed{0,1,2}.json`)

The content-matched adapter's own Self reference is **0.964/0.964/0.893**
(mean 0.940), not the unadapted 0.899. The K-only recovery to 0.792 must be
compared against 0.940; conclusions about "recovering most" should use this
same-condition reference.

### C1 — Repaired-regime layer scrambling (`reports/phaseB_alignment_repaired_1.7B_0.6B_seed0.json`)

With the OutAware value mapper (which raises V-only EM to 0.91 under the
proportional map), scrambling the layer map **does** hurt:

| Layer map | V-only EM | Joint ΔLL |
|---|---|---|
| proportional / identity | **0.91** | −0.90 |
| offset +3 | 0.21 | +0.34 |
| offset −3 | 0.23 | +2.82 |
| random permutation | **0.11** | +0.96 |

This reverses the earlier "layer alignment alone does not explain the failure"
conclusion: the earlier null result was a floor effect of the affine mapper.
Under a consumer-aware mapper, V-only EM falls from 0.91 to 0.11–0.23 when the
alignment is broken.

### C2 — 1.7B Self anomaly is not a formatting artifact (`reports/phaseB_selfdiag_seed0.json`)

| Student | raw EM | normalized EM | token F1 | raw-fail/norm-pass |
|---|---|---|---|---|
| 0.6B | 0.911 | 0.911 | 0.304 | 0 |
| 1.7B | 0.429 | 0.429 | 0.136 | 0 |
| 4B | 0.804 | 0.804 | 0.233 | 0 |

Raw and normalized EM coincide in every student, and 1.7B has the lowest token
F1. The non-monotonic Self is genuine, not a metric artifact.

### C3 — Document-clustered EM CI (`reports/phaseB_fourarm_seed0.json`)

Corrected EM with document-clustered bootstrap (8 documents) for the six-pair
scan: every transfer arm's clustered CI touches 0 (e.g. 8B→0.6B K/V [0,0];
4B→1.7B K [0.018,0.107]; 1.7B→0.6B K [0,0.089], V [0.018,0.107]). The
conclusion 0.90 vs 0.00 does not depend on the CI width.

### B2 — Adapter content causality (`reports/phaseB_adapter_causal_1.7B_0.6B_seed0.json`)

With the content-matched adapter held fixed (1.7B→0.6B, seed 0, rank 8,
10 epochs), injecting different states into the four/test arms gives:

| Injected state | EM |
|---|---|
| correct teacher Joint (K/V) | **0.625** |
| wrong-document teacher Joint | 0.000 |
| moment-matched random KV | 0.018 |
| zero KV | 0.000 |
| (K-only / V-only under same adapter) | 0.857 / 0.821 |

The adapter's benefit requires the correct document content: correct ≫ wrong ≈
random ≈ zero. This closes the audit2 §6 concern that the adapter might be
learning the task rather than reading the transferred state.

### B2 (8B) and C4 — completed

**B2 adapter causality, 8B→0.6B** (`reports/phaseB_adapter_causal_8B_0.6B_seed0.json`,
rank 8, 10 epochs): with the content-matched adapter fixed, correct teacher Joint
EM is 0.304, wrong-document 0.143, random 0.161, zero 0.161; K-only 0.964,
V-only 0.232, Self 0.982. Correct > controls but the gap is small, i.e. a
larger generic teacher-state component on the large-gap pair than at 1.7B
(correct 0.625 vs controls ≈0). Reported honestly as partial content causality.

**C4 cross-domain repair** (`phaseB_squad_outaware_seed0.json`,
`phaseB_squad_adapter_seed0.json`): applying the synthetic-trained OutAware
mapper on SQuAD raises the V-only LL gain to +4.62 but corrected EM stays 0.000;
the synthetic-trained adapter raises SQuAD Self EM 0.367→0.600 but K/J EM remain
0.000 and V-only 0.033. The constructive fix does **not** transfer across
domains; the repair is task-distribution-specific.

**C3 completion** (`phaseB_fourarm_8B_4B_seed0.json`): 8B→4B Self 0.804,
K 0.143, V 0.429 (clustered CI [0.375,0.482]), Joint 0.018. V-only is the only
arm whose clustered CI excludes zero, and it remains far below Self.


---

## 8. Paper-side integration of the revision-2 results (W20b)

Section 7 records what the GPU machine produced. This section records the
derived publication quantities the paper now prints, so that every number in the
manuscript can be checked against a report without re-running anything.

**Independently recomputed 3-seed aggregates** (all from `reports/*.json`, none
hand-written):

| Quantity | Report(s) | Value used in the paper |
|---|---|---|
| 8B→0.6B controls, `Shuf_K` / `Shuf_V` | `phaseB_controls_8B_0.6B_seed{0_fixed,1,2}.json` | ΔLL −1.19±0.10 / −1.28±0.04, EM 0.000 (3/3) |
| 8B→0.6B `Shuf_KV` invariance check | same | ΔLL −0.01±0.00, EM 0.887±0.037 |
| Random-K EM (3 seeds, was seed-0 only) | same | 0.012±0.010 |
| Wrong-document Joint ΔLL dispersion | same | +4.50±0.18 (paper previously printed it bare) |
| Adapter Self column, both pairs | `phaseB_adapter_{1.7B,8B}_0.6B_*_seed*.json` | 0.940±0.041 (teacher); 0.976±0.021 / 0.411±0.099 (8B student / shuffled) |
| Proportional V-only EM under the consumption-space mapper | `phaseB_outaware_1.7B_0.6B_seed{0,1,2}.json` | 0.911 / 0.929 / 0.964 |
| Post-hoc oracle layer pair (12,20) | `phaseB_layers_heldout_8B_0.6B_seed{0,1,2}.json` | 0.131 (vs 0.351 validation-selected) |
| Projection ablation, `v_proj` / `q_proj,k_proj` | `phaseB_adapter_1.7B_0.6B_{v_proj,q_proj_k_proj}_seed0.json` | joint 0.964 / 0.625 (seed 0) |

**Wording correction carried into the paper.** `Shuf_K` and `Shuf_V` are built
from the *student's own* cache (`phaseB_controls.py:141-148`:
`shuf = shuffled_kv(eval_s[i])`, `Shuf_K = KV(k=shuf.k, v=sv)`), not from the
teacher state. They are therefore pair-independent by construction, which is why
1.7B and 8B report identical values. The paper describes them as
correspondence-breaking controls on the student's cache and says so explicitly;
it does not read their cross-pair agreement as an independent replication.

**Conclusion that reversed.** Section 3b (B1, affine mapper) reported that
scrambling the layer map does not reduce transfer. Section 7 C1 shows that result
was a floor effect. The paper now reports both regimes in Section 6.1, retitled
"Layer alignment matters once the consumer reads the state", and the old blanket
claim is asserted absent across the whole manuscript by
`paper/audit/verify_audit2_edits.py`.

**Contaminated artifact (removed).** `reports/phaseB_controls_8B_0.6B_seed0.json`
was a stale legacy-evaluator run (Self 0.429, `Zero_KV` EM 1.000) that predated
the controls fix. No paper number ever used it; it was renamed
`phaseB_controls_8B_0.6B_seed0_LEGACY_DO_NOT_USE.json` so that it could not be
mistaken for the corrected seed-0 run (`..._seed0_fixed.json`, Self 0.911,
`Zero_KV` EM 0.000), and the W23 repository cleanup deleted it outright (it
remains in git history only). The remaining `phaseB_controls_*` reports are all
corrected-evaluator runs; `scripts/cluster_stats_audit3.py` still skips any file
whose name contains `LEGACY` in case an old copy resurfaces.

**Traceability.** `paper/audit/verify_audit2_edits.py` asserts, per audit2
finding, that the superseded strings are absent and the replacements present,
including one assertion per new number above. It distinguishes PART I (W19b,
edit-only findings) from PART II (W20b, findings that needed new GPU runs) and
runs clean.

---

## 9. Revision-3 results (audit3 / REVISION_PLAN3)

### A6 — Document-clustered EM for every seed, computed on the CPU (W22)

**Source:** `reports/cluster_stats_audit3.json`, produced by
`scripts/cluster_stats_audit3.py` from the archived reports plus
`data/test_v2_seed{0,1,2}.json`. Cluster unit = document; 8 clusters per seed;
10,000 cluster-bootstrap resamples. No GPU and no re-run were needed: the
four-arm reports carry `doc_id` on every row for all three seeds, and the
control and mapper reports carry `id`, which joins to the dataset.

**Why this was a plan item.** The paper asserted that "document-clustered EM
intervals exist only in the seed-0 reports, where the clustering fix landed
after seeds 1 and 2 had been written". That was true when written and is now
false. audit3 (par.10) asked for all-seed clustered inference; the archived rows
already contained what it needed.

**Key numbers** (also quoted in Sections 5.1 and 7.3 of the paper):

| Pair, arm | seed 0 | seed 1 | seed 2 | clustered CI95 (per seed) |
|---|---|---|---|---|
| 8B→4B V-only | 0.429 | 0.446 | 0.446 | `[0.375,0.482]`, `[0.429,0.482]`, `[0.429,0.482]` |
| 8B→4B K-only | 0.143 | 0.161 | 0.143 | `[0.071,0.214]`, `[0.089,0.250]`, `[0.089,0.196]` |
| 8B→4B Joint | 0.018 | 0.054 | 0.125 | `[0.000,0.054]`, `[0.018,0.107]`, `[0.071,0.179]` |
| 8B→4B Self | 0.804 | 0.839 | 0.804 | `[0.732,0.857]` to `[0.750,0.839]` |
| 1.7B→0.6B V-only (affine) | 0.054 | 0.143 | 0.143 | `[0.018,0.107]`, `[0.071,0.214]`, `[0.071,0.214]` |
| 1.7B→0.6B V-only (attention-output-aware) | 0.911 | 0.929 | 0.964 | `[0.839,0.982]`, `[0.857,0.982]`, `[0.911,1.000]` |

**Effect on the paper.** No arm ordering changes. Clustered intervals are wider
than per-sample ones, every 8B→4B V-only interval excludes zero while remaining
far below the same seed's Self, and the affine-versus-consumption-space value
intervals stay disjoint in all three seeds. The Limitations bullet "Corpus
effects" and the Table 1 caption were rewritten accordingly; the superseded
sentences are asserted absent by `paper/audit/verify_audit3_edits.py`.

**Deliberately not available.** Adapter and causality reports carry no
per-sample rows, so those results cannot receive clustered intervals from the
archive. `phaseB_adapter.py --dump-rows` was added and the A2 runs regenerate
them with rows; the same script picks them up automatically
(`adapter_causality` section of the JSON currently reports
"regenerated without --dump-rows" for the two archived single-seed runs).

### A1–A5 — pending GPU runs

The GPU-side code for A1 (evaluator residual and attribution probe), A2
(20-epoch causality, three seeds, `--dump-rows`), A3 (within-SQuAD repair) and
A4 (state/consumption error budget) is committed and queued by
`scripts/run_audit3_gpu_queue.sh`. Nothing here should be cited until the
corresponding `reports/*.json` land and are registered in a section 10.

### A7 — Self-baseline failure taxonomy (optional item, audit3 par.13, CPU)

**Source:** `reports/error_taxonomy_selfdiag.json`, produced by
`scripts/error_taxonomy_selfdiag.py` from the generations already stored in
`reports/phaseB_selfdiag_seed0.json` (the `return_gen` change of REVISION_PLAN2
I-4). Rule-based and deterministic: empty or punct-only, denial phrase, 2-gram
repeat covering half the tokens, generation at the 16-token decode cap, an
entity/number that is not the gold answer, or other free text.

| Student | correct | failures | failure classes |
|---|---|---|---|
| 0.6B | 51 | 5 | `other_text` 5 |
| 1.7B | 24 | 32 | `wrong_entity` 32 |
| 4B | 45 | 11 | `wrong_entity` 11 |

**Effect on the paper.** The 1.7B dip is now characterised rather than merely
reported: every one of its 32 failures is a well-formed answer naming the wrong
component, plant, number, or yes/no polarity, and none of the four artefact
classes (refusal, repetition loop, empty output, truncation at the cap) occurs
in any student. The Limitations bullet "Unexplained baseline spread" was
rewritten accordingly. This closes audit3 par.13's request and strengthens the
audit2 C2 finding without changing any number.

### Text-only audit3 items landed in the same round

T1 (key-side routing versus value-side consumption, four sites plus the
Analysis 6.3 subsection title), T1b (Discussion 7.1 restructured into the four
layers of audit3 par.16), T2 (the two evaluator defects are attributed to our
own earlier implementation, with an explicit statement that third-party
implementations were not audited), T3 ("end-to-end equivalent to full prefill"
replaced by "closely agrees ... on normalized answer exact match"). Each is
pinned by `paper/audit/verify_audit3_edits.py`.

---
## 10. Revision-3 GPU results, consumed into the paper (W27)

Source: commit `e7d0048` (W26) ran `scripts/run_audit3_gpu_queue.sh` end to end
(18 reports plus the queue log) and fixed two runtime blockers
(`phaseB_errorbudget` compared `d @ Wq` with an untransposed `o_proj` slice, and
the container's cgroup watchdog killed the 8B load until
`low_cpu_mem_usage=True` + `device_map={"": 0}` streamed the weights; the change
was verified numerically identical, max abs logit diff `0.0`). Every number below
was recomputed here from the archived reports before it entered `main.tex`.

### A1 — evaluator residual, attributed (audit3 par.3; paper: Abstract, 3.3, 5.3, Limitations)

| student | top-1 agreement | median KL (nats) | max \|Δ logit\| | normalized-EM agreement |
|---|---|---|---|---|
| 0.6B | 0.929 | 0.0047 | 1.344 | 0.964 |
| 1.7B | 0.929 | 0.0015 | 0.875 | 0.929 |
| 4B | 0.982 | 0.0010 | 0.969 | 0.982 |
| SQuAD 0.6B | 0.933 | 0.0052 | — | 0.933 |

Attribution probe (`phaseB_evalcheck_attrib.json`): repeated capture is
bit-identical (0.0), the numpy cache round-trip is exact (max |Δ| = 0.0, top-1
1.000), explicit `position_ids` are exact (0.0), and the entire residual sits on
the attention kernel path (max |Δ| = 0.844, median KL = 0.005) which does not
shrink under eager attention (max 0.9375, top-1 0.875 on the 8-item probe).

**Pre-registered gate, adjudicated in writing rather than relaxed.** The gate
said row 1 requires "top-1 agreement 1.000 and median KL ≤ 0.01", and row 2
triggers the stop rule on "top-1 disagreement or median KL > 0.1". Top-1
agreement is 0.929–0.982, so row 1 is not literally met, and row 2's trigger
literally fires on the disagreement. Row 2's stated diagnostic intent, however,
is a residual *implementation* discrepancy, and that is excluded three ways: the
median KL is 20–100× below the 0.1 stop threshold, the attribution probe explains
the flips (cache construction and position handling are exact; the difference is
the one-token-versus-full-sequence kernel path), and the same magnitude appears
under eager attention, i.e. it is a bf16 numerical effect rather than our logic.
We therefore (a) record the gate as not literally satisfied, (b) state the
amended reading explicitly here and in the paper's Limitations, and (c) keep the
paper away from any "equivalent to full prefill" wording: the claim is agreement
on the extracted answer plus top-1 agreement and KL on the first-token
distribution. Nothing in the paper was changed by lowering a threshold.

### A2 — causality with the final 20-epoch adapter (audit3 par.6/7; paper: `tab:causal`, 6.4, Abstract, Limitations)

| injected state | 1.7B→0.6B (3 seeds) | 8B→0.6B (3 seeds) |
|---|---|---|
| Correct teacher KV (joint) | **0.905 ± 0.083** | **0.321 ± 0.129** |
| Wrong-document teacher KV | 0.131 ± 0.021 | 0.131 ± 0.041 |
| Moment-matched random KV | 0.095 ± 0.055 | 0.125 ± 0.018 |
| Zero KV | 0.107 ± 0.071 | 0.214 ± 0.095 |
| Self (own cache, adapted) | 0.952 ± 0.054 | 0.929 ± 0.095 |

Per-seed margins (correct − worst destroyed): 1.7B `+0.714 / +0.679 / +0.857`
(document-clustered CI95 excludes zero in all three seeds); 8B
`+0.179 / +0.071 / +0.036`, mean `+0.095`, with the clustered margin intervals
of seeds 1 and 2 reaching or containing zero (`[0.000, +0.125]`,
`[-0.036, +0.089]`) while seed 0's `[+0.107, +0.232]` excludes it. Gate verdict:
**established on the equal-depth pair, unresolved on the flagship pair** — the
paper says exactly that and does not upgrade it. Caveat carried into the paper:
repeating the 8B training at a fixed seed gave 0.679 and 0.429 on the correct
arm, so the margin is comparable to adapter training variance. The per-seed
margins were re-derived from the reports in W29; see Section 12 for the
correction and the clustered intervals.

### A3 — within-domain repair on the second domain (audit3 par.8; paper: `tab:squadrepair`, 5.3, 6.4, Abstract, Limitations)

Fifteen SQuAD calibration documents and fifteen held-out documents, three
document splits: held-out Self `0.333 / 0.200 / 0.400`, and **every** transfer
arm (K-only, V-only under affine / output-aware / $W_O$-aware, joint, and the
same set with a within-domain adapter) answers `0.000`. Adapted Self rises to
`0.667 / 0.400 / 0.400`, i.e. the adapter improves the student's reading of its
own state without making the transferred state usable. Gate verdict: the
**stronger negative** branch — consumer compatibility is bound to the task
distribution, not merely non-transferable across domains. Caveats recorded in
the paper: 15 calibration documents only, and a held-out Self ceiling of
0.20–0.40.

### A4 — state-space versus consumption-space error budget (audit3 par.11; paper: `tab:errorbudget`, 6.3, 7.1, Abstract)

Pooled over both pairs and three seeds (`phaseB_errorbudget_*.json`),
with rank correlation against V-only EM over the five variants:

| mapper | $e_{raw}$ | $e_{attn}$ | $e_{W_O}$ | V-only EM |
|---|---|---|---|---|
| raw layer selection | 3.73 | 3.72 | 3.65 | 0.000 |
| affine | **0.267** | 3.20 | 3.75 | 0.057 |
| attention-output-aware | 0.772 | 0.073 | 0.082 | **0.592** |
| shuffled target | 5.58 | 0.903 | 1.09 | 0.071 |
| $W_O$-aware | 0.699 | **0.076** | **0.081** | 0.586 |

Spearman with EM: `-0.10` (raw), `-1.00` (attention output), `-0.80`
($o$-projection space). Gate verdict: the paper's central claim needs **no
downgrade** — raw representation error does not rank the mappers, and error in
the receiver's consumption space does. The shuffled-target mapper is reported in
the paper as the explicit exception (low attention-output error against the wrong
document's targets, floor EM).

### Where each result landed

W1 (Abstract + 3.3 + Fig. 2 caption + Limitations "Evaluator provenance"), W2
(5.3 validator sentence, 6.4 within-domain paragraph, `tab:squadrepair`), W3
(`tab:causal` and the 6.4 causality paragraph, Abstract, Introduction,
Limitations "Flagship causality" and the new "Adapter training variance" bullet),
W4 (a new 6.3 paragraph, `tab:errorbudget`, Discussion 7.1(ii), Conclusion),
W5/W6 unchanged. The
Limitations "Partial seed coverage" bullet now lists only the repaired-regime
layer scrambling and the projection ablation as seed-0 results. All of it is
pinned by `paper/audit/verify_audit3_edits.py` (tags A1–A4, including report
cross-checks) and `paper/audit/verify_audit2_edits.py`.

---

## 11. Revision-4 paper-side changes (W28, audit4 / REVISION_PLAN4 PART I)

Audit4 (7/10 Accept) asked for a framing upgrade rather than more experiments.
PART I of the plan was executed with no GPU work; the ten text items, and the
three factual corrections audit4 did not raise, are recorded here because this
file is the provenance record for every paper-side change.

### T4/§0.5 — the cost crossover disagreed with the Method

`main.tex` 6.6 quoted `2050` tokens for a `58.8`M-parameter fp32 mapper
(`224.2` MB). That figure traces to `experiments/phase3_rate_law.py:50-55,173`
(`reports/archive/phase3_rate_law_seed0.json`), which counts one
`1024x1024` map per layer for K and V: `2 x 28 x (1024^2 + 1024) = 58,777,600`
parameters. The Method defines the mapper actually used in this paper as
per-(layer, head) affine maps (`fit_mapper` -> `AffineMapper`, `D = 128`;
`phaseB_errorbudget.apply_oa` indexes `W[(l, h)]` of shape `128x128`).
Recomputed from that definition:

| quantity | value |
|---|---|
| per-head K+V parameters (`2 x 28 x 8 x (128^2+128)`) | `7,397,376` (`7.40`M) |
| footprint at fp32 | `28.2` MiB |
| fp16 KV per token (`28 x 2 x 8 x 128 x 2 B`) | `112` KiB |
| byte crossover | `~260` tokens |

The paragraph was moved to Appendix B with the corrected numbers; the superseded
figure is not quoted in the paper (the older number was `8x` too large, and the
two parameterisations are not comparable). The consumption adapter's `~0.7`M
figure was re-derived the same way (`28 x (1024x8 + 2048x8) = 688,128`) and is
consistent with the paper.

### T7/§0.1 — the rank correlations were reported only as pooled values

The paper's `-1.00 / -0.80 / -0.10` are computed on means pooled over six runs
(two pairs, three seeds) and then ranked over five mapper variants. The
`reports/phaseB_errorbudget_*.json` files carry a run-to-run range the paper did
not disclose, and the report notes already said "n is too small for a p-value".
Recomputed per run (tag T7 in `verify_audit4_edits.py`):

| run | `rho(e_raw, EM)` | `rho(e_attn, EM)` | `rho(e_wo, EM)` |
|---|---|---|---|
| 1.7B->0.6B s0 | -0.40 | -1.00 | -1.00 |
| 1.7B->0.6B s1 | -0.60 | -0.90 | -0.90 |
| 1.7B->0.6B s2 | -0.60 | -0.80 | -0.80 |
| 8B->0.6B s0 | -0.20 | -0.80 | -0.80 |
| 8B->0.6B s1 | -0.20 | -0.80 | -0.90 |
| 8B->0.6B s2 | -0.10 | -0.90 | -0.90 |
| pooled (paper) | **-0.10** | **-1.00** | **-0.80** |

The Abstract, 6.3, 7.1(ii) and the `tab:errorbudget` caption now report the
pooled value together with the per-run range and state that five variants cannot
support a p-value. Direction is unchanged in all six runs; the sweep planned as
REVISION_PLAN4 A2 is what would turn this into an estimated law.

### The other eight items

| tag | change | sites |
|---|---|---|
| T1 | title -> *Cross-Model KV Transfer Needs Functional Compatibility: Why Representation Alignment Is Not Enough* | `main.tex` (title, `pdftitle`, keywords), `arxiv_submission/main.tex`, `README.md`, `arxiv_metadata.md` |
| T2 | contributions rewritten as four items (evaluation principle / decomposition / consumer-space alignment / repairability and boundary), with the general principle and our own implementation defect kept in separate sentences | Introduction |
| T3 | new Method 3.3 *Functional compatibility decomposition* (`eq:decomp`), stated as a perturbation identity with the explicit caveat that it does not make EM additive and does not identify the binding term | Method |
| T4 | factorial decomposition + Table (`tab:factorial`) and the cost heuristic moved to Appendices A/B after the bibliography | 6.1 pointer, Appendix A/B |
| T5 | "joint arm is the weakest arm everywhere" replaced by the four-pair statement; Related Work no longer blames a "standard evaluation" | 7.1(iv), Related Work |
| T6 | *task-conditioned* naming introduced with its qualifiers (one second domain, fifteen calibration documents, held-out Self 0.20-0.40, calibration volume not excluded) | Abstract, 5.3, 6.4, 7.2, Limitations |
| T9 | the four v1 documents (`BRIEF`, `project_context`, `architecture`, `v3_data_baseline`) moved to `paper/archive/*_v1_stale.md` with a STALE header | `paper/archive/` |
| T10 | all `Section~N` literals replaced by `\label`/`\ref` (8 sites); the guard also checks for undefined and duplicate labels | whole paper |

### Verification

`paper/audit/verify_audit4_edits.py` (new) pins T1-T10 plus the recomputed
correlations and the cost arithmetic; the four superseded assertions in
`verify_audit2_edits.py` (B3, B4) and `verify_audit3_edits.py` (T1, A4) were
repointed at the current wording with a comment naming this plan. `main.tex`
compiles with 0 errors and 0 overfull boxes; underfull hbox warnings remain in
the Reproducibility statement paragraph (24 pages, re-verified by a Tectonic
build in W29), and `arxiv_submission/main.tex` is byte-identical to it.

## 12. W29 correction: flagship causality margins (paper-side numbers)

The per-seed margins printed for the causality test did not reproduce from the
committed reports, although every other number in the same table did.

| quantity | previously printed | recomputed from `reports/phaseB_adapter_causal20_*.json` |
|---|---|---|
| 1.7B->0.6B per-seed margins | `+0.714 / +0.643 / +0.857` | `+0.714 / +0.679 / +0.857` |
| 1.7B->0.6B mean | `+0.738` | `+0.750` |
| 8B->0.6B per-seed margins | `+0.107 / +0.071 / +0.036` | `+0.179 / +0.071 / +0.036` |
| 8B->0.6B mean | `+0.071` | `+0.095` |
| zero-containing clustered intervals | "one seed" | seeds 1 and 2 |

Both aggregation paths inside those reports (`results.causal` and
`rows_by_condition["adapter(joint)"]`) agree with the recomputed column, and
`ITERATION_LOG.md` (W26) already recorded `+0.750` and `+0.095`: the reports and
the log were consistent, and the transcribed text had drifted.

Document-clustered bootstrap CI95 for the margin (cluster = document, 8 clusters,
10,000 resamples; bounds stable across five bootstrap seeds):

| run | margin | clustered CI95 | excludes 0 |
|---|---|---|---|
| 1.7B->0.6B s0 | `+0.714` | `[+0.625, +0.786]` | yes |
| 1.7B->0.6B s1 | `+0.679` | `[+0.625, +0.714]` | yes |
| 1.7B->0.6B s2 | `+0.857` | `[+0.857, +0.857]` | yes |
| 8B->0.6B s0 | `+0.179` | `[+0.107, +0.232]` | yes |
| 8B->0.6B s1 | `+0.071` | `[+0.000, +0.125]` | no (touches 0) |
| 8B->0.6B s2 | `+0.036` | `[-0.036, +0.089]` | no |

Gate verdict unchanged: **established on the equal-depth pair, unresolved on the
flagship pair** (8B mean `+0.095` is below `0.10`). Text, the Table 5 caption,
the Abstract and the Limitations entry were corrected;
`paper/audit/verify_audit3_edits.py` now pins the per-seed margins and both means
so the numbers cannot drift again, and fails if the flagship mean reaches `0.10`
while the paper still reports it as unresolved.
