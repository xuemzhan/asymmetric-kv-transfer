# W21 - End-to-end self-review before reviewer submission

**Date:** 2026-09-13
**Scope:** full read of `paper/main.tex` at commit `09dca3b` (W20b), every
number-bearing sentence and every table cell re-checked against `reports/*.json`.
**Method:** `paper/audit/verify_audit2_edits.py` PART III now pins each of these
corrections so they cannot silently regress.

Status key: **FIXED** = paper changed. **VERIFIED** = checked, no change needed.

---

## 1. Numerical errors corrected

### E1 - `tab:main`, `4B->0.6B` V-only `dLL` std  **FIXED**
`paper/main.tex` printed `-1.22 \pm 0.06`. `phaseB_fourarm_seed{0,1,2}.json`
gives per-seed `-1.26696 / -1.23287 / -1.15985`, i.e. sample std `0.0547` ->
`0.05` at two decimals. `tab:factorial` already printed `0.05` for the same
quantity, and its caption asserts the two tables agree. Now `-1.22 \pm 0.05`.

### E2 - `tab:adapter`, `1.7B->0.6B` block, four mis-rounded cells  **FIXED**
Source: `phaseB_adapter_1.7B_0.6B_seed0.json` +
`phaseB_adapter_1.7B_0.6B_o_proj_seed{1,2}.json`.

| cell | printed | correct |
|---|---|---|
| `none`, V-only std | 0.051 | **0.052** (0.051549) |
| `student KV`, K-only mean | 0.203 | **0.202** (0.202381) |
| `shuffled`, K-only mean/std | 0.559 \pm 0.020 | **0.560 \pm 0.021** (0.559524 / 0.020620) |
| `shuffled`, V-only std | 0.054 | **0.055** (0.054554) |

The matching sentences in the body (`0.203/0.345/0.179`, `0.559/0.345/0.458`)
were updated with them. The `8B->0.6B` block needed no change; it was already
correct to three decimals.

### E3 - Analysis 6.1, `8B->0.6B` returned-EM claim  **FIXED**
Printed: "corrected V-only EM is 0.000 for every map and seed, and K-only never
exceeds 0.107". `phaseB_alignment_8B_0.6B_seed{0,1,2}.json` (6 maps x 3 seeds
= 18 cells) gives V-only EM `0.000` in 17 cells, with one exception (seed 2,
learned top-1 V selection, `0.107`), and K-only peaking at `0.179` (seed 1,
learned top-1 V selection). The first clause was true only for the
proportional/offset/permuted maps; the second was off by 0.072.
Now: "0.000 for every map and seed except one learned-selection cell (0.107),
and K-only never exceeds 0.179".

### E4 - Limitations, "Partial seed coverage"  **FIXED**
Printed: the `1.7B->0.6B` control suite, the identity V-only control and the
random-KV control are seed 0 only. All three have three seeds
(`phaseB_controls_1.7B_0.6B_seed{0,1,2}_fixed.json`, and the `Identity_*` /
`Rand_*` arms are present in every control report for both pairs). The Results
text itself already said "The same pattern holds over three seeds for the
equal-depth pair 1.7B->0.6B", so the paper contradicted itself. Only the
repaired-regime layer scrambling, the projection ablation and the two causality
runs are genuinely seed 0. Corrected to "Four results are seed 0 only", and the
three-seed coverage of both control suites is now stated.

### E5 - first-token logit error, five sites, three different values  **FIXED**
`phaseB_evalcheck_seed0.json` maxima are `1.34375 / 0.875 / 0.96875` (0.6B /
1.7B / 4B); the digest rounds the first to `1.344`. The paper printed
`at most 1.35` (Abstract, Contributions), `is 1.34, 0.88, and 0.97` (Method)
and `within 1.34` (figure caption), and the figure's own data table uses
`1.344`. Standardised on `1.344` everywhere; `0.875` and `0.969` adopted from
the same table.

### E6 - routing total-variation, two sites  **FIXED**
Printed `0.165` and `0.245` next to top-1 agreement `0.70 / 0.59` and cosine
`0.97 / 0.93`, all presented as means. The top-1 and cosine values are
three-seed means; the TV values were the seed-0 per-layer means
(`phaseB_mechanism_*_seed0.json`). Three-seed means are `0.167` and `0.251`.
Now reported as seed means in both places.

---

## 2. Claim provenance corrected

### E7 - G0 gate: undisclosed re-run, and a log statistic that does not reproduce  **FIXED**
The paper said the gate's likelihood clause "stands" without saying that the
pre-registered gate was defined on the earlier `42`/`14` split and later re-run
at `70`/`56` after a baseline double-counting bug fix.

While drafting that disclosure, `ITERATION_LOG.md:129` ("V-only vs Self:
Wilcoxon p = 0.074, Cohen d = 1.90") was checked against
`reports/g0_seed0.json`. **It does not reproduce.** Applying the repository's
own `stats_utils.paired_wilcoxon_test` to the 14 archived per-sample pairs gives
`p = 0.119`, `d = 0.492`. No pairing of any arm against `Self`, `student_full`
or `teacher_full` in that report yields `p = 0.074`; the value `d ~ 1.9` appears
only for the *K-only* arm (`d = 1.952`). The log entry is therefore treated as
unsourced and is **not** quoted in the paper.

The paragraph now states only what the archived report supports: on the `42`/`14`
split V-only sat `+1.75` above Self (`-9.97 \pm 2.60` against
`-11.72 \pm 2.07`, 95% intervals overlapping; `ci95` in that report is
`1.96 \cdot SEM`), the task-metric clause is withdrawn, and the likelihood
clause was re-run at `70`/`56` with the same verdict
(`dLL = -0.045 / -0.219 / -0.415`, `p = 0.024 / 0.125 / 0.144` from
`reports/g0_v2_summary.json`).

### E8 - SQuAD comparison was a seed-0 difference presented as a protocol difference  **FIXED**
"`dLL +4.62` against `+1.05` under the default protocol": `+4.62` is seed 0
(`phaseB_squad_outaware_seed0.json`, the only seed for that run) and `+1.05` is
seed 0 of the default protocol (`1.0527`); the three-seed default mean is
`+0.97`. Now reads "at seed 0, against +1.05 for the default protocol at the
same seed".

### E9 - adapted-Self baseline sentence was self-contradictory  **FIXED**
"an unadapted Self baseline would have understated the gap on the flagship pair
and overstated it on no pair" - on the equal-depth pair the adapted Self is
`0.940` and the recovered K/V arms are also `0.940`, so against the unadapted
`0.899` those arms would appear to *beat* Self by `0.041`, which is neither an
over- nor an understatement of the gap. Rewritten to state the arithmetic
directly.

---

## 3. Checked and confirmed correct (no change)

- `tab:controls` - all nine rows, `dLL`, EM and share, from
  `phaseB_controls_8B_0.6B_seed{0_fixed,1,2}.json`.
- `tab:factorial` - all 18 cells match `factorial_analysis.json.aggregate`.
- `tab:mappers` - all eight cells reproduce from the `phaseB_outaware_*` rows.
- `tab:causal` - all ten cells reproduce from `phaseB_adapter_causal_*.json`
  (rank 8, 10 epochs).
- `tab:ablation` - all 24 cells reproduce from the three
  `phaseB_adapter_1.7B_0.6B_*_seed0.json` reports.
- `tab:squadrepair` - all twelve cells reproduce from the three SQuAD reports.
- Data split sentence "train 70 / validation 28 / test 56": confirmed against
  `data/{train,val,test}_v2_seed{0,1,2}.json`.
- Largest single-seed K / Joint EM `0.161` / `0.125`: confirmed as 8B->4B seed 1
  and seed 2.
- Seed-0 document-clustered EM intervals for 8B->4B (`[0.375, 0.482]`,
  `[0.071, 0.214]`, `[0.000, 0.054]`) and the control intervals
  (`[+2.21, +2.64]`, `[+2.49, +3.00]`).
- Held-out layer selection: `(12, 16)` and `L12` first in all three seeds;
  test EM `0.339 / 0.357 / 0.357` (mean `0.351`), post-hoc `(8, 12)` `0.220`,
  post-hoc `(12, 20)` `0.131`, all-layers `0.000`, Self `0.899`.
- Routing diagnostics: top-1 `0.698-0.701` and `0.577-0.590`, cosine
  `0.970-0.971` and `0.925-0.934`.
- Repaired-regime scrambling (1.7B->0.6B, seed 0): V-only EM `0.911` -> `0.214`,
  `0.232`, `0.107`, clustered intervals `[0.839, 0.982]` vs `[0.125, 0.304]`,
  `[0.143, 0.321]`, `[0.054, 0.179]`.
- Affine-regime layer map (1.7B->0.6B, 3 seeds): proportional V-only EM
  `0.05-0.14`, offsets and permutation `0.000` in every seed, permutation V
  `dLL` `+1.35` vs `+0.83` at seed 0.
- SQuAD per-seed p-values (`2.8e-6`, `4.7e-7`, `2.1e-7`), `Self = 0.367`,
  legacy `0.733` / `0.767`.
- Cost heuristic: `58.78M` params, `224.2` MiB (`ITERATION_LOG.md` section on
  the Phase 3 crossover), `~2050` tokens.
- `~0.7M` adapter parameters is arithmetic, not a measured quantity:
  rank-8 on a `0.6B` `o_proj` (`8x2048 + 1024x8 = 24576` per layer over 28
  layers is 0.688M). Recorded in `METRIC_CORRECTION.md` section 8.

---

## 4. Observations left as-is

- The `1.7B` row of `tab:mappers` prints bare means while the `8B` row prints
  mean `\pm` std, although both come from three-seed runs. The values are
  correct; only the dispersion column is asymmetric. Left unchanged to avoid
  introducing numbers the caption does not currently account for.
- `\label{sec:factorial}` is defined but never referenced, and the
  `\need{}` "NEEDS DATA" macro is now unused. Cosmetic.
- `figures/fig_cost.pdf`, `fig_problem.pdf`, `fig_cca.pdf`,
  `fig_corrected_main.pdf`, `fig_exploration.pdf`, `fig_main.pdf`,
  `fig_protocol.pdf`, `fig_readouts.pdf` and `fig_squad.pdf` still track in
  `paper/figures/` but are not included by `main.tex`; the arXiv bundle carries
  only the seven figures the paper uses.
- `phaseB_controls_8B_0.6B_seed0_fixed.json` / `seed1.json` / `seed2.json`
  naming is not uniform (seed 0 carries a `_fixed` suffix, seeds 1-2 do not).
  The paper's reproducibility section refers to `phaseB_controls_*.json`, which
  matches all of them, so the glob is safe.
