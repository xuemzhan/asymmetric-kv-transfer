# `reports/` — provenance rules for the experiment JSONs

Every number that enters `paper/main.tex` is transcribed from a file in this
directory. These JSONs are deliberately tracked by git (unlike `data/*.json`,
which is regenerable) because they are the only record of what was run.

**Two different evaluators wrote files in here, and one of them was wrong.**
Read this before citing anything.

## The rule

`experiments/phaseB_common.py` implements the corrected evaluator
(`greedy_answer_fixed` + `score_arm`). Anything produced by an earlier script
used the defective greedy exact-match path in `experiments/phase0_g0.py`
(a `DynamicCache` advanced through the query and gold answer, plus a re-fed
final query token). See `paper/audit/METRIC_CORRECTION.md`.

> **Log-likelihood fields survive either way.** The defect is confined to the
> generation path, so `*_ll`, `summary_ll`, `delta_vs_self`, `wilcoxon_*` and the
> rank/correlation fields are valid in both generations of report. Only EM is
> affected — but per the control suite, a likelihood gain is not evidence of
> content transfer either, so cite LL only as a likelihood result.

## EM is INVALID in these files (legacy evaluator)

| files | producing script |
|---|---|
| `g0_v2_seed{0,1,2}.json`, `g0_v2_summary.json` | `experiments/phase0_g0_v2.py` |
| `phase1_causal_v2_seed{0,1,2}.json` | `experiments/phase1_causal_v2.py` |
| `phase4_scaling_law_v2_seed{0,1,2}.json` | `experiments/phase4_scaling_law_v2.py` |
| `phase7_second_domain_seed{0,1,2}.json` | `experiments/phase7_second_domain.py` |
| `w2_baseline_fix.json` | `tests/test_w2_baseline_fix.py` |
| `archive/*.json` | the superseded v1 pipeline |

Concretely, `phase4_scaling_law_v2_seed*.json` still carries
`pair_results[].exact_match` with K-only EM of 0.71–0.95 — the very figures the
paper withdrew. The corrected value is 0.000–0.161. **Do not cite those files'
EM fields, and do not "rediscover" them as an inconsistency.**

## EM is VALID in these files (corrected evaluator)

`phaseB_fourarm*`, `phaseB_controls*`, `phaseB_alignment*`, `phaseB_alignment_repaired*`,
`phaseB_mechanism*`, `phaseB_outaware*`, `phaseB_adapter*`, `phaseB_errorbudget*`,
`phaseB_layers_heldout*`, `phaseB_evalcheck*`, `phaseB_selfdiag_seed0.json`,
`phaseB_squad*`, `phaseB_squadwithin*`, and the derived summaries
`cluster_stats_audit3.json`, `error_taxonomy_selfdiag.json`.

Superseded runs kept for the record: `phaseB_adapter_causal_{1.7B,8B}_0.6B_seed0.json`
are the 10-epoch causality runs; the paper uses the 20-epoch
`phaseB_adapter_causal20_*` files (audit3 §6).

## Known traps

- **`factorial_analysis.json` is mixed.** Its `pair_effects_seed_means` and
  per-sample blocks are likelihood effects and are valid; its `retention` block
  was computed on legacy EM and contradicts the corrected four-arm table
  (it marks 8B→4B V-only as `success=True` at retention 0.875, while the
  corrected EM is 0.44). The paper no longer contains any retention or
  non-inferiority criterion, so the block must not be revived.
- **`six_pair_per_sample` is `null` on purpose.** `experiments/phase4_scaling_law_v2.py`
  now dumps per-sample `rows`, but the three committed `phase4_scaling_law_v2_*`
  reports were written before that change, so they carry no rows. Re-running that
  script on the GPU host would populate it; until then the five non-flagship
  interactions have no per-sample interval and are reported as exploratory means
  only.
- **`cluster_stats_audit3.json`**: use `EM_ci95_clustered` (cluster = document,
  8 clusters) for anything inferential. The unclustered `EM_ci95` used to be
  degenerate — it came out exactly equal to the mean — and was fixed in W29.
- **`w4_cca_perhead_extra.json` / `w4_cca_perhead_orig3.json`** are unreferenced
  variants of `w4_cca_perhead.json`; no script or paper text points at them.
- **`phase1_causal_v2_*` and `phase7_second_domain_*` are not cited** anywhere in
  the paper or in `scripts/`; `phaseB_squad.py` supersedes the latter.

## No filename warns you

Nothing in a legacy filename says "invalid EM", which is how the stale
`phase4` figures stayed quotable. A key-name heuristic does not separate the two
generations reliably (`phase1_causal_v2_*` and `phase7_second_domain_*` carry
`EM`/`self_EM` keys but no `exact_match`). Use the producing-script table above.
`scripts/verify_corrected_paper.py`, `paper/audit/verify_audit3_edits.py` and
`paper/audit/verify_audit4_edits.py` read only files from the valid column.
