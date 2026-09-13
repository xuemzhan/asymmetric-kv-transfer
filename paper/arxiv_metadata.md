# arXiv Submission Metadata

Prepared 2026-09-12 for the bundle in `paper/arxiv_submission/`.

## Files to upload

```
main.tex
figures/fig_setting.pdf
figures/fig_fourarm.pdf
figures/fig_controls.pdf
figures/fig_mappers.pdf
figures/fig_adapter.pdf
figures/fig_layers.pdf
```

No `.bbl` is required: the bibliography is inline (`thebibliography`, 18
entries). The bundle compiles standalone from an empty directory with no other
repository files present (verified with Tectonic 0.17.0; 0 errors, 0 undefined
references).

## Title

Cross-Model KV Transfer Needs Functional Compatibility: Why Representation
Alignment Is Not Enough

## Authors

**BLOCKING.** arXiv does not accept anonymous submissions. The bundle currently
reads `Anonymous Author(s)` in two places:

- `main.tex` line 39: `\author{Anonymous Author(s) \\ \texttt{anonymous@example.com}}`
- `main.tex` line 30: `pdfauthor={Anonymous Author(s)}`

Supply the real author list (names, affiliations, emails, order) and both will
be updated.

## Categories

- Primary: `cs.LG`
- Cross-list: `cs.CL`, `cs.AI`

## Comments field

Suggested: `16 pages, 5 tables, 6 figures. Code and result summaries accompanying the paper.`

**OPEN:** no repository URL is cited anywhere in the paper. The git remote for
this working copy is `https://github.com/xuemzhan/asymmetric-kv-transfer`. Add
it (or a DOI/Zenodo mirror) to the Comments field and to the Reproducibility
statement only if you intend to make it public.

## License

**NEEDS INPUT.** Choose one, e.g. arXiv perpetual non-exclusive license to
distribute 1.0.

## Endorsement

If this is a first submission to `cs.LG`, arXiv endorsement may be required.
## Abstract

Transferring key-value (KV) caches between language models promises to avoid
recomputation when a model is upgraded or an ensemble is assembled. We revisit
this claim within the Qwen3 family (0.6B-8B; six teacher-student pairs; three
seeds) and report a diagnostic negative result followed by a constructive fix.

The standard protocol overstates transfer. Its greedy exact-match (EM) metric is
computed on a cache that has already been advanced through the query and the gold
answer, and it re-feeds the final query token as the first generation input; we
confirm both defects in the published evaluation code. Under a corrected
evaluator, K-only reaches at most 0.149 EM and Joint at most 0.065, against
0.44-0.90 for the student's own cache; on the flagship 8B->0.6B pair all three
teacher arms are 0.000, and V-only exceeds 0.15 on one pair only (8B->4B, 0.440
against a Self of 0.815). Teacher-forced log-likelihood, which earlier work
reports as transfer, is not a transfer metric: over three seeds a wrong document
(+2.61 vs +2.62), a moment-matched random Gaussian key (+2.33), and an all-zero
cache (+2.75) reproduce 89-105% of the K-only gain while each yields EM ~ 0; a
token-shuffled student cache keeps EM at 0.89 +/- 0.04. SQuAD replicates the
negative result (Self 0.367; K-only, V-only, and Joint 0.000; K-only dLL +5.24).

The failure is consumer-side. Offsetting or randomly permuting the layer map does
not reduce transfer, so layer alignment is not the controlling variable; under
mapped teacher keys the student's top-1 attention agreement with its own routing
is 0.59-0.70. Fitting the value mapper where the student reads value raises
V-only EM from 0.113 to 0.90-0.94 (1.7B->0.6B) and from 0.000 to 0.25-0.27
(8B->0.6B), while a shuffled-target mapper stays at 0.08/0.07. A rank-8
correction of the student's output projections (~0.7M parameters), trained only
on calibration teacher states, restores K/V/Joint EM to 0.94/0.94/0.83
(1.7B->0.6B; Joint spread 0.55-0.96 across seeds) and 0.79/0.36/0.47
(8B->0.6B). The same adapter trained on student states stays much lower
(0.20/0.35/0.18 and 0.11/0.11/0.12), and one trained on shuffled-document
teacher states is intermediate (Joint 0.46 and 0.13). All experiments are within
a single model family, and the adapter is trained on the task distribution it is
evaluated on.

## Pre-submission checklist

- [ ] Real author list substituted (2 places; see above)
- [ ] License selected
- [ ] Repository URL added, or the omission deliberately accepted
- [ ] Endorsement for the primary category, if required
