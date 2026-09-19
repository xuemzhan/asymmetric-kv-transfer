# arXiv Submission Metadata

Prepared 2026-09-12; metadata and abstract re-synced with `paper/main.tex` on 2026-09-19.

## Files to upload

```
main.tex
figures/fig_setting.pdf
figures/fig_evalcheck.pdf
figures/fig_fourarm.pdf
figures/fig_controls.pdf
figures/fig_mappers.pdf
figures/fig_adapter.pdf
figures/fig_layers.pdf
```

No `.bbl` is required: the bibliography is inline (`thebibliography`, 18
entries). The bundle compiles standalone from an empty directory with no other
repository files present (re-verified in W29 with Tectonic 0.17.0: 0 errors, 0
undefined references).

## Title

Cross-Model KV Transfer Needs Functional Compatibility: Why Representation
Alignment Is Not Enough

## Authors

**BLOCKING.** arXiv does not accept anonymous submissions. The bundle currently
reads `Anonymous Author(s)` in two places:

- `main.tex` line 40: `\author{Anonymous Author(s) \\ \texttt{anonymous@example.com}}`
- `main.tex` line 31: `pdfauthor={Anonymous Author(s)}`

Supply the real author list (names, affiliations, emails, order) and both will
be updated.

## Categories

- Primary: `cs.LG`
- Cross-list: `cs.CL`, `cs.AI`

## Comments field

Suggested: `24 pages, 9 tables, 7 figures. Code and result summaries accompanying the paper.`

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

The evaluation protocol we used in earlier versions of this work overstates
transfer. Its greedy exact-match (EM) metric is computed on a cache that has
already been advanced through the query and the gold answer, and it re-feeds the
final query token as the first generation input; both defects are in that
implementation, and we validate the corrected evaluator against a full-prefill
reference on the whole Self test set (normalized-EM agreement 0.929-0.982;
first-token top-1 agreement 0.929-0.982; median KL 0.001-0.005 nats). The
residual is a low-precision attention-kernel path difference rather than a defect
in cache construction or position handling: a cache round-trip and an explicit
position offset each reproduce the reference bit for bit, and the same residual
appears under eager attention. The validator repeats on SQuAD (top-1 agreement
0.933). Under that evaluator, no pair averages more than 0.149 EM for keys or
0.065 for the joint arm over three seeds (largest single seed 0.161), against
0.44-0.90 for the student's own cache; on the flagship 8B->0.6B pair all three
teacher arms are 0.000, and V-only exceeds 0.15 on one pair only (8B->4B, 0.440
against a Self of 0.815). Teacher-forced log-likelihood, which earlier work
reports as transfer, is not a transfer metric: over three seeds a wrong document
(+2.61 vs +2.62), a moment-matched random Gaussian key (+2.33), and an all-zero
cache (+2.75) reproduce 89-105% of the K-only gain while each yields EM ~ 0,
against 0.899 +/- 0.010 for the student's own cache. A control that breaks the
key-value correspondence instead of permuting both drives EM to 0.000, so the
evaluator does detect content corruption. SQuAD replicates the negative result
(Self 0.367; K-only, V-only, and Joint 0.000; K-only dLL +5.24).

The failure lies in how the receiving model reads the state. Under a mapper fit in
representation space, layer alignment looks irrelevant: scrambling the layer map
costs nothing. Under a mapper fit in consumption space the same scrambling
collapses V-only EM from 0.91 to 0.11-0.23, so the earlier null result was a floor
effect of the mapper and alignment does matter once the consumer reads the state
correctly. Two failures are kept apart below: under mapped teacher keys the
student's top-1 attention agreement with its own routing is 0.59-0.70 (a key-side
diagnostic), while the value arm, which never sees a mapped key, is limited by how
the receiver consumes it. Fitting the value mapper where the student reads value
raises V-only EM from 0.113 to 0.90-0.94 (1.7B->0.6B) and from 0.000 to 0.25-0.27
(8B->0.6B), while a shuffled-target mapper stays at 0.08/0.07. Raw representation
error does not rank these variants (rank correlation -0.10 pooled over five
variants and six runs, per-run range [-0.60,-0.10]), whereas error measured after
the student's attention output and after its o-projection does (-1.00 and -0.80,
range [-1.00,-0.80] in every run). A rank-8 correction of the student's output
projections (~0.7M parameters), trained by next-token cross-entropy on calibration
answers under injected teacher KV, restores K/V/Joint EM to 0.94/0.94/0.83
(1.7B->0.6B; Joint spread 0.55-0.96 across seeds) and 0.79/0.36/0.47 (8B->0.6B),
measured against an adapted Self of 0.940 on the flagship pair rather than the
unadapted 0.899. Holding the final adapter fixed and destroying the injected
content drops EM to the floor on 1.7B->0.6B, where correct teacher KV answers
0.905 +/- 0.083 of questions against at most 0.179 for wrong-document, random, and
zero caches (three seeds); on 8B->0.6B the same margin is +0.095 on average, which
our pre-registered gate does not let us separate from no margin, and we report it
as unresolved. The same adapter trained on student states stays much lower
(0.20/0.35/0.18 and 0.11/0.11/0.12), and one trained on shuffled-document teacher
states is intermediate (Joint 0.46 and 0.13). The repair does not transfer across
domains, and on SQuAD it does not reappear within one either: retrained on SQuAD
calibration documents, every transfer arm still answers 0.000 of the held-out
questions across three document splits, so what the synthetic domain repairs is
bound to that task distribution; we name this task-conditioned functional
compatibility. All experiments are within a single model family, and the adapter
is trained on the task distribution it is evaluated on.

## Pre-submission checklist

- [ ] Real author list substituted (2 places; see above)
- [ ] License selected
- [ ] Repository URL added, or the omission deliberately accepted
- [ ] Endorsement for the primary category, if required
