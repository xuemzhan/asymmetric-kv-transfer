# Asymmetric KV Transfer

Research artifact for the paper

> **Beyond Representation Fidelity: Functional Compatibility for Cross-Model KV Transfer**
> Zhang Xuemin, Zhong Liheng, Hu Liangbin, Yu Junpeng, Yi Kun

The paper is built from [`paper/main.tex`](paper/main.tex) in English and
Chinese; see [Building the paper](#building-the-paper). Compiled PDFs are not
committed, so build them locally. arXiv: _not posted yet._

---

## What this repository is

A KV cache is the state a language model builds while reading a document. Reusing
one model's cache in a different model would let you skip re-reading the document
when you upgrade a model, swap a large model for a small one, or wire several
models into a pipeline. A line of work makes that reuse work by fitting a linear
map between the two models' caches and reporting that it succeeds.

This repository is the artifact for a paper that asks a narrower question: does
the receiving model actually **use** the handed-off state? Our answer is that
measuring the distance between the mapped cache and the receiver's own cache is
the wrong test, and that several published evaluation shortcuts pass that test
while the receiver still cannot answer a single question.

The repository contains the paper (English and Chinese), every experiment script,
the raw result JSONs behind every number in the paper, the figures, and a set of
scripts that fail loudly if the paper and the results ever disagree.

**Scope.** All experiments are inside the Qwen3 family (0.6B–8B), with six
teacher→student pairs, three seeds, and 56 held-out questions per seed.

---

## What we find

1. **Task performance does not survive the handoff.** Under a corrected evaluator,
   no pair averages more than 0.149 exact match for keys or 0.065 for the joint
   arm over three seeds, against 0.44–0.90 for the student reading its own cache.
   On 8B→0.6B all three transfer arms are exactly 0.000.

2. **Teacher-forced likelihood is not a transfer metric.** Three caches that
   contain no usable content — a *different document's* cache, a moment-matched
   random Gaussian key, and an all-zero cache — reproduce 89–105% of the
   likelihood gain that a real teacher cache produces, while each yields exact
   match ≈ 0. A likelihood-only protocol cannot tell these apart from success.

3. **The measurement that does predict transfer is taken where the receiver
   consumes the state.** Sweeping a one-parameter mapper objective over 198
   configurations, raw state-space error is *positively* related to task score
   inside every run (ρ +0.89 to +0.99): the mapper that reconstructs the state
   best answers the fewest questions. Error measured after the receiver's
   attention output, or after its output projection, is strongly negatively
   related (ρ −0.93 to −0.99), with bootstrap intervals disjoint from the raw
   interval in every run.

4. **The two failures are different, and this matters for what to fix.** Mapped
   teacher *keys* move the receiver's attention routing (top-1 agreement
   0.59–0.70) while the attention distribution stays close (cosine 0.93–0.97).
   A key mapper that reduces routing divergence by 17–43% still leaves the key
   arm at the floor, so **routing alignment alone is not sufficient**. The
   *value* arm never sees a mapped key, and is limited by how the receiver
   consumes the value; fitting the value mapper in the receiver's consumption
   space raises V-only exact match from 0.113 to 0.90–0.94 on 1.7B→0.6B (a
   shuffled-target control stays at 0.077).

5. **Repairability is real but conditional.** A rank-8 correction of the
   receiver's output projections (~0.7M parameters, receiver and mapper frozen)
   restores K/V/joint exact match to 0.94/0.94/0.83 on 1.7B→0.6B. It does not
   carry to 8B→0.6B (0.79/0.36/0.47), and it does not reappear on a second domain
   even when retrained inside that domain — a calibration set of the *same size*
   from the other domain fails identically, and 85 documents mixing both still
   fail. The paper reads this as evidence that functional compatibility is
   **task-conditioned**, not a property of the model pair.

### Transfer quality, all six pairs (corrected exact match, mean ± std over 3 seeds)

| pair | Self (own cache) | K-only | V-only | Joint |
|---|---|---|---|---|
| 8B→0.6B | 0.899 ± 0.010 | 0.000 | 0.000 | 0.000 |
| 4B→0.6B | 0.899 ± 0.010 | 0.000 | 0.000 | 0.006 ± 0.010 |
| 1.7B→0.6B | 0.899 ± 0.010 | 0.030 ± 0.010 | 0.113 ± 0.052 | 0.000 |
| 8B→1.7B | 0.440 ± 0.010 | 0.000 | 0.000 | 0.012 ± 0.021 |
| 4B→1.7B | 0.440 ± 0.010 | 0.054 ± 0.018 | 0.000 | 0.024 ± 0.010 |
| 8B→4B | 0.815 ± 0.021 | 0.149 ± 0.010 | 0.440 ± 0.010 | 0.065 ± 0.055 |

`Self` is the student reading its own cache and is the ceiling. `K-only`,
`V-only` and `Joint` are the three handoff arms. The single arm above 0.15 is
V-only on the equal-depth 8B→4B pair, a little over half of that pair's Self.

---

## What we do **not** claim

We state these explicitly, because it is easy to read more into the numbers above
than they support.

- **Not** that cross-model cache transfer is impossible. Recent systems that fit
  the mapper to receiver behaviour — by distillation, by a nonlinear map, by a
  trainable correction stage — report near-native quality, and we cite them as
  concurrent work. What we show is that *state reconstruction fidelity* does not
  predict whether the state is usable.
- **Not** that routing compatibility is necessary. We only show that improving
  routing alignment is not sufficient. No experiment here destroys routing
  compatibility in a condition that otherwise transfers, so the converse is
  untested.
- **Not** a general law about model families. One family, four sizes, linear
  mappers, exact match on a synthetic reading-comprehension corpus plus one
  second domain.
- **Not** a proposed method. The rank-8 adapter appears as an *intervention* that
  establishes repairability, not as something we recommend deploying.

---

## Using this repository

There are two tiers. Tier 1 needs no GPU, no model weights and no external
package, and is enough to reproduce every table, figure and check in the paper.
Tier 2 re-runs the experiments themselves.

### Tier 1 — reproduce the paper from the stored results (CPU only)

Requirements: Python 3.10+, `numpy`, `matplotlib`.

Run every command below from the repository root, and substitute `python` for
`python3` if that is what your interpreter is called (Windows usually).

```bash
# Regenerate all eight figures from the tracked result JSONs
python3 paper/figures/gen_paper_figures.py

# Assert every number in the paper against reports/
python3 scripts/verify_corrected_paper.py
python3 paper/audit/verify_audit2_edits.py
python3 paper/audit/verify_audit3_edits.py
python3 paper/audit/verify_audit4_edits.py

# Assert the Chinese edition mirrors the English one exactly
python3 paper/audit/verify_zh_parity.py

# Statistics unit tests
python3 tests/test_stats_utils.py
```

Every number in `paper/main.tex` is transcribed from a file in
[`reports/`](reports/), which is tracked by git for exactly this reason.

### Tier 2 — re-run the experiments (CUDA GPU)

Additional requirements:

- `torch==2.5.1+cu124`, `transformers==4.52.4`, `modelscope==1.39.1`,
  `numpy==1.26.4` — the versions this artifact was run with.
- A CUDA GPU; an RTX 4090 is enough for the 0.6B–8B pairs.
- Qwen3 snapshots for 0.6B, 1.7B, 4B and 8B.
- **The sibling `apcs` package**, which supplies the mapper and RoPE helpers
  (`apcs.mapper.math`, `apcs.rope.runner`). ⚠️ It is **not** vendored here; the
  experiment scripts will fail on import without it. See
  [Known gaps](#known-gaps).

Paths are configurable through environment variables, so nothing is hard-coded to
one machine:

| variable | default | what it points at |
|---|---|---|
| `V3_ROOT` | repository root | base for the two below |
| `V3_DATA_DIR` | `$V3_ROOT/data` | generated data JSONs |
| `V3_REPORT_DIR` | `$V3_ROOT/reports` | experiment result JSONs |
| `V3_MODELS_DIR` | `/root/.cache/modelscope/models` | Qwen3 snapshots |
| `V3_APCS_DIR` | `/workspace/apcs` | the sibling `apcs` package |

The data JSONs are **not** tracked (they are regenerable) and must exist first:

```bash
python3 data/build_ood.py --seeds 0,1,2   # synthetic reading-comprehension domain
python3 data/build_nq.py                  # SQuAD subset (needs network)
```

Then, from the repository root:

```bash
# The main four-arm table (one seed at a time; rerun for 0, 1, 2)
python3 experiments/phaseB_fourarm.py --seed 0 --n-calib 70 --n-eval 56

# Fast smoke test: one small pair, a handful of items
python3 experiments/phaseB_fourarm.py --seed 0 --n-calib 4 --n-eval 2 \
    --pairs 1.7B_0.6B --output /tmp/smoke_fourarm.json

# The full GPU queue (evaluator validation -> causality -> alignment -> error budget),
# resumable, each step writing its own JSON under reports/
bash scripts/run_audit3_gpu_queue.sh
```

### Building the paper

The paper is self-contained LaTeX (no `\input` of section files) and uses vector
PDFs from `paper/figures/`. You need a LaTeX engine; the script is written for
[Tectonic](https://github.com/tectonic-typesetting/tectonic/releases) 0.17.0 but
any toolchain that can run XeTeX will do. The Tectonic binary is **not** committed
(it is a ~20 MB download), so put it at `tools/tectonic/tectonic.exe`, pass
`-TectonicPath`, or install it on `PATH`. Then:

```bash
powershell -File tools/compile_paper.ps1
```

This is the single build entry point. It copies `paper/main.tex` to
`paper/arxiv_submission/main.tex` and asserts the two are byte-identical, then
builds all three editions and prints a page count plus the undefined-reference,
overfull-box, LaTeX-warning and missing-glyph counts for each:

| target | source | output |
|---|---|---|
| English edition | `paper/main.tex` | `paper/main.pdf` |
| Chinese edition | `paper/main_zh.tex` | `paper/main_zh.pdf` |
| arXiv bundle | `paper/arxiv_submission/main.tex` | `paper/arxiv_submission/main.pdf` |

Pass `-SkipArxiv` to build only the first two. Any LaTeX toolchain works; the
Chinese edition needs XeTeX plus a CJK font set (it uses `ctex`). Compiled PDFs
are gitignored — the sources are the deliverable.

---

## Repository layout

```
.
├── paper/                  # the paper
│   ├── main.tex            #   English edition (canonical, self-contained)
│   ├── main_zh.tex         #   Chinese edition, number-for-number mirror
│   ├── arxiv_submission/   #   arXiv bundle (main.tex + figures/)
│   ├── figures/            #   vector PDFs + gen_paper_figures.py
│   └── audit/              #   review rounds, revision plans, consistency guards
├── experiments/            # experiment code
│   ├── phase0_g0.py        #   shared harness: load / capture / mappers / scoring
│   ├── phaseB_common.py    #   corrected evaluator + arbitrary-pair harness
│   └── phaseB_*.py         #   four-arm scan, controls, alignment, adapter, ...
├── data/                   # data builders (JSON outputs are regenerable)
├── reports/                # result JSONs — the provenance for every paper number
├── scripts/                # verification, clustered statistics, GPU queue
├── tests/                  # unit and smoke tests
├── tools/                  # compile_paper.ps1 (bring your own Tectonic)
├── README.md               # this file (for readers)
└── ITERATION_LOG.md        # full session-by-session history (see the warning below)
```

---

## Provenance and correctness

Three rules keep the paper and the data from drifting apart.

**The result JSONs are the source of truth.** Every empirical number in
`paper/main.tex` is transcribed from a file in `reports/`, and
`scripts/verify_corrected_paper.py` plus the three `paper/audit/verify_*_edits.py`
guards assert that they still agree. If you edit the paper or a report, run them.

**An earlier version of this artifact used a broken evaluator, and its numbers are
withdrawn.** Its greedy exact-match metric scored a cache that had already been
advanced through the query *and the gold answer*, and it re-fed the final query
token as the first generation input. Both defects are fixed in
`experiments/phaseB_common.py` (`greedy_answer_fixed` + `score_arm`), and the
corrected evaluator is validated against a full-prefill reference on the whole
Self test set: normalized-EM agreement 0.929–0.982, first-token top-1 agreement
0.929–0.982, median KL 0.001–0.005 nats, largest single logit error 1.344.
[`reports/README.md`](reports/README.md) records which stored reports carry a
valid exact-match field and which predate the fix — **read it before citing
anything from `reports/` directly.**

**The Chinese edition is checked, not trusted.** `paper/audit/verify_zh_parity.py`
asserts that every numeric literal, label, reference, citation, bibliography
entry, figure and table of the English edition reappears in the Chinese one.

---

## Known gaps

- ⚠️ **The `apcs` package is external.** The experiment scripts import mapper and
  RoPE helpers from `apcs` (`V3_APCS_DIR`), which is not part of this repository.
  Tier 1 above does not need it; Tier 2 does. Until it is vendored or published,
  the GPU-side results can be inspected and re-checked but not re-run from a
  clean clone.
- **Small test sets, few clusters.** 56 held-out questions per seed, drawn from 8
  documents, so per-item intervals are wide; the paper reports
  document-clustered intervals wherever a claim depends on one. Treat single-seed
  cells in the table above as descriptive, not as precise estimates.
- **Linear mappers only.** Everything the paper reports is a linear (ridge) map,
  optionally with a one-parameter interpolation. Nonlinear mappers are untested
  here, and concurrent work suggests they change the picture.
- **The key side is not tested under the objective the paper advocates.** The
  symmetric experiment — regressing on the receiver's pre-softmax scores, so that
  `Q_S K̂ᵀ Mᵀ ≈ Q_S K_Sᵀ` — is described in the paper but not run.
- **The flagship causality test is underpowered.** On 8B→0.6B the correct-cache
  margin is +0.095 against a pre-registered 0.10 gate. The paper reports it as
  unresolved rather than lowering the bar.
- **Absolute exact match is not comparable across pairs.** Self is 0.440 for 1.7B
  students and 0.899 for 0.6B students, so every claim in the paper is a
  within-pair comparison; do not read the table above as an absolute ranking.

---

## Appendix: the full history (author-facing)

If you want to see how this work arrived at its current claims — including the
claims it withdrew — read [`ITERATION_LOG.md`](ITERATION_LOG.md), the sole
session-by-session record, and the review rounds in
[`paper/audit/`](paper/audit/):

| file | what it is |
|---|---|
| `paper/audit/audit1-4.md` | earlier review rounds and the revisions they forced |
| `paper/audit/METRIC_CORRECTION.md` | the evaluator defect, the withdrawn numbers, and the mapping from each revised claim to its report |
| `paper/audit/audit5_external_2026-11.md` | literature re-audit two months after the first draft |
| `paper/audit/audit6.md` | the review that fixed the final title and claim scope |
| `paper/audit/response_to_audit6.md` | what was changed in response, and what was not |

⚠️ **These documents contain superseded numbers and superseded claims.** They are
kept as history. Where they disagree with `paper/main.tex`, the paper wins.

---

## Citation

```bibtex
@misc{zhang2026fidelity,
  title  = {Beyond Representation Fidelity: Functional Compatibility for
            Cross-Model KV Transfer},
  author = {Zhang, Xuemin and Zhong, Liheng and Hu, Liangbin and Yu, Junpeng
            and Yi, Kun},
  year   = {2026},
  note   = {Manuscript; arXiv identifier to be added}
}
```

Please cite the paper rather than this repository. If you use the code or the
result JSONs, say so explicitly and point at the commit you used.

## License

This repository carries **no license file yet**, which means the default
all-rights-reserved terms apply. If you want to reuse the code, the figures or the
result JSONs, contact the authors first. The license chosen for the paper when it
is posted to arXiv will be recorded here.
