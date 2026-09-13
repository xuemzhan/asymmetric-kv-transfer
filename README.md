# Asymmetric KV Transfer (V3)

Research artifact for the paper **“Cross-Model KV Transfer Needs a Compatible
Consumer: Diagnosis and a Consumption-Side Fix”** (`paper/main.tex`).

We revisit KV-cache transfer between language models within the Qwen3 family
(0.6B–8B, six teacher–student pairs, three seeds). The artifact contains a
diagnostic **negative result** — under a corrected evaluator, teacher KV does
not transfer task performance — followed by a constructive **consumption-side
fix** that recovers much of it by changing how the receiving model reads the
state.

## Headline findings

- **The greedy-EM metric used by earlier versions of this work was invalid.**
  It scored a cache already advanced through the query and gold answer and
  re-fed the final query token. The corrected evaluator (`experiments/phaseB_common.py`,
  `greedy_answer_fixed` + `score_arm`) agrees with a full-prefill reference on
  the Self test set at normalized-EM 0.929–0.982, with max first-token logit
  error 1.344.
- **Corrected transfer is near zero.** Across pairs, keys peak at 0.149 EM and
  the joint arm at 0.065 EM over three seeds, against 0.44–0.90 for the
  student's own cache; on 8B→0.6B all teacher arms are 0.000.
- **Teacher-forced log-likelihood is not a transfer metric.** A wrong document
  (+2.61 vs +2.62), a moment-matched random Gaussian key (+2.33), and an
  all-zero cache (+2.75) reproduce 89–105% of the K-only LL gain while each
  yields EM ≈ 0.
- **Two separate failures.** Mapped teacher keys perturb routing (top-1
  attention agreement 0.59–0.70); the value arm is limited by how the receiver
  consumes it. Consumption-space (output-aware / W_O-aware) value mappers raise
  V-only EM from 0.113 to 0.90–0.94 (1.7B→0.6B) and from 0.000 to 0.25–0.27
  (8B→0.6B), while a shuffled-target mapper stays at 0.08/0.07.
- **A small adapter restores transfer.** A rank-8 correction of the student's
  output projections (~0.7M params), trained by next-token cross-entropy under
  injected teacher KV, restores K/V/Joint EM to 0.94/0.94/0.83 (1.7B→0.6B) and
  0.79/0.36/0.47 (8B→0.6B).
- **Scope.** Within-Qwen3, synthetic OOD domain plus a SQuAD second domain
  (which replicates the negative result).

## Repository layout

```
.
├── data/                 # Data builders (*.json outputs are regenerable, git-ignored)
│   ├── build_ood.py      #   synthetic OOD domain (entity-cluster splits, hop-1..4)
│   └── build_nq.py       #   SQuAD subset + nq_open novelty probe
├── experiments/          # All experiment scripts (shared stats_utils.py)
│   ├── phase0_g0.py      #   shared harness: load / capture / mappers / scoring
│   ├── phaseB_common.py  #   corrected evaluator + arbitrary-pair harness
│   └── phaseB_*.py       #   controls / alignment / mechanism / adapter / ...
├── tests/                # Unit + smoke tests
├── scripts/              # Verification, analysis, GPU queue
│   ├── verify_corrected_paper.py   # asserts paper numbers against reports/
│   ├── cluster_stats_audit3.py     # document-clustered statistics
│   └── run_audit3_gpu_queue.sh     # resumable A1 -> A4 GPU queue
├── reports/              # Experiment JSON provenance (the number source of truth)
│   └── archive/          #   superseded v1 reports
├── paper/                # Paper sources
│   ├── main.tex          #   canonical, self-contained paper
│   ├── arxiv_submission/ #   arXiv bundle
│   ├── figures/          #   active PDFs + gen_paper_figures.py
│   ├── audit/            #   audit1-3, revision plans, paper-number guards
│   └── archive/          #   superseded v1 drafts
├── tools/                # compile_paper.ps1 (Windows Tectonic wrapper)
├── runs/                 # scratch run logs (git-ignored)
├── ITERATION_LOG.md      # SOLE iteration record (read first)
├── AGENTS.md             # Knowledge base / commands
└── README.md
```

## Environment

- Python 3.10, `torch==2.5.1+cu124`, `transformers==4.52.4`,
  `modelscope==1.39.1`, `numpy==1.26.4` (the versions this artifact was run with).
- A CUDA GPU (an RTX 4090 is enough for the 0.6B–8B pairs).
- Qwen3 model snapshots at
  `/root/.cache/modelscope/models/Qwen--Qwen3-{0.6B,1.7B,4B,8B}/snapshots/master`.
- The mapper/rope math is reused from the sibling `apcs` package
  (`/workspace/apcs`, typically a symlink to `/workspace/KVCache`) via
  `sys.path`; the repository is also symlinked as `/workspace/v3`, which the
  scripts use for absolute data/report paths.

## Data

The data JSONs are **not tracked** (they are regenerable) and must exist before
running experiments:

```bash
# Synthetic OOD domain, all three seeds used by the phaseB pipeline
python3 data/build_ood.py --seeds 0,1,2

# Second domain: SQuAD subset + nq_open novelty probe (needs network)
python3 data/build_nq.py
```

`reports/*.json` **are** tracked: every empirical number in the paper is
transcribed from them and cross-checked by the guards below.

## Quick start

```bash
# Main corrected four-arm table (single seed; rerun for seeds 0/1/2)
python3 experiments/phaseB_fourarm.py --seed 0 --n-calib 70 --n-eval 56

# Small GPU smoke test (fast, loads one small pair)
python3 experiments/phaseB_fourarm.py --seed 0 --n-calib 4 --n-eval 2 \
    --pairs 1.7B_0.6B --output /tmp/smoke_fourarm.json

# Resumable audit3 GPU queue (A1 evaluator check -> A2 causality -> A3 -> A4)
bash scripts/run_audit3_gpu_queue.sh

# Regenerate the paper figures (writes paper/figures/*.pdf)
python3 paper/figures/gen_paper_figures.py

# Unit tests
python3 tests/test_stats_utils.py

# Paper-number guards (run after editing the paper or any report)
python3 scripts/verify_corrected_paper.py
python3 paper/audit/verify_audit2_edits.py
python3 paper/audit/verify_audit3_edits.py
```

All experiment scripts assume the current working directory is the repository
root.

## Building the paper

`paper/main.tex` is self-contained (no `\input` of section files) and uses the
vector PDFs in `paper/figures/`. Build with any LaTeX toolchain, e.g.:

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

On Windows the bundled Tectonic wrapper is `tools/compile_paper.ps1`
(`powershell -File tools/compile_paper.ps1`).

## Reproducibility and provenance

- `ITERATION_LOG.md` is the sole iteration record; `reports/*.json` are the
  provenance for every number that enters `paper/main.tex`.
- `paper/audit/` holds the review rounds (`audit1-3.md`), the corresponding
  revision plans, the metric-correction writeup, and guard scripts.
- Numbers are asserted by `scripts/verify_corrected_paper.py`,
  `paper/audit/verify_audit2_edits.py`, and `paper/audit/verify_audit3_edits.py`.
  The pre-correction guard is kept for history only at
  `scripts/archive/verify_paper_numbers_v1_stale.py` (it targets the superseded
  v1 paper and is expected to fail).

## Status

The paper is an anonymous 14-page draft under review; the author list and
citation will be filled in before posting.
