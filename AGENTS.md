# V3 PROJECT KNOWLEDGE BASE

**Generated:** 2026-08-29
**Branch:** main

## OVERVIEW

V3 is a research project investigating KV cache transfer between teacher and student models. The core hypothesis is that task advantage lives in content streams (value/residual/MLP computation), not in addressing geometry (key/attention routing).

## STRUCTURE

```
/workspace/asymmetric-kv-transfer/   (symlinked as /workspace/v3)
├── data/                # Data builders (build_ood.py, build_nq.py); *.json are regenerable
├── experiments/         # All experiment scripts + shared stats_utils.py
│   ├── phase0_g0.py     # Shared harness: load/capture/mapper/scoring
│   ├── phaseB_common.py # Corrected evaluator + arbitrary-pair harness
│   └── phaseB_*.py      # Review-response experiments (controls/alignment/...)
├── tests/               # Unit + smoke tests (test_stats_utils.py, test_w2_baseline_fix.py)
├── scripts/             # Verification, analysis, GPU queue
├── reports/             # Experiment JSON provenance (archive/ holds superseded v1)
├── paper/               # Paper sources
│   ├── main.tex         # Canonical paper (self-contained)
│   ├── arxiv_submission/# arXiv bundle (main.tex + figures/)
│   ├── figures/         # Active vector PDFs + gen_paper_figures.py
│   ├── audit/           # audit1-3, revision plans, guards
│   └── archive/         # Superseded v1 drafts/figures
├── ITERATION_LOG.md     # SOLE iteration log (read first every session)
└── AGENTS.md
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Data generation | `data/build_ood.py` | Synthetic OOD domain, entity-cluster splits |
| Shared harness | `experiments/phase0_g0.py` | load_model / capture_kv / mappers / scoring |
| Corrected evaluator | `experiments/phaseB_common.py` | `greedy_answer_fixed` + `score_arm` (legacy EM is a bug) |
| Main 4-arm table | `experiments/phaseB_fourarm.py` | 6 pairs × 3 seeds |
| Project state | `ITERATION_LOG.md` | SOLE source of truth for iteration history |
| Model paths | `experiments/phaseB_common.py:42-60` | Qwen3 8B/4B/1.7B/0.6B, PAIRS map |
| Mapper math | `/workspace/apcs/apcs/mapper/math.py` | Reused: AffineMapper, CCAMapper, ProcrustesMapper, RidgePerHeadMapper, WhitenedMapper |
| RoPE utils | `/workspace/apcs/apcs/rope/runner.py` | de_rope, _rope_pairs |
| Figure generation | `paper/figures/gen_paper_figures.py` | Produces the 7 figures used in `main.tex` |
| Number guard | `scripts/verify_corrected_paper.py` | Asserts paper numbers against `reports/` |

## CONVENTIONS

- **ITERATION_LOG.md is SOLE iteration record** - never scatter conclusions in chat
- **Read ITERATION_LOG.md first** every new session
- **Gate decisions are pre-registered** - no post-hoc threshold lowering
- **K/V reported separately** - never average to hide single-arm failure
- **3 independent seeds** required for statistical claims
- **Real task metrics only** - KV cosine is NOT a substitute for task score

## ANTI-PATTERNS (THIS PROJECT)

- ❌ Using KV cosine as retention metric (forbidden by methodological discipline)
- ❌ Post-hoc Gate threshold adjustment
- ❌ Averaging K/V results to hide single-arm failure
- ❌ Skipping seed replication
- ❌ Claims without CI95 confidence intervals

## COMMANDS

```bash
# Main corrected 4-arm table (single seed)
python3 experiments/phaseB_fourarm.py --seed 0 --n-calib 70 --n-eval 56

# Regenerate data
python3 data/build_ood.py

# GPU audit3 queue (A1 -> A4, resumable)
bash scripts/run_audit3_gpu_queue.sh

# Paper-number guards (run after any paper/report edit)
python3 scripts/verify_corrected_paper.py
python3 paper/audit/verify_audit3_edits.py

# Unit tests
python3 tests/test_stats_utils.py

# Check GPU availability
nvidia-smi
```

## NOTES

- All experiment scripts assume CWD = repository root and reuse `/workspace/apcs` via `sys.path`.
- Models cached at: `/root/.cache/modelscope/models/Qwen--Qwen3-{0.6B,1.7B,4B,8B}/snapshots/master`
- Architecture: 8B/4B (L36,H8,D128) / 1.7B/0.6B (L28,H8,D128); layers aligned via proportional mapping
- GPU: RTX 4090 24GB
- Dependencies: torch 2.5.1+cu124, transformers 4.52.4, modelscope 1.39.1
