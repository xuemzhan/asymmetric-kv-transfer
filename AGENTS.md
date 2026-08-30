# V3 PROJECT KNOWLEDGE BASE

**Generated:** 2026-08-29
**Branch:** main

## OVERVIEW

V3 is a research project investigating KV cache transfer between teacher and student models. The core hypothesis is that task advantage lives in content streams (value/residual/MLP computation), not in addressing geometry (key/attention routing).

## STRUCTURE

```
/workspace/v3/
├── data/
│   ├── build_ood.py      # Synthetic OOD domain generator
│   ├── train.json        # Training samples (42)
│   ├── val.json          # Validation samples (14)
│   ├── test.json         # Test samples (14)
│   └── graph.json        # Knowledge graph
├── phase0_g0.py          # Gate G0 experiment: strong mapper ablation
├── ITERATION_LOG.md      # SOLE iteration log (read first every session)
├── reports/              # Experiment output directory
└── runs/                 # Run artifacts
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Data generation | `data/build_ood.py` | Synthetic OOD domain, entity-cluster splits |
| G0 experiment | `phase0_g0.py` | Strong mapper ablation (Affine/Whitened/Procrustes/CCA/RidgePerHead) |
| Project state | `ITERATION_LOG.md` | SOLE source of truth for iteration history |
| Model paths | `phase0_g0.py:54-57` | Qwen3-8B (teacher) / Qwen3-0.6B (student) |
| Mapper math | `/workspace/apcs/apcs/mapper/math.py` | Reused: AffineMapper, CCAMapper, ProcrustesMapper, RidgePerHeadMapper, WhitenedMapper |
| RoPE utils | `/workspace/apcs/apcs/rope/runner.py` | de_rope, _rope_pairs |

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
# Run G0 experiment (single seed)
python3 phase0_g0.py --seed 0 --n-calib 42 --n-eval 14

# Regenerate data
python3 data/build_ood.py

# Check GPU availability
nvidia-smi
```

## NOTES

- Models cached at: `/root/.cache/modelscope/models/Qwen--Qwen3-{0.6B,8B}/snapshots/master`
- Architecture: 8B(L36,H8,D128) / 0.6B(L28,H8,D128) - layers aligned via proportional mapping
- GPU: RTX 4090 24GB
- Dependencies: torch 2.5.1+cu124, transformers 4.52.4, modelscope 1.39.1
