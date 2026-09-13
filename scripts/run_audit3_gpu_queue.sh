#!/usr/bin/env bash
# audit3 / REVISION_PLAN3 GPU queue (A1 -> A2 -> A3 -> A4).
#
# Runs the four GPU-side items in dependency order and skips a step whose
# output already exists, so the queue can be resumed after an interruption.
# A1 comes first because its pre-registered gate is a stop rule: if the
# corrected evaluator disagrees with the full-prefill reference at the
# distribution level (top-1 agreement below 1 or median KL above 0.1 nats),
# the remaining items must not be interpreted until the evaluator is fixed.
#
# Usage (on the GPU machine, from the repository root):
#   bash scripts/run_audit3_gpu_queue.sh 2>&1 | tee reports/audit3_gpu_queue.log
#
# Notes
# -----
# - `OMP_NUM_THREADS` is pinned because a concurrent session on the same host
#   made numpy/BLAS oversubscribe the CPU in the audit2 round.
# - Every step writes into reports/; commit those JSONs before interpreting.
# - A5 (wide-document split) is optional and deliberately not in this queue.
set -u
cd "$(dirname "$0")/.." || exit 1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
mkdir -p reports

run_step() {
  local name="$1"; shift
  local sentinel="$1"; shift
  if [ -f "$sentinel" ]; then
    echo "== ${name}: SKIP (found ${sentinel}) =="
    return 0
  fi
  echo "== ${name}: START $(date -Is) =="
  if "$@"; then
    echo "== ${name}: DONE $(date -Is) =="
  else
    echo "== ${name}: FAILED $(date -Is) =="
    exit 1
  fi
}

# ---------------------------------------------------------------- A1 --------
# Evaluator residual: distributional metrics (KL / p95 / top-1), the SQuAD
# validator, and the attribution probe.
run_step "A1-evalcheck-synthetic" reports/phaseB_evalcheck_residual_seed0.json \
  python3 experiments/phaseB_evalcheck.py --seed 0 --n-eval 56 --students 0.6B 1.7B 4B \
    --output reports/phaseB_evalcheck_residual_seed0.json

run_step "A1-evalcheck-squad" reports/phaseB_evalcheck_squad.json \
  python3 experiments/phaseB_evalcheck.py --domain squad --n-eval 30 --students 0.6B \
    --output reports/phaseB_evalcheck_squad.json

run_step "A1-attribution" reports/phaseB_evalcheck_attrib.json \
  python3 experiments/phaseB_evalcheck.py --attrib --students 0.6B --n-eval 8 \
    --output reports/phaseB_evalcheck_attrib.json

# ---------------------------------------------------------------- A2 --------
# Causality with the final 20-epoch adapter, three seeds, both pairs, with
# per-sample rows so document-clustered inference is possible.
for S in 0 1 2; do
  run_step "A2-causal20-1.7B-seed${S}" \
    "reports/phaseB_adapter_causal20_1.7B_0.6B_seed${S}.json" \
    python3 experiments/phaseB_adapter.py --pair 1.7B_0.6B --seed "$S" --rank 8 \
      --epochs 20 --conditions joint --causal --n-calib 70 --n-eval 56 \
      --dump-rows \
      --output "reports/phaseB_adapter_causal20_1.7B_0.6B_seed${S}.json"
done

for S in 0 1 2; do
  run_step "A2-causal20-8B-seed${S}" \
    "reports/phaseB_adapter_causal20_8B_0.6B_seed${S}.json" \
    python3 experiments/phaseB_adapter.py --pair 8B_0.6B --seed "$S" --rank 8 \
      --epochs 20 --conditions joint --causal --n-calib 70 --n-eval 56 \
      --dump-rows \
      --output "reports/phaseB_adapter_causal20_8B_0.6B_seed${S}.json"
done

# ---------------------------------------------------------------- A3 --------
# Within-domain repair on the second domain: 15 calibration / 15 held-out
# SQuAD documents, three document splits.
for D in 0 1 2; do
  run_step "A3-squad-within-split${D}" \
    "reports/phaseB_squadwithin_split${D}.json" \
    python3 experiments/phaseB_squad_within.py --split-seed "$D" --n-calib 15 --n-eval 15 \
      --v-mappers affine,outaware,wo --adapter --epochs 20 --rank 8 \
      --output "reports/phaseB_squadwithin_split${D}.json"
done

# ---------------------------------------------------------------- A4 --------
# State-space vs consumption-space error budget, both pairs, three seeds.
for S in 0 1 2; do
  run_step "A4-errorbudget-1.7B-seed${S}" \
    "reports/phaseB_errorbudget_1.7B_0.6B_seed${S}.json" \
    python3 experiments/phaseB_errorbudget.py --pair 1.7B_0.6B --seed "$S" \
      --n-calib 70 --n-eval 56 \
      --output "reports/phaseB_errorbudget_1.7B_0.6B_seed${S}.json"
done

for S in 0 1 2; do
  run_step "A4-errorbudget-8B-seed${S}" \
    "reports/phaseB_errorbudget_8B_0.6B_seed${S}.json" \
    python3 experiments/phaseB_errorbudget.py --pair 8B_0.6B --seed "$S" \
      --n-calib 70 --n-eval 56 \
      --output "reports/phaseB_errorbudget_8B_0.6B_seed${S}.json"
done

echo "== audit3 GPU queue finished $(date -Is) =="
echo "Next: git add reports/ && git commit, then on the editing machine:"
echo "  python scripts/cluster_stats_audit3.py"
echo "  python paper/audit/verify_audit3_edits.py"
