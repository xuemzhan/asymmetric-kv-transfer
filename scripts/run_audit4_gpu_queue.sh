#!/usr/bin/env bash
# audit4 / REVISION_PLAN4 GPU queue (A1 -> A2 -> A3).
#
# Runs the three GPU-side items in ROI order and skips a step whose output
# already exists, so the queue can be resumed after an interruption.
#
# Usage (on the GPU machine, from the repository root):
#   bash scripts/run_audit4_gpu_queue.sh 2>&1 | tee reports/audit4_gpu_queue.log
#
# Pre-registered gates live in paper/audit/REVISION_PLAN4.md PART II; results
# (reports/*.json) must be committed before any prose change.
#
# Cost reference (W26, RTX 4090): A2-causal20 ~13 min/run, A4-errorbudget
# ~6.3 min/run. A2 sweeps 33 mapper configurations per run, so its wall time
# is dominated by the per-config V-only EM generation.
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
# Routing-aware K mapper: K-side intervention, 2 pairs x 3 seeds.
for P in 1.7B_0.6B 8B_0.6B; do
  for S in 0 1 2; do
    run_step "A1-routingkey-${P}-seed${S}" \
      "reports/phaseB_routingkey_${P}_seed${S}.json" \
      python3 experiments/phaseB_routingkey.py --pair "$P" --seed "$S" \
        --n-calib 70 --n-eval 56 \
        --output "reports/phaseB_routingkey_${P}_seed${S}.json"
  done
done

# ---------------------------------------------------------------- A2 --------
# Mapper objective sweep: alpha in {0..1} x lambda in {1e-4,1e-3,1e-2},
# 2 pairs x 3 seeds. This supersedes the five-variant point estimate of A4.
for P in 1.7B_0.6B 8B_0.6B; do
  for S in 0 1 2; do
    run_step "A2-mappersweep-${P}-seed${S}" \
      "reports/phaseB_mappersweep_${P}_seed${S}.json" \
      python3 experiments/phaseB_mappersweep.py --pair "$P" --seed "$S" \
        --n-calib 70 --n-eval 56 \
        --alphas 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
        --lams 1e-4,1e-3,1e-2 \
        --output "reports/phaseB_mappersweep_${P}_seed${S}.json"
  done
done

# Pool the six sweep runs into one correlation table.
run_step "A2-mappersweep-aggregate" \
  "reports/phaseB_mappersweep_aggregate.json" \
  python3 experiments/phaseB_mappersweep.py --aggregate \
    'reports/phaseB_mappersweep_[18]*_seed*.json'

# ---------------------------------------------------------------- A3 --------
# Calibration-volume / distribution control: the C2 (15 synthetic) and C3
# (15 SQuAD + 70 synthetic) cells of REVISION_PLAN4 II.A3. C1 (SQuAD only) is
# the already-archived 3-split result; three document splits are run for C2/C3.
for MIX in synthetic squad+synthetic; do
  for D in 0 1 2; do
    TAG="${MIX/+/_}"
    run_step "A3-squadwithin-${TAG}-split${D}" \
      "reports/phaseB_squadwithin_${TAG}_split${D}.json" \
      python3 experiments/phaseB_squad_within.py --split-seed "$D" \
        --n-calib 15 --n-eval 15 --calib-mix "$MIX" --n-synth 70 \
        --v-mappers affine,outaware,wo --adapter --epochs 20 --rank 8 \
        --output "reports/phaseB_squadwithin_${TAG}_split${D}.json"
  done
done

echo "== audit4 GPU queue finished $(date -Is) =="
echo "Next: git add reports/ && git commit, then on the editing machine:"
echo "  python3 -c \"import json; print(json.load(open('reports/phaseB_mappersweep_aggregate.json'))['rho_pooled'])\""
echo "  python3 paper/audit/verify_audit4_edits.py"
