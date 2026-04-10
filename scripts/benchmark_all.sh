#!/usr/bin/env bash
# =============================================================================
# benchmark_all.sh — Run the complete PHYSAI-RL-ROBOT-X benchmark suite
# Evaluates all algorithms × tasks × noise levels × latency levels
# =============================================================================
set -euo pipefail

N_EPISODES=${N_EPISODES:-100}
SEED=${SEED:-42}
RESULTS_DIR=${RESULTS_DIR:-results}

echo "============================================================"
echo "  PHYSAI-RL-ROBOT-X — Full Benchmark Suite"
echo "============================================================"
echo "  Episodes per condition: $N_EPISODES"
echo "  Seed: $SEED"
echo "  Output: $RESULTS_DIR"
echo ""

# Step 1: Train all baselines (skip if checkpoints already exist)
if [ ! -f "models/checkpoints/ppo_pick_and_place_final.zip" ]; then
  echo "==> [1/4] Training RL-only PPO baseline"
  bash scripts/train_rl_baseline.sh
else
  echo "==> [1/4] PPO baseline checkpoint found — skipping training"
fi

# Step 2: Evaluate RL-only baseline across noise levels
echo ""
echo "==> [2/4] Evaluating RL-only baseline (noise sweep)"
ALGORITHM=rl_only_ppo N_EPISODES=$N_EPISODES bash scripts/evaluate_noise_suite.sh

# Step 3: Evaluate hybrid system across noise levels
echo ""
echo "==> [3/4] Evaluating hybrid planner+RL system (noise sweep)"
ALGORITHM=hybrid_ppo N_EPISODES=$N_EPISODES bash scripts/evaluate_noise_suite.sh

# Step 4: Latency sweep for both systems
echo ""
echo "==> [4/4] Latency sensitivity benchmark"
for ALGORITHM in rl_only_ppo hybrid_ppo; do
  for LATENCY in none low medium high; do
    echo "  Algorithm: $ALGORITHM | Latency: $LATENCY"
    physai-eval \
      --task pick_and_place \
      --algorithm "$ALGORITHM" \
      --n-episodes "$N_EPISODES" \
      --noise-level none \
      --latency-level "$LATENCY" \
      --seed "$SEED" \
      --results-dir "$RESULTS_DIR"
  done
done

echo ""
echo "==> All benchmarks complete. Exporting results …"
physai-benchmark
bash scripts/export_results.sh

echo ""
echo "============================================================"
echo "  Benchmark complete!"
echo "  Results: $RESULTS_DIR/tables/benchmark_results.csv"
echo "  Report:  $RESULTS_DIR/benchmark_report.md"
echo "============================================================"
