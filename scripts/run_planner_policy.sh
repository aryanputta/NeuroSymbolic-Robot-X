#!/usr/bin/env bash
# =============================================================================
# run_planner_policy.sh — Run the hybrid planner+RL policy on a task
# =============================================================================
set -euo pipefail

TASK=${TASK:-pick_and_place}
ALGORITHM=${ALGORITHM:-ppo}
PLANNER=${PLANNER:-mock}           # mock | llm | vla
CHECKPOINT=${CHECKPOINT:-""}
N_EPISODES=${N_EPISODES:-10}
NOISE_LEVEL=${NOISE_LEVEL:-none}   # none | low | medium | high
LATENCY_LEVEL=${LATENCY_LEVEL:-none}
SEED=${SEED:-42}
RESULTS_DIR=${RESULTS_DIR:-results}

echo "==> Hybrid planner+RL policy evaluation"
echo "    Task:         $TASK"
echo "    Algorithm:    $ALGORITHM"
echo "    Planner:      $PLANNER"
echo "    Noise level:  $NOISE_LEVEL"
echo "    Latency:      $LATENCY_LEVEL"
echo "    Episodes:     $N_EPISODES"
echo ""

physai-eval \
  --task "$TASK" \
  --algorithm "$ALGORITHM" \
  --n-episodes "$N_EPISODES" \
  --noise-level "$NOISE_LEVEL" \
  --latency-level "$LATENCY_LEVEL" \
  --seed "$SEED" \
  --results-dir "$RESULTS_DIR" \
  ${CHECKPOINT:+--checkpoint "$CHECKPOINT"}

echo "==> Evaluation complete. Results in: $RESULTS_DIR/tables/"
