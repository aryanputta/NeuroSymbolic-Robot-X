#!/usr/bin/env bash
# =============================================================================
# train_rl_baseline.sh — Train the RL-only baseline agent
# =============================================================================
set -euo pipefail

TASK=${TASK:-pick_and_place}
ALGORITHM=${ALGORITHM:-ppo}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1000000}
N_ENVS=${N_ENVS:-4}
SEED=${SEED:-42}
LOG_DIR=${LOG_DIR:-results/logs}

echo "==> Training RL-only baseline"
echo "    Task:       $TASK"
echo "    Algorithm:  $ALGORITHM"
echo "    Timesteps:  $TOTAL_TIMESTEPS"
echo "    Envs:       $N_ENVS"
echo "    Seed:       $SEED"
echo ""

physai-train \
  --task "$TASK" \
  --algorithm "$ALGORITHM" \
  --mode rl_only \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --n-envs "$N_ENVS" \
  --seed "$SEED" \
  --log-dir "$LOG_DIR"

echo "==> RL baseline training complete"
echo "    Checkpoints saved to: models/checkpoints/"
echo "    Logs saved to:        $LOG_DIR"
