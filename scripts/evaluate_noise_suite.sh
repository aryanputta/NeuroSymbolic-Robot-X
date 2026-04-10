#!/usr/bin/env bash
# =============================================================================
# evaluate_noise_suite.sh — Run the full noise robustness benchmark suite
# =============================================================================
set -euo pipefail

ALGORITHM=${ALGORITHM:-ppo}
N_EPISODES=${N_EPISODES:-50}
SEED=${SEED:-42}
RESULTS_DIR=${RESULTS_DIR:-results}
TASKS=${TASKS:-"pick_and_place sort_objects stack_blocks"}

echo "==> Noise robustness evaluation suite"
echo "    Algorithm:  $ALGORITHM"
echo "    Episodes:   $N_EPISODES per condition"
echo "    Tasks:      $TASKS"
echo ""

for TASK in $TASKS; do
  echo "--- Task: $TASK ---"
  for NOISE in none low medium high; do
    echo "  Noise level: $NOISE"
    physai-eval \
      --task "$TASK" \
      --algorithm "$ALGORITHM" \
      --n-episodes "$N_EPISODES" \
      --noise-level "$NOISE" \
      --latency-level none \
      --seed "$SEED" \
      --results-dir "$RESULTS_DIR"
  done
done

echo ""
echo "==> Noise suite complete. Exporting CSV …"
python -c "
from src.evaluation.benchmark import BenchmarkRunner
runner = BenchmarkRunner(results_dir='$RESULTS_DIR', n_episodes=$N_EPISODES)
runner.export_csv()
print('CSV exported.')
print(runner.generate_summary_report())
"

echo "==> Results saved to: $RESULTS_DIR/tables/"
