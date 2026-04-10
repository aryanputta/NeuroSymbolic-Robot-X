#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — Bootstrap the PHYSAI-RL-ROBOT-X development environment
# =============================================================================
set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}
EXTRAS=${EXTRAS:-""}  # set to "dev" to install dev deps

echo "==> PHYSAI-RL-ROBOT-X environment setup"

# --- Python version check
$PYTHON -c "import sys; assert sys.version_info >= (3,10), f'Python 3.10+ required, got {sys.version}'"
echo "    Python OK: $($PYTHON --version)"

# --- Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating virtual environment at $VENV_DIR"
  $PYTHON -m venv "$VENV_DIR"
fi

# --- Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "==> Virtual environment activated"

# --- Upgrade pip
pip install --upgrade pip setuptools wheel

# --- Install project
if [ -n "$EXTRAS" ]; then
  echo "==> Installing package with extras: $EXTRAS"
  pip install -e ".[$EXTRAS]"
else
  echo "==> Installing package (base)"
  pip install -e "."
fi

# --- Create required data directories with .gitkeep
for dir in \
  data/raw data/processed data/trajectories data/failure_logs \
  models/checkpoints results/logs results/figures results/tables results/videos \
  docs/architecture docs/experiments docs/failure_analysis; do
  mkdir -p "$dir"
  touch "$dir/.gitkeep"
done

echo ""
echo "==> Setup complete!"
echo "    Activate with: source $VENV_DIR/bin/activate"
echo "    Train:         physai-train --help"
echo "    Evaluate:      physai-eval --help"
echo "    Benchmark:     physai-benchmark"
echo "    Tests:         pytest tests/ -v"
