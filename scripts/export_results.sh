#!/usr/bin/env bash
# =============================================================================
# export_results.sh — Export all benchmark results to CSV, JSON, and figures
# =============================================================================
set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-results}

echo "==> Exporting results from $RESULTS_DIR"

python - <<'PYEOF'
import sys
import os
import json
import csv
from pathlib import Path

results_dir = Path(os.environ.get("RESULTS_DIR", "results"))
tables_dir = results_dir / "tables"
figures_dir = results_dir / "figures"
tables_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

# Export JSONL -> CSV
jsonl_path = tables_dir / "benchmark_results.jsonl"
if jsonl_path.exists():
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except:
                    pass
    if records:
        csv_path = tables_dir / "benchmark_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"  CSV:  {csv_path} ({len(records)} rows)")

        # Summary JSON
        summary_path = tables_dir / "summary.json"
        by_algo = {}
        for r in records:
            algo = r.get("algorithm", "unknown")
            if algo not in by_algo:
                by_algo[algo] = []
            by_algo[algo].append(r)
        summary = {
            algo: {
                "mean_success_rate": sum(r["success_rate"] for r in rs) / len(rs),
                "mean_reward": sum(r["avg_reward"] for r in rs) / len(rs),
                "n_conditions": len(rs),
            }
            for algo, rs in by_algo.items()
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  JSON summary: {summary_path}")
else:
    print("  No benchmark_results.jsonl found — run benchmarks first")

# Try plotting if matplotlib is available
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if 'records' in dir() and records:
        # Success rate by algorithm
        algos = list({r["algorithm"] for r in records})
        fig, ax = plt.subplots(figsize=(8, 5))
        for algo in algos:
            algo_records = [r for r in records if r["algorithm"] == algo]
            noise_levels = [r.get("noise_level", "none") for r in algo_records]
            success_rates = [r["success_rate"] for r in algo_records]
            ax.plot(noise_levels, success_rates, marker="o", label=algo)
        ax.set_xlabel("Noise Level")
        ax.set_ylabel("Task Success Rate")
        ax.set_title("Robustness: Success Rate vs Noise Level")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig_path = figures_dir / "noise_robustness.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure: {fig_path}")
except ImportError:
    print("  matplotlib not installed — skipping figures")

print("\n==> Export complete")
PYEOF
