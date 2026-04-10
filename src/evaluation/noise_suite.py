"""
Noise robustness evaluation suite.

Systematically sweeps over noise and latency levels, runs the full
benchmark, and generates robustness curves showing performance degradation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

NOISE_LEVELS = ["none", "low", "medium", "high"]
LATENCY_LEVELS = ["none", "low", "medium", "high"]


class NoiseSuite:
    """
    Runs a factorial sweep over noise × latency conditions.

    For each condition, delegates to :class:`BenchmarkRunner` and
    collects results into a structured matrix for plotting.

    Args:
        runner:        BenchmarkRunner instance.
        noise_levels:  Subset of noise levels to evaluate.
        latency_levels: Subset of latency levels to evaluate.
    """

    def __init__(
        self,
        runner,
        noise_levels: list[str] | None = None,
        latency_levels: list[str] | None = None,
    ) -> None:
        self._runner = runner
        self._noise_levels = noise_levels or NOISE_LEVELS
        self._latency_levels = latency_levels or ["none"]

    def run_noise_sweep(
        self,
        env_factory_fn,
        executor_factory,
        task_factory,
        algorithm: str,
        task_name: str,
    ) -> dict[str, Any]:
        """
        Sweep over all noise levels (fixed latency=none).

        Returns a dict mapping noise_level -> BenchmarkResult.
        """
        results = {}
        for nl in self._noise_levels:
            def _env_factory(noise=nl):
                return env_factory_fn(noise, "none")
            result = self._runner.run_task_benchmark(
                _env_factory, executor_factory, task_factory,
                algorithm=algorithm, task_name=task_name,
                noise_level=nl, latency_level="none",
            )
            results[nl] = result
            logger.info("Noise=%s: success_rate=%.3f", nl, result.success_rate)
        return results

    def run_latency_sweep(
        self,
        env_factory_fn,
        executor_factory,
        task_factory,
        algorithm: str,
        task_name: str,
    ) -> dict[str, Any]:
        """Sweep over all latency levels (fixed noise=none)."""
        results = {}
        for ll in self._latency_levels:
            def _env_factory(latency=ll):
                return env_factory_fn("none", latency)
            result = self._runner.run_task_benchmark(
                _env_factory, executor_factory, task_factory,
                algorithm=algorithm, task_name=task_name,
                noise_level="none", latency_level=ll,
            )
            results[ll] = result
            logger.info("Latency=%s: success_rate=%.3f", ll, result.success_rate)
        return results

    def run_full_factorial(
        self,
        env_factory_fn,
        executor_factory,
        task_factory,
        algorithm: str,
        task_name: str,
    ) -> dict[tuple[str, str], Any]:
        """Full noise × latency factorial sweep."""
        results = {}
        for nl in self._noise_levels:
            for ll in self._latency_levels:
                def _env_factory(n=nl, l=ll):
                    return env_factory_fn(n, l)
                result = self._runner.run_task_benchmark(
                    _env_factory, executor_factory, task_factory,
                    algorithm=algorithm, task_name=task_name,
                    noise_level=nl, latency_level=ll,
                )
                results[(nl, ll)] = result
        return results

    def generate_robustness_matrix(
        self,
        results: dict[str, Any],
        metric: str = "success_rate",
    ) -> dict:
        """Convert sweep results to a matrix dict for plotting."""
        levels = list(results.keys())
        values = [getattr(results[l], metric, 0.0) for l in levels]
        return {
            "levels": levels,
            "values": values,
            "metric": metric,
            "degradation_pct": self._compute_degradation(values),
        }

    @staticmethod
    def _compute_degradation(values: list[float]) -> float:
        """Percentage degradation from 'none' to worst level."""
        if not values or values[0] == 0:
            return 0.0
        return round((values[0] - min(values)) / max(values[0], 1e-6) * 100, 2)

    def save_robustness_report(
        self,
        noise_results: dict,
        output_path: str = "results/tables/robustness_report.json",
    ) -> None:
        """Save robustness matrix as JSON for downstream plotting."""
        report = {}
        for metric in ("success_rate", "avg_reward", "avg_collision_count"):
            report[metric] = self.generate_robustness_matrix(noise_results, metric)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Robustness report saved to '%s'", output_path)
