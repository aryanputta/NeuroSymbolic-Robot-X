"""
Metrics computation for benchmarking the hybrid robot system.

Covers primary metrics (success rate, completion time, collision count,
recovery rate, replan count) and secondary metrics (path efficiency,
policy smoothness, inference latency).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EpisodeMetrics:
    """Per-episode metrics record."""
    episode_id: str
    task_name: str
    algorithm: str
    success: bool
    total_steps: int
    total_reward: float
    completion_time: float
    collision_count: int
    replan_count: int
    recovery_success: bool
    planner_latency_ms: float = 0.0
    control_loop_hz: float = 0.0
    path_length: float = 0.0
    path_efficiency: float = 0.0
    action_smoothness: float = 0.0
    noise_level: str = "none"
    latency_level: str = "none"


@dataclass
class BenchmarkResult:
    """Aggregated metrics over N episodes."""
    task_name: str
    algorithm: str
    n_episodes: int
    noise_level: str = "none"
    latency_level: str = "none"

    # Primary metrics
    success_rate: float = 0.0
    avg_completion_time: float = 0.0
    avg_collision_count: float = 0.0
    recovery_success_rate: float = 0.0
    avg_replan_count: float = 0.0
    avg_reward: float = 0.0

    # Secondary metrics
    avg_path_efficiency: float = 0.0
    avg_action_smoothness: float = 0.0
    avg_planner_latency_ms: float = 0.0
    avg_control_hz: float = 0.0

    # Distribution stats
    reward_std: float = 0.0
    success_rate_95ci: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "algorithm": self.algorithm,
            "n_episodes": self.n_episodes,
            "noise_level": self.noise_level,
            "latency_level": self.latency_level,
            "success_rate": round(self.success_rate, 4),
            "avg_completion_time": round(self.avg_completion_time, 3),
            "avg_collision_count": round(self.avg_collision_count, 3),
            "recovery_success_rate": round(self.recovery_success_rate, 4),
            "avg_replan_count": round(self.avg_replan_count, 3),
            "avg_reward": round(self.avg_reward, 4),
            "avg_path_efficiency": round(self.avg_path_efficiency, 4),
            "avg_action_smoothness": round(self.avg_action_smoothness, 4),
            "avg_planner_latency_ms": round(self.avg_planner_latency_ms, 2),
            "reward_std": round(self.reward_std, 4),
            "success_rate_95ci": [round(v, 4) for v in self.success_rate_95ci],
        }


class MetricsComputer:
    """
    Computes and aggregates episode metrics into benchmark results.

    Also provides specialised metrics for:
    - Policy smoothness (mean absolute action change between steps).
    - Path efficiency (optimal / actual path length ratio).
    - Planner latency distribution (percentiles).
    """

    def aggregate(self, episodes: list[EpisodeMetrics]) -> BenchmarkResult:
        """Aggregate a list of episode metrics into a BenchmarkResult."""
        if not episodes:
            return BenchmarkResult(task_name="", algorithm="", n_episodes=0)

        ep = episodes[0]
        n = len(episodes)

        successes = [e.success for e in episodes]
        rewards = [e.total_reward for e in episodes]

        sr = np.mean(successes)
        ci_lo, ci_hi = self._wilson_ci(int(sum(successes)), n)

        recovery_eps = [e for e in episodes if e.replan_count > 0]
        recovery_sr = np.mean([e.recovery_success for e in recovery_eps]) if recovery_eps else 0.0

        return BenchmarkResult(
            task_name=ep.task_name,
            algorithm=ep.algorithm,
            n_episodes=n,
            noise_level=ep.noise_level,
            latency_level=ep.latency_level,
            success_rate=float(sr),
            avg_completion_time=float(np.mean([e.completion_time for e in episodes])),
            avg_collision_count=float(np.mean([e.collision_count for e in episodes])),
            recovery_success_rate=float(recovery_sr),
            avg_replan_count=float(np.mean([e.replan_count for e in episodes])),
            avg_reward=float(np.mean(rewards)),
            reward_std=float(np.std(rewards)),
            avg_path_efficiency=float(np.mean([e.path_efficiency for e in episodes])),
            avg_action_smoothness=float(np.mean([e.action_smoothness for e in episodes])),
            avg_planner_latency_ms=float(np.mean([e.planner_latency_ms for e in episodes])),
            avg_control_hz=float(np.mean([e.control_loop_hz for e in episodes if e.control_loop_hz > 0])),
            success_rate_95ci=(float(ci_lo), float(ci_hi)),
        )

    @staticmethod
    def compute_action_smoothness(actions: list[np.ndarray]) -> float:
        """Mean absolute difference between consecutive actions (lower = smoother)."""
        if len(actions) < 2:
            return 0.0
        diffs = [float(np.mean(np.abs(actions[i+1] - actions[i]))) for i in range(len(actions) - 1)]
        return float(np.mean(diffs))

    @staticmethod
    def compute_path_efficiency(
        ee_trajectory: list[np.ndarray],
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
    ) -> float:
        """
        Ratio of straight-line distance to actual path length.

        1.0 = perfectly straight; < 1.0 = detours taken.
        """
        if len(ee_trajectory) < 2:
            return 0.0
        optimal = float(np.linalg.norm(goal_pos - start_pos))
        actual = sum(
            float(np.linalg.norm(ee_trajectory[i+1] - ee_trajectory[i]))
            for i in range(len(ee_trajectory) - 1)
        )
        return optimal / max(actual, 1e-6)

    @staticmethod
    def planner_latency_stats(latencies_ms: list[float]) -> dict:
        """Return descriptive stats for planner latency distribution."""
        if not latencies_ms:
            return {}
        arr = np.array(latencies_ms)
        return {
            "mean_ms": float(np.mean(arr)),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(np.max(arr)),
            "min_ms": float(np.min(arr)),
        }

    @staticmethod
    def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
        """Wilson score confidence interval for a proportion."""
        if n == 0:
            return 0.0, 0.0
        p = successes / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return max(0.0, centre - margin), min(1.0, centre + margin)

    def compare_algorithms(
        self,
        results: list[BenchmarkResult],
    ) -> dict[str, Any]:
        """
        Compare multiple BenchmarkResults (one per algorithm).

        Returns a dict keyed by metric name with algorithm rankings.
        """
        comparison: dict[str, Any] = {}
        metrics_to_compare = [
            ("success_rate", True),         # higher is better
            ("avg_completion_time", False),  # lower is better
            ("avg_collision_count", False),
            ("recovery_success_rate", True),
            ("avg_reward", True),
        ]
        for metric, higher_better in metrics_to_compare:
            values = [(r.algorithm, getattr(r, metric)) for r in results]
            values.sort(key=lambda x: x[1], reverse=higher_better)
            comparison[metric] = {
                "ranking": [v[0] for v in values],
                "values": {v[0]: v[1] for v in values},
                "best": values[0][0] if values else None,
            }
        return comparison
