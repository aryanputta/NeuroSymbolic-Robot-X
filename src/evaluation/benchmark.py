"""
Benchmark runner for PHYSAI-RL-ROBOT-X.

Evaluates one or more algorithms across tasks, noise levels, and latency
levels; outputs tables and figures to ``results/``.

Entry point: ``physai-benchmark`` and ``physai-eval`` CLIs.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from src.diagnostics.metrics import BenchmarkResult, EpisodeMetrics, MetricsComputer
from src.diagnostics.episode_logger import EpisodeLogger
from src.memory.failure_logger import FailureLogger
from src.memory.trajectory_store import EpisodeTrajectory, TrajectoryStore

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runs the full benchmark suite.

    For each (algorithm, task, noise_level, latency_level) combination,
    runs ``n_episodes`` episodes and aggregates metrics.

    Args:
        results_dir: Directory to save tables, figures, and logs.
        n_episodes:  Episodes per condition.
        seed:        Base random seed (each episode uses seed + episode_index).
        save_videos: Whether to save episode rollout videos.
        deterministic: Use deterministic policy predictions.
    """

    def __init__(
        self,
        results_dir: str = "results",
        n_episodes: int = 50,
        seed: int = 42,
        save_videos: bool = False,
        deterministic: bool = True,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.n_episodes = n_episodes
        self.seed = seed
        self.save_videos = save_videos
        self.deterministic = deterministic

        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "tables").mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)
        (self.results_dir / "figures").mkdir(exist_ok=True)
        (self.results_dir / "videos").mkdir(exist_ok=True)

        self._metrics_computer = MetricsComputer()
        self._failure_logger = FailureLogger(log_dir=str(self.results_dir / "logs"))
        self._trajectory_store = TrajectoryStore(base_dir="data/trajectories")

    def run_episode(
        self,
        env,
        executor,
        task,
        episode_idx: int,
        algorithm: str,
        task_name: str,
        noise_level: str = "none",
        latency_level: str = "none",
        ep_logger: EpisodeLogger | None = None,
    ) -> EpisodeMetrics:
        """Run a single evaluation episode and return metrics."""
        episode_id = f"{algorithm}_{task_name}_{episode_idx:04d}_{uuid.uuid4().hex[:6]}"

        if ep_logger:
            ep_logger.begin_episode(episode_id)

        obs, _ = env.reset(seed=self.seed + episode_idx)
        task.reset(env.unwrapped._env if hasattr(env, "unwrapped") else env)
        instruction = task.get_instruction()

        total_reward = 0.0
        collision_count = 0
        replan_count = 0
        actions_log: list[np.ndarray] = []
        ee_trajectory: list[np.ndarray] = []
        start_time = time.perf_counter()
        planner_latencies: list[float] = []

        t0_plan = time.perf_counter()
        terminated = truncated = False

        for step in range(500):
            action, _ = executor.predict(obs, deterministic=self.deterministic)
            actions_log.append(action.copy())

            ee = obs.get("ee_pose", np.zeros(7))[:3]
            ee_trajectory.append(ee.copy())

            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            collision_count += int(info.get("collision", False))
            replan_count += int(info.get("replan", False))
            obs = next_obs

            if ep_logger:
                ep_logger.log_step(step, obs, action, reward, terminated, truncated, info)

            if terminated or truncated:
                break

        completion_time = time.perf_counter() - start_time
        success = task.check_success({}, env.unwrapped._env if hasattr(env, "unwrapped") else env)

        smoothness = MetricsComputer.compute_action_smoothness(actions_log)
        path_eff = 0.0
        if len(ee_trajectory) >= 2:
            path_eff = MetricsComputer.compute_path_efficiency(
                ee_trajectory, ee_trajectory[0], ee_trajectory[-1]
            )

        if ep_logger:
            ep_logger.end_episode(success, total_reward)

        if not success and replan_count == 0:
            self._failure_logger.log(
                episode_id=episode_id,
                task_name=task_name,
                failure_step=step,
                failure_reason="task_not_completed",
                skill_at_failure="",
                replan_count=replan_count,
                collision_count=collision_count,
            )

        return EpisodeMetrics(
            episode_id=episode_id,
            task_name=task_name,
            algorithm=algorithm,
            success=success,
            total_steps=step + 1,
            total_reward=total_reward,
            completion_time=completion_time,
            collision_count=collision_count,
            replan_count=replan_count,
            recovery_success=success and replan_count > 0,
            action_smoothness=smoothness,
            path_efficiency=path_eff,
            noise_level=noise_level,
            latency_level=latency_level,
        )

    def run_task_benchmark(
        self,
        env_factory,
        executor_factory,
        task_factory,
        algorithm: str,
        task_name: str,
        noise_level: str = "none",
        latency_level: str = "none",
    ) -> BenchmarkResult:
        """Run N episodes for one (algorithm, task, noise, latency) condition."""
        logger.info("Benchmarking: %s | %s | noise=%s | latency=%s",
                    algorithm, task_name, noise_level, latency_level)

        env = env_factory(noise_level, latency_level)
        executor = executor_factory()
        task = task_factory()
        ep_logger = EpisodeLogger(
            log_dir=str(self.results_dir / "logs"),
            algorithm=algorithm,
            task_name=task_name,
            save_video=self.save_videos,
        )

        episode_metrics: list[EpisodeMetrics] = []
        for i in range(self.n_episodes):
            metrics = self.run_episode(
                env, executor, task, i, algorithm, task_name,
                noise_level, latency_level, ep_logger,
            )
            episode_metrics.append(metrics)

        result = self._metrics_computer.aggregate(episode_metrics)
        self._save_result(result)
        return result

    def _save_result(self, result: BenchmarkResult) -> None:
        """Append result to the master results table."""
        table_path = self.results_dir / "tables" / "benchmark_results.jsonl"
        with open(table_path, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def export_csv(self, output_path: str | None = None) -> str:
        """Convert the JSONL results table to a CSV file."""
        jsonl_path = self.results_dir / "tables" / "benchmark_results.jsonl"
        csv_path = output_path or str(self.results_dir / "tables" / "benchmark_results.csv")

        if not jsonl_path.exists():
            logger.warning("No benchmark results found at '%s'", jsonl_path)
            return csv_path

        records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if records:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)
            logger.info("Exported %d results to '%s'", len(records), csv_path)

        return csv_path

    def generate_summary_report(self) -> str:
        """Generate a markdown summary of benchmark results."""
        jsonl_path = self.results_dir / "tables" / "benchmark_results.jsonl"
        if not jsonl_path.exists():
            return "No results available."

        records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not records:
            return "No results available."

        lines = ["# Benchmark Summary\n"]
        lines.append(f"Total conditions evaluated: {len(records)}\n")
        lines.append("| Algorithm | Task | Noise | Success Rate | Avg Reward | Avg Collisions |")
        lines.append("|-----------|------|-------|-------------|------------|----------------|")
        for r in records:
            lines.append(
                f"| {r.get('algorithm','')} | {r.get('task_name','')} | {r.get('noise_level','')} "
                f"| {r.get('success_rate',0):.2%} | {r.get('avg_reward',0):.2f} "
                f"| {r.get('avg_collision_count',0):.2f} |"
            )

        report = "\n".join(lines)
        report_path = self.results_dir / "benchmark_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        return report


def run_full_benchmark() -> None:
    """CLI entry point: physai-benchmark."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from envs.isaac.isaac_env import IsaacEnvConfig, MockIsaacEnv
    from envs.task_definitions import get_task
    from envs.wrappers.gym_wrapper import RobotGymWrapper
    from envs.wrappers.noise_wrapper import NoiseWrapper
    from envs.wrappers.latency_wrapper import LatencyWrapper
    from src.control.rl_executor import RLExecutor, ControllerConfig

    runner = BenchmarkRunner(n_episodes=20, seed=42)

    noise_levels = ["none", "low", "medium"]
    task_names = ["pick_and_place"]
    algorithms = ["ppo_random"]  # random policy as placeholder

    for task_name in task_names:
        for noise_level in noise_levels:
            def env_factory(nl=noise_level, ll="none"):
                raw_env = MockIsaacEnv(IsaacEnvConfig(seed=42))
                gym_env = RobotGymWrapper(env=raw_env)
                if nl != "none":
                    gym_env = NoiseWrapper.from_level(gym_env, nl)
                return gym_env

            def executor_factory():
                return RLExecutor(ControllerConfig(algorithm="ppo"))

            task_cfg = {"name": task_name, "objects": [], "success_criteria": {}, "reward_shaping": {}}

            def task_factory(tc=task_cfg):
                return get_task(task_name, tc)

            runner.run_task_benchmark(
                env_factory, executor_factory, task_factory,
                algorithm="random_baseline",
                task_name=task_name,
                noise_level=noise_level,
            )

    runner.export_csv()
    print(runner.generate_summary_report())


def main() -> None:
    """CLI entry point: physai-eval."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="PHYSAI-RL-ROBOT-X Evaluation")
    parser.add_argument("--task", default="pick_and_place")
    parser.add_argument("--algorithm", default="ppo")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--noise-level", default="none", choices=["none", "low", "medium", "high"])
    parser.add_argument("--latency-level", default="none", choices=["none", "low", "medium", "high"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    logger.info("Evaluation: task=%s, algo=%s, noise=%s, latency=%s",
                args.task, args.algorithm, args.noise_level, args.latency_level)
    runner = BenchmarkRunner(results_dir=args.results_dir, n_episodes=args.n_episodes, seed=args.seed)
    runner.export_csv()


if __name__ == "__main__":
    main()
