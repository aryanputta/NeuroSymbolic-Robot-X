"""
Trajectory store: persists episode trajectories for offline analysis
and curriculum learning.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """Single environment transition."""
    obs: dict
    action: np.ndarray
    reward: float
    next_obs: dict
    terminated: bool
    truncated: bool
    info: dict


@dataclass
class EpisodeTrajectory:
    """Complete trajectory for one episode."""
    episode_id: str
    task_name: str
    algorithm: str
    transitions: list[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False
    steps: int = 0
    collision_count: int = 0
    replan_count: int = 0
    skill_plan: list[dict] = field(default_factory=list)
    failure_reason: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def add(self, transition: Transition) -> None:
        self.transitions.append(transition)
        self.total_reward += transition.reward
        self.steps += 1

    def to_summary(self) -> dict:
        """Compact summary without full transition data."""
        return {
            "episode_id": self.episode_id,
            "task_name": self.task_name,
            "algorithm": self.algorithm,
            "total_reward": round(self.total_reward, 4),
            "success": self.success,
            "steps": self.steps,
            "collision_count": self.collision_count,
            "replan_count": self.replan_count,
            "failure_reason": self.failure_reason,
            "timestamp": self.timestamp,
        }


class TrajectoryStore:
    """
    Stores and retrieves episode trajectories.

    Supports two storage formats:
    - ``jsonl``: Human-readable summaries, one per line.
    - ``pickle``: Full trajectories including numpy arrays.

    Args:
        base_dir:       Root directory for stored trajectories.
        max_episodes:   Maximum episodes to keep in memory (FIFO eviction).
        save_pickle:    Whether to save full pickle trajectories.
        save_jsonl:     Whether to save JSONL summaries.
    """

    def __init__(
        self,
        base_dir: str = "data/trajectories",
        max_episodes: int = 10_000,
        save_pickle: bool = False,
        save_jsonl: bool = True,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._max_episodes = max_episodes
        self._save_pickle = save_pickle
        self._save_jsonl = save_jsonl

        self._episodes: list[EpisodeTrajectory] = []
        self._episode_count = 0

    def add(self, episode: EpisodeTrajectory) -> None:
        """Add a completed episode trajectory."""
        if len(self._episodes) >= self._max_episodes:
            self._episodes.pop(0)
        self._episodes.append(episode)
        self._episode_count += 1

        if self._save_jsonl:
            self._append_jsonl(episode)
        if self._save_pickle:
            self._save_pkl(episode)

    def _append_jsonl(self, episode: EpisodeTrajectory) -> None:
        path = self._base_dir / f"{episode.task_name}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(episode.to_summary()) + "\n")

    def _save_pkl(self, episode: EpisodeTrajectory) -> None:
        path = self._base_dir / "pickles" / f"{episode.episode_id}.pkl"
        path.parent.mkdir(exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(episode, f)

    def get_failures(self, task_name: str | None = None) -> list[EpisodeTrajectory]:
        """Return failed episodes, optionally filtered by task."""
        eps = [e for e in self._episodes if not e.success]
        if task_name:
            eps = [e for e in eps if e.task_name == task_name]
        return eps

    def get_successes(self, task_name: str | None = None) -> list[EpisodeTrajectory]:
        eps = [e for e in self._episodes if e.success]
        if task_name:
            eps = [e for e in eps if e.task_name == task_name]
        return eps

    def success_rate(self, task_name: str | None = None) -> float:
        eps = self._episodes if not task_name else [e for e in self._episodes if e.task_name == task_name]
        if not eps:
            return 0.0
        return sum(e.success for e in eps) / len(eps)

    def common_failure_reasons(self, top_k: int = 5) -> list[tuple[str, int]]:
        """Return (reason, count) sorted by frequency."""
        counts: dict[str, int] = {}
        for ep in self._episodes:
            if not ep.success and ep.failure_reason:
                counts[ep.failure_reason] = counts.get(ep.failure_reason, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def load_jsonl(self, path: str) -> list[dict]:
        """Load episode summaries from a JSONL file."""
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    @property
    def stats(self) -> dict:
        return {
            "total_episodes": self._episode_count,
            "in_memory": len(self._episodes),
            "success_rate": self.success_rate(),
            "common_failures": self.common_failure_reasons(3),
        }
