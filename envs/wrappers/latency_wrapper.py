"""
Latency injection wrapper for stress-testing robot policies under delays.

Simulates observation lag, action delay, dropped frames, and optional
artificial sleep to mimic planner call overhead.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass
class LatencyConfig:
    """Parameters for latency injection."""
    obs_delay_steps: int = 0        # delay observations by N timesteps
    action_delay_steps: int = 0     # delay action execution by N timesteps
    dropped_frame_prob: float = 0.0 # probability observation is replaced by last
    planning_latency_ms: float = 0.0  # artificial sleep per step (ms)

    @classmethod
    def from_level(cls, level: str) -> "LatencyConfig":
        presets = {
            "none":   cls(),
            "low":    cls(obs_delay_steps=2,  action_delay_steps=1, dropped_frame_prob=0.02,
                          planning_latency_ms=10.0),
            "medium": cls(obs_delay_steps=5,  action_delay_steps=3, dropped_frame_prob=0.05,
                          planning_latency_ms=50.0),
            "high":   cls(obs_delay_steps=10, action_delay_steps=5, dropped_frame_prob=0.10,
                          planning_latency_ms=150.0),
        }
        if level not in presets:
            raise ValueError(f"Unknown latency level '{level}'. Choose from {list(presets)}")
        return presets[level]


class LatencyWrapper(gym.Wrapper):
    """
    Wraps a Gymnasium environment and injects configurable latency.

    Three mechanisms are available:
    1. **Observation delay**: returns observations from N steps ago.
    2. **Action delay**: executes actions N steps after they were submitted.
    3. **Frame drop**: probabilistically repeats the previous observation.
    4. **Planning sleep**: inserts a ``time.sleep`` to simulate planner overhead.

    All buffers use circular (deque) semantics for efficiency.

    Args:
        env:    Wrapped Gymnasium environment.
        config: :class:`LatencyConfig`.
        seed:   RNG seed for frame-drop reproducibility.
    """

    def __init__(self, env: gym.Env, config: LatencyConfig | None = None, seed: int = 0) -> None:
        super().__init__(env)
        self.latency_cfg = config or LatencyConfig()
        self._rng = np.random.default_rng(seed)

        obs_buf_size = max(1, self.latency_cfg.obs_delay_steps + 1)
        act_buf_size = max(1, self.latency_cfg.action_delay_steps + 1)

        self._obs_buffer: deque = deque(maxlen=obs_buf_size)
        self._act_buffer: deque = deque(maxlen=act_buf_size)

        self._last_obs: Any = None
        self._null_action: np.ndarray | None = None

        # Stats
        self._total_steps = 0
        self._dropped_frames = 0
        self._cumulative_sleep_ms = 0.0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> tuple[Any, dict]:
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.clear()
        self._act_buffer.clear()
        # Pre-fill observation buffer with the reset observation
        for _ in range(self.latency_cfg.obs_delay_steps + 1):
            self._obs_buffer.append(obs)
        self._last_obs = obs
        self._total_steps = 0
        self._dropped_frames = 0
        self._cumulative_sleep_ms = 0.0
        return self._get_delayed_obs(obs), info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        cfg = self.latency_cfg

        # Optional planning latency sleep
        if cfg.planning_latency_ms > 0:
            sleep_s = cfg.planning_latency_ms / 1000.0
            time.sleep(sleep_s)
            self._cumulative_sleep_ms += cfg.planning_latency_ms

        # Resolve delayed action to actually execute
        delayed_action = self._get_delayed_action(action)

        obs, reward, terminated, truncated, info = self.env.step(delayed_action)

        self._total_steps += 1
        delayed_obs = self._get_delayed_obs(obs)

        info["latency_stats"] = {
            "obs_delay_steps": cfg.obs_delay_steps,
            "action_delay_steps": cfg.action_delay_steps,
            "dropped_frames": self._dropped_frames,
            "total_sleep_ms": self._cumulative_sleep_ms,
        }
        return delayed_obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Delay mechanics
    # ------------------------------------------------------------------

    def _get_delayed_obs(self, current_obs: Any) -> Any:
        """Push current obs onto buffer and return the oldest buffered obs."""
        cfg = self.latency_cfg

        # Frame drop: repeat last with probability p
        if cfg.dropped_frame_prob > 0 and self._rng.random() < cfg.dropped_frame_prob:
            self._dropped_frames += 1
            self._obs_buffer.append(self._last_obs)
        else:
            self._last_obs = current_obs
            self._obs_buffer.append(current_obs)

        # Return the oldest item (i.e. N steps ago)
        return self._obs_buffer[0]

    def _get_delayed_action(self, action: np.ndarray) -> np.ndarray:
        """Push action onto buffer; execute oldest buffered action."""
        cfg = self.latency_cfg
        if cfg.action_delay_steps <= 0:
            return action

        if self._null_action is None:
            self._null_action = np.zeros_like(action)

        self._act_buffer.append(action)

        if len(self._act_buffer) <= cfg.action_delay_steps:
            return self._null_action

        return self._act_buffer[0]

    @classmethod
    def from_level(cls, env: gym.Env, level: str, seed: int = 0) -> "LatencyWrapper":
        """Convenience constructor using a named latency level."""
        return cls(env, LatencyConfig.from_level(level), seed=seed)

    @property
    def effective_drop_rate(self) -> float:
        return self._dropped_frames / max(1, self._total_steps)
