"""
Gymnasium-compatible wrapper around BaseRobotEnv.

Provides proper observation_space / action_space definitions and tracks
per-episode statistics for training and evaluation.
"""
from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.isaac.isaac_env import BaseRobotEnv, IsaacEnvConfig, MockIsaacEnv, RobotObservation

logger = logging.getLogger(__name__)

# Observation dimensions
_ROBOT_STATE_DIM = 18    # 9 joint positions + 9 joint velocities
_OBJECT_STATE_DIM = 7    # pos(3) + quat(4) per object
_MAX_OBJECTS = 5
_TASK_GOAL_DIM = 64      # placeholder text-embedding dimension
_ACTION_DIM = 9          # 7 arm joints + 2 finger joints


class RobotGymWrapper(gym.Env):
    """
    Gymnasium wrapper for any BaseRobotEnv backend.

    Observation space (Dict):
        - ``robot_state``:   float32 (18,) — joint positions + velocities
        - ``object_states``: float32 (max_objects * 7,) — object poses (padded)
        - ``task_goal``:     float32 (64,) — task goal embedding (zeros if unused)
        - ``ee_pose``:       float32 (7,)  — end-effector pose

    Action space:
        - Box(-1, 1, shape=(9,), dtype=float32) — normalised joint deltas

    Args:
        env:            A BaseRobotEnv instance (or None to create MockIsaacEnv).
        config:         IsaacEnvConfig (used only when env is None).
        action_scale:   Multiplier applied to normalised actions before sending.
        max_objects:    Maximum number of objects in observation (padded with zeros).
        task_goal:      Optional fixed goal embedding (64,).
        max_episode_steps: Episode truncation limit.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        env: BaseRobotEnv | None = None,
        config: IsaacEnvConfig | None = None,
        action_scale: float = 2.0,
        max_objects: int = _MAX_OBJECTS,
        task_goal: np.ndarray | None = None,
        max_episode_steps: int = 500,
    ) -> None:
        super().__init__()
        if env is None:
            env = MockIsaacEnv(config or IsaacEnvConfig())
        self._env = env
        self._action_scale = action_scale
        self._max_objects = max_objects
        self._task_goal = task_goal if task_goal is not None else np.zeros(_TASK_GOAL_DIM, dtype=np.float32)
        self._max_episode_steps = max_episode_steps

        # Build spaces
        self.observation_space = spaces.Dict({
            "robot_state": spaces.Box(-np.inf, np.inf, shape=(_ROBOT_STATE_DIM,), dtype=np.float32),
            "object_states": spaces.Box(-np.inf, np.inf, shape=(max_objects * _OBJECT_STATE_DIM,), dtype=np.float32),
            "task_goal": spaces.Box(-np.inf, np.inf, shape=(_TASK_GOAL_DIM,), dtype=np.float32),
            "ee_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(_ACTION_DIM,), dtype=np.float32)

        # Episode tracking
        self._episode_step = 0
        self._episode_reward = 0.0
        self._episode_collisions = 0
        self._total_episodes = 0
        self._total_successes = 0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            super().reset(seed=seed)
        raw_obs = self._env.reset(seed=seed)
        self._episode_step = 0
        self._episode_reward = 0.0
        self._episode_collisions = 0
        return self._encode_obs(raw_obs), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        scaled_action = np.clip(action, -1.0, 1.0) * self._action_scale
        raw_obs, reward, terminated, truncated, info = self._env.step(scaled_action)

        self._episode_step += 1
        self._episode_reward += reward
        if self._env.check_collision():
            self._episode_collisions += 1

        if self._episode_step >= self._max_episode_steps:
            truncated = True

        if terminated or truncated:
            self._total_episodes += 1
            if info.get("success", False):
                self._total_successes += 1
            info["episode"] = {
                "reward": self._episode_reward,
                "length": self._episode_step,
                "collisions": self._episode_collisions,
                "success": info.get("success", False),
            }

        return self._encode_obs(raw_obs), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        obs = self._env.get_observation()
        return obs.rgb_overhead

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_obs(self, raw_obs: RobotObservation) -> dict[str, np.ndarray]:
        """Convert RobotObservation to flat gymnasium observation dict."""
        robot_state = np.concatenate([
            raw_obs.joint_positions.astype(np.float32),
            raw_obs.joint_velocities.astype(np.float32),
        ])

        # Pack object states into fixed-size array (padded with zeros)
        obj_array = np.zeros(self._max_objects * _OBJECT_STATE_DIM, dtype=np.float32)
        for i, pose in enumerate(list(raw_obs.object_states.values())[:self._max_objects]):
            obj_array[i * _OBJECT_STATE_DIM: (i + 1) * _OBJECT_STATE_DIM] = pose.astype(np.float32)

        return {
            "robot_state": robot_state,
            "object_states": obj_array,
            "task_goal": self._task_goal.copy(),
            "ee_pose": raw_obs.end_effector_pose.astype(np.float32),
        }

    @property
    def success_rate(self) -> float:
        """Episode success rate over the current run."""
        return self._total_successes / max(1, self._total_episodes)

    def set_task_goal(self, goal_embedding: np.ndarray) -> None:
        """Update the task-goal embedding (e.g. from a text encoder)."""
        assert goal_embedding.shape == (_TASK_GOAL_DIM,)
        self._task_goal = goal_embedding.astype(np.float32)
