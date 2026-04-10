"""
Noise injection wrapper for stress-testing robot policies.

Injects Gaussian noise, salt-and-pepper noise, and frame dropouts
into observations, and additive noise into actions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass
class NoiseConfig:
    """Parameters for each noise injection channel."""
    camera_noise_std: float = 0.0       # std of Gaussian noise on image pixels (0-1 scale)
    depth_noise_std: float = 0.0        # std of Gaussian noise on depth values (metres)
    joint_noise_std: float = 0.0        # std of Gaussian noise on joint obs
    actuation_noise_std: float = 0.0    # std of Gaussian noise added to actions
    obs_dropout_prob: float = 0.0       # probability of dropping an observation frame
    salt_pepper_prob: float = 0.0       # probability of salt-and-pepper pixel corruption

    @classmethod
    def from_level(cls, level: str) -> "NoiseConfig":
        """Factory for named noise levels."""
        presets = {
            "none":   cls(),
            "low":    cls(camera_noise_std=0.02, depth_noise_std=0.005,
                          joint_noise_std=0.005, actuation_noise_std=0.01,
                          obs_dropout_prob=0.02, salt_pepper_prob=0.01),
            "medium": cls(camera_noise_std=0.05, depth_noise_std=0.015,
                          joint_noise_std=0.01,  actuation_noise_std=0.03,
                          obs_dropout_prob=0.05, salt_pepper_prob=0.03),
            "high":   cls(camera_noise_std=0.10, depth_noise_std=0.03,
                          joint_noise_std=0.02,  actuation_noise_std=0.06,
                          obs_dropout_prob=0.10, salt_pepper_prob=0.07),
        }
        if level not in presets:
            raise ValueError(f"Unknown noise level '{level}'. Choose from {list(presets)}")
        return presets[level]


class NoiseWrapper(gym.Wrapper):
    """
    Wraps any Gymnasium environment and injects configurable noise.

    Noise is applied:
    - **Observations**: Gaussian noise on vector obs; pixel noise + dropout on images.
    - **Actions**: Additive Gaussian noise before execution (for actuation realism).

    Args:
        env:    Wrapped Gymnasium environment.
        config: :class:`NoiseConfig` specifying noise levels per channel.
        seed:   RNG seed for reproducibility.
    """

    def __init__(self, env: gym.Env, config: NoiseConfig | None = None, seed: int = 0) -> None:
        super().__init__(env)
        self.noise_cfg = config or NoiseConfig()
        self._rng = np.random.default_rng(seed)
        self._last_obs: dict | None = None

        # Episode statistics
        self._dropped_frames = 0
        self._total_frames = 0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> tuple[Any, dict]:
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._dropped_frames = 0
        self._total_frames = 0
        return self._corrupt_obs(obs), info

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        noisy_action = self._corrupt_action(action)
        obs, reward, terminated, truncated, info = self.env.step(noisy_action)
        self._total_frames += 1
        noisy_obs = self._corrupt_obs(obs)
        info["noise_stats"] = {
            "dropped_frames": self._dropped_frames,
            "drop_rate": self._dropped_frames / max(1, self._total_frames),
        }
        return noisy_obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Noise injection
    # ------------------------------------------------------------------

    def _corrupt_obs(self, obs: dict) -> dict:
        """Apply all configured noise sources to the observation."""
        cfg = self.noise_cfg

        # Frame dropout: repeat last observation with probability p
        if cfg.obs_dropout_prob > 0 and self._rng.random() < cfg.obs_dropout_prob:
            self._dropped_frames += 1
            if self._last_obs is not None:
                return self._last_obs
        self._last_obs = obs

        corrupted = {}
        for key, val in obs.items():
            if not isinstance(val, np.ndarray):
                corrupted[key] = val
                continue

            arr = val.copy().astype(np.float32)

            if key in ("robot_state", "ee_pose"):
                if cfg.joint_noise_std > 0:
                    arr += self._rng.normal(0, cfg.joint_noise_std, arr.shape).astype(np.float32)

            elif key == "object_states":
                # Light positional noise on object state observations
                if cfg.joint_noise_std > 0:
                    arr += self._rng.normal(0, cfg.joint_noise_std * 0.5, arr.shape).astype(np.float32)

            elif key == "task_goal":
                pass  # no noise on goal embedding

            corrupted[key] = arr.astype(val.dtype)

        return corrupted

    def _corrupt_action(self, action: np.ndarray) -> np.ndarray:
        """Add actuation noise to the action vector."""
        if self.noise_cfg.actuation_noise_std <= 0:
            return action
        noise = self._rng.normal(0, self.noise_cfg.actuation_noise_std, action.shape)
        return np.clip(action + noise, self.action_space.low, self.action_space.high).astype(action.dtype)

    @staticmethod
    def _add_salt_pepper(image: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
        """Apply salt-and-pepper noise to a uint8 image."""
        img = image.copy()
        mask_salt = rng.random(img.shape[:2]) < prob / 2
        mask_pepper = rng.random(img.shape[:2]) < prob / 2
        img[mask_salt] = 255
        img[mask_pepper] = 0
        return img

    @classmethod
    def from_level(cls, env: gym.Env, level: str, seed: int = 0) -> "NoiseWrapper":
        """Convenience constructor using a named noise level."""
        return cls(env, NoiseConfig.from_level(level), seed=seed)

    @property
    def drop_rate(self) -> float:
        return self._dropped_frames / max(1, self._total_frames)
