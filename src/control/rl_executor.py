"""
RL executor: closed-loop low-level controller using a trained RL policy.

Wraps PPO/SAC policies from stable-baselines3 and exposes a clean interface
for the SkillRouter and training pipeline.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ControllerConfig:
    """Configuration for the RL executor."""
    algorithm: str = "ppo"         # "ppo" | "sac" | "td3"
    checkpoint_path: str = ""
    device: str = "auto"
    deterministic: bool = True
    action_scale: float = 1.0
    obs_normalize: bool = True


class RLExecutor:
    """
    Closed-loop RL policy executor.

    Loads a trained stable-baselines3 policy and runs it step-by-step.
    Supports PPO, SAC, and TD3.  Falls back to a random policy when no
    checkpoint is available (for scaffolding / integration testing).

    Args:
        config: :class:`ControllerConfig`
    """

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.cfg = config or ControllerConfig()
        self._policy = None
        self._vec_normalize = None
        self._obs_dim: int | None = None
        self._act_dim: int | None = None

        if self.cfg.checkpoint_path:
            self._load_checkpoint(self.cfg.checkpoint_path)

    def _load_checkpoint(self, path: str) -> None:
        """Load a saved SB3 policy from disk."""
        try:
            from stable_baselines3 import PPO, SAC, TD3  # type: ignore[import]
            algo_map = {"ppo": PPO, "sac": SAC, "td3": TD3}
            AlgoCls = algo_map.get(self.cfg.algorithm.lower())
            if AlgoCls is None:
                raise ValueError(f"Unknown algorithm '{self.cfg.algorithm}'")

            self._policy = AlgoCls.load(path, device=self.cfg.device)
            logger.info("RLExecutor: loaded %s checkpoint from '%s'", self.cfg.algorithm.upper(), path)

            # Load optional VecNormalize stats
            norm_path = Path(path).with_suffix(".vecnorm.pkl")
            if norm_path.exists():
                from stable_baselines3.common.vec_env import VecNormalize  # type: ignore[import]
                self._vec_normalize = VecNormalize.load(str(norm_path), None)
                logger.info("RLExecutor: loaded VecNormalize stats from '%s'", norm_path)

        except (ImportError, FileNotFoundError) as exc:
            logger.warning("RLExecutor: could not load checkpoint (%s). Using random policy.", exc)

    def predict(
        self,
        observation: dict[str, np.ndarray] | np.ndarray,
        state: Any = None,
        episode_start: bool | None = None,
    ) -> tuple[np.ndarray, Any]:
        """
        Run one policy inference step.

        Args:
            observation: Gym observation dict or flat numpy array.
            state:       Recurrent state (None for non-recurrent policies).
            episode_start: Whether this is the first step of an episode.

        Returns:
            (action, state) where action is a float32 numpy array.
        """
        if self._policy is not None:
            obs = self._preprocess_obs(observation)
            action, state = self._policy.predict(
                obs,
                state=state,
                episode_start=episode_start,
                deterministic=self.cfg.deterministic,
            )
        else:
            # Random policy fallback
            action = self._random_action(observation)

        action = action.astype(np.float32) * self.cfg.action_scale
        return action, state

    def _preprocess_obs(self, obs: dict | np.ndarray) -> np.ndarray | dict:
        """Apply optional normalisation to the observation."""
        if self._vec_normalize is not None and isinstance(obs, np.ndarray):
            return self._vec_normalize.normalize_obs(obs)
        return obs

    def _random_action(self, obs: dict | np.ndarray) -> np.ndarray:
        """Produce a small random action (used as fallback)."""
        dim = 9  # default Franka action dim
        if isinstance(obs, np.ndarray):
            self._obs_dim = obs.shape[-1]
        return np.random.uniform(-0.1, 0.1, size=(dim,)).astype(np.float32)

    def save(self, path: str) -> None:
        """Save the current policy to disk."""
        if self._policy is not None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._policy.save(path)
            logger.info("RLExecutor: saved checkpoint to '%s'", path)

    @property
    def is_loaded(self) -> bool:
        return self._policy is not None


class EnsembleRLExecutor:
    """
    Ensemble of multiple RL executors that vote on the action.

    Useful for uncertainty estimation: actions with high inter-policy
    disagreement can be flagged for replanning.

    Args:
        executors:     List of trained RLExecutors.
        aggregation:   How to combine actions: ``"mean"`` or ``"vote"``.
    """

    def __init__(self, executors: list[RLExecutor], aggregation: str = "mean") -> None:
        self._executors = executors
        self._aggregation = aggregation

    def predict(
        self,
        observation: dict | np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, float]:
        """
        Predict action and return (action, disagreement_score).

        Disagreement score is the mean standard deviation across ensemble
        members — high values suggest the policy is uncertain.
        """
        actions = []
        for ex in self._executors:
            a, _ = ex.predict(observation, **kwargs)
            actions.append(a)

        arr = np.stack(actions, axis=0)
        mean_action = arr.mean(axis=0)
        disagreement = float(arr.std(axis=0).mean())

        if self._aggregation == "mean":
            return mean_action.astype(np.float32), disagreement
        else:
            # Majority-vote per dimension: use median
            return np.median(arr, axis=0).astype(np.float32), disagreement
