"""
PPO policy builder and training helpers.

Provides a thin wrapper around stable-baselines3 PPO with project-specific
defaults, curriculum support, and callback hooks.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def build_ppo(
    env,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    learning_rate: float = 3e-4,
    net_arch: list[int] | None = None,
    device: str = "auto",
    verbose: int = 1,
    tensorboard_log: str | None = None,
    seed: int = 42,
):
    """
    Build a PPO agent with sensible defaults for robot manipulation.

    Args:
        env:    Gymnasium environment (or VecEnv).
        net_arch: Hidden layer sizes, default [256, 256, 128].
        ...     All standard SB3 PPO kwargs.

    Returns:
        stable-baselines3 PPO instance.
    """
    try:
        import torch.nn as nn
        from stable_baselines3 import PPO  # type: ignore[import]
        from stable_baselines3.common.torch_layers import FlattenExtractor  # noqa

        if net_arch is None:
            net_arch = [256, 256, 128]

        policy_kwargs: dict[str, Any] = {
            "net_arch": net_arch,
            "activation_fn": nn.ReLU,
            "ortho_init": True,
        }

        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            seed=seed,
            device=device,
        )
        logger.info("PPO policy built: net_arch=%s, lr=%g", net_arch, learning_rate)
        return model

    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required to build PPO. "
            "Install it with: pip install stable-baselines3"
        ) from exc


def build_ppo_callbacks(
    eval_env,
    eval_freq: int = 50_000,
    n_eval_episodes: int = 20,
    save_path: str = "models/checkpoints",
    log_path: str = "results/logs",
    verbose: int = 1,
) -> list:
    """Build standard SB3 callbacks for training: eval + checkpoint + stop-on-nan."""
    try:
        from stable_baselines3.common.callbacks import (  # type: ignore[import]
            CallbackList,
            CheckpointCallback,
            EvalCallback,
            StopTrainingOnNaNLoss,
        )

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=verbose,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=eval_freq * 2,
            save_path=save_path,
            name_prefix="ppo_robot",
            verbose=verbose,
        )
        nan_callback = StopTrainingOnNaNLoss(verbose=verbose)

        return [eval_callback, checkpoint_callback, nan_callback]

    except ImportError:
        logger.warning("Could not build full SB3 callbacks (stable-baselines3 not installed)")
        return []
