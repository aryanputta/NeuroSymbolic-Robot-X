"""SAC policy builder for continuous robot control."""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def build_sac(
    env,
    buffer_size: int = 1_000_000,
    learning_starts: int = 10_000,
    batch_size: int = 256,
    tau: float = 0.005,
    gamma: float = 0.99,
    train_freq: int = 1,
    gradient_steps: int = 1,
    ent_coef: str | float = "auto",
    target_entropy: str | float = "auto",
    learning_rate: float = 3e-4,
    net_arch: list[int] | None = None,
    device: str = "auto",
    verbose: int = 1,
    tensorboard_log: str | None = None,
    seed: int = 42,
):
    """
    Build a SAC agent suited for continuous manipulation control.

    SAC is often preferred over PPO for continuous-action tasks because it:
    - Is sample-efficient via an off-policy replay buffer.
    - Automatically tunes entropy via ``ent_coef='auto'``.
    - Handles high-dimensional continuous spaces well (e.g. 9-DoF joints).

    Args:
        env:         Gymnasium environment.
        net_arch:    Hidden layer sizes (default [256, 256]).
        ...          Standard SB3 SAC kwargs.

    Returns:
        stable-baselines3 SAC instance.
    """
    try:
        import torch.nn as nn
        from stable_baselines3 import SAC  # type: ignore[import]

        if net_arch is None:
            net_arch = [256, 256]

        policy_kwargs: dict[str, Any] = {
            "net_arch": net_arch,
            "activation_fn": nn.ReLU,
        }

        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            seed=seed,
            device=device,
        )
        logger.info("SAC policy built: net_arch=%s, lr=%g", net_arch, learning_rate)
        return model

    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required to build SAC. "
            "Install it with: pip install stable-baselines3"
        ) from exc


def build_sac_callbacks(
    eval_env,
    eval_freq: int = 50_000,
    n_eval_episodes: int = 20,
    save_path: str = "models/checkpoints",
    log_path: str = "results/logs",
    verbose: int = 1,
) -> list:
    """Build standard SB3 callbacks for SAC training."""
    try:
        from stable_baselines3.common.callbacks import (  # type: ignore[import]
            CheckpointCallback,
            EvalCallback,
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
            name_prefix="sac_robot",
            verbose=verbose,
        )
        return [eval_callback, checkpoint_callback]

    except ImportError:
        logger.warning("Could not build SAC callbacks (stable-baselines3 not installed)")
        return []
