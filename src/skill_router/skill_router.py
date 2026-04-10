"""
Skill router: maps high-level plan steps to low-level RL controller calls.

The router maintains a registry of named skills, executes them in sequence,
handles failures, and triggers replanning when a skill cannot be completed.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from src.planner.base_planner import PlanResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skill execution result
# ---------------------------------------------------------------------------

@dataclass
class SkillResult:
    """Outcome of a single skill execution."""
    skill: str
    target: Any
    success: bool
    steps_taken: int
    reward: float
    info: dict = field(default_factory=dict)
    failure_reason: str = ""


@dataclass
class RouterResult:
    """Outcome of executing a full skill plan."""
    skill_results: list[SkillResult] = field(default_factory=list)
    plan_success: bool = False
    total_steps: int = 0
    total_reward: float = 0.0
    replan_count: int = 0
    failure_step: int = -1
    failure_reason: str = ""

    @property
    def success_rate(self) -> float:
        if not self.skill_results:
            return 0.0
        return sum(r.success for r in self.skill_results) / len(self.skill_results)


# ---------------------------------------------------------------------------
# Skill implementations
# ---------------------------------------------------------------------------

SkillFn = Callable[[dict, Any, dict], SkillResult]


def _skill_move_to(params: dict, env: Any, state: dict) -> SkillResult:
    """Move end-effector to target object/position."""
    target = params.get("target", "")
    approach_height = params.get("approach_height", 0.1)
    max_steps = 50
    total_reward = 0.0

    for step in range(max_steps):
        obj_states = getattr(env, "_object_states", {})
        ee_pose = getattr(env, "_ee_pose", np.zeros(7))[:3]

        # Determine goal position
        if target in obj_states:
            goal = obj_states[target][:3].copy()
            goal[2] += approach_height
        elif isinstance(target, list) and len(target) == 3:
            goal = np.array(target)
        else:
            goal = np.array([0.5, 0.0, 0.55])

        delta = goal - ee_pose
        dist = float(np.linalg.norm(delta))
        if dist < 0.02:
            return SkillResult(skill="move_to", target=target, success=True,
                               steps_taken=step + 1, reward=total_reward)

        # Proportional control → joint delta action
        action = np.zeros(9)
        action[0] = np.clip(delta[1] * 3.0, -1.0, 1.0)
        action[1] = np.clip(delta[0] * 3.0, -1.0, 1.0)
        action[3] = np.clip(delta[2] * 3.0, -1.0, 1.0)
        action[7] = 1.0  # keep gripper open

        _, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if term or trunc:
            break

    return SkillResult(skill="move_to", target=target, success=False,
                       steps_taken=max_steps, reward=total_reward,
                       failure_reason="timeout")


def _skill_grasp(params: dict, env: Any, state: dict) -> SkillResult:
    """Close gripper to grasp target."""
    target = params.get("target", "")
    action = np.zeros(9)
    action[7] = -1.0  # close gripper

    total_reward = 0.0
    for step in range(10):
        _, reward, term, trunc, info = env.step(action)
        total_reward += reward

    held = getattr(env, "_held_object", None)
    success = held == target or held is not None
    return SkillResult(skill="grasp", target=target, success=success,
                       steps_taken=10, reward=total_reward,
                       failure_reason="" if success else f"object '{target}' not grasped")


def _skill_release(params: dict, env: Any, state: dict) -> SkillResult:
    """Open gripper to release held object."""
    target = params.get("target", "")
    action = np.zeros(9)
    action[7] = 1.0  # open gripper

    total_reward = 0.0
    for step in range(10):
        _, reward, term, trunc, info = env.step(action)
        total_reward += reward

    held = getattr(env, "_held_object", None)
    success = held is None
    return SkillResult(skill="release", target=target, success=success,
                       steps_taken=10, reward=total_reward)


def _skill_wait(params: dict, env: Any, state: dict) -> SkillResult:
    """Wait (execute null action) for specified duration."""
    duration = params.get("duration", 0.5)
    steps = max(1, int(duration / 0.01))
    action = np.zeros(9)
    total_reward = 0.0
    for _ in range(steps):
        _, r, term, trunc, _ = env.step(action)
        total_reward += r
    return SkillResult(skill="wait", target=duration, success=True,
                       steps_taken=steps, reward=total_reward)


def _skill_reset(params: dict, env: Any, state: dict) -> SkillResult:
    """Reset robot to home position."""
    env.reset()
    return SkillResult(skill="reset", target=None, success=True, steps_taken=1, reward=0.0)


# ---------------------------------------------------------------------------
# Skill router
# ---------------------------------------------------------------------------

class SkillRouter:
    """
    Executes a :class:`PlanResult` skill sequence on an environment.

    Maintains a registry of skill functions.  On skill failure, it can
    optionally trigger replanning via a provided replan callback.

    Args:
        replan_callback: Callable ``(scene, goal) -> PlanResult`` invoked on failure.
        max_replan_attempts: Maximum replanning attempts per episode.
        skill_timeout_steps: Max env steps per skill before declaring failure.
    """

    _DEFAULT_SKILLS: dict[str, SkillFn] = {
        "move_to":     _skill_move_to,
        "grasp":       _skill_grasp,
        "release":     _skill_release,
        "place_at":    _skill_move_to,   # alias: move_to then release
        "wait":        _skill_wait,
        "reset":       _skill_reset,
    }

    def __init__(
        self,
        replan_callback: Callable | None = None,
        max_replan_attempts: int = 3,
        skill_timeout_steps: int = 100,
    ) -> None:
        self._skills: dict[str, SkillFn] = dict(self._DEFAULT_SKILLS)
        self._replan_callback = replan_callback
        self._max_replan_attempts = max_replan_attempts
        self._skill_timeout_steps = skill_timeout_steps

    def register_skill(self, name: str, fn: SkillFn) -> None:
        """Register a custom skill function."""
        self._skills[name] = fn
        logger.debug("SkillRouter: registered skill '%s'", name)

    def execute(self, plan: PlanResult, env: Any, state: dict | None = None) -> RouterResult:
        """
        Execute a skill plan on the given environment.

        Args:
            plan:  PlanResult from a planner.
            env:   Any environment with a ``step(action)`` interface.
            state: Optional mutable state dict shared across skills.

        Returns:
            :class:`RouterResult`
        """
        state = state or {}
        result = RouterResult()

        if not plan.valid or not plan.steps:
            result.failure_reason = plan.validation_error or "Empty plan"
            return result

        replan_attempts = 0
        remaining_steps = list(plan.steps)

        while remaining_steps:
            step_dict = remaining_steps.pop(0)
            skill_name = step_dict.get("skill", "")
            params = {k: v for k, v in step_dict.items() if k != "skill"}

            skill_fn = self._skills.get(skill_name)
            if skill_fn is None:
                logger.warning("SkillRouter: unknown skill '%s' — skipping", skill_name)
                result.skill_results.append(SkillResult(
                    skill=skill_name, target=params.get("target"), success=False,
                    steps_taken=0, reward=0.0, failure_reason="unknown skill",
                ))
                continue

            logger.debug("SkillRouter: executing %s(%s)", skill_name, params)
            skill_result = skill_fn(params, env, state)
            result.skill_results.append(skill_result)
            result.total_steps += skill_result.steps_taken
            result.total_reward += skill_result.reward

            if not skill_result.success:
                result.failure_step = len(result.skill_results) - 1
                result.failure_reason = skill_result.failure_reason
                logger.info("SkillRouter: skill '%s' failed: %s", skill_name, skill_result.failure_reason)

                if self._replan_callback and replan_attempts < self._max_replan_attempts:
                    replan_attempts += 1
                    result.replan_count += 1
                    new_plan = self._replan_callback(state.get("scene"), state.get("goal", ""))
                    if new_plan.valid and new_plan.steps:
                        logger.info("SkillRouter: replanning succeeded (attempt %d)", replan_attempts)
                        remaining_steps = list(new_plan.steps)
                        continue
                    logger.warning("SkillRouter: replanning failed (attempt %d)", replan_attempts)

                # Give up
                break

        result.plan_success = (
            len(result.skill_results) > 0 and
            result.failure_step == -1 and
            all(r.success for r in result.skill_results)
        )
        return result
