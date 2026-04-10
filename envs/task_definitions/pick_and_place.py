"""Pick-and-place task definition."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TaskResult:
    """Outcome of a completed task episode."""
    success: bool
    steps_taken: int
    collision_count: int
    replan_count: int
    completion_time: float
    reward_total: float
    failure_reason: str = ""


class BaseTask(ABC):
    """Abstract base for all manipulation task definitions."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self._instruction_templates: list[str] = []
        self._step = 0

    @abstractmethod
    def reset(self, env) -> dict[str, Any]:
        """Reset the task: spawn objects and return task info."""

    @abstractmethod
    def check_success(self, obs: dict, env) -> bool:
        """Return True if the task goal has been achieved."""

    @abstractmethod
    def compute_reward(
        self,
        obs: dict,
        action: np.ndarray,
        next_obs: dict,
        info: dict,
        env,
    ) -> float:
        """Return the shaped reward for the current transition."""

    def get_instruction(self) -> str:
        """Return a random natural-language instruction for this task."""
        if not self._instruction_templates:
            return "Complete the task."
        return random.choice(self._instruction_templates)

    @abstractmethod
    def get_skill_plan(self) -> list[dict]:
        """Return a hardcoded fallback skill plan as a list of skill dicts."""

    def get_info(self) -> dict:
        """Return task metadata for logging."""
        return {"task": self.config.get("name", "unknown"), "step": self._step}


class PickAndPlaceTask(BaseTask):
    """
    Pick a target object and place it inside the goal bin.

    Shaped reward:
      - approach_bonus: gripper within 0.15 m of object
      - grasp_bonus:    gripper closes on object (object leaves table)
      - lift_bonus:     object above lift threshold
      - transport_bonus: object moving toward goal
      - place_bonus:    object dropped near goal
      - success_bonus:  object stable inside goal bin
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._instruction_templates = config.get("instruction_templates", [
            "Pick up the red block and place it in the blue bin.",
            "Move the red cube into the blue container.",
            "Grasp the red block and put it in bin A.",
        ])
        self._target_object: str = ""
        self._goal_bin: str = ""
        self._initial_object_pos: np.ndarray | None = None
        self._grasp_bonus_given = False
        self._lift_bonus_given = False
        self._place_bonus_given = False
        self._success_steps = 0
        self._pos_tol: float = config.get("success_criteria", {}).get("position_tolerance", 0.05)
        self._stable_steps: int = config.get("success_criteria", {}).get("stable_steps", 10)

    # ------------------------------------------------------------------
    # BaseTask implementation
    # ------------------------------------------------------------------

    def reset(self, env) -> dict[str, Any]:
        """Spawn objects at randomised positions and return task info."""
        objects_cfg = self.config.get("objects", [])
        rng = np.random.default_rng()

        for obj_cfg in objects_cfg:
            name = obj_cfg["name"]
            region = obj_cfg.get("spawn_region", {})
            x = rng.uniform(*region.get("x", [0.4, 0.5]))
            y = rng.uniform(*region.get("y", [-0.15, 0.15]))
            z = region.get("z", [0.45, 0.45])[0]
            pos = np.array([x, y, z])
            if hasattr(env, "add_object"):
                env.add_object(name, pos, mass=obj_cfg.get("mass", 0.1))
            if obj_cfg.get("type") != "bin":
                self._target_object = name
                self._initial_object_pos = pos.copy()
            else:
                self._goal_bin = name

        self._grasp_bonus_given = False
        self._lift_bonus_given = False
        self._place_bonus_given = False
        self._success_steps = 0
        self._step = 0
        return self.get_info()

    def check_success(self, obs: dict, env) -> bool:
        """Check if target object is stably inside the goal bin."""
        if not (self._target_object and self._goal_bin):
            return False
        obj_states = {}
        if hasattr(env, "_object_states"):
            obj_states = env._object_states
        if self._target_object not in obj_states or self._goal_bin not in obj_states:
            return False
        obj_pos = obj_states[self._target_object][:3]
        bin_pos = obj_states[self._goal_bin][:3]
        dist = float(np.linalg.norm(obj_pos - bin_pos))
        if dist < self._pos_tol:
            self._success_steps += 1
        else:
            self._success_steps = 0
        return self._success_steps >= self._stable_steps

    def compute_reward(
        self,
        obs: dict,
        action: np.ndarray,
        next_obs: dict,
        info: dict,
        env,
    ) -> float:
        self._step += 1
        reward_cfg = self.config.get("reward_shaping", {})
        reward = 0.0

        obj_states = getattr(env, "_object_states", {})
        ee_pose = getattr(env, "_ee_pose", np.zeros(3))[:3]

        if self._target_object not in obj_states:
            return reward

        obj_pos = obj_states[self._target_object][:3]
        dist_to_obj = float(np.linalg.norm(ee_pose - obj_pos))

        # Approach bonus
        if dist_to_obj < 0.15:
            reward += reward_cfg.get("approach_bonus", 0.5) * (1.0 - dist_to_obj / 0.15)

        # Grasp bonus
        held = getattr(env, "_held_object", None)
        if held == self._target_object and not self._grasp_bonus_given:
            reward += reward_cfg.get("grasp_bonus", 2.0)
            self._grasp_bonus_given = True

        # Lift bonus
        if obj_pos[2] > 0.55 and not self._lift_bonus_given:
            reward += reward_cfg.get("lift_bonus", 1.0)
            self._lift_bonus_given = True

        # Transport / progress toward goal
        if self._goal_bin in obj_states:
            bin_pos = obj_states[self._goal_bin][:3]
            dist_to_goal = float(np.linalg.norm(obj_pos - bin_pos))
            if self._initial_object_pos is not None:
                init_dist = float(np.linalg.norm(self._initial_object_pos - bin_pos))
                progress = (init_dist - dist_to_goal) / max(init_dist, 1e-6)
                reward += reward_cfg.get("transport_bonus", 1.0) * max(0.0, progress) * 0.02

            # Place bonus
            if dist_to_goal < 0.08 and not self._place_bonus_given:
                reward += reward_cfg.get("place_bonus", 3.0)
                self._place_bonus_given = True

        # Success bonus
        if self.check_success(obs, env):
            reward += reward_cfg.get("success_bonus", 10.0)

        # Collision penalty
        if info.get("collision", False):
            reward -= 2.0

        # Time penalty
        reward -= 0.01

        return reward

    def get_skill_plan(self) -> list[dict]:
        """Fallback deterministic skill plan."""
        return [
            {"skill": "move_to",   "target": self._target_object, "approach_height": 0.1},
            {"skill": "grasp",     "target": self._target_object},
            {"skill": "move_to",   "target": self._goal_bin,      "approach_height": 0.15},
            {"skill": "release",   "target": self._target_object},
        ]

    def get_info(self) -> dict:
        base = super().get_info()
        base.update({
            "target_object": self._target_object,
            "goal_bin": self._goal_bin,
            "success_steps": self._success_steps,
        })
        return base
