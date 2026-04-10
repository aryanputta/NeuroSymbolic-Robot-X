"""Sort-objects task: categorise objects into matching bins."""
from __future__ import annotations

import random
from typing import Any

import numpy as np

from envs.task_definitions.pick_and_place import BaseTask, TaskResult  # noqa: F401


class SortObjectsTask(BaseTask):
    """
    Sort multiple objects into category-matched bins (e.g. red→red bin, blue→blue bin).

    Success: every non-bin object is stably placed in the bin that matches its category.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._instruction_templates = config.get("instruction_templates", [
            "Sort the blocks by colour — red blocks in the red bin, blue blocks in the blue bin.",
            "Separate the red and blue cubes into their matching coloured bins.",
            "Place each block into the bin that matches its colour.",
        ])
        # name -> category string
        self._object_categories: dict[str, str] = {}
        # category -> bin name
        self._category_bins: dict[str, str] = {}

        self._pos_tol: float = config.get("success_criteria", {}).get("position_tolerance", 0.06)
        self._stable_steps: int = config.get("success_criteria", {}).get("stable_steps", 15)
        self._stable_counters: dict[str, int] = {}
        self._bonus_given: dict[str, bool] = {}

    def reset(self, env) -> dict[str, Any]:
        self._object_categories = {}
        self._category_bins = {}
        self._stable_counters = {}
        self._bonus_given = {}
        self._step = 0

        rng = np.random.default_rng()
        for obj_cfg in self.config.get("objects", []):
            name = obj_cfg["name"]
            region = obj_cfg.get("spawn_region", {})
            x = rng.uniform(*region.get("x", [0.35, 0.55]))
            y = rng.uniform(*region.get("y", [-0.25, 0.25]))
            z = region.get("z", [0.45, 0.45])[0]
            pos = np.array([x, y, z])
            if hasattr(env, "add_object"):
                env.add_object(name, pos, mass=obj_cfg.get("mass", 0.1))

            if obj_cfg.get("type") == "bin":
                category_accepted = obj_cfg.get("accepts", "")
                self._category_bins[category_accepted] = name
            else:
                category = obj_cfg.get("category", "unknown")
                self._object_categories[name] = category
                self._stable_counters[name] = 0
                self._bonus_given[name] = False

        return self.get_info()

    def check_success(self, obs: dict, env) -> bool:
        """All sortable objects must be stably inside their matching bin."""
        obj_states = getattr(env, "_object_states", {})
        all_placed = True
        for obj_name, category in self._object_categories.items():
            bin_name = self._category_bins.get(category)
            if bin_name is None or obj_name not in obj_states or bin_name not in obj_states:
                all_placed = False
                continue
            obj_pos = obj_states[obj_name][:3]
            bin_pos = obj_states[bin_name][:3]
            dist = float(np.linalg.norm(obj_pos - bin_pos))
            if dist < self._pos_tol:
                self._stable_counters[obj_name] += 1
            else:
                self._stable_counters[obj_name] = 0
            if self._stable_counters[obj_name] < self._stable_steps:
                all_placed = False
        return all_placed

    def compute_reward(
        self,
        obs: dict,
        action: np.ndarray,
        next_obs: dict,
        info: dict,
        env,
    ) -> float:
        self._step += 1
        reward = 0.0
        obj_states = getattr(env, "_object_states", {})
        ee_pose = getattr(env, "_ee_pose", np.zeros(3))[:3]

        for obj_name, category in self._object_categories.items():
            bin_name = self._category_bins.get(category)
            if obj_name not in obj_states or bin_name not in obj_states:
                continue
            obj_pos = obj_states[obj_name][:3]
            bin_pos = obj_states[bin_name][:3]
            dist_to_goal = float(np.linalg.norm(obj_pos - bin_pos))

            # Approach reward
            dist_to_obj = float(np.linalg.norm(ee_pose - obj_pos))
            if dist_to_obj < 0.15:
                reward += 0.3 * (1.0 - dist_to_obj / 0.15)

            # Per-object placement bonus (once)
            if dist_to_goal < self._pos_tol and not self._bonus_given.get(obj_name, True):
                reward += 3.0
                self._bonus_given[obj_name] = True

        # Completion bonus
        if self.check_success(obs, env):
            reward += 15.0

        if info.get("collision", False):
            reward -= 2.0
        reward -= 0.01
        return reward

    def get_skill_plan(self) -> list[dict]:
        plan = []
        for obj_name, category in self._object_categories.items():
            bin_name = self._category_bins.get(category, "bin")
            plan += [
                {"skill": "move_to", "target": obj_name,  "approach_height": 0.1},
                {"skill": "grasp",   "target": obj_name},
                {"skill": "move_to", "target": bin_name,  "approach_height": 0.15},
                {"skill": "release", "target": obj_name},
            ]
        return plan

    def get_info(self) -> dict:
        base = super().get_info()
        base.update({
            "object_categories": self._object_categories,
            "category_bins": self._category_bins,
            "stable_counters": dict(self._stable_counters),
        })
        return base
