"""Stack-blocks task: stack N blocks in a specified order."""
from __future__ import annotations

from typing import Any

import numpy as np

from envs.task_definitions.pick_and_place import BaseTask


class StackBlocksTask(BaseTask):
    """
    Stack a sequence of blocks in a specified bottom-to-top order.

    Success: each block is resting on the one below it (within tolerance),
    and all blocks are stacked stably for ``stable_steps`` consecutive steps.
    """

    # Approximate block half-height for stacking z offsets
    _BLOCK_HEIGHT = 0.05

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._instruction_templates = config.get("instruction_templates", [
            "Stack the blocks from bottom to top: red, blue, green.",
            "Build a tower by stacking the blocks in order.",
            "Place the blocks on top of each other to form a stack.",
        ])
        # Ordered list of block names from bottom to top
        self._stack_order: list[str] = config.get("stack_order", [])
        self._base_position: np.ndarray = np.array(config.get("stack_base_position", [0.5, 0.0, 0.45]))
        self._pos_tol: float = config.get("position_tolerance", 0.04)
        self._stable_steps: int = config.get("stable_steps", 15)
        self._stable_counter: int = 0
        self._layer_bonus_given: dict[int, bool] = {}

    def reset(self, env) -> dict[str, Any]:
        self._stable_counter = 0
        self._layer_bonus_given = {}
        self._step = 0

        rng = np.random.default_rng()
        objects_cfg = self.config.get("objects", [])
        if not objects_cfg:
            # Generate default objects from stack_order
            objects_cfg = [
                {"name": n, "type": "cube", "mass": 0.1,
                 "spawn_region": {"x": [0.3, 0.5], "y": [-0.2, 0.2], "z": [0.45, 0.45]}}
                for n in self._stack_order
            ]

        for obj_cfg in objects_cfg:
            name = obj_cfg["name"]
            region = obj_cfg.get("spawn_region", {})
            x = rng.uniform(*region.get("x", [0.3, 0.5]))
            y = rng.uniform(*region.get("y", [-0.2, 0.2]))
            z = region.get("z", [0.45, 0.45])[0]
            if hasattr(env, "add_object"):
                env.add_object(name, np.array([x, y, z]), mass=obj_cfg.get("mass", 0.1))

        return self.get_info()

    def check_success(self, obs: dict, env) -> bool:
        """Check whether blocks form a stable vertical stack in the correct order."""
        obj_states = getattr(env, "_object_states", {})
        if not self._stack_order:
            return False

        for i, name in enumerate(self._stack_order):
            if name not in obj_states:
                return False
            pos = obj_states[name][:3]
            expected_z = self._base_position[2] + i * self._BLOCK_HEIGHT
            expected_xy = self._base_position[:2]
            xy_dist = float(np.linalg.norm(pos[:2] - expected_xy))
            z_dist = abs(pos[2] - expected_z)
            if xy_dist > self._pos_tol or z_dist > self._pos_tol:
                self._stable_counter = 0
                return False

        self._stable_counter += 1
        return self._stable_counter >= self._stable_steps

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

        for i, name in enumerate(self._stack_order):
            if name not in obj_states:
                continue
            pos = obj_states[name][:3]
            expected_pos = self._base_position.copy()
            expected_pos[2] += i * self._BLOCK_HEIGHT

            # Approach reward for bottom-most unplaced block
            dist_to_obj = float(np.linalg.norm(ee_pose - pos))
            if dist_to_obj < 0.15:
                reward += 0.2 * (1.0 - dist_to_obj / 0.15)

            # Per-layer placement bonus
            dist_to_target = float(np.linalg.norm(pos - expected_pos))
            if dist_to_target < self._pos_tol and not self._layer_bonus_given.get(i, False):
                reward += 2.0 * (i + 1)  # higher layers worth more
                self._layer_bonus_given[i] = True

        # Height bonus: reward taller stacks
        max_z = max(
            (obj_states[n][2] for n in self._stack_order if n in obj_states),
            default=0.45,
        )
        reward += 0.1 * max(0.0, max_z - 0.5)

        # Success
        if self.check_success(obs, env):
            reward += 20.0

        if info.get("collision", False):
            reward -= 2.0
        reward -= 0.01
        return reward

    def get_skill_plan(self) -> list[dict]:
        plan = []
        for i, name in enumerate(self._stack_order):
            if i == 0:
                target = f"position:{self._base_position.tolist()}"
            else:
                target = self._stack_order[i - 1]
            plan += [
                {"skill": "move_to", "target": name,   "approach_height": 0.1},
                {"skill": "grasp",   "target": name},
                {"skill": "move_to", "target": target, "approach_height": self._BLOCK_HEIGHT * 1.2},
                {"skill": "release", "target": name},
            ]
        return plan

    def get_info(self) -> dict:
        base = super().get_info()
        base.update({
            "stack_order": self._stack_order,
            "stable_counter": self._stable_counter,
            "layers_placed": sum(self._layer_bonus_given.values()),
        })
        return base
