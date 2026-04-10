"""
Regression tests: verify that task success detection is stable.

These tests prevent silent regressions where a code change causes
success to never trigger (success_rate = 0) or always trigger (= 1).
"""
import numpy as np
import pytest

from envs.isaac.isaac_env import IsaacEnvConfig, MockIsaacEnv
from envs.task_definitions.pick_and_place import PickAndPlaceTask
from envs.task_definitions.sort_objects import SortObjectsTask
from envs.task_definitions.stack_blocks import StackBlocksTask


def _basic_pnp_config():
    return {
        "name": "pick_and_place",
        "objects": [
            {"name": "red_block", "type": "cube", "mass": 0.1,
             "spawn_region": {"x": [0.45, 0.45], "y": [0.0, 0.0], "z": [0.45, 0.45]}},
            {"name": "bin_A", "type": "bin", "mass": 0.5,
             "spawn_region": {"x": [0.6, 0.6], "y": [0.3, 0.3], "z": [0.42, 0.42]}},
        ],
        "success_criteria": {"position_tolerance": 0.05, "stable_steps": 2},
        "reward_shaping": {},
        "instruction_templates": ["Pick up the red block."],
    }


class TestPickAndPlaceSuccessRegression:
    def test_success_not_triggered_at_start(self):
        """Objects start far from the bin — success should be False initially."""
        env = MockIsaacEnv(IsaacEnvConfig(seed=42))
        task = PickAndPlaceTask(_basic_pnp_config())
        task.reset(env)
        assert not task.check_success({}, env)

    def test_success_triggered_when_object_in_bin(self):
        """Manually place object in bin — success should trigger."""
        env = MockIsaacEnv(IsaacEnvConfig(seed=42))
        task = PickAndPlaceTask(_basic_pnp_config())
        task.reset(env)
        # Move red_block to the same position as bin_A
        bin_pos = env._object_states["bin_A"][:3].copy()
        env._object_states["red_block"][:3] = bin_pos
        # Need stable_steps=2 consecutive successes
        for _ in range(3):
            task.check_success({}, env)
        assert task.check_success({}, env)

    def test_success_resets_after_object_leaves_bin(self):
        """If the object moves away mid-episode, success counter resets."""
        env = MockIsaacEnv(IsaacEnvConfig(seed=42))
        task = PickAndPlaceTask(_basic_pnp_config())
        task.reset(env)
        bin_pos = env._object_states["bin_A"][:3].copy()
        # Place in bin for 1 step (not enough for stable_steps=2)
        env._object_states["red_block"][:3] = bin_pos
        task.check_success({}, env)
        # Move away
        env._object_states["red_block"][:3] = np.array([0.3, 0.0, 0.45])
        task.check_success({}, env)
        assert task._success_steps == 0

    def test_info_returns_expected_keys(self):
        env = MockIsaacEnv(IsaacEnvConfig(seed=42))
        task = PickAndPlaceTask(_basic_pnp_config())
        task.reset(env)
        info = task.get_info()
        assert "target_object" in info
        assert "goal_bin" in info


class TestSortObjectsSuccessRegression:
    def test_success_false_at_start(self):
        config = {
            "name": "sort_objects",
            "objects": [
                {"name": "red_block_1", "type": "cube", "mass": 0.1, "category": "red",
                 "spawn_region": {"x": [0.4, 0.4], "y": [0.0, 0.0], "z": [0.45, 0.45]}},
                {"name": "red_bin", "type": "bin", "mass": 0.5,
                 "spawn_region": {"x": [0.6, 0.6], "y": [-0.3, -0.3], "z": [0.42, 0.42]},
                 "accepts": "red"},
            ],
            "success_criteria": {"position_tolerance": 0.06, "stable_steps": 2},
        }
        env = MockIsaacEnv(IsaacEnvConfig(seed=42))
        task = SortObjectsTask(config)
        task.reset(env)
        assert not task.check_success({}, env)

    def test_all_sorted_triggers_success(self):
        config = {
            "name": "sort_objects",
            "objects": [
                {"name": "red_block_1", "type": "cube", "mass": 0.1, "category": "red",
                 "spawn_region": {"x": [0.4, 0.4], "y": [0.0, 0.0], "z": [0.45, 0.45]}},
                {"name": "red_bin", "type": "bin", "mass": 0.5,
                 "spawn_region": {"x": [0.6, 0.6], "y": [-0.3, -0.3], "z": [0.42, 0.42]},
                 "accepts": "red"},
            ],
            "success_criteria": {"position_tolerance": 0.06, "stable_steps": 2},
        }
        env = MockIsaacEnv(IsaacEnvConfig(seed=42))
        task = SortObjectsTask(config)
        task.reset(env)
        bin_pos = env._object_states["red_bin"][:3].copy()
        env._object_states["red_block_1"][:3] = bin_pos
        for _ in range(3):
            task.check_success({}, env)
        assert task.check_success({}, env)
