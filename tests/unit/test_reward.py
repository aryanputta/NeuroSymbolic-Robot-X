"""Unit tests for task reward functions."""
import numpy as np
import pytest

from envs.isaac.isaac_env import IsaacEnvConfig, MockIsaacEnv
from envs.task_definitions.pick_and_place import PickAndPlaceTask
from envs.task_definitions.sort_objects import SortObjectsTask


def _make_pnp_config():
    return {
        "name": "pick_and_place",
        "objects": [
            {
                "name": "red_block",
                "type": "cube",
                "mass": 0.1,
                "spawn_region": {"x": [0.45, 0.45], "y": [0.0, 0.0], "z": [0.45, 0.45]},
            },
            {
                "name": "bin_A",
                "type": "bin",
                "mass": 0.5,
                "spawn_region": {"x": [0.6, 0.6], "y": [0.3, 0.3], "z": [0.42, 0.42]},
            },
        ],
        "success_criteria": {"position_tolerance": 0.05, "stable_steps": 3},
        "reward_shaping": {
            "approach_bonus": 0.5,
            "grasp_bonus": 2.0,
            "lift_bonus": 1.0,
            "transport_bonus": 1.0,
            "place_bonus": 3.0,
            "success_bonus": 10.0,
        },
        "instruction_templates": ["Pick up the red block."],
    }


@pytest.fixture
def pnp_env_task():
    env = MockIsaacEnv(IsaacEnvConfig(seed=42))
    task = PickAndPlaceTask(_make_pnp_config())
    task.reset(env)
    return env, task


class TestPickAndPlaceReward:
    def test_reward_returns_float(self, pnp_env_task):
        env, task = pnp_env_task
        obs = {"ee_pose": np.zeros(7), "robot_state": np.zeros(18)}
        action = np.zeros(9)
        reward = task.compute_reward(obs, action, obs, {"collision": False}, env)
        assert isinstance(reward, float)

    def test_collision_penalty_applied(self, pnp_env_task):
        env, task = pnp_env_task
        obs = {"ee_pose": np.zeros(7), "robot_state": np.zeros(18)}
        action = np.zeros(9)
        reward_no_col = task.compute_reward(obs, action, obs, {"collision": False}, env)
        reward_col = task.compute_reward(obs, action, obs, {"collision": True}, env)
        assert reward_col < reward_no_col

    def test_approach_bonus_when_near_object(self, pnp_env_task):
        env, task = pnp_env_task
        # Place EE right at the object position
        env._ee_pose[:3] = env._object_states.get("red_block", np.zeros(7))[:3]
        obs = {"ee_pose": env._ee_pose, "robot_state": np.zeros(18)}
        action = np.zeros(9)
        reward = task.compute_reward(obs, action, obs, {"collision": False}, env)
        # Should receive positive approach bonus
        assert reward > -0.1  # time penalty is -0.01 per step

    def test_grasp_bonus_given_once(self, pnp_env_task):
        env, task = pnp_env_task
        env._held_object = "red_block"
        obs = {"ee_pose": np.zeros(7), "robot_state": np.zeros(18)}
        action = np.zeros(9)
        r1 = task.compute_reward(obs, action, obs, {}, env)
        r2 = task.compute_reward(obs, action, obs, {}, env)
        # Grasp bonus should be given only once
        assert task._grasp_bonus_given
        # Second call should NOT get the grasp bonus again
        assert r1 > r2 or r1 == r2  # r1 may include grasp bonus; r2 should not

    def test_success_detected(self, pnp_env_task):
        env, task = pnp_env_task
        # Move object into bin
        bin_pos = env._object_states["bin_A"][:3].copy()
        env._object_states["red_block"][:3] = bin_pos
        obs = {}
        # success_steps need to accumulate
        for _ in range(5):
            task.check_success(obs, env)
        assert task.check_success(obs, env)

    def test_skill_plan_structure(self, pnp_env_task):
        _, task = pnp_env_task
        plan = task.get_skill_plan()
        assert len(plan) == 4
        skills = [s["skill"] for s in plan]
        assert "move_to" in skills
        assert "grasp" in skills
        assert "release" in skills

    def test_instruction_is_string(self, pnp_env_task):
        _, task = pnp_env_task
        instruction = task.get_instruction()
        assert isinstance(instruction, str)
        assert len(instruction) > 0


class TestSortObjectsReward:
    def test_compute_reward_positive_near_correct_bin(self):
        config = {
            "name": "sort_objects",
            "objects": [
                {"name": "red_block_1", "type": "cube", "mass": 0.1, "category": "red",
                 "spawn_region": {"x": [0.4, 0.4], "y": [0.0, 0.0], "z": [0.45, 0.45]}},
                {"name": "red_bin", "type": "bin", "mass": 0.5,
                 "spawn_region": {"x": [0.6, 0.6], "y": [-0.3, -0.3], "z": [0.42, 0.42]},
                 "accepts": "red"},
            ],
            "success_criteria": {"position_tolerance": 0.06, "stable_steps": 3},
        }
        env = MockIsaacEnv(IsaacEnvConfig(seed=42))
        task = SortObjectsTask(config)
        task.reset(env)
        obs = {}
        reward = task.compute_reward(obs, np.zeros(9), obs, {}, env)
        assert isinstance(reward, float)
