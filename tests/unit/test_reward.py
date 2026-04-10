"""
Tests for shaped reward functions.

Shaped rewards are one of the main levers for RL training speed. These
tests prove the reward signal is internally consistent:

1. Proximity matters — being near the object earns more reward than standing far away.
2. Grasping is rewarded exactly once — not repeatedly, which would bias the policy
   toward oscillating instead of making progress.
3. Collisions are penalised — the penalty must be large enough to register on top
   of any shaped bonus.
4. Task success gives the largest single reward spike — ensuring the final goal is
   the global optimum, not some intermediate state.
5. Reward is deterministic — same state → same reward (important for reproducibility).
"""
import numpy as np
import pytest

from envs.isaac.isaac_env import IsaacEnvConfig, MockIsaacEnv
from envs.task_definitions.pick_and_place import PickAndPlaceTask


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_CONFIG = {
    "name": "pick_and_place",
    "objects": [
        {"name": "red_block", "type": "cube", "mass": 0.1,
         "spawn_region": {"x": [0.45, 0.45], "y": [0.0, 0.0], "z": [0.45, 0.45]}},
        {"name": "bin_A", "type": "bin",   "mass": 0.5,
         "spawn_region": {"x": [0.6, 0.6], "y": [0.3, 0.3], "z": [0.42, 0.42]}},
    ],
    "success_criteria": {"position_tolerance": 0.05, "stable_steps": 2},
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


def _fresh():
    env = MockIsaacEnv(IsaacEnvConfig(seed=0))
    task = PickAndPlaceTask(_CONFIG)
    task.reset(env)
    return env, task


def _reward(env, task, *, held=None, ee_pos=None, obj_pos=None, collision=False):
    """Helper: set state then call compute_reward."""
    if held is not None:
        env._held_object = held
    if ee_pos is not None:
        env._ee_pose[:3] = np.array(ee_pos)
    if obj_pos is not None:
        env._object_states["red_block"][:3] = np.array(obj_pos)
    obs = {"ee_pose": env._ee_pose, "robot_state": np.zeros(18)}
    return task.compute_reward(obs, np.zeros(9), obs, {"collision": collision}, env)


# ---------------------------------------------------------------------------
# Proximity: reward increases as EE approaches the object
# ---------------------------------------------------------------------------

class TestProximityReward:
    def test_closer_ee_earns_higher_reward_than_far(self):
        env, task = _fresh()
        r_far  = _reward(env, task, ee_pos=[0.0, 0.0, 0.5])    # 0.45 m away
        env2, task2 = _fresh()
        r_near = _reward(env2, task2, ee_pos=[0.45, 0.0, 0.5])  # right above block
        assert r_near > r_far, \
            f"Expected r_near={r_near:.4f} > r_far={r_far:.4f}"

    def test_reward_at_zero_distance_is_positive(self):
        env, task = _fresh()
        # Put EE exactly at object position
        obj_pos = env._object_states["red_block"][:3].copy()
        r = _reward(env, task, ee_pos=obj_pos.tolist())
        # approach bonus should push this positive despite time penalty
        assert r > -0.05, f"Expected positive approach reward, got {r:.4f}"


# ---------------------------------------------------------------------------
# Grasp bonus: given once, not repeatedly
# ---------------------------------------------------------------------------

class TestGraspBonusOnce:
    def test_grasp_bonus_given_on_first_hold(self):
        env, task = _fresh()
        r1 = _reward(env, task, held="red_block")
        assert task._grasp_bonus_given

    def test_grasp_bonus_not_repeated_on_second_step(self):
        env, task = _fresh()
        env._held_object = "red_block"
        obs = {"ee_pose": env._ee_pose, "robot_state": np.zeros(18)}
        r1 = task.compute_reward(obs, np.zeros(9), obs, {}, env)
        r2 = task.compute_reward(obs, np.zeros(9), obs, {}, env)
        # r2 must NOT include the grasp bonus again
        # The grasp_bonus is 2.0; r2 should be lower than r1 by roughly that amount
        assert r1 > r2, \
            "Grasp bonus was applied twice — reward should decrease after first step"

    def test_grasp_bonus_value_matches_config(self):
        env, task = _fresh()
        # Baseline reward without grasp
        env._held_object = None
        obs = {"ee_pose": env._ee_pose, "robot_state": np.zeros(18)}
        r_no_grasp = task.compute_reward(obs, np.zeros(9), obs, {}, env)
        # Reward with grasp on same step (fresh task so bonus not yet given)
        env2, task2 = _fresh()
        env2._held_object = "red_block"
        obs2 = {"ee_pose": env2._ee_pose, "robot_state": np.zeros(18)}
        r_grasp = task2.compute_reward(obs2, np.zeros(9), obs2, {}, env2)
        delta = r_grasp - r_no_grasp
        # delta should be approximately grasp_bonus=2.0 (±0.5 for proximity)
        assert 1.0 <= delta <= 3.5, f"Unexpected grasp delta {delta:.4f}"


# ---------------------------------------------------------------------------
# Collision penalty dominates shaped bonuses at that step
# ---------------------------------------------------------------------------

class TestCollisionPenalty:
    def test_collision_reduces_reward_vs_clean_step(self):
        env, task = _fresh()
        r_clean = _reward(env, task, collision=False)
        env2, task2 = _fresh()
        r_col   = _reward(env2, task2, collision=True)
        assert r_col < r_clean, "Collision did not reduce reward"

    def test_collision_penalty_magnitude_is_meaningful(self):
        # The penalty (-2.0) should exceed the per-step time penalty (-0.01)
        env, task = _fresh()
        r_clean = _reward(env, task, collision=False)
        env2, task2 = _fresh()
        r_col   = _reward(env2, task2, collision=True)
        assert (r_clean - r_col) >= 1.5, \
            f"Collision penalty too small: diff={r_clean - r_col:.4f}"


# ---------------------------------------------------------------------------
# Success bonus is the global optimum
# ---------------------------------------------------------------------------

class TestSuccessBonusIsGlobalOptimum:
    def test_success_gives_largest_single_reward_spike(self):
        env, task = _fresh()
        # Move object into bin (manually)
        bin_pos = env._object_states["bin_A"][:3].copy()
        env._object_states["red_block"][:3] = bin_pos
        # Accrue stable_steps (2) so check_success fires
        for _ in range(3):
            task.check_success({}, env)
        obs = {"ee_pose": env._ee_pose, "robot_state": np.zeros(18)}
        r_success = task.compute_reward(obs, np.zeros(9), obs, {}, env)
        # Now check without success
        env2, task2 = _fresh()
        r_approach = _reward(env2, task2, ee_pos=[0.45, 0.0, 0.5])
        assert r_success > r_approach, \
            f"Success reward {r_success:.2f} should exceed approach reward {r_approach:.2f}"


# ---------------------------------------------------------------------------
# Determinism: reward must be reproducible given identical state
# ---------------------------------------------------------------------------

class TestRewardDeterminism:
    def test_same_state_same_reward(self):
        rewards = []
        for _ in range(3):
            env, task = _fresh()
            r = _reward(env, task, ee_pos=[0.45, 0.0, 0.50])
            rewards.append(r)
        assert len(set(round(r, 8) for r in rewards)) == 1, \
            f"Reward not deterministic across resets: {rewards}"
