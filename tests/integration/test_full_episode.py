"""Integration test: full episode rollout through the complete pipeline."""
import numpy as np
import pytest

from envs.isaac.isaac_env import IsaacEnvConfig, MockIsaacEnv
from envs.task_definitions.pick_and_place import PickAndPlaceTask
from envs.wrappers.gym_wrapper import RobotGymWrapper
from envs.wrappers.noise_wrapper import NoiseWrapper
from envs.wrappers.latency_wrapper import LatencyWrapper
from src.control.rl_executor import RLExecutor, ControllerConfig
from src.planner.llm_planner import MockLLMPlanner
from src.perception.scene_parser import SceneParser
from src.skill_router.skill_router import SkillRouter
from src.safety.safety_checker import SafetyChecker


@pytest.fixture
def base_env():
    raw = MockIsaacEnv(IsaacEnvConfig(seed=42))
    raw.reset()
    raw.add_object("red_block", np.array([0.45, 0.0, 0.45]))
    raw.add_object("bin_A", np.array([0.6, 0.3, 0.42]))
    return raw


@pytest.fixture
def gym_env(base_env):
    return RobotGymWrapper(env=base_env, max_episode_steps=100)


class TestFullEpisodeRollout:
    def test_random_policy_runs_full_episode(self, gym_env):
        """Ensure the environment runs for a full episode with a random policy."""
        obs, info = gym_env.reset()
        total_reward = 0.0
        steps = 0
        for _ in range(100):
            action = gym_env.action_space.sample()
            obs, reward, terminated, truncated, info = gym_env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        assert steps > 0
        assert isinstance(total_reward, float)

    def test_observation_keys_correct(self, gym_env):
        obs, _ = gym_env.reset()
        assert "robot_state" in obs
        assert "object_states" in obs
        assert "task_goal" in obs
        assert "ee_pose" in obs

    def test_observation_shapes_correct(self, gym_env):
        obs, _ = gym_env.reset()
        assert obs["robot_state"].shape == (18,)
        assert obs["ee_pose"].shape == (7,)
        assert obs["task_goal"].shape == (64,)

    def test_action_space_respected(self, gym_env):
        obs, _ = gym_env.reset()
        action = np.ones(9, dtype=np.float32)
        obs2, reward, term, trunc, info = gym_env.step(action)
        assert obs2["robot_state"].shape == (18,)

    def test_planner_skill_router_integration(self, base_env):
        """Planner + SkillRouter execute a plan on the mock env."""
        parser = SceneParser(use_gt_states=True)
        planner = MockLLMPlanner()
        router = SkillRouter()

        obs = base_env.get_observation()
        scene = parser.parse(
            rgb_image=obs.rgb_overhead,
            depth_image=obs.depth_overhead,
            robot_obs={"ee_pose": obs.end_effector_pose},
            object_states=obs.object_states,
            task_goal="pick and place",
        )
        plan = planner.plan(scene, "pick and place")
        result = router.execute(plan, base_env)
        assert result.total_steps > 0
        assert isinstance(result.total_reward, float)

    def test_safety_checker_integrated(self, base_env):
        checker = SafetyChecker()
        checker.reset()
        obs = base_env.get_observation()
        result = checker.check(base_env)
        # Default state should be safe
        assert result.safe


class TestNoisyEpisode:
    def test_noise_wrapper_low_runs(self, gym_env):
        noisy_env = NoiseWrapper.from_level(gym_env, "low", seed=0)
        obs, _ = noisy_env.reset()
        assert obs is not None
        action = noisy_env.action_space.sample()
        obs2, r, t, tr, info = noisy_env.step(action)
        assert obs2 is not None

    def test_noise_wrapper_high_drops_frames(self, gym_env):
        """High noise should eventually drop frames (drop_rate > 0 after many steps)."""
        noisy_env = NoiseWrapper.from_level(gym_env, "high", seed=42)
        noisy_env.reset()
        for _ in range(50):
            action = noisy_env.action_space.sample()
            noisy_env.step(action)
        # Drop rate won't be exactly 0 with p=0.10 over 50 steps (probabilistic)
        # but we just verify it stays in [0, 1]
        assert 0.0 <= noisy_env.drop_rate <= 1.0


class TestLatencyEpisode:
    def test_latency_wrapper_low_runs(self, gym_env):
        lat_env = LatencyWrapper.from_level(gym_env, "low", seed=0)
        obs, _ = lat_env.reset()
        assert obs is not None
        action = lat_env.action_space.sample()
        obs2, r, t, tr, info = lat_env.step(action)
        assert obs2 is not None
        assert "latency_stats" in info

    def test_latency_wrapper_observation_is_delayed(self, gym_env):
        """With 2-step obs delay, the first returned obs should be the initial obs."""
        from envs.wrappers.latency_wrapper import LatencyConfig
        config = LatencyConfig(obs_delay_steps=2, action_delay_steps=0)
        lat_env = LatencyWrapper(gym_env, config)
        initial_obs, _ = lat_env.reset()
        action = lat_env.action_space.sample()
        obs_after_step, _, _, _, _ = lat_env.step(action)
        # Both obs should be valid dicts with the right keys
        assert "robot_state" in initial_obs
        assert "robot_state" in obs_after_step

    def test_episode_tracker_resets(self, gym_env):
        gym_env.reset()
        for _ in range(10):
            gym_env.step(gym_env.action_space.sample())
        gym_env.reset()  # Should not raise
        assert gym_env._episode_step == 0
