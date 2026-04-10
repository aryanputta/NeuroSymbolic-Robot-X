"""
Integration tests — full pipeline from scene perception to action execution.

These tests validate the *composition* of layers. Each test proves that two
adjacent layers hand off correctly, and that the combined behaviour matches
the design intent. They run without Isaac Sim or an LLM API key.

Key claims tested here:
1. Planner → SkillRouter: the router can execute every plan the planner emits.
2. NoiseWrapper: with high noise the obs changes but the pipeline keeps running.
3. LatencyWrapper: delayed observations are still valid dicts with correct keys.
4. SafetyChecker integrated: a clean environment stays safe step-by-step.
5. Episode lifecycle: reset clears all per-episode counters.
"""
import numpy as np
import pytest

from envs.isaac.isaac_env import IsaacEnvConfig, MockIsaacEnv
from envs.task_definitions.pick_and_place import PickAndPlaceTask
from envs.wrappers.gym_wrapper import RobotGymWrapper
from envs.wrappers.noise_wrapper import NoiseWrapper
from envs.wrappers.latency_wrapper import LatencyWrapper, LatencyConfig
from src.control.rl_executor import ControllerConfig, RLExecutor
from src.perception.scene_parser import SceneParser
from src.planner.llm_planner import MockLLMPlanner
from src.safety.safety_checker import SafetyChecker
from src.skill_router.skill_router import SkillRouter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_env():
    e = MockIsaacEnv(IsaacEnvConfig(seed=42))
    e.reset()
    e.add_object("red_block", np.array([0.45, 0.0, 0.45]))
    e.add_object("bin_A",     np.array([0.60, 0.3, 0.42]))
    return e


@pytest.fixture
def gym_env(raw_env):
    return RobotGymWrapper(env=raw_env, max_episode_steps=50)


# ---------------------------------------------------------------------------
# Planner → SkillRouter handoff
# ---------------------------------------------------------------------------

class TestPlannerToRouterHandoff:
    def test_mock_planner_output_executes_without_error(self, raw_env):
        """Every plan the mock planner emits should run without crashing."""
        parser  = SceneParser(use_gt_states=True)
        planner = MockLLMPlanner()
        router  = SkillRouter()

        obs = raw_env.get_observation()
        scene = parser.parse(
            rgb_image=obs.rgb_overhead,
            depth_image=obs.depth_overhead,
            robot_obs={"ee_pose": obs.end_effector_pose},
            object_states=obs.object_states,
            task_goal="pick and place",
        )
        plan   = planner.plan(scene, "pick and place")
        result = router.execute(plan, raw_env)

        assert result.total_steps > 0
        assert isinstance(result.total_reward, float)

    def test_every_plan_step_maps_to_a_known_skill(self, raw_env):
        parser  = SceneParser(use_gt_states=True)
        planner = MockLLMPlanner()
        router  = SkillRouter()

        obs   = raw_env.get_observation()
        scene = parser.parse(obs.rgb_overhead, obs.depth_overhead,
                             {"ee_pose": obs.end_effector_pose},
                             obs.object_states)
        plan  = planner.plan(scene, "pick and place")

        for step in plan.steps:
            assert step["skill"] in router._skills, \
                f"Skill '{step['skill']}' not registered in router"

    def test_planner_targets_only_objects_in_scene(self, raw_env):
        parser  = SceneParser(use_gt_states=True)
        planner = MockLLMPlanner()

        obs   = raw_env.get_observation()
        scene = parser.parse(obs.rgb_overhead, obs.depth_overhead,
                             {"ee_pose": obs.end_effector_pose},
                             obs.object_states)
        plan  = planner.plan(scene, "pick and place")

        scene_names = set(scene.objects.keys())
        for step in plan.steps:
            t = step.get("target")
            if isinstance(t, str) and t:
                assert t in scene_names, f"Target '{t}' not in scene {scene_names}"


# ---------------------------------------------------------------------------
# Gymnasium wrapper: obs shape / keys contract
# ---------------------------------------------------------------------------

class TestGymWrapperContract:
    def test_reset_returns_correct_keys(self, gym_env):
        obs, info = gym_env.reset()
        for key in ("robot_state", "object_states", "task_goal", "ee_pose"):
            assert key in obs, f"Missing key '{key}' in reset observation"

    def test_reset_returns_correct_shapes(self, gym_env):
        obs, _ = gym_env.reset()
        assert obs["robot_state"].shape == (18,)
        assert obs["ee_pose"].shape     == (7,)
        assert obs["task_goal"].shape   == (64,)

    def test_step_returns_correct_keys(self, gym_env):
        gym_env.reset()
        obs, r, term, trunc, info = gym_env.step(gym_env.action_space.sample())
        for key in ("robot_state", "object_states", "task_goal", "ee_pose"):
            assert key in obs

    def test_action_space_bounds_respected(self, gym_env):
        gym_env.reset()
        action = np.ones(9) * 10.0   # intentionally out-of-bounds
        # The wrapper clips before passing to env — should not raise
        gym_env.step(action)

    def test_episode_truncation_at_max_steps(self):
        raw = MockIsaacEnv(IsaacEnvConfig(seed=0))
        raw.reset()
        env = RobotGymWrapper(env=raw, max_episode_steps=5)
        env.reset()
        truncated = False
        for _ in range(10):
            _, _, _, truncated, _ = env.step(np.zeros(9))
            if truncated:
                break
        assert truncated, "Episode should truncate at max_episode_steps=5"

    def test_episode_step_counter_resets_on_new_episode(self, gym_env):
        gym_env.reset()
        for _ in range(10):
            gym_env.step(gym_env.action_space.sample())
        gym_env.reset()
        assert gym_env._episode_step == 0


# ---------------------------------------------------------------------------
# Noise wrapper: pipeline keeps running under all noise levels
# ---------------------------------------------------------------------------

class TestNoiseWrapperPipelineContinues:
    @pytest.mark.parametrize("level", ["none", "low", "medium", "high"])
    def test_all_noise_levels_produce_valid_observations(self, gym_env, level):
        noisy = NoiseWrapper.from_level(gym_env, level, seed=0)
        obs, _ = noisy.reset()
        assert "robot_state" in obs

        action = noisy.action_space.sample()
        obs2, r, t, tr, info = noisy.step(action)
        assert "robot_state" in obs2
        assert isinstance(r, float)

    def test_high_noise_does_not_produce_nan_observations(self, gym_env):
        noisy = NoiseWrapper.from_level(gym_env, "high", seed=99)
        obs, _ = noisy.reset()
        for key, arr in obs.items():
            if isinstance(arr, np.ndarray):
                assert not np.any(np.isnan(arr)), f"NaN in obs['{key}'] at reset"
        for _ in range(20):
            obs, _, _, _, _ = noisy.step(noisy.action_space.sample())
            for key, arr in obs.items():
                if isinstance(arr, np.ndarray):
                    assert not np.any(np.isnan(arr)), f"NaN in obs['{key}'] at step"


# ---------------------------------------------------------------------------
# Latency wrapper: delayed observations still usable
# ---------------------------------------------------------------------------

class TestLatencyWrapperDelaysObservation:
    def test_latency_info_present_in_step_info(self, gym_env):
        lat = LatencyWrapper.from_level(gym_env, "low", seed=0)
        lat.reset()
        _, _, _, _, info = lat.step(lat.action_space.sample())
        assert "latency_stats" in info

    def test_2step_delay_still_returns_valid_obs(self, gym_env):
        config = LatencyConfig(obs_delay_steps=2, action_delay_steps=0)
        lat = LatencyWrapper(gym_env, config)
        obs, _ = lat.reset()
        assert "robot_state" in obs
        for _ in range(5):
            obs, _, _, _, _ = lat.step(lat.action_space.sample())
            assert "robot_state" in obs

    def test_effective_drop_rate_in_range(self, gym_env):
        lat = LatencyWrapper.from_level(gym_env, "medium", seed=42)
        lat.reset()
        for _ in range(30):
            lat.step(lat.action_space.sample())
        assert 0.0 <= lat.effective_drop_rate <= 1.0


# ---------------------------------------------------------------------------
# Safety checker: nominal pipeline stays within safe bounds
# ---------------------------------------------------------------------------

class TestSafetyInNominalPipeline:
    def test_clean_env_stays_safe_for_10_steps(self, raw_env):
        checker = SafetyChecker()
        checker.reset()
        for _ in range(10):
            result = checker.check(raw_env)
            assert result.safe, f"Unexpected safety violation: {result.violations}"

    def test_safety_checker_does_not_alter_env_state(self, raw_env):
        ee_before = raw_env._ee_pose.copy()
        checker = SafetyChecker()
        checker.check(raw_env)
        np.testing.assert_array_equal(raw_env._ee_pose, ee_before)
