"""
Safety filter tests: verify that safety-critical invariants hold regardless
of what the planner produces.
"""
import numpy as np
import pytest

from envs.isaac.isaac_env import IsaacEnvConfig, MockIsaacEnv
from src.safety.safety_checker import SafetyChecker
from src.planner.base_planner import BasePlanner, PlanResult
from src.planner.llm_planner import MockLLMPlanner
from src.perception.scene_parser import SceneState, ObjectInfo


def _make_scene_state(object_names: list[str]) -> SceneState:
    objects = {}
    for name in object_names:
        objects[name] = ObjectInfo(
            name=name,
            position=np.array([0.4, 0.0, 0.45]),
            orientation=np.array([0., 0., 0., 1.]),
            confidence=1.0,
            is_graspable=True,
            is_container=False,
        )
    return SceneState(objects=objects)


class TestSafetyFilterPlanner:
    """The safety checker must reject plans targeting objects not in the scene."""

    def test_reject_plan_with_unknown_target(self):
        checker = SafetyChecker()
        scene = _make_scene_state(["red_block"])
        step = {"skill": "grasp", "target": "invisible_cube"}
        valid, err = checker.validate_plan_step(step, scene)
        assert not valid
        assert "invisible_cube" in err

    def test_accept_plan_with_known_target(self):
        checker = SafetyChecker()
        scene = _make_scene_state(["red_block"])
        step = {"skill": "grasp", "target": "red_block"}
        valid, err = checker.validate_plan_step(step, scene)
        assert valid, err

    def test_wait_skill_always_accepted(self):
        checker = SafetyChecker()
        scene = _make_scene_state([])
        step = {"skill": "wait", "target": None, "duration": 1.0}
        valid, err = checker.validate_plan_step(step, scene)
        assert valid

    def test_reset_skill_always_accepted(self):
        checker = SafetyChecker()
        scene = _make_scene_state([])
        step = {"skill": "reset", "target": None}
        valid, err = checker.validate_plan_step(step, scene)
        assert valid


class TestControlAgentHaltsOnCollision:
    """The safety checker must signal abort after max_collision_count collisions."""

    def test_abort_after_max_collisions(self):
        env = MockIsaacEnv(IsaacEnvConfig(seed=0))
        env.reset()
        checker = SafetyChecker(max_collision_count=2, abort_on_critical=True)
        env._collision = True

        results = [checker.check(env) for _ in range(3)]
        # At least one result must have should_abort == True
        assert any(r.should_abort for r in results)

    def test_no_abort_below_max_collisions(self):
        env = MockIsaacEnv(IsaacEnvConfig(seed=0))
        env.reset()
        checker = SafetyChecker(max_collision_count=5, abort_on_critical=True)
        env._collision = True
        result = checker.check(env)
        # Only 1 collision — should not abort yet
        assert not result.should_abort


class TestTimeoutHandling:
    """Episode truncation from the gym wrapper simulates timeout."""

    def test_episode_truncated_at_max_steps(self):
        from envs.wrappers.gym_wrapper import RobotGymWrapper
        env = MockIsaacEnv(IsaacEnvConfig(seed=42))
        env.reset()
        env.add_object("red_block", np.array([0.45, 0.0, 0.45]))
        gym_env = RobotGymWrapper(env=env, max_episode_steps=5)
        gym_env.reset()
        action = np.zeros(9)
        truncated = False
        for _ in range(10):
            _, _, _, truncated, _ = gym_env.step(action)
            if truncated:
                break
        assert truncated, "Episode should have been truncated at max_episode_steps=5"


class TestInvalidObjectTargetHandling:
    """SkillRouter must not crash when given an invalid target object."""

    def test_router_does_not_raise_on_invalid_target(self):
        from src.skill_router.skill_router import SkillRouter
        from src.planner.base_planner import PlanResult
        env = MockIsaacEnv(IsaacEnvConfig(seed=42))
        env.reset()
        router = SkillRouter()
        plan = PlanResult(steps=[{"skill": "grasp", "target": "completely_nonexistent"}])
        # Must not raise
        result = router.execute(plan, env)
        assert not result.plan_success
        assert len(result.skill_results) == 1
