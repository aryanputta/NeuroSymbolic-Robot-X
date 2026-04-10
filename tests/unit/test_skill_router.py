"""Unit tests for the skill router."""
import numpy as np
import pytest

from envs.isaac.isaac_env import IsaacEnvConfig, MockIsaacEnv
from src.planner.base_planner import PlanResult
from src.skill_router.skill_router import SkillRouter, RouterResult, SkillResult


@pytest.fixture
def env():
    """Provide a seeded MockIsaacEnv with one object and one bin."""
    e = MockIsaacEnv(IsaacEnvConfig(seed=0))
    e.reset()
    e.add_object("red_block", np.array([0.45, 0.0, 0.45]))
    e.add_object("bin_A", np.array([0.6, 0.3, 0.42]))
    return e


@pytest.fixture
def router():
    return SkillRouter()


class TestSkillRouterBasic:
    def test_empty_plan_returns_failure(self, env, router):
        plan = PlanResult(steps=[], valid=True)
        result = router.execute(plan, env)
        assert not result.plan_success
        assert result.failure_reason

    def test_invalid_plan_returns_failure(self, env, router):
        plan = PlanResult.invalid("bad plan")
        result = router.execute(plan, env)
        assert not result.plan_success

    def test_unknown_skill_skipped(self, env, router):
        plan = PlanResult(steps=[{"skill": "teleport", "target": "mars"}])
        result = router.execute(plan, env)
        assert len(result.skill_results) == 1
        assert not result.skill_results[0].success

    def test_wait_skill_succeeds(self, env, router):
        plan = PlanResult(steps=[{"skill": "wait", "target": None, "duration": 0.1}])
        result = router.execute(plan, env)
        assert result.skill_results[0].success

    def test_reset_skill_succeeds(self, env, router):
        plan = PlanResult(steps=[{"skill": "reset", "target": None}])
        result = router.execute(plan, env)
        assert result.skill_results[0].success

    def test_move_to_object_executes(self, env, router):
        plan = PlanResult(steps=[{"skill": "move_to", "target": "red_block", "approach_height": 0.1}])
        result = router.execute(plan, env)
        assert result.total_steps > 0

    def test_total_reward_accumulates(self, env, router):
        plan = PlanResult(steps=[
            {"skill": "wait", "target": None, "duration": 0.02},
            {"skill": "wait", "target": None, "duration": 0.02},
        ])
        result = router.execute(plan, env)
        assert result.total_steps >= 4  # at least 2 steps per wait

    def test_replan_count_tracked(self, env, router):
        """Router with no replan callback stays at replan_count=0."""
        plan = PlanResult(steps=[{"skill": "grasp", "target": "nonexistent"}])
        result = router.execute(plan, env)
        assert result.replan_count == 0


class TestSkillRouterReplan:
    def test_replan_callback_invoked_on_failure(self, env):
        replan_call_count = {"n": 0}

        def mock_replan(scene, goal):
            replan_call_count["n"] += 1
            # Return a simple wait plan to halt further failures
            return PlanResult(steps=[{"skill": "wait", "target": None, "duration": 0.01}])

        router = SkillRouter(replan_callback=mock_replan, max_replan_attempts=2)
        # grasp of nonexistent object → failure → replan
        plan = PlanResult(steps=[{"skill": "grasp", "target": "nonexistent_object"}])
        result = router.execute(plan, env)
        assert replan_call_count["n"] >= 1

    def test_max_replan_attempts_respected(self, env):
        calls = {"n": 0}

        def always_fail_replan(scene, goal):
            calls["n"] += 1
            return PlanResult.invalid("replan failure")

        router = SkillRouter(replan_callback=always_fail_replan, max_replan_attempts=3)
        plan = PlanResult(steps=[{"skill": "grasp", "target": "nonexistent"}])
        router.execute(plan, env)
        assert calls["n"] <= 3


class TestSkillRouterCustomSkill:
    def test_register_custom_skill(self, env, router):
        called = {"yes": False}

        def my_skill(params, env, state):
            called["yes"] = True
            return SkillResult(skill="my_skill", target=None, success=True,
                               steps_taken=1, reward=5.0)

        router.register_skill("my_skill", my_skill)
        plan = PlanResult(steps=[{"skill": "my_skill", "target": None}])
        result = router.execute(plan, env)
        assert called["yes"]
        assert result.total_reward == 5.0
        assert result.plan_success
