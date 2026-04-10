"""
Tests for the skill router.

The skill router is the bridge between the planner's high-level language
and the RL controller's low-level actions. These tests prove:

1. Skill ordering is respected — the router executes steps in plan order
   and stops at the first failure (it doesn't skip ahead).
2. Replanning actually fires — when a skill fails and a replan callback is
   registered, the new plan replaces the remainder of the old one.
3. Replanning is bounded — the router won't loop forever on repeated failures.
4. Custom skills extend the router without touching existing code.
5. Total reward is accumulated correctly across skills.
"""
import numpy as np
import pytest

from envs.isaac.isaac_env import IsaacEnvConfig, MockIsaacEnv
from src.planner.base_planner import PlanResult
from src.skill_router.skill_router import SkillResult, SkillRouter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    e = MockIsaacEnv(IsaacEnvConfig(seed=0))
    e.reset()
    e.add_object("red_block", np.array([0.45, 0.0, 0.45]))
    e.add_object("bin_A",     np.array([0.60, 0.3, 0.42]))
    return e


def _plan(*steps) -> PlanResult:
    return PlanResult(steps=list(steps))


def _wait(duration=0.05):
    return {"skill": "wait", "target": None, "duration": duration}


# ---------------------------------------------------------------------------
# Execution ordering: steps run in sequence, stop at first failure
# ---------------------------------------------------------------------------

class TestExecutionOrder:
    def test_steps_run_in_declared_order(self, env):
        execution_log = []

        def make_skill(name):
            def _skill(params, e, state):
                execution_log.append(name)
                return SkillResult(skill=name, target=None, success=True,
                                   steps_taken=1, reward=1.0)
            return _skill

        router = SkillRouter()
        router.register_skill("first",  make_skill("first"))
        router.register_skill("second", make_skill("second"))
        router.register_skill("third",  make_skill("third"))

        plan = _plan(
            {"skill": "first",  "target": None},
            {"skill": "second", "target": None},
            {"skill": "third",  "target": None},
        )
        router.execute(plan, env)
        assert execution_log == ["first", "second", "third"]

    def test_execution_stops_at_first_failure(self, env):
        executed = []

        def fail_skill(params, e, state):
            executed.append("fail")
            return SkillResult(skill="fail", target=None, success=False,
                               steps_taken=1, reward=0.0, failure_reason="deliberate")

        def after_skill(params, e, state):
            executed.append("after")
            return SkillResult(skill="after", target=None, success=True,
                               steps_taken=1, reward=1.0)

        router = SkillRouter()   # no replan callback
        router.register_skill("fail",  fail_skill)
        router.register_skill("after", after_skill)

        plan = _plan({"skill": "fail", "target": None},
                     {"skill": "after", "target": None})
        result = router.execute(plan, env)

        assert "after" not in executed, "Router should stop after first failure"
        assert result.failure_step == 0

    def test_invalid_plan_never_executes(self, env):
        executed = []

        def track(params, e, state):
            executed.append(True)
            return SkillResult(skill="x", target=None, success=True,
                               steps_taken=1, reward=0.0)

        router = SkillRouter()
        router.register_skill("x", track)
        router.execute(PlanResult.invalid("bad"), env)
        assert not executed


# ---------------------------------------------------------------------------
# Replanning: callback fires on failure and its plan replaces remainder
# ---------------------------------------------------------------------------

class TestReplanningFiresOnFailure:
    def test_replan_callback_called_on_skill_failure(self, env):
        calls = {"n": 0}

        def replan(scene, goal):
            calls["n"] += 1
            return _plan(_wait(0.01))  # recovery plan: just wait

        fail_skill = lambda p, e, s: SkillResult("fail", None, False, 1, 0.0,
                                                  failure_reason="test failure")

        router = SkillRouter(replan_callback=replan, max_replan_attempts=2)
        router.register_skill("fail", fail_skill)
        router.execute(_plan({"skill": "fail", "target": None}), env)

        assert calls["n"] >= 1

    def test_replan_result_replaces_remaining_steps(self, env):
        recovery_executed = []

        def replan(scene, goal):
            return _plan({"skill": "recovery", "target": None})

        fail_skill    = lambda p, e, s: SkillResult("fail", None, False, 1, 0.0,
                                                     failure_reason="oops")
        recovery_skill = lambda p, e, s: (
            recovery_executed.append(True) or
            SkillResult("recovery", None, True, 1, 5.0)
        )

        router = SkillRouter(replan_callback=replan, max_replan_attempts=1)
        router.register_skill("fail",     fail_skill)
        router.register_skill("recovery", recovery_skill)

        result = router.execute(_plan({"skill": "fail", "target": None}), env)

        assert recovery_executed, "Recovery skill from replan was never executed"
        assert result.replan_count == 1

    def test_max_replan_attempts_is_a_hard_cap(self, env):
        replan_calls = {"n": 0}

        def always_fail_replan(scene, goal):
            replan_calls["n"] += 1
            return PlanResult.invalid("replan also failed")

        fail_skill = lambda p, e, s: SkillResult("fail", None, False, 1, 0.0,
                                                  failure_reason="x")

        router = SkillRouter(replan_callback=always_fail_replan, max_replan_attempts=3)
        router.register_skill("fail", fail_skill)
        router.execute(_plan({"skill": "fail", "target": None}), env)

        assert replan_calls["n"] <= 3, \
            f"Expected ≤3 replan attempts, got {replan_calls['n']}"


# ---------------------------------------------------------------------------
# Reward accumulation: total is the sum across all skills
# ---------------------------------------------------------------------------

class TestRewardAccumulation:
    def test_total_reward_is_sum_of_skill_rewards(self, env):
        rewards = [1.0, 2.5, 0.75]

        for i, r in enumerate(rewards):
            env.router_reward = r
            def make_skill(reward_val=r):
                def _skill(params, e, state):
                    return SkillResult(f"s{reward_val}", None, True, 1, reward_val)
                return _skill
            SkillRouter().register_skill  # just to show the pattern

        router = SkillRouter()
        for i, r in enumerate(rewards):
            def make_skill(rv=r, idx=i):
                def _skill(params, e, state):
                    return SkillResult(f"s{idx}", None, True, 1, rv)
                return _skill
            router.register_skill(f"s{i}", make_skill())

        plan = _plan(*[{"skill": f"s{i}", "target": None} for i in range(len(rewards))])
        result = router.execute(plan, env)

        assert abs(result.total_reward - sum(rewards)) < 1e-6, \
            f"Expected {sum(rewards)}, got {result.total_reward}"

    def test_built_in_wait_skill_accumulates_steps(self, env):
        router = SkillRouter()
        plan = _plan(_wait(0.05), _wait(0.05))
        result = router.execute(plan, env)
        # Each 0.05 s wait = 5 steps at 0.01 physics_dt
        assert result.total_steps >= 8


# ---------------------------------------------------------------------------
# Custom skill registration: extend without modifying existing code
# ---------------------------------------------------------------------------

class TestCustomSkillRegistration:
    def test_custom_skill_is_callable(self, env):
        invoked = {"yes": False}

        def my_skill(params, env, state):
            invoked["yes"] = True
            return SkillResult("my_skill", params.get("target"), True, 1, 99.0)

        router = SkillRouter()
        router.register_skill("my_skill", my_skill)
        result = router.execute(_plan({"skill": "my_skill", "target": "x"}), env)

        assert invoked["yes"]
        assert result.total_reward == 99.0
        assert result.plan_success

    def test_custom_skill_overwrites_builtin(self, env):
        """A registered skill with the same name as a built-in overrides it."""
        def loud_wait(params, e, state):
            return SkillResult("wait", None, True, 999, 42.0)

        router = SkillRouter()
        router.register_skill("wait", loud_wait)
        result = router.execute(_plan(_wait()), env)

        assert result.skill_results[0].steps_taken == 999
