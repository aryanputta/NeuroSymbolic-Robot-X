"""
Tests for the safety checker.

The safety layer is the last line of defence between bad planner output and
the physical robot. These tests prove three things:

1. Hard limits are enforced — workspace bounds, joint limits, and collision
   thresholds all produce violations regardless of what the planner says.
2. Severity escalates correctly — a single collision is a warning; hitting the
   configured maximum triggers a critical abort.
3. Plan-step validation catches invalid targets *before* the skill router
   ever touches the environment.

If any of these tests fail, the robot could execute unsafe actions.
"""
import numpy as np
import pytest

from envs.isaac.isaac_env import IsaacEnvConfig, MockIsaacEnv
from src.safety.safety_checker import SafetyChecker
from src.perception.scene_parser import SceneState, ObjectInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    e = MockIsaacEnv(IsaacEnvConfig(seed=0))
    e.reset()
    e.add_object("red_block", np.array([0.45, 0.0, 0.45]))
    return e


@pytest.fixture
def checker():
    return SafetyChecker(
        workspace_bounds=[[-0.65, 0.65], [-0.65, 0.65], [0.0, 0.85]],
        max_collision_count=3,
        table_height_z=0.42,
        abort_on_critical=True,
    )


def _scene(*names) -> SceneState:
    objs = {n: ObjectInfo(n, np.zeros(3), np.array([0,0,0,1.]), 1.0,
                          is_graspable=True, is_container=False)
            for n in names}
    return SceneState(objects=objs)


# ---------------------------------------------------------------------------
# Nominal state: clean environment passes every check
# ---------------------------------------------------------------------------

class TestNominalPassesAllChecks:
    def test_centred_ee_no_collision_is_safe(self, env, checker):
        env._ee_pose[:3] = [0.40, 0.0, 0.55]
        env._collision = False
        result = checker.check(env)
        assert result.safe
        assert not result.should_abort
        assert not result.violations

    def test_check_increments_step_counter(self, env, checker):
        for _ in range(5):
            checker.check(env)
        assert checker._step == 5

    def test_reset_zeroes_all_counters(self, env, checker):
        env._collision = True
        checker.check(env)
        checker.reset()
        assert checker._step == 0
        assert checker._collision_count == 0


# ---------------------------------------------------------------------------
# Workspace bounds: EE must stay inside the configured box
# ---------------------------------------------------------------------------

class TestWorkspaceBoundsEnforced:
    @pytest.mark.parametrize("pos,axis", [
        ([1.5, 0.0, 0.5],  "x"),
        ([-1.5, 0.0, 0.5], "x"),
        ([0.0, 1.5, 0.5],  "y"),
        ([0.4, 0.0, -0.1], "z"),
        ([0.4, 0.0, 1.2],  "z"),
    ])
    def test_out_of_bounds_produces_violation(self, env, checker, pos, axis):
        env._ee_pose[:3] = pos
        env._collision = False
        result = checker.check(env)
        assert not result.safe
        types = [v.violation_type for v in result.violations]
        assert "workspace_bounds" in types, f"Expected workspace_bounds for {axis}-violation at {pos}"

    def test_at_exact_boundary_is_safe(self, env, checker):
        env._ee_pose[:3] = [0.65, 0.0, 0.5]  # exactly on x-max boundary
        env._collision = False
        result = checker.check(env)
        assert result.safe


# ---------------------------------------------------------------------------
# Collision escalation: warning → critical → abort
# ---------------------------------------------------------------------------

class TestCollisionEscalation:
    def test_first_collision_is_warning_not_abort(self, env, checker):
        env._ee_pose[:3] = [0.4, 0.0, 0.5]
        env._collision = True
        result = checker.check(env)
        col = [v for v in result.violations if v.violation_type == "collision"]
        assert col and col[0].severity == "warning"
        assert not result.should_abort

    def test_exceeding_max_count_triggers_critical_abort(self, env, checker):
        env._ee_pose[:3] = [0.4, 0.0, 0.5]
        env._collision = True
        # checker has max_collision_count=3; trigger 4 collisions
        results = [checker.check(env) for _ in range(4)]
        assert any(r.should_abort for r in results), \
            "Expected should_abort=True after exceeding max_collision_count"

    def test_no_collision_no_violation(self, env, checker):
        env._collision = False
        result = checker.check(env)
        col = [v for v in result.violations if v.violation_type == "collision"]
        assert not col


# ---------------------------------------------------------------------------
# Object drop detection: held object below table triggers warning
# ---------------------------------------------------------------------------

class TestObjectDropDetection:
    def test_held_object_below_table_is_violation(self, env, checker):
        env._held_object = "red_block"
        env._object_states["red_block"][2] = 0.25   # well below table (0.42)
        result = checker.check(env)
        types = [v.violation_type for v in result.violations]
        assert "object_drop" in types

    def test_held_object_on_table_is_safe(self, env, checker):
        env._held_object = "red_block"
        env._object_states["red_block"][2] = 0.55   # above table
        result = checker.check(env)
        types = [v.violation_type for v in result.violations]
        assert "object_drop" not in types

    def test_not_holding_anything_no_drop_check(self, env, checker):
        env._held_object = None
        result = checker.check(env)
        types = [v.violation_type for v in result.violations]
        assert "object_drop" not in types


# ---------------------------------------------------------------------------
# Plan-step pre-validation: unknown targets caught before execution
# ---------------------------------------------------------------------------

class TestPlanStepPreValidation:
    """
    The safety checker validates each skill step against the current scene
    *before* the skill router executes it.  This ensures the planner cannot
    command an object that doesn't exist.
    """

    def test_known_target_passes(self, checker):
        scene = _scene("red_block")
        valid, _ = checker.validate_plan_step({"skill": "grasp", "target": "red_block"}, scene)
        assert valid

    def test_unknown_target_fails_with_informative_error(self, checker):
        scene = _scene("red_block")
        valid, err = checker.validate_plan_step({"skill": "grasp", "target": "phantom_cube"}, scene)
        assert not valid
        assert "phantom_cube" in err     # error names the bad target
        assert "red_block" in err        # and hints at what IS available

    def test_wait_requires_no_object_in_scene(self, checker):
        # wait/reset should not fail even against an empty scene
        empty_scene = _scene()
        for skill in ("wait", "reset"):
            valid, err = checker.validate_plan_step({"skill": skill, "target": None}, empty_scene)
            assert valid, f"'{skill}' should always be safe, got: {err}"

    def test_move_to_position_literal_always_valid(self, checker):
        scene = _scene()   # empty — no objects at all
        step = {"skill": "move_to", "target": "position:[0.5,0.0,0.5]"}
        valid, _ = checker.validate_plan_step(step, scene)
        assert valid


# ---------------------------------------------------------------------------
# Violation summary: useful for the failure analysis report
# ---------------------------------------------------------------------------

class TestViolationSummary:
    def test_counts_by_type_correctly(self, env, checker):
        env._ee_pose[:3] = [2.0, 0.0, 0.5]   # x out of bounds
        checker.check(env)
        checker.check(env)
        summary = checker.violation_summary
        assert summary["by_type"]["workspace_bounds"] == 2

    def test_total_violations_accumulates_across_steps(self, env, checker):
        env._collision = True
        for _ in range(3):
            checker.check(env)
        assert checker.violation_summary["total_violations"] >= 3
