"""Unit tests for the safety checker."""
import numpy as np
import pytest

from envs.isaac.isaac_env import IsaacEnvConfig, MockIsaacEnv
from src.safety.safety_checker import SafetyChecker, SafetyViolation


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


class TestSafetyCheckerNominal:
    def test_safe_state_passes(self, env, checker):
        env._ee_pose[:3] = np.array([0.4, 0.0, 0.5])
        env._collision = False
        result = checker.check(env)
        assert result.safe
        assert not result.should_abort

    def test_reset_clears_counters(self, env, checker):
        env._collision = True
        checker.check(env)
        checker.reset()
        assert checker._collision_count == 0
        assert checker._step == 0


class TestWorkspaceBoundsViolation:
    def test_ee_out_of_bounds_x(self, env, checker):
        env._ee_pose[:3] = np.array([1.5, 0.0, 0.5])  # x > 0.65
        result = checker.check(env)
        assert not result.safe
        types = [v.violation_type for v in result.violations]
        assert "workspace_bounds" in types

    def test_ee_below_z_min(self, env, checker):
        env._ee_pose[:3] = np.array([0.4, 0.0, -0.1])  # z < 0.0
        result = checker.check(env)
        assert not result.safe

    def test_safe_at_boundary(self, env, checker):
        env._ee_pose[:3] = np.array([0.65, 0.0, 0.5])  # exactly at x bound
        env._collision = False
        result = checker.check(env)
        # At exact boundary is safe
        assert result.safe


class TestCollisionViolation:
    def test_collision_warning_below_max(self, env, checker):
        env._collision = True
        result = checker.check(env)
        col_violations = [v for v in result.violations if v.violation_type == "collision"]
        assert len(col_violations) == 1
        assert col_violations[0].severity == "warning"

    def test_collision_critical_at_max(self, env, checker):
        env._collision = True
        checker.reset()
        # Trigger max_collision_count (3) collisions
        for _ in range(3):
            checker.check(env)
        result = checker.check(env)
        col_violations = [v for v in result.violations if v.violation_type == "collision"]
        assert any(v.severity == "critical" for v in col_violations)
        assert result.should_abort


class TestObjectDropDetection:
    def test_drop_detected_below_table(self, env, checker):
        env._held_object = "red_block"
        env._object_states["red_block"][2] = 0.30  # well below table (0.42)
        result = checker.check(env)
        types = [v.violation_type for v in result.violations]
        assert "object_drop" in types

    def test_no_drop_when_on_table(self, env, checker):
        env._held_object = "red_block"
        env._object_states["red_block"][2] = 0.50  # above table
        result = checker.check(env)
        types = [v.violation_type for v in result.violations]
        assert "object_drop" not in types


class TestPlanStepValidation:
    def test_valid_target_passes(self, env, checker):
        from src.perception.scene_parser import SceneState, ObjectInfo
        scene = SceneState(objects={"red_block": ObjectInfo(
            name="red_block", position=np.zeros(3), orientation=np.zeros(4),
            confidence=1.0, is_graspable=True, is_container=False,
        )})
        step = {"skill": "grasp", "target": "red_block"}
        valid, err = checker.validate_plan_step(step, scene)
        assert valid, err

    def test_unknown_target_fails(self, env, checker):
        from src.perception.scene_parser import SceneState
        scene = SceneState(objects={})
        step = {"skill": "grasp", "target": "invisible_cube"}
        valid, err = checker.validate_plan_step(step, scene)
        assert not valid
        assert "invisible_cube" in err

    def test_wait_skill_needs_no_target_check(self, env, checker):
        step = {"skill": "wait", "target": None, "duration": 1.0}
        valid, err = checker.validate_plan_step(step, None)
        assert valid


class TestViolationSummary:
    def test_summary_counts_by_type(self, env, checker):
        env._ee_pose[:3] = np.array([1.5, 0.0, 0.5])
        checker.check(env)
        summary = checker.violation_summary
        assert summary["total_violations"] >= 1
        assert "workspace_bounds" in summary["by_type"]
