"""Unit tests for planner modules."""
import pytest
import numpy as np

from src.planner.base_planner import BasePlanner, PlanResult, PLAN_SCHEMA
from src.planner.llm_planner import MockLLMPlanner
from src.perception.scene_parser import SceneState, ObjectInfo


def _make_scene(objects: dict | None = None, goal: str = "pick up the red block") -> SceneState:
    if objects is None:
        objects = {
            "red_block": ObjectInfo(
                name="red_block",
                position=np.array([0.45, 0.0, 0.45]),
                orientation=np.array([0., 0., 0., 1.]),
                confidence=1.0,
                color="red",
                shape="block",
                is_graspable=True,
                is_container=False,
            ),
            "bin_A": ObjectInfo(
                name="bin_A",
                position=np.array([0.6, 0.3, 0.42]),
                orientation=np.array([0., 0., 0., 1.]),
                confidence=1.0,
                color="blue",
                shape="bin",
                is_graspable=False,
                is_container=True,
            ),
        }
    return SceneState(objects=objects, task_goal=goal)


class TestPlanResult:
    def test_len(self):
        plan = PlanResult(steps=[{"skill": "grasp", "target": "obj"}])
        assert len(plan) == 1

    def test_iter(self):
        steps = [{"skill": "move_to", "target": "obj"}, {"skill": "grasp", "target": "obj"}]
        plan = PlanResult(steps=steps)
        assert list(plan) == steps

    def test_invalid_factory(self):
        plan = PlanResult.invalid("test error")
        assert not plan.valid
        assert plan.validation_error == "test error"
        assert len(plan) == 0

    def test_to_json(self):
        import json
        steps = [{"skill": "grasp", "target": "x"}]
        plan = PlanResult(steps=steps)
        parsed = json.loads(plan.to_json())
        assert parsed == steps


class TestBasePlannerValidation:
    """Test the JSON schema validation logic on BasePlanner."""

    class _ConcretePlanner(BasePlanner):
        def plan(self, scene, goal):
            return PlanResult(steps=[])

    def setup_method(self):
        self.planner = self._ConcretePlanner()

    def test_valid_plan(self):
        steps = [
            {"skill": "move_to", "target": "red_block"},
            {"skill": "grasp", "target": "red_block"},
            {"skill": "release", "target": "red_block"},
        ]
        valid, err = self.planner.validate_plan(steps)
        assert valid, err

    def test_empty_plan_invalid(self):
        valid, err = self.planner.validate_plan([])
        assert not valid

    def test_unknown_skill_invalid(self):
        steps = [{"skill": "fly_to_moon", "target": "mars"}]
        valid, err = self.planner.validate_plan(steps)
        assert not valid
        assert "fly_to_moon" in err

    def test_missing_target_invalid(self):
        steps = [{"skill": "grasp"}]  # missing "target"
        valid, err = self.planner.validate_plan(steps)
        assert not valid

    def test_too_many_steps_invalid(self):
        steps = [{"skill": "wait", "target": None, "duration": 1.0}] * 25
        valid, err = self.planner.validate_plan(steps)
        assert not valid

    def test_parse_json_response_direct(self):
        raw = '[{"skill":"grasp","target":"obj"}]'
        result = self.planner._parse_json_response(raw)
        assert result == [{"skill": "grasp", "target": "obj"}]

    def test_parse_json_response_markdown(self):
        raw = '```json\n[{"skill":"move_to","target":"bin"}]\n```'
        result = self.planner._parse_json_response(raw)
        assert result is not None
        assert result[0]["skill"] == "move_to"

    def test_parse_json_response_garbage(self):
        result = self.planner._parse_json_response("This is not JSON at all!")
        assert result is None


class TestMockLLMPlanner:
    def test_returns_valid_plan(self):
        planner = MockLLMPlanner()
        scene = _make_scene()
        result = planner.plan(scene, "pick up the red block")
        assert result.valid
        assert len(result.steps) > 0
        assert result.source == "mock"

    def test_plan_uses_graspable_objects(self):
        planner = MockLLMPlanner()
        scene = _make_scene()
        result = planner.plan(scene, "pick and place")
        targets = {s.get("target") for s in result.steps}
        assert "red_block" in targets

    def test_plan_includes_container(self):
        planner = MockLLMPlanner()
        scene = _make_scene()
        result = planner.plan(scene, "put red block in bin")
        targets = {s.get("target") for s in result.steps}
        assert "bin_A" in targets

    def test_empty_scene_returns_invalid(self):
        planner = MockLLMPlanner()
        scene = _make_scene(objects={})
        result = planner.plan(scene, "do something")
        assert not result.valid

    def test_stats_tracked(self):
        planner = MockLLMPlanner()
        scene = _make_scene()
        planner.plan(scene, "test 1")
        planner.plan(scene, "test 2")
        assert planner.stats["total_calls"] == 2

    def test_schema_prompt_string(self):
        scene = _make_scene()
        text = scene.to_prompt_string()
        assert "red_block" in text
        assert "bin_A" in text
        assert "Task:" in text
