"""
Tests for the planner layer.

These tests verify the core design decisions:
1. The planner output is schema-constrained — hallucinated skills are caught
   before they ever reach the robot.
2. Plans are validated against the live scene — you can't command an object
   that isn't there.
3. The JSON parsing is robust to LLM formatting quirks (markdown, whitespace).
4. The mock planner always produces a sensible plan from scene structure alone,
   so the rest of the pipeline can run without an API key.
"""
import pytest
import numpy as np

from src.planner.base_planner import BasePlanner, PlanResult
from src.planner.llm_planner import MockLLMPlanner
from src.perception.scene_parser import SceneState, ObjectInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scene(names_graspable: list[str], names_container: list[str]) -> SceneState:
    objects = {}
    for n in names_graspable:
        objects[n] = ObjectInfo(n, np.array([0.4, 0.0, 0.45]), np.array([0,0,0,1]),
                                1.0, is_graspable=True, is_container=False)
    for n in names_container:
        objects[n] = ObjectInfo(n, np.array([0.6, 0.3, 0.42]), np.array([0,0,0,1]),
                                1.0, is_graspable=False, is_container=True)
    return SceneState(objects=objects, task_goal="test")


class _Planner(BasePlanner):
    """Minimal concrete planner for testing the base class."""
    def plan(self, scene, goal):
        return PlanResult(steps=[])


# ---------------------------------------------------------------------------
# Design claim: hallucinated skills are rejected before execution
# ---------------------------------------------------------------------------

class TestSchemaEnforcesSkillWhitelist:
    """
    The planner must reject any skill not in the allowed set.
    This prevents LLM hallucinations from reaching the robot controller.
    """
    p = _Planner()

    def test_fly_to_moon_rejected(self):
        valid, err = self.p.validate_plan([{"skill": "fly_to_moon", "target": "mars"}])
        assert not valid
        assert "fly_to_moon" in err

    def test_every_allowed_skill_accepted(self):
        for skill in BasePlanner.ALLOWED_SKILLS:
            valid, _ = self.p.validate_plan([{"skill": skill, "target": "obj"}])
            assert valid, f"Allowed skill '{skill}' was incorrectly rejected"

    def test_empty_plan_rejected(self):
        # A plan with zero steps means the robot does nothing — that's an error
        valid, err = self.p.validate_plan([])
        assert not valid

    def test_plan_longer_than_20_rejected(self):
        # Enforcing a max length prevents runaway plans
        plan = [{"skill": "wait", "target": None, "duration": 0.1}] * 25
        valid, _ = self.p.validate_plan(plan)
        assert not valid

    def test_missing_required_target_field_rejected(self):
        # The schema requires "target" — its absence is caught
        valid, _ = self.p.validate_plan([{"skill": "grasp"}])
        assert not valid


# ---------------------------------------------------------------------------
# Design claim: JSON parsing tolerates real-world LLM output formatting
# ---------------------------------------------------------------------------

class TestJSONParsingRobustness:
    p = _Planner()

    def test_bare_json_array(self):
        raw = '[{"skill":"grasp","target":"block"}]'
        result = self.p._parse_json_response(raw)
        assert result == [{"skill": "grasp", "target": "block"}]

    def test_markdown_fenced_json(self):
        raw = '```json\n[{"skill":"move_to","target":"bin"}]\n```'
        result = self.p._parse_json_response(raw)
        assert result is not None and result[0]["skill"] == "move_to"

    def test_json_buried_in_prose(self):
        raw = 'Here is the plan:\n[{"skill":"release","target":"block"}]\nDone.'
        result = self.p._parse_json_response(raw)
        assert result is not None and result[0]["skill"] == "release"

    def test_garbage_response_returns_none(self):
        assert self.p._parse_json_response("I cannot help with that.") is None

    def test_plain_string_returns_none(self):
        assert self.p._parse_json_response("step 1: move, step 2: grasp") is None


# ---------------------------------------------------------------------------
# Design claim: mock planner derives a correct plan from scene structure
# ---------------------------------------------------------------------------

class TestMockPlannerDerivesFromScene:
    """
    The mock planner must always produce a plan that:
    - References objects that actually exist in the scene
    - Follows move_to → grasp → move_to → release ordering
    - Routes each object to the correct container
    """
    planner = MockLLMPlanner()

    def test_plan_only_targets_existing_objects(self):
        scene = _scene(["red_block"], ["bin_A"])
        result = self.planner.plan(scene, "pick and place")
        assert result.valid
        for step in result.steps:
            t = step.get("target")
            assert t in ("red_block", "bin_A"), f"Unexpected target '{t}'"

    def test_plan_includes_move_grasp_release(self):
        scene = _scene(["red_block"], ["bin_A"])
        result = self.planner.plan(scene, "pick and place")
        skills = [s["skill"] for s in result.steps]
        assert "move_to" in skills
        assert "grasp" in skills
        assert "release" in skills

    def test_grasp_always_preceded_by_move_to(self):
        scene = _scene(["red_block"], ["bin_A"])
        result = self.planner.plan(scene, "pick and place")
        skills = [s["skill"] for s in result.steps]
        for i, skill in enumerate(skills):
            if skill == "grasp":
                assert i > 0 and skills[i - 1] == "move_to", \
                    f"grasp at index {i} not preceded by move_to"

    def test_multi_object_scene_all_objects_planned(self):
        scene = _scene(["red_block", "blue_block"], ["bin_A"])
        result = self.planner.plan(scene, "sort objects")
        targets = {s["target"] for s in result.steps}
        assert "red_block" in targets
        assert "blue_block" in targets

    def test_empty_scene_plan_is_invalid(self):
        scene = _scene([], [])
        result = self.planner.plan(scene, "do something")
        assert not result.valid

    def test_plan_source_tagged_as_mock(self):
        scene = _scene(["block"], ["bin"])
        result = self.planner.plan(scene, "test")
        assert result.source == "mock"

    def test_call_count_tracked(self):
        planner = MockLLMPlanner()
        scene = _scene(["block"], ["bin"])
        planner.plan(scene, "a")
        planner.plan(scene, "b")
        planner.plan(scene, "c")
        assert planner.stats["total_calls"] == 3
        assert planner.stats["error_count"] == 0

    def test_plan_result_is_json_serialisable(self):
        import json
        scene = _scene(["block"], ["bin"])
        result = self.planner.plan(scene, "test")
        # Should not raise
        json.loads(result.to_json())
