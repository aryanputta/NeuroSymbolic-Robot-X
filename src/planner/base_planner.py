"""
Abstract planner interface and shared data structures.

All planners (LLM, VLA, scripted) implement BasePlanner and return a
PlanResult that the SkillRouter can execute.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import jsonschema

from src.perception.scene_parser import SceneState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON schema for validating planner output
# ---------------------------------------------------------------------------

SKILL_STEP_SCHEMA = {
    "type": "object",
    "required": ["skill", "target"],
    "properties": {
        "skill": {"type": "string"},
        "target": {},
        "approach_height": {"type": "number"},
        "grip_force": {"type": "number"},
        "direction": {"type": "array", "items": {"type": "number"}},
        "distance": {"type": "number"},
        "duration": {"type": "number"},
    },
}

PLAN_SCHEMA = {
    "type": "array",
    "items": SKILL_STEP_SCHEMA,
    "minItems": 1,
    "maxItems": 20,
}


@dataclass
class PlanResult:
    """Output of a planner call."""
    steps: list[dict[str, Any]]
    raw_response: str = ""
    latency_ms: float = 0.0
    valid: bool = True
    validation_error: str = ""
    source: str = "unknown"   # "llm" | "vla" | "scripted" | "cache"

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def to_json(self) -> str:
        return json.dumps(self.steps, indent=2)

    @classmethod
    def invalid(cls, reason: str) -> "PlanResult":
        return cls(steps=[], valid=False, validation_error=reason)


class BasePlanner(ABC):
    """
    Abstract base class for all task planners.

    Subclasses must implement :meth:`plan` which takes a :class:`SceneState`
    and a natural-language goal string, then returns a :class:`PlanResult`.
    """

    ALLOWED_SKILLS: frozenset[str] = frozenset({
        "move_to", "grasp", "release", "push", "pull",
        "place_at", "open_drawer", "close_drawer",
        "open_gripper", "close_gripper", "reset", "wait",
    })

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self._call_count = 0
        self._error_count = 0

    @abstractmethod
    def plan(self, scene: SceneState, goal: str) -> PlanResult:
        """Generate a skill plan for the given scene and goal."""

    def validate_plan(self, steps: list[dict]) -> tuple[bool, str]:
        """
        Validate plan against the JSON schema and skill whitelist.

        Returns:
            (is_valid, error_message)
        """
        try:
            jsonschema.validate(steps, PLAN_SCHEMA)
        except jsonschema.ValidationError as exc:
            return False, f"Schema validation failed: {exc.message}"

        for i, step in enumerate(steps):
            skill = step.get("skill", "")
            if skill not in self.ALLOWED_SKILLS:
                return False, f"Step {i}: unknown skill '{skill}'. Allowed: {sorted(self.ALLOWED_SKILLS)}"

        return True, ""

    def _parse_json_response(self, text: str) -> list[dict] | None:
        """Extract and parse a JSON array from a possibly verbose LLM response."""
        # Try direct parse
        text = text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        import re
        pattern = r"```(?:json)?\s*(\[.*?\])\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try extracting bare array
        match = re.search(r"(\[.*\])", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse JSON from planner response: %r", text[:200])
        return None

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._call_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._call_count),
        }
