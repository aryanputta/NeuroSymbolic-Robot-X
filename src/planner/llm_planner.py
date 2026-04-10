"""
LLM-based task planner.

Sends the structured scene state to an OpenAI-compatible chat API and
parses the response into a validated skill plan.  Falls back to a
MockLLMPlanner when no API key is available.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from src.perception.scene_parser import SceneState
from src.planner.base_planner import BasePlanner, PlanResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a robot task planner for a Franka Panda arm in simulation.
Given a scene description and a task goal, output a JSON array of skill steps.

Available skills (use ONLY these):
- move_to:    {"skill":"move_to","target":"<object_or_pos>","approach_height":0.1}
- grasp:      {"skill":"grasp","target":"<object>"}
- release:    {"skill":"release","target":"<object>"}
- place_at:   {"skill":"place_at","target":"<object_or_pos>"}
- push:       {"skill":"push","target":"<object>","direction":[dx,dy],"distance":0.1}
- open_drawer:{"skill":"open_drawer","target":"<drawer>"}
- wait:       {"skill":"wait","duration":0.5}
- reset:      {"skill":"reset"}

Rules:
1. Output ONLY valid JSON array — no explanation, no markdown, no extra text.
2. Always move_to before grasp.
3. Always grasp before release.
4. Use object names exactly as they appear in the scene.
5. Maximum 10 steps.
6. If the task requires multiple objects, handle them one at a time.
"""


def _build_user_message(scene: SceneState, goal: str) -> str:
    return f"{scene.to_prompt_string()}\n\nGoal: {goal}\n\nOutput the JSON plan:"


# ---------------------------------------------------------------------------
# Real LLM planner
# ---------------------------------------------------------------------------

class LLMPlanner(BasePlanner):
    """
    Task planner backed by an OpenAI-compatible chat API.

    Supports:
    - OpenAI (gpt-4o, gpt-4-turbo, etc.)
    - Any local OpenAI-compatible server (Ollama, vLLM, LMStudio).

    When ``OPENAI_API_KEY`` is not set the planner automatically switches to
    :class:`MockLLMPlanner`.

    Args:
        model:          Model identifier (default ``"gpt-4o"``).
        temperature:    Sampling temperature (default 0.1 for determinism).
        max_tokens:     Max tokens in response.
        timeout:        Request timeout in seconds.
        retry_attempts: Number of retries on transient failure.
        base_url:       Override API base URL for local inference servers.
        cache_plans:    Cache identical (scene_hash, goal) pairs.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: float = 10.0,
        retry_attempts: int = 3,
        base_url: str | None = None,
        cache_plans: bool = True,
    ) -> None:
        super().__init__()
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._retry_attempts = retry_attempts
        self._cache: dict[tuple, PlanResult] = {}
        self._cache_enabled = cache_plans
        self._client = self._build_client(base_url)

    def _build_client(self, base_url: str | None) -> Any | None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("LLMPlanner: OPENAI_API_KEY not set — using mock responses")
            return None
        try:
            import openai
            kwargs: dict = {"api_key": api_key, "timeout": self._timeout}
            if base_url:
                kwargs["base_url"] = base_url
            return openai.OpenAI(**kwargs)
        except ImportError:
            logger.warning("LLMPlanner: openai package not installed — using mock responses")
            return None

    def plan(self, scene: SceneState, goal: str) -> PlanResult:
        self._call_count += 1

        if self._client is None:
            return MockLLMPlanner().plan(scene, goal)

        cache_key = (scene.to_prompt_string()[:200], goal)
        if self._cache_enabled and cache_key in self._cache:
            cached = self._cache[cache_key]
            return PlanResult(
                steps=cached.steps,
                raw_response=cached.raw_response,
                latency_ms=0.0,
                valid=True,
                source="cache",
            )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(scene, goal)},
        ]

        last_error = ""
        for attempt in range(self._retry_attempts):
            try:
                t0 = time.perf_counter()
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                raw = response.choices[0].message.content or ""
                steps = self._parse_json_response(raw)
                if steps is None:
                    last_error = "Failed to parse JSON from response"
                    continue
                valid, err = self.validate_plan(steps)
                if not valid:
                    last_error = err
                    continue
                result = PlanResult(steps=steps, raw_response=raw,
                                    latency_ms=latency_ms, valid=True, source="llm")
                if self._cache_enabled:
                    self._cache[cache_key] = result
                return result

            except Exception as exc:
                last_error = str(exc)
                logger.warning("LLMPlanner attempt %d/%d failed: %s", attempt + 1, self._retry_attempts, exc)
                time.sleep(0.5 * (2 ** attempt))

        self._error_count += 1
        logger.error("LLMPlanner: all retries failed. Last error: %s", last_error)
        return PlanResult.invalid(last_error)


# ---------------------------------------------------------------------------
# Mock planner — no API required
# ---------------------------------------------------------------------------

class MockLLMPlanner(BasePlanner):
    """
    Deterministic mock planner for testing and CI.

    Generates a hardcoded skill plan derived from the scene state:
    - Sorts graspable objects by name.
    - For each object, move_to → grasp → move_to (nearest container) → release.

    This ensures tests pass without an LLM API key.
    """

    def plan(self, scene: SceneState, goal: str) -> PlanResult:
        self._call_count += 1
        t0 = time.perf_counter()

        graspable = scene.get_graspable_objects()
        containers = scene.get_containers()

        if not graspable:
            return PlanResult.invalid("No graspable objects in scene")

        steps: list[dict] = []
        target_bin = containers[0] if containers else "table_center"

        for obj_name in sorted(graspable):
            steps += [
                {"skill": "move_to", "target": obj_name, "approach_height": 0.1},
                {"skill": "grasp", "target": obj_name},
                {"skill": "move_to", "target": target_bin, "approach_height": 0.15},
                {"skill": "release", "target": obj_name},
            ]
            if len(steps) >= 16:  # respect max 10 meaningful actions
                break

        latency_ms = (time.perf_counter() - t0) * 1000
        return PlanResult(steps=steps, raw_response="[mock]",
                          latency_ms=latency_ms, valid=True, source="mock")
