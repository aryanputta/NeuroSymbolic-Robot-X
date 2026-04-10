"""
Failure logger: mines past failures to improve the planner and curriculum.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FailureRecord:
    """Structured record of one episode failure."""
    episode_id: str
    task_name: str
    failure_step: int
    failure_reason: str
    failure_type: str   # "perception" | "planning" | "control" | "safety" | "timeout"
    skill_at_failure: str
    replan_count: int
    collision_count: int
    scene_snapshot: dict = field(default_factory=dict)
    suggested_fix: str = ""


class FailureClassifier:
    """
    Classifies failure reasons into high-level categories.

    Categories:
    - ``perception``: Object not found, wrong position estimate.
    - ``planning``:   Invalid skill, impossible target, hallucinated action.
    - ``control``:    Skill timeout, grasp failure, drop.
    - ``safety``:     Collision, out-of-bounds, joint limit.
    - ``timeout``:    Episode time limit exceeded.
    """

    _PERCEPTION_KEYWORDS = ("not found", "not in scene", "target not detected", "unknown object")
    _PLANNING_KEYWORDS = ("unknown skill", "validation", "schema", "hallucinated", "impossible")
    _CONTROL_KEYWORDS = ("timeout", "not grasped", "drop", "failed to grasp")
    _SAFETY_KEYWORDS = ("collision", "bounds", "joint limit", "velocity limit")

    def classify(self, failure_reason: str) -> str:
        reason_lower = failure_reason.lower()
        if any(k in reason_lower for k in self._SAFETY_KEYWORDS):
            return "safety"
        if any(k in reason_lower for k in self._PERCEPTION_KEYWORDS):
            return "perception"
        if any(k in reason_lower for k in self._PLANNING_KEYWORDS):
            return "planning"
        if any(k in reason_lower for k in self._CONTROL_KEYWORDS):
            return "control"
        if "timeout" in reason_lower or "time limit" in reason_lower:
            return "timeout"
        return "unknown"


class FailureLogger:
    """
    Logs, analyses, and proposes fixes for episode failures.

    Maintains a rolling log of structured failure records and provides
    statistics for the improvement agent.

    Args:
        log_dir:        Directory to write failure logs.
        max_records:    Maximum in-memory failure records.
        classifier:     FailureClassifier instance.
    """

    def __init__(
        self,
        log_dir: str = "data/failure_logs",
        max_records: int = 5_000,
        classifier: FailureClassifier | None = None,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._max_records = max_records
        self._classifier = classifier or FailureClassifier()
        self._records: list[FailureRecord] = []

    def log(
        self,
        episode_id: str,
        task_name: str,
        failure_step: int,
        failure_reason: str,
        skill_at_failure: str = "",
        replan_count: int = 0,
        collision_count: int = 0,
        scene_snapshot: dict | None = None,
    ) -> FailureRecord:
        """Record a new failure and persist it."""
        failure_type = self._classifier.classify(failure_reason)
        record = FailureRecord(
            episode_id=episode_id,
            task_name=task_name,
            failure_step=failure_step,
            failure_reason=failure_reason,
            failure_type=failure_type,
            skill_at_failure=skill_at_failure,
            replan_count=replan_count,
            collision_count=collision_count,
            scene_snapshot=scene_snapshot or {},
            suggested_fix=self._suggest_fix(failure_type, failure_reason),
        )

        if len(self._records) >= self._max_records:
            self._records.pop(0)
        self._records.append(record)
        self._persist(record)
        return record

    def _persist(self, record: FailureRecord) -> None:
        path = self._log_dir / f"{record.task_name}_failures.jsonl"
        entry = {
            "episode_id": record.episode_id,
            "task_name": record.task_name,
            "failure_step": record.failure_step,
            "failure_reason": record.failure_reason,
            "failure_type": record.failure_type,
            "skill_at_failure": record.skill_at_failure,
            "replan_count": record.replan_count,
            "collision_count": record.collision_count,
            "suggested_fix": record.suggested_fix,
        }
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    @staticmethod
    def _suggest_fix(failure_type: str, reason: str) -> str:
        suggestions = {
            "perception": "Increase detection confidence threshold; add re-observation before grasp.",
            "planning": "Constrain planner output schema; add skill-library validation layer.",
            "control": "Increase skill timeout; add approach retry; tune RL reward shaping.",
            "safety": "Tighten workspace bounds in planner; add collision avoidance reward term.",
            "timeout": "Reduce plan length; increase control frequency; simplify task decomposition.",
            "unknown": "Review failure trajectory; consider curriculum learning on this failure mode.",
        }
        return suggestions.get(failure_type, "No specific suggestion.")

    def failure_attribution(self) -> dict[str, float]:
        """Return fraction of failures attributed to each category."""
        if not self._records:
            return {}
        counts = Counter(r.failure_type for r in self._records)
        total = len(self._records)
        return {k: round(v / total, 4) for k, v in counts.most_common()}

    def most_problematic_skills(self, top_k: int = 5) -> list[tuple[str, int]]:
        """Return (skill_name, failure_count) for the most-failing skills."""
        counts = Counter(r.skill_at_failure for r in self._records if r.skill_at_failure)
        return counts.most_common(top_k)

    def generate_report(self) -> str:
        """Human-readable failure analysis report."""
        attribution = self.failure_attribution()
        skills = self.most_problematic_skills()
        lines = [
            f"=== Failure Analysis Report ({len(self._records)} failures) ===",
            "\nFailure attribution:",
        ]
        for ftype, frac in attribution.items():
            lines.append(f"  {ftype:15s}: {frac*100:.1f}%")
        lines.append("\nMost problematic skills:")
        for skill, count in skills:
            lines.append(f"  {skill:20s}: {count} failures")
        return "\n".join(lines)

    @property
    def total_failures(self) -> int:
        return len(self._records)
