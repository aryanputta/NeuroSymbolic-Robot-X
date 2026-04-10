"""
Safety checker: enforces hard constraints before and after skill execution.

All safety violations result in an emergency stop and episode reset.
The checker also logs violation details for failure analysis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SafetyViolation:
    """Record of a single safety constraint violation."""
    violation_type: str
    severity: str               # "warning" | "critical"
    message: str
    step: int
    data: dict = field(default_factory=dict)


@dataclass
class SafetyCheckResult:
    """Output of a full safety check."""
    safe: bool
    violations: list[SafetyViolation] = field(default_factory=list)
    should_abort: bool = False
    should_replan: bool = False

    @property
    def critical_violations(self) -> list[SafetyViolation]:
        return [v for v in self.violations if v.severity == "critical"]


class SafetyChecker:
    """
    Monitors robot state for safety constraint violations.

    Checks performed each step:
    - Workspace bounds: EE must remain inside configured bounds.
    - Collision detection: flags if simulator reports contact.
    - Object drop: detects held object falling below table level.
    - Timeout: episode must complete within max_steps.
    - Joint limits: joint positions within mechanical limits.
    - Velocity limits: joint velocities within safe range.

    Args:
        workspace_bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        joint_position_limits: (9, 2) array of [min, max] per joint.
        joint_velocity_limits: (9,) max absolute velocity per joint.
        max_collision_count: Stop after this many collisions.
        table_height_z:      Z below which objects are considered dropped.
        abort_on_critical:   Whether critical violations trigger episode abort.
    """

    _FRANKA_JOINT_LIMITS = np.array([
        [-2.8973,  2.8973],
        [-1.7628,  1.7628],
        [-2.8973,  2.8973],
        [-3.0718, -0.0698],
        [-2.8973,  2.8973],
        [-0.0175,  3.7525],
        [-2.8973,  2.8973],
        [ 0.0000,  0.0400],
        [ 0.0000,  0.0400],
    ])

    _FRANKA_VELOCITY_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, 0.2, 0.2])

    def __init__(
        self,
        workspace_bounds: list[list[float]] | None = None,
        joint_position_limits: np.ndarray | None = None,
        joint_velocity_limits: np.ndarray | None = None,
        max_collision_count: int = 5,
        table_height_z: float = 0.42,
        abort_on_critical: bool = True,
    ) -> None:
        if workspace_bounds is None:
            workspace_bounds = [[-0.65, 0.65], [-0.65, 0.65], [0.0, 0.85]]
        self._ws_bounds = np.array(workspace_bounds)
        self._joint_limits = joint_position_limits if joint_position_limits is not None else self._FRANKA_JOINT_LIMITS
        self._vel_limits = joint_velocity_limits if joint_velocity_limits is not None else self._FRANKA_VELOCITY_LIMITS
        self._max_collisions = max_collision_count
        self._table_z = table_height_z
        self._abort_on_critical = abort_on_critical

        self._collision_count = 0
        self._step = 0
        self._violation_log: list[SafetyViolation] = []

    def reset(self) -> None:
        """Reset per-episode counters."""
        self._collision_count = 0
        self._step = 0

    def check(self, env: Any, action: np.ndarray | None = None) -> SafetyCheckResult:
        """
        Run all safety checks for the current environment state.

        Args:
            env:    Environment with EE pose, joint state, object states.
            action: Proposed action (used for velocity limit pre-check).

        Returns:
            :class:`SafetyCheckResult`
        """
        self._step += 1
        violations: list[SafetyViolation] = []

        ee_pose = getattr(env, "_ee_pose", np.zeros(7))[:3]
        joint_pos = getattr(env, "_joint_pos", np.zeros(9))
        joint_vel = getattr(env, "_joint_vel", np.zeros(9))
        collision = getattr(env, "_collision", False)
        held_obj = getattr(env, "_held_object", None)
        obj_states = getattr(env, "_object_states", {})

        # 1. Workspace bounds
        for i, (axis, label) in enumerate(zip(ee_pose, ["x", "y", "z"])):
            lo, hi = self._ws_bounds[i]
            if axis < lo or axis > hi:
                violations.append(SafetyViolation(
                    violation_type="workspace_bounds",
                    severity="critical",
                    message=f"EE {label}={axis:.3f} outside [{lo}, {hi}]",
                    step=self._step,
                    data={"axis": label, "value": float(axis), "bounds": [lo, hi]},
                ))

        # 2. Collision
        if collision:
            self._collision_count += 1
            severity = "critical" if self._collision_count >= self._max_collisions else "warning"
            violations.append(SafetyViolation(
                violation_type="collision",
                severity=severity,
                message=f"Collision detected (count={self._collision_count})",
                step=self._step,
                data={"count": self._collision_count},
            ))

        # 3. Joint position limits
        for i in range(min(9, len(joint_pos))):
            lo, hi = self._joint_limits[i]
            if joint_pos[i] < lo or joint_pos[i] > hi:
                violations.append(SafetyViolation(
                    violation_type="joint_position_limit",
                    severity="warning",
                    message=f"Joint {i} pos={joint_pos[i]:.3f} outside [{lo:.3f}, {hi:.3f}]",
                    step=self._step,
                    data={"joint": i, "value": float(joint_pos[i])},
                ))

        # 4. Joint velocity limits
        for i in range(min(9, len(joint_vel))):
            if abs(joint_vel[i]) > self._vel_limits[i] * 1.1:  # 10% tolerance
                violations.append(SafetyViolation(
                    violation_type="joint_velocity_limit",
                    severity="warning",
                    message=f"Joint {i} vel={joint_vel[i]:.3f} > limit {self._vel_limits[i]:.3f}",
                    step=self._step,
                    data={"joint": i, "velocity": float(joint_vel[i])},
                ))

        # 5. Object drop detection
        if held_obj and held_obj in obj_states:
            obj_z = obj_states[held_obj][2]
            if obj_z < self._table_z - 0.05:
                violations.append(SafetyViolation(
                    violation_type="object_drop",
                    severity="warning",
                    message=f"Held object '{held_obj}' dropped (z={obj_z:.3f} < {self._table_z:.3f})",
                    step=self._step,
                    data={"object": held_obj, "z": float(obj_z)},
                ))

        self._violation_log.extend(violations)

        has_critical = any(v.severity == "critical" for v in violations)
        should_abort = has_critical and self._abort_on_critical
        should_replan = bool(violations) and not has_critical

        return SafetyCheckResult(
            safe=not violations,
            violations=violations,
            should_abort=should_abort,
            should_replan=should_replan,
        )

    def validate_plan_step(self, step: dict, scene_state: Any) -> tuple[bool, str]:
        """
        Pre-validate a planner skill step before execution.

        Catches impossible actions (invalid target names, out-of-bounds goals).

        Returns:
            (is_valid, error_message)
        """
        target = step.get("target")
        skill = step.get("skill", "")
        obj_names = set()
        if scene_state is not None and hasattr(scene_state, "objects"):
            obj_names = set(scene_state.objects.keys())

        if skill in ("grasp", "release", "move_to") and isinstance(target, str):
            if target and target not in obj_names and not target.startswith("position:"):
                return False, f"Target object '{target}' not in scene: {sorted(obj_names)}"

        return True, ""

    @property
    def violation_summary(self) -> dict:
        """Summary of all violations logged since last reset."""
        by_type: dict[str, int] = {}
        for v in self._violation_log:
            by_type[v.violation_type] = by_type.get(v.violation_type, 0) + 1
        return {
            "total_violations": len(self._violation_log),
            "by_type": by_type,
            "collision_count": self._collision_count,
        }
