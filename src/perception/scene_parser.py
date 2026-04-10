"""
Scene parser: converts raw sensor data into a structured scene description.

Takes camera images + depth + robot state and produces a ``SceneState``
that the planner can reason over in natural language.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.perception.object_detector import DetectionResult, ObjectDetector


@dataclass
class ObjectInfo:
    """Rich description of a single detected object."""
    name: str
    position: np.ndarray        # (3,) world-frame xyz
    orientation: np.ndarray     # (4,) quaternion
    confidence: float
    color: str = "unknown"
    shape: str = "unknown"
    is_graspable: bool = True
    is_container: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "position": self.position.tolist(),
            "orientation": self.orientation.tolist(),
            "confidence": round(self.confidence, 3),
            "color": self.color,
            "shape": self.shape,
            "is_graspable": self.is_graspable,
            "is_container": self.is_container,
        }


@dataclass
class SceneState:
    """Structured scene representation consumed by the planner."""
    objects: dict[str, ObjectInfo] = field(default_factory=dict)
    robot_joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(9))
    robot_ee_pose: np.ndarray = field(default_factory=lambda: np.zeros(7))
    held_object: str | None = None
    task_goal: str = ""
    step: int = 0

    def to_prompt_string(self) -> str:
        """Serialise scene to a compact string for LLM prompting."""
        lines = [f"Task: {self.task_goal}", "Objects:"]
        for name, obj in self.objects.items():
            pos_str = f"[{obj.position[0]:.3f}, {obj.position[1]:.3f}, {obj.position[2]:.3f}]"
            lines.append(f"  - {name}: pos={pos_str}, color={obj.color}, container={obj.is_container}")
        ee = self.robot_ee_pose
        lines.append(f"EE pose: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]")
        if self.held_object:
            lines.append(f"Held: {self.held_object}")
        return "\n".join(lines)

    def get_object_names(self) -> list[str]:
        return list(self.objects.keys())

    def get_graspable_objects(self) -> list[str]:
        return [n for n, o in self.objects.items() if o.is_graspable]

    def get_containers(self) -> list[str]:
        return [n for n, o in self.objects.items() if o.is_container]


class SceneParser:
    """
    Fuses detection results and ground-truth object states into a SceneState.

    In mock/simulation mode, object states from the simulator are used
    directly.  In real-sensor mode, detections from the ObjectDetector are
    lifted to 3-D using the depth map and camera intrinsics.

    Args:
        detector:           ObjectDetector instance.
        camera_intrinsics:  (3, 3) K matrix (used for 3-D back-projection).
        use_gt_states:      If True, trust simulator object states directly.
        label_to_color:     Mapping from object label to colour string.
        label_to_shape:     Mapping from object label to shape string.
        container_labels:   Set of labels that are containers/bins.
    """

    def __init__(
        self,
        detector: ObjectDetector | None = None,
        camera_intrinsics: np.ndarray | None = None,
        use_gt_states: bool = True,
        label_to_color: dict[str, str] | None = None,
        label_to_shape: dict[str, str] | None = None,
        container_labels: set[str] | None = None,
    ) -> None:
        self._detector = detector or ObjectDetector(use_mock=True)
        self._K = camera_intrinsics if camera_intrinsics is not None else np.eye(3, dtype=np.float32)
        self._use_gt = use_gt_states
        self._label_to_color = label_to_color or {}
        self._label_to_shape = label_to_shape or {}
        self._container_labels = container_labels or {"bin", "bin_a", "bin_b", "tray", "drawer"}

    def parse(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        robot_obs: dict,
        object_states: dict[str, np.ndarray] | None = None,
        task_goal: str = "",
        held_object: str | None = None,
        step: int = 0,
    ) -> SceneState:
        """
        Build a SceneState from sensor inputs.

        Args:
            rgb_image:    (H, W, 3) uint8 image.
            depth_image:  (H, W) float32 depth map.
            robot_obs:    Dict with ``robot_state`` and ``ee_pose`` arrays.
            object_states: Optional ground-truth {name: pose_7d} dict.
            task_goal:    Natural-language task goal string.
            held_object:  Name of currently held object (if any).
            step:         Current environment step.

        Returns:
            :class:`SceneState`
        """
        objects: dict[str, ObjectInfo] = {}

        if self._use_gt and object_states:
            for name, pose in object_states.items():
                objects[name] = self._pose_to_object_info(name, pose)
        else:
            det_result: DetectionResult = self._detector.detect(rgb_image, object_states)
            for det in det_result.detections:
                if det.estimated_pos_3d is not None:
                    pos_3d = det.estimated_pos_3d
                else:
                    pos_3d = self._detector.project_to_3d(det, depth_image, self._K)
                objects[det.label] = ObjectInfo(
                    name=det.label,
                    position=pos_3d,
                    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                    confidence=det.confidence,
                    color=self._infer_color(det.label),
                    shape=self._infer_shape(det.label),
                    is_graspable=not self._is_container(det.label),
                    is_container=self._is_container(det.label),
                )

        ee_pose = robot_obs.get("ee_pose", np.zeros(7))
        joint_pos = robot_obs.get("robot_state", np.zeros(18))[:9]

        return SceneState(
            objects=objects,
            robot_joint_positions=joint_pos,
            robot_ee_pose=ee_pose,
            held_object=held_object,
            task_goal=task_goal,
            step=step,
        )

    def _pose_to_object_info(self, name: str, pose: np.ndarray) -> ObjectInfo:
        return ObjectInfo(
            name=name,
            position=pose[:3],
            orientation=pose[3:7] if len(pose) >= 7 else np.array([0.0, 0.0, 0.0, 1.0]),
            confidence=1.0,
            color=self._infer_color(name),
            shape=self._infer_shape(name),
            is_graspable=not self._is_container(name),
            is_container=self._is_container(name),
        )

    def _infer_color(self, label: str) -> str:
        if label in self._label_to_color:
            return self._label_to_color[label]
        for color in ("red", "blue", "green", "yellow", "orange", "purple", "white", "black"):
            if color in label.lower():
                return color
        return "unknown"

    def _infer_shape(self, label: str) -> str:
        if label in self._label_to_shape:
            return self._label_to_shape[label]
        for shape in ("block", "cube", "sphere", "cylinder", "bin", "tray", "drawer"):
            if shape in label.lower():
                return shape
        return "unknown"

    def _is_container(self, label: str) -> bool:
        return any(c in label.lower() for c in self._container_labels)
