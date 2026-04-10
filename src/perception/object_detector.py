"""
Object detection module.

Wraps a YOLO-based detector (or a lightweight mock) that takes an RGB image
and returns structured detections with class labels, bounding boxes, and
confidence scores.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single object detection result."""
    label: str
    confidence: float
    bbox_xyxy: np.ndarray   # shape (4,): [x1, y1, x2, y2] in pixels
    center_px: np.ndarray   # shape (2,): [cx, cy] in pixels
    estimated_pos_3d: np.ndarray | None = None  # (3,) world-frame estimate


@dataclass
class DetectionResult:
    """All detections for one frame."""
    detections: list[Detection] = field(default_factory=list)
    image_shape: tuple[int, ...] = (224, 224, 3)
    inference_time_ms: float = 0.0

    def get_by_label(self, label: str) -> list[Detection]:
        return [d for d in self.detections if d.label == label]

    def best(self, label: str) -> Detection | None:
        """Return highest-confidence detection with this label."""
        matches = self.get_by_label(label)
        return max(matches, key=lambda d: d.confidence) if matches else None


class ObjectDetector:
    """
    YOLO-based object detector with a mock fallback.

    If ``ultralytics`` is not installed or ``use_mock=True`` the detector
    runs a lightweight mock that returns plausible detections derived from
    the ground-truth ``object_states`` dict (useful for unit testing).

    Args:
        model_path: Path to YOLO weights (e.g. ``yolov8n.pt``).
        device:     Torch device string (``"cuda"`` or ``"cpu"``).
        conf_threshold: Minimum detection confidence.
        use_mock:   Force mock mode regardless of availability.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "cpu",
        conf_threshold: float = 0.4,
        use_mock: bool = False,
    ) -> None:
        self._conf_threshold = conf_threshold
        self._model = None
        self._use_mock = use_mock

        if not use_mock:
            try:
                from ultralytics import YOLO  # type: ignore[import]
                self._model = YOLO(model_path)
                self._model.to(device)
                logger.info("ObjectDetector: loaded YOLO from '%s' on %s", model_path, device)
            except (ImportError, Exception) as exc:
                logger.warning("ObjectDetector falling back to mock (%s)", exc)
                self._use_mock = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        object_states: dict[str, np.ndarray] | None = None,
    ) -> DetectionResult:
        """
        Run detection on a single RGB frame.

        Args:
            image:         uint8 (H, W, 3) RGB image.
            object_states: Optional ground-truth states used by the mock.

        Returns:
            :class:`DetectionResult`
        """
        import time
        t0 = time.perf_counter()

        if self._use_mock:
            result = self._mock_detect(image, object_states or {})
        else:
            result = self._yolo_detect(image)

        result.inference_time_ms = (time.perf_counter() - t0) * 1000
        return result

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _yolo_detect(self, image: np.ndarray) -> DetectionResult:
        results = self._model(image, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self._conf_threshold:
                    continue
                cls_id = int(box.cls[0])
                label = r.names.get(cls_id, f"class_{cls_id}")
                xyxy = box.xyxy[0].cpu().numpy().astype(np.float32)
                cx = (xyxy[0] + xyxy[2]) / 2
                cy = (xyxy[1] + xyxy[3]) / 2
                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    bbox_xyxy=xyxy,
                    center_px=np.array([cx, cy]),
                ))
        return DetectionResult(detections=detections, image_shape=image.shape)

    def _mock_detect(
        self,
        image: np.ndarray,
        object_states: dict[str, np.ndarray],
    ) -> DetectionResult:
        """Generate synthetic detections from ground-truth object states."""
        h, w = image.shape[:2]
        detections = []
        for name, state in object_states.items():
            pos = state[:3]
            # Simple pinhole projection (approximate)
            cx = int((pos[0] - 0.1) / 0.8 * w)
            cy = int(h - (pos[2] - 0.4) / 0.5 * h * 0.7)
            cx = int(np.clip(cx, 10, w - 10))
            cy = int(np.clip(cy, 10, h - 10))
            half = 12
            xyxy = np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)
            # Add slight positional noise to the mock
            noise = np.random.normal(0, 2, 4).astype(np.float32)
            xyxy += noise
            conf = float(np.clip(np.random.normal(0.85, 0.05), 0.5, 0.99))
            detections.append(Detection(
                label=name,
                confidence=conf,
                bbox_xyxy=xyxy,
                center_px=np.array([cx, cy], dtype=np.float32),
                estimated_pos_3d=state[:3].copy(),
            ))
        return DetectionResult(detections=detections, image_shape=image.shape)

    def project_to_3d(
        self,
        detection: Detection,
        depth_map: np.ndarray,
        camera_intrinsics: np.ndarray,
    ) -> np.ndarray:
        """
        Back-project a 2-D detection into 3-D world coordinates.

        Args:
            detection:         2-D detection with ``center_px``.
            depth_map:         (H, W) float32 depth in metres.
            camera_intrinsics: (3, 3) K matrix.

        Returns:
            (3,) estimated 3-D point in camera frame.
        """
        cx, cy = detection.center_px
        cx_int, cy_int = int(np.clip(cx, 0, depth_map.shape[1] - 1)), \
                         int(np.clip(cy, 0, depth_map.shape[0] - 1))
        z = float(depth_map[cy_int, cx_int])
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        px, py = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        x = (cx - px) * z / max(fx, 1e-6)
        y = (cy - py) * z / max(fy, 1e-6)
        return np.array([x, y, z], dtype=np.float32)
