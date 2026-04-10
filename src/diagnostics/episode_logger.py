"""
Episode logger: writes per-step and per-episode logs to disk.
Supports CSV, JSON, and optional video frame buffering.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EpisodeLogger:
    """
    Structured logger for robot manipulation episodes.

    Writes:
    - ``episodes.jsonl``: One JSON summary per completed episode.
    - ``steps.csv``:      Per-step tabular data (reward, collision, etc.).
    - Optional video frames buffered in memory and written via OpenCV.

    Args:
        log_dir:        Root output directory.
        algorithm:      Algorithm name for tagging logs.
        task_name:      Task name for tagging logs.
        save_video:     Whether to buffer frames for MP4 export.
        video_fps:      Frames per second for video export.
    """

    def __init__(
        self,
        log_dir: str = "results/logs",
        algorithm: str = "unknown",
        task_name: str = "unknown",
        save_video: bool = False,
        video_fps: int = 30,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._algorithm = algorithm
        self._task_name = task_name
        self._save_video = save_video
        self._video_fps = video_fps

        self._episode_id: str = ""
        self._episode_start: float = 0.0
        self._step_records: list[dict] = []
        self._video_frames: list[np.ndarray] = []
        self._episode_count = 0

        # Open persistent JSONL file for episode summaries
        self._ep_log_path = self._log_dir / f"{algorithm}_{task_name}_episodes.jsonl"
        # Open persistent CSV for step data
        self._step_log_path = self._log_dir / f"{algorithm}_{task_name}_steps.csv"
        self._csv_initialized = False

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def begin_episode(self, episode_id: str | None = None) -> None:
        """Call at the start of each episode."""
        self._episode_count += 1
        self._episode_id = episode_id or f"ep_{self._episode_count:06d}"
        self._episode_start = time.time()
        self._step_records = []
        self._video_frames = []
        logger.debug("EpisodeLogger: episode %s started", self._episode_id)

    def log_step(
        self,
        step: int,
        obs: dict,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
        frame: np.ndarray | None = None,
    ) -> None:
        """Log a single environment step."""
        record = {
            "episode_id": self._episode_id,
            "step": step,
            "reward": round(float(reward), 5),
            "terminated": terminated,
            "truncated": truncated,
            "collision": bool(info.get("collision", False)),
            "replan": bool(info.get("replan", False)),
            "ee_x": round(float(obs.get("ee_pose", [0]*7)[0]), 4),
            "ee_y": round(float(obs.get("ee_pose", [0]*7)[1]), 4),
            "ee_z": round(float(obs.get("ee_pose", [0]*7)[2]), 4),
        }
        self._step_records.append(record)
        self._write_csv_row(record)

        if self._save_video and frame is not None:
            self._video_frames.append(frame)

    def end_episode(
        self,
        success: bool,
        total_reward: float,
        info: dict | None = None,
    ) -> dict:
        """Call at episode end; write summary and optionally export video."""
        duration = time.time() - self._episode_start
        summary = {
            "episode_id": self._episode_id,
            "task_name": self._task_name,
            "algorithm": self._algorithm,
            "success": success,
            "total_reward": round(total_reward, 4),
            "steps": len(self._step_records),
            "duration_s": round(duration, 3),
            "collision_count": sum(r["collision"] for r in self._step_records),
            "replan_count": sum(r["replan"] for r in self._step_records),
            "timestamp": self._episode_start,
            **(info or {}),
        }
        with open(self._ep_log_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

        if self._save_video and self._video_frames:
            self._export_video()

        logger.debug("EpisodeLogger: episode %s ended (success=%s, steps=%d)",
                     self._episode_id, success, len(self._step_records))
        return summary

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _write_csv_row(self, record: dict) -> None:
        write_header = not self._csv_initialized and not self._step_log_path.exists()
        self._csv_initialized = True
        with open(self._step_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(record)

    # ------------------------------------------------------------------
    # Video export
    # ------------------------------------------------------------------

    def _export_video(self) -> None:
        try:
            import cv2  # type: ignore[import]
            video_dir = self._log_dir.parent / "videos"
            video_dir.mkdir(exist_ok=True)
            path = str(video_dir / f"{self._episode_id}.mp4")
            h, w = self._video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(path, fourcc, self._video_fps, (w, h))
            for frame in self._video_frames:
                if frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                out.write(frame_bgr)
            out.release()
            logger.info("EpisodeLogger: video saved to '%s'", path)
        except ImportError:
            logger.warning("EpisodeLogger: opencv-python not installed, skipping video export")
        except Exception as exc:
            logger.warning("EpisodeLogger: video export failed: %s", exc)
