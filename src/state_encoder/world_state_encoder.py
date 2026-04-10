"""
World-state encoder: converts raw observations into a structured tensor
representation suitable for RL policy input and planner prompting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.perception.scene_parser import SceneState


@dataclass
class EncodedState:
    """Flattened numerical state for RL policy consumption."""
    robot_state: np.ndarray         # (18,)  joint pos + vel
    ee_pose: np.ndarray             # (7,)   end-effector pose
    object_features: np.ndarray     # (max_objects * object_feat_dim,)
    goal_embedding: np.ndarray      # (goal_dim,)
    held_object_flag: np.ndarray    # (1,)   1 if holding something
    scene_state: SceneState | None = None  # original scene for debugging

    def to_flat_vector(self) -> np.ndarray:
        """Concatenate all features into a single 1-D float32 vector."""
        return np.concatenate([
            self.robot_state,
            self.ee_pose,
            self.object_features,
            self.goal_embedding,
            self.held_object_flag,
        ]).astype(np.float32)

    @property
    def dim(self) -> int:
        return (len(self.robot_state) + len(self.ee_pose) +
                len(self.object_features) + len(self.goal_embedding) + 1)


class WorldStateEncoder:
    """
    Encodes a :class:`SceneState` (from SceneParser) into fixed-size tensors
    usable by the RL policy and diagnostics layer.

    Object features per object (per-slot):
        - position (3)
        - quaternion (4)
        - is_container flag (1)
        - is_graspable flag (1)
        - is_held flag (1)
        total = 10 per object

    Args:
        max_objects:    Maximum number of object slots.
        goal_dim:       Dimensionality of goal embedding.
        normalize_pos:  If True, normalise positions by workspace bounds.
        workspace_bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    """

    _OBJECT_FEAT_DIM = 10

    def __init__(
        self,
        max_objects: int = 5,
        goal_dim: int = 64,
        normalize_pos: bool = True,
        workspace_bounds: list[list[float]] | None = None,
    ) -> None:
        self.max_objects = max_objects
        self.goal_dim = goal_dim
        self.normalize_pos = normalize_pos
        if workspace_bounds is None:
            workspace_bounds = [[-0.6, 0.6], [-0.6, 0.6], [0.0, 0.8]]
        self._ws_bounds = np.array(workspace_bounds, dtype=np.float32)

        # Optional: lightweight text encoder for goal embedding
        self._text_encoder: Any = None

    def encode(
        self,
        scene: SceneState,
        gym_obs: dict[str, np.ndarray] | None = None,
    ) -> EncodedState:
        """
        Produce an :class:`EncodedState` from a parsed scene.

        Args:
            scene:    SceneState from SceneParser.
            gym_obs:  Optional raw gym obs dict (used as fallback for robot_state).

        Returns:
            :class:`EncodedState`
        """
        # Robot state
        if gym_obs is not None:
            robot_state = gym_obs.get("robot_state", np.zeros(18)).astype(np.float32)
            ee_pose = gym_obs.get("ee_pose", scene.robot_ee_pose).astype(np.float32)
        else:
            robot_state = np.concatenate([
                scene.robot_joint_positions,
                np.zeros(9),   # velocities not available from scene alone
            ]).astype(np.float32)
            ee_pose = scene.robot_ee_pose.astype(np.float32)

        # Object features
        obj_feats = np.zeros(self.max_objects * self._OBJECT_FEAT_DIM, dtype=np.float32)
        for i, (name, obj) in enumerate(list(scene.objects.items())[:self.max_objects]):
            pos = self._normalise_pos(obj.position)
            quat = obj.orientation.astype(np.float32)
            feats = np.concatenate([
                pos,
                quat,
                [float(obj.is_container)],
                [float(obj.is_graspable)],
                [float(scene.held_object == name)],
            ])
            offset = i * self._OBJECT_FEAT_DIM
            obj_feats[offset: offset + self._OBJECT_FEAT_DIM] = feats

        # Goal embedding
        goal_emb = self._encode_goal(scene.task_goal)

        held_flag = np.array([1.0 if scene.held_object else 0.0], dtype=np.float32)

        return EncodedState(
            robot_state=robot_state,
            ee_pose=ee_pose,
            object_features=obj_feats,
            goal_embedding=goal_emb,
            held_object_flag=held_flag,
            scene_state=scene,
        )

    def _normalise_pos(self, pos: np.ndarray) -> np.ndarray:
        """Map position to [-1, 1] using workspace bounds."""
        if not self.normalize_pos:
            return pos.astype(np.float32)
        mins = self._ws_bounds[:, 0]
        maxs = self._ws_bounds[:, 1]
        ranges = np.maximum(maxs - mins, 1e-6)
        normalised = 2.0 * (pos - mins) / ranges - 1.0
        return np.clip(normalised, -1.0, 1.0).astype(np.float32)

    def _encode_goal(self, task_goal: str) -> np.ndarray:
        """Encode task goal to a fixed-dim embedding.

        Uses a sentence encoder if available, otherwise falls back to a
        deterministic hash-based pseudo-embedding for testing.
        """
        if not task_goal:
            return np.zeros(self.goal_dim, dtype=np.float32)

        if self._text_encoder is not None:
            return self._text_encoder(task_goal)

        # Deterministic pseudo-embedding from character codes
        chars = np.array([ord(c) for c in task_goal[:self.goal_dim]], dtype=np.float32)
        emb = np.zeros(self.goal_dim, dtype=np.float32)
        emb[:len(chars)] = (chars - 64.0) / 64.0  # rough normalisation
        return emb

    def set_text_encoder(self, encoder) -> None:
        """Attach an external text encoder (e.g. sentence-transformers)."""
        self._text_encoder = encoder

    @property
    def state_dim(self) -> int:
        """Total dimensionality of the encoded state vector."""
        return 18 + 7 + self.max_objects * self._OBJECT_FEAT_DIM + self.goal_dim + 1
