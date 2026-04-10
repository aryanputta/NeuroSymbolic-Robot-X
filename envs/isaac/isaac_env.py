"""
Isaac Sim environment interface for PHYSAI-RL-ROBOT-X.

Provides IsaacEnv (real Isaac Sim) and MockIsaacEnv (fully functional mock
for unit testing and CI without a GPU/Isaac licence).
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RobotObservation:
    """Single timestep observation from the robot environment."""
    rgb_wrist: np.ndarray          # (H, W, 3) uint8
    rgb_overhead: np.ndarray       # (H, W, 3) uint8
    depth_overhead: np.ndarray     # (H, W) float32 in metres
    joint_positions: np.ndarray    # (9,) radians
    joint_velocities: np.ndarray   # (9,) rad/s
    end_effector_pose: np.ndarray  # (7,) [x,y,z, qx,qy,qz,qw]
    object_states: dict[str, np.ndarray]  # name -> (7,) [pos(3)+quat(4)]
    timestamp: float               # wall-clock seconds


@dataclass
class IsaacEnvConfig:
    """Configuration for the robot simulation environment."""
    backend: str = "mock"           # "isaac" | "mock"
    physics_dt: float = 0.01       # 100 Hz
    rendering_dt: float = 0.033    # ~30 Hz
    gravity: tuple[float, ...] = (0.0, 0.0, -9.81)
    headless: bool = True
    gpu_id: int = 0
    seed: int = 42

    # Robot
    robot_name: str = "franka_panda"
    n_dof: int = 9
    workspace_x: tuple[float, float] = (-0.6, 0.6)
    workspace_y: tuple[float, float] = (-0.6, 0.6)
    workspace_z: tuple[float, float] = (0.0, 0.8)

    # Cameras
    wrist_cam_resolution: tuple[int, int] = (224, 224)
    overhead_cam_resolution: tuple[int, int] = (480, 640)

    # Table
    table_position: tuple[float, ...] = (0.5, 0.0, 0.4)
    table_size: tuple[float, ...] = (1.2, 0.8, 0.05)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseRobotEnv(ABC):
    """Abstract interface every robot environment backend must implement."""

    def __init__(self, config: IsaacEnvConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    @abstractmethod
    def reset(self, seed: int | None = None) -> RobotObservation:
        """Reset the scene and return the initial observation."""

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        """Apply action; return (obs, reward, terminated, truncated, info)."""

    @abstractmethod
    def get_observation(self) -> RobotObservation:
        """Return the current observation without stepping."""

    @abstractmethod
    def get_object_states(self) -> dict[str, np.ndarray]:
        """Return {object_name: pose_7d} for all objects."""

    @abstractmethod
    def apply_action(self, action: np.ndarray) -> None:
        """Send joint-delta action to the robot."""

    @abstractmethod
    def check_collision(self) -> bool:
        """Return True if a collision is currently detected."""

    @abstractmethod
    def get_ee_pose(self) -> np.ndarray:
        """Return end-effector pose as (7,) [x,y,z, qx,qy,qz,qw]."""

    @abstractmethod
    def close(self) -> None:
        """Shut down the simulation."""


# ---------------------------------------------------------------------------
# Real Isaac Sim backend
# ---------------------------------------------------------------------------

class IsaacEnv(BaseRobotEnv):
    """
    Isaac Sim environment wrapper.

    Requires NVIDIA Isaac Sim to be installed.  If the ``omni`` package is not
    available the constructor raises ``ImportError`` with a helpful message.
    """

    def __init__(self, config: IsaacEnvConfig) -> None:
        super().__init__(config)
        try:
            import omni.isaac.core as isaac_core  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Isaac Sim (omni.isaac.core) is not installed. "
                "Install NVIDIA Isaac Sim or use MockIsaacEnv for testing.\n"
                f"Original error: {exc}"
            ) from exc

        self._sim = None
        self._robot = None
        self._cameras: dict[str, Any] = {}
        self._objects: dict[str, Any] = {}
        self._setup_simulation()

    def _setup_simulation(self) -> None:
        """Initialise the Isaac Sim application and physics context."""
        from omni.isaac.core import SimulationApp  # type: ignore[import]
        self._app = SimulationApp({"headless": self.config.headless, "gpu_id": self.config.gpu_id})
        from omni.isaac.core import World  # type: ignore[import]
        self._world = World(physics_dt=self.config.physics_dt, rendering_dt=self.config.rendering_dt)
        self._setup_scene()
        self._setup_robot()
        self._setup_cameras()
        logger.info("IsaacEnv initialised (headless=%s)", self.config.headless)

    def _setup_scene(self) -> None:
        """Add table and ground plane to the scene."""
        from omni.isaac.core.objects import FixedCuboid  # type: ignore[import]
        table_pos = list(self.config.table_position)
        table_size = list(self.config.table_size)
        self._table = FixedCuboid(
            prim_path="/World/Table",
            name="table",
            position=np.array(table_pos),
            size=np.array(table_size),
        )
        self._world.scene.add(self._table)

    def _setup_robot(self) -> None:
        """Load the Franka Panda URDF/USD."""
        from omni.isaac.franka import Franka  # type: ignore[import]
        self._robot = Franka(prim_path="/World/Franka", name="franka")
        self._world.scene.add(self._robot)

    def _setup_cameras(self) -> None:
        """Attach RGB and depth cameras to the scene."""
        from omni.isaac.core.utils.viewports import set_camera_view  # type: ignore[import]
        set_camera_view(eye=[1.5, 0.0, 1.5], target=[0.5, 0.0, 0.4])

    def reset(self, seed: int | None = None) -> RobotObservation:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._world.reset()
        return self.get_observation()

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict]:
        self.apply_action(action)
        self._world.step(render=not self.config.headless)
        obs = self.get_observation()
        return obs, 0.0, False, False, {}

    def get_observation(self) -> RobotObservation:
        h_w, h_oh = self.config.wrist_cam_resolution, self.config.overhead_cam_resolution
        return RobotObservation(
            rgb_wrist=np.zeros((*h_w, 3), dtype=np.uint8),
            rgb_overhead=np.zeros((*h_oh, 3), dtype=np.uint8),
            depth_overhead=np.zeros(h_oh, dtype=np.float32),
            joint_positions=self._robot.get_joint_positions() if self._robot else np.zeros(9),
            joint_velocities=self._robot.get_joint_velocities() if self._robot else np.zeros(9),
            end_effector_pose=self.get_ee_pose(),
            object_states=self.get_object_states(),
            timestamp=time.time(),
        )

    def get_object_states(self) -> dict[str, np.ndarray]:
        states = {}
        for name, obj in self._objects.items():
            pos, quat = obj.get_world_pose()
            states[name] = np.concatenate([pos, quat])
        return states

    def apply_action(self, action: np.ndarray) -> None:
        if self._robot is not None:
            from omni.isaac.core.utils.types import ArticulationAction  # type: ignore[import]
            current = self._robot.get_joint_positions()
            target = current + action * self.config.physics_dt
            self._robot.apply_action(ArticulationAction(joint_positions=target))

    def check_collision(self) -> bool:
        return False  # Real implementation queries PhysX contact reports

    def get_ee_pose(self) -> np.ndarray:
        if self._robot is not None:
            pos, quat = self._robot.end_effector.get_world_pose()
            return np.concatenate([pos, quat])
        return np.zeros(7)

    def close(self) -> None:
        if hasattr(self, "_app") and self._app is not None:
            self._app.close()
        logger.info("IsaacEnv closed")


# ---------------------------------------------------------------------------
# Mock backend — fully functional, no GPU required
# ---------------------------------------------------------------------------

class MockIsaacEnv(BaseRobotEnv):
    """
    Deterministic mock of an Isaac Sim environment.

    Simulates a 9-DoF Franka Panda arm with simple forward-kinematics,
    object state tracking, grasp/release logic, and configurable sensor noise.
    Suitable for unit testing, CI, and rapid prototyping.
    """

    # Franka Panda home joint angles (radians)
    _HOME_JOINTS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04])

    def __init__(self, config: IsaacEnvConfig | None = None) -> None:
        if config is None:
            config = IsaacEnvConfig(backend="mock")
        super().__init__(config)

        self._joint_pos = self._HOME_JOINTS.copy()
        self._joint_vel = np.zeros(9)
        self._ee_pose = np.array([0.3, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
        self._gripper_open: bool = True
        self._held_object: str | None = None

        # Object table: name -> (7,) pose [pos(3)+quat(4)]
        self._object_states: dict[str, np.ndarray] = {}
        self._object_masses: dict[str, float] = {}

        self._step_count = 0
        self._collision = False

        logger.info("MockIsaacEnv initialised (seed=%d)", config.seed)

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------

    def add_object(
        self,
        name: str,
        position: np.ndarray,
        mass: float = 0.1,
        quat: np.ndarray | None = None,
    ) -> None:
        """Register a new object in the scene."""
        if quat is None:
            quat = np.array([0.0, 0.0, 0.0, 1.0])
        self._object_states[name] = np.concatenate([position, quat])
        self._object_masses[name] = mass

    def remove_object(self, name: str) -> None:
        self._object_states.pop(name, None)
        self._object_masses.pop(name, None)
        if self._held_object == name:
            self._held_object = None

    # ------------------------------------------------------------------
    # BaseRobotEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> RobotObservation:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._joint_pos = self._HOME_JOINTS.copy()
        self._joint_vel = np.zeros(9)
        self._ee_pose = np.array([0.3, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
        self._gripper_open = True
        self._held_object = None
        self._step_count = 0
        self._collision = False
        return self.get_observation()

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict]:
        assert action.shape == (9,), f"Expected action shape (9,), got {action.shape}"

        self.apply_action(action)
        self._step_count += 1

        # Simple EE kinematics: EE moves proportionally to the first 3 joint deltas
        ee_delta = np.zeros(3)
        ee_delta[0] = action[1] * 0.05
        ee_delta[1] = action[0] * 0.05
        ee_delta[2] = action[3] * 0.05
        self._ee_pose[:3] = np.clip(
            self._ee_pose[:3] + ee_delta,
            [self.config.workspace_x[0], self.config.workspace_y[0], 0.05],
            [self.config.workspace_x[1], self.config.workspace_y[1], self.config.workspace_z[1]],
        )

        # Drag held object with EE
        if self._held_object is not None:
            self._object_states[self._held_object][:3] = self._ee_pose[:3] - np.array([0.0, 0.0, 0.05])

        # Collision: if EE goes below table surface
        self._collision = self._ee_pose[2] < 0.42

        obs = self.get_observation()
        return obs, 0.0, False, False, {"step": self._step_count, "collision": self._collision}

    def get_observation(self) -> RobotObservation:
        noise = self._rng.normal(0, 0.01, (3,))
        h_w = self.config.wrist_cam_resolution
        h_oh = self.config.overhead_cam_resolution
        return RobotObservation(
            rgb_wrist=self._render_mock_image(h_w),
            rgb_overhead=self._render_mock_image(h_oh),
            depth_overhead=self._render_mock_depth(h_oh),
            joint_positions=self._joint_pos.copy() + self._rng.normal(0, 0.001, 9),
            joint_velocities=self._joint_vel.copy() + self._rng.normal(0, 0.001, 9),
            end_effector_pose=self._ee_pose.copy() + np.concatenate([noise, np.zeros(4)]),
            object_states={k: v.copy() for k, v in self._object_states.items()},
            timestamp=time.time(),
        )

    def get_object_states(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._object_states.items()}

    def apply_action(self, action: np.ndarray) -> None:
        """Apply joint delta action; last 2 joints control the gripper."""
        self._joint_pos[:7] = np.clip(
            self._joint_pos[:7] + action[:7] * self.config.physics_dt,
            -np.pi, np.pi,
        )
        # Gripper: action[7] > 0 → open, < 0 → close
        gripper_cmd = action[7] if len(action) > 7 else 0.0
        if gripper_cmd > 0.1 and not self._gripper_open:
            self._open_gripper()
        elif gripper_cmd < -0.1 and self._gripper_open:
            self._close_gripper()
        self._joint_vel = action.copy()

    def check_collision(self) -> bool:
        return self._collision

    def get_ee_pose(self) -> np.ndarray:
        return self._ee_pose.copy()

    def close(self) -> None:
        logger.info("MockIsaacEnv closed after %d steps", self._step_count)

    # ------------------------------------------------------------------
    # Gripper helpers
    # ------------------------------------------------------------------

    def _open_gripper(self) -> None:
        self._gripper_open = True
        self._joint_pos[7:] = 0.04
        if self._held_object is not None:
            # Drop object at current EE position
            self._object_states[self._held_object][:3] = self._ee_pose[:3] - np.array([0.0, 0.0, 0.05])
            self._held_object = None

    def _close_gripper(self) -> None:
        self._joint_pos[7:] = 0.0
        # Check if any object is close enough to grasp
        ee_pos = self._ee_pose[:3]
        for name, state in self._object_states.items():
            obj_pos = state[:3]
            dist = float(np.linalg.norm(ee_pos - obj_pos))
            if dist < 0.08 and self._held_object is None:
                self._held_object = name
                self._gripper_open = False
                logger.debug("MockIsaacEnv: grasped '%s'", name)
                return
        self._gripper_open = False

    # ------------------------------------------------------------------
    # Mock rendering
    # ------------------------------------------------------------------

    def _render_mock_image(self, resolution: tuple[int, int]) -> np.ndarray:
        """Generate a plausible-looking synthetic image for testing."""
        h, w = resolution
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Table surface (brown)
        table_y = int(h * 0.6)
        img[table_y:, :] = [139, 90, 43]
        # Sky/background (light grey)
        img[:table_y, :] = [200, 200, 200]
        # Draw objects as coloured rectangles
        for name, state in self._object_states.items():
            pos = state[:3]
            px = int((pos[0] - 0.3) / 0.6 * w * 0.5 + w * 0.3)
            py = int(table_y - (pos[2] - 0.42) / 0.4 * h * 0.3)
            color = [200, 50, 50] if "red" in name else [50, 50, 200]
            x0, x1 = max(0, px - 10), min(w, px + 10)
            y0, y1 = max(0, py - 10), min(h, py + 10)
            img[y0:y1, x0:x1] = color
        return img

    def _render_mock_depth(self, resolution: tuple[int, int]) -> np.ndarray:
        """Generate a synthetic depth map."""
        h, w = resolution
        depth = np.ones((h, w), dtype=np.float32) * 1.5
        table_y = int(h * 0.6)
        depth[table_y:, :] = 0.8
        return depth
