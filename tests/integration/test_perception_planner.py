"""Integration tests: perception → scene parser → planner handoff."""
import numpy as np
import pytest

from envs.isaac.isaac_env import IsaacEnvConfig, MockIsaacEnv
from src.perception.object_detector import ObjectDetector
from src.perception.scene_parser import SceneParser
from src.state_encoder.world_state_encoder import WorldStateEncoder
from src.planner.llm_planner import MockLLMPlanner


@pytest.fixture
def full_env():
    env = MockIsaacEnv(IsaacEnvConfig(seed=42))
    env.reset()
    env.add_object("red_block", np.array([0.45, 0.0, 0.45]))
    env.add_object("bin_A", np.array([0.6, 0.3, 0.42]))
    return env


class TestPerceptionPlannerHandoff:
    def test_scene_parser_produces_scene_state(self, full_env):
        obs = full_env.get_observation()
        parser = SceneParser(use_gt_states=True)
        scene = parser.parse(
            rgb_image=obs.rgb_overhead,
            depth_image=obs.depth_overhead,
            robot_obs={"ee_pose": obs.end_effector_pose, "robot_state": np.zeros(18)},
            object_states=obs.object_states,
            task_goal="pick and place",
        )
        assert "red_block" in scene.objects
        assert "bin_A" in scene.objects
        assert scene.task_goal == "pick and place"

    def test_scene_contains_correct_positions(self, full_env):
        obs = full_env.get_observation()
        parser = SceneParser(use_gt_states=True)
        scene = parser.parse(
            rgb_image=obs.rgb_overhead,
            depth_image=obs.depth_overhead,
            robot_obs={"ee_pose": obs.end_effector_pose},
            object_states=obs.object_states,
        )
        pos = scene.objects["red_block"].position
        assert abs(pos[0] - 0.45) < 0.05
        assert abs(pos[2] - 0.45) < 0.05

    def test_scene_classifies_containers(self, full_env):
        obs = full_env.get_observation()
        parser = SceneParser(use_gt_states=True, container_labels={"bin_a", "bin"})
        scene = parser.parse(
            rgb_image=obs.rgb_overhead,
            depth_image=obs.depth_overhead,
            robot_obs={"ee_pose": obs.end_effector_pose},
            object_states=obs.object_states,
        )
        assert scene.objects["bin_A"].is_container
        assert not scene.objects["red_block"].is_container

    def test_planner_receives_scene_and_returns_plan(self, full_env):
        obs = full_env.get_observation()
        parser = SceneParser(use_gt_states=True)
        scene = parser.parse(
            rgb_image=obs.rgb_overhead,
            depth_image=obs.depth_overhead,
            robot_obs={"ee_pose": obs.end_effector_pose},
            object_states=obs.object_states,
            task_goal="pick up the red block and place it in the bin",
        )
        planner = MockLLMPlanner()
        result = planner.plan(scene, "pick up the red block and place it in the bin")
        assert result.valid
        assert len(result.steps) >= 2

    def test_world_state_encoder_produces_correct_dim(self, full_env):
        obs = full_env.get_observation()
        parser = SceneParser(use_gt_states=True)
        scene = parser.parse(
            rgb_image=obs.rgb_overhead,
            depth_image=obs.depth_overhead,
            robot_obs={"ee_pose": obs.end_effector_pose, "robot_state": np.zeros(18)},
            object_states=obs.object_states,
        )
        encoder = WorldStateEncoder(max_objects=5, goal_dim=64)
        encoded = encoder.encode(scene)
        flat = encoded.to_flat_vector()
        assert flat.shape == (encoder.state_dim,)

    def test_encoded_state_has_correct_dtype(self, full_env):
        obs = full_env.get_observation()
        parser = SceneParser(use_gt_states=True)
        scene = parser.parse(
            rgb_image=obs.rgb_overhead,
            depth_image=obs.depth_overhead,
            robot_obs={"ee_pose": obs.end_effector_pose},
            object_states=obs.object_states,
        )
        encoder = WorldStateEncoder()
        encoded = encoder.encode(scene)
        assert encoded.to_flat_vector().dtype == np.float32

    def test_prompt_string_non_empty(self, full_env):
        obs = full_env.get_observation()
        parser = SceneParser(use_gt_states=True)
        scene = parser.parse(
            rgb_image=obs.rgb_overhead,
            depth_image=obs.depth_overhead,
            robot_obs={"ee_pose": obs.end_effector_pose},
            object_states=obs.object_states,
            task_goal="test goal",
        )
        prompt = scene.to_prompt_string()
        assert "test goal" in prompt
        assert "red_block" in prompt
