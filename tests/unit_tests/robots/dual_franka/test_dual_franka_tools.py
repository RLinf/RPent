# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline dual-Franka primitive and state-capture tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robots.dual_franka.perception import back_project, segment
from robots.dual_franka.tasks import CLEAN_DESK_VLA_PROMPT
from robots.dual_franka.tools import (
    DualFrankaPrimitives,
    coerce_arm,
    coerce_vec3,
    dump_state,
    view_env_state,
)
from robots.franka.runtime_config import set_calibration_path
from robots.franka.tools import view_camera_meta
from rpent.session import EnvState
from rpent.tools.toolkit import ToolResult


class FakeEnv:
    def __init__(self) -> None:
        self.moves: list[tuple[str, np.ndarray]] = []
        self.rotations: list[tuple[str, np.ndarray]] = []
        self.grippers: list[tuple[str, bool]] = []
        self.chunks: list[np.ndarray] = []

    def reset(self):
        return {"ok": True}

    def move_delta(self, arm, value):
        self.moves.append((arm, np.asarray(value)))
        return {"ok": True, "arm": arm}

    def rotate_delta(self, arm, value):
        self.rotations.append((arm, np.asarray(value)))
        return {"ok": True, "arm": arm}

    def set_gripper(self, arm, *, open: bool):
        self.grippers.append((arm, open))
        return {"ok": True, "arm": arm, "open": open}

    def recover_joint_posture(self, *, reason="", return_to_start=True):
        return {
            "ok": True,
            "reason": reason,
            "return_to_start": return_to_start,
        }

    def get_observation(self):
        return {
            "main_images": np.zeros((8, 8, 3), dtype=np.uint8),
            "extra_view_images": np.ones((2, 8, 8, 3), dtype=np.uint8),
            "main_depths": np.ones((8, 8), dtype=np.float32),
            "extra_view_depths": np.ones((2, 8, 8), dtype=np.float32) * 2,
            "d455_images": np.ones((8, 8, 3), dtype=np.uint8) * 3,
            "d455_depths": np.ones((8, 8), dtype=np.float32) * 4,
            "raw_camera_frames": {
                "base_0_rgb": np.full((10, 12, 3), 7, dtype=np.uint8),
            },
            "raw_camera_depths": {
                "base_0_rgb": np.full((10, 12), 9, dtype=np.float32),
            },
            "states": np.zeros(20, dtype=np.float32),
        }

    def get_robot_state(self):
        return {
            "left_arm": {
                "tcp_pose": [0.5, -0.2, 0.5, 0.0, 0.0, 0.0, 1.0],
                "gripper_open": True,
            },
            "right_arm": {
                "tcp_pose": [0.5, 0.2, 0.5, 0.0, 0.0, 0.0, 1.0],
                "gripper_open": True,
            },
        }

    def get_camera_meta(self):
        return {
            "cameras": {"left_wrist_0_rgb": {"serial": "left", "type": "zed"}},
            "observation_camera_map": {
                "main": "left_wrist_0_rgb",
                "extra_0": "base_0_rgb",
                "extra_1": "right_wrist_0_rgb",
            },
            "agent_observation": {
                "inline_cameras": ["d455"],
                "auxiliary_cameras": ["left_wrist", "base", "right_wrist"],
            },
            "projection_views": {
                "base": {"raw_key": "base_0_rgb", "calibration_key": "base_camera"},
                "d455": {"raw_key": "d455_rgb", "calibration_key": "d455_camera"},
            },
        }

    def chunk_step(self, actions):
        self.chunks.append(np.asarray(actions))
        return {"terminated": False, "truncated": False}


class FakeModel:
    def __init__(self, expected_prompt="hand over the cube") -> None:
        self.expected_prompt = expected_prompt

    def predict(self, observation, options=None):
        assert observation["task_descriptions"] == self.expected_prompt
        assert options == {"mode": "eval"}
        return np.zeros((2, 20), dtype=np.float32)


class FakeSam3Client:
    def segment(self, image, *, text_prompt=None, point=None, min_score=0.2):
        assert text_prompt == "white cardboard box interior"
        assert point is None
        mask = np.zeros((8, 8), dtype=bool)
        mask[3:6, 3:6] = True
        return type(
            "FakeSam3Result",
            (),
            {
                "found": True,
                "score": 0.91,
                "box": [3.0, 3.0, 5.0, 5.0],
                "mask": mask,
                "mask_shape": mask.shape,
                "reason": None,
            },
        )()


class BoundaryEnv(FakeEnv):
    def __init__(self) -> None:
        super().__init__()
        self._right_open = True
        self._right_z = 0.5

    def get_robot_state(self):
        return {
            "left_arm": {
                "tcp_pose": [0.5, -0.2, 0.5, 0.0, 0.0, 0.0, 1.0],
                "gripper_open": True,
            },
            "right_arm": {
                "tcp_pose": [0.5, 0.2, self._right_z, 0.0, 0.0, 0.0, 1.0],
                "gripper_open": self._right_open,
            },
        }

    def chunk_step(self, actions):
        self.chunks.append(np.asarray(actions))
        self._right_open = False
        self._right_z += 0.08
        return {"terminated": False, "truncated": False}


def _primitives(env: FakeEnv, *, model=None, check_cancelled=lambda: None):
    return DualFrankaPrimitives(
        env=env,
        model=model,
        task_description="default task",
        check_cancelled=check_cancelled,
    )


def test_arm_and_vec3_validation_and_motion_forwarding():
    env = FakeEnv()
    primitives = _primitives(env)

    primitives.move_delta("left", [0.01, 0.0, -0.02])
    primitives.rotate_delta("right", [0.0, 0.0, 0.1])
    primitives.open_gripper("left")
    primitives.close_gripper("right")

    assert env.moves[0][0] == "left"
    np.testing.assert_allclose(env.moves[0][1], [0.01, 0.0, -0.02])
    assert env.rotations[0][0] == "right"
    assert env.grippers == [("left", True), ("right", False)]

    assert coerce_arm("LEFT") == "left"
    with pytest.raises(ValueError, match="left.*right"):
        coerce_arm("both")
    with pytest.raises(ValueError, match="exactly 3"):
        coerce_vec3([1.0, 2.0], name="delta")


def test_dump_state_saves_three_camera_artifacts(tmp_path: Path):
    env = FakeEnv()
    primitives = _primitives(env)
    state = EnvState(tmp_path)

    record = dump_state(
        primitives,
        state,
        command={"action": "move_delta"},
        result={"ok": True},
        elapsed_s=0.2,
    )

    assert record.artifacts == {
        "left_wrist.png",
        "left_wrist_depth.npy",
        "base.png",
        "base_depth.npy",
        "right_wrist.png",
        "right_wrist_depth.npy",
        "d455.png",
        "d455_depth.npy",
        "camera_meta.json",
    }
    output = view_env_state(state=state)
    assert output["_image_bytes"]
    assert "_image_nav_bytes" not in output
    assert "_image_cam_bytes" not in output
    assert "_image_wrist_bytes" not in output
    assert output["artifact_images"] == ["base", "d455", "left_wrist", "right_wrist"]
    assert output["image_block_order"] == ["d455"]
    np.testing.assert_array_equal(state.load("base.png"), 7)
    np.testing.assert_array_equal(state.load("base_depth.npy"), 9)
    camera_meta = view_camera_meta(state=state)["camera_meta"]
    assert camera_meta["observation_camera_map"]["main"] == "left_wrist_0_rgb"


def test_view_env_state_emits_multimodal_image_blocks(tmp_path: Path):
    env = FakeEnv()
    primitives = _primitives(env)
    state = EnvState(tmp_path)

    dump_state(primitives, state, command=None, result=None, elapsed_s=None)
    output = view_env_state(state=state)
    # Routine planner snapshots inline only D455, while auxiliary camera
    # artifacts stay available through returned paths/read_image.
    assert output["images"] == ["d455"]
    assert output["image_block_order"] == output["images"]
    assert output["artifact_images"] == ["base", "d455", "left_wrist", "right_wrist"]

    result = ToolResult(name="view_env_state", result=output)
    image_blocks = [b for b in result.content_blocks if b.get("type") == "image"]
    assert len(image_blocks) == 1
    text_block = next(b for b in result.content_blocks if b.get("type") == "text")
    # Image bytes must be lifted out of the text block, not serialized into it.
    assert "_image_" not in text_block["text"]


def test_back_project_reads_rpent_state_artifacts(tmp_path: Path):
    state = EnvState(tmp_path)
    with state.record_step(
        state={
            "raw": {
                "left": {"tcp_pose": [0, 0, 0, 0, 0, 0, 1]},
                "right": {"tcp_pose": [0, 0, 0, 0, 0, 0, 1]},
            }
        }
    ) as step:
        state.save("base_depth.npy", np.full((4, 4), 0.5), step=step)
        state.save(
            "camera_meta.json",
            {
                "base_0_rgb": {
                    "color_intrinsics": {
                        "fx": 100,
                        "fy": 100,
                        "ppx": 2,
                        "ppy": 2,
                    }
                }
            },
            step=step,
        )

    set_calibration_path(
        Path(__file__).parent / "fixtures" / "hand_eye_calibration.json"
    )
    result = back_project(camera="base", row=2, col=2, state=state)

    assert result["coordinate_frame"] == "right_base"
    assert result["depth_m"] == 0.5
    assert len(result["point_xyz"]) == 3


def test_back_project_returns_annotated_image_block(tmp_path: Path):
    state = EnvState(tmp_path)
    with state.record_step(
        state={
            "raw": {
                "left": {"tcp_pose": np.array([0, 0, 0, 0, 0, 0, 1])},
                "right": {"tcp_pose": [0, 0, 0, 0, 0, 0, 1]},
            }
        }
    ) as step:
        state.save("d455.png", np.zeros((8, 8, 3), dtype=np.uint8), step=step)
        state.save("d455_depth.npy", np.full((8, 8), 0.5), step=step)
        state.save(
            "camera_meta.json",
            {
                "d455_rgb": {
                    "color_intrinsics": {
                        "fx": 100,
                        "fy": 100,
                        "ppx": 4,
                        "ppy": 4,
                    }
                }
            },
            step=step,
        )

    set_calibration_path(
        Path(__file__).parent / "fixtures" / "hand_eye_calibration.json"
    )
    result = back_project(row=4, col=4, state=state)

    assert result["coordinate_frame"] == "right_base"
    assert result["tcp_delta_coordinate_frame"] == "right_base"
    assert result["left_tcp_xyz"] == [0.01, 0.69, 0.0]
    assert result["right_tcp_xyz"] == [0.0, 0.0, 0.0]
    point = np.asarray(result["point_xyz"])
    np.testing.assert_allclose(
        result["delta_left_tcp_to_point_xyz"],
        point - np.asarray(result["left_tcp_xyz"]),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        result["delta_right_tcp_to_point_xyz"],
        point - np.asarray(result["right_tcp_xyz"]),
        atol=1e-5,
    )
    assert result["diagnostic_artifacts"]["annotated_image"].endswith(".png")
    assert result["_image_cam_bytes"]
    assert result["image_block_order"] == ["d455_selection_diagnostic"]

    tool_result = ToolResult(name="back_project", result=result)
    image_blocks = [b for b in tool_result.content_blocks if b.get("type") == "image"]
    assert len(image_blocks) == 1
    text_block = next(b for b in tool_result.content_blocks if b.get("type") == "text")
    assert "_image_" not in text_block["text"]


def test_segment_returns_mask_overlay_and_world_point(tmp_path: Path):
    state = EnvState(tmp_path)
    with state.record_step(
        state={
            "raw": {
                "left": {"tcp_pose": [0, 0, 0, 0, 0, 0, 1]},
                "right": {"tcp_pose": [0, 0, 0, 0, 0, 0, 1]},
            }
        }
    ) as step:
        state.save("d455.png", np.zeros((8, 8, 3), dtype=np.uint8), step=step)
        state.save("d455_depth.npy", np.full((8, 8), 0.5), step=step)
        state.save(
            "camera_meta.json",
            {
                "d455_rgb": {
                    "color_intrinsics": {
                        "fx": 100,
                        "fy": 100,
                        "ppx": 4,
                        "ppy": 4,
                    }
                }
            },
            step=step,
        )

    set_calibration_path(
        Path(__file__).parent / "fixtures" / "hand_eye_calibration.json"
    )
    result = segment(
        prompt="white cardboard box interior",
        target_name="cardboard_box_interior",
        min_valid_depth_pixels=1,
        state=state,
        sam3_client=FakeSam3Client(),
    )

    assert result["ok"]
    assert result["found"]
    assert result["coordinate_frame"] == "right_base"
    assert result["tcp_delta_coordinate_frame"] == "right_base"
    assert "delta_left_tcp_to_point_xyz" in result
    assert "delta_right_tcp_to_point_xyz" in result
    assert len(result["point_xyz"]) == 3
    assert result["centroid_pixel"] == [4, 4]
    assert result["segment_artifact"].startswith("d455_segment_")
    assert result["overlay_artifact"].startswith("d455_segment_overlay_")
    assert result["_image_cam_bytes"]
    assert result["image_block_order"] == ["d455_segment_overlay"]

    tool_result = ToolResult(name="segment", result=result)
    image_blocks = [b for b in tool_result.content_blocks if b.get("type") == "image"]
    assert len(image_blocks) == 1


def test_segment_without_sam3_client_falls_back(tmp_path: Path):
    state = EnvState(tmp_path)
    with state.record_step(state={}):
        pass

    result = segment(prompt="cup", state=state, sam3_client=None)

    assert not result["ok"]
    assert "SAM3 client is not configured" in result["error"]
    assert "back_project" in result["fallback"]


def test_vla_grasp_runs_bounded_chunks():
    env = FakeEnv()
    primitives = _primitives(env, model=FakeModel())

    result = primitives.vla_grasp("hand over the cube", max_chunks=3)

    assert result["chunks_executed"] == 3
    assert len(env.chunks) == 3


def test_recover_joint_posture_forwards_to_env():
    env = FakeEnv()
    primitives = _primitives(env)

    result = primitives.recover_joint_posture(reason="joint drift")

    assert result["ok"]
    assert result["reason"] == "joint drift"


def test_named_clean_desk_vla_uses_fixed_prompt_and_semantic_boundary():
    env = BoundaryEnv()
    primitives = _primitives(
        env,
        model=FakeModel(expected_prompt=CLEAN_DESK_VLA_PROMPT),
    )

    result = primitives.vla_right_grasp(
        prompt="grasp the next task-allowed object", max_chunks=2
    )

    assert result["ok"]
    assert result["skill_name"] == "vla_right_grasp"
    assert result["prompt_overridden"]
    assert result["boundary"] == "grasp"
    assert result["boundary_reached"]
    assert len(env.chunks) == 3
