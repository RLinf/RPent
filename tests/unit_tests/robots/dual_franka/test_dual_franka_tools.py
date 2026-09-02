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

from robots.dual_franka.perception import back_project_base_pixel
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
            "left_arm": {"tcp_pose": [0.5, -0.2, 0.5, 0.0, 0.0, 0.0, 1.0]},
            "right_arm": {"tcp_pose": [0.5, 0.2, 0.5, 0.0, 0.0, 0.0, 1.0]},
        }

    def get_camera_meta(self):
        return {
            "cameras": {"left_wrist_0_rgb": {"serial": "left", "type": "zed"}},
            "observation_camera_map": {"main": "left_wrist_0_rgb"},
        }

    def chunk_step(self, actions):
        self.chunks.append(np.asarray(actions))
        return {"terminated": False, "truncated": False}


class FakeModel:
    def predict(self, observation, options=None):
        assert observation["task_descriptions"] == "hand over the cube"
        assert options == {"mode": "eval"}
        return np.zeros((2, 20), dtype=np.float32)


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
    assert output["_image_cam_bytes"]
    assert output["_image_wrist_bytes"]
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
    # The text must name the views in the same order the image blocks are emitted.
    assert output["images"] == ["left_wrist", "base", "right_wrist"]

    result = ToolResult(name="view_env_state", result=output)
    image_blocks = [b for b in result.content_blocks if b.get("type") == "image"]
    assert len(image_blocks) == 3
    text_block = next(b for b in result.content_blocks if b.get("type") == "text")
    # Image bytes must be lifted out of the text block, not serialized into it.
    assert "_image_" not in text_block["text"]


def test_back_project_base_pixel_reads_rpent_state_artifacts(tmp_path: Path):
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
    result = back_project_base_pixel(row=2, col=2, state=state)

    assert result["coordinate_frame"] == "right_base"
    assert result["depth_m"] == 0.5
    assert len(result["point_xyz"]) == 3


def test_vla_grasp_runs_bounded_chunks():
    env = FakeEnv()
    primitives = _primitives(env, model=FakeModel())

    result = primitives.vla_grasp("hand over the cube", max_chunks=3)

    assert result["chunks_executed"] == 3
    assert len(env.chunks) == 3
