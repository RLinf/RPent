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

from robots.dual_franka.perception import (
    back_project_base_pixel,
    load_calibration_bundle,
)
from robots.dual_franka.runtime_config import DEFAULT_CONFIG
from robots.dual_franka.tools import (
    DualFrankaPrimitives,
    coerce_arm,
    coerce_vec3,
    dump_state,
    view_env_state,
)
from robots.franka.runtime_config import (
    set_calibration_path,
    set_robot_config_path,
)
from robots.franka.tools import view_camera_meta
from rpent.session import EnvState
from rpent.tools.toolkit import ToolResult


class FakeEnv:
    def __init__(self) -> None:
        self.moves: list[tuple[str, np.ndarray]] = []
        self.rotations: list[tuple[str, np.ndarray]] = []
        self.grippers: list[tuple[str, bool]] = []
        self.chunks: list[np.ndarray] = []
        self.observation_calls = 0

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

    def _obs(self):
        return {
            "main_images": np.zeros((8, 8, 3), dtype=np.uint8),
            "extra_view_images": np.ones((2, 8, 8, 3), dtype=np.uint8),
            "main_depths": np.ones((8, 8), dtype=np.float32),
            "extra_view_depths": np.ones((2, 8, 8), dtype=np.float32) * 2,
            "d455_images": np.ones((8, 8, 3), dtype=np.uint8) * 3,
            "d455_depths": np.ones((8, 8), dtype=np.float32) * 4,
            "raw_camera_frames": {
                "left_wrist_0_rgb": np.full((10, 12, 3), 5, dtype=np.uint8),
                "base_0_rgb": np.full((10, 12, 3), 7, dtype=np.uint8),
                "right_wrist_0_rgb": np.full((10, 12, 3), 6, dtype=np.uint8),
            },
            "raw_camera_depths": {
                "left_wrist_0_rgb": np.full((10, 12), 8, dtype=np.float32),
                "base_0_rgb": np.full((10, 12), 9, dtype=np.float32),
                "right_wrist_0_rgb": np.full((10, 12), 11, dtype=np.float32),
            },
            "states": np.zeros(20, dtype=np.float32),
        }

    def get_observation(self):
        self.observation_calls += 1
        return self._obs()

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
        return {"terminated": False, "truncated": False, "observation": self._obs()}


def _primitives(env: FakeEnv, *, check_cancelled=lambda: None):
    return DualFrankaPrimitives(
        env=env,
        model=None,
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
    # Every VLA camera persists its raw frame as the single canonical version.
    np.testing.assert_array_equal(state.load("left_wrist.png"), 5)
    np.testing.assert_array_equal(state.load("base.png"), 7)
    np.testing.assert_array_equal(state.load("right_wrist.png"), 6)
    np.testing.assert_array_equal(state.load("left_wrist_depth.npy"), 8)
    np.testing.assert_array_equal(state.load("base_depth.npy"), 9)
    np.testing.assert_array_equal(state.load("right_wrist_depth.npy"), 11)
    camera_meta = view_camera_meta(state=state)["camera_meta"]
    assert camera_meta["observation_camera_map"]["main"] == "left_wrist_0_rgb"


def test_dump_state_falls_back_to_policy_view_when_raw_missing(tmp_path: Path):
    env = FakeEnv()
    full_obs = env.get_observation()

    def base_raw_only() -> dict:
        obs = dict(full_obs)
        obs["raw_camera_frames"] = {
            "base_0_rgb": full_obs["raw_camera_frames"]["base_0_rgb"]
        }
        obs["raw_camera_depths"] = {
            "base_0_rgb": full_obs["raw_camera_depths"]["base_0_rgb"]
        }
        return obs

    env.get_observation = base_raw_only
    state = EnvState(tmp_path)
    dump_state(_primitives(env), state, command=None, result=None, elapsed_s=None)

    # base keeps its raw frame; the wrists fall back to the policy views.
    np.testing.assert_array_equal(state.load("base.png"), 7)
    np.testing.assert_array_equal(state.load("left_wrist.png"), 0)  # main_images
    np.testing.assert_array_equal(
        state.load("right_wrist.png"), 1
    )  # extra_view_images[1]


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
    set_robot_config_path(DEFAULT_CONFIG)
    result = back_project_base_pixel(row=2, col=2, state=state)

    assert result["coordinate_frame"] == "right_base"
    assert result["depth_m"] == 0.5
    assert len(result["point_xyz"]) == 3


def test_load_calibration_bundle_follows_robot_config_override(tmp_path: Path):
    config = tmp_path / "robot_config.yaml"
    config.write_text(
        "perception:\n"
        "  localization_validity:\n"
        "    base_camera:\n"
        "      depth_m: [0.2, 0.9]\n"
        "  base_frames:\n"
        "    T_right_base_left_base:\n"
        "      matrix:\n"
        "        - [1.0, 0.0, 0.0, 0.02]\n"
        "        - [0.0, 1.0, 0.0, 0.7]\n"
        "        - [0.0, 0.0, 1.0, 0.0]\n"
        "        - [0.0, 0.0, 0.0, 1.0]\n"
    )
    set_calibration_path(
        Path(__file__).parent / "fixtures" / "hand_eye_calibration.json"
    )
    set_robot_config_path(config)
    try:
        bundle = load_calibration_bundle()
    finally:
        set_robot_config_path(None)

    assert bundle["base_camera"]["localization_validity"] == {"depth_m": [0.2, 0.9]}
    assert bundle["base_frames"]["T_right_base_left_base"]["matrix"][0][3] == 0.02
    # Hand-eye transforms from the calibration bundle survive the merge.
    assert "transformation" in bundle["d455_camera"]
