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

"""Saved camera/state results, pixel geometry and segmentation on the selected frame."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
import pytest

from robots.behavior.primitives import BehaviorPrimitives
from robots.behavior.toolkit import BehaviorToolkit
from robots.behavior.tools import (
    estimate_mask_position_in_robot_base,
    extract_public_proprio,
    locate_pixel_in_robot_base,
    prepare_behavior_observation,
)
from rpent.robots.components.pi05_vla_client import Pi05VLAClient
from tests.unit_tests.robots.behavior.conftest import (
    _GEOMETRY_CAMERAS,
    _SAM,
    _geometry_observation,
    _GeometryEnvironment,
    _GripperEnvironment,
    _GripperModel,
    _template_toolkit,
)

_CAMERAS = ("zed_link", "left_realsense_link", "right_realsense_link")


def _raw(batched=False, channels=4):
    raw = {
        f"robot::{camera}:Camera:0::rgb": np.full((5, 7, channels), i, np.uint8)
        for i, camera in enumerate(_CAMERAS)
    }
    raw["robot::proprio"] = np.arange(61, dtype=np.float32)
    raw["task_id"] = np.array(40)
    raw["robot::zed_link:Camera:0::depth_linear"] = np.full((5, 7, 1), 2.0, np.float32)
    if batched:
        raw = {key: value[None] for key, value in raw.items()}
    raw["oracle_object_positions"] = {"secret": [1, 2, 3]}
    return raw


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("channels", [3, 4])
def test_sensor_arrays_preserved_without_model_transforms_or_oracles(batched, channels):
    raw = _raw(batched, channels)
    for value in raw.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    result = prepare_behavior_observation(raw, task_id=40, task_language="popcorn")
    assert result["proprio"].shape == (61,)
    assert result["views"]["head"]["rgb"].shape == (5, 7, 3)
    assert result["views"]["head"]["depth"].shape == (5, 7)
    np.testing.assert_array_equal(result["views"]["head"]["depth"], 2.0)
    policy = result["policy_observation"]
    assert "oracle_object_positions" not in policy
    assert policy["robot::zed_link:Camera:0::rgb"].shape == (5, 7, channels)
    assert policy["robot::proprio"].shape == (61,)
    native_depth = policy["robot::zed_link:Camera:0::depth_linear"]
    assert native_depth.shape == (5, 7, 1)
    assert native_depth.dtype == np.float32
    np.testing.assert_array_equal(native_depth, 2.0)
    assert np.shares_memory(result["proprio"], raw["robot::proprio"])
    assert np.shares_memory(policy["robot::proprio"], result["proprio"])
    assert np.shares_memory(native_depth, raw["robot::zed_link:Camera:0::depth_linear"])
    assert np.shares_memory(result["views"]["head"]["depth"], native_depth)
    for camera in _CAMERAS:
        key = f"robot::{camera}:Camera:0::rgb"
        assert np.shares_memory(policy[key], raw[key])
    if channels == 3:
        assert np.shares_memory(
            result["views"]["head"]["rgb"], raw["robot::zed_link:Camera:0::rgb"]
        )
    encoded = Pi05VLAClient(None, embodiment="behavior").encode_obs(
        {**policy, "task_descriptions": "popcorn"}
    )
    assert encoded["main_images"].shape == (1, 5, 7, 3)
    assert encoded["wrist_images"].shape == (1, 2, 5, 7, 3)
    np.testing.assert_array_equal(encoded["wrist_images"][0, 0], 1)
    np.testing.assert_array_equal(encoded["wrist_images"][0, 1], 2)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("byte_keys", [False, True])
def test_robot_state_reading_needs_no_cameras_and_checks_task_id(batched, byte_keys):
    proprio = _proprio()
    proprio.setflags(write=False)
    state_key = b"other::proprio" if byte_keys else "other::proprio"
    task_key = b"task_id" if byte_keys else "task_id"
    raw = {state_key: proprio[None] if batched else proprio, task_key: np.array([40])}
    primitives = BehaviorPrimitives(
        env=None,
        model=None,
        task_id=40,
        task_language="popcorn",
        check_cancelled=lambda: None,
        execution_horizon=1,
    )
    assert np.shares_memory(extract_public_proprio(raw, task_id=40), proprio)
    state = primitives._robot_state(raw)
    np.testing.assert_array_equal(state["left"]["arm_qpos"], proprio[3:10])
    np.testing.assert_array_equal(state["right"]["eef_position"], proprio[42:45])
    raw[task_key] = np.array([41])
    with pytest.raises(ValueError, match="task_id does not match"):
        primitives._robot_state(raw)


_SENSORS = {
    "head": "zed_link",
    "left_wrist": "left_realsense_link",
    "right_wrist": "right_realsense_link",
}


def _observation_geometry_raw(
    *, batched=False, native=False, depth_suffix="depth_linear"
):
    raw = {"robot_r1::proprio": np.zeros(61), "task_id": np.array([40])}
    for name, sensor in _SENSORS.items():
        size = (720 if name == "head" else 480) if native else 224
        prefix = f"robot_r1::{sensor}:Camera:0"
        raw[f"{prefix}::rgb"] = np.zeros((size, size, 4), dtype=np.uint8)
        raw[f"{prefix}::{depth_suffix}"] = np.full((size, size), 2.0, dtype=np.float32)
    # r1pro.yaml's public order is left_wrist, right_wrist, head, each xyz+xyzw.
    raw["robot_r1::cam_rel_poses"] = np.array(
        [[10, 20, 30, 0, 0, 0, 1], [40, 50, 60, 0, 0, 0, 1], [1, 2, 3, 0, 0, 0, 1]],
        dtype=np.float32,
    ).reshape(21)
    return {key: value[None] for key, value in raw.items()} if batched else raw


def _snapshot(raw):
    return prepare_behavior_observation(raw, task_id=40, task_language="make popcorn")


@pytest.mark.parametrize(("batched", "native"), [(True, False), (False, True)])
def test_camera_order_calibration_and_unchanged_vla_payload(batched, native):
    raw = _observation_geometry_raw(batched=batched, native=native)
    snapshot = _snapshot(raw)
    original_policy = _snapshot(
        {key: value for key, value in raw.items() if not key.endswith("cam_rel_poses")}
    )["policy_observation"]
    assert snapshot["policy_observation"].keys() == original_policy.keys()
    for key, value in original_policy.items():
        np.testing.assert_array_equal(snapshot["policy_observation"][key], value)
        assert snapshot["policy_observation"][key].dtype == value.dtype
    for camera, position, native_size, focal in (
        ("head", [1, 2, 3], 720, 306.0),
        ("left_wrist", [10, 20, 30], 480, 388.6639),
        ("right_wrist", [40, 50, 60], 480, 388.6639),
    ):
        metadata = snapshot["views"][camera]["camera_meta"]
        size = native_size if native else 224
        assert metadata["pose_xyz_xyzw"][:3] == position
        assert metadata["pose_source"] == "robot_r1::cam_rel_poses"
        assert metadata["depth_type"] == "distance_to_image_plane"
        assert metadata["target_frame"] == "robot_base"
        np.testing.assert_allclose(
            metadata["intrinsic_K"],
            [
                [focal * size / native_size, 0, size / 2],
                [0, focal * size / native_size, size / 2],
                [0, 0, 1],
            ],
        )
        json.dumps(metadata, allow_nan=False)


def test_off_center_pixel_respects_row_col_and_xyzw_opengl_axes():
    raw = _observation_geometry_raw(native=True)
    # Rotate the sensor 90 degrees around the robot base's z axis.
    raw["robot_r1::cam_rel_poses"].reshape(3, 7)[2, 3:] = [0, 0, 2**-0.5, 2**-0.5]
    view = _snapshot(raw)["views"]["head"]
    # Optical [2, 1, 2] -> OpenGL [2, -1, -2] -> base [2, 4, 1].
    result = locate_pixel_in_robot_base(view, [513, 666])
    np.testing.assert_allclose(result["point_robot_base"], [2, 4, 1], atol=1e-6)


def test_radial_depth_uses_unit_ray_while_linear_depth_uses_optical_z():
    linear_view = _snapshot(_observation_geometry_raw(native=True))["views"]["head"]
    radial_view = _snapshot(
        _observation_geometry_raw(native=True, depth_suffix="depth")
    )["views"]["head"]
    linear = locate_pixel_in_robot_base(linear_view, [360, 666])
    radial = locate_pixel_in_robot_base(radial_view, [360, 666])
    np.testing.assert_allclose(linear["point_robot_base"], [3, 2, 1])
    np.testing.assert_allclose(radial["point_robot_base"], [1 + 2**0.5, 2, 3 - 2**0.5])
    assert radial["depth_type"] == "distance_to_camera"
    for view, expected in (
        (linear_view, [0, 2, 0]),
        (radial_view, [1 - 2**-0.5, 2, 3 - 3 * 2**-0.5]),
    ):
        mask = np.zeros(view["depth"].shape, dtype=bool)
        mask[360, [54, 207, 666]] = True
        view["depth"][360, [54, 207, 666]] = [1, 10, 3]
        mask[0, 0] = True
        view["depth"][0, 0] = np.nan
        summary = estimate_mask_position_in_robot_base(view, mask)
        # Median of projected XYZ, not projection of median pixel/depth.
        np.testing.assert_allclose(summary["point_robot_base"], expected)
        assert summary["valid_depth_pixels"] == 3


def _toolkit(
    tmp_path: Path, env: _GeometryEnvironment
) -> AbstractContextManager[BehaviorToolkit]:
    return _template_toolkit(
        tmp_path,
        ["view_env_state", "vla_act", "pixel_to_robot_base", "finish"],
        env=env,
    )


def test_geometry_uses_the_requested_saved_frame_without_reading_or_moving_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    env = _GeometryEnvironment(_geometry_observation())
    with _toolkit(tmp_path, env) as toolkit:
        definition = next(
            item for item in toolkit.list_tools() if item.name == "pixel_to_robot_base"
        )
        assert definition.readonly
        assert "state" not in definition.input_schema["properties"]
        action = toolkit.execute_tool("vla_act", {})
        assert not action.is_error
        assert env.actions == 32 and toolkit.state.latest_step == 1
        reads = env.reads
        # A later live pose/depth must not leak into either stored record.
        env.raw = _geometry_observation(offset=99, depth=7)
        old = toolkit.execute_tool(
            "pixel_to_robot_base", {"camera": "head", "pixel": [112, 112], "step": 0}
        )
        latest = toolkit.execute_tool(
            "pixel_to_robot_base", {"camera": "head", "pixel": [112, 112]}
        )
        assert not old.is_error and not latest.is_error
        np.testing.assert_allclose(old.data["point_robot_base"], [3, 0, -2])
        np.testing.assert_allclose(latest.data["point_robot_base"], [13, 0, -4])
        assert old.data["step"] == old.data["coordinate_frame_step"] == 0
        assert latest.data["step"] == latest.data["coordinate_frame_step"] == 1
        assert old.data["frame"] == "robot_base" and old.data["unit"] == "metre"
        assert old.data["pixel_order"] == "row_col"
        assert env.reads == reads and env.actions == 32
        assert toolkit.state.latest_step == 1
        assert old.images == latest.images == []


@pytest.mark.parametrize("missing", ["calibration", "depth", "unsupported_resolution"])
def test_unavailable_geometry_returns_an_error_without_a_fabricated_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    raw = _geometry_observation(size=64 if missing == "unsupported_resolution" else 224)
    if missing == "calibration":
        raw.pop("robot_r1::cam_rel_poses")
    if missing == "depth":
        raw.pop(_GEOMETRY_CAMERAS[0].replace("::rgb", "::depth_linear"))
    env = _GeometryEnvironment(raw)
    with _toolkit(tmp_path, env) as toolkit:
        reads = env.reads
        result = toolkit.execute_tool(
            "pixel_to_robot_base", {"camera": "head", "pixel": [10, 10]}
        )
        assert result.is_error
        assert result.data["code"] == "geometry_unavailable"
        assert "point_robot_base" not in result.data
        assert env.reads == reads and env.actions == 0
        assert toolkit.state.latest_step == 0


def test_observation_includes_current_measured_state_without_reader_tool(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    env = _GripperEnvironment()
    env.raw["r1::proprio"][17:20] = [0.81234, -0.12346, 0.91234]
    env.raw["r1::proprio"][:3] = [0.12, -0.03, 0.2]
    before = env.raw["r1::proprio"].copy()
    with _template_toolkit(
        tmp_path,
        ["view_env_state", "vla_act", "finish"],
        env=env,
        model=_GripperModel(),
        execution_horizon=2,
    ) as toolkit:
        initial = toolkit.execute_tool("view_env_state", {})
        summary = initial.data["state"]["robot_state"]
        assert not initial.is_error and len(initial.images) == 3
        assert summary["available"] and summary["source_step"] == 0
        assert summary["left"]["eef_position"] == [0.8123, -0.1235, 0.9123]
        assert summary["left"]["finger_qpos"] == [0.04, 0.04]
        assert summary["base_qvel"] == [0.12, -0.03, 0.2]
        assert (
            summary["units"]["arm_qpos"] == summary["units"]["trunk_qpos"] == "radian"
        )
        assert summary["units"]["finger_qpos"] == "metre per finger"
        assert summary["units"]["base_qvel"] == [
            "metre/second",
            "metre/second",
            "radian/second",
        ]
        assert "arm_qvel" not in summary["left"]
        assert "read_robot_state" not in {tool.name for tool in toolkit.list_tools()}
        assert not env.actions
        np.testing.assert_array_equal(toolkit.state.load("proprio.npy", step=0), before)
        env.raw["r1::proprio"][17] = 0.62346
        action = toolkit.execute_tool("vla_act", {})
        assert action.data["state"]["robot_state"]["source_step"] == 1
        assert action.data["state"]["robot_state"]["left"]["eef_position"][0] == 0.6235
        assert action.data["log"]["result"]["issued_actions"] == 2
        assert len(env.actions) == 1


def _native_tools_toolkit(tmp_path, env, sam):
    return _template_toolkit(
        tmp_path,
        ["view_env_state", "vla_act", "segment", "set_gripper", "release", "finish"],
        env=env,
        sam=sam,
    )


def test_native_segment_uses_same_saved_rgb_and_behavior_base_frame(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    env, sam = _GeometryEnvironment(_geometry_observation()), _SAM()
    with _native_tools_toolkit(tmp_path, env, sam) as toolkit:
        declarations = {tool.name: tool for tool in toolkit.list_tools()}
        assert {"segment", "set_gripper", "release"} <= declarations.keys()
        assert declarations["segment"].readonly
        assert {"state", "segmenter"}.isdisjoint(
            declarations["segment"].input_schema["properties"]
        )
        toolkit.execute_tool("vla_act", {})
        reads, actions = env.reads, env.actions
        result = toolkit.execute_tool("segment", {"prompt": "popcorn bag", "step": 0})
        assert result.data["found"] is True
        assert result.data["geometry_valid"] is True
        assert result.data["frame"] == "robot_base"
        assert result.data["coordinate_frame_step"] == 0
        assert "world_xyz" not in result.data
        np.testing.assert_allclose(result.data["point_robot_base"], [3, 0, -2])
        assert sam.calls[0][0] == toolkit.state.load_bytes("head_rgb.png", step=0)
        assert sam.calls[0][1]["text_prompt"] == "popcorn bag"
        artifact_name = result.data["segment_artifact"]
        artifact = toolkit.state.load(artifact_name, step=0)
        assert artifact["source_step"] == artifact["coordinate_frame_step"] == 0
        assert artifact["point_robot_base"] == result.data["point_robot_base"]
        assert not toolkit.state.exists(artifact_name, step=1)
        assert len(result.images) == 1
        overlay = toolkit.state.load(result.data["overlay_artifact"], step=0)
        assert overlay[112, 112, 0] > 0
        assert np.all(toolkit.state.load("head_rgb.png", step=0) == 0)
        assert (
            env.reads == reads
            and env.actions == actions
            and toolkit.state.latest_step == 1
        )


def test_native_segment_keeps_mask_when_behavior_depth_is_unavailable(tmp_path):
    raw = _geometry_observation()
    raw.pop("robot_r1::robot_r1:zed_link:Camera:0::depth_linear")
    with _native_tools_toolkit(tmp_path, _GeometryEnvironment(raw), _SAM()) as toolkit:
        result = toolkit.execute_tool("segment", {"point": [112, 112]})
        assert result.data["found"] is True
        assert result.data["geometry_valid"] is False
        assert result.data["point_robot_base"] is None and result.data["geometry_error"]
        assert len(result.images) == 1


def _proprio():
    result = np.zeros(61, dtype=np.float64)
    result[3:10] = np.arange(7) * 0.02
    result[28:35] = np.arange(7) * -0.03
    result[17:20] = [0.4, 0.2, 0.8]
    result[42:45] = [0.4, -0.2, 0.8]
    result[[23, 48]] = 1
    result[24:26] = [0.02, 0.02]
    result[49:51] = [0.01, 0.01]
    result[53:57] = [0.1, 0.2, 0.3, 0.4]
    return result


def test_robot_reader_uses_saved_record_without_advancing_or_calling_vla(tmp_path):
    env, model = _GripperEnvironment(), _GripperModel()
    env.raw["r1::proprio"] = _proprio()
    with _template_toolkit(
        tmp_path, ["view_env_state", "read_robot_state", "finish"], env=env, model=model
    ) as toolkit:
        env.raw["r1::proprio"][17] = 9
        result = toolkit.execute_tool("read_robot_state", {"step": 0})
        assert result.data["robot_state"]["left"]["eef_position"][0] == 0.4
        assert env.actions == [] and model.calls == 0
        assert toolkit.state.latest_step == 0
