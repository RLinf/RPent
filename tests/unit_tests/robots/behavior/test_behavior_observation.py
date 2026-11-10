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

"""Public observations, robot state, geometry and SAM3 segmentation tools."""

from __future__ import annotations

import argparse
import base64
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import imageio.v2 as imageio
import numpy as np
import pytest

from robots.behavior.env_client import env_runtime_contract
from robots.behavior.robot_spec import get_robot_spec
from robots.behavior.toolkit import BehaviorToolkit
from robots.behavior.tools import (
    SavedImageSegmenter,
    SegmentProjection,
    dump_observation,
    extract_robot_state,
    snapshot_observation,
    view_env_state,
)
from robots.behavior.tools import project_pixel as pixel_to_robot_base
from rpent.dashboard.events import NullDashboardEventSink
from rpent.robots.components.sam3_client import Sam3Client, Sam3Result
from rpent.session import EnvState
from tests.unit_tests.robots.behavior.conftest import (
    _GEOMETRY_CAMERAS,
    _SAM,
    _args,
    _geometry_observation,
    _GeometryEnvironment,
    _GripperEnvironment,
    _GripperModel,
    _template_toolkit,
)

_CAMERAS = ("zed_link", "left_realsense_link", "right_realsense_link")


def _raw(batched=False):
    raw = {
        f"robot::{camera}:Camera:0::rgb": np.full((5, 7, 4), i, np.uint8)
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
def test_sensor_arrays_preserved_without_model_transforms_or_oracles(batched):
    raw = _raw(batched)
    result = snapshot_observation(raw, task_id=40, task_language="popcorn")
    assert result["proprio"].shape == (61,)
    assert result["views"]["head"]["rgb"].shape == (5, 7, 3)
    assert result["views"]["head"]["depth"].shape == (5, 7)
    np.testing.assert_array_equal(result["views"]["head"]["depth"], 2.0)
    policy = result["policy_observation"]
    assert "oracle_object_positions" not in policy
    assert policy["robot::zed_link:Camera:0::rgb"].shape == (5, 7, 4)
    assert policy["robot::proprio"].shape == (61,)
    native_depth = policy["robot::zed_link:Camera:0::depth_linear"]
    assert native_depth.shape == (5, 7, 1)
    assert native_depth.dtype == np.float32
    np.testing.assert_array_equal(native_depth, 2.0)
    native_depth[:] = 7.0
    np.testing.assert_array_equal(result["views"]["head"]["depth"], 2.0)
    np.testing.assert_array_equal(raw["robot::zed_link:Camera:0::depth_linear"], 2.0)
    result["views"]["head"]["rgb"][:] = 99
    assert not np.any(raw["robot::zed_link:Camera:0::rgb"] == 99)


def test_multi_environment_batch_is_rejected():
    raw = _raw(True)
    raw["robot::proprio"] = np.zeros((2, 61), dtype=np.float32)
    with pytest.raises(ValueError, match="one environment"):
        snapshot_observation(raw, task_id=40, task_language="popcorn")


def test_task_mismatch_is_rejected():
    with pytest.raises(ValueError, match="task_id"):
        snapshot_observation(_raw(), task_id=0, task_language="radio")


def test_ambiguous_robot_sensors_are_rejected():
    raw = _raw()
    raw["other_robot::proprio"] = np.zeros(61)
    with pytest.raises(ValueError, match="expected one sensor"):
        snapshot_observation(raw, task_id=40, task_language="popcorn")


def test_depth_rgb_misalignment_is_rejected():
    raw = _raw()
    raw["robot::zed_link:Camera:0::depth_linear"] = np.zeros((6, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="dimensions must match"):
        snapshot_observation(raw, task_id=40, task_language="popcorn")


def test_mixed_batch_layout_is_rejected():
    raw = _raw(True)
    raw["robot::zed_link:Camera:0::rgb"] = np.zeros((5, 7, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="single-slot batch"):
        snapshot_observation(raw, task_id=40, task_language="popcorn")


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
    return snapshot_observation(raw, task_id=40, task_language="make popcorn")


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


def test_public_pose_metadata_owns_its_snapshot_copy():
    raw = _observation_geometry_raw()
    snapshot = _snapshot(raw)
    raw["robot_r1::cam_rel_poses"][:] = 99
    view = snapshot["views"]["head"]
    assert pixel_to_robot_base(view, [112, 112])["point_robot_base"] == [1, 2, 1]
    raw = _observation_geometry_raw()
    snapshot = _snapshot(raw)
    snapshot["views"]["head"]["camera_meta"]["pose_xyz_xyzw"][0] = 999
    assert raw["robot_r1::cam_rel_poses"].reshape(3, 7)[2, 0] == 1


def test_missing_or_different_robot_pose_preserves_vla_only_observations():
    raw = _observation_geometry_raw()
    raw["another_robot::cam_rel_poses"] = raw.pop("robot_r1::cam_rel_poses")
    snapshot = _snapshot(raw)
    assert all("camera_meta" not in view for view in snapshot["views"].values())
    with pytest.raises(ValueError, match="no verified public camera calibration"):
        pixel_to_robot_base(snapshot["views"]["head"], [112, 112])


@pytest.mark.parametrize("invalid", ["shape", "nonfinite", "nonunit"])
def test_invalid_camera_pose_rejected_at_observation_boundary(invalid):
    raw = _observation_geometry_raw()
    poses = raw["robot_r1::cam_rel_poses"]
    if invalid == "shape":
        raw["robot_r1::cam_rel_poses"] = poses.reshape(3, 7)
    elif invalid == "nonfinite":
        poses.reshape(-1)[0] = np.nan
    else:
        poses.reshape(3, 7)[1, 6] = 2
    with pytest.raises(ValueError, match="cam_rel_poses|unit xyzw quaternion"):
        _snapshot(raw)


def test_unverified_resolution_does_not_guess_intrinsics():
    raw = _observation_geometry_raw()
    raw["robot_r1::zed_link:Camera:0::rgb"] = np.zeros((5, 7, 4), dtype=np.uint8)
    raw["robot_r1::zed_link:Camera:0::depth_linear"] = np.ones((5, 7), np.float32)
    view = _snapshot(raw)["views"]["head"]
    assert view["camera_meta"]["calibration_available"] is False
    assert "intrinsic_K" not in view["camera_meta"]
    with pytest.raises(ValueError, match="no verified public camera calibration"):
        pixel_to_robot_base(view, [2, 3])


def test_center_pixel_projects_forward_using_camera_to_base_translation():
    result = pixel_to_robot_base(
        _snapshot(_observation_geometry_raw())["views"]["head"], [112, 112]
    )
    assert result["point_robot_base"] == [1.0, 2.0, 1.0]
    assert result["frame"] == "robot_base"
    assert result["unit"] == "metre"
    assert result["pixel_order"] == "row_col"
    assert result["valid_depth_pixels"] == 1
    assert result["depth_sampling"] == "single_pixel"


def test_off_center_pixel_respects_row_col_and_xyzw_opengl_axes():
    raw = _observation_geometry_raw(native=True)
    # Rotate the sensor 90 degrees around the robot base's z axis.
    raw["robot_r1::cam_rel_poses"].reshape(3, 7)[2, 3:] = [0, 0, 2**-0.5, 2**-0.5]
    view = _snapshot(raw)["views"]["head"]
    # Optical [2, 1, 2] -> OpenGL [2, -1, -2] -> base [2, 4, 1].
    result = pixel_to_robot_base(view, [513, 666])
    np.testing.assert_allclose(result["point_robot_base"], [2, 4, 1], atol=1e-6)


def test_radial_depth_uses_unit_ray_while_linear_depth_uses_optical_z():
    linear_view = _snapshot(_observation_geometry_raw(native=True))["views"]["head"]
    radial_view = _snapshot(
        _observation_geometry_raw(native=True, depth_suffix="depth")
    )["views"]["head"]
    linear = pixel_to_robot_base(linear_view, [360, 666])
    radial = pixel_to_robot_base(radial_view, [360, 666])
    np.testing.assert_allclose(linear["point_robot_base"], [3, 2, 1])
    np.testing.assert_allclose(radial["point_robot_base"], [1 + 2**0.5, 2, 3 - 2**0.5])
    assert radial["depth_type"] == "distance_to_camera"


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf])
def test_invalid_center_depth_does_not_produce_a_point(invalid):
    view = _snapshot(_observation_geometry_raw())["views"]["head"]
    view["depth"][112, 112] = invalid
    with pytest.raises(ValueError, match="no finite positive depth"):
        pixel_to_robot_base(view, [112, 112])


def test_optional_neighborhood_uses_only_valid_depth_and_records_estimate():
    view = _snapshot(_observation_geometry_raw())["views"]["head"]
    view["depth"][111:114, 111:114] = [[np.nan, 2, 3], [0, np.inf, 5], [-1, 7, 20]]
    result = pixel_to_robot_base(view, [112, 112], neighborhood=1)
    assert result["depth_m"] == 5
    assert result["point_robot_base"] == [1, 2, -2]
    assert result["valid_depth_pixels"] == 5
    assert result["depth_sampling"] == "median_neighborhood"


@pytest.mark.parametrize("pixel", [[-1, 10], [224, 10], [1.5, 10], [True, 10], [10]])
def test_invalid_pixel_is_rejected(pixel):
    with pytest.raises(ValueError, match="pixel"):
        pixel_to_robot_base(
            _snapshot(_observation_geometry_raw())["views"]["head"], pixel
        )


@pytest.mark.parametrize("radius", [-1, 33, True, 1.0])
def test_invalid_neighborhood_radius_is_rejected(radius):
    with pytest.raises(ValueError, match="neighborhood"):
        pixel_to_robot_base(
            _snapshot(_observation_geometry_raw())["views"]["head"],
            [112, 112],
            neighborhood=radius,
        )


@pytest.mark.parametrize(
    "problem", ["unit", "shape", "quantized", "transform", "intrinsic"]
)
def test_incompatible_depth_or_calibration_is_rejected(problem):
    view = _snapshot(_observation_geometry_raw())["views"]["head"]
    if problem == "unit":
        view["camera_meta"]["depth_unit"] = "millimetre"
    elif problem == "shape":
        view["depth"] = view["depth"][:-1]
    elif problem == "quantized":
        view["depth"] = view["depth"].astype(np.uint16)
    elif problem == "transform":
        view["camera_meta"]["camera_to_base_gl"][0][0] = 2
    else:
        view["camera_meta"]["intrinsic_K"][0][0] = 0
    with pytest.raises(ValueError):
        pixel_to_robot_base(view, [112, 112])


def test_rgb_only_observation_cannot_be_backprojected():
    raw = {
        key: value
        for key, value in _observation_geometry_raw().items()
        if "depth_linear" not in key
    }
    view = _snapshot(raw)["views"]["head"]
    with pytest.raises(ValueError, match="metric depth"):
        pixel_to_robot_base(view, [112, 112])


def _toolkit(tmp_path: Path, env: _GeometryEnvironment) -> BehaviorToolkit:
    return _template_toolkit(
        tmp_path,
        ["view_env_state", "vla_act", "pixel_to_robot_base", "finish"],
        env=env,
    )


def test_camera_metadata_is_persisted_with_its_image_shape_and_not_aliased(
    tmp_path: Path,
) -> None:
    snapshot = snapshot_observation(
        _geometry_observation(), task_id=40, task_language="make popcorn"
    )
    state = EnvState(tmp_path)
    record = dump_observation(snapshot, env_state=state, log=None)
    snapshot["views"]["head"]["camera_meta"]["camera_to_base_gl"][0][3] = 999
    saved = json.loads((tmp_path / "states.json").read_text())["steps"][0]
    for view_spec in (
        record.state["view_specs"]["head"],
        saved["state"]["view_specs"]["head"],
    ):
        assert view_spec["image_shape"] == [224, 224]
        assert view_spec["camera_meta"]["calibration_available"] is True
        assert view_spec["camera_meta"]["camera_to_base_gl"][0][3] == 3
        assert view_spec["camera_meta"]["pose_index"] == 2


def test_geometry_uses_the_requested_saved_frame_without_reading_or_moving_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    env = _GeometryEnvironment(_geometry_observation())
    toolkit = _toolkit(tmp_path, env)
    try:
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
    finally:
        toolkit.close()


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
    toolkit = _toolkit(tmp_path, env)
    try:
        reads = env.reads
        result = toolkit.execute_tool(
            "pixel_to_robot_base", {"camera": "head", "pixel": [10, 10]}
        )
        assert result.is_error
        assert result.data["code"] == "geometry_unavailable"
        assert "point_robot_base" not in result.data
        assert env.reads == reads and env.actions == 0
        assert toolkit.state.latest_step == 0
    finally:
        toolkit.close()


def test_geometry_is_opt_in_for_scripted_runs_and_advertised_for_agent_runs(
    tmp_path: Path,
) -> None:
    spec = get_robot_spec()
    args = argparse.Namespace(
        output_dir=tmp_path,
        task_id=40,
        task_language="make popcorn",
        execution_horizon=32,
    )
    legacy = spec.parse_config(args)
    assert set(legacy.task_desc["tool_allowlist"]) == {
        "view_env_state",
        "vla_act",
        "finish",
    }
    assert "pixel_to_robot_base" not in spec.prompts.render(
        "system", variables=legacy.prompt_vars
    )
    args.control_mode = "agent"
    agent = spec.parse_config(args)
    assert "pixel_to_robot_base" in agent.task_desc["tool_allowlist"]
    prompt = spec.prompts.render("system", variables=agent.prompt_vars)
    assert "pixel_to_robot_base" in prompt
    assert "pixel=[row,col]" in prompt
    assert "cannot\nmove the robot" in prompt


def test_observation_includes_current_measured_state_without_reader_tool(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    env = _GripperEnvironment()
    env.raw["r1::proprio"][17:20] = [0.81234, -0.12346, 0.91234]
    env.raw["r1::proprio"][:3] = [0.12, -0.03, 0.2]
    before = env.raw["r1::proprio"].copy()
    toolkit = _template_toolkit(
        tmp_path,
        ["view_env_state", "vla_act", "finish"],
        env=env,
        model=_GripperModel(),
        execution_horizon=2,
    )
    try:
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
    finally:
        toolkit.close()


@pytest.mark.parametrize("original_size", [61, 256])
def test_old_observation_gains_readonly_summary_without_rewriting_history(
    tmp_path, original_size
):
    state = EnvState(tmp_path)
    raw = np.zeros(original_size)
    if original_size == 61:
        raw[[23, 48]] = 1
    with state.record_step(state={"old_format": True}):
        state.save("proprio.npy", raw)
        for name in ("head", "left_wrist", "right_wrist"):
            state.save(f"{name}_rgb.png", np.zeros((2, 3, 3), dtype=np.uint8))
    original = (tmp_path / "states.json").read_bytes()
    result = view_env_state(0, state=state)
    assert not result.is_error and len(result.images) == 3
    assert result.data["state"]["old_format"] is True
    assert result.data["state"]["robot_state"]["available"] == (original_size == 61)
    assert "robot_state" not in state.get(0).state
    assert (tmp_path / "states.json").read_bytes() == original
    np.testing.assert_array_equal(state.load("proprio.npy", step=0), raw)


def test_reported_joint_units_match_the_sourced_robot_definition():
    root = Path(__file__).resolve().parents[4]
    robot = ET.parse(root / "robots/behavior/assets/r1pro/r1pro_base.urdf").getroot()
    types = {
        joint.attrib["name"]: joint.attrib["type"] for joint in robot.findall("joint")
    }
    assert [types[f"torso_joint{i}"] for i in range(1, 5)] == ["revolute"] * 4
    for arm in ("left", "right"):
        assert [types[f"{arm}_arm_joint{i}"] for i in range(1, 8)] == ["revolute"] * 7
        assert [types[f"{arm}_gripper_finger_joint{i}"] for i in range(1, 3)] == [
            "prismatic"
        ] * 2


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
    toolkit = _native_tools_toolkit(tmp_path, env, sam)
    try:
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
    finally:
        toolkit.close()


def test_native_segment_keeps_mask_when_behavior_depth_is_unavailable(tmp_path):
    raw = _geometry_observation()
    raw.pop("robot_r1::robot_r1:zed_link:Camera:0::depth_linear")
    toolkit = _native_tools_toolkit(tmp_path, _GeometryEnvironment(raw), _SAM())
    try:
        result = toolkit.execute_tool("segment", {"point": [112, 112]})
        assert result.data["found"] is True
        assert result.data["geometry_valid"] is False
        assert result.data["point_robot_base"] is None and result.data["geometry_error"]
        assert len(result.images) == 1
    finally:
        toolkit.close()


def test_unconfigured_native_sam_is_rejected_before_observation_capture(tmp_path):
    env = _GeometryEnvironment(_geometry_observation())
    with pytest.raises(ValueError, match="SAM3 client"):
        _native_tools_toolkit(tmp_path, env, None)
    assert env.reads == 0


def test_native_config_requires_only_the_selected_optional_service(tmp_path):
    spec = get_robot_spec()
    args = _args(tmp_path)
    args.tool_allowlist = ["view_env_state", "set_gripper", "release", "finish"]
    spec.validate_args(args)
    config = spec.parse_config(args)
    prompt = spec.prompts.render("system", variables=config.prompt_vars)
    assert "set_gripper" in prompt and "release" in prompt
    assert "SAM3" not in prompt and "object transport" not in prompt
    args.tool_allowlist.append("segment")
    with pytest.raises(ValueError, match="sam3_endpoint"):
        spec.validate_args(args)
    args.sam3_endpoint = "http://localhost:21002"
    spec.validate_args(args)
    prompt = spec.prompts.render(
        "system", variables=spec.parse_config(args).prompt_vars
    )
    assert "SAM3" in prompt and "coordinate_frame_step" in prompt
    args.sam3_endpoint = "ws://localhost:21002"
    with pytest.raises(ValueError, match="RPC endpoint"):
        spec.validate_args(args)


def test_native_runtime_borrows_sam_without_reset_shutdown_or_inference(
    tmp_path, monkeypatch
):
    calls, endpoints = [], []

    class Transport:
        def call(self, method, **kwargs):
            calls.append(method)
            if method == "healthz":
                return {"ok": True}
            assert method == "env.get_env_meta"
            return env_runtime_contract(40)

    def make_client(endpoint):
        endpoints.append(endpoint)
        return Transport()

    monkeypatch.setattr("rpent.utils.rpc.make_rpc_client", make_client)
    args = _args(tmp_path, "--sam3-endpoint", "http://localhost:21002")
    args.tool_allowlist = ["view_env_state", "segment", "finish"]
    daemons, runtime = get_robot_spec().init_runtime(
        args, tmp_path, NullDashboardEventSink(), None
    )
    assert daemons == [] and isinstance(runtime["sam3_client"], Sam3Client)
    assert endpoints == [args.env_endpoint, args.am_endpoint, args.sam3_endpoint]
    assert calls.count("healthz") == 3
    assert not {"env.reset", "shutdown", "sam3.segment"}.intersection(calls)


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


def test_public_proprio_decoding_keeps_xyzw_and_raw_joint_semantics():
    state = extract_robot_state(_proprio())
    assert state["frame"] == "robot_base"
    assert state["left"]["eef_position"] == [0.4, 0.2, 0.8]
    assert state["right"]["eef_position"] == [0.4, -0.2, 0.8]
    assert state["left"]["eef_quaternion_xyzw"] == [0, 0, 0, 1]
    assert state["left"]["finger_qpos"] == [0.02, 0.02]
    assert state["right"]["finger_qpos"] == [0.01, 0.01]
    assert state["trunk_qpos"] == [0.1, 0.2, 0.3, 0.4]


@pytest.mark.parametrize("values", [np.zeros(256), np.zeros(61), np.full(61, np.nan)])
def test_model_padding_or_invalid_public_pose_is_not_motion_state(values):
    with pytest.raises(ValueError):
        extract_robot_state(values)


def test_robot_reader_uses_saved_record_without_advancing_or_calling_vla(tmp_path):
    env, model = _GripperEnvironment(), _GripperModel()
    env.raw["r1::proprio"] = _proprio()
    toolkit = _template_toolkit(
        tmp_path, ["view_env_state", "read_robot_state", "finish"], env=env, model=model
    )
    try:
        env.raw["r1::proprio"][17] = 9
        result = toolkit.execute_tool("read_robot_state", {"step": 0})
        assert result.data["robot_state"]["left"]["eef_position"][0] == 0.4
        assert env.actions == [] and model.calls == 0
        assert toolkit.state.latest_step == 0
    finally:
        toolkit.close()


@pytest.fixture
def state(tmp_path):
    state = EnvState(tmp_path / "state")
    with state.record_step(state={}):
        state.save("camera.png", np.zeros((4, 4, 3), dtype=np.uint8))
    return state


def _segment(segmenter, state, **kwargs):
    return segmenter.segment(
        state=state, step=0, camera="camera", image_artifact="camera.png", **kwargs
    )


def test_no_mask_preserves_native_reason_and_point_query(state):
    reason = "no mask exceeds min_score"
    project = Mock(return_value=SegmentProjection())
    infer = Mock(return_value=Sam3Result(found=False, reason=reason))
    result = _segment(
        SavedImageSegmenter(SimpleNamespace(segment=infer)),
        state,
        point=[1, 2],
        project_mask=project,
    )
    project.assert_called_once_with(None, reason)
    artifact = state.load(result.data["segment_artifact"], step=0)
    assert artifact["error"] == result.data["error"] == reason
    assert artifact["mode"] == "point"
    assert artifact["point"] == [1, 2]
    assert "prompt" not in artifact
    assert "overlay_artifact" not in result.data
    assert result.images == []


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"prompt": "bowl", "point": [1, 2]}, "exactly one"),
        ({"point": [1]}, "point must be [row, col]"),
        ({"prompt": "bowl", "min_score": 2.0}, "min_score must be between"),
    ],
)
def test_invalid_query_does_not_call_rpc_or_save_artifacts(state, kwargs, error):
    rpc = Mock()
    result = _segment(SavedImageSegmenter(Sam3Client(rpc)), state, **kwargs)
    assert error in result.data["error"]
    rpc.call.assert_not_called()
    assert state.get(0).artifacts == {"camera.png"}


def test_missing_source_does_not_call_service_or_save_artifacts(state):
    infer = Mock()
    result = SavedImageSegmenter(SimpleNamespace(segment=infer)).segment(
        state=state, step=0, camera="camera", image_artifact="absent.png", prompt="bowl"
    )
    assert "segmentation service call failed" in result.data["error"]
    assert result.data["image_artifact"] == "absent.png"
    infer.assert_not_called()
    assert state.get(0).artifacts == {"camera.png"}


def test_invalid_mask_response_preserves_client_error_and_index(state):
    buffer = io.BytesIO()
    imageio.imwrite(buffer, np.ones((2, 2), dtype=np.uint8), format="png")
    rpc = Mock()
    rpc.call.return_value = {
        "found": True,
        "mask_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "mask_shape": [4, 4],
    }
    segmenter = SavedImageSegmenter(Sam3Client(rpc))
    result = _segment(segmenter, state, prompt="bowl")
    assert "SAM3 mask shape mismatch" in result.data["error"]
    assert state.get(0).artifacts == {"camera.png"}
    rpc.call.return_value = {"found": False, "reason": "no mask"}
    result = _segment(segmenter, state, prompt="bowl")
    assert result.data["segment_artifact"] == "segment_00.json"


def test_overlay_failure_does_not_claim_a_saved_image(state, monkeypatch):
    infer = Mock(return_value=Sam3Result(found=True, mask=np.ones((4, 4), dtype=bool)))
    save = state.save

    def fail_overlay(name, value, **kwargs):
        return None if name.endswith(".png") else save(name, value, **kwargs)

    monkeypatch.setattr(state, "save", fail_overlay)
    result = _segment(
        SavedImageSegmenter(SimpleNamespace(segment=infer)), state, prompt="bowl"
    )
    assert not result.is_error
    assert state.exists(result.data["segment_artifact"], step=0)
    assert "overlay_artifact" not in result.data
    assert result.images == []
