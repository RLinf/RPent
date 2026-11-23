# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Model-visible frame invariance, native TCP control and geometry consumers."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robots.rl import tools
from robots.rl.toolkit import RLToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from tests.unit_tests.robots.rl.test_libero_rl_contracts import (
    GRIPPER_DOWN,
    SIZE,
    FakeEnv,
    FakeSam3,
    bar_mask,
)


class CalibratedEnv(FakeEnv):
    def __init__(
        self, origin=(0, 0, 0), yaw=0, pad_offset=(0.002, 0, -0.0036), **kwargs
    ):
        self.pad_offset = np.asarray(pad_offset)
        self.origin = np.asarray(origin, dtype=float)
        self.base_rotation = Rotation.from_euler("z", yaw).as_matrix()
        self.depth_calls = 0
        self._client = SimpleNamespace(call=lambda method: {"max_episode_steps": 10000})
        super().__init__(**kwargs)
        self.xyz = self.origin + self.base_rotation @ np.array([0.1, -0.2, 1.0])
        self.rotation = Rotation.from_matrix(self.base_rotation @ GRIPPER_DOWN)
        self._observe()

    def _observe(self):
        super()._observe()
        self.last_obs["control_state"].update(
            coordinate_calibration={
                "base_xyz": self.origin.copy(),
                "base_rotation": self.base_rotation.copy(),
                "table_top_xyz": self.origin + np.array([0, 0, 0.12]),
            },
            pad_tcp_xyz=self.xyz + self.rotation.as_matrix() @ self.pad_offset,
            jaw_slide_axes_site=[[1, 0, 0], [-1, 0, 0]],
            wrist_to_site_in_site=[0, 0, 0.097],
        )

    def render_camera(self, camera_name, height, width, depth=False):
        rgb, buffer = super().render_camera(camera_name, height, width, depth)
        self.depth_calls += int(depth)
        return (rgb, buffer) if depth else rgb

    def get_camera_meta(self, camera_name, height, width):
        result = super().get_camera_meta(camera_name, height, width)
        transform = np.eye(4)
        transform[:3, :3] = self.base_rotation
        transform[:3, 3] = self.origin
        result["extrinsic_cam2world"] = transform.tolist()
        return result


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(tools, "IMAGE_SIZE", SIZE)
    monkeypatch.setattr("rpent.tools.toolkit.substitute", lambda value: value)
    monkeypatch.setattr(
        "rpent.utils.templates.default_variables", lambda: {"output_dir": "/offline"}
    )


def make(tmp_path, *, env=None, workflow="episode", world_cord="general", **options):
    env = env or CalibratedEnv()
    toolkit = RLToolkit(
        runtime_kwargs={"env": env, "sam3_client": FakeSam3(bar_mask())},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        observation_mode="adaptive",
        output_dir=tmp_path / "run",
        workflow=workflow,
        world_cord=world_cord,
        **options,
    )
    return toolkit, env


@pytest.mark.parametrize("mode", ["base", "general"])
def test_floor_height_is_explicit_and_not_reported_as_a_table(mode):
    from robots.rl.coordinates import Coordinates

    env = CalibratedEnv([4, -3, 2])
    control = env.last_obs["control_state"]
    calibration = control["coordinate_calibration"]
    del calibration["table_top_xyz"]
    calibration["floor_top_xyz"] = [4, -3, 1.95]
    frame = Coordinates.from_control(mode, control)
    for metadata in (frame.metadata(), frame.description()):
        assert metadata["support_surface"] == "floor"
        assert metadata["floor_height_above_base_m"] == pytest.approx(-0.05)
        assert metadata["table_height_above_base_m"] is None


def test_episode_returns_frame_invariant_robot_and_geometry(tmp_path):
    first, _ = make(tmp_path / "a")
    second, _ = make(tmp_path / "b", env=CalibratedEnv([4, -3, 2], 0.73))
    for toolkit in (first, second):
        assert toolkit._primitives.coordinates.table_height_m == pytest.approx(0.12)
    for name in ("xyz", "orientation_rpy"):
        np.testing.assert_allclose(
            first._robot_state()["pose"][name], second._robot_state()["pose"][name]
        )
    for camera in ("agentview", "wrist"):
        a, b = [
            tk.execute_tool("segment", {"prompt": "cube", "camera": camera, "step": 0})
            for tk in (first, second)
        ]
        for key in ("world_xyz", "top_z_m", "size_xyz_m", "axis_yaw"):
            assert a.data[key] == pytest.approx(b.data[key], abs=1e-4)
        a, b = [
            tk.execute_tool(
                "back_project", {"pixel": [10, 12], "camera": camera, "step": 0}
            )
            for tk in (first, second)
        ]
        assert a.data["world_xyz"] == pytest.approx(b.data["world_xyz"], abs=1e-4)


def test_regions_filter_general_z_and_recompute_rotated_extents(tmp_path):
    toolkit, _ = make(tmp_path, env=CalibratedEnv([10, -4, 2], 0.8))
    found = toolkit.execute_tool("segment", {"prompt": "cube", "step": 0})
    assert found.data["top_z_m"] == pytest.approx(1.0, abs=1e-4)
    assert found.data["size_xyz_m"][0] > 4 * found.data["size_xyz_m"][1]
    assert found.data["axis_yaw"] == pytest.approx(0, abs=1e-4)
    for region in (
        {"box": [8, 4, 12, 28], "step": 0},
        {"mask_id": found.data["mask_id"]},
    ):
        accepted = toolkit.execute_tool(
            "back_project", {**region, "z_range_m": [0.99, 1.01]}
        )
        rejected = toolkit.execute_tool(
            "back_project", {**region, "z_range_m": [2.99, 3.01]}
        )
        assert accepted.data["world_xyz"][2] == pytest.approx(1, abs=1e-4)
        assert rejected.is_error
    raw = toolkit.state.load("agentview_world.npz", step=0)
    assert np.median(raw[..., 2]) == pytest.approx(3.0)
    assert (
        toolkit.state.load("coordinates.json", step=None)["raw_point_map_frame"]
        == "sim_world"
    )


def test_absolute_tcp_targets_and_omitted_position_survive_rotation(tmp_path):
    toolkit, env = make(
        tmp_path,
        env=CalibratedEnv([1.5, -2, 0.8], 0.6, pad_offset=[0.018, 0.005, -0.025]),
    )
    from tests.unit_tests.robots.rl.test_libero_rl_contracts import execute_item

    result = execute_item(
        toolkit,
        "move_to",
        {
            "xyz": [0.15, -0.18, 1.01],
            "orientation_rpy": [2.9, 0.1, 0.3],
        },
    )
    assert result.data.get("status") == "reached", result.to_text()
    expected = env.origin + env.base_rotation @ np.array([0.15, -0.18, 1.01])
    np.testing.assert_allclose(env.xyz, expected, atol=0.004)
    np.testing.assert_allclose(
        env.rotation.as_matrix(),
        env.base_rotation @ Rotation.from_euler("xyz", [2.9, 0.1, 0.3]).as_matrix(),
        atol=0.02,
    )
    before = toolkit._robot_state()["pose"]["xyz"]
    result = execute_item(toolkit, "move_to", {"orientation_rpy": [2.7, -0.1, -0.2]})
    assert result.data.get("status") == "reached", result.to_text()
    np.testing.assert_allclose(
        toolkit._robot_state()["pose"]["xyz"], before, atol=0.004
    )
    closed = execute_item(toolkit, "set_gripper", {"opening": 0})
    assert closed.data["opening"] <= 0.08
    np.testing.assert_allclose(
        toolkit._robot_state()["pose"]["xyz"], before, atol=0.004
    )


def test_general_requires_calibration_but_base_keeps_original_control_point(tmp_path):
    with pytest.raises(ValueError, match="calibration"):
        make(tmp_path / "missing", env=FakeEnv())
    toolkit, env = make(tmp_path / "base", world_cord="base")
    assert toolkit._robot_state()["pose"]["xyz"] == env.xyz.tolist()
    assert not np.allclose(env.xyz, env.last_obs["control_state"]["pad_tcp_xyz"])
    frame = toolkit._primitives.coordinates
    native = np.array([[0, 0, 0], [np.nan, 0, 1], [1, 2, 3]])
    np.testing.assert_allclose(toolkit._primitives.surface_points(native), [[1, 2, 3]])
    assert frame.mode == "base"


def test_removed_prompt_variants_rejected():
    from robots.rl.prompt_bundle import system_prompt

    for variant in ("base", "base_gripper", "base_gripper_camera"):
        with pytest.raises(ValueError, match="fixed prompt"):
            system_prompt({"prompt": variant})


def test_model_frame_description_hides_simulator_origin_but_audit_keeps_it(tmp_path):
    a, _ = make(tmp_path / "a", workflow="episode")
    b, _ = make(
        tmp_path / "b", workflow="episode", env=CalibratedEnv([40, -30, 20], 0.7)
    )
    first, second = a.initial_observation(), b.initial_observation()
    assert first.data["coordinates"] == second.data["coordinates"]
    assert "origin_in_sim_world" not in str(first.data)
    assert "rotation_to_sim_world" not in str(second.data)
    assert (
        a.state.load("coordinates.json", step=None)["origin_in_sim_world"]
        != b.state.load("coordinates.json", step=None)["origin_in_sim_world"]
    )
