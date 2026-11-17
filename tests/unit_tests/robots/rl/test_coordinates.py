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


@pytest.mark.parametrize(
    "workflow", ["episode", "episode_image", "zap_episode", "zap_turn", "zap_subtask"]
)
def test_all_workflows_return_frame_invariant_robot_and_geometry(tmp_path, workflow):
    first, _ = make(tmp_path / "a", workflow=workflow)
    second, _ = make(
        tmp_path / "b", workflow=workflow, env=CalibratedEnv([4, -3, 2], 0.73)
    )
    for toolkit in (first, second):
        assert toolkit._primitives.coordinates.table_height_m == pytest.approx(0.12)
    np.testing.assert_allclose(
        first._robot_state()["pose"]["xyz"], second._robot_state()["pose"]["xyz"]
    )
    np.testing.assert_allclose(
        first._robot_state()["pose"]["orientation_rpy"],
        second._robot_state()["pose"]["orientation_rpy"],
    )
    if workflow.startswith("zap"):
        for toolkit in (first, second):
            toolkit._primitives._sam3.segment_queries = lambda raw, queries: [
                [toolkit._primitives._sam3.segment(raw)]
            ]
            toolkit.snapshot.queries = ["cube"]
        a, b = first.observe_snapshot(), second.observe_snapshot()
        for camera in range(2):
            for key in ("world_xyz", "top_z_m", "size_xyz_m", "axis_yaw"):
                assert a.data["perception"][camera]["objects"][0]["surface"][
                    key
                ] == pytest.approx(
                    b.data["perception"][camera]["objects"][0]["surface"][key], abs=1e-4
                )
    else:
        for camera in ("agentview", "wrist"):
            kwargs = {"prompt": "cube", "camera": camera}
            if workflow == "episode":
                kwargs["step"] = 0
            a, b = (
                first.execute_tool("segment", kwargs),
                second.execute_tool("segment", kwargs),
            )
            for key in ("world_xyz", "top_z_m", "size_xyz_m", "axis_yaw"):
                assert a.data[key] == pytest.approx(b.data[key], abs=1e-4)
            kwargs = {"pixel": [10, 12], "camera": camera}
            if workflow == "episode":
                kwargs["step"] = 0
            assert first.execute_tool("back_project", kwargs).data[
                "world_xyz"
            ] == pytest.approx(
                second.execute_tool("back_project", kwargs).data["world_xyz"], abs=1e-4
            )


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


def test_batched_observe_uses_same_frame_for_image_mask_and_dependants(tmp_path):
    toolkit, _ = make(
        tmp_path, env=CalibratedEnv([5, 2, 3], -0.4), perception_mode="batch"
    )
    toolkit.initial_observation()
    result = toolkit.execute_tool(
        "observe",
        {
            "calls": [
                {"id": "v", "op": "view_env", "camera": "agentview"},
                {
                    "id": "s",
                    "op": "segment",
                    "source": "@v.agentview",
                    "prompt": "cube",
                },
                {
                    "id": "p",
                    "op": "back_project",
                    "source": "@s",
                    "z_range_m": [0.99, 1.01],
                },
            ]
        },
    )
    assert not result.is_error
    rows = {row["id"]: row for row in result.data["results"]}
    assert rows["s"]["result"]["world_xyz"] == rows["p"]["result"]["world_xyz"]


def test_absolute_tcp_targets_and_omitted_position_survive_rotation(tmp_path):
    toolkit, env = make(
        tmp_path,
        env=CalibratedEnv([1.5, -2, 0.8], 0.6, pad_offset=[0.018, 0.005, -0.025]),
    )
    result = toolkit.execute_tool(
        "move_to", {"xyz": [0.15, -0.18, 1.01], "orientation_rpy": [2.9, 0.1, 0.3]}
    )
    assert result.data["status"] == "reached"
    expected = env.origin + env.base_rotation @ np.array([0.15, -0.18, 1.01])
    np.testing.assert_allclose(
        env.last_obs["control_state"]["pad_tcp_xyz"], expected, atol=0.004
    )
    np.testing.assert_allclose(
        env.rotation.as_matrix(),
        env.base_rotation @ Rotation.from_euler("xyz", [2.9, 0.1, 0.3]).as_matrix(),
        atol=0.02,
    )
    before = toolkit._robot_state()["pose"]["xyz"]
    result = toolkit.execute_tool("move_to", {"orientation_rpy": [2.7, -0.1, -0.2]})
    assert result.data["status"] == "reached"
    np.testing.assert_allclose(
        toolkit._robot_state()["pose"]["xyz"], before, atol=0.004
    )
    closed = toolkit.execute_tool("set_gripper", {"opening": 0})
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


def test_optional_gripper_prompt_describes_the_selected_control_point():
    from robots.rl.prompt_bundle import system_prompt

    options = {"prompt": "base_gripper_camera", "workflow": "episode"}
    base = system_prompt({**options, "world_cord": "base"})
    general = system_prompt({**options, "world_cord": "general"})
    assert "9.7 cm beyond the wrist flange" in base
    assert "1.2 cm above and 0.4 cm below" in base
    assert "midpoint of the two finger-pad geometry centres" in general
    assert "9.7 cm" not in general
    assert "1.2 cm above and 0.4 cm below" not in general


def test_model_frame_description_hides_simulator_origin_but_audit_keeps_it(tmp_path):
    a, _ = make(tmp_path / "a", workflow="pear")
    b, _ = make(tmp_path / "b", workflow="pear", env=CalibratedEnv([40, -30, 20], 0.7))
    first, second = a.initial_observation(), b.initial_observation()
    assert first.data["coordinates"] == second.data["coordinates"]
    assert "origin_in_sim_world" not in str(first.data)
    assert "rotation_to_sim_world" not in str(second.data)
    assert (
        a.state.load("coordinates.json", step=None)["origin_in_sim_world"]
        != b.state.load("coordinates.json", step=None)["origin_in_sim_world"]
    )
