# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Physical equivalence across displayed units and stale-frame rejection."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robots.rl import tools
from robots.rl.coordinates import Coordinates, grasp_axes
from robots.rl.prompt_bundle import system_prompt
from robots.rl.toolkit import RLToolkit
from robots.rl.units import PositionUnits
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from tests.unit_tests.robots.rl.test_libero_rl_contracts import (
    SIZE,
    FakeEnv,
    FakeSam3,
    bar_mask,
)


class CalibratedEnv(FakeEnv):
    def _observe(self):
        super()._observe()
        self.last_obs["control_state"].update(
            coordinate_calibration={
                "base_xyz": [0, 0, 0.1],
                "base_rotation": np.eye(3),
                "table_top_xyz": [0, 0, 0.3],
                "tcp": "grip_site",
            },
            jaw_slide_axes_site=[[1, 0, 0], [-1, 0, 0]],
            wrist_to_site_in_site=[0, 0, 0.097],
        )


@pytest.fixture
def factory(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "IMAGE_SIZE", SIZE)
    monkeypatch.setattr("rpent.tools.toolkit.substitute", lambda value: value)
    monkeypatch.setattr(
        "rpent.utils.templates.default_variables", lambda: {"output_dir": str(tmp_path)}
    )

    def make(unit):
        env = CalibratedEnv()
        tk = RLToolkit(
            runtime_kwargs={"env": env, "sam3_client": FakeSam3(bar_mask())},
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / unit / "memory"),
            observation_mode="adaptive",
            output_dir=tmp_path / unit,
            world_cord="general",
            position_unit=unit,
            action_mode="batch",
        )
        return tk, env

    return make


def test_metres_centimetres_execute_identical_physical_sequence(factory):
    metre, a = factory("m")
    centimetre, b = factory("cm")
    for tk, scale in [(metre, 1), (centimetre, 100)]:
        result = tk.execute_tool(
            "execute_actions",
            {
                "actions": [
                    {
                        "action": "move_to",
                        "xyz": [v * scale for v in [0.12, -0.21, 0.88]],
                        "orientation_rpy": [3.1, 0.05, 0.2],
                    },
                    {"action": "set_gripper", "opening": 0.5},
                ],
            },
        )
        assert result.data["executed_count"] == 2, result.to_text()
        assert result.data["final_state"]["pose"]["xyz"][0] == pytest.approx(
            a.xyz[0] * scale, abs=0.001
        )
        assert result.data["coordinates"]["frame_version"] == 0
    np.testing.assert_allclose(a.actions, b.actions, atol=1e-10)
    np.testing.assert_allclose(a.xyz, b.xyz, atol=1e-10)
    np.testing.assert_allclose(
        a.rotation.as_matrix(), b.rotation.as_matrix(), atol=1e-10
    )


def test_action_frame_version_is_receipt_only(factory):
    toolkit, env = factory("cm")
    action = {"actions": [{"action": "move_to", "xyz": [10, -20, 80]}]}
    assert toolkit.execute_tool(
        "execute_actions", {**action, "frame_version": 0}
    ).is_error
    assert env.actions == []
    result = toolkit.execute_tool("execute_actions", action)
    assert result.data["executed_count"] == 1
    assert result.data["results"][0]["start_frame_version"] == 0
    assert result.data["results"][0]["frame_version"] == 0


def test_units_convert_lengths_not_angles_pixels_time_or_opening():
    original = {
        "xyz": [0.123456, 0, -0.4],
        "top_z_m": 0.2,
        "size_xyz_m": [0.01, 0.02, 0.03],
        "position_error_m": 0.004,
        "z_range_m": [-0.1, 0.5],
        "vx": 0.1,
        "opening": 0.5,
        "duration_s": 2,
        "axis_yaw": 0.3,
        "orientation_rpy": [1, 2, 3],
        "pixel": [20, 30],
        "box": [1, 2, 3, 4],
    }
    units = PositionUnits("cm")
    external = units.payload(original)
    assert external["xyz"] == pytest.approx([12.3456, 0, -40])
    assert external["top_z_cm"] == 20
    assert external["position_error_cm"] == 0.4
    for name in (
        "pixel",
        "box",
        "axis_yaw",
        "orientation_rpy",
        "opening",
        "duration_s",
    ):
        assert external[name] == original[name]
    for name, value in units.payload(external, to_model=False).items():
        assert value == pytest.approx(original[name])
    assert "ccm" not in units.description("Speed 0.1 m/s, maximum 2 metres.")


def test_native_tcp_metadata_distinguishes_wrist_from_grip_site():
    for control, expected in (
        ({}, "grip_site"),
        ({"native_tcp": "native_wrist_ik"}, "native_wrist_ik"),
    ):
        frame = Coordinates.from_control("base", control)
        assert frame.metadata()["tcp"] == expected
        assert frame.description()["tcp"] == expected
        np.testing.assert_array_equal(
            frame.world_position([0.1, 0.2, 0.3]), [0.1, 0.2, 0.3]
        )


def test_common_tool_axes_roundtrip_under_rotated_base():
    c = grasp_axes([0, 1, 0], [1, 0, 0])
    native = Rotation.from_euler("xyz", [0.4, -0.5, 1.2]).as_matrix()
    base = Coordinates(
        "general", np.array([3, -1, 0.7]), Rotation.from_euler("z", 0.9).as_matrix()
    )
    visible = base.orientation(native @ c)
    np.testing.assert_allclose(
        base.world_orientation(visible) @ c.T, native, atol=1e-12
    )
    np.testing.assert_allclose(c[:, 0], [0, 1, 0])
    np.testing.assert_allclose(c[:, 2], [1, 0, 0])


def test_frame_version_changes_only_for_mount_motion():
    base = Coordinates("general", np.array([0.0, 0.0, 0.0]), np.eye(3))
    assert base.after(base).version == 0
    lifted = Coordinates("general", np.array([0.0, 0.0, 0.05]), np.eye(3)).after(base)
    assert lifted.version == 1
    assert lifted.after(lifted).version == 1
    turned = Coordinates(
        "general", lifted.origin, Rotation.from_euler("z", 0.2).as_matrix()
    ).after(lifted)
    assert turned.version == 2


def test_current_libero_base_prompt_uses_batched_actions_and_metre_poses():
    prompt = system_prompt(
        {"environment": "libero", "world_cord": "base", "position_unit": "m"}
    )
    assert "grip_site" in prompt and "world metres" in prompt
    assert "execute_actions" in prompt
    assert "finish ends" not in prompt and "frame_version" not in prompt


@pytest.mark.parametrize("backend", ["libero", "robocasa", "robotwin", "robodojo"])
def test_every_backend_transforms_points_before_median_extent_and_z_filter(backend):
    from robots.rl import geometry
    from robots.rl.toolkit import primitive_owner

    obj = object.__new__(primitive_owner(backend, "general"))
    yaw = Rotation.from_euler("z", 0.73).as_matrix()
    origin = np.array([4.0, -3.0, 2.0])
    obj.coordinates = Coordinates("general", origin, yaw)
    # Asymmetric sampling exposes transform(median(world)) != median(transform(world)).
    expected = np.array(
        [
            [x, y, z]
            for x, y, z in [
                (0.1, 0.8, 0.2),
                (0.2, 0.2, 0.3),
                (0.8, 0.1, 0.4),
                (0.9, 0.15, 0.2),
                (0.3, 0.3, 0.7),
                (0.2, 0.9, 0.3),
                (0.7, 0.2, 0.4),
                (0.15, 0.2, 0.3),
                (0.1, 0.7, 0.2),
            ]
        ]
    )
    expected = np.repeat(expected, 10, axis=0)
    world = expected @ yaw.T + origin
    world = np.vstack([world, [np.nan, np.nan, np.nan], [np.nan, 1, 2]])
    visible, centre = obj._surface_sample(world, [0.19, 0.41])
    keep = expected[expected[:, 2] < 0.5]
    np.testing.assert_allclose(visible, keep, atol=1e-12)
    np.testing.assert_allclose(centre, np.median(keep, axis=0), atol=1e-12)
    a, b = geometry.surface_geometry(visible), geometry.surface_geometry(keep)
    for key in ("world_xyz", "top_z_m", "size_xyz_m", "axis_yaw"):
        assert a[key] == pytest.approx(b[key], abs=1e-8)


def test_general_world_origin_is_a_valid_depth_point():
    from robots.rl.geometry import valid_points

    points = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    np.testing.assert_array_equal(
        valid_points(points, zero_is_invalid=False), [[0.0, 0.0, 0.0]]
    )
    assert len(valid_points(points)) == 0  # frozen base-mode zero sentinel
