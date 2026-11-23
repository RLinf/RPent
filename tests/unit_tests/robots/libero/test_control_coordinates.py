# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Native calibration reads actual model geometry and resets with the scene."""

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robots.libero.control import ControlObservation


def native_scene():
    rotation = Rotation.from_euler("z", 0.8).as_matrix()
    model = SimpleNamespace(
        body_name2id=lambda name: {"robot0_base": 0, "hand": 1, "table": 2}[name],
        nbody=3,
        ngeom=3,
        body_parentid=np.array([0, 0, 0]),
        geom_bodyid=np.array([1, 1, 2]),
        geom_type=np.array([6, 6, 6]),
        geom_contype=np.ones(3),
        geom_conaffinity=np.ones(3),
        geom_size=np.array([[0.01] * 3, [0.01] * 3, [0.02, 0.4, 0.6]]),
        geom_name2id=lambda name: {"left_pad": 0, "right_pad": 1}[name],
    )
    data = SimpleNamespace(
        body_xpos=np.array([[4.0, -3.0, 0.8], [4.2, -3.0, 1.1], [4.0, -3.0, 0]]),
        body_xmat=np.array(
            [rotation.flatten(), np.eye(3).flatten(), np.eye(3).flatten()]
        ),
        site_xpos=np.array([[4.2, -3.0, 1.2]]),
        site_xmat=np.array([np.eye(3).flatten()]),
        geom_xpos=np.array([[4.24, -3.0, 1.19], [4.16, -3.0, 1.19], [4, -3, 0.88]]),
        geom_xmat=np.array(
            [
                np.eye(3).flatten(),
                np.eye(3).flatten(),
                Rotation.from_euler("y", np.pi / 2).as_matrix().flatten(),
            ]
        ),
    )
    controller = SimpleNamespace(
        name="OSC_POSE",
        input_type="delta",
        input_ref_frame="world",
        input_min=-np.ones(6),
        input_max=np.ones(6),
        output_min=-np.ones(6),
        output_max=np.ones(6),
    )
    gripper = SimpleNamespace(
        joints=[],
        important_geoms={
            "left_fingerpad": ["left_pad"],
            "right_fingerpad": ["right_pad"],
        },
    )
    robot = SimpleNamespace(
        arms=["right"],
        part_controllers={"right": controller},
        eef_site_id={"right": 0},
        robot_model=SimpleNamespace(
            root_body="robot0_base", eef_name={"right": "hand"}
        ),
        gripper={"right": gripper},
    )
    return SimpleNamespace(
        robots=[robot],
        sim=SimpleNamespace(model=model, data=data),
        model=SimpleNamespace(
            mujoco_arena=SimpleNamespace(
                table_body=ET.Element("body", name="table"),
                table_top_abs=np.array([4, -3, 999]),
            )
        ),
        reset=lambda: None,
    )


def test_native_calibration_is_fixed_per_reset_but_tcp_is_measured():
    scene = native_scene()
    wrapped = ControlObservation(scene, world_cord="general")
    first = wrapped.rpent_control_state
    np.testing.assert_allclose(first["xyz"], [4.2, -3.0, 1.2])
    assert first["coordinate_calibration"]["tcp"] == "grip_site"
    scene.sim.data.geom_xpos[:, 0] += 0.01
    scene.sim.data.site_xpos[:, 0] += 0.02
    scene.sim.data.body_xpos[0, 0] += 2
    second = wrapped.rpent_control_state
    np.testing.assert_allclose(second["xyz"], [4.22, -3.0, 1.2])
    np.testing.assert_allclose(
        second["coordinate_calibration"]["base_xyz"], [4, -3, 0.8]
    )
    wrapped.reset()
    np.testing.assert_allclose(
        wrapped.rpent_control_state["coordinate_calibration"]["base_xyz"], [6, -3, 0.8]
    )


def test_legacy_control_does_not_require_new_calibration():
    scene = native_scene()
    del scene.model.mujoco_arena
    state = ControlObservation(scene).rpent_control_state
    assert state["reference"] == "grip_site"
    assert "coordinate_calibration" not in state


@pytest.mark.parametrize(
    "field", ["table_body", "living_room_table_body", "coffee_table_body"]
)
def test_table_height_uses_compiled_collision_surface_not_nominal_offset(field):
    scene = native_scene()
    scene.model.mujoco_arena = SimpleNamespace(
        **{
            field: ET.Element("body", name="table"),
            "table_top_abs": np.array([0, 0, 999]),
        }
    )
    measured = ControlObservation(scene, world_cord="general").rpent_control_state
    assert measured["coordinate_calibration"]["table_top_xyz"][2] == pytest.approx(0.9)


def test_missing_table_geometry_is_an_error_not_a_height_guess():
    scene = native_scene()
    scene.sim.model.geom_contype[2] = scene.sim.model.geom_conaffinity[2] = 0
    with pytest.raises(ValueError, match="collision boxes"):
        ControlObservation(scene, world_cord="general").rpent_control_state


def test_floor_calibration_uses_plane_position_without_inventing_a_table():
    scene = native_scene()
    scene.model.mujoco_arena = SimpleNamespace(
        floor_body=ET.Element("body", name="table")
    )
    scene.sim.model.geom_type[2] = 0
    scene.sim.data.geom_xmat[2] = np.eye(3).flatten()
    scene.sim.model.geom_size[2] = [3, 3, 0.125]
    state = ControlObservation(scene, world_cord="general").rpent_control_state
    calibration = state["coordinate_calibration"]
    np.testing.assert_allclose(calibration["floor_top_xyz"], [4, -3, 0.88])
    assert "table_top_xyz" not in calibration
    scene.sim.data.geom_xmat[2] = Rotation.from_euler("x", np.pi).as_matrix().flatten()
    with pytest.raises(ValueError, match="upward floor plane"):
        ControlObservation(scene, world_cord="general").rpent_control_state


def test_rgb_only_environment_disables_automatic_depth_observables():
    from robots.libero.env_server import build_env_cfg

    assert build_env_cfg().init_params.camera_depths is True
    assert build_env_cfg(camera_depths=False).init_params.camera_depths is False
