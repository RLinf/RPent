# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""The RoboTwin episode prompt and tools share one model-facing frame."""

from types import SimpleNamespace

import numpy as np

from robots.rl.prompt_bundle import system_prompt
from robots.rl.remote import remote_options
from robots.rl.toolkit import RLToolkit
from robots.robotwin.general import mounting_calibration
from rpent.utils import logging as rpent_logging
from tests.unit_tests.robots.rl.test_backend_contracts import Twin

OPTIONS = {
    "environment": "robotwin",
}


def test_robotwin_episode_contract_and_prompt():
    options = remote_options(OPTIONS)
    assert options["max_actions_per_call"] == 3
    prompt = system_prompt(OPTIONS)
    for phrase in (
        "execute_actions",
        "orientation_rpy",
        "metres",
        "left_wrist",
        "right_wrist",
        "back_project",
        "partial",
    ):
        assert phrase in prompt
    assert "quaternions" not in prompt


def test_robotwin_episode_tools_report_and_execute_in_the_same_frame(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(rpent_logging, "_output_dir", tmp_path)
    env = Twin()
    offset = [0.02, 0.0, 0.0]
    calibrate = {
        "coordinate_calibration": mounting_calibration(
            [-0.3, -0.65, 0.78], [0.3, -0.65, 0.78], [0, 0, 0.74]
        ),
        "tool_rotation_to_native": {
            arm: np.eye(3).tolist() for arm in ("left", "right")
        },
    }
    native_call = env._client.call
    pad_deflection = {"left": np.zeros(3), "right": np.zeros(3)}

    def call(name, kwargs=None):
        if name == "env.coordinate_calibration":
            return calibrate
        grippers = native_call(name)
        if kwargs and kwargs.get("with_pad"):
            for arm, gripper in grippers.items():
                native = env.last_info["robot_state"][f"{arm}_tcp_pose"]
                gripper["pad_tcp_xyz"] = (
                    np.asarray(native[:3]) + offset + pad_deflection[arm]
                ).tolist()
        return grippers

    env._client.call = call
    env.get_task_language = lambda: "Move both arms forward."
    env.get_camera_meta = lambda _: {
        "extrinsic_cv": np.eye(4),
        "intrinsic_K": np.eye(3),
    }
    env.render_camera = lambda camera, depth: (
        np.zeros((4, 6, 3), dtype=np.uint8),
        np.ones((4, 6)),
    )
    toolkit = RLToolkit(
        runtime_kwargs={"env": env, "sam3_client": None},
        dashboard_events=SimpleNamespace(emit=lambda _event: None),
        memory=None,
        output_dir=tmp_path,
        **OPTIONS,
    )
    initial = toolkit.initial_observation().data
    left = initial["state"]["left"]["pose"]
    np.testing.assert_allclose(left["xyz"], [65, 0, -38])
    assert initial["coordinates"]["tcp"] == "native_grasp_tcp"
    assert {tool.name for tool in toolkit.list_tools()} == {
        "view_env",
        "segment",
        "back_project",
        "execute_actions",
    }
    images = toolkit.execute_tool("view_env", {"camera": "all"})
    assert len(images.images) == 3
    point = toolkit.execute_tool(
        "back_project", {"pixel": [0, 0], "step": 0, "camera": "head"}
    )
    np.testing.assert_allclose(point.data["world_xyz"], [65, 0, 22])
    invalid_height = toolkit.execute_tool(
        "back_project",
        {"box": [0, 0, 4, 6], "camera": "head", "step": 0, "z_range_cm": [0]},
    )
    assert invalid_height.is_error

    move = {"action": "move_to", "arm": "left", "xyz": [67, 0, -38]}
    move_right = {"action": "move_to", "arm": "right", "xyz": [68, 0, -38]}
    grip = {"action": "set_gripper", "arm": "right", "opening": 1}
    for invalid in (
        {"actions": [move, {"action": "set_gripper", "opening": 0}]},
        {"actions": [{**move, "other_xyz": move_right["xyz"]}]},
        {"actions": [move] * 4},
    ):
        assert toolkit.execute_tool("execute_actions", invalid).is_error
        assert not env.plans
    result = toolkit.execute_tool(
        "execute_actions", {"actions": [move, move_right, grip]}
    )
    assert result.data["executed_count"] == 3
    assert result.data["results"][0]["status"] == "reached"
    assert [arm for arm, _ in env.plans] == ["left", "right"]
    np.testing.assert_allclose(env.plans[0][1][:3], [0, 0.02, 0.3], atol=1e-9)
    np.testing.assert_allclose(env.plans[1][1][:3], [0, 0.03, 0.3], atol=1e-9)
    np.testing.assert_allclose(
        result.data["final_state"]["right"]["pose"]["xyz"], move_right["xyz"]
    )
    assert not result.images
    env.failed_arm = "right"
    stopped = toolkit.execute_tool("execute_actions", {"actions": [move_right, grip]})
    assert stopped.data["executed_count"] == 1
    assert stopped.data["stop_reason"] == "plan_failed"
    np.testing.assert_allclose(
        stopped.data["final_state"]["right"]["pose"]["xyz"], move_right["xyz"]
    )
    env.failed_arm = None
    pad_deflection["right"][1] = 0
    native_state = env.last_info["robot_state"]
    for arm in ("left", "right"):
        tcp = native_state[f"{arm}_tcp_pose"]
        native_state[f"{arm}_eef_pose"] = [
            *(np.asarray(tcp[:3]) - [0, 0, 0.1]),
            *tcp[3:],
        ]
    native_step = env.chunk_step

    def step_and_displace_tcp(actions, *, action_type):
        result = native_step(actions, action_type=action_type)
        env.last_info["robot_state"]["right_tcp_pose"][1] += 0.012
        return result

    env.chunk_step = step_and_displace_tcp
    stopped = toolkit.execute_tool("execute_actions", {"actions": [move_right, grip]})
    assert stopped.data["executed_count"] == 1
    assert stopped.data["stop_reason"] == "not_at_target"


def test_finger_pad_reference_uses_opposed_faces():
    from robots.robotwin.general import finger_contact_centres

    far_jaw = np.array(
        [
            [-0.02, -0.05, -0.03],
            [-0.02, -0.05, 0.03],
            [0.055, -0.02, -0.01],
            [0.055, -0.02, 0.01],
            [0.075, -0.02, -0.01],
            [0.075, -0.02, 0.01],
        ]
    )
    near_jaw = far_jaw * [1, -1, 1]
    np.testing.assert_allclose(
        finger_contact_centres(near_jaw, far_jaw),
        [[0.065, 0.02, 0], [0.065, -0.02, 0]],
    )


def test_collision_calibration_applies_mesh_scale_before_local_pose():
    from robots.robotwin.general import _collision_vertices

    local = np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]])
    shape = SimpleNamespace(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        scale=np.array([2, 3, 4]),
        get_local_pose=lambda: SimpleNamespace(to_transformation_matrix=lambda: local),
    )
    link = SimpleNamespace(get_collision_shapes=lambda: [shape])
    np.testing.assert_allclose(
        _collision_vertices(link),
        [[10, 20, 30], [10, 22, 30], [7, 20, 30], [10, 20, 34]],
    )
