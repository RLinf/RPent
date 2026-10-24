# Copyright 2026 The RPent Authors.
"""CPU contracts for native motion composition and saved-frame perception."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robodojo import agent_support, robot_spec, tools
from robots.robodojo.primitives import RoboDojoPrimitives
from robots.robodojo.toolkit import RoboDojoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.tools.toolkit import ToolCancelled


class MotionEnv:
    def __init__(self):
        self.step_count = 0
        self.step_limit = 100
        self.success = False
        self.updates = []
        self.plans = []
        self.resets = 0
        self.reads = 0
        self.depth = 2.0
        self.pose = [0.1, 0.2, 0.8, 1.0, 0.0, 0.0, 0.0]
        self.opening = {"left": 1.0, "right": 0.3}
        self.path = np.arange(60, dtype=float).reshape(10, 6) / 100
        self.plan_status = "Success"

    def get_status(self):
        return {
            "step": self.step_count,
            "step_limit": self.step_limit,
            "success": self.success,
            "safety": {"hidden_object_pose": [9, 9, 9]},
            "score": 100,
        }

    def get_motion_state(self):
        return {
            **{f"{arm}_eef_pose": self.pose.copy() for arm in self.opening},
            **{f"{arm}_gripper": value for arm, value in self.opening.items()},
        }

    def plan_arm_path(self, arm, target_pose):
        self.plans.append((arm, target_pose))
        return {"status": self.plan_status, "position": self.path.tolist()}

    def step_joint_update(self, arm, arm_qpos=None, gripper=None):
        self.updates.append({"arm": arm, "arm_qpos": arm_qpos, "gripper": gripper})
        self.step_count += 1
        if gripper is not None:
            self.opening[arm] = gripper
        return {"executed_actions": 1, "episode_status": self.get_status()}

    def get_obs(self):
        self.reads += 1
        return {
            "instruction": "Move the object.",
            "state": {},
            "vision": {
                "cam_head": {
                    "color": np.full((3, 3, 3), self.reads, np.uint8),
                    "depth": np.full((3, 3), self.depth),
                    "intrinsic_matrix": [[2, 0, 1], [0, 2, 1], [0, 0, 1]],
                    "extrinsic_matrix": np.eye(4),
                }
            },
        }

    def get_task_language(self):
        return "Move the object."

    def is_success(self):
        return self.success

    def reset(self):
        self.resets += 1
        self.success = False
        self.step_count = 0


@pytest.fixture
def motion():
    env = MotionEnv()
    return env, RoboDojoPrimitives(env=env, check_cancelled=lambda: None)


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_gripper_endpoints_interpolation_and_other_target(motion, arm, value):
    env, primitive = motion
    before = env.opening.copy()
    result = primitive.set_gripper(arm=arm, val=value).data
    np.testing.assert_allclose(
        [u["gripper"] for u in env.updates], np.linspace(before[arm], value, 11)[1:]
    )
    assert all(u["arm_qpos"] is None for u in env.updates)
    other = "right" if arm == "left" else "left"
    assert env.opening[other] == before[other]
    assert result["completed"] and result["executed_steps"] == 10
    assert result["gripper_val"] == value
    assert (
        "safety" not in result["episode_status"]
        and "score" not in result["episode_status"]
    )


@pytest.mark.parametrize("value", [-1.0, 1.1, np.nan, np.inf])
def test_invalid_gripper_never_moves(motion, value):
    env, primitive = motion
    with pytest.raises(ValueError):
        primitive.set_gripper(arm="left", val=value)
    with pytest.raises(ValueError):
        primitive.release(arm="left", val=value)
    assert not env.updates


def test_release_defaults_open_and_budget_is_partial(motion):
    env, primitive = motion
    env.opening["left"] = 0
    env.step_limit = 4
    result = primitive.release(arm="left").data
    assert not result["completed"]
    assert result["executed_steps"] == 4 and result["requested_steps"] == 10
    assert result["stop_reason"] == "budget_exhausted"
    assert env.opening["left"] == pytest.approx(0.4)


def test_cancellation_retains_acknowledged_progress(motion):
    env, primitive = motion

    def cancel():
        if len(env.updates) == 4:
            raise ToolCancelled("stop")

    primitive._check_cancelled = cancel
    result = primitive.set_gripper(arm="left", val=0).data
    assert result["executed_steps"] == 4
    assert not result["completed"] and result["interrupted"]


def test_ambiguous_mutation_timeout_is_not_retried(motion):
    env, primitive = motion

    def fail(**kwargs):
        env.updates.append(kwargs)
        raise TimeoutError("reply lost")

    env.step_joint_update = fail
    result = primitive.release(arm="right").data
    assert len(env.updates) == 1
    assert result["execution_uncertain"] and not result["completed"]
    assert result["executed_steps"] == 0 and result["error"] == "reply lost"


def test_native_runtime_stop_is_not_reported_as_success(motion):
    env, primitive = motion
    env.step_joint_update = lambda **kwargs: {
        "executed_actions": 0,
        "episode_status": env.get_status(),
    }
    result = primitive.release(arm="left").data
    assert not result["success"] and not result["completed"]
    assert result["stop_reason"] == "runtime_failure"
    assert result["executed_steps"] == 0
    assert "execution_uncertain" not in result


@pytest.mark.parametrize(
    "substeps,indices",
    [(0, list(range(10))), (1, [9]), (3, [0, 4, 9]), (25, list(range(10)))],
)
def test_move_path_sampling_and_measured_feedback(motion, substeps, indices):
    env, primitive = motion
    result = primitive.move_to(arm="left", xyz=[0.1, 0.2, 0.9], substeps=substeps).data
    np.testing.assert_allclose([u["arm_qpos"] for u in env.updates], env.path[indices])
    assert env.plans[0][1][3:] == env.pose[3:]
    assert all(u["gripper"] is None for u in env.updates)
    assert result["completed"] and result["final_dist_m"] == pytest.approx(0.1)
    assert "reached" not in result


def test_planning_failure_does_not_execute_or_change_orientation(motion):
    env, primitive = motion
    env.plan_status = "Fail"
    quat = [0.0, 1.0, 0.0, 0.0]
    result = primitive.move_to(arm="right", xyz=[0, 0, 1], quat=quat).data
    assert len(env.plans) == 1 and env.plans[0][1][3:] == quat
    assert not env.updates and result["stop_reason"] == "plan_failed"


def test_rotation_is_world_z_not_local_axis(motion):
    env, primitive = motion
    env.pose[3:] = [np.sqrt(0.5), np.sqrt(0.5), 0, 0]
    primitive.rotate_wrist(arm="left", delta_yaw_deg=90)
    np.testing.assert_allclose(env.plans[0][1][3:], [0.5, 0.5, 0.5, 0.5])
    assert env.plans[0][1][:3] == env.pose[:3]


def make_native():
    robots = [
        SimpleNamespace(
            type="target",
            arm_name=f"{arm}_arm",
            gripper_name=f"{arm}_ee",
            robot_name=arm,
            arm_joint_indices=list(range(6)),
            entity_origin_pose=[0, 0, 0, 1, 0, 0, 0],
            gripper_scale=[-0.01, 0.044],
            gripper_move={"sign": 1},
        )
        for arm in ("left", "right")
    ]
    previous = {
        f"{arm}_arm_joint_state": {"position": [offset] * 6}
        for arm, offset in (("left", 0.4), ("right", 0.7))
    }
    previous.update(
        {
            f"{arm}_ee_joint_state": {"position": [val, val]}
            for arm, val in (("left", -0.01), ("right", 0.017))
        }
    )
    plans = []

    def plan(**kwargs):
        plans.append(kwargs)
        return {"status": "Success", "position": np.zeros((3, 6))}

    manager = SimpleNamespace(
        robot_list=robots,
        robot_key=[
            SimpleNamespace(
                data=SimpleNamespace(joint_pos_limits=np.tile([-3.14, 3.14], (1, 6, 1)))
            )
            for _ in robots
        ],
        control_manager=SimpleNamespace(prev_control=[previous]),
        scene=SimpleNamespace(env_origins=np.array([[1, 2, 3]])),
        process_name=lambda name: name + "_joint_state",
        get_joint=lambda robot, **kw: {0: np.zeros(6)},
        get_real_endpose=lambda robot, **kw: {0: np.array([1, 2, 3, 1, 0, 0, 0])},
        get_end_effector_real_val=lambda robot, **kw: {0: np.array([0.017, 0.017])},
        planner={arm: SimpleNamespace(plan_path=plan) for arm in ("left", "right")},
    )
    return SimpleNamespace(
        robot_manager=manager, take_action_cnt=[0], step_lim=10, end_flag=[False]
    ), plans


def test_native_state_distinguishes_command_from_measurement():
    env, _ = make_native()
    state = agent_support._motion_state(env)
    assert state["left_gripper"] == pytest.approx(0.5)
    assert state["targets"]["left_ee_joint_state"] == [0.0]
    assert state["targets"]["left_arm_joint_state"] == [0.4] * 6


@pytest.mark.parametrize("arm", ["left", "right"])
def test_native_waypoint_holds_all_other_targets(make_agent, arm):
    env, _ = make_native()
    agent = make_agent(env)
    agent.get_status = lambda: {
        "step": env.take_action_cnt[0],
        "step_limit": env.step_lim,
        "success": False,
    }
    captured = []

    def step(action, *, expected_step=None):
        captured.append(deepcopy(action))
        env.take_action_cnt[0] += 1
        return (
            {"feedback": "post-action"},
            0.0,
            False,
            {"executed_actions": 1, "status": agent.get_status()},
        )

    agent.step_native = step
    agent._observation = lambda: {"feedback": "post-action"}
    before = agent.get_motion_state()["targets"]
    result = agent.step_joint_update(arm, gripper=0.25)
    expected = deepcopy(before)
    expected[f"{arm}_ee_joint_state"] = [0.25]
    assert captured == [expected] and result["executed_actions"] == 1
    assert result["observation"] == {"feedback": "post-action"}
    assert agent.get_motion_state()["targets"] == before
    with pytest.raises(ValueError, match="joint limits"):
        agent.step_joint_update(arm, arm_qpos=[100.0] * 6)
    with pytest.raises(ValueError, match="finite"):
        agent.step_joint_update(arm, gripper=float("nan"))
    assert len(captured) == 1
    env.take_action_cnt[0] = env.step_lim
    assert agent.step_joint_update(arm, gripper=0)["executed_actions"] == 0
    assert len(captured) == 1


def test_native_plan_transforms_world_frame_without_motion():
    env, plans = make_native()
    result = agent_support._plan_arm_path(env, "left", [1.1, 2.2, 3.3, 2, 0, 0, 0])
    np.testing.assert_allclose(plans[0]["target_ee_pose"], [0.1, 0.2, 0.3, 1, 0, 0, 0])
    assert result["status"] == "Success" and env.take_action_cnt == [0]
    with pytest.raises(ValueError, match="quaternion"):
        agent_support._plan_arm_path(env, "left", [1, 2, 3, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="finite"):
        agent_support._plan_arm_path(env, "left", [np.nan, 2, 3, 1, 0, 0, 0])
    assert len(plans) == 1


@pytest.fixture
def toolkit_factory(tmp_path, monkeypatch):
    from rpent.utils import logging

    monkeypatch.setattr(logging, "_output_dir", tmp_path)

    def create(sam3=None):
        env = MotionEnv()
        toolkit = RoboDojoToolkit(
            primitives_kwargs={"env": env, "sam3_client": sam3},
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "memory"),
            state_output_dir=tmp_path / "states",
        )
        return env, toolkit

    return create


def test_queries_and_segmentation_use_requested_saved_frame(toolkit_factory):
    images = []
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False

    def segment(image, **kwargs):
        images.append(image)
        return SimpleNamespace(found=True, score=0.9, mask=mask, box=[0, 0, 3, 3])

    env, toolkit = toolkit_factory(sam3=SimpleNamespace(segment=segment))
    env.depth = 4
    toolkit.execute_tool("render", {})
    before = env.reads
    sample = toolkit.execute_tool(
        "sample_world_xyz",
        {"view": "head", "pixels": [[1, 1]], "step": 0, "neighborhood": 0},
    ).data
    assert sample["samples"][0]["xyz"] == [0, 0, -2]
    assert toolkit.execute_tool(
        "back_project", {"row": 1, "col": 1, "step": 1}
    ).data == {
        "camera": "cam_head",
        "step_idx": 1,
        "pixel": [1, 1],
        "depth_m": 4.0,
        "world_xyz": [0, 0, -4],
    }
    assert toolkit.execute_tool("back_project", {"row": 1, "col": 1, "step": 0}).data[
        "world_xyz"
    ] == [0, 0, -2]
    response = toolkit.execute_tool("segment", {"text_prompt": "ring", "step": 0})
    result = response.data
    assert result["step_idx"] == 0 and result["centroid_in_mask"] is False
    assert mask[tuple(result["point_rc"])]
    np.testing.assert_array_equal(images[0], np.ones((3, 3, 3)))
    assert env.reads == before
    assert toolkit._state.latest_step == 1
    assert len(response.images) == 1
    overlay = toolkit._state.load(result["overlay_artifact"], step=0)
    np.testing.assert_array_equal(overlay[1, 1], [1, 1, 1])
    assert overlay[0, 0, 0] > overlay[0, 0, 1]
    assert (
        toolkit._state.load(result["segment_artifact"], step=0)["point_rc"]
        == result["point_rc"]
    )
    again = toolkit.execute_tool("segment", {"text_prompt": "ring", "step": 0}).data
    assert again["segment_artifact"] != result["segment_artifact"]


def test_segmentation_failure_keeps_manual_localization_available(toolkit_factory):
    def fail(*args, **kwargs):
        raise TimeoutError("SAM3 unavailable")

    env, toolkit = toolkit_factory(sam3=SimpleNamespace(segment=fail))
    reads = env.reads
    result = toolkit.execute_tool("segment", {"text_prompt": "bottle"})
    assert result.is_error and "SAM3 unavailable" in result.data["error"]
    assert (
        result.data["fallback"]
        == "Inspect another saved view or revise the description; use only available tools."
    )
    assert not toolkit.execute_tool("back_project", {"row": 1, "col": 1}).is_error
    assert env.reads == reads


def test_segmentation_no_mask_explains_manual_fallback(toolkit_factory):
    from rpent.robots.components.sam3_client import Sam3Result

    _, toolkit = toolkit_factory(
        sam3=SimpleNamespace(
            segment=lambda *args, **kwargs: Sam3Result(
                found=False, score=0.1, reason="score below threshold"
            )
        )
    )
    response = toolkit.execute_tool("segment", {"text_prompt": "bottle"})
    assert not response.is_error
    assert response.data["found"] is False
    assert response.data["score"] == 0.1
    assert response.data["reason"] == "score below threshold"
    assert (
        response.data["fallback"]
        == "Inspect another saved view or revise the description; use only available tools."
    )
    assert not response.images


def test_segmentation_reports_artifact_write_failure(toolkit_factory, monkeypatch):
    from rpent.robots.components.sam3_client import Sam3Result

    _, toolkit = toolkit_factory(
        sam3=SimpleNamespace(
            segment=lambda *args, **kwargs: Sam3Result(
                found=True, score=0.9, mask=np.ones((3, 3), dtype=bool)
            )
        )
    )
    monkeypatch.setattr(toolkit._state, "save", lambda *args, **kwargs: None)
    response = toolkit.execute_tool("segment", {"text_prompt": "bottle"})
    assert response.is_error
    assert "persist" in response.data["error"]
    assert "overlay_artifact" not in response.data
    assert len(response.images) == 1


def test_invalid_depth_and_half_open_regions(toolkit_factory):
    env, toolkit = toolkit_factory()
    record = toolkit._state.latest_record()
    world = toolkit._state.load("head_world_xyz.npy", step=0)
    world[0, 0] = np.inf
    world[0, 1] = np.nan
    toolkit._state.save("head_world_xyz.npy", world, step=0)
    for col in (0, 1):
        assert toolkit.execute_tool("back_project", {"row": 0, "col": col}).is_error
    assert toolkit.execute_tool("back_project", {"row": 3, "col": 0}).is_error
    result = tools.query_world_map(
        toolkit._state, view="head", bbox=[0, 0, 2, 2], max_points=1
    ).data
    assert result["valid_points"] == 2 and result["returned_points"] == 1
    assert all(0 <= coordinate < 2 for coordinate in result["points"][0]["pixel"])
    result = tools.sample_world_xyz(toolkit._state, view="head", pixels=[[0, 0]]).data
    assert np.isfinite(result["samples"][0]["xyz"]).all()
    assert result["samples"][0]["valid_points"] == 2
    assert tools.sample_world_xyz(toolkit._state, view="head", pixels=[[3, 0]]).is_error
    assert record.step_idx == 0
    depth = np.array([[0, -1], [np.inf, np.nan]])
    assert np.isnan(tools._world_from_depth(depth, np.eye(3), np.eye(4))).all()


def test_scene_identity_in_config(tmp_path):
    import argparse

    parser = argparse.ArgumentParser()
    robot_spec._add_cli_args(parser, False)
    args = parser.parse_args(
        ["--task", "put_bottles_into_dustbin", "--seed", "2", "--layout", "3"]
    )
    args.output_dir = tmp_path
    config = robot_spec._parse_config(args)
    assert config.task_desc["seed"] == 2 and config.task_desc["layout"] == 3
    assert config.recipe_tag.endswith("s2_l3")
    assert robot_spec.get_robot_spec().supports_exploration is False


@pytest.mark.parametrize("frame", ["world", "tool"])
@pytest.mark.parametrize("angle", [37.0, -29.0])
@pytest.mark.parametrize("sign", [1, -1])
def test_move_delta_arbitrary_axis_and_nonidentity_pose(motion, frame, angle, sign):
    from scipy.spatial.transform import Rotation

    env, primitive = motion
    initial = Rotation.from_euler("xyz", [23, -14, 71], degrees=True)
    env.pose[3:] = (sign * initial.as_quat(scalar_first=True)).tolist()
    axis = np.array([2.0, -3.0, 4.0])
    delta = np.array([0.01, -0.02, 0.03])
    rotation = Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.deg2rad(angle))
    result = primitive.move_delta(
        arm="right",
        dxyz=delta.tolist(),
        axis=axis.tolist(),
        angle_deg=angle,
        frame=frame,
        substeps=1,
    ).data
    target = env.plans[-1][1]
    expected_rotation = rotation * initial if frame == "world" else initial * rotation
    expected_delta = delta if frame == "world" else initial.apply(delta)
    np.testing.assert_allclose(
        target[:3], np.array(env.pose[:3]) + expected_delta, atol=1e-12
    )
    actual = Rotation.from_quat(target[3:], scalar_first=True)
    assert (actual.inv() * expected_rotation).magnitude() < 1e-12
    assert result["executed_actions"] == 1


def test_move_delta_noop_and_explicit_gripper_target(motion):
    env, primitive = motion
    result = primitive.move_delta(arm="left").data
    assert result["executed_actions"] == 0 and result["stop_reason"] == "no_op"
    assert result["episode_status"]["step"] == 0
    assert not env.plans and not env.updates
    primitive.move_delta(arm="left", gripper=0.5, substeps=1)
    assert len(env.updates) == 1 and env.opening["left"] == 0.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"axis": [0, 0, 0], "angle_deg": 1},
        {"angle_deg": np.nan},
        {"dxyz": [0, np.inf, 0]},
        {"frame": "camera"},
        {"substeps": -1},
        {"substeps": True},
        {"axis": [1, 2]},
        {"gripper": -0.1},
    ],
)
def test_invalid_delta_never_plans_or_moves(motion, kwargs):
    env, primitive = motion
    with pytest.raises(ValueError):
        primitive.move_delta(arm="left", **kwargs)
    assert not env.plans and not env.updates


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_invalid_hold_never_moves(motion, steps):
    env, primitive = motion
    with pytest.raises(ValueError):
        primitive.hold(steps=steps)
    assert not env.updates
