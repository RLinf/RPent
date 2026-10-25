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
    assert result["episode_status"]["remaining_steps"] == 0
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
    assert result["episode_status"]["remaining_steps"] == 96


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

    def step(action):
        captured.append(deepcopy(action))
        env.take_action_cnt[0] += 1
        return (
            {"feedback": "post-action"},
            0.0,
            False,
            {"executed_actions": 1, "status": agent.get_status()},
        )

    agent.step_native = step
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

    def create(mode="evaluation", attempts=5, sam3=None):
        env = MotionEnv()
        toolkit = RoboDojoToolkit(
            primitives_kwargs={"env": env, "sam3_client": sam3},
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "memory"),
            mode=mode,
            attempts_per_session=attempts,
            state_output_dir=tmp_path / mode,
        )
        return env, toolkit

    return create


@pytest.mark.parametrize("mode", ["evaluation", "exploration"])
def test_observations_and_motion_report_budget_at_capture_time(toolkit_factory, mode):
    env, toolkit = toolkit_factory(mode=mode)
    initial_budget = {"step": 0, "step_limit": 100, "remaining_steps": 100}
    initial = toolkit.execute_tool("view_env_state", {}).data
    assert initial["episode_status"] == initial_budget
    calls = [
        ("set_gripper", {"arm": "left", "val": 0, "steps": 4}, 4),
        ("move_to", {"arm": "left", "xyz": [0.1, 0.2, 0.9], "substeps": 5}, 9),
        ("rotate_wrist", {"arm": "left", "delta_yaw_deg": 1, "substeps": 1}, 10),
        ("release", {"arm": "left", "steps": 2}, 12),
    ]
    for name, arguments, used in calls:
        result = toolkit.execute_tool(name, arguments)
        assert not result.is_error
        budget = {"step": used, "step_limit": 100, "remaining_steps": 100 - used}
        assert result.data["episode_status"] == budget
        assert result.data["log"]["result"]["episode_status"] == {
            **budget,
            "success": False,
        }
    assert result.data["step"] == 4
    reads = env.reads
    assert toolkit.execute_tool("view_env_state", {}).data["episode_status"] == budget
    assert (
        toolkit.execute_tool("view_env_state", {"step": 0}).data["episode_status"]
        == initial_budget
    )
    assert env.reads == reads and env.step_count == 12
    if mode == "exploration":
        reset = toolkit.execute_tool("reset", {}).data
        assert reset["episode_status"] == initial_budget
        assert (
            toolkit.execute_tool("view_env_state", {"step": 4}).data["episode_status"]
            == budget
        )
    toolkit.close()


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
    assert "back_project" in result.data["fallback"]
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
    assert "back_project" in response.data["fallback"]
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


def test_reset_attempt_budget_and_history_are_runtime_enforced(toolkit_factory):
    env, toolkit = toolkit_factory(mode="exploration")
    assert env.resets == 1
    assert toolkit.execute_tool(
        "finish", {"status": "failure", "summary": "try"}
    ).is_error
    for attempt in range(2, 6):
        result = toolkit.execute_tool("reset", {}).data
        assert result["log"]["result"]["attempt"] == attempt
    assert env.resets == 5
    assert toolkit.execute_tool("reset", {}).is_error
    assert env.resets == 5 and toolkit._state.get(0).step_idx == 0
    assert toolkit.execute_tool(
        "finish", {"status": "failure", "summary": "done"}
    ).data["_finish"]
    toolkit.close()
    # No env.close: the same runtime serves the next planner session.


def test_solved_reset_refused_and_only_successful_attempt_exported(toolkit_factory):
    env, toolkit = toolkit_factory(mode="exploration")
    toolkit.execute_tool("set_gripper", {"arm": "left", "val": 0, "steps": 1})
    toolkit.execute_tool("reset", {})
    original = env.step_joint_update

    def succeed(**kwargs):
        result = original(**kwargs)
        env.success = True
        return result

    env.step_joint_update = succeed
    toolkit.execute_tool("release", {"arm": "left", "steps": 1})
    assert toolkit.execute_tool("reset", {}).is_error
    path = toolkit.write_recipe("example")
    import json
    from pathlib import Path

    recipe = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert recipe == [{"action": "release", "arm": "left", "steps": 1}]


def test_recipe_preserves_sam3_call_order_and_excludes_failed_attempts(toolkit_factory):
    import json
    from pathlib import Path

    def segment(image, *, text_prompt, **kwargs):
        if text_prompt == "service error":
            raise RuntimeError("SAM3 unavailable")
        return SimpleNamespace(
            found=text_prompt != "missing",
            score=0.9,
            mask=np.ones(image.shape[:2], dtype=bool),
            box=[0, 0, 3, 3],
        )

    env, toolkit = toolkit_factory(
        mode="exploration", sam3=SimpleNamespace(segment=segment)
    )
    toolkit.execute_tool("segment", {"text_prompt": "failed attempt"})
    toolkit.execute_tool("set_gripper", {"arm": "left", "val": 0, "steps": 1})
    assert toolkit.write_recipe("unsolved") == ""
    toolkit.execute_tool("segment", {"text_prompt": "before reset"})
    toolkit.execute_tool("reset", {})
    source_step = toolkit._state.latest_step
    first = {"text_prompt": "object", "step": source_step}
    toolkit.execute_tool("segment", first)
    toolkit.execute_tool("segment", {"text_prompt": "missing"})
    toolkit.execute_tool("segment", {"text_prompt": "service error"})
    toolkit.execute_tool("back_project", {"row": 1, "col": 1})
    toolkit.execute_tool("set_gripper", {"arm": "left", "val": 0, "steps": 1})

    # Two later queries of the same saved image must stay after the motion,
    # rather than being sorted by image step or deduplicated.
    later = {"text_prompt": "object again", "step": source_step, "min_score": 0.5}
    toolkit.execute_tool("segment", later)
    toolkit.execute_tool("segment", later)
    original = env.step_joint_update

    def succeed(**kwargs):
        result = original(**kwargs)
        env.success = True
        return result

    env.step_joint_update = succeed
    toolkit.execute_tool("release", {"arm": "left", "steps": 1})
    final = {"text_prompt": "final observation"}
    toolkit.execute_tool("segment", final)
    toolkit.execute_tool("finish", {"status": "success", "summary": "done"})
    path = toolkit.write_recipe("example")
    recipe = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert recipe == [
        {"action": "segment", **first},
        {"action": "set_gripper", "arm": "left", "val": 0, "steps": 1},
        {"action": "segment", **later},
        {"action": "segment", **later},
        {"action": "release", "arm": "left", "steps": 1},
        {"action": "segment", **final},
    ]
    toolkit.close()


def test_failed_reset_does_not_consume_an_attempt(toolkit_factory):
    env, toolkit = toolkit_factory(mode="exploration")
    original = env.reset

    def fail():
        raise RuntimeError("reset failed")

    env.reset = fail
    assert toolkit.execute_tool("reset", {}).is_error
    env.reset = original
    result = toolkit.execute_tool("reset", {}).data
    assert result["log"]["result"]["attempt"] == 2


def test_explore_defaults_and_scene_identity_in_config(tmp_path):
    import argparse

    parser = argparse.ArgumentParser()
    robot_spec._add_cli_args(parser, False)
    args = parser.parse_args(
        ["--task", "put_bottles_into_dustbin", "--seed", "2", "--layout", "3"]
    )
    args.output_dir = tmp_path
    args.explore = True
    config = robot_spec._parse_config(args)
    assert args.explore_attempts_per_session == 5 and args.explore_sessions == 3
    assert config.task_desc["seed"] == 2 and config.task_desc["layout"] == 3
    assert config.recipe_tag.endswith("s2_l3")
    prompt = robot_spec.get_robot_spec().prompts.render(
        "system", variables={**config.prompt_vars, "output_dir": str(tmp_path)}
    )
    assert "5 attempts" in " ".join(prompt.split())
    args.random = True
    with pytest.raises(ValueError, match="fixed seed/layout"):
        robot_spec._validate_args(args)


@pytest.mark.parametrize("task", ["stack_blocks", "stack_blocks_random"])
def test_evaluation_references_seed_zero_without_changing_output_identity(
    tmp_path, task
):
    import argparse

    parser = argparse.ArgumentParser()
    robot_spec._add_cli_args(parser, False)
    args = parser.parse_args(["--task", task, "--seed", "1", "--layout", "3"])
    args.output_dir = tmp_path
    config = robot_spec._parse_config(args)
    assert config.recipe_tag == f"{task}_s1_l3"
    assert config.prompt_vars["reference_tag"] == f"{task}_s0_l0"
    prompt = robot_spec.get_robot_spec().prompts.render(
        "system", variables={**config.prompt_vars, "output_dir": str(tmp_path)}
    )
    memory = config.prompt_vars["memory_dir"]
    assert f"{memory}/task-specific/{task}_s0_l0.json" in prompt
    assert f"{memory}/task-specific/{task}_s0_l0_recipe.jsonl" in prompt
    assert f"{memory}/task-family/task-family_robodojo_arx_x5_t{task}.md" in prompt
    assert f"{tmp_path}/{task}_s1_l3.json" in prompt
    assert f"{memory}/task-specific/{task}_s1_l3" not in prompt
    other = "stack_blocks" if task.endswith("_random") else "stack_blocks_random"
    assert f"{memory}/task-specific/{other}_s0_l0" not in prompt
    assert "{{" not in prompt


def test_exploration_memory_can_be_published_and_read_by_evaluation(
    tmp_path, monkeypatch
):
    import json

    from rpent.robots.robot_spec import RunConfig
    from rpent.utils import logging

    monkeypatch.setattr(logging, "_output_dir", tmp_path)
    root = tmp_path / "memory"
    tag = "general_pickup_s0_l0"
    config = RunConfig(
        recipe_tag=tag,
        output_dir=tmp_path,
        prompt_vars={"memory_dir": str(root)},
        task_desc={"task": "general_pickup", "seed": 0, "layout": 0},
    )
    env = MotionEnv()
    toolkit = robot_spec.get_toolkit(
        runtime_kwargs={"env": env},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        mode="exploration",
        attempts_per_session=2,
        state_output_dir=tmp_path / "sessions" / "session_001",
    )
    proposal = root / "_internal" / "inbox" / tag / "task-family_draft.md"
    content = (
        "---\nscope: task-family\nsuite: robodojo\nregime: arx_x5\n"
        "task_id: general_pickup\ntask_language: Move the object.\n"
        f"evidence:\n  cells: [{tag}]\nconfidence: single-shot\n---\n"
        "Re-localize the target after a reset.\n"
    )
    assert not toolkit.execute_tool(
        "write_text_file", {"path": str(proposal), "content": content}
    ).is_error
    assert toolkit.execute_tool(
        "write_text_file", {"path": str(root / "MEMORY.md"), "content": "overwrite"}
    ).is_error
    original = env.step_joint_update

    def succeed(**kwargs):
        result = original(**kwargs)
        env.success = True
        return result

    env.step_joint_update = succeed
    toolkit.execute_tool("release", {"arm": "left", "steps": 1})
    toolkit.write_recipe(tag)
    (tmp_path / f"{tag}.json").write_text(json.dumps({"success": True}))
    merged = toolkit.memory.merge_memory(
        cell_tag=tag, run_state_dir=tmp_path, solved=True
    )
    assert merged["task-family"] == 1 and merged["task"] == 1
    assert toolkit.memory.validate() == []
    toolkit.close()

    evaluation = robot_spec.get_toolkit(
        runtime_kwargs={"env": MotionEnv()},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        state_output_dir=tmp_path / "evaluation",
    )
    leaf = next((root / "task-family").glob("*.md"))
    result = evaluation.execute_tool("read_text_file", {"path": str(leaf)})
    assert "Re-localize" in result.data["content"]
    assert evaluation.execute_tool(
        "write_text_file", {"path": str(proposal), "content": content}
    ).is_error
    assert "reset" not in {tool.name for tool in evaluation.list_tools()}
    evaluation.close()
