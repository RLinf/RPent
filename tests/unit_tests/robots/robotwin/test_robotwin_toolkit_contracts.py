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

"""Native RoboTwin action, observation, cancellation, and lifecycle contracts."""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from robots.robotwin import tools
from robots.robotwin.robot_spec import ROBOTWIN_CAMERA_NAMES
from rpent.dashboard.events import StepRecordEvent
from rpent.tools import ToolCancelled, ToolContext
from rpent.tools.common_tools import COMMON_TOOLS


def test_initial_observation_and_readonly_tools(robotwin):
    tk = robotwin.toolkit
    assert {t.name for t in tk.list_tools()} == {
        t.name for t in (*COMMON_TOOLS, *tools.ROBOTWIN_TOOLS)
    }
    assert {t.name for t in tools.ROBOTWIN_TOOLS if t.parallel} == {
        "view_env_state",
        "sample_world_xyz",
        "query_world_map",
    }
    assert tools.finish.readonly and not tools.finish.parallel
    assert robotwin.env.renders == [(camera, True) for camera in ROBOTWIN_CAMERA_NAMES]
    initial = tk.state.get(0)
    assert initial.command == {"action": "reset"}
    assert initial.result == {**robotwin.env.last_reset_info, "success": True}
    assert len(initial.artifacts) == 12
    result = tk.execute_tool("view_env_state", {})
    assert not result.is_error
    assert len(result.images) == 3
    assert result.images == [
        tk.state.load_bytes(f"{c}_rgb.png", step=0) for c in ROBOTWIN_CAMERA_NAMES
    ]
    assert "_image_bytes" not in result.data
    assert "log" not in result.data["state"]
    assert tk.state.latest_step == 0
    assert len(robotwin.env.renders) == 3
    assert not tk.solved()
    assert tk.execute_tool("view_env_state", {"step": 99}).is_error


@pytest.mark.parametrize(
    "substeps,indices",
    [(0, [0, 1, 2, 3, 4, 5]), (1, [5]), (3, [0, 2, 5]), (25, [0, 1, 2, 3, 4, 5])],
)
def test_move_preserves_path_sampling_and_fresh_other_arm_state(
    robotwin, substeps, indices
):
    env = robotwin.env

    def change_other_arm():
        env.last_info["robot_state"]["qpos_target14"][7] += 1

    env.on_step = change_other_arm
    result = robotwin.toolkit.execute_tool(
        "move_to",
        {
            "arm": "left",
            "xyz": [0.2, 0.3, 0.4],
            "gripper": 0.7,
            "substeps": substeps,
        },
    )
    assert not result.is_error
    np.testing.assert_allclose(env.plans[0][1], [0.2, 0.3, 0.4, 1, 0, 0, 0])
    np.testing.assert_allclose(np.asarray(env.steps)[:, :6], env.path[indices])
    np.testing.assert_allclose(np.asarray(env.steps)[:, 6], 0.7)
    np.testing.assert_allclose(np.asarray(env.steps)[:, 7], np.arange(len(indices)))
    assert result.data["log"]["result"]["executed_steps"] == len(indices)
    assert result.data["state"]["episode_status"]["native_actions"] == len(indices)
    assert result.data["state"]["episode_status"]["policy_actions"] == 0
    assert len(robotwin.toolkit._frames) == len(indices)
    assert robotwin.toolkit.state.latest_step == 1
    assert len(env.renders) == 6


def test_rotate_uses_world_z_and_only_captures_once(robotwin):
    result = robotwin.toolkit.execute_tool(
        "rotate_wrist",
        {
            "arm": "right",
            "delta_yaw_deg": 90,
            "substeps": 1,
        },
    )
    assert not result.is_error
    arm, target = robotwin.env.plans[0]
    assert arm == "right"
    np.testing.assert_allclose(
        target, [0.4, 0.5, 0.6, np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    )
    assert result.data["log"]["result"]["requested_delta_yaw_deg"] == 90
    assert robotwin.toolkit.state.latest_step == 1
    assert len(robotwin.env.renders) == 6


@pytest.mark.parametrize(
    "name,args,target",
    [
        ("set_gripper", {"val": 0.8}, 0.8),
        ("release", {}, 1.0),
    ],
)
def test_gripper_interpolation_and_release_composition(robotwin, name, args, target):
    result = robotwin.toolkit.execute_tool(name, {"arm": "right", "steps": 4, **args})
    assert not result.is_error
    np.testing.assert_allclose(
        np.asarray(robotwin.env.steps)[:, 13], np.arange(1, 5) * target / 4
    )
    np.testing.assert_allclose(np.asarray(robotwin.env.steps)[:, :13], 0)
    assert result.data["log"]["result"]["gripper_val"] == target
    assert robotwin.toolkit.state.latest_step == 1


@pytest.mark.parametrize("empty_path", [False, True])
def test_plan_failure_reports_native_error_and_captures_without_recipe(
    robotwin, empty_path
):
    if empty_path:
        robotwin.env.path = np.empty((0, 6))
    else:
        robotwin.env.plan_status = "Failure"
    result = robotwin.toolkit.execute_tool("move_to", {"arm": "left", "xyz": [1, 2, 3]})
    assert result.is_error
    assert result.data["log"]["result"]["stop_reason"] == "plan_failed"
    assert "error" not in result.data["log"]["result"]
    assert "error" in robotwin.toolkit.state.get(1).result
    assert len(result.images) == 3
    assert not robotwin.env.steps
    recipe = robotwin.toolkit.write_recipe("failure")
    assert (robotwin.output_dir / recipe).read_text() == ""


def test_partial_execution_failure_keeps_counts_frames_and_observation(robotwin):
    original = robotwin.env.step

    def step(action, **kwargs):
        if len(robotwin.env.steps) == 2:
            raise RuntimeError("offline step failure")
        return original(action, **kwargs)

    robotwin.env.step = step
    result = robotwin.toolkit.execute_tool("set_gripper", {"arm": "left", "val": 1})
    assert result.is_error and "offline step failure" in result.error
    assert len(robotwin.env.steps) == len(robotwin.toolkit._frames) == 2
    assert result.data["state"]["episode_status"]["native_actions"] == 2
    assert robotwin.toolkit.state.latest_step == 1


def _context(robotwin):
    cancel = threading.Event()
    tk = robotwin.toolkit
    return ToolContext(
        state=tk.state,
        memory=tk.memory,
        robot=tk._robot,
        output_dir=robotwin.output_dir,
        record_frame=tk.record_frame,
        _cancel_event=cancel,
    ), cancel


def test_cancel_stops_before_next_waypoint_and_keeps_completed_count(robotwin):
    ctx, cancel = _context(robotwin)
    robotwin.env.on_step = cancel.set
    with pytest.raises(ToolCancelled):
        tools.move_to.handler(arm="left", xyz=[0.1, 0.2, 0.3], ctx=ctx)
    assert len(robotwin.env.steps) == 1
    assert ctx.robot.native_actions == 1
    assert len(robotwin.toolkit._frames) == 1


@pytest.mark.parametrize("cancel_at,executed", [("infer", 0), ("chunk", 50)])
def test_lingbot_cancellation_boundaries_preserve_counters(
    robotwin, cancel_at, executed
):
    ctx, cancel = _context(robotwin)
    if cancel_at == "infer":
        robotwin.model.on_infer = cancel.set
    else:
        robotwin.env.on_chunk = cancel.set
    with pytest.raises(ToolCancelled):
        tools.lingbot_act.handler(chunks=2, ctx=ctx)
    assert ctx.robot.policy_actions == ctx.robot.native_actions == executed
    assert len(robotwin.toolkit._frames) == executed
    assert len(robotwin.env.chunks) == (1 if executed else 0)


def test_lingbot_native_instruction_rgb_only_inference_and_all_frame_recording(
    robotwin,
):
    result = robotwin.toolkit.execute_tool(
        "lingbot_act", {"chunks": 2, "prompt": "ignored instruction"}
    )
    assert not result.is_error
    assert len(robotwin.model.observations) == 2
    for observation in robotwin.model.observations:
        assert observation["task_language"] == robotwin.env.get_task_language()
        assert all(set(view) == {"rgb"} for view in observation["views"].values())
    assert robotwin.env.renders[3:9] == [
        (camera, False) for _ in range(2) for camera in ROBOTWIN_CAMERA_NAMES
    ]
    assert all(
        actions.shape == (50, 16) and all_frames
        for actions, all_frames in robotwin.env.chunks
    )
    action = result.data["log"]["result"]
    assert action["prompt"] == robotwin.env.get_task_language()
    assert action["agent_prompt_ignored"] is True
    assert action["ignored_agent_prompt"] == "ignored instruction"
    assert action["requested_steps"] == action["executed_steps"] == 100
    assert len(robotwin.toolkit._frames) == 100
    assert result.data["state"]["episode_status"]["native_actions"] == 100
    assert result.data["state"]["episode_status"]["policy_actions"] == 100
    assert robotwin.toolkit.state.latest_step == 1


def test_lingbot_records_final_frame_without_all_frames_capability(robotwin):
    robotwin.env.execution_capabilities = {"chunk_step_all_frames": False}
    result = robotwin.toolkit.execute_tool("lingbot_act", {"chunks": 1})
    assert not result.is_error
    assert robotwin.env.chunks[0][1] is False
    assert len(robotwin.toolkit._frames) == 1


@pytest.mark.parametrize(
    "tool_name,arguments",
    [
        ("lingbot_act", {"chunks": 2}),
        ("set_gripper", {"arm": "left", "val": 1}),
    ],
)
@pytest.mark.parametrize("success", [True, False])
def test_native_success_and_budget_exhaustion_stop_execution(
    robotwin, tool_name, arguments, success
):
    if success:
        robotwin.env.success_at = 2
    else:
        robotwin.env.last_info["episode_status"]["step_lim"] = 2
    result = robotwin.toolkit.execute_tool(tool_name, arguments)
    assert not result.is_error
    action = result.data["log"]["result"]
    assert action["executed_steps"] == 2
    assert action["completed"] is False
    assert action["stop_reason"] == (
        "native_success" if success else "budget_exhausted"
    )
    assert result.data["terminated"] is success
    assert result.data["truncated"] is (not success)
    assert robotwin.toolkit.solved() is success
    if tool_name == "lingbot_act":
        robotwin.toolkit.execute_tool(tool_name, arguments)
        assert len(robotwin.model.observations) == 1


@pytest.mark.parametrize(
    "native_success,requested,reported",
    [
        (False, "success", "failure"),
        (True, "failure", "success"),
        (False, "stuck", "stuck"),
    ],
)
def test_finish_verifies_native_status_and_does_not_capture(
    robotwin, native_success, requested, reported
):
    robotwin.env.last_info["episode_status"]["eval_success"] = native_success
    result = robotwin.toolkit.execute_tool(
        "finish", {"status": requested, "summary": "done"}
    )
    assert not result.is_error
    assert result.data["status"] == reported
    assert result.data["success"] is native_success
    assert robotwin.toolkit.finish_result == {"status": reported, "summary": "done"}
    assert robotwin.toolkit.state.latest_step == 0
    assert len(robotwin.env.renders) == 3


def test_finish_still_ends_planner_when_native_status_is_unavailable(robotwin):
    def failed_status():
        raise RuntimeError("status unavailable")

    robotwin.toolkit._robot.status = failed_status
    result = robotwin.toolkit.execute_tool(
        "finish", {"status": "success", "summary": "done"}
    )
    assert not result.is_error
    assert result.data["runtime_error"] == "RuntimeError: status unavailable"
    assert robotwin.toolkit.finish_result == {"status": "error", "summary": "done"}


def test_persisted_perception_uses_same_step_and_view_without_rendering(robotwin):
    tk = robotwin.toolkit
    sampled = tk.execute_tool(
        "sample_world_xyz", {"view": "head", "pixels": [[1, 2]], "neighborhood": 0}
    )
    assert not sampled.is_error
    np.testing.assert_allclose(sampled.data["samples"][0]["xyz"], [1, -0.5, -1])
    queried = tk.execute_tool(
        "query_world_map", {"view": "left_wrist", "bbox": [0, 0, 2, 3], "max_points": 2}
    )
    assert not queried.is_error
    assert queried.data["valid_points"] == 6
    assert queried.data["returned_points"] == 2
    assert [p["pixel"] for p in queried.data["points"]] == [[0, 0], [1, 2]]
    assert tk.state.latest_step == 0
    assert len(robotwin.env.renders) == 3
    outside = tk.execute_tool("sample_world_xyz", {"view": "head", "pixels": [[99, 0]]})
    assert outside.is_error and outside.data["code"] == "pixel_out_of_bounds"
    missing = tk.execute_tool(
        "query_world_map", {"view": "missing", "bbox": [0, 0, 1, 1]}
    )
    assert missing.is_error and missing.data["code"] == "view_not_found"


def test_recording_dashboard_and_recipe_export_use_common_toolkit(
    robotwin, monkeypatch
):
    tk = robotwin.toolkit
    events = []

    class Sink:
        enabled = True
        emit = staticmethod(events.append)

    tk._dashboard_events = Sink()
    videos = []
    save = tk.state.save

    def save_artifact(name, value, **kwargs):
        if name.endswith(".mp4"):
            videos.append((name, len(value), kwargs))
            if kwargs["step"] is not None:
                tk.state.latest_record().artifacts.add(name)
            return None
        return save(name, value, **kwargs)

    monkeypatch.setattr(tk.state, "save", save_artifact)
    result = tk.execute_tool("release", {"arm": "left", "steps": 3})
    assert not result.is_error
    assert "action_release.mp4" in result.data["artifacts"]
    assert len(events) == 1 and isinstance(events[0], StepRecordEvent)
    tk.execute_tool("finish", {"status": "failure", "summary": "offline"})
    recipe = tk.write_recipe("offline")
    assert [
        json.loads(line)
        for line in (robotwin.output_dir / recipe).read_text().splitlines()
    ] == [{"action": "release", "arm": "left", "val": 1.0, "steps": 3}]
    tk.close()
    assert videos == [
        ("action_release.mp4", 3, {"step": 1, "fps": 20}),
        ("episode.mp4", 3, {"step": None, "fps": 20}),
    ]
    assert tk.execute_tool("render", {}).is_error


def test_malformed_planner_waypoint_cannot_broadcast_into_joint_targets(robotwin):
    robotwin.env.path = np.ones((2, 1))
    result = robotwin.toolkit.execute_tool("move_to", {"arm": "left", "xyz": [1, 2, 3]})
    assert result.is_error and "shape [N,6]" in result.error
    assert not robotwin.env.steps
    assert robotwin.toolkit.state.latest_step == 1


def test_perception_reports_missing_initial_record(robotwin):
    robotwin.toolkit.state.reset()
    result = robotwin.toolkit.execute_tool(
        "sample_world_xyz", {"view": "head", "pixels": [[0, 0]]}
    )
    assert result.is_error and result.data["code"] == "state_not_found"
    assert robotwin.toolkit.execute_tool("view_env_state", {}).is_error
