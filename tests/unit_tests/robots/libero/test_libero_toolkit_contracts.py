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

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from robots.libero import robot_spec
from rpent.dashboard.events import NullDashboardEventSink
from rpent.robots import RunConfig
from rpent.robots.components.sam3_client import Sam3Result


@pytest.mark.parametrize("reset_after_success", [False, True])
def test_robot_finish_uses_cumulative_solved_state(make_toolkit, reset_after_success):
    toolkit, env, _ = make_toolkit(mode="exploration", attempts=4)
    assert toolkit._robot.mode == "exploration"
    assert not toolkit.solved()
    env.after_step = lambda: setattr(env, "terminated", True)
    assert not toolkit.execute_tool("set_gripper", {"steps": 1}).is_error
    assert toolkit.solved() and toolkit._robot.solved
    if reset_after_success:
        assert not toolkit.execute_tool(
            "reset", {"reason": "record another attempt"}
        ).is_error
        assert not env.terminated
        assert toolkit.solved() and toolkit._robot.solved
    record_count = len(toolkit.state.records())
    result = toolkit.execute_tool("finish", {"status": "success", "summary": "完成"})
    assert not result.is_error
    assert toolkit.finish_result == {"status": "success", "summary": "完成"}
    assert len(toolkit.state.records()) == record_count


@pytest.mark.parametrize("mode, attempts", [("evaluation", 4), ("exploration", 0)])
def test_robot_finish_preserves_unrestricted_modes(make_toolkit, mode, attempts):
    toolkit, _, _ = make_toolkit(mode=mode, attempts=attempts)
    assert not toolkit.solved()
    result = toolkit.execute_tool("finish", {"status": "stuck", "summary": "无法完成"})
    assert result.data == {"_finish": True, "status": "stuck", "summary": "无法完成"}
    assert not result.is_error
    assert not toolkit.solved()


def test_modes_filters_directories_and_exploration_guards(make_toolkit):
    evaluation, env, _ = make_toolkit()
    assert env.reset_calls == 1
    assert "reset" not in {tool.name for tool in evaluation.list_tools()}
    assert evaluation.execute_tool("reset", {"reason": "again"}).error.startswith(
        "Unknown tool: "
    )
    exploration, env, _ = make_toolkit(mode="exploration", attempts=3)
    assert "reset" in {t.name for t in exploration.list_tools()}
    assert exploration.execute_tool(
        "finish", {"status": "success", "summary": "early"}
    ).is_error
    assert exploration.finish_result is None
    for attempt in [2, 3]:
        reset = exploration.execute_tool("reset", {"reason": "new strategy"})
        assert not reset.is_error
        assert reset.data["log"]["result"]["attempt"] == attempt
        assert reset.data["step"] == attempt - 1
    assert exploration.execute_tool("reset", {"reason": "too many"}).is_error
    assert env.reset_calls == 3
    accepted = exploration.execute_tool(
        "finish", {"status": "failure", "summary": "spent"}
    )
    assert accepted.data == {"_finish": True, "status": "failure", "summary": "spent"}
    assert exploration.finish_result == {"status": "failure", "summary": "spent"}


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("move_to", {"xyz": [0, 0, 0.3]}),
        ("move_pose", {"xyz": [0, 0, 0.3]}),
        ("rotate_wrist", {"target_yaw": 0}),
        ("rotate_pitch", {"target_pitch": 0}),
        ("move_pose", {"xyz": [1, 1, 1], "max_steps": 0}),
        ("move_to", {"xyz": [1, 1, 1], "max_steps": 0}),
    ],
)
def test_already_reached_and_zero_budget_report_zero_actions(make_toolkit, name, args):
    toolkit, env, _ = make_toolkit()
    result = toolkit.execute_tool(name, args)
    assert not result.is_error
    assert result.data["log"]["result"]["steps_used"] == 0
    assert result.data["step"] == 1
    assert env.actions == []
    assert len(result.images) == 3
    assert all(image.startswith(b"\x89PNG\r\n\x1a\n") for image in result.images)


@pytest.mark.parametrize(
    ("name", "args", "field"),
    [
        ("move_to", {"xyz": [1, 0, 0.3], "max_steps": 4}, "steps_used"),
        ("move_pose", {"xyz": [1, 0, 0.3], "max_steps": 4}, "steps_used"),
        ("rotate_wrist", {"target_yaw": 1, "max_steps": 4}, "steps_used"),
        ("rotate_pitch", {"target_pitch": 1, "max_steps": 4}, "steps_used"),
        ("set_gripper", {"steps": 4}, "steps"),
        ("release", {"max_steps": 4}, "steps_used"),
    ],
)
def test_early_termination_counts_and_records_only_sent_actions(
    make_toolkit, name, args, field
):
    toolkit, env, _ = make_toolkit()
    env.after_step = lambda: setattr(env, "terminated", True)
    result = toolkit.execute_tool(name, args)
    assert not result.is_error
    assert (
        result.data["log"]["result"][field]
        == len(env.actions)
        == len(toolkit._frames)
        == 1
    )
    assert toolkit.solved()
    assert result.data["terminated"]


ACTION_CASES = [
    ("move_to", {"xyz": [1, 0, 0.3], "max_steps": 4}, 1),
    ("move_pose", {"xyz": [1, 0, 0.3], "max_steps": 4}, 1),
    ("rotate_wrist", {"target_yaw": 1, "max_steps": 4}, 1),
    ("rotate_pitch", {"target_pitch": 1, "max_steps": 4}, 1),
    ("set_gripper", {"steps": 4}, 1),
    ("release", {"max_steps": 4}, 1),
    ("pi0_pick", {"prompt": "pick bowl", "max_chunks": 2}, 3),
    ("pi0_doubled", {"prompt": "touch bowl", "max_chunks": 2}, 3),
]


@pytest.mark.parametrize(("name", "args", "completed"), ACTION_CASES)
def test_cancellation_preserves_partial_execution_capture_and_resume(
    make_toolkit, name, args, completed
):
    toolkit, env, _ = make_toolkit()
    stepped, release_step = threading.Event(), threading.Event()

    def after_step():
        stepped.set()
        assert release_step.wait(3)

    env.after_step = after_step
    with ThreadPoolExecutor(max_workers=2) as pool:
        action = pool.submit(toolkit.execute_tool, name, args)
        assert stepped.wait(3)
        cancellation = pool.submit(toolkit.cancel_active_and_wait)
        # Confirm cancellation was delivered before allowing another action boundary.
        assert toolkit._active_operation.cancel_event.wait(3)
        release_step.set()
        result = action.result(3)
        cancellation.result(3)
    assert result.is_error and result.error == "Tool call cancelled."
    assert len(env.actions) == toolkit._robot.executed_steps == completed
    assert len(toolkit._frames) == completed
    assert toolkit.state.latest_record().result == {"error": result.error}
    assert result.data["step"] == 1
    env.after_step = lambda: None
    assert not toolkit.execute_tool("set_gripper", {"steps": 1}).is_error
    assert len(env.actions) == completed + 1


@pytest.mark.parametrize(("name", "args", "completed"), ACTION_CASES)
def test_action_failures_keep_completed_steps_and_capture(
    make_toolkit, monkeypatch, name, args, completed
):
    toolkit, env, _ = make_toolkit()
    step = env.step

    def fail_after_completed(action):
        if toolkit._robot.executed_steps >= completed:
            raise TypeError("driver defect")
        return step(action)

    monkeypatch.setattr(env, "step", fail_after_completed)
    result = toolkit.execute_tool(name, args)
    assert result.is_error
    assert result.data["log"]["result"] == {}
    assert toolkit._robot.executed_steps == completed
    assert toolkit.state.latest_record().result == {"error": result.error}
    assert len(env.actions) == len(toolkit._frames) == completed
    assert result.data["step"] == 1
    assert len(result.images) == 3


def test_model_results_keep_original_observation_shape_and_full_disk_history(
    make_toolkit,
):
    toolkit, _, _ = make_toolkit()
    result = toolkit.execute_tool("rotate_wrist", {})
    model_text = result.to_text()
    assert model_text.count(result.error) == 1
    assert "error" not in result.data["log"]["result"]
    payload = json.loads(model_text)
    assert set(payload) == {
        "step",
        "terminated",
        "truncated",
        "state",
        "artifacts",
        "task_language",
        "log",
        "agent_elapsed_s",
        "error",
    }
    assert payload["error"] == result.error
    assert payload["log"]["result"] == {"name": "rotate_wrist"}
    # Rendering must not strip the error from the stored observation/history.
    assert toolkit.state.latest_record().result["error"] == result.error
    observed = toolkit.execute_tool("view_env_state", {})
    historical = json.loads(observed.to_text())
    assert "error" not in historical
    assert historical["log"]["result"] == {
        "name": "rotate_wrist",
        "error": result.error,
    }

    manifest = json.loads((toolkit.state._output_dir / "states.json").read_text())
    record = manifest["steps"][-1]
    assert record["command"] == {
        "action": "rotate_wrist",
        **toolkit._tools.get("rotate_wrist").args_schema().model_dump(),
    }
    assert record["result"] == {"name": "rotate_wrist", "error": result.error}
    assert record["elapsed_s"] >= 0


def test_validation_precedes_execution_and_uses_model_defaults(make_toolkit):
    toolkit, env, _ = make_toolkit(mode="exploration")
    assert toolkit.execute_tool("move_to", {"xyz": [0, 1]}).error.startswith(
        "Invalid arguments for "
    )
    assert toolkit.execute_tool("reset", {}).error.startswith("Invalid arguments for ")
    assert len(toolkit.state.records()) == 1
    result = toolkit.execute_tool(
        "set_gripper", {"steps": "2", "ctx": "ignored", "extra": 1}
    )
    assert not result.is_error
    assert result.data["log"]["result"]["steps"] == 2
    assert len(env.actions) == 2


def test_segment_is_exclusive_and_captures_after_saving_artifacts(make_toolkit):
    toolkit, env, _ = make_toolkit()
    definition = next(t for t in toolkit.list_tools() if t.name == "segment")
    assert not definition.readonly
    for index in [0, 1]:
        result = toolkit.execute_tool("segment", {"prompt": "bowl", "step": 0})
        assert not result.is_error
        data = result.data["log"]["result"]
        assert data["step"] == 0
        assert data["segment_artifact"] == f"segment_{index:02d}.json"
        assert toolkit.state.load(data["segment_artifact"], step=0)["prompt"] == "bowl"
        assert result.data["step"] == index + 1
        assert result.images[0].startswith(b"\x89PNG\r\n\x1a\n")
    assert len(toolkit.state.records()) == 3
    assert env.actions == []
    assert not toolkit.execute_tool("back_project", {"row": 4, "col": 4}).is_error
    assert not toolkit.execute_tool("view_camera_meta", {}).is_error


@pytest.mark.parametrize("save_fails", [False, True])
def test_segment_errors_appear_once_and_keep_diagnostics(
    make_toolkit, monkeypatch, save_fails
):
    toolkit, env, _ = make_toolkit()
    reason = "SAM3 found no matching mask"
    monkeypatch.setattr(
        toolkit._robot._sam3_client,
        "segment",
        lambda *args, **kwargs: Sam3Result(found=False, reason=reason),
    )
    if save_fails:
        save = toolkit.state.save

        def fail_segment_save(name, *args, **kwargs):
            if name.startswith("segment_"):
                return None
            return save(name, *args, **kwargs)

        monkeypatch.setattr(toolkit.state, "save", fail_segment_save)

    result = toolkit.execute_tool("segment", {"prompt": "bowl"})
    data = result.data["log"]["result"]
    assert not ({"code", "segmentation_error", "world_error"} & data.keys())
    assert data["found"] is False
    assert data["world_xyz"] is None
    assert result.to_text().count(reason) == 1
    assert len(toolkit.state.records()) == 2
    assert env.actions == []
    if save_fails:
        assert result.is_error
        assert json.loads(result.error.split("\n", 1)[1]) == {
            "segmentation_error": reason
        }
        assert "segment_artifact" not in data
    else:
        assert result.is_error
        assert result.error == reason
        assert toolkit.state.load(data["segment_artifact"], step=0)["error"] == reason


@pytest.mark.parametrize("name", ["pi0_pick", "pi0_doubled"])
def test_vla_prompt_chunks_recording_and_unsolved_result_are_preserved(
    make_toolkit, name
):
    toolkit, env, model = make_toolkit()
    result = toolkit.execute_tool(name, {"prompt": "touch bowl", "max_chunks": 2})
    assert not result.is_error
    assert result.data["log"]["result"]["success"] is False
    assert result.data["log"]["result"]["chunks_used"] == 2
    assert model.instructions == ["touch bowl", "touch bowl"]
    assert toolkit._robot._last_obs["task_descriptions"] == "original task"
    assert (
        len(env.actions) == toolkit._robot.executed_steps == len(toolkit._frames) == 6
    )


def test_factory_binds_memory_permissions_and_task_root(monkeypatch, tmp_path):
    from robots.libero import toolkit as module

    captured = []
    monkeypatch.setattr(
        module, "LiberoToolkit", lambda **kwargs: captured.append(kwargs) or kwargs
    )
    config = RunConfig(
        recipe_tag="cell",
        output_dir=tmp_path / "run",
        prompt_vars={"memory_dir": str(tmp_path / "memory")},
        task_desc={},
    )
    for mode in ["evaluation", "exploration"]:
        result = robot_spec.get_toolkit(
            runtime_kwargs={"env": "offline"},
            dashboard_events=NullDashboardEventSink(),
            config=config,
            mode=mode,
            state_output_dir=tmp_path / "state",
        )
        assert result["runtime_kwargs"] == {"env": "offline"}
        assert result["output_dir"] == config.output_dir
        path = tmp_path / "memory" / "_internal" / "inbox" / "cell" / "draft.md"
        if mode == "evaluation":
            with pytest.raises(PermissionError):
                result["memory"].authorize_write(path)
        else:
            assert result["memory"].authorize_write(path) == path
