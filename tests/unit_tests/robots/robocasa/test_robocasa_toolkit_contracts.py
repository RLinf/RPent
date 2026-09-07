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

"""Native RoboCasa execution, observation, and lifecycle contracts."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from rpent.tools import ToolContext

PERCEPTION = {
    "view_env_state",
    "back_project_batch",
    "query_world_map",
}
COMMON = {"read_text_file", "write_text_file", "list_dir", "read_image", "finish"}


def test_toolkit_initializes_environment_and_tool_policies(make_toolkit):
    run = make_toolkit()
    tk = run.toolkit
    assert run.env.reset_calls == 1
    assert not run.env.actions
    assert (
        run.config.output_dir / "success_criteria.md"
    ).read_text() == "offline success criteria"
    for definition in tk.list_tools():
        if definition.name not in COMMON:
            assert definition.readonly is (definition.name in PERCEPTION)
            assert definition.parallel is (definition.name in PERCEPTION)


def test_observation_images_and_errors_are_native_and_saved(make_toolkit):
    run = make_toolkit()
    tk = run.toolkit
    initial = tk.execute_tool("view_env_state", {"step": 0})
    assert initial.data["step"] == 0
    assert initial.data["task_language"] == run.env.language
    names = [image["artifact"] for image in initial.data["images"]]
    assert names == ["agentview_high.png", "navview.png", "wrist.png"]
    assert initial.images == [tk.state.load_bytes(name, step=0) for name in names]
    assert all(image.startswith(b"\x89PNG") for image in initial.images)
    json.dumps(initial.to_dict())
    result = tk.execute_tool("move_to", {"xyz": [2, 0, 1], "max_steps": 1})
    assert result.is_error
    assert "did not reach" in result.error
    assert result.data["step"] == 1
    assert result.data["log"]["result"]["ok"] is False
    assert "error" not in result.data["log"]["result"]
    assert tk.state.get(1).result["error"] == result.error
    assert result.images and result.data["artifacts"]
    historical = tk.execute_tool("view_env_state", {"step": 1})
    assert historical.data["log"]["result"]["error"] == result.error
    assert not historical.is_error
    assert tk.state.latest_record().step_idx == 1


@pytest.mark.parametrize(
    ("name", "arguments", "expected_steps"),
    [
        ("set_gripper", {"steps": 3}, 3),
        ("release", {"steps": 4}, 4),
        ("rotate_pitch", {"n": 5}, 5),
        ("move_base", {"forward": 0.2, "steps": 3}, 3),
        ("move_to", {"xyz": [2, 0, 1], "max_steps": 2}, 11),
        ("move_delta", {"dxyz": [2, 0, 0], "max_steps": 2}, 11),
        ("navigate_to", {"xy": [2, 0], "max_steps": 2}, 8),
    ],
)
def test_every_physical_step_records_one_frame(
    make_toolkit, name, arguments, expected_steps
):
    run = make_toolkit()
    result = run.toolkit.execute_tool(name, arguments)
    assert result.data["step"] == 1
    assert len(run.env.actions) == expected_steps
    assert len(run.toolkit._frames) == expected_steps
    assert [int(frame[0, 0, 0]) for frame in run.toolkit._frames] == list(
        range(1, expected_steps + 1)
    )
    assert run.toolkit._robot._vla_desync is True


def test_scripted_grasp_records_all_stages_and_stops_on_failure(make_toolkit):
    run = make_toolkit()
    result = run.toolkit.execute_tool("scripted_grasp", {"xyz": [0, 0, 1]})
    assert not result.is_error
    assert len(run.toolkit._frames) == len(run.env.actions) > 18
    assert np.all([action[6] == -1 for action in run.env.actions[:4]])
    failed = make_toolkit()
    failed.toolkit._robot._pos_jac = np.zeros((3, 3))
    result = failed.toolkit.execute_tool("scripted_grasp", {"xyz": [2, 0, 1]})
    assert result.is_error
    assert result.data["log"]["result"]["stage"] == "approach"
    assert len(failed.env.actions) == 204
    assert len(failed.toolkit._frames) == 204


def test_reset_guard_preserves_evaluation_and_clears_exploration_state(make_toolkit):
    run = make_toolkit()
    denied = run.toolkit.execute_tool("reset", {})
    assert denied.is_error and "DISABLED" in denied.error
    assert run.env.reset_calls == 1
    assert not run.env.actions
    explore = make_toolkit(RLDX_ALLOW_RESET=1)
    runtime = explore.toolkit._robot
    runtime._pos_jac = np.eye(3)
    runtime._fwd_offset = 1.0
    runtime._vla_desync = False
    assert not explore.toolkit.execute_tool("reset", {}).is_error
    assert explore.env.reset_calls == 2
    assert runtime._pos_jac is None and runtime._fwd_offset is None
    assert runtime._vla_desync is True
    assert explore.model.resets == 1


def test_cancellation_stops_remaining_steps_and_uses_fresh_call_signal(
    make_toolkit, monkeypatch
):
    run = make_toolkit()
    entered = threading.Event()
    resume = threading.Event()
    cancelled = threading.Event()
    check = ToolContext.check_cancelled

    def checkpoint(ctx):
        try:
            check(ctx)
        finally:
            if ctx._cancel_event.is_set():
                cancelled.set()

    def on_step():
        entered.set()
        assert resume.wait(5)

    monkeypatch.setattr(ToolContext, "check_cancelled", checkpoint)
    run.env.on_step = on_step
    with ThreadPoolExecutor(max_workers=2) as pool:
        action = pool.submit(run.toolkit.execute_tool, "move_base", {"steps": 20})
        assert entered.wait(5)
        # Request cancellation while the first step is in progress.
        stop = pool.submit(run.toolkit.cancel_active_and_wait)
        # Wait for the scheduler to signal cancellation before releasing the step.
        with run.toolkit._scheduler._condition:
            assert run.toolkit._scheduler._condition.wait_for(
                lambda: run.toolkit._scheduler._state == "paused", timeout=5
            )
        resume.set()
        result = action.result(timeout=5)
        stop.result(timeout=5)
    assert cancelled.is_set()
    assert result.is_error and "cancelled" in result.error
    assert len(run.env.actions) == len(run.toolkit._frames) == 1
    assert result.data["task_progress"]["steps"] == 1
    run.env.on_step = lambda: None
    run.toolkit.resume_calls()
    assert not run.toolkit.execute_tool("release", {"steps": 1}).is_error
    assert len(run.env.actions) == len(run.toolkit._frames) == 2


def test_partial_failure_captures_current_state_and_retains_both_errors(make_toolkit):
    run = make_toolkit()

    def fail_step():
        raise RuntimeError("actuator failed after moving")

    run.env.on_step = fail_step
    result = run.toolkit.execute_tool("move_base", {"forward": 1, "steps": 5})
    assert result.is_error and "actuator failed" in result.error
    assert result.data["state"]["robot0_base_pos"][0] == pytest.approx(0.01)
    assert len(run.env.actions) == 1

    def fail_render(*args, **kwargs):
        raise RuntimeError("camera unavailable")

    run.env.render_camera = fail_render
    result = run.toolkit.execute_tool("move_base", {"forward": 1, "steps": 5})
    assert "actuator failed" in result.error and "camera unavailable" in result.error
    assert "state" not in result.data


def test_finish_and_recipe_keep_environment_success_source(make_toolkit):
    run = make_toolkit()
    tk = run.toolkit
    assert not tk.execute_tool("release", {"steps": 2}).is_error
    finish = tk.execute_tool(
        "finish", {"status": "success", "summary": "planner claim"}
    )
    assert not finish.is_error
    assert tk.finish_result == {"status": "success", "summary": "planner claim"}
    assert tk.solved() is False
    run.env.success = True
    tk.execute_tool("release", {"steps": 1})
    assert tk.solved() is True
    run.env.success = False
    tk.execute_tool("release", {"steps": 1})
    assert tk.solved() is False
    path = run.config.output_dir / tk.write_recipe("OpenDrawer_s1")
    assert [json.loads(line)["action"] for line in path.read_text().splitlines()] == [
        "release"
    ] * 3


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("move_to", {"xyz": [1, 2]}),
        ("navigate_to", {"xy": [1, 2, 3]}),
        ("back_project_batch", {"pixels": [[1, 2, 3]]}),
        ("back_project_batch", {"pixels": [[1, 2]] * 51}),
        ("back_project_batch", {"pixels": [[1, 2]], "camera": "typo"}),
        ("query_world_map", {"x_range": [1]}),
        ("finish", {"status": "invalid", "summary": "bad"}),
    ],
)
def test_model_argument_validation_precedes_execution(make_toolkit, name, arguments):
    run = make_toolkit()
    result = run.toolkit.execute_tool(name, arguments)
    assert result.is_error and "Invalid arguments" in result.error
    assert not run.env.actions
    assert run.toolkit.state.latest_record().step_idx == 0


def test_perception_uses_saved_maps_and_reports_missing_artifacts(make_toolkit):
    run = make_toolkit(RLDX_KEEP_HEAVY_NPY=1)
    tk = run.toolkit
    result = tk.execute_tool(
        "back_project_batch", {"pixels": [[0, 0], [1, 1], [-1, 0]]}
    )
    assert not result.is_error
    assert result.data["summary"]["valid_count"] == 2
    assert result.data["summary"]["median_xyz"] == [1.0, 1.0, 0.9]
    assert not result.data["results"][2]["valid"]
    assert tk.execute_tool("query_world_map", {"min_cluster_size": 1}).data["clusters"]
    assert tk.execute_tool(
        "back_project_batch",
        {"pixels": [[0, 0]], "camera": "wrist", "resolution": "high"},
    ).is_error
    tk.execute_tool("release", {"steps": 1})
    pruned = tk.execute_tool("back_project_batch", {"pixels": [[0, 0]], "step": 0})
    assert pruned.is_error and "not found" in pruned.error
    assert tk.execute_tool("view_env_state", {"step": 0}).images
