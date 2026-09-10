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

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rpent.dashboard.events import (
    RunStartedEvent,
    RuntimeStatusEvent,
    StepRecordEvent,
    TranscriptEvent,
    UsageEvent,
)
from rpent.dashboard.interaction import (
    DashboardInteractionError,
    DashboardMessageConflictError,
    InteractionUnavailableError,
    UnknownDashboardMessageError,
)
from rpent.dashboard.server import DashboardServer
from rpent.dashboard.spec import DashboardSpec
from rpent.dashboard.state import DashboardState
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, readonly, tool

DASHBOARD_SPEC: DashboardSpec = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <mode> <seed>",
        "fields": (
            {"name": "mode", "choices": ("pick", "place")},
            {"name": "seed", "kind": "integer", "minimum": 0},
        ),
        "display": "{mode} / seed {seed}",
        "output_slug": "{mode}_s{seed}",
    },
    "runtime_components": (
        {"name": "model", "label": "MODEL", "scope": "shared"},
        {"name": "env", "label": "ENV", "scope": "unique"},
    ),
    "primitives": ("move_to",),
}


def _state(tmp_path: Path) -> DashboardState:
    return DashboardState(
        output_dir=tmp_path,
        dashboard_spec=DASHBOARD_SPEC,
    )


def _ready_state(tmp_path: Path) -> DashboardState:
    state = _state(tmp_path)
    state.shared_services_ready()
    return state


def _claim_started_task(
    state: DashboardState,
    command: str = "/rpent-task pick 1",
):
    state.submit_input(command)
    claimed = state.wait_for_task(timeout=0)
    assert claimed is not None
    state.emit(RunStartedEvent())
    return claimed


def test_dashboard_task_commands_validate_and_latest_unclaimed_request_wins(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    with pytest.raises(InteractionUnavailableError, match="still starting"):
        state.submit_input("/rpent-task pick 0")

    state.shared_services_ready()
    for command, message in (
        ("/rpent-other pick 0", "unknown Dashboard command"),
        ("/rpent-task pick", "expected /rpent-task"),
        ("/rpent-task inspect 0", "unsupported mode"),
        ("/rpent-task pick seed", "seed must be an integer"),
        ("/rpent-task pick -1", "seed must be at least 0"),
    ):
        with pytest.raises(ValueError, match=message):
            state.submit_input(command)

    state.submit_input("/rpent-task pick 1")
    state.submit_input("/rpent-task place 2")
    claimed = state.wait_for_task(timeout=0)

    assert claimed is not None
    assert claimed.number == 1
    assert claimed.request == {"mode": "place", "seed": 2}
    assert claimed.output_dir == tmp_path / "tasks" / "0001_place_s2"
    assert state.wait_for_task(timeout=0) is None
    snapshot = state.snapshot()
    assert snapshot["current_task"] == {"label": "place / seed 2"}
    assert snapshot["pending_task"] is None
    assert snapshot["control_error"] is None


def test_dashboard_task_replacement_seals_input_and_resets_only_unique_runtime(
    tmp_path: Path,
) -> None:
    state = _ready_state(tmp_path)
    first = _claim_started_task(state)
    state.emit(RuntimeStatusEvent("model", "ready"))
    state.emit(RuntimeStatusEvent("env", "ready"))
    state.set_planner_activity("idle", accepting_input=True)
    pending = state.submit_input("try a different grasp")

    state.submit_input("/rpent-task place 3")

    replacing = state.snapshot()
    assert first.number == 1
    assert state.task_replacement_requested is True
    assert replacing["session_state"] == "switch_pending"
    assert replacing["interaction"]["messages"] == [
        {
            "message_id": pending.message_id,
            "text": "try a different grasp",
            "status": "unsent",
            "error": None,
        }
    ]
    assert state.claim_next_pending_message() is None

    state.complete_task_replacement()
    state.complete_task(state="cancelled")

    completed = state.snapshot()
    assert completed["runtime"] == {
        "model": {"status": "ready", "error": None},
        "env": {"status": "pending", "error": None},
    }
    assert completed["session_state"] == "task_starting"
    assert completed["pending_task"] == {"label": "place / seed 3"}
    second = state.wait_for_task(timeout=0)
    assert second is not None
    assert second.number == 2
    assert second.request == {"mode": "place", "seed": 3}


def test_dashboard_message_lifecycle_is_ordered_and_returns_detached_values(
    tmp_path: Path,
) -> None:
    state = _ready_state(tmp_path)
    _claim_started_task(state)
    state.set_planner_activity("idle", accepting_input=True)

    withdrawn = state.submit_input("withdraw this")
    withdrawn.status = "sent"
    assert state.withdraw_message(withdrawn.message_id).status == "withdrawn"

    sent = state.submit_input("send this")
    assert state.claim_next_pending_message().message_id == sent.message_id
    assert state.mark_message_sent(sent.message_id).status == "sent"
    with pytest.raises(DashboardMessageConflictError, match="not sending"):
        state.mark_message_sent(sent.message_id)

    failed = state.submit_input("fail this")
    assert state.claim_next_pending_message().message_id == failed.message_id
    marked_failed = state.mark_message_failed(failed.message_id, "backend failed")
    assert marked_failed.status == "failed"
    assert marked_failed.error == "backend failed"

    unsent = state.submit_input("restore this")
    assert state.claim_next_pending_message().message_id == unsent.message_id
    assert state.mark_message_unsent(unsent.message_id).status == "unsent"
    with pytest.raises(UnknownDashboardMessageError, match="unknown Dashboard message"):
        state.mark_message_sent("missing")

    assert [item["status"] for item in state.snapshot()["interaction"]["messages"]] == [
        "withdrawn",
        "sent",
        "failed",
        "unsent",
    ]


def test_dashboard_interrupt_lifecycle_accepts_only_busy_active_tasks(
    tmp_path: Path,
) -> None:
    state = _ready_state(tmp_path)
    assert state.request_interrupt() == "noop"

    _claim_started_task(state)
    state.set_planner_activity("idle", accepting_input=True)
    pending = state.submit_input("queued during the run")
    assert state.request_interrupt() == "noop"
    state.set_planner_activity("busy")

    assert state.request_interrupt() == "accepted"
    assert state.request_interrupt() == "duplicate"
    assert state.claim_next_pending_message() is None
    assert state.claim_interrupt_request() is True
    assert state.claim_interrupt_request() is False
    state.complete_interrupt("backend interrupt failed")

    interaction = state.snapshot()["interaction"]
    assert interaction["interrupt_requested"] is False
    assert interaction["last_error"] == "backend interrupt failed"
    assert state.claim_next_pending_message().message_id == pending.message_id
    with pytest.raises(DashboardInteractionError, match="no interrupt request"):
        state.complete_interrupt()


def test_dashboard_primitives_are_available_only_while_planner_is_idle(
    tmp_path: Path,
) -> None:
    state = _ready_state(tmp_path)
    _claim_started_task(state)
    toolkit = MagicMock(spec=Toolkit)

    @tool
    def move_to(*, ctx) -> ToolResult:
        """Move the robot."""
        return ToolResult(data={"ok": True})

    toolkit.list_tools.return_value = (move_to,)
    tool_result = ToolResult(data={"ok": True})
    toolkit.execute_tool.return_value = tool_result
    state.bind_toolkit(toolkit)

    assert state.snapshot()["primitives_available"] is False
    with pytest.raises(InteractionUnavailableError, match="not available"):
        state.primitive_specs()
    with pytest.raises(InteractionUnavailableError, match="not available"):
        state.execute_primitive("move_to", {})

    state.set_planner_activity("idle", accepting_input=True)
    assert state.snapshot()["primitives_available"] is True
    assert state.primitive_specs() == [
        {"name": "move_to", "input_schema": move_to.input_schema}
    ]
    assert state.execute_primitive("move_to", {}) is tool_result

    state.set_planner_activity("busy")
    assert state.snapshot()["primitives_available"] is False
    with pytest.raises(InteractionUnavailableError, match="not available"):
        state.execute_primitive("move_to", {})


def test_dashboard_discovers_step_pngs_and_serves_only_current_frames(tmp_path):
    from fastapi.testclient import TestClient

    from rpent.dashboard.server import DashboardServer

    state = _ready_state(tmp_path)
    claimed = _claim_started_task(state)
    env_state = EnvState(claimed.output_dir)
    with TestClient(DashboardServer(state=state)._app) as client:
        assert client.get("/api/session/state").json()["frame_available"] == {}
        with env_state.record_step(state={}):
            env_state.save("navview.png", np.zeros((2, 2, 3), dtype=np.uint8))
            env_state.save("overhead.png", np.ones((2, 2, 3), dtype=np.uint8))
            env_state.save("depth.npz", np.zeros((2, 2)))
            env_state.save("metadata.json", {"camera": "overhead"})
            env_state.save("expired.png", np.zeros((2, 2, 3), dtype=np.uint8))
        env_state.artifact_path("expired.png").unlink()
        state.emit(StepRecordEvent(env_state.get(0), env_state))
        assert client.get("/api/session/state").json()["frame_available"] == {
            "navview": True,
            "overhead": True,
        }
        response = client.get("/api/session/frame", params={"kind": "overhead"})
        assert response.headers["content-type"] == "image/png"
        assert response.content == env_state.load_bytes("overhead.png")
        assert (
            client.get("/api/session/frame", params={"kind": "depth"}).status_code
            == 404
        )

        with env_state.record_step(state={}):
            env_state.save("wrist.png", np.ones((2, 2, 3), dtype=np.uint8))
        state.emit(StepRecordEvent(env_state.get(1), env_state))
        assert client.get("/api/session/state").json()["frame_available"] == {
            "wrist": True
        }
        assert (
            client.get("/api/session/frame", params={"kind": "overhead"}).status_code
            == 404
        )


def test_dashboard_events_project_runtime_usage_timeline_and_frames(
    tmp_path: Path,
) -> None:
    state = _ready_state(tmp_path)
    _claim_started_task(state)
    state.emit(RuntimeStatusEvent("model", "ready"))
    state.emit(UsageEvent(inp=3, out=4, tool_calls=1))
    state.emit(TranscriptEvent({"type": "assistant", "text": "working"}))
    state.begin_planner_session()
    state.emit(UsageEvent(inp=2, out=1, tool_calls=2))
    env_state = EnvState(tmp_path / "env")
    with env_state.record_step(
        state={"phase": 1},
        terminated=True,
        command={"action": "move_to", "arm": "left"},
        result={
            "position": np.asarray([1.0, 2.0, 3.0]),
            "path": Path("artifact.json"),
        },
        elapsed_s=0.5,
    ):
        env_state.save("camera.png", np.zeros((2, 2, 3), dtype=np.uint8))
    record = env_state.latest_record()
    assert record is not None
    state.emit(StepRecordEvent(record, env_state))

    detail = state.session_detail()
    assert detail["usage"] == {"in": 5, "out": 5, "tool_calls": 3}
    assert detail["runtime"]["model"] == {"status": "ready", "error": None}
    assert detail["timeline"] == [
        {
            "step": 0,
            "action": "move_to",
            "args": {"arm": "left"},
            "result": {"position": [1.0, 2.0, 3.0], "path": "artifact.json"},
            "elapsed_s": 0.5,
            "terminated": True,
            "truncated": False,
            "has_action_video": False,
        }
    ]
    assert state.events_since(0) == [{"type": "assistant", "text": "working"}]
    assert state.frame("camera") == env_state.load_bytes("camera.png")
    assert state.frame("unknown") is None
    with pytest.raises(ValueError, match="unknown runtime component"):
        state.emit(RuntimeStatusEvent("missing", "ready"))
    with pytest.raises(ValueError, match="unknown runtime status"):
        state.emit(RuntimeStatusEvent("model", "unknown"))


def test_dashboard_step_events_offset_new_traces_and_resolve_action_video(
    tmp_path: Path,
) -> None:
    state = _ready_state(tmp_path / "dashboard")
    _claim_started_task(state)

    first_env = EnvState(tmp_path / "first")
    with first_env.record_step(
        state={"phase": 1},
        command={"action": "move_to", "arm": "left"},
        result={"position": np.asarray([1, 2, 3])},
        elapsed_s=0.25,
    ):
        first_env.save("camera.png", np.zeros((2, 2, 3), dtype=np.uint8))
        first_env.save("action.mp4", b"first-video")
    first_record = first_env.latest_record()
    assert first_record is not None
    state.emit(StepRecordEvent(first_record, first_env))

    second_env = EnvState(tmp_path / "second")
    with second_env.record_step(
        state={"phase": 2},
        terminated=True,
        command={"action": "release", "arm": "right"},
        result={"released": True},
        elapsed_s=0.5,
    ):
        second_env.save("camera.png", np.ones((2, 2, 3), dtype=np.uint8))
    second_record = second_env.latest_record()
    assert second_record is not None
    state.emit(StepRecordEvent(second_record, second_env))

    detail = state.session_detail()
    assert [item["step"] for item in detail["timeline"]] == [0, 1]
    assert detail["timeline"][0]["result"] == {"position": [1, 2, 3]}
    assert detail["timeline"][1]["terminated"] is True
    assert state.frame("camera") == second_env.load_bytes("camera.png")
    assert state.action_video_path(0) == first_env.artifact_path("action.mp4", step=0)


def test_primitive_http_uses_native_validation_results_and_observation(tmp_path):
    state = _ready_state(tmp_path)
    _claim_started_task(state)
    calls = []

    @tool
    def move_to(distance: int, *, ctx) -> ToolResult:
        """Move a test robot."""
        calls.append(distance)
        if distance < 0:
            return ToolResult(error="motion rejected")
        return ToolResult(data={"distance": distance})

    @tool
    @readonly
    def finish(status: str, summary: str, *, ctx) -> ToolResult:
        """Finish the test task."""
        return ToolResult(data={"status": status, "summary": summary})

    class RobotToolkit(Toolkit):
        def _capture_observation(self, *, command, result, elapsed_s):
            with self.state.record_step(
                state={"distance": calls[-1]},
                command=command,
                result=result.to_dict(),
                elapsed_s=elapsed_s,
            ):
                self.state.save("overhead.png", np.zeros((2, 2, 3), dtype=np.uint8))
            return {"distance": calls[-1]}, []

    toolkit = RobotToolkit(
        robot=None,
        tools=(move_to, finish),
        state=EnvState(tmp_path / "env"),
        memory=MemoryManager(root=tmp_path / "memory"),
        output_dir=tmp_path,
        dashboard_events=state,
    )
    state.bind_toolkit(toolkit)
    state.set_planner_activity("idle", accepting_input=True)
    with TestClient(DashboardServer(state=state)._app) as client:
        specs = client.get("/api/session/primitives").json()["primitives"]
        assert specs == [{"name": "move_to", "input_schema": move_to.input_schema}]
        for arguments in ({}, {"distance": "bad"}):
            response = client.post(
                "/api/session/primitive",
                json={"name": "move_to", "arguments": arguments},
            )
            assert response.status_code == 422
            assert "Invalid arguments for move_to" in response.json()["error"]
        assert calls == []
        response = client.post(
            "/api/session/primitive", json={"name": "finish", "arguments": {}}
        )
        assert response.status_code == 403
        # Coercion is the same native validation used by planner tool calls.
        response = client.post(
            "/api/session/primitive",
            json={"name": "move_to", "arguments": {"distance": "2"}},
        )
        assert response.json() == {"ok": True}
        assert calls == [2]
        snapshot = client.get("/api/session/state").json()
        assert snapshot["frame_available"] == {"overhead": True}
        frame = client.get("/api/session/frame", params={"kind": "overhead"})
        assert frame.headers["content-type"] == "image/png"
        assert frame.content == toolkit.state.load_bytes("overhead.png")
        assert (
            client.get("/api/session/frame", params={"kind": "missing"}).status_code
            == 404
        )
        response = client.post(
            "/api/session/primitive",
            json={"name": "move_to", "arguments": {"distance": -1}},
        )
        assert response.status_code == 422
        assert response.json() == {"error": "motion rejected"}
        state.set_planner_activity("busy")
        assert (
            client.post(
                "/api/session/primitive",
                json={"name": "move_to", "arguments": {"distance": 3}},
            ).status_code
            == 409
        )
        assert calls == [2, -1]
    state.unbind_toolkit(toolkit)
    toolkit.close()
