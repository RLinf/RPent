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

import base64
import copy
import json
import threading
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import pytest
from pydantic import Field

from rpent.dashboard.events import StepRecordEvent
from rpent.memory import MemoryManager
from rpent.memory import manager as memory_module
from rpent.planner.utils.http_mcp_server import mcp_result
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, common, iter_tools, tool
from rpent.tools import base as tool_base


class _RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    @property
    def enabled(self) -> bool:
        return True

    def emit(self, event: Any) -> None:
        self.events.append(event)


class _ContractToolkit(Toolkit):
    def __init__(
        self,
        output_dir: Path,
        *,
        memory: MemoryManager | None = None,
    ) -> None:
        self.events = _RecordingEventSink()
        self.capture_calls: list[dict[str, Any]] = []
        self.capture_error: Exception | None = None
        super().__init__(
            dashboard_events=self.events,
            state=EnvState(output_dir),
            memory=memory or MemoryManager(output_dir / "memory"),
        )

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
        if self.capture_error is not None:
            raise self.capture_error
        call = {
            "command": copy.deepcopy(command),
            "result": copy.deepcopy(result),
            "elapsed_s": elapsed_s,
        }
        self.capture_calls.append(call)
        with self.state.record_step(
            state={"capture_count": len(self.capture_calls)},
            command=command,
            result=result,
            elapsed_s=elapsed_s,
        ):
            pass
        return ToolResult(data={"observation": len(self.capture_calls)})

    def solved(self) -> bool:
        return False


def test_tool_result_builds_text_and_images_without_mutating_result() -> None:
    payload = {"status": "ok", "count": 2}
    images = [b"main", b"camera", b"navigation", b"wrist"]
    result = ToolResult(data=payload, images=images)
    blocks = mcp_result(result)["content"]

    assert payload == {"status": "ok", "count": 2}
    assert images == [b"main", b"camera", b"navigation", b"wrist"]
    assert json.loads(blocks[0]["text"]) == payload
    assert [block["type"] for block in blocks] == ["text"] + ["image"] * 4
    assert [base64.b64decode(block["data"]) for block in blocks[1:]] == images
    assert all(block["mimeType"] == "image/png" for block in blocks[1:])


@pytest.mark.parametrize("value", ["plain text", 17, ["one", "two"]])
def test_tool_result_encodes_payload_values(value: Any) -> None:
    result = ToolResult(data={"value": value})
    assert json.loads(result.to_text()) == {"value": value}


@pytest.mark.parametrize("finished", [True, False])
def test_tool_result_preserves_finish_signal(finished: bool) -> None:
    result = ToolResult(data={"_finish": finished})
    assert result.to_dict()["_finish"] is finished
    assert json.loads(result.to_text())["_finish"] is finished


def test_tool_result_text_limit_preserves_internal_payload(monkeypatch) -> None:
    monkeypatch.setattr(tool_base, "MAX_TOOL_TEXT_BYTES", 30)
    payload = {"value": "界" * 10}
    result = ToolResult(data=payload)
    text = result.to_text()
    assert len(text.encode("utf-8")) <= 30
    assert text.endswith("\n[truncated]")
    assert result.data == payload


def test_toolkit_registers_common_specs_with_fresh_placeholder_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run-output"
    original = common.CommonTools.list_dir
    original_schema = original.input_schema
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: output_dir)
    toolkit = _ContractToolkit(tmp_path / "state")

    first = toolkit.list_tools()
    second = toolkit.list_tools()

    assert [definition.name for definition in first] == [
        "read_text_file",
        "write_text_file",
        "list_dir",
        "finish",
    ]
    list_dir = next(definition for definition in first if definition.name == "list_dir")
    assert str(output_dir) in list_dir.description
    assert common.CommonTools.list_dir is original
    assert original.input_schema == original_schema
    assert first == second
    assert first is not second
    list_dir.input_schema["properties"].clear()
    assert list_dir.input_schema == original_schema


def test_common_file_tools_dispatch_offline_without_capturing_robot_state(
    tmp_path: Path,
) -> None:
    toolkit = _ContractToolkit(tmp_path / "state")
    text_file = tmp_path / "files" / "note.txt"

    written = toolkit.execute_tool(
        "write_text_file",
        {"path": str(text_file), "content": "hello 世界"},
    )
    read = toolkit.execute_tool(
        "read_text_file",
        {"path": str(text_file), "max_chars": 7},
    )
    listed = toolkit.execute_tool("list_dir", {"path": str(text_file.parent)})
    finished = toolkit.execute_tool(
        "finish",
        {"status": "success", "summary": "done"},
    )

    assert written.data == {
        "path": str(text_file),
        "bytes_written": len("hello 世界".encode()),
    }
    assert read.data["path"] == str(text_file)
    assert read.data["size"] == len("hello 世界")
    assert read.data["content"].startswith("hello 世")
    assert "[TRUNCATED" in read.data["content"]
    assert listed.data == {
        "path": str(text_file.parent),
        "count": 1,
        "files": ["note.txt"],
    }
    assert finished.data == {
        "_finish": True,
        "status": "success",
        "summary": "done",
    }
    assert finished.data.get("_finish", False) is True
    assert toolkit.capture_calls == []
    assert toolkit.events.events == []


def test_common_file_tools_enforce_memory_manager_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    memory_root = repo_root / "memory" / "libero"
    published = memory_root / "global" / "strategy.md"
    published.parent.mkdir(parents=True)
    published.write_text("published")
    (memory_root / "MEMORY.md").write_text("index")
    root_leaf = memory_root / "notes.md"
    root_leaf.write_text("root-level note")
    evaluation_inbox = memory_root / "_internal" / "inbox" / "current-cell"
    evaluation_inbox.mkdir(parents=True)
    (evaluation_inbox / "draft.md").write_text("private draft")
    foreign = repo_root / "memory" / "robotwin" / "global" / "x.md"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("foreign")
    monkeypatch.setattr(memory_module, "get_repo_root", lambda: repo_root)

    read_only_memory = MemoryManager(memory_root)
    evaluation = _ContractToolkit(
        tmp_path / "evaluation-state",
        memory=read_only_memory,
    )
    read_published = evaluation.execute_tool(
        "read_text_file",
        {"path": "memory/libero/global/strategy.md"},
    )
    read_root_leaf = evaluation.execute_tool(
        "read_text_file",
        {"path": str(root_leaf)},
    )
    list_published = evaluation.execute_tool(
        "list_dir",
        {"path": str(published.parent)},
    )
    write_published = evaluation.execute_tool(
        "write_text_file",
        {"path": str(published), "content": "changed"},
    )
    read_foreign = evaluation.execute_tool(
        "read_text_file",
        {"path": str(foreign)},
    )
    read_evaluation_inbox = evaluation.execute_tool(
        "read_text_file",
        {"path": str(evaluation_inbox / "draft.md")},
    )

    assert evaluation.memory is read_only_memory
    assert read_published.data["content"] == "published"
    assert read_root_leaf.data["content"] == "root-level note"
    assert list_published.data["files"] == ["strategy.md"]
    assert "writing to memory is denied" in write_published.data["error"]
    assert "another robot's memory is denied" in read_foreign.data["error"]
    assert "reading this memory path is denied" in read_evaluation_inbox.data["error"]

    exploration = _ContractToolkit(
        tmp_path / "exploration-state",
        memory=MemoryManager(
            memory_root,
            memory_access="inbox_write",
            inbox_cell_tag="current-cell",
        ),
    )
    own_draft = memory_root / "_internal" / "inbox" / "current-cell" / "draft.md"
    other_draft = memory_root / "_internal" / "inbox" / "other-cell" / "draft.md"
    write_own = exploration.execute_tool(
        "write_text_file",
        {"path": str(own_draft), "content": "draft"},
    )
    read_own = exploration.execute_tool(
        "read_text_file",
        {"path": str(own_draft)},
    )
    read_other = exploration.execute_tool(
        "read_text_file",
        {"path": str(other_draft)},
    )
    inbox_escape = own_draft.parent / "published-link.md"
    inbox_escape.symlink_to(published)
    write_through_symlink = exploration.execute_tool(
        "write_text_file",
        {"path": str(inbox_escape), "content": "escaped"},
    )

    assert write_own.data["bytes_written"] == 5
    assert read_own.data["content"] == "draft"
    assert "reading this memory path is denied" in read_other.data["error"]
    assert "writing to memory is denied" in write_through_symlink.data["error"]
    assert published.read_text() == "published"
    assert evaluation.capture_calls == []
    assert exploration.capture_calls == []


def test_toolkit_reports_unknown_tools_and_invalid_arguments(tmp_path: Path) -> None:
    toolkit = _ContractToolkit(tmp_path)

    unknown = toolkit.execute_tool("missing", {"value": 1})
    invalid = toolkit.execute_tool("read_text_file", {"unexpected": True})

    assert unknown.data == {"error": "unknown tool: missing"}
    assert "bad arguments for read_text_file" in invalid.data["error"]
    assert {error["type"] for error in invalid.data["errors"]} == {
        "missing",
        "extra_forbidden",
    }
    assert toolkit.capture_calls == []


def test_readonly_declarations_bind_methods_and_execution_guards(
    tmp_path: Path,
) -> None:
    toolkit = _ContractToolkit(tmp_path)

    @tool(name="function", readonly=True)
    def readonly_function(value: str) -> ToolResult:
        return ToolResult(data={"value": value})

    class Handler:
        @tool(name="method", readonly=True)
        def readonly_method(self, *, prefix: str, value: str) -> ToolResult:
            return ToolResult(data={"value": prefix + value})

    handler = Handler()
    toolkit.add_tool(readonly_function)
    toolkit.add_tools(iter_tools(handler))
    calls = []

    def guarded(**kwargs):
        calls.append(kwargs)
        return handler.readonly_method(**kwargs)

    toolkit.add_tool(handler.readonly_method.with_handler(guarded), replace=True)
    assert toolkit.execute_tool("function", {"value": "plain"}).data == {
        "value": "plain"
    }
    assert toolkit.execute_tool(
        "method", {"prefix": "pre-", "value": "bound"}
    ).data == {"value": "pre-bound"}
    assert calls == [{"prefix": "pre-", "value": "bound"}]
    assert toolkit.capture_calls == []
    assert toolkit.events.events == []


def test_internal_resource_binding_is_not_a_model_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    resource = {"position": 3}

    @tool(readonly=True, exclude=("resource",))
    def observe(label: str, *, resource: dict[str, int]) -> ToolResult:
        """Read a named resource.

        Args:
            label: Label to attach to the observation.
        """
        return ToolResult(data={"label": label, "position": resource["position"]})

    toolkit = _ContractToolkit(tmp_path)
    toolkit.add_tool(observe.with_handler(partial(observe, resource=resource)))
    schema = toolkit.list_tools()[-1].input_schema
    assert set(schema["properties"]) == {"label"}
    assert schema["properties"]["label"]["description"] == (
        "Label to attach to the observation."
    )
    assert toolkit.execute_tool("observe", {"label": "current"}).data == {
        "label": "current",
        "position": 3,
    }
    rejected = toolkit.execute_tool(
        "observe", {"label": "forged", "resource": {"position": 99}}
    )
    assert rejected.is_error
    assert rejected.data["errors"][0]["type"] == "extra_forbidden"
    assert resource == {"position": 3}


@pytest.mark.parametrize(
    "value", [[1.0, 2.0], [1.0, 2.0, "3"], [1.0, 2.0, float("nan")]]
)
def test_invalid_vectors_never_execute_or_capture_state(tmp_path: Path, value) -> None:
    calls = []

    @tool
    def move(
        xyz: Annotated[list[float], Field(min_length=3, max_length=3)],
    ) -> ToolResult:
        """Move to the requested point."""
        calls.append(xyz)
        return ToolResult(data={"moved": True})

    toolkit = _ContractToolkit(tmp_path)
    toolkit.add_tool(move)
    result = toolkit.execute_tool("move", {"xyz": value})
    assert result.is_error
    assert calls == []
    assert toolkit.capture_calls == []


def test_stateful_dispatch_captures_state_and_emits_the_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((10.0, 10.25))
    monkeypatch.setattr(
        "rpent.tools.toolkit.time.perf_counter",
        lambda: next(clock),
    )
    toolkit = _ContractToolkit(tmp_path)
    handler_result = {"moved": True}

    @tool
    def move(*, distance: int) -> ToolResult:
        assert distance == 3
        return ToolResult(data=handler_result)

    toolkit.add_tool(move)

    result = toolkit.execute_tool("move", {"distance": 3})

    assert result.data == {"observation": 1}
    assert handler_result == {"moved": True}
    assert toolkit.capture_calls[0]["command"] == {
        "action": "move",
        "distance": 3,
    }
    assert toolkit.capture_calls[0]["result"] == {"moved": True}
    assert toolkit.capture_calls[0]["elapsed_s"] == 0.25
    assert len(toolkit.events.events) == 1
    event = toolkit.events.events[0]
    assert isinstance(event, StepRecordEvent)
    assert event.record.step_idx == 0
    assert event.record.command == {"action": "move", "distance": 3}
    assert event.env_state is toolkit.state


def test_handler_error_is_retained_when_state_capture_also_fails(
    tmp_path: Path,
) -> None:
    toolkit = _ContractToolkit(tmp_path)
    toolkit.capture_error = RuntimeError("capture exploded")

    @tool
    def fail() -> ToolResult:
        raise ValueError("handler exploded")

    @tool(readonly=True)
    def probe() -> ToolResult:
        return ToolResult(data={"ready": True})

    toolkit.add_tool(fail)
    toolkit.add_tool(probe)

    failed = toolkit.execute_tool("fail", {})

    assert failed.data["error"] == "handler exploded"
    assert failed.data["state_capture_error"] == "capture exploded"
    assert "ValueError: handler exploded" in failed.data["traceback"]
    assert toolkit.events.events == []
    assert toolkit.execute_tool("probe", {}).data == {"ready": True}


@pytest.mark.timeout(5)
def test_toolkit_rejects_overlapping_operations_and_cleans_up_after_success(
    tmp_path: Path,
) -> None:
    toolkit = _ContractToolkit(tmp_path)
    started = threading.Event()
    release = threading.Event()
    results: list[ToolResult] = []
    worker_errors: list[BaseException] = []

    @tool(readonly=True)
    def blocking() -> ToolResult:
        started.set()
        assert release.wait(2), "test did not release the blocking handler"
        return ToolResult(data={"released": True})

    toolkit.add_tool(blocking)

    def run_blocking() -> None:
        try:
            results.append(toolkit.execute_tool("blocking", {}))
        except BaseException as error:
            worker_errors.append(error)

    worker = threading.Thread(target=run_blocking, daemon=True)
    worker.start()
    try:
        assert started.wait(2), "blocking handler did not start"
        overlap = toolkit.execute_tool("finish", {"status": "failure", "summary": "x"})
        assert overlap.data == {"error": "another tool operation is still active"}
    finally:
        release.set()
        worker.join(2)

    assert not worker.is_alive()
    assert worker_errors == []
    assert len(results) == 1
    assert results[0].data == {"released": True}
    assert toolkit.execute_tool(
        "finish", {"status": "success", "summary": "clean"}
    ).data.get("_finish", False)


def test_toolkit_cleans_up_operation_after_handler_failure(tmp_path: Path) -> None:
    toolkit = _ContractToolkit(tmp_path)

    @tool(readonly=True)
    def fail() -> ToolResult:
        raise RuntimeError("tool failed")

    toolkit.add_tool(fail)

    failed = toolkit.execute_tool("fail", {})

    assert failed.data["error"] == "tool failed"
    assert "RuntimeError: tool failed" in failed.data["traceback"]
    assert toolkit.execute_tool(
        "finish", {"status": "failure", "summary": "recovered"}
    ).data.get("_finish", False)


@pytest.mark.timeout(5)
def test_toolkit_cooperatively_cancels_and_cleans_up_active_operation(
    tmp_path: Path,
) -> None:
    toolkit = _ContractToolkit(tmp_path)
    started = threading.Event()
    stop_polling = threading.Event()
    results: list[ToolResult] = []

    @tool
    def cancellable() -> ToolResult:
        started.set()
        while not stop_polling.wait(0.01):
            toolkit.raise_if_cancelled()
        return ToolResult(data={"unexpected": True})

    toolkit.add_tool(cancellable)
    worker = threading.Thread(
        target=lambda: results.append(toolkit.execute_tool("cancellable", {})),
        daemon=True,
    )
    worker.start()
    try:
        assert started.wait(2), "cancellable handler did not start"
        toolkit.cancel_active_and_wait()
    finally:
        stop_polling.set()
        worker.join(2)

    assert not worker.is_alive()
    assert results[0].data["code"] == "tool_cancelled"
    assert results[0].data["interrupted"] is True
    assert results[0].data["error"] == "tool operation interrupted"
    assert toolkit.capture_calls[0]["result"]["code"] == "tool_cancelled"
    assert len(toolkit.events.events) == 1
    assert toolkit.execute_tool(
        "finish", {"status": "failure", "summary": "cancelled"}
    ).data.get("_finish", False)
