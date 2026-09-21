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

import copy
import json
import threading
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from rpent.dashboard.events import StepRecordEvent
from rpent.memory import MemoryManager
from rpent.memory import manager as memory_manager
from rpent.session import EnvState
from rpent.tools import base as tools_base
from rpent.tools import common, iter_tools, tool
from rpent.tools.toolkit import Toolkit, ToolResult
from rpent.utils.templates import substitute


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


@pytest.mark.parametrize("declaration", ["bare", "default", "writable"])
def test_decorated_method_uses_existing_capture_and_result_contract(
    tmp_path, monkeypatch, declaration
) -> None:
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    declare = {
        "bare": tool,
        "default": tool(),
        "writable": tool(readonly=False),
    }[declaration]

    class Primitives:
        def __init__(self):
            self.moves = []

        @declare
        def move(
            self,
            distance: Annotated[
                float, Field(gt=0, json_schema_extra={"default": 0.1})
            ] = 0.1,
        ) -> ToolResult:
            """Move this robot by the requested distance."""
            self.moves.append(distance)
            return ToolResult(data={"moved": distance})

    primitives = Primitives()
    toolkit = _ContractToolkit(tmp_path)
    toolkit.add_tool(primitives.move)

    spec = next(
        item.input_schema for item in toolkit.list_tools() if item.name == "move"
    )
    assert set(spec["properties"]) == {"distance"}
    assert spec["properties"]["distance"]["default"] == 0.1
    result = toolkit.execute_tool("move", {})
    assert primitives.moves == [0.1]
    assert toolkit.capture_calls[0]["command"] == {"action": "move", "distance": 0.1}
    assert toolkit.capture_calls[0]["result"] == {"moved": 0.1}
    assert len(toolkit.events.events) == 1
    assert result.to_dict() == {"observation": 1}
    assert json.loads(result.to_text()) == {"observation": 1}

    for arguments in (
        {"distance": -1},
        {"distance": float("nan")},
        {"distance": float("inf")},
        {"distance": "invalid"},
        {"self": primitives},
        {"unknown": 1},
    ):
        result = toolkit.execute_tool("move", arguments)
        assert result.to_dict()["error"] == "bad arguments for move"
        assert result.to_dict()["errors"]
    assert primitives.moves == [0.1]
    assert len(toolkit.capture_calls) == len(toolkit.events.events) == 1

    with pytest.raises(TypeError, match="unbound method"):
        toolkit.add_tool(Primitives.move)


def test_decorated_readonly_tools_keep_finish_and_image_contracts(tmp_path) -> None:
    class Primitives:
        @tool(readonly=True, exclude=("state",))
        def observe(self, *, label: str = "bowl", state: EnvState) -> ToolResult:
            """Read a saved view."""
            assert state is toolkit.state
            return ToolResult(data={"label": label}, images=[b"png"])

    @tool(readonly=True)
    def finish(status: str) -> ToolResult:
        """Report the outcome."""
        return ToolResult(data={"_finish": True, "status": status})

    toolkit = _ContractToolkit(tmp_path)
    observe = Primitives().observe
    toolkit.add_tool(observe.with_handler(partial(observe, state=toolkit.state)))
    toolkit.add_tool(finish, replace=True)
    rejected = toolkit.execute_tool("observe", {"state": "model-provided"})
    assert rejected.is_error
    observed = toolkit.execute_tool("observe", {})
    assert json.loads(observed.to_text()) == {"label": "bowl"}
    assert observed.images[0] == b"png"
    result = toolkit.execute_tool("finish", {"status": "success"})
    assert toolkit.finish_result is not None
    assert result.to_dict() == {"_finish": True, "status": "success"}
    assert toolkit.capture_calls == toolkit.events.events == []


def test_decorated_parameters_keep_python_types_but_log_json(tmp_path) -> None:
    class Target(BaseModel):
        label: str

    @tool
    def inspect_target(target: Target) -> ToolResult:
        return ToolResult(data={"label": target.label})

    toolkit = _ContractToolkit(tmp_path)
    toolkit.add_tool(inspect_target)
    result = toolkit.execute_tool("inspect_target", {"target": {"label": "bowl"}})
    assert result.to_dict() == {"observation": 1}
    assert toolkit.capture_calls[0]["result"] == {"label": "bowl"}
    assert toolkit.capture_calls[0]["command"] == {
        "action": "inspect_target",
        "target": {"label": "bowl"},
    }


def test_tool_result_builds_text_and_images_without_mutating_result() -> None:
    images = [b"main", b"camera", b"navigation", b"wrist"]
    result = {"status": "ok", "count": 2}
    original = copy.deepcopy(result)
    tool_result = ToolResult(data=result, images=images)
    assert json.loads(tool_result.to_text()) == original
    assert result == original
    assert not tool_result.is_error
    assert tool_result.images == images


@pytest.mark.parametrize("value", ["plain text", 17, ["one", "two"]])
def test_tool_result_serializes_scalar_and_list_data(value) -> None:
    result = ToolResult(data={"value": value})
    assert json.loads(result.to_text()) == {"value": value}


def test_toolkit_accepts_finish_only_after_successful_execution(tmp_path) -> None:
    toolkit = _ContractToolkit(tmp_path)

    @tool(readonly=True)
    def finish(status: str) -> ToolResult:
        if status == "refused":
            return ToolResult(error="finish refused")
        return ToolResult(data={"status": status, "operator_notes": "confirmed"})

    toolkit.add_tool(finish, replace=True)
    rejected = toolkit.execute_tool("finish", {"status": "refused"})
    assert rejected.is_error and toolkit.finish_result is None
    toolkit.execute_tool("finish", {"status": "success"})
    assert toolkit.finish_result == {"status": "success", "operator_notes": "confirmed"}
    copy = toolkit.finish_result
    copy["status"] = "mutated"
    assert toolkit.finish_result["status"] == "success"


def test_native_error_flag_is_independent_of_business_data(tmp_path) -> None:
    toolkit = _ContractToolkit(tmp_path)

    @tool(readonly=True)
    def inspect_scene() -> ToolResult:
        return ToolResult(data={"error": {"count": 0}})

    toolkit.add_tool(inspect_scene)
    result = toolkit.execute_tool("inspect_scene", {})
    assert not result.is_error
    assert result.data == {"error": {"count": 0}}


@pytest.mark.parametrize("value", ["界" * 10, "x" * 100])
def test_tool_result_text_limit_preserves_complete_data(monkeypatch, value) -> None:
    monkeypatch.setattr(tools_base, "MAX_TOOL_TEXT_BYTES", 20)
    result = ToolResult(data={"value": value})
    text = result.to_text()
    assert len(text.encode("utf-8")) <= 20
    assert text.endswith("[truncated]")
    assert result.data == {"value": value}


def test_toolkit_registers_common_specs_with_fresh_placeholder_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run-output"
    original = common.CommonTools.list_dir.description
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: output_dir)
    toolkit = _ContractToolkit(tmp_path / "state")
    first = toolkit.list_tools()
    second = toolkit.list_tools()
    assert [item.name for item in first] == [
        "read_text_file",
        "write_text_file",
        "list_dir",
        "finish",
    ]
    listed = next(item for item in first if item.name == "list_dir")
    assert str(output_dir) in listed.description
    assert "Published memory is read-only." in listed.description
    assert (
        str(output_dir)
        in substitute(listed.input_schema)["properties"]["path"]["description"]
    )
    assert common.CommonTools.list_dir.description == original
    assert first == second
    assert first is not second


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

    assert written.to_dict() == {
        "path": str(text_file),
        "bytes_written": len("hello 世界".encode()),
    }
    assert read.to_dict()["path"] == str(text_file)
    assert read.to_dict()["size"] == len("hello 世界")
    assert read.to_dict()["content"].startswith("hello 世")
    assert "[TRUNCATED" in read.to_dict()["content"]
    assert listed.to_dict() == {
        "path": str(text_file.parent),
        "count": 1,
        "files": ["note.txt"],
    }
    assert finished.to_dict() == {
        "_finish": True,
        "status": "success",
        "summary": "done",
    }
    assert toolkit.finish_result is not None
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
    monkeypatch.setattr(memory_manager, "get_repo_root", lambda: repo_root)

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
    assert read_published.to_dict()["content"] == "published"
    assert read_root_leaf.to_dict()["content"] == "root-level note"
    assert list_published.to_dict()["files"] == ["strategy.md"]
    assert "writing to memory is denied" in write_published.to_dict()["error"]
    assert "another robot's memory is denied" in read_foreign.to_dict()["error"]
    assert (
        "reading this memory path is denied" in read_evaluation_inbox.to_dict()["error"]
    )

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

    assert write_own.to_dict()["bytes_written"] == 5
    assert read_own.to_dict()["content"] == "draft"
    assert "reading this memory path is denied" in read_other.to_dict()["error"]
    assert "writing to memory is denied" in write_through_symlink.to_dict()["error"]
    assert published.read_text() == "published"
    assert evaluation.capture_calls == []
    assert exploration.capture_calls == []


def test_toolkit_reports_unknown_tools_and_invalid_arguments(tmp_path: Path) -> None:
    toolkit = _ContractToolkit(tmp_path)

    unknown = toolkit.execute_tool("missing", {"value": 1})
    invalid = toolkit.execute_tool("read_text_file", {"unexpected": True})

    assert unknown.to_dict() == {"error": "unknown tool: missing"}
    assert "bad arguments for read_text_file" in invalid.to_dict()["error"]
    assert {
        (tuple(error["loc"]), error["type"]) for error in invalid.data["errors"]
    } == {
        (("path",), "missing"),
        (("unexpected",), "extra_forbidden"),
    }
    assert toolkit.capture_calls == []


def test_readonly_declarations_bind_methods_and_nested_partials(tmp_path: Path) -> None:
    toolkit = _ContractToolkit(tmp_path)

    @tool(name="function", readonly=True)
    def readonly_function(value: str) -> ToolResult:
        return ToolResult(data={"value": value})

    class Handler:
        @tool(name="method", readonly=True)
        def readonly_method(self, *, prefix: str, value: str) -> ToolResult:
            return ToolResult(data={"value": prefix + value})

    @tool(name="partial", readonly=True, exclude=("prefix", "value"))
    def bound(*, prefix: str, value: str) -> ToolResult:
        return ToolResult(data={"value": prefix + value})

    handler = Handler()
    toolkit.add_tools(
        [
            readonly_function,
            handler.readonly_method,
            bound.with_handler(partial(partial(bound, prefix="pre-"), value="bound")),
        ]
    )
    assert toolkit.execute_tool("function", {"value": "plain"}).to_dict() == {
        "value": "plain"
    }
    assert toolkit.execute_tool(
        "method", {"prefix": "pre-", "value": "bound"}
    ).to_dict() == {"value": "pre-bound"}
    assert toolkit.execute_tool("partial", {}).to_dict() == {"value": "pre-bound"}
    assert toolkit.capture_calls == toolkit.events.events == []


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

    def move(*, distance: int) -> ToolResult:
        assert distance == 3
        return ToolResult(data=handler_result)

    toolkit.add_tool(tool(move))

    result = toolkit.execute_tool("move", {"distance": 3})

    assert result.to_dict() == {"observation": 1}
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

    assert failed.to_dict()["error"] == "handler exploded"
    assert failed.to_dict()["state_capture_error"] == "capture exploded"
    assert "ValueError: handler exploded" in failed.to_dict()["traceback"]
    assert toolkit.events.events == []
    assert toolkit.execute_tool("probe", {}).to_dict() == {"ready": True}


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
        assert overlap.to_dict() == {"error": "another tool operation is still active"}
    finally:
        release.set()
        worker.join(2)

    assert not worker.is_alive()
    assert worker_errors == []
    assert len(results) == 1
    assert results[0].to_dict() == {"released": True}
    assert (
        toolkit.execute_tool(
            "finish", {"status": "success", "summary": "clean"}
        ).is_error
        is False
    )


def test_toolkit_cleans_up_operation_after_handler_failure(tmp_path: Path) -> None:
    toolkit = _ContractToolkit(tmp_path)

    @tool(readonly=True)
    def fail() -> ToolResult:
        raise RuntimeError("tool failed")

    toolkit.add_tool(fail)

    failed = toolkit.execute_tool("fail", {})

    assert failed.to_dict()["error"] == "tool failed"
    assert "RuntimeError: tool failed" in failed.to_dict()["traceback"]
    assert (
        toolkit.execute_tool(
            "finish", {"status": "failure", "summary": "recovered"}
        ).is_error
        is False
    )


@pytest.mark.timeout(5)
def test_toolkit_cooperatively_cancels_and_cleans_up_active_operation(
    tmp_path: Path,
) -> None:
    toolkit = _ContractToolkit(tmp_path)
    started = threading.Event()
    stop_polling = threading.Event()
    results: list[ToolResult] = []

    def cancellable() -> ToolResult:
        started.set()
        while not stop_polling.wait(0.01):
            toolkit.raise_if_cancelled()
        return ToolResult(data={"unexpected": True})

    toolkit.add_tool(tool(cancellable))
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
    assert results[0].to_dict()["code"] == "tool_cancelled"
    assert results[0].to_dict()["interrupted"] is True
    assert results[0].to_dict()["error"] == "tool operation interrupted"
    assert toolkit.capture_calls[0]["result"]["code"] == "tool_cancelled"
    assert len(toolkit.events.events) == 1
    assert (
        toolkit.execute_tool(
            "finish", {"status": "failure", "summary": "cancelled"}
        ).is_error
        is False
    )


def test_registration_requires_explicit_replacement_and_keeps_bound_owner(tmp_path):
    class Sensor:
        def __init__(self, label: str):
            self.label = label

        @tool(readonly=True)
        def read(self) -> ToolResult:
            return ToolResult(data={"sensor": self.label})

    toolkit = _ContractToolkit(tmp_path)
    first, second = Sensor("first"), Sensor("second")
    toolkit.add_tools(iter_tools(first))
    with pytest.raises(ValueError, match="already registered: read"):
        toolkit.add_tool(second.read)
    assert toolkit.execute_tool("read", {}).data == {"sensor": "first"}
    toolkit.add_tool(second.read, replace=True)
    assert toolkit.execute_tool("read", {}).data == {"sensor": "second"}
    assert toolkit.capture_calls == []
