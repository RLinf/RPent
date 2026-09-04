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
from typing import Any

import pytest

from rpent.dashboard.events import StepRecordEvent
from rpent.memory import MemoryManager
from rpent.memory import tools as memory_tools
from rpent.session import EnvState
from rpent.tools import common
from rpent.tools.toolkit import Toolkit, ToolResult, readonly


class _RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    @property
    def enabled(self) -> bool:
        return True

    def emit(self, event: Any) -> None:
        self.events.append(event)


class _ContractToolkit(Toolkit):
    _FRAME_ARTIFACTS = {"primary": "frame.png"}

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
    ) -> dict[str, Any]:
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
        return {"observation": len(self.capture_calls)}

    def solved(self) -> bool:
        return False


def test_tool_result_builds_text_and_images_without_mutating_result() -> None:
    image_payloads = {
        "_image_bytes": b"main",
        "_image_cam_bytes": b"camera",
        "_image_nav_bytes": b"navigation",
        "_image_wrist_bytes": b"wrist",
    }
    result = {"status": "ok", "count": 2, **image_payloads}
    original = copy.deepcopy(result)

    tool_result = ToolResult(name="observe", result=result, call_id="call-1")

    assert result == original
    assert tool_result.call_id == "call-1"
    assert tool_result.is_finish is False
    assert json.loads(tool_result.content_blocks[0]["text"]) == {
        "status": "ok",
        "count": 2,
    }
    assert [block["type"] for block in tool_result.content_blocks] == [
        "text",
        "image",
        "image",
        "image",
        "image",
    ]
    assert [
        base64.b64decode(block["source"]["data"])
        for block in tool_result.content_blocks[1:]
    ] == list(image_payloads.values())
    assert all(
        block["source"]["media_type"] == "image/png"
        for block in tool_result.content_blocks[1:]
    )


@pytest.mark.parametrize(
    ("raw_result", "expected_text"),
    [
        ("plain text", "plain text"),
        (17, "17"),
        (["one", "two"], "['one', 'two']"),
    ],
)
def test_tool_result_converts_ordinary_results_to_text(
    raw_result: Any,
    expected_text: str,
) -> None:
    tool_result = ToolResult(name="ordinary", result=raw_result)

    assert tool_result.content_blocks == [{"type": "text", "text": expected_text}]


def test_tool_result_recognizes_finish_only_from_truthy_dict_sentinel() -> None:
    assert ToolResult("finish", {"_finish": True}).is_finish is True
    assert ToolResult("finish", {"_finish": False}).is_finish is False
    assert ToolResult("finish", "finished").is_finish is False


@pytest.mark.parametrize(
    ("raw_result", "expected_plain_text"),
    [
        pytest.param(
            "界" * 10,
            "界" * 6,
            id="ordinary-unicode-text",
        ),
        pytest.param(
            {"value": "界" * 10},
            None,
            id="unicode-json-dict",
        ),
    ],
)
def test_tool_result_text_limit_counts_utf8_bytes(
    monkeypatch: pytest.MonkeyPatch,
    raw_result: Any,
    expected_plain_text: str | None,
) -> None:
    monkeypatch.setattr(ToolResult, "MAX_TEXT_BYTES_IN_RESULT", 20)

    text = ToolResult(name="unicode", result=raw_result).content_blocks[0]["text"]

    assert len(text.encode("utf-8")) <= 20
    assert text.encode("utf-8").decode("utf-8") == text
    if expected_plain_text is not None:
        assert text == expected_plain_text


def test_toolkit_registers_common_specs_with_fresh_placeholder_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run-output"
    original_specs = copy.deepcopy(common.TOOLS_SPEC)
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: output_dir)
    toolkit = _ContractToolkit(tmp_path / "state")

    first = toolkit.get_tools_spec()
    second = toolkit.get_tools_spec()

    assert [spec["name"] for spec in first] == [
        "read_text_file",
        "write_text_file",
        "list_dir",
        "finish",
    ]
    list_dir_spec = next(spec for spec in first if spec["name"] == "list_dir")
    assert str(output_dir) in list_dir_spec["description"]
    assert memory_tools.MEMORY_BOUNDARY_NOTE in list_dir_spec["description"]
    assert (
        str(output_dir)
        in list_dir_spec["input_schema"]["properties"]["path"]["description"]
    )
    assert common.TOOLS_SPEC == original_specs
    assert first == second
    assert first is not second
    assert first[0] is not common.TOOLS_SPEC[0]


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

    assert written.result == {
        "path": str(text_file),
        "bytes_written": len("hello 世界".encode()),
    }
    assert read.result["path"] == str(text_file)
    assert read.result["size"] == len("hello 世界")
    assert read.result["content"].startswith("hello 世")
    assert "[TRUNCATED" in read.result["content"]
    assert listed.result == {
        "path": str(text_file.parent),
        "count": 1,
        "files": ["note.txt"],
    }
    assert finished.result == {
        "_finish": True,
        "status": "success",
        "summary": "done",
    }
    assert finished.is_finish is True
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
    monkeypatch.setattr(memory_tools, "get_repo_root", lambda: repo_root)
    monkeypatch.setattr(common, "get_repo_root", lambda: repo_root)

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
    assert read_published.result["content"] == "published"
    assert read_root_leaf.result["content"] == "root-level note"
    assert list_published.result["files"] == ["strategy.md"]
    assert "writing to memory is denied" in write_published.result["error"]
    assert "another robot's memory is denied" in read_foreign.result["error"]
    assert "reading this memory path is denied" in read_evaluation_inbox.result["error"]

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

    assert write_own.result["bytes_written"] == 5
    assert read_own.result["content"] == "draft"
    assert "reading this memory path is denied" in read_other.result["error"]
    assert "writing to memory is denied" in write_through_symlink.result["error"]
    assert published.read_text() == "published"
    assert evaluation.capture_calls == []
    assert exploration.capture_calls == []


def test_toolkit_reports_unknown_tools_and_invalid_arguments(tmp_path: Path) -> None:
    toolkit = _ContractToolkit(tmp_path)

    unknown = toolkit.execute_tool("missing", {"value": 1})
    invalid = toolkit.execute_tool("read_text_file", {"unexpected": True})

    assert unknown.result == {"error": "unknown tool: missing"}
    assert "bad arguments for read_text_file" in invalid.result["error"]
    assert invalid.result["got"] == {"unexpected": True}
    assert toolkit.capture_calls == []


def test_readonly_marker_handles_functions_bound_methods_and_nested_partials(
    tmp_path: Path,
) -> None:
    toolkit = _ContractToolkit(tmp_path)

    @readonly
    def readonly_function(value: str) -> dict[str, str]:
        return {"value": value}

    class Handler:
        @readonly
        def readonly_method(self, *, prefix: str, value: str) -> dict[str, str]:
            return {"value": prefix + value}

    handler = Handler()
    toolkit.add_tool("function", {"name": "function"}, readonly_function)
    toolkit.add_tool("method", {"name": "method"}, handler.readonly_method)
    toolkit.add_tool(
        "partial",
        {"name": "partial"},
        partial(partial(handler.readonly_method, prefix="pre-"), value="bound"),
    )

    assert toolkit.execute_tool("function", {"value": "plain"}).result == {
        "value": "plain"
    }
    assert toolkit.execute_tool(
        "method", {"prefix": "pre-", "value": "bound"}
    ).result == {"value": "pre-bound"}
    assert toolkit.execute_tool("partial", {}).result == {"value": "pre-bound"}
    assert toolkit.capture_calls == []
    assert toolkit.events.events == []


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

    def move(*, distance: int) -> dict[str, bool]:
        assert distance == 3
        return handler_result

    toolkit.add_tool("move", {"name": "move"}, move)

    result = toolkit.execute_tool("move", {"distance": 3})

    assert result.result == {"observation": 1}
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
    assert event.frame_artifacts == {"primary": "frame.png"}


def test_handler_error_is_retained_when_state_capture_also_fails(
    tmp_path: Path,
) -> None:
    toolkit = _ContractToolkit(tmp_path)
    toolkit.capture_error = RuntimeError("capture exploded")

    def fail() -> dict[str, Any]:
        raise ValueError("handler exploded")

    @readonly
    def probe() -> dict[str, bool]:
        return {"ready": True}

    toolkit.add_tool("fail", {"name": "fail"}, fail)
    toolkit.add_tool("probe", {"name": "probe"}, probe)

    failed = toolkit.execute_tool("fail", {})

    assert failed.result["error"] == "handler exploded"
    assert failed.result["state_capture_error"] == "capture exploded"
    assert "ValueError: handler exploded" in failed.result["traceback"]
    assert toolkit.events.events == []
    assert toolkit.execute_tool("probe", {}).result == {"ready": True}


@pytest.mark.timeout(5)
def test_toolkit_rejects_overlapping_operations_and_cleans_up_after_success(
    tmp_path: Path,
) -> None:
    toolkit = _ContractToolkit(tmp_path)
    started = threading.Event()
    release = threading.Event()
    results: list[ToolResult] = []
    worker_errors: list[BaseException] = []

    @readonly
    def blocking() -> dict[str, bool]:
        started.set()
        assert release.wait(2), "test did not release the blocking handler"
        return {"released": True}

    toolkit.add_tool("blocking", {"name": "blocking"}, blocking)

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
        assert overlap.result == {"error": "another tool operation is still active"}
    finally:
        release.set()
        worker.join(2)

    assert not worker.is_alive()
    assert worker_errors == []
    assert len(results) == 1
    assert results[0].result == {"released": True}
    assert toolkit.execute_tool(
        "finish", {"status": "success", "summary": "clean"}
    ).is_finish


def test_toolkit_cleans_up_operation_after_handler_failure(tmp_path: Path) -> None:
    toolkit = _ContractToolkit(tmp_path)

    @readonly
    def fail() -> dict[str, Any]:
        raise RuntimeError("tool failed")

    toolkit.add_tool("fail", {"name": "fail"}, fail)

    failed = toolkit.execute_tool("fail", {})

    assert failed.result["error"] == "tool failed"
    assert "RuntimeError: tool failed" in failed.result["traceback"]
    assert toolkit.execute_tool(
        "finish", {"status": "failure", "summary": "recovered"}
    ).is_finish


@pytest.mark.timeout(5)
def test_toolkit_cooperatively_cancels_and_cleans_up_active_operation(
    tmp_path: Path,
) -> None:
    toolkit = _ContractToolkit(tmp_path)
    started = threading.Event()
    stop_polling = threading.Event()
    results: list[ToolResult] = []

    def cancellable() -> dict[str, bool]:
        started.set()
        while not stop_polling.wait(0.01):
            toolkit.raise_if_cancelled()
        return {"unexpected": True}

    toolkit.add_tool("cancellable", {"name": "cancellable"}, cancellable)
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
    assert results[0].result["code"] == "tool_cancelled"
    assert results[0].result["interrupted"] is True
    assert results[0].result["error"] == "tool operation interrupted"
    assert toolkit.capture_calls[0]["result"]["code"] == "tool_cancelled"
    assert len(toolkit.events.events) == 1
    assert toolkit.execute_tool(
        "finish", {"status": "failure", "summary": "cancelled"}
    ).is_finish
