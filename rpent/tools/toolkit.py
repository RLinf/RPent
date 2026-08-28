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

"""Base class for agent tools.

``Toolkit`` is the agent-facing tool container. Subclasses can register tools
during ``__init__`` via :meth:`Toolkit.add_tool`; the planner calls the tools through :meth:`Toolkit.get_tools_spec` and
:meth:`Toolkit.execute_tool`.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

from rpent.dashboard.events import DashboardEventSink, StepRecordEvent
from rpent.utils.templates import substitute

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager
    from rpent.session import EnvState, StepRecord


@dataclass(slots=True)
class _ToolOperation:
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)


class ToolCancelled(Exception):
    """Raised when an environment reaches a safe cancellation boundary."""


def readonly(func):
    """Mark a tool handler as not advancing environment state.

    Tool handlers capture a fresh observation (:meth:`Toolkit.get_env_state`)
    by default. Apply this marker to observational and file/IO tools that do
    not move the robot or otherwise change the environment.
    """
    func._readonly = True
    return func


def _is_readonly(handler: Callable[..., Any]) -> bool:
    """Whether ``handler`` was marked with :func:`readonly`."""
    target = handler
    while isinstance(target, partial):
        target = target.func
    target = getattr(target, "__func__", target)
    return bool(getattr(target, "_readonly", False))


@dataclass
class ToolResult:
    """Result of executing one tool call.

    Carries the raw result dict (for logging and finish-signal detection)
    alongside the Anthropic-shaped content blocks the LLM consumes.
    """

    name: str
    result: dict[str, Any]
    call_id: str | None = None

    content_blocks: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    is_finish: bool = field(default=False, init=False)

    #: Max bytes of the text block emitted in :attr:`content_blocks`.
    MAX_TEXT_BYTES_IN_RESULT: ClassVar[int] = 60000

    def __post_init__(self) -> None:
        self.content_blocks = self._build_content_blocks()
        self.is_finish = bool(
            isinstance(self.result, dict) and self.result.get("_finish")
        )

    def _build_content_blocks(self) -> list[dict[str, Any]]:
        """Build Anthropic-shaped content blocks (text + optional images).

        Strips image byte payloads from the text block and emits them as
        separate base64 image blocks so the LLM receives the state images as
        multimodal content.
        """
        result = self.result
        if not isinstance(result, dict):
            return [
                {"type": "text", "text": str(result)[: self.MAX_TEXT_BYTES_IN_RESULT]}
            ]

        result_for_text = dict(result)
        image = result_for_text.pop("_image_bytes", None)
        image_cam = result_for_text.pop("_image_cam_bytes", None)
        image_nav = result_for_text.pop("_image_nav_bytes", None)
        image_wrist = result_for_text.pop("_image_wrist_bytes", None)
        text = json.dumps(result_for_text, indent=2, default=str)
        if len(text) > self.MAX_TEXT_BYTES_IN_RESULT:
            text = text[: self.MAX_TEXT_BYTES_IN_RESULT] + "\n[truncated]"

        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]

        def _add_image_bytes(data_bytes: bytes) -> None:
            data = base64.b64encode(data_bytes).decode("utf-8")
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": data,
                    },
                }
            )

        if image:
            _add_image_bytes(image)
        if image_cam:
            _add_image_bytes(image_cam)
        if image_nav:
            _add_image_bytes(image_nav)
        if image_wrist:
            _add_image_bytes(image_wrist)
        return blocks


class Toolkit:
    """Base toolkit: registers common tools and dispatches tool calls.

    Subclasses extend ``__init__`` (calling ``super().__init__()`` first)
    and register additional tools with :meth:`add_tool`. Robot-specific
    subclasses receive their env/model/etc. as constructor arguments and
    build the underlying env Primitives in ``__init__``; the toolkit
    base class only contributes the common file/IO tools. Override
    :meth:`close` to release robot-side primitives / servers at the end of the run.
    """

    def __init__(
        self,
        *,
        dashboard_events: DashboardEventSink,
        state: Any = None,
        memory: "MemoryManager",
    ) -> None:
        self._tools: dict[
            str,
            tuple[dict[str, Any], Callable[..., Any]],
        ] = {}
        self._dashboard_events = dashboard_events
        self._state = state
        self._memory = memory
        self._operation_lock = threading.Lock()
        self._active_operation: _ToolOperation | None = None
        self._register_common_tools()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_tool(
        self,
        name: str,
        spec: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """Register one tool under ``name`` with its schema and handler.

        Args:
            name: Tool name as the LLM sees it (e.g. ``"read_text_file"``).
            spec: Anthropic-shaped tool schema dict (``name``,
                ``description``, ``input_schema``).
            handler: Callable invoked with the tool's input kwargs; returns
                a result dict. Decorate read-only handlers with
                :func:`readonly`; all other handlers capture state.
        """
        self._tools[name] = (spec, handler)

    def _register_common_tools(self) -> None:
        """Register the file/IO tools shared by every run."""
        from rpent.tools import common

        memory_bindings = self._memory.get_common_tool_bindings()
        for spec in common.TOOLS_SPEC:
            name = spec["name"]
            binding = memory_bindings.get(name)
            if binding is None:
                binding = (spec, common.TOOL_HANDLERS[name])
            tool_spec, handler = binding
            self.add_tool(name, tool_spec, handler)

    # ------------------------------------------------------------------
    # Planner-facing API
    # ------------------------------------------------------------------

    @property
    def memory(self) -> "MemoryManager":
        """Return the toolkit's memory manager."""
        return self._memory

    @property
    def state(self) -> EnvState:
        """Return the run's artifact and step store."""
        if self._state is None:
            raise RuntimeError("toolkit has no environment state")
        return self._state

    def get_tools_spec(self) -> list[dict[str, Any]]:
        """Return the tool schemas the LLM sees."""
        return substitute([spec for spec, _ in self._tools.values()])

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Dispatch a tool call to its registered handler."""
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(name=name, result={"error": f"unknown tool: {name}"})
        _, handler = entry

        with self._operation_lock:
            if self._active_operation is not None:
                return ToolResult(
                    name=name,
                    result={"error": "another tool operation is still active"},
                )
            operation = _ToolOperation()
            self._active_operation = operation

        try:
            started = time.perf_counter()
            failed = False
            try:
                result = handler(**input_dict)
            except TypeError as e:
                result = {
                    "error": f"bad arguments for {name}: {e}",
                    "got": input_dict,
                }
                failed = True
            except ToolCancelled as e:
                result = {
                    "error": str(e),
                    "code": "tool_cancelled",
                    "interrupted": True,
                }
                failed = True
            except Exception as e:
                result = {"error": str(e), "traceback": traceback.format_exc()}
                failed = True

            if not _is_readonly(handler):
                elapsed_s = round(time.perf_counter() - started, 2)
                result_dict = result if isinstance(result, dict) else {"value": result}
                command = {"action": name, **input_dict}
                record: StepRecord | None = None
                try:
                    captured = self.get_env_state(
                        command=command,
                        result=result_dict,
                        elapsed_s=elapsed_s,
                    )
                except Exception as e:
                    captured = result_dict
                    captured["state_capture_error"] = str(e)
                    captured.setdefault(
                        "error", f"failed to capture state after {name}: {e}"
                    )
                    captured.setdefault("traceback", traceback.format_exc())
                else:
                    record = self._state.latest_record()
                result = captured
                if failed:
                    for key, value in result_dict.items():
                        result.setdefault(key, value)
                if record is not None:
                    self._publish_step(record)

            return ToolResult(name=name, result=result)
        finally:
            with self._operation_lock:
                self._active_operation = None
                operation.done_event.set()

    def _publish_step(self, record: StepRecord) -> None:
        """Publish one recorded environment step to the dashboard sink."""
        self._dashboard_events.emit(
            StepRecordEvent(
                record=record,
                env_state=self._state,
                frame_artifacts=dict(getattr(type(self), "_FRAME_ARTIFACTS", {})),
            )
        )

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        """Capture and return the observation produced by a stateful tool."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Server lifecycle hooks (overridden by robot toolkits)
    # ------------------------------------------------------------------

    def cancel_active_and_wait(self) -> None:
        """Request cancellation and wait for the active tool to return."""
        with self._operation_lock:
            operation = self._active_operation
            if operation is None:
                return
            operation.cancel_event.set()
        operation.done_event.wait()

    def raise_if_cancelled(self) -> None:
        """Raise at an environment-defined safe cancellation boundary."""
        with self._operation_lock:
            operation = self._active_operation
        if operation is not None and operation.cancel_event.is_set():
            raise ToolCancelled("tool operation interrupted")

    def close(self) -> None:
        """Release the robot-side primitives / servers at end of run. Default: no-op."""

    def solved(self) -> bool:
        """Whether the env has reported the task complete.

        Ground truth for the session loop: an agent may call ``finish`` with
        ``status="success"`` on a cell it did not actually finish, so the
        handoff decision reads the environment, not the agent.
        """
        raise NotImplementedError

    def write_recipe(self, recipe_tag: str) -> str | None:
        """Write a replay recipe for this robot, if supported."""
        return None
