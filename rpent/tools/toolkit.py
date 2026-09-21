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
during ``__init__`` via :meth:`Toolkit.add_tool`; planners discover declarations
through :meth:`Toolkit.list_tools` and dispatch through :meth:`Toolkit.execute_tool`.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from rpent.dashboard.events import DashboardEventSink, StepRecordEvent
from rpent.tools.base import Tool, ToolResult, iter_tools
from rpent.utils.logging import get_logger
from rpent.utils.templates import substitute

logger = get_logger("toolkit")

if TYPE_CHECKING:
    from rpent.memory.manager import MemoryManager
    from rpent.session import EnvState, StepRecord


@dataclass(slots=True)
class _ToolOperation:
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)


class ToolCancelled(Exception):
    """Raised when an environment reaches a safe cancellation boundary."""


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
        self._tools: dict[str, Tool] = {}
        self._dashboard_events = dashboard_events
        self._state = state
        self._memory = memory
        self._operation_lock = threading.Lock()
        self._active_operation: _ToolOperation | None = None
        self._finish_result: dict[str, Any] | None = None
        self._register_common_tools()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_tool(self, tool: Tool, *, replace: bool = False) -> None:
        """Register a bound declaration, explicitly opting into replacement.

        Args:
            tool: A decorated function or bound instance method.
            replace: Replace an existing declaration, for robot-specific tools
                or execution guards such as an exploration finish check.
        """
        if tool._unbound_method:
            raise TypeError("Register an instance's tool method, not an unbound method")
        if tool.name in self._tools and not replace:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def add_tools(self, tools: Iterable[Tool]) -> None:
        """Register declarations collected from explicitly chosen tool owners."""
        for tool in tools:
            self.add_tool(tool)

    def _register_common_tools(self) -> None:
        from rpent.tools.common import CommonTools

        self.add_tools(iter_tools(CommonTools(memory=self._memory)))

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

    @property
    def finish_result(self) -> dict[str, Any] | None:
        """Return the accepted finish payload, including robot-specific metadata."""
        return dict(self._finish_result) if self._finish_result is not None else None

    def list_tools(self) -> tuple[Tool, ...]:
        """Return the native declarations with run-specific descriptions resolved."""
        return tuple(
            replace(tool, description=substitute(tool.description))
            for tool in self._tools.values()
        )

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Validate one call, execute it, and capture state for advancing tools."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(error=f"unknown tool: {name}")
        try:
            parameters = tool.args_schema.model_validate(input_dict)
        except ValidationError as exc:
            return ToolResult(
                error=f"bad arguments for {name}",
                data={
                    "errors": exc.errors(
                        include_url=False,
                        include_context=False,
                        include_input=False,
                    )
                },
            )
        kwargs = {
            field: getattr(parameters, field) for field in type(parameters).model_fields
        }
        command_input = parameters.model_dump(mode="json")

        with self._operation_lock:
            if self._active_operation is not None:
                return ToolResult(
                    error="another tool operation is still active",
                )
            operation = _ToolOperation()
            self._active_operation = operation

        try:
            started = time.perf_counter()
            try:
                native = tool(**kwargs)
            except ToolCancelled as exc:
                native = ToolResult(
                    error=str(exc),
                    data={"code": "tool_cancelled", "interrupted": True},
                )
            except Exception as exc:
                logger.exception("Tool %s failed", name)
                native = ToolResult(
                    error=str(exc), data={"traceback": traceback.format_exc()}
                )

            if not tool.readonly:
                elapsed_s = round(time.perf_counter() - started, 2)
                result_dict = native.to_dict()
                command = {"action": name, **command_input}
                record: StepRecord | None = None
                try:
                    observed = self.get_env_state(
                        command=command,
                        result=result_dict,
                        elapsed_s=elapsed_s,
                    )
                except Exception as exc:
                    logger.exception("State capture failed after %s", name)
                    native.data["state_capture_error"] = str(exc)
                    if not native.is_error:
                        native.error = f"failed to capture state after {name}: {exc}"
                    native.data.setdefault("traceback", traceback.format_exc())
                else:
                    record = self._state.latest_record()
                    if native.is_error:
                        for key, value in native.data.items():
                            observed.data.setdefault(key, value)
                    native = ToolResult(
                        data=observed.data,
                        images=native.images + observed.images,
                        error=native.error if native.is_error else observed.error,
                    )
                if record is not None:
                    self._publish_step(record)

            try:
                json.dumps(native.to_dict(), allow_nan=False, default=str)
            except (TypeError, ValueError) as exc:
                native = ToolResult(
                    error=f"Tool result serialization failed: {exc}",
                    images=native.images,
                )
            if (
                name == "finish"
                and not native.is_error
                and native.data.get("_finish", True)
            ):
                self._finish_result = {
                    key: value for key, value in native.data.items() if key != "_finish"
                }
            return native
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
            )
        )

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> ToolResult:
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
