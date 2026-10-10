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


@dataclass(eq=False)
class _ToolOperation:
    readonly: bool
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done: bool = False


class _Scheduler:
    """Share readonly calls; run queued exclusive calls in arrival order.

    A plain RWLock only provides mutual exclusion. Tool calls also need ordered
    writers, cancellation of queued and active calls, and pause/close with drain.
    One condition keeps admission and these lifecycle transitions synchronized.
    """

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.pending: list[_ToolOperation] = []
        self.active: set[_ToolOperation] = set()
        self.paused = False
        self.closed = False

    def acquire(self, *, readonly: bool) -> _ToolOperation:
        with self.condition:
            if self.closed:
                raise ToolCancelled("Toolkit is closed.")
            if self.paused:
                raise ToolCancelled("Tool calls are paused.")
            call = _ToolOperation(readonly=readonly)
            self.pending.append(call)
            self.condition.notify_all()
            try:
                while True:
                    if call.cancel_event.is_set():
                        raise ToolCancelled("tool operation interrupted")
                    exclusive = next(
                        (item for item in self.pending if not item.readonly), None
                    )
                    if (
                        readonly
                        and exclusive is None
                        and all(item.readonly for item in self.active)
                    ) or (not readonly and not self.active and call is exclusive):
                        self.pending.remove(call)
                        self.active.add(call)
                        return call
                    self.condition.wait()
            except BaseException:
                self.pending.remove(call)
                call.done = True
                self.condition.notify_all()
                raise

    def release(self, call: _ToolOperation) -> None:
        with self.condition:
            self.active.remove(call)
            call.done = True
            self.condition.notify_all()

    def cancel(self, *, close: bool = False) -> list[_ToolOperation]:
        with self.condition:
            self.paused = True
            self.closed |= close
            calls = [*self.pending, *self.active]
            for call in calls:
                call.cancel_event.set()
            self.condition.notify_all()
            return calls

    def wait(self, calls: list[_ToolOperation]) -> None:
        with self.condition:
            self.condition.wait_for(lambda: all(call.done for call in calls))

    def resume(self) -> None:
        with self.condition:
            if self.active or self.pending:
                raise RuntimeError("Wait for tool cleanup before resuming.")
            if not self.closed:
                self.paused = False


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
        self._scheduler = _Scheduler()
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

        self._memory.check_layout()
        for declaration in iter_tools(CommonTools(memory=self._memory)):
            if declaration.name in {"read_text_file", "list_dir"}:
                declaration = replace(
                    declaration,
                    description=declaration.description
                    + self._memory.file_tool_description,
                )
            self.add_tool(declaration)

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
            return ToolResult(data={"error": f"unknown tool: {name}"})
        try:
            parameters = tool.args_schema.model_validate(input_dict, strict=True)
        except ValidationError as exc:
            return ToolResult(
                data={
                    "error": f"bad arguments for {name}",
                    "errors": exc.errors(
                        include_url=False,
                        include_context=False,
                        include_input=False,
                    ),
                }
            )
        kwargs = {
            field: getattr(parameters, field) for field in type(parameters).model_fields
        }

        try:
            operation = self._scheduler.acquire(readonly=tool.readonly)
        except ToolCancelled as exc:
            return ToolResult(
                data={"error": str(exc), "code": "tool_cancelled", "interrupted": True}
            )

        try:
            started = time.perf_counter()
            try:
                self.raise_if_cancelled()
                native = tool(**kwargs)
            except TypeError as exc:
                native = ToolResult(
                    data={
                        "error": f"bad arguments for {name}: {exc}",
                        "got": input_dict,
                    }
                )
            except ToolCancelled as exc:
                native = ToolResult(
                    data={
                        "error": str(exc),
                        "code": "tool_cancelled",
                        "interrupted": True,
                    }
                )
            except Exception as exc:
                logger.exception("Tool %s failed", name)
                native = ToolResult(
                    data={"error": str(exc), "traceback": traceback.format_exc()}
                )

            if not tool.readonly:
                elapsed_s = round(time.perf_counter() - started, 2)
                result_dict = native.to_dict()
                command = {"action": name, **input_dict}
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
                        native.data["error"] = (
                            f"failed to capture state after {name}: {exc}"
                        )
                    native.data.setdefault("traceback", traceback.format_exc())
                else:
                    record = self._state.latest_record()
                    if native.is_error:
                        for key, value in native.data.items():
                            observed.data.setdefault(key, value)
                        observed.data["error"] = native.data["error"]
                    native = ToolResult(
                        data=observed.data,
                        images=native.images + observed.images,
                    )
                if record is not None:
                    self._publish_step(record)

            try:
                json.dumps(native.to_dict(), allow_nan=False, default=str)
            except (TypeError, ValueError) as exc:
                native = ToolResult(
                    data={"error": f"Tool result serialization failed: {exc}"},
                    images=native.images,
                )
            return native
        finally:
            self._scheduler.release(operation)

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
        """Pause admission, cancel queued and active calls, and wait for cleanup."""
        self._scheduler.wait(self._scheduler.cancel())

    def resume_calls(self) -> None:
        """Reopen admission after interrupted calls and their workers have drained."""
        self._scheduler.resume()

    def raise_if_cancelled(self) -> None:
        """Raise at an environment-defined safe cancellation boundary."""
        with self._scheduler.condition:
            if any(call.cancel_event.is_set() for call in self._scheduler.active):
                raise ToolCancelled("tool operation interrupted")

    def close(self) -> None:
        """Close admission and drain calls before subclasses release resources."""
        self._scheduler.cancel(close=True)
        self.cancel_active_and_wait()

    def exploration_continuation(self, *, explicit: bool = False) -> str | None:
        """Return a rule-checked continuation at an idle planner boundary."""
        return None

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
