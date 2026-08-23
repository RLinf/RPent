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
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

from rpent.dashboard.events import DashboardEventSink, StepRecordEvent
from rpent.utils.templates import substitute

if TYPE_CHECKING:
    from rpent.tools.state import EnvState, StepRecord


@dataclass(slots=True, eq=False)
class _ToolOperation:
    is_parallel: bool
    cancel_event: threading.Event = field(default_factory=threading.Event)
    cancelled_before_start: bool = False


class ToolCancelled(Exception):
    """Raised when an environment reaches a safe cancellation boundary."""


def readonly(func):
    """Mark a tool handler as not advancing environment state."""
    func._readonly = True
    return func


def parallel(func):
    """Mark a read-only tool handler as safe to run with other readers."""
    func._parallel = True
    return func


def _tool_policy(handler: Callable[..., Any]) -> tuple[bool, bool]:
    """Return ``(is_parallel, updates_env)`` for a registered handler."""
    target = handler
    while isinstance(target, partial):
        target = target.func
    target = getattr(target, "__func__", target)
    is_parallel = bool(getattr(target, "_parallel", False))
    updates_env = not bool(getattr(target, "_readonly", False))
    return is_parallel, updates_env


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
            return [{"type": "text", "text": str(result)[:self.MAX_TEXT_BYTES_IN_RESULT]}]

        result_for_text = dict(result)
        image = result_for_text.pop("_image_bytes", None)
        image_cam = result_for_text.pop("_image_cam_bytes", None)
        image_nav = result_for_text.pop("_image_nav_bytes", None)
        image_wrist = result_for_text.pop("_image_wrist_bytes", None)
        text = json.dumps(result_for_text, indent=2, default=str)
        if len(text) > self.MAX_TEXT_BYTES_IN_RESULT:
            text = text[:self.MAX_TEXT_BYTES_IN_RESULT] + "\n[truncated]"

        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]

        def _add_image_bytes(data_bytes: bytes) -> None:
            data = base64.b64encode(data_bytes).decode("utf-8")
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": data,
                },
            })

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
    and register additional tools with :meth:`add_tool`. Env-specific
    subclasses receive their env/model/etc. as constructor arguments and
    build the underlying env Primitives in ``__init__``; the toolkit
    base class only contributes the common file/IO tools. Override
    :meth:`close` to release env-side primitives / servers at the end of the run,
    and call ``super().close()`` before releasing them.
    """

    def __init__(
        self,
        *,
        dashboard_events: DashboardEventSink,
        state: Any = None,
    ) -> None:
        self._tools: dict[
            str,
            tuple[dict[str, Any], Callable[..., Any]],
        ] = {}
        self._dashboard_events = dashboard_events
        self._state = state
        self._operation_condition = threading.Condition()
        self._admission_queue: deque[_ToolOperation] = deque()
        # Includes both queued and running calls so cancellation can drain all
        # calls admitted before the gate closed.
        self._registered_operations: set[_ToolOperation] = set()
        self._running_readers = 0
        self._active_exclusive_operation: _ToolOperation | None = None
        self._admission_paused = False
        self._closed = False
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
                a result dict. Decorate handlers that do not advance the
                environment with :func:`readonly`, and add :func:`parallel`
                when they are also safe to run concurrently. All other
                handlers capture state after execution.
        """
        is_parallel, updates_env = _tool_policy(handler)
        if is_parallel and updates_env:
            raise ValueError(
                f"parallel tool {name!r} must also be marked readonly"
            )
        self._tools[name] = (spec, handler)

    def _register_common_tools(self) -> None:
        """Register the file/IO tools shared by every run."""
        from rpent.tools import common

        for spec in common.TOOLS_SPEC:
            name = spec["name"]
            self.add_tool(name, spec, common.TOOL_HANDLERS[name])

    # ------------------------------------------------------------------
    # Planner-facing API
    # ------------------------------------------------------------------

    @property
    def state(self) -> EnvState:
        """Return the run's artifact and step store."""
        if self._state is None:
            raise RuntimeError("toolkit has no environment state")
        return self._state

    def get_tools_spec(self) -> list[dict[str, Any]]:
        """Return the tool schemas the LLM sees."""
        return substitute(
            [spec for spec, _ in self._tools.values()]
        )

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Dispatch a tool call to its registered handler."""
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(name=name, result={"error": f"unknown tool: {name}"})
        _, handler = entry
        is_parallel, updates_env = _tool_policy(handler)
        operation = self._begin_operation(is_parallel=is_parallel)
        if operation is None:
            return ToolResult(
                name=name,
                result={
                    "error": "tool operation interrupted",
                    "code": "tool_cancelled",
                    "interrupted": True,
                },
            )

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

            if updates_env:
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
            self._end_operation(operation)

    def _begin_operation(self, *, is_parallel: bool) -> _ToolOperation | None:
        """Register this call and wait for FIFO admission."""
        with self._operation_condition:
            if self._admission_paused or self._closed:
                return None

            operation = _ToolOperation(is_parallel=is_parallel)
            self._registered_operations.add(operation)
            self._admission_queue.append(operation)
            self._operation_condition.wait_for(
                lambda: operation.cancelled_before_start
                or self._can_start_operation(operation)
            )

            if operation.cancelled_before_start:
                self._registered_operations.discard(operation)
                self._operation_condition.notify_all()
                return None

            self._admission_queue.popleft()
            if operation.is_parallel:
                self._running_readers += 1
            else:
                self._active_exclusive_operation = operation
            # Wake the next queued reader so a consecutive reader group can
            # enter without waiting for the first reader to finish.
            self._operation_condition.notify_all()
            return operation

    def _can_start_operation(self, operation: _ToolOperation) -> bool:
        """Return whether the FIFO head can enter the running set."""
        if (
            not self._admission_queue
            or self._admission_queue[0] is not operation
        ):
            return False
        if self._active_exclusive_operation is not None:
            return False
        return operation.is_parallel or self._running_readers == 0

    def _end_operation(self, operation: _ToolOperation) -> None:
        with self._operation_condition:
            if operation.is_parallel:
                self._running_readers -= 1
            else:
                if self._active_exclusive_operation is operation:
                    self._active_exclusive_operation = None
            self._registered_operations.discard(operation)
            self._operation_condition.notify_all()

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
    # Server lifecycle hooks (overridden by env toolkits)
    # ------------------------------------------------------------------

    def cancel_active_and_wait(self) -> None:
        """Pause admission, cancel queued calls, and drain running calls."""
        with self._operation_condition:
            self._admission_paused = True
            for operation in self._admission_queue:
                operation.cancelled_before_start = True
            self._admission_queue.clear()
            if self._active_exclusive_operation is not None:
                self._active_exclusive_operation.cancel_event.set()
            self._operation_condition.notify_all()
            self._operation_condition.wait_for(
                lambda: not self._registered_operations
            )

    def resume_operations(self) -> None:
        """Resume admission after a successful interrupt of the same run."""
        with self._operation_condition:
            if self._closed or not self._admission_paused:
                return
            if self._registered_operations:
                raise RuntimeError("cannot resume toolkit before operations drain")
            self._admission_paused = False

    def raise_if_cancelled(self) -> None:
        """Raise at an environment-defined safe cancellation boundary."""
        with self._operation_condition:
            operation = self._active_exclusive_operation
        if operation is not None and operation.cancel_event.is_set():
            raise ToolCancelled("tool operation interrupted")

    def close(self) -> None:
        """Permanently stop tool admission and drain this toolkit."""
        with self._operation_condition:
            self._closed = True
        self.cancel_active_and_wait()

    def write_recipe(self, recipe_tag: str) -> str | None:
        """Write a replay recipe for this env, if supported."""
        return None
