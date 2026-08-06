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
from typing import TYPE_CHECKING, Any, ClassVar

from rpent.dashboard.events import DashboardEventSink, StepRecordEvent
from rpent.utils.templates import substitute

if TYPE_CHECKING:
    from rpent.tools.state import StepRecord


@dataclass(slots=True)
class _ToolOperation:
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)


class ToolCancelled(Exception):
    """Raised when an environment reaches a safe cancellation boundary."""


def updatestate(func):
    """Mark a tool handler as one that advances environment state.

    The toolkit captures a fresh observation (:meth:`Toolkit.get_env_state`)
    after a handler carrying this marker runs (or raises). Apply it to
    primitive-driver methods that move the robot; read-only tools (file IO,
    ``view_env_state``, ``back_project``, ``segment``, ...) are left
    unmarked and skip state capture.

    The marker is read off the underlying function, so registering a bound
    method (``getattr(self._driver, name)``) inherits it automatically.
    ``functools.partial`` does not delegate attribute access, so a partial
    wrapping a marked method is treated as read-only (intentional -- see the
    LIBERO ``segment`` tool).
    """
    func._updates_state = True
    return func


def _updates_state(handler: Callable[..., Any]) -> bool:
    """Whether ``handler`` was marked with :func:`updatestate`.

    Resolves through ``__func__`` so bound methods (the common case for
    primitive-driver tools) report the marker set on their underlying
    function. Plain functions, closures, and ``functools.partial`` objects
    do not delegate, so undecorated read-only handlers read as ``False``.
    """
    target = getattr(handler, "__func__", handler)
    return bool(getattr(target, "_updates_state", False))


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
        if image_wrist:
            _add_image_bytes(image_wrist)
        return blocks


class Toolkit:
    """Base toolkit: registers common tools and dispatches tool calls.

    Subclasses extend ``__init__`` (calling ``super().__init__()`` first)
    and register additional tools with :meth:`add_tool`. Env-specific
    subclasses receive their env/model/etc. as constructor arguments and
    build the underlying LiberoPrimitives in ``__init__``; the toolkit
    base class only contributes the common file/IO tools. Override
    :meth:`close` to release env-side primitives / servers at the end of the run.
    """

    def __init__(
        self,
        *,
        dashboard_events: DashboardEventSink,
        state: Any = None,
    ) -> None:
        self._tools: dict[
            str,
            tuple[dict[str, Any], Callable[..., Any], bool],
        ] = {}
        self._dashboard_events = dashboard_events
        self._state = state
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
                a result dict. Decorate state-advancing primitives with
                :func:`updatestate`.
        """
        self._tools[name] = (spec, handler, _updates_state(handler))

    def _register_common_tools(self) -> None:
        """Register the file/IO tools shared by every run."""
        from rpent.tools import common

        for spec in common.TOOLS_SPEC:
            name = spec["name"]
            self.add_tool(name, spec, common.TOOL_HANDLERS[name])

    # ------------------------------------------------------------------
    # Planner-facing API
    # ------------------------------------------------------------------

    def get_tools_spec(self) -> list[dict[str, Any]]:
        """Return the tool schemas the LLM sees."""
        return substitute(
            [spec for spec, _, _ in self._tools.values()]
        )

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Dispatch a tool call to its registered handler."""
        entry = self._tools.get(name)
        if entry is None:
            return self._finish_tool(name, {"error": f"unknown tool: {name}"})
        _, handler, captures_state = entry

        with self._operation_lock:
            if self._active_operation is not None:
                return self._finish_tool(
                    name,
                    {"error": "another tool operation is still active"},
                )
            operation = _ToolOperation()
            self._active_operation = operation

        try:
            started = time.perf_counter()
            failed = False
            try:
                result = handler(**input_dict)
            except TypeError as e:
                return self._finish_tool(
                    name,
                    {"error": f"bad arguments for {name}: {e}", "got": input_dict},
                )
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

            if captures_state:
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
                    result.setdefault("error", result_dict["error"])
                    if "traceback" in result_dict:
                        result.setdefault("traceback", result_dict["traceback"])
                if record is not None:
                    self._publish_step(record)

            return self._finish_tool(name, result)
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

    def _finish_tool(self, name: str, result: Any) -> ToolResult:
        return ToolResult(name=name, result=result)

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
        """Release the env-side primitives / servers at end of run. Default: no-op."""

    def write_recipe(self, recipe_tag: str) -> str | None:
        """Write a replay recipe for this env, if supported."""
        return None
