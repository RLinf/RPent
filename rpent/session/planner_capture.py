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

"""Capture planner tasks, model calls, and tool feedback."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter
from pydantic_ai import ModelRetry, RunContext, ToolCallPart
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_core import to_jsonable_python

from rpent.session import EnvState, MissingInfo, Payload
from rpent.session.env_capture import ToolCallCapture
from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from rpent.planner.base import PlannerResult

logger = get_logger("task_capture")

SDK_VERSION = version("pydantic-ai-slim")
_REQUEST_PARAMETERS = TypeAdapter(ModelRequestParameters)
_SDK_VALUES = TypeAdapter(Any)


def start_planner_task(
    state: EnvState, *, instruction: str, config: dict[str, Any]
) -> EnvState:
    """Start task capture with a serializable snapshot of the execution config."""
    state.start_task(
        instruction=instruction,
        config=Payload(
            format="rpent.task_config",
            format_version="1",
            data=to_jsonable_python(config),
            completeness="complete",
        ),
    )
    return state


def end_planner_task(
    state: EnvState | None,
    result: PlannerResult | None,
    *,
    finish_result: dict | None,
    environment_success: bool | None,
    exception: BaseException | None,
    cancelled_reason: str | None = None,
) -> None:
    """Close capture after toolkit cleanup, keeping outcome separate from success."""
    if state is None:
        return
    error = result.error if result is not None else None
    reason = getattr(result, "stop_reason", None)
    if exception is not None:
        error = f"{type(exception).__name__}: {exception}"
        status = (
            "cancelled"
            if isinstance(exception, (KeyboardInterrupt, asyncio.CancelledError))
            else "failed"
        )
        reason = "operator" if status == "cancelled" else "execution_error"
    elif cancelled_reason is not None:
        status, reason = "cancelled", cancelled_reason
    elif (finish_result or {}).get("operator_aborted"):
        status, reason = "cancelled", "operator"
    elif (finish_result or {}).get("operator_finished"):
        status, reason = "completed", "operator_finished"
    elif reason == "cancelled":
        status = "cancelled"
    elif error is not None or result is None:
        status, reason = "failed", reason or "planner_error"
    else:
        status = "completed"
        reason = reason or ("finish" if finish_result is not None else "model_output")
    state.end_task(
        status=status,
        stop_reason=reason,
        result=Payload(
            format="rpent.task_result",
            format_version="1",
            data={
                "finish_result": finish_result,
                "environment_success": environment_success,
                "stats": result.stats if result is not None else {},
                "error": error,
            },
            completeness="complete",
        ),
    )
    try:
        path = state.save_planner_trajectory()
    except Exception:
        logger.exception("Failed to save planner trajectory")
    else:
        logger.info("Planner trajectory saved to %s", path)


def _sdk_data(value: Any) -> Any:
    """Normalize a Python-mode SDK snapshot while retaining binary attachments."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, dict):
        return {key: _sdk_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sdk_data(item) for item in value]
    return to_jsonable_python(value, bytes_mode="base64")


def sdk_payload(format: str, data: Any) -> Payload:
    """Preserve SDK values, including binary content, in a serializable snapshot."""
    return Payload(
        format=format,
        format_version=SDK_VERSION,
        data=_sdk_data(_SDK_VALUES.dump_python(data, mode="python")),
        completeness="complete",
    )


def import_codex_trace(state: EnvState, trace_root: Path) -> dict[str, Any]:
    """Import a settled native Codex trace and associate its tool executions."""
    stats = {
        "bundles": 0,
        "planners": 0,
        "tool_calls": 0,
        "linked_tool_calls": 0,
        "unmatched_mcp_calls": 0,
        "unlinked_tool_calls": 0,
        "errors": [],
    }
    tools_by_mcp_id = {
        call.source_ids["codex_bridge_mcp_call_id"]: call
        for call in state.tool_call_records()
        if "codex_bridge_mcp_call_id" in call.source_ids
    }
    trace_files = sorted(trace_root.rglob("trace.jsonl"))
    if not trace_files:
        stats["errors"].append(f"No Codex trace.jsonl found under {trace_root}.")

    for trace_file in trace_files:
        bundle = trace_file.parent.resolve()
        events = []
        try:
            with trace_file.open(encoding="utf-8") as file:
                for line_number, line in enumerate(file, 1):
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        stats["errors"].append(f"{trace_file}:{line_number}: {exc}")
                        continue
                    if event["schema_version"] != 1:
                        raise ValueError(
                            f"{trace_file}: unsupported Codex trace schema "
                            f"{event['schema_version']}"
                        )
                    events.append(event)
        except OSError as exc:
            stats["errors"].append(f"{trace_file}: {exc}")
            continue
        events.sort(key=lambda event: event["seq"])
        stats["bundles"] += 1

        def read_payload(reference: dict | None, format: str) -> Payload:
            if reference is None:
                return Payload(
                    format,
                    None,
                    format_version="1",
                    completeness="partial",
                    missing={
                        "": MissingInfo("not_recorded", "No native payload reference.")
                    },
                )
            try:
                path = (bundle / reference["path"]).resolve()
                if not path.is_relative_to(bundle):
                    raise ValueError("payload path is outside the trace bundle")
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("expected a native payload object")
            except (OSError, ValueError) as exc:
                detail = f"{trace_file}: {reference['path']}: {exc}"
                stats["errors"].append(detail)
                return Payload(
                    format,
                    None,
                    format_version="1",
                    completeness="partial",
                    missing={"": MissingInfo("capture_failed", detail)},
                )
            return Payload(format, data, format_version="1", completeness="complete")

        planners: dict[str, str] = {}
        for event in events:
            payload = event["payload"]
            event_type = payload["type"]
            if event_type not in {
                "inference_started",
                "inference_completed",
                "inference_failed",
                "inference_cancelled",
            }:
                continue
            inference_id = payload["inference_call_id"]
            timestamp = datetime.fromtimestamp(
                event["wall_time_unix_ms"] / 1000, timezone.utc
            )
            source_ids = {
                "codex.inference_call_id": inference_id,
                "codex.rollout_id": event["rollout_id"],
                **{
                    f"codex.{key}": event[key]
                    for key in ("thread_id", "codex_turn_id")
                    if event.get(key) is not None
                },
            }
            if inference_id not in planners:
                started = event_type == "inference_started"
                missing = {
                    field: MissingInfo("not_recorded", "No terminal inference event.")
                    for field in ("response", "ended_at")
                }
                if not started:
                    missing.update(
                        {
                            field: MissingInfo(
                                "not_recorded", "No inference_started event."
                            )
                            for field in ("request", "started_at")
                        }
                    )
                planners[inference_id] = state.register_planner(
                    backend="codex",
                    request=read_payload(
                        payload["request_payload"], "codex.inference_request"
                    )
                    if started
                    else None,
                    source_ids=source_ids,
                    started_at=timestamp if started else None,
                    missing=missing,
                )
                stats["planners"] += 1
            if event_type == "inference_started":
                continue
            status = event_type.removeprefix("inference_")
            reference = payload.get("response_payload") or payload.get(
                "partial_response_payload"
            )
            response = (
                read_payload(reference, "codex.inference_response")
                if reference
                else None
            )
            detail = (
                payload.get("error")
                or payload.get("reason")
                or "No native response payload."
            )
            if response is not None and status != "completed":
                response.completeness = "partial"
                response.missing["completion"] = MissingInfo("not_recorded", detail)
            state.update_planner(
                planners[inference_id],
                status=status,
                response=response,
                ended_at=timestamp,
                source_ids={
                    key: payload[native_key]
                    for key, native_key in (
                        ("provider.response_id", "response_id"),
                        ("provider.request_id", "upstream_request_id"),
                    )
                    if payload.get(native_key) is not None
                },
                missing={
                    "response": None
                    if response is not None
                    else MissingInfo("not_recorded", detail),
                    "ended_at": None,
                },
            )

        model_calls = {}
        for planner_order, planner_id in enumerate(planners.values()):
            planner = state.get_planner(planner_id)
            if planner.response is None or planner.response.data is None:
                continue
            for index, item in enumerate(planner.response.data["output_items"]):
                if item.get("call_id") is not None:
                    model_calls[
                        (planner.source_ids.get("codex.thread_id"), item["call_id"])
                    ] = (planner_id, planner_order, index)

        tool_starts, tool_ends, mcp_ids, code_cells = {}, {}, {}, {}
        for event in events:
            payload = event["payload"]
            event_type = payload["type"]
            thread_id = event["thread_id"]
            if event_type == "tool_call_started":
                tool_starts[(thread_id, payload["tool_call_id"])] = event
            elif event_type == "tool_call_ended":
                tool_ends[(thread_id, payload["tool_call_id"])] = event
            elif event_type == "mcp_tool_call_correlation_assigned":
                # Correlations can omit thread_id; execution IDs are bundle-wide.
                mcp_ids[payload["tool_call_id"]] = payload["mcp_call_id"]
            elif event_type == "code_cell_started":
                code_cells[(thread_id, payload["runtime_cell_id"])] = payload[
                    "model_visible_call_id"
                ]

        ordered_calls = []
        for key, event in tool_starts.items():
            payload = event["payload"]
            requester = payload["requester"]
            origin = "program" if requester["type"] == "code_cell" else "model"
            call_id = payload.get("model_visible_call_id")
            source_ids = {
                "codex.rollout_id": event["rollout_id"],
                "codex.thread_id": key[0],
                "codex.tool_call_id": key[1],
            }
            if call_id is not None:
                source_ids["codex.model_visible_call_id"] = call_id
            if origin == "program":
                cell_id = requester["runtime_cell_id"]
                call_id = code_cells.get((key[0], cell_id))
                source_ids["codex.runtime_cell_id"] = cell_id
                source_ids["codex.code_mode_runtime_tool_id"] = payload[
                    "code_mode_runtime_tool_id"
                ]
                if call_id is not None:
                    source_ids["codex.parent_model_visible_call_id"] = call_id
            planner_id, planner_order, index = model_calls.get(
                (key[0], call_id), (None, len(planners), 0)
            )
            ordered_calls.append(
                (
                    (planner_order, index, event["seq"]),
                    key,
                    event,
                    planner_id,
                    origin,
                    source_ids,
                )
            )

        for _, key, event, planner_id, origin, source_ids in sorted(ordered_calls):
            payload = event["payload"]
            mcp_id = mcp_ids.get(key[1])
            if mcp_id is not None:
                source_ids["codex_bridge_mcp_call_id"] = mcp_id
            missing = {
                "planner_id": None
                if planner_id is not None
                else MissingInfo(
                    "not_recorded", "No matching call_id in a captured model response."
                )
            }
            if mcp_id in tools_by_mcp_id:
                call = tools_by_mcp_id[mcp_id]
                state.update_tool_call(
                    call.tool_call_id,
                    planner_id=planner_id,
                    origin=origin,
                    source_ids=source_ids,
                    missing=missing,
                )
                if call.planner_id is None and planner_id is not None:
                    stats["linked_tool_calls"] += 1
                continue

            invocation = read_payload(
                payload.get("invocation_payload"), "codex.tool_invocation"
            )
            invocation_data = invocation.data or {}
            name = invocation_data.get("tool_name") or payload["summary"].get("label")
            if name is None:
                stats["errors"].append(f"{trace_file}: no tool name for {key[1]}.")
                continue
            if (
                name.startswith("mcp__rpent__")
                or invocation_data.get("tool_namespace") == "mcp__rpent"
                or payload["kind"].get("server") == "rpent"
            ):
                stats["unmatched_mcp_calls"] += 1
                continue
            terminal = tool_ends.get(key)
            feedback = None
            status = "unknown"
            ended_at = None
            if terminal is not None:
                status = terminal["payload"]["status"]
                if status == "aborted":
                    status = "cancelled"
                ended_at = datetime.fromtimestamp(
                    terminal["wall_time_unix_ms"] / 1000, timezone.utc
                )
                if terminal["payload"].get("result_payload") is not None:
                    feedback = read_payload(
                        terminal["payload"]["result_payload"], "codex.tool_result"
                    )
            missing = {
                field: reason for field, reason in missing.items() if reason is not None
            }
            missing["raw_result"] = MissingInfo(
                "not_exposed", "Native trace records the Codex adapter result."
            )
            if feedback is None:
                missing["feedback"] = MissingInfo(
                    "not_recorded", "No native tool result payload."
                )
            if terminal is None:
                missing["ended_at"] = MissingInfo(
                    "not_recorded", "No tool_call_ended event."
                )
            state.register_tool_call(
                planner_id=planner_id,
                origin=origin,
                name=name,
                arguments=invocation,
                status=status,
                source_ids=source_ids,
                started_at=datetime.fromtimestamp(
                    event["wall_time_unix_ms"] / 1000, timezone.utc
                ),
                ended_at=ended_at,
                feedback=feedback,
                missing=missing,
            )
            stats["tool_calls"] += 1
    stats["unlinked_tool_calls"] = sum(
        call.planner_id is None for call in state.tool_call_records()
    )
    return stats


class CapturingModel(WrapperModel):
    """Capture model requests, responses, and tool executions in one conversation."""

    def __init__(self, wrapped: Model, state: EnvState, conversation_id: str) -> None:
        super().__init__(wrapped)
        self.state = state
        self.conversation_id = conversation_id
        self.run_key: tuple[str | None, int] = (None, 0)
        self.calls: dict[tuple[str | None, int, str], ToolCallCapture] = {}

    def before_request(self, ctx: RunContext) -> None:
        self.run_key = (ctx.run_id, ctx.run_step)

    def response(self, planner_id: str, response: ModelResponse) -> None:
        """Register every issued call before SDK validation or dispatch."""
        for part in response.parts:
            if not isinstance(part, ToolCallPart):
                continue
            execution_missing = MissingInfo(
                "not_applicable"
                if part.tool_name in {"read_image", "finish"}
                else "not_recorded",
                "This adapter operation uses no execution model or instruction."
                if part.tool_name in {"read_image", "finish"}
                else "Robot execution metadata is not captured.",
            )
            arguments = sdk_payload("pydantic_ai.tool_args", part.args)
            if response.state in {"incomplete", "interrupted"}:
                arguments.completeness = "partial"
                arguments.missing["completion"] = MissingInfo(
                    "not_recorded", "Model stream ended before the response completed."
                )
            tool_id = self.state.register_tool_call(
                planner_id=planner_id,
                name=part.tool_name,
                arguments=arguments,
                source_ids={"pydantic_ai.tool_call_id": part.tool_call_id},
                missing={
                    "raw_result": MissingInfo("pending", "Waiting for execution."),
                    "feedback": MissingInfo("pending", "Waiting for the adapter."),
                    "effective_instruction": execution_missing,
                    "model_config": execution_missing,
                    "before_observation_id": MissingInfo(
                        "not_recorded", "Observation capture is not connected."
                    ),
                    "after_observation_id": MissingInfo(
                        "not_recorded", "Observation capture is not connected."
                    ),
                },
            )
            self.calls[(*self.run_key, part.tool_call_id)] = ToolCallCapture(
                self.state, tool_id
            )

    @contextmanager
    def tool(self, ctx: RunContext) -> Iterator[ToolCallCapture]:
        """Capture adapter exceptions after physical workers have settled."""
        call = self.calls[(ctx.run_id, ctx.run_step, ctx.tool_call_id)]
        with call.bind():
            try:
                yield call
            except ModelRetry as exc:
                self.feedback(ctx, "pydantic_ai.model_retry", {"message": str(exc)})
                raise
            except BaseException:
                self.state.update_tool_call(
                    call.tool_call_id,
                    missing={
                        "feedback": MissingInfo(
                            "not_recorded", "Adapter did not return feedback."
                        )
                    },
                )
                raise
            finally:
                if not call.finished:
                    call.finish("failed", "Adapter exited before execution completed.")

    def feedback(self, ctx: RunContext, format: str, value: Any) -> None:
        call = self.calls[(ctx.run_id, ctx.run_step, ctx.tool_call_id)]
        self.state.update_tool_call(
            call.tool_call_id,
            feedback=sdk_payload(format, value),
            missing={"feedback": None},
        )

    def settle(self, run_id: str | None, detail: str, step: int | None = None) -> None:
        """Close undispatched calls and release settled capture handles."""
        for key, call in list(self.calls.items()):
            call_run, call_step, _ = key
            if call_run != run_id or (step is not None and call_step != step):
                continue
            if not call.finished:
                call.finish("cancelled", detail)
                self.state.update_tool_call(
                    call.tool_call_id,
                    missing={"feedback": MissingInfo("not_applicable", detail)},
                )
            del self.calls[key]

    def _start(
        self,
        messages: list[ModelMessage],
        settings: ModelSettings | None,
        parameters: ModelRequestParameters,
    ) -> str:
        source_ids = {"pydantic_ai.conversation_id": self.conversation_id}
        if self.run_key[0] is not None:
            source_ids["pydantic_ai.run_id"] = self.run_key[0]
        return self.state.register_planner(
            backend="api",
            request=Payload(
                format="pydantic_ai.model_request",
                format_version=SDK_VERSION,
                data={
                    "model": {"provider": self.system, "name": self.model_name},
                    "messages": _sdk_data(
                        ModelMessagesTypeAdapter.dump_python(messages, mode="python")
                    ),
                    "model_settings": to_jsonable_python(settings, bytes_mode="base64"),
                    "model_request_parameters": _REQUEST_PARAMETERS.dump_python(
                        parameters, mode="json"
                    ),
                },
                completeness="complete",
            ),
            source_ids=source_ids,
            started_at=datetime.now(timezone.utc),
            missing={"response": MissingInfo("pending", "Waiting for response.")},
        )

    def _end(
        self,
        planner_id: str,
        response: ModelResponse | None,
        error: BaseException | None,
    ) -> None:
        status = "completed"
        if error is not None:
            status = (
                "cancelled"
                if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt))
                else "failed"
            )
        elif response is not None and response.state in {"incomplete", "interrupted"}:
            status = "cancelled" if response.state == "interrupted" else "failed"
        payload = None
        ids = {}
        missing = {"response": None}
        if response is not None:
            payload = Payload(
                format="pydantic_ai.model_response",
                format_version=SDK_VERSION,
                data=_sdk_data(
                    ModelMessagesTypeAdapter.dump_python([response], mode="python")[0]
                ),
                completeness="complete" if status == "completed" else "partial",
            )
            if status != "completed":
                payload.missing["completion"] = MissingInfo(
                    "not_recorded",
                    str(error)
                    if error is not None and str(error)
                    else "Response did not complete.",
                )
            if response.provider_response_id is not None:
                ids["provider.response_id"] = response.provider_response_id
        else:
            missing["response"] = MissingInfo(
                "capture_failed", f"{type(error).__name__}: {error}"
            )
        self.state.update_planner(
            planner_id,
            status=status,
            response=payload,
            source_ids=ids,
            ended_at=datetime.now(timezone.utc),
            missing=missing,
        )
        if response is not None:
            self.response(planner_id, response)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        planner_id = self._start(messages, model_settings, model_request_parameters)
        try:
            response = await self.wrapped.request(
                messages, model_settings, model_request_parameters
            )
        except BaseException as exc:
            self._end(planner_id, None, exc)
            raise
        self._end(planner_id, response, None)
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        planner_id = self._start(messages, model_settings, model_request_parameters)
        stream = None
        error = None
        try:
            async with self.wrapped.request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as stream:
                yield stream
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._end(planner_id, stream.get() if stream is not None else None, error)
