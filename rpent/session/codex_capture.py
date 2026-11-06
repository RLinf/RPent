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

"""Capture planner records from native Codex traces."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rpent.session import EnvState, MissingInfo, Payload, PlannerRecord, ToolCallRecord


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

        planner_ids = _import_codex_planners(state, events, trace_file, stats)
        _import_codex_tools(
            state, events, planner_ids, tools_by_mcp_id, trace_file, stats
        )
    stats["unlinked_tool_calls"] = sum(
        call.planner_id is None for call in state.tool_call_records()
    )
    return stats


def _read_codex_payload(
    trace_file: Path,
    reference: dict[str, Any] | None,
    format: str,
    errors: list[str],
) -> Payload:
    """Read a native payload and retain the reason for missing content."""
    if reference is None:
        reason = MissingInfo("not_recorded", "No native payload reference.")
    else:
        bundle = trace_file.parent.resolve()
        try:
            path = (bundle / reference["path"]).resolve()
            if not path.is_relative_to(bundle):
                raise ValueError("payload path is outside the trace bundle")
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("expected a native payload object")
        except (OSError, ValueError) as exc:
            detail = f"{trace_file}: {reference['path']}: {exc}"
            errors.append(detail)
            reason = MissingInfo("capture_failed", detail)
        else:
            return Payload(
                format,
                data,
                format_version="1",
                completeness="complete",
                base64_paths=_codex_image_paths(data),
            )
    return Payload(
        format,
        None,
        format_version="1",
        completeness="partial",
        missing={"": reason},
    )


def _codex_image_paths(value: Any, pointer: str = "") -> dict[str, str]:
    """Locate embedded images along native Codex message and tool-result fields."""
    paths = {}
    if isinstance(value, list):
        for index, item in enumerate(value):
            paths.update(_codex_image_paths(item, f"{pointer}/{index}"))
    elif isinstance(value, dict):
        if value.get("type") == "input_image":
            url = value.get("image_url")
            if isinstance(url, str):
                prefix, separator, _ = url.partition(",")
                if (
                    separator
                    and prefix.startswith("data:image/")
                    and prefix.endswith(";base64")
                ):
                    paths[f"{pointer}/image_url"] = prefix + separator
        for key in ("input", "output_items", "response_item", "content", "output"):
            if key in value:
                paths.update(_codex_image_paths(value[key], f"{pointer}/{key}"))
    return paths


def _expand_codex_request(request: Payload, parent: PlannerRecord | None) -> Payload:
    """Expand a native Codex request using its captured parent response."""
    expanded = replace(request, format="codex.expanded_request")
    if request.data is None:
        return expanded
    data = dict(request.data)
    previous_response_id = data.pop("previous_response_id", None)
    if previous_response_id is not None:
        if (
            parent is None
            or parent.request is None
            or parent.request.completeness != "complete"
            or parent.response is None
            or parent.response.completeness != "complete"
        ):
            expanded.completeness = "partial"
            expanded.missing = {
                **request.missing,
                "/input": MissingInfo(
                    "not_recorded",
                    f"Cannot expand previous_response_id {previous_response_id!r}: "
                    "parent request or response is missing or incomplete.",
                ),
            }
        else:
            data["input"] = (
                parent.request.data["input"]
                + parent.response.data["output_items"]
                + data["input"]
            )
    expanded.data = data
    expanded.base64_paths = _codex_image_paths(data)
    return expanded


def _import_codex_planners(
    state: EnvState,
    events: list[dict[str, Any]],
    trace_file: Path,
    stats: dict[str, Any],
) -> list[str]:
    """Import native model calls and return their planner IDs in registration order."""
    planners: dict[str, str] = {}
    planners_by_response_id: dict[str, str] = {}
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
            request = None
            missing = {
                field: MissingInfo("not_recorded", "No terminal inference event.")
                for field in ("response", "ended_at")
            }
            if started:
                request = _read_codex_payload(
                    trace_file,
                    payload["request_payload"],
                    "codex.inference_request",
                    stats["errors"],
                )
                previous_response_id = (request.data or {}).get("previous_response_id")
                parent_id = planners_by_response_id.get(previous_response_id)
                if previous_response_id is not None:
                    source_ids["provider.previous_response_id"] = previous_response_id
                request = _expand_codex_request(
                    request,
                    state.get_planner(parent_id) if parent_id is not None else None,
                )
                if "/input" in request.missing:
                    stats["errors"].append(
                        f"{trace_file}: {inference_id}: {request.missing['/input'].detail}"
                    )
            else:
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
                request=request,
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
            _read_codex_payload(
                trace_file, reference, "codex.inference_response", stats["errors"]
            )
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
        if payload.get("response_id") is not None:
            planners_by_response_id[payload["response_id"]] = planners[inference_id]
    return list(planners.values())


def _import_codex_tools(
    state: EnvState,
    events: list[dict[str, Any]],
    planner_ids: list[str],
    tools_by_mcp_id: dict[str, ToolCallRecord],
    trace_file: Path,
    stats: dict[str, Any],
) -> None:
    """Associate captured MCP calls and import other native tool executions."""
    model_calls = {}
    for planner_order, planner_id in enumerate(planner_ids):
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
            (key[0], call_id), (None, len(planner_ids), 0)
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

        invocation = _read_codex_payload(
            trace_file,
            payload.get("invocation_payload"),
            "codex.tool_invocation",
            stats["errors"],
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
                feedback = _read_codex_payload(
                    trace_file,
                    terminal["payload"]["result_payload"],
                    "codex.tool_result",
                    stats["errors"],
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
