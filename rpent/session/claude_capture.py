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

"""Import model calls from Claude Code's local Messages API recordings."""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from packaging.version import Version

from rpent.session import EnvState, MissingInfo, Payload, ToolCallRecord
from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions

logger = get_logger("claude_capture")


class ClaudeTraceCapture:
    """Record and import the API bodies written by one Claude run."""

    def __init__(
        self, state: EnvState, run_id: str, prompt: str, options: ClaudeAgentOptions
    ) -> None:
        from claude_agent_sdk import __version__

        if Version(__version__) < Version("0.2.154"):
            raise RuntimeError(
                f"Claude trace capture requires claude-agent-sdk >= 0.2.154, found {__version__}."
            )
        from claude_agent_sdk._internal.transport.subprocess_cli import (
            SubprocessCLITransport,
        )

        self.transport = SubprocessCLITransport(prompt, options)
        cli_path = options.cli_path or self.transport._find_cli()
        cli_version = subprocess.check_output(
            [cli_path, "--version"], text=True, timeout=10
        ).split()[0]
        if Version(cli_version) < Version("2.1.274"):
            raise RuntimeError(
                f"Claude trace capture requires Claude Code >= 2.1.274, found {cli_version}."
            )
        self._state = state
        self._run_dir = (state.trajectory_dir / "claude" / run_id).resolve()
        self._run_dir.mkdir(parents=True)
        options.env["OTEL_LOG_RAW_API_BODIES"] = f"file:{self._run_dir}"
        self.stats: dict[str, Any] = {
            "sdk_version": __version__,
            "cli_version": cli_version,
        }

    async def finish(self) -> None:
        """Import after the process that writes the native recording has exited."""
        try:
            # Breaking out of sdk.query() can leave the CLI flushing in the
            # background. Stateful sessions already await their client's exit.
            if self.transport._process is not None:
                await self.transport._process.wait()
            self.stats.update(import_claude_trace(self._state, self._run_dir))
        except Exception as exc:
            self.stats["errors"] = [f"{type(exc).__name__}: {exc}"]
        for issue in self.stats["errors"]:
            logger.warning("Claude trace capture: %s", issue)


def _mark_content(payload: Payload, value: Any, pointer: str = "") -> None:
    """Locate native content blocks, including images nested in tool results."""
    if isinstance(value, list):
        for index, item in enumerate(value):
            _mark_content(payload, item, f"{pointer}/{index}")
    elif isinstance(value, dict):
        kind = value.get("type")
        source = value.get("source")
        if (
            kind == "image"
            and isinstance(source, dict)
            and source.get("type") == "base64"
        ):
            payload.base64_paths[f"{pointer}/source/data"] = ""
        elif kind in ("thinking", "redacted_thinking"):
            payload.completeness = "partial"
            payload.missing[pointer] = MissingInfo(
                "not_exposed",
                "Claude Code redacts extended thinking from API recordings.",
            )
        for key in ("system", "messages", "content", "source"):
            if key in value:
                _mark_content(payload, value[key], f"{pointer}/{key}")


def _read_body(path: Path, kind: str, errors: list[str]) -> Payload:
    format_name = f"claude.messages_{kind}"
    try:
        if path.resolve().parent != path.parent:
            raise ValueError("body is outside the Claude run directory")
        data = json.loads(path.read_bytes())
        if not isinstance(data, dict):
            raise ValueError("expected a Messages API object")
    except (OSError, ValueError) as exc:
        detail = f"{path}: {exc}"
        errors.append(detail)
        return Payload(
            format_name,
            None,
            completeness="partial",
            missing={"": MissingInfo("capture_failed", detail)},
        )
    payload = Payload(format_name, data, completeness="complete")
    _mark_content(payload, data)
    return payload


def _request_entries(run_dir: Path, errors: list[str]) -> dict[Path, dict[str, Any]]:
    """Keep index order for successful calls, then include unindexed attempts."""
    entries = {}
    index = run_dir / "index.jsonl"
    try:
        with index.open("rb") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    entry = json.loads(line)
                    request = run_dir / entry["request_file"]
                    response = run_dir / entry["response_file"]
                    if request.parent != run_dir or response.parent != run_dir:
                        raise ValueError(
                            "expected body files in the Claude run directory"
                        )
                    entries.setdefault(request, entry)
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"{index}:{number}: {exc}")
    except FileNotFoundError:
        pass  # The CLI creates the index only after a successful response.
    except OSError as exc:
        errors.append(f"{index}: {exc}")
    for request in sorted(run_dir.glob("*.request.json")):
        entries.setdefault(request, {})
    if not entries and not errors:
        errors.append(f"No Claude request recordings found in {run_dir}.")
    return entries


def _blocks(message: dict[str, Any], kind: str) -> Iterator[tuple[int, dict[str, Any]]]:
    content = message.get("content")
    if isinstance(content, list):
        for index, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == kind:
                yield index, block


def _request_tool_results(request: Payload | None) -> dict[str, dict[str, Any] | None]:
    """Find model-visible results, excluding history before the latest use of an ID."""
    results = {}
    messages = request.data.get("messages") if request and request.data else None
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                for _, block in _blocks(message, "tool_use"):
                    if isinstance(key := block.get("id"), str):
                        results.pop(key, None)
            elif message.get("role") == "user":
                for _, block in _blocks(message, "tool_result"):
                    key = block.get("tool_use_id")
                    if not isinstance(key, str):
                        continue
                    if key not in results:
                        results[key] = block
                    elif results[key] != block:
                        results[key] = None
    return results


def _import_native_tool(
    state: EnvState,
    planner_id: str,
    use: dict[str, Any],
    source_ids: dict[str, str],
    previous: ToolCallRecord | None,
    result: dict[str, Any] | MissingInfo,
) -> None:
    """Record a Claude-side tool request and the first result sent back to the model."""
    if previous is not None and previous.feedback is not None:
        if previous.source_ids != source_ids:
            state.update_tool_call(previous.tool_call_id, source_ids=source_ids)
        return
    feedback = None
    if isinstance(result, MissingInfo):
        status = "requested"
        missing = {"feedback": result}
    else:
        status = "failed" if result.get("is_error") else "completed"
        missing = {}
        feedback = Payload("claude.tool_result", result, completeness="complete")
        _mark_content(feedback, result)
    if previous is not None:
        state.update_tool_call(
            previous.tool_call_id,
            status=status,
            feedback=feedback,
            source_ids=source_ids,
            missing={"feedback": missing.get("feedback")},
        )
        return
    arguments = Payload("claude.tool_input", use.get("input"), completeness="complete")
    if "input" not in use:
        arguments.completeness = "partial"
        arguments.missing[""] = MissingInfo(
            "not_recorded", "Claude tool_use has no input."
        )
    no_time = MissingInfo("not_exposed", "Claude does not record tool execution times.")
    state.register_tool_call(
        planner_id=planner_id,
        name=use["name"],
        status=status,
        arguments=arguments,
        source_ids=source_ids,
        feedback=feedback,
        missing={
            **missing,
            "started_at": no_time,
            "ended_at": no_time,
            "raw_result": MissingInfo(
                "not_exposed", "Claude records only the result sent to the model."
            ),
        },
    )


def _import_tools(
    state: EnvState, planner_ids: list[str], run_id: str | None, errors: list[str]
) -> None:
    """Match callback executions once per run and attach them in response order."""
    uses = []
    uses_by_id = defaultdict(list)
    results = {}
    for planner_id in planner_ids:
        planner = state.get_planner(planner_id)
        for key, result in _request_tool_results(planner.request).items():
            candidates = uses_by_id.get(key, ())
            if len(candidates) == 1:
                location, _ = candidates[0]
                # Later requests can repeat or shorten the same conversation history.
                results.setdefault(location, result)
        if planner.response is None or planner.response.data is None:
            continue
        for index, block in _blocks(planner.response.data, "tool_use"):
            key = block.get("id")
            if isinstance(key, str) and key:
                uses_by_id[key].append(((planner_id, index), block.get("name")))
            else:
                key = None
            if not isinstance(block.get("name"), str) or not block["name"]:
                errors.append(
                    f"{planner_id} /content/{index}: Claude tool_use has no name."
                )
                continue
            uses.append((planner_id, index, key, block))

    executions = defaultdict(list)
    native = {}
    for call in state.tool_call_records():
        if "claude.tool_use_index" in call.source_ids:
            if call.planner_id in planner_ids:
                native[call.planner_id, call.source_ids["claude.tool_use_index"]] = call
        elif run_id is not None and call.source_ids.get("claude.run_id") == run_id:
            executions[call.source_ids.get("claude.tool_use_id")].append(call)

    matched = {}
    for key, calls in executions.items():
        candidates = uses_by_id.get(key, ())
        if not key:
            reason = MissingInfo("not_exposed", "Claude did not provide a tool use ID.")
        elif len(calls) > 1 or len(candidates) > 1:
            reason = MissingInfo(
                "not_recorded",
                f"Ambiguous Claude tool_use ID {key!r}: {len(candidates)} model calls and {len(calls)} executions.",
            )
        elif not candidates:
            reason = MissingInfo(
                "not_recorded",
                f"No Claude model response contains tool_use ID {key!r}.",
            )
        elif candidates[0][1] != f"mcp__rpent__{calls[0].name}":
            reason = MissingInfo(
                "not_recorded",
                f"The tool name in the response for {key!r} differs from the executed tool.",
            )
        else:
            matched[key] = calls[0]
            continue
        for call in calls:
            if call.planner_id is None or call.planner_id in planner_ids:
                state.update_tool_call(
                    call.tool_call_id,
                    unlink_planner=call.planner_id is not None,
                    origin="unknown",
                    missing={"planner_id": reason},
                )

    positions: Counter[str] = Counter()
    for planner_id, index, key, use in uses:
        if use["name"].startswith("mcp__rpent__"):
            call = matched.get(key)
            if call is None:
                continue
            if call.planner_id is None:
                state.update_tool_call(
                    call.tool_call_id,
                    planner_id=planner_id,
                    response_tool_index=positions[planner_id],
                    origin="model",
                    missing={"planner_id": None},
                )
        else:
            source_ids = {"claude.tool_use_index": str(index)}
            if run_id is not None:
                source_ids["claude.run_id"] = run_id
            if key is not None:
                source_ids["claude.tool_use_id"] = key
            location = (planner_id, index)
            if location in results:
                result = results[location] or MissingInfo(
                    "not_recorded", "Conflicting Claude tool_result blocks."
                )
            elif key is None or len(uses_by_id[key]) > 1:
                result = MissingInfo(
                    "not_recorded", "Missing or ambiguous Claude tool_use ID."
                )
            else:
                result = MissingInfo(
                    "not_recorded", "No subsequent Claude tool_result."
                )
            _import_native_tool(
                state,
                planner_id,
                use,
                source_ids,
                native.get((planner_id, str(index))),
                result,
            )
        positions[planner_id] += 1


def import_claude_trace(
    state: EnvState, run_dir: Path, *, run_id: str | None = None
) -> dict[str, Any]:
    """Import one Claude run into an active task, once per native request file.

    Args:
        state: Task that owns the model calls.
        run_dir: Claude Code directory containing the index and saved API bodies.
        run_id: RPent solve ID used by the tool callbacks. A recording under the
            task's ``claude/<run_id>`` directory supplies it automatically. Pass the
            original ID to associate executions from a separately stored recording.

    Returns:
        The number of newly registered model calls and any recording errors.
    """
    run_dir = run_dir.resolve()
    relative_dir = Path(os.path.relpath(run_dir, state.trajectory_dir.resolve()))
    existing = {
        request_file: record
        for record in state.planner_records()
        if record.backend == "claude_code"
        and (request_file := record.source_ids.get("claude.request_file"))
        and Path(request_file).parent == relative_dir
    }
    saved_run = next(
        (
            record.source_ids["claude.run_id"]
            for record in existing.values()
            if "claude.run_id" in record.source_ids
        ),
        None,
    )
    if run_id is None:
        run_id = saved_run
        if run_id is None and relative_dir.parent == Path("claude"):
            run_id = run_dir.name
    elif saved_run is not None and saved_run != run_id:
        return {
            "planners": 0,
            "errors": [f"{run_dir} already belongs to Claude run {saved_run!r}."],
        }
    errors: list[str] = []
    imported = 0
    planner_ids = []
    run_source = {"claude.run_id": run_id} if run_id is not None else {}
    for path, entry in _request_entries(run_dir, errors).items():
        request_file = str(relative_dir / path.name)
        previous = existing.get(request_file)
        if previous is None:
            no_response = MissingInfo("not_recorded", "No indexed Claude response.")
            planner_id = state.register_planner(
                backend="claude_code",
                request=_read_body(path, "request", errors),
                source_ids={"claude.request_file": request_file, **run_source},
                missing={
                    "started_at": MissingInfo(
                        "not_exposed", "Claude does not index request start times."
                    ),
                    "response": no_response,
                    "ended_at": no_response,
                },
            )
            imported += 1
        else:
            planner_id = previous.planner_id
            if run_source and "claude.run_id" not in previous.source_ids:
                state.update_planner(planner_id, source_ids=run_source)
        planner_ids.append(planner_id)
        if not entry or (previous is not None and previous.response is not None):
            continue

        response_path = run_dir / entry["response_file"]
        response = _read_body(response_path, "response", errors)
        source_ids = {"claude.response_file": str(relative_dir / response_path.name)}
        for field, key in (
            ("session_id", "claude.session_id"),
            ("query_source", "claude.query_source"),
            ("message_uuid", "claude.message_uuid"),
            ("message_id", "provider.response_id"),
            ("request_id", "provider.request_id"),
        ):
            if isinstance(entry.get(field), str):
                source_ids[key] = entry[field]
        ended_at = None
        if isinstance(timestamp := entry.get("timestamp"), str):
            try:
                ended_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError as exc:
                errors.append(f"{path.name}: {exc}")
        state.update_planner(
            planner_id,
            status="completed",
            response=response if response.data is not None else None,
            source_ids=source_ids,
            ended_at=ended_at,
            missing={
                "response": response.missing.get(""),
                "ended_at": None
                if ended_at
                else MissingInfo("not_recorded", "No valid Claude response timestamp."),
            },
        )
    _import_tools(state, planner_ids, run_id, errors)
    return {"planners": imported, "errors": errors}
