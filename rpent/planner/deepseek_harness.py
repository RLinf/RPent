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

"""DeepSeek Harness runtime with RPent's prompt, tools, and lifecycle boundary.

The runtime owns model inference and conversation compaction. RPent owns the
Toolkit and accepts completion only from its actual tool result. A runtime
plugin enforces the step budget and tool scope before execution; an independent
Python deadline also bounds the SDK's blocking notification queue.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from rpent.dashboard.events import DashboardEventSink, TranscriptEvent, UsageEvent
from rpent.dashboard.interaction import DashboardInteractionPort
from rpent.planner.base import Planner, PlannerResult, add_mcp_prefix, strip_mcp_prefix
from rpent.planner.utils.http_mcp_server import HttpMcpServer, mcp_result
from rpent.tools import ToolResult
from rpent.tools.toolkit import Toolkit
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger

logger = get_logger("deepseek_harness")
PROVIDER = "openai-codex"
SDK_VERSION = "0.2.1rc2"


def _redact(text: str) -> str:
    from rpent.planner.check import redact_secrets

    text = redact_secrets(text)
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~-]+", r"\1<redacted>", text)
    return re.sub(
        r"(?i)((?:(?:access|refresh|id)[_-]?token|api[_-]?key)(?:\\?[\"'])?\s*[:=]\s*(?:\\?[\"'])?)[^\\\s,\"'}]+",
        r"\1<redacted>",
        text,
    )


def _sanitize(value: Any) -> Any:
    """Redact diagnostics recursively before escaping them into artifact JSON."""
    if isinstance(value, str):
        return _redact(value)
    if isinstance(value, dict):
        secrets = {"accesstoken", "refreshtoken", "idtoken", "apikey", "authorization"}
        return {
            key: "<redacted>"
            if re.sub(r"[_-]", "", str(key)).lower() in secrets and item is not None
            else _sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def resolve_dsh_home(value: str | Path | None) -> Path:
    """Resolve the explicitly selected credential home without copying logins."""
    selected = value or os.environ.get("DSH_HOME")
    if not selected or not str(selected).strip():
        raise ValueError("deepseek_harness requires --dsh-home or DSH_HOME")
    return Path(selected).expanduser().resolve()


def _attachment_home() -> Path:
    """Allocate storage supporting dsh's cross-directory atomic hard links."""
    candidates = [tempfile.gettempdir()]
    if os.name == "posix" and Path("/tmp").is_dir():
        candidates.append("/tmp")
    for directory in dict.fromkeys(candidates):
        home = None
        try:
            home = Path(
                tempfile.mkdtemp(prefix="rpent-dsh-attachments-", dir=directory)
            )
            source, target = home / "probe-source", home / "probe-target"
            source.mkdir()
            target.mkdir()
            (source / "image").write_bytes(b"attachment filesystem probe")
            os.link(source / "image", target / "image")
            shutil.rmtree(source)
            shutil.rmtree(target)
            return home
        except OSError:
            if home is not None:
                shutil.rmtree(home, ignore_errors=True)
    raise RuntimeError(
        "dsh image storage needs cross-directory hard links; set TMPDIR to a local writable filesystem"
    )


class _ToolBoundary:
    """Serialize MCP calls and latch native completion before returning over HTTP."""

    def __init__(self, toolkit: Toolkit, ledger_path: Path | None = None):
        self.toolkit = toolkit
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.finish_result: dict | None = None
        self.completed_calls = 0
        self.execution_uncertain = False
        self.ledger_path = ledger_path

    def list_tools(self):
        return self.toolkit.list_tools()

    def execute_tool(self, name: str, arguments: dict) -> ToolResult:
        with self.lock:
            if self.stopped.is_set() or self.finish_result is not None:
                return ToolResult(
                    data={"error": "RPent run has stopped; tool refused."}
                )
            try:
                result = self.toolkit.execute_tool(name, arguments)
            except BaseException:
                self.execution_uncertain = True
                self.stopped.set()
                raise
            self.execution_uncertain |= result.execution_uncertain
            if self.execution_uncertain:
                self.stopped.set()
            self.completed_calls += 1
            if (
                not result.is_error
                and not result.execution_uncertain
                and result.data.get("_finish") is True
            ):
                self.finish_result = dict(result.data)
            if self.ledger_path is not None:
                business_error_hash = None
                if (
                    result.is_error
                    and not result.execution_uncertain
                    and not result.images
                ):
                    try:
                        text = mcp_result(result, include_structured=True)["content"][
                            0
                        ]["text"]
                        # MCP errors become text-only runtime exceptions; the
                        # structured copy cannot rescue a truncated text block.
                        json.loads(text)
                    except Exception:
                        # A native refusal whose payload cannot be serialized
                        # is not a correctly delivered business error. The MCP
                        # failure must close admission just like lost output.
                        pass
                    else:
                        business_error_hash = hashlib.sha256(text.encode()).hexdigest()
                temporary = self.ledger_path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(
                        {
                            "seq": self.completed_calls,
                            "name": name,
                            "isError": result.is_error,
                            "executionUncertain": result.execution_uncertain,
                            "canonicalBusinessErrorSha256": business_error_hash,
                            "finishResult": self.finish_result,
                        }
                    )
                )
                temporary.replace(self.ledger_path)
            return result

    def cancel_active_and_wait(self) -> None:
        self.stopped.set()
        self.toolkit.cancel_active_and_wait()
        # Toolkit cancellation may return before the MCP wrapper records its
        # result. Hold this lock once to establish the final action certainty.
        with self.lock:
            pass

    def resume_calls(self) -> None:
        """Reopen admission only after the driver establishes runtime quiescence."""
        with self.lock:
            if self.execution_uncertain or self.finish_result is not None:
                raise RuntimeError("Cannot resume a terminated or uncertain dsh run")
            self.toolkit.resume_calls()
            self.stopped.clear()


class _Recorder:
    def __init__(self, sink: DashboardEventSink, *, terminal: bool = False):
        self.sink = sink
        self.terminal = terminal
        self.steps = 0
        self.tool_calls = 0
        self.tool_results = 0
        self.delivery_errors: list[dict] = []
        self.pending_calls: set[str] = set()
        self.usage: dict[str, int] = {}
        self._input_usage_complete = True
        self._output_usage_complete = True
        self.messages: list[dict] = []
        self.consumed_user_receipts: dict[str, int | None] = {}
        self.seen: set[int] = set()
        self.received = False
        self.idle = False
        self.turn_reason: dict = {}
        self.failed_attempts: list[dict] = []
        self.retries_scheduled: list[dict] = []
        self.retries_started: list[dict] = []
        self.compactions = 0
        self.image_offloads = 0

    def observe(self, notification: Any, receipt: str) -> None:
        payload = _sanitize(notification.payload)
        if notification.method == "session.status":
            self.idle = self.received and payload.get("status") == "idle"
            return
        if notification.method != "session.event":
            return
        event = payload.get("event", {})
        seq = event.get("seq")
        if seq is not None:
            if seq in self.seen:
                return
            self.seen.add(seq)
        kind, data = event.get("type"), event.get("data", {})
        if kind == "agent/inbox/spliced":
            self.received |= any(
                m.get("id") == receipt for m in data.get("inserted", [])
            )
        elif kind == "user/message":
            if data.get("id") != receipt:
                self.messages.append(data)
                if isinstance(data.get("id"), str):
                    self.consumed_user_receipts[data["id"]] = seq
        elif kind == "turn/start":
            self.idle = False
        elif kind == "step/start":
            self.steps += 1
        elif kind == "turn/end":
            self.turn_reason = data.get("reason", {})
        elif kind == "assistant/attempt":
            chunks = [item.get("chunk", {}) for item in data.get("stream", [])]
            usage = next(
                (
                    item.get("usage")
                    for item in reversed(chunks)
                    if item.get("type") == "usage"
                ),
                None,
            )
            self.failed_attempts.append(
                {
                    "turn": data.get("turn"),
                    "step": data.get("step"),
                    "usage": usage,
                    "usage_known": isinstance(usage, dict),
                    "reason": next(
                        (
                            item.get("reason")
                            for item in reversed(chunks)
                            if item.get("type") == "finish"
                        ),
                        None,
                    ),
                }
            )
        elif kind == "llm/retry":
            self.retries_scheduled.append(data)
        elif kind == "llm/retry-started":
            self.retries_started.append(data)
        elif kind == "compaction/summary":
            self.compactions += 1
        elif kind == "image/offload":
            self.image_offloads += 1
        elif kind == "assistant/message":
            message = data.get("message", {})
            self.messages.append(message)
            for block in message.get("content", []):
                if block.get("type") in ("text", "reasoning"):
                    if self.terminal:
                        logger.info("[%s] %s", block["type"], block.get("text", ""))
                    self.sink.emit(
                        TranscriptEvent(
                            {
                                "type": "thinking"
                                if block["type"] == "reasoning"
                                else "text",
                                "text": block.get("text", ""),
                            }
                        )
                    )
            self._record_usage(data.get("usage"))
            if {"total_input_tokens", "total_output_tokens"} <= self.usage.keys():
                self.sink.emit(
                    UsageEvent(
                        inp=self.usage["total_input_tokens"],
                        out=self.usage["total_output_tokens"],
                        tool_calls=self.tool_calls,
                    )
                )
        elif kind == "tool/call":
            self.tool_calls += 1
            if self.terminal:
                logger.info("[tool] %s", strip_mcp_prefix(data.get("name", "")))
            if data.get("callId"):
                self.pending_calls.add(data["callId"])
            self.sink.emit(
                TranscriptEvent(
                    {
                        "type": "tool_call",
                        "tool": strip_mcp_prefix(data.get("name", "")),
                        "args": data.get("arguments", {}),
                    }
                )
            )
        elif kind == "tool/result":
            self.tool_results += 1
            message = data.get("message", {})
            call_id = message.get("toolCallId") or message.get("source", {}).get(
                "callId"
            )
            self.pending_calls.discard(call_id)
            if (data.get("error") or {}).get("code") == "INVALID_TOOL_OUTPUT":
                # A runtime rejection of the returned value is not delivery of
                # the native action result, even though call IDs are paired.
                self.delivery_errors.append(
                    {"seq": seq, "call_id": call_id, "error": data["error"]}
                )
            self.messages.append(message)
            self.sink.emit(TranscriptEvent({"type": "tool_result", "result": data}))

    def _record_usage(self, usage: Any) -> None:
        usage = usage if isinstance(usage, dict) else {}
        counts = {
            key: value
            for key, value in usage.items()
            if type(value) is int and 0 <= value <= 2**53 - 1
        }
        for source, target in (
            ("inputTokens", "total_uncached_input_tokens"),
            ("outputTokens", "total_output_tokens"),
            ("cacheReadTokens", "total_cached_input_tokens"),
            ("cacheWriteTokens", "total_cache_creation_input_tokens"),
            ("reasoningTokens", "total_reasoning_output_tokens"),
        ):
            if source in counts:
                self.usage[target] = self.usage.get(target, 0) + counts[source]
        self._output_usage_complete &= "outputTokens" in counts
        if not self._output_usage_complete:
            self.usage.pop("total_output_tokens", None)

        # dsh input/read/write buckets are disjoint. An absent cache bucket is
        # unknown, so use the exact provider total or all three input buckets.
        prompt_fields = ("inputTokens", "cacheReadTokens", "cacheWriteTokens")
        known_prompt = sum(counts.get(key, 0) for key in prompt_fields)
        complete_prompt = all(key in counts for key in prompt_fields)
        exact_input = None
        if {"inputTokens", "outputTokens"} <= counts.keys() and all(
            key not in usage or key in counts for key in prompt_fields
        ):
            if "totalTokens" in counts:
                candidate = counts["totalTokens"] - counts["outputTokens"]
                if candidate >= known_prompt and (
                    not complete_prompt or candidate == known_prompt
                ):
                    exact_input = candidate
            elif "totalTokens" not in usage and complete_prompt:
                exact_input = known_prompt
        self._input_usage_complete &= exact_input is not None
        if self._input_usage_complete:
            self.usage["total_input_tokens"] = (
                self.usage.get("total_input_tokens", 0) + exact_input
            )
        else:
            self.usage.pop("total_input_tokens", None)


class DeepSeekHarnessPlanner(Planner):
    """Run a bounded task with optional same-process interactive continuation."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        recipe_tag: str,
        dashboard_events: DashboardEventSink,
        model: str | None,
        dsh_home: str | Path | None = None,
        reasoning_effort: str = "high",
        timeout_s: float = 1200,
        max_tokens: int = 8192,
        repo_root: str | Path | None = None,
    ):
        """Resolve configuration before creating any simulator resources.

        Args:
            output_dir: Parent for the transcript and runtime control artifacts.
            recipe_tag: Human-readable run label.
            dashboard_events: Transcript consumer (CLI uses a null consumer).
            model: Exact openai-codex model id; no model substitution is performed.
            dsh_home: Explicit login home, falling back to DSH_HOME.
            reasoning_effort: Requested low, medium, high, or xhigh effort.
            timeout_s: Deadline covering runtime startup, inference, and tools.
            max_tokens: Runtime output budget; the subscription provider may not
                enforce it as a wire-level output cap.
            repo_root: Runtime working directory.
        """
        if not model:
            raise ValueError("deepseek_harness requires --model")
        if reasoning_effort not in ("low", "medium", "high", "xhigh"):
            raise ValueError("deepseek_harness requires low/medium/high/xhigh effort")
        if not math.isfinite(timeout_s) or timeout_s <= 0 or max_tokens <= 0:
            raise ValueError("planner timeout and max_tokens must be positive")
        self.home = resolve_dsh_home(dsh_home)
        self.model = model
        self.effort = reasoning_effort
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.recipe_tag = recipe_tag
        self.cwd = str(Path(repo_root or get_repo_root()).expanduser().resolve())
        self.sink = dashboard_events
        try:
            installed = {
                version(package)
                for package in ("deepseek-harness-sdk", "deepseek-harness-runtime-bin")
            }
        except PackageNotFoundError as exc:
            raise RuntimeError(
                "Install the optional rpent[deepseek-harness] dependency"
            ) from exc
        if installed != {SDK_VERSION}:
            raise RuntimeError(
                f"Install rpent[deepseek-harness] (SDK/runtime {SDK_VERSION})"
            )

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
        dashboard_interaction: DashboardInteractionPort | None = None,
    ) -> PlannerResult:
        """Run until native finish, budget, error, interruption, or deadline."""
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        return self._solve(
            system_prompt,
            user_message,
            toolkit,
            max_turns,
            input_queue=input_queue,
            dashboard_interaction=dashboard_interaction,
        )

    def _solve(
        self,
        system_prompt,
        user_message,
        toolkit,
        max_steps,
        *,
        probe=False,
        input_queue=None,
        dashboard_interaction=None,
    ):
        from deepseek_harness import HarnessClient, HarnessConfig

        work = self.output_dir / f"dsh-{uuid.uuid4().hex[:12]}"
        work.mkdir(parents=True)
        raw_path, control_path = work / "events.jsonl", work / "control.json"
        recorder = _Recorder(self.sink, terminal=input_queue is not None)
        recorder.messages = _sanitize(
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {"role": "user", "content": [{"type": "text", "text": user_message}]},
            ]
        )
        boundary = _ToolBoundary(toolkit, work / "tool-ledger.json")
        server = HttpMcpServer(boundary, include_structured=True)
        outcome: dict[str, Any] = {}
        done = threading.Event()
        incoming = queue.Queue()
        interactive = input_queue is not None or dashboard_interaction is not None
        started = time.monotonic()
        client = None
        worker = None
        attachment_home = None
        attachment_retained = False
        error = None
        stop_reason = "runtime_error"
        control = {}
        cleanup_ok = True
        process_interrupted = False
        active_elapsed = 0.0
        session_id = f"rpent-{uuid.uuid4().hex}"
        try:
            attachment_home = _attachment_home()
            url = server.start(ready_timeout_s=min(30, self.timeout_s))
            patch = self._write_patch(
                work,
                control_path,
                url,
                boundary,
                system_prompt,
                max_steps,
                attachment_home,
            )
            self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
            client = HarnessClient(
                HarnessConfig(
                    profile="sdk",
                    dsh_home=str(self.home),
                    cwd=self.cwd,
                    patches=(str(patch),),
                    env={
                        "DSH_TELEMETRY_DISABLED": "1",
                        "NO_PROXY": ",".join(
                            filter(
                                None,
                                (
                                    os.environ.get(
                                        "NO_PROXY", os.environ.get("no_proxy", "")
                                    ),
                                    "127.0.0.1",
                                    "localhost",
                                    "::1",
                                ),
                            )
                        ),
                        "no_proxy": ",".join(
                            filter(
                                None,
                                (
                                    os.environ.get(
                                        "no_proxy", os.environ.get("NO_PROXY", "")
                                    ),
                                    "127.0.0.1",
                                    "localhost",
                                    "::1",
                                ),
                            )
                        ),
                    },
                    initialize_timeout_seconds=min(30, self.timeout_s),
                    request_timeout_seconds=self.timeout_s,
                    shutdown_timeout_seconds=1,
                )
            )
            # Start on the owner thread: cleanup cannot race a not-yet-started child.
            client.start()

            def run():
                try:
                    client.initialize(
                        cwd=self.cwd,
                        provider=PROVIDER,
                        model=self.model,
                        reasoning_effort=self.effort,
                        max_tokens=self.max_tokens,
                    )
                    with (
                        raw_path.open("w") as raw,
                        client.subscribe_session_notifications(
                            session_id
                        ) as subscription,
                    ):
                        receipt = client.session_prompt(
                            session_id,
                            [{"type": "text", "text": user_message}],
                            notification_subscription=subscription,
                        )
                        outcome["receipt"] = receipt
                        received = False
                        while True:
                            event = subscription.next()
                            raw.write(
                                json.dumps(
                                    _sanitize(
                                        {
                                            "method": event.method,
                                            "payload": event.payload,
                                        }
                                    ),
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            raw.flush()
                            incoming.put(event)
                            if event.method == "session.event":
                                data = event.payload.get("event", {})
                                if data.get("type") == "agent/inbox/spliced":
                                    received |= any(
                                        m.get("id") == receipt
                                        for m in data.get("data", {}).get(
                                            "inserted", []
                                        )
                                    )
                            if (
                                not interactive
                                and received
                                and event.method == "session.status"
                                and event.payload.get("status") == "idle"
                            ):
                                break
                except Exception as exc:
                    outcome["error"] = exc
                finally:
                    done.set()

            worker = threading.Thread(target=run, name="rpent-dsh", daemon=True)
            worker.start()
            stop_reason, error, active_elapsed = asyncio.run(
                self._drive(
                    recorder=recorder,
                    incoming=incoming,
                    outcome=outcome,
                    done=done,
                    boundary=boundary,
                    control_path=control_path,
                    started=started,
                    host_path=attachment_home / "host.sock",
                    run_id=work.name,
                    input_queue=input_queue,
                    interaction=dashboard_interaction,
                    probe=probe,
                )
            )
        except KeyboardInterrupt:
            process_interrupted = True
            stop_reason, error = "interrupted", "DeepSeek Harness interrupted"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            # Stop Python admission first. Closing dsh alone cannot stop a tool.
            boundary.stopped.set()
            try:
                boundary.cancel_active_and_wait()
            except Exception as exc:
                cleanup_ok = False
                error = error or f"Toolkit cleanup failed: {exc}"
            finally:
                if client is not None:
                    try:
                        client.close()
                    except Exception as exc:
                        cleanup_ok = False
                        error = error or f"DeepSeek Harness cleanup failed: {exc}"
                if worker is not None:
                    worker.join(timeout=5)
                    if worker.is_alive():
                        cleanup_ok = False
                        error = error or "DeepSeek Harness worker did not stop"
                try:
                    server.stop()
                except Exception as exc:
                    cleanup_ok = False
                    error = error or f"MCP cleanup failed: {exc}"
            if attachment_home is not None:
                try:
                    source = attachment_home / "attachments"
                    if source.exists():
                        shutil.copytree(source, work / "attachments")
                    shutil.rmtree(attachment_home)
                except OSError as exc:
                    cleanup_ok = False
                    attachment_retained = True
                    error = (
                        error
                        or f"Cannot archive dsh images; retained {attachment_home}: {exc}"
                    )
            if dashboard_interaction is not None:
                dashboard_interaction.seal_interaction()
        # The deadline or host control can end before the notification consumer
        # catches up. Reconcile everything actually received before freezing
        # delivery certainty; never assume an HTTP ledger write is a model ACK.
        while not incoming.empty():
            recorder.observe(incoming.get_nowait(), outcome.get("receipt", ""))
        result_delivery_complete = (
            not recorder.pending_calls
            and not recorder.delivery_errors
            and recorder.tool_results >= boundary.completed_calls
        )
        if client is not None:
            # The pinned SDK keeps a bounded stderr deque but has no public export.
            diagnostics = _redact("\n".join(getattr(client, "_stderr_lines", ())))
            (work / "runtime.stderr.log").write_text(diagnostics)
        if control_path.exists():
            try:
                control = json.loads(control_path.read_text())
            except (OSError, ValueError) as exc:
                error = error or f"Cannot read runtime control result: {exc}"
        result_delivery_complete &= control.get("stopReason") not in {
            "tool_transport_error",
            "image_pipeline_error",
        }
        reconcile_consumed = getattr(
            dashboard_interaction, "reconcile_consumed_message", None
        )
        if reconcile_consumed is not None:
            # Closing the interaction can precede the last user/message event.
            # Only a consumed receipt proves delivery; queued/discarded input
            # stays unsent. The state rejects IDs outside its current TaskRun.
            for receipt in control.get("receipts", {}).values():
                message_id = receipt.get("messageId")
                if receipt.get("status") == "consumed" and isinstance(message_id, str):
                    reconcile_consumed(message_id)
        if recorder.delivery_errors:
            stop_reason = "tool_transport_error"
            error = (
                error or "Runtime rejected a native tool result; delivery is incomplete"
            )
        elif control.get("stopReason") == "model_mismatch":
            stop_reason = "model_mismatch"
            error = (
                error
                or "Runtime model selection differs from the requested configuration"
            )
        elif control.get("stopReason") == "tool_transport_error":
            stop_reason = "tool_transport_error"
            error = error or "Tool outcome is uncertain; no further action was issued"
        elif control.get("stopReason") == "image_pipeline_error":
            stop_reason = "image_pipeline_error"
            error = (
                error
                or "Observation images could not reach the model without loss or coordinate changes"
            )
        elif (
            control.get("stopReason") == "budget"
            and stop_reason == "missing_finish"
            and recorder.turn_reason.get("kind") == "blocked"
            and cleanup_ok
            and not boundary.execution_uncertain
            and result_delivery_complete
            and "error" not in outcome
        ):
            # The worker can publish idle after _drive reads an older snapshot.
            # Reconcile that provisional classification using the final record;
            # never clear a transport, delivery, execution, or cleanup failure.
            stop_reason, error = "budget", None
        if control.get("startupError") and error:
            error += f"; bridge startup: {control['startupError']}"
        error = _redact(error) if error else None
        from rpent.planner.deepseek_context import consumed_operator_inputs

        operator_inputs = _sanitize(
            consumed_operator_inputs(
                outcome.get("operator_inputs", {}),
                recorder.consumed_user_receipts,
                control.get("receipts", {}),
            )
        )
        stats = {
            "backend": "deepseek_harness",
            "sdk_version": SDK_VERSION,
            "runtime_version": SDK_VERSION,
            "provider": PROVIDER,
            "model": self.model,
            "reasoning_effort": self.effort,
            "session_id": session_id,
            "stop_reason": stop_reason,
            "elapsed_s": round(time.monotonic() - started, 3),
            "active_elapsed_s": round(active_elapsed, 3),
            "turns_used": recorder.steps,
            "tool_calls": recorder.tool_calls,
            "completed_tool_calls": boundary.completed_calls,
            "observed_tool_results": recorder.tool_results,
            "tool_delivery_errors": recorder.delivery_errors,
            "raw_stream_path": str(raw_path),
            "control_path": str(control_path),
            "runtime_selection": control.get("observedSelection"),
            "inference": {
                "root_steps": recorder.steps,
                "failed_attempts": recorder.failed_attempts,
                "stream_requests": control.get("llmRequests", []),
                "provider_attempts": None,
                "retries_scheduled": recorder.retries_scheduled,
                "retries_started": recorder.retries_started,
                "usage_scope": "accepted_root_assistant_messages",
                "input_usage_complete": recorder._input_usage_complete,
                "output_usage_complete": recorder._output_usage_complete,
                "failed_attempts_with_unknown_usage": sum(
                    not attempt["usage_known"] for attempt in recorder.failed_attempts
                ),
                "image_offloads": recorder.image_offloads,
                "compactions": recorder.compactions,
            },
            "image_error": _redact(str(control["imageError"]))
            if control.get("imageError")
            else None,
            "attachment_archive": str(work / "attachments"),
            "attachment_scratch": str(attachment_home) if attachment_retained else None,
            **recorder.usage,
            **(
                {"consumed_operator_inputs": operator_inputs} if operator_inputs else {}
            ),
        }
        execution_certain = (
            cleanup_ok
            and not boundary.execution_uncertain
            and stop_reason
            not in {"tool_transport_error", "runtime_error", "control_error"}
        )
        stop_info = {
            "reason": stop_reason,
            # A host SIGINT ends the Dashboard process; an interaction pause does not.
            "process_interrupted": process_interrupted,
            "execution_certain": execution_certain,
            "result_delivery_complete": result_delivery_complete,
            "pending_tool_call_ids": sorted(recorder.pending_calls),
            "allow_new_session": execution_certain
            and result_delivery_complete
            and stop_reason in {"finish", "budget", "timeout"},
            "cleanup_complete": cleanup_ok,
        }
        stats["stop_info"] = stop_info
        (work / "result.json").write_text(
            json.dumps(
                {
                    "stats": stats,
                    "finish_result": _sanitize(boundary.finish_result),
                    "error": error,
                    "stop_info": stop_info,
                },
                indent=2,
            )
        )
        return PlannerResult(
            finish_result=_sanitize(boundary.finish_result),
            messages=recorder.messages,
            stats=stats,
            error=error,
            stop_info=stop_info,
        )

    async def _drive(
        self,
        *,
        recorder,
        incoming,
        outcome,
        done,
        boundary,
        control_path,
        started,
        host_path,
        run_id,
        input_queue,
        interaction,
        probe,
    ):
        """Own event projection and interactive state on one asynchronous loop."""
        from rpent.cli.tui import QUIT_TOKENS
        from rpent.dashboard.interaction import DashboardMessage
        from rpent.dashboard.planner_control import DashboardPlannerControl
        from rpent.planner.base import cancel_and_wait
        from rpent.planner.deepseek_session import DeepSeekSessionDriver, HostControl

        interactive = input_queue is not None or interaction is not None
        driver = DeepSeekSessionDriver(
            HostControl(host_path, run_id),
            boundary,
            operator_inputs=outcome.setdefault("operator_inputs", {}),
        )
        dashboard = None
        dashboard_task = None
        active_elapsed = time.monotonic() - started
        tick = time.monotonic()
        active = True
        pending_completions = 0
        waiting = False
        state = {}

        def emit_user(text):
            self.sink.emit(TranscriptEvent({"type": "user", "text": text}))

        try:
            while True:
                now = time.monotonic()
                if active or not interactive:
                    active_elapsed += now - tick
                tick = now
                try:
                    state = json.loads(control_path.read_text())
                except FileNotFoundError:
                    # Some shared filesystems briefly hide the destination
                    # during atomic replacement. Runtime policy still owns
                    # admission; retain the last snapshot until the next poll.
                    pass
                if (
                    interaction is not None
                    and dashboard is None
                    and "receipt" in outcome
                ):
                    dashboard = DashboardPlannerControl(
                        interaction=interaction,
                        cancel_active_and_wait=boundary.cancel_active_and_wait,
                        # Keep model admission closed while manual operations
                        # are allowed at a paused boundary.
                        resume_calls=boundary.toolkit.resume_calls,
                        emit_user=emit_user,
                        emit_initial_user=lambda: self.sink.emit(
                            TranscriptEvent({"type": "initial_prompt"})
                        ),
                        defer_message_ack=True,
                        submit_while_busy=True,
                    )
                    driver.control = dashboard
                    await dashboard.start()
                    dashboard_task = asyncio.create_task(dashboard.run(driver))
                while not incoming.empty():
                    notification = incoming.get_nowait()
                    recorder.observe(notification, outcome.get("receipt", ""))
                    if notification.method == "session.event":
                        event = notification.payload.get("event", {})
                        kind, data = event.get("type"), event.get("data", {})
                        if kind == "user/message":
                            driver.message_consumed(data.get("id", ""))
                        elif kind == "turn/end":
                            pending_completions += 1
                if recorder.delivery_errors:
                    boundary.stopped.set()
                    return (
                        "tool_transport_error",
                        "Runtime rejected a native tool result; continuation is disabled",
                        active_elapsed,
                    )
                terminal = state.get("stopReason")
                if terminal in {
                    "model_mismatch",
                    "tool_transport_error",
                    "image_pipeline_error",
                    "control_error",
                    "host_closed",
                    "compaction_error",
                }:
                    return (
                        terminal,
                        f"DeepSeek Harness stopped: {terminal}",
                        active_elapsed,
                    )
                if interactive and state.get("quiescent") and dashboard is not None:
                    while pending_completions:
                        pending_completions -= 1
                        await dashboard.complete(driver)
                if driver.error:
                    return "control_error", driver.error, active_elapsed
                if driver.stop_reason == "budget":
                    return "budget", None, active_elapsed
                if interaction is not None and getattr(
                    interaction, "manual_execution_uncertain", False
                ):
                    boundary.execution_uncertain = True
                    boundary.stopped.set()
                    return (
                        "tool_transport_error",
                        "Manual action outcome is uncertain; continuation is disabled",
                        active_elapsed,
                    )
                if "error" in outcome:
                    raise outcome["error"]
                if boundary.finish_result is not None and (
                    recorder.idle or done.is_set()
                ):
                    return "finish", None, active_elapsed
                if terminal == "budget" and (recorder.idle or done.is_set()):
                    return "budget", None, active_elapsed
                if not interactive and done.is_set():
                    if probe and any(
                        m.get("role") == "assistant" for m in recorder.messages
                    ):
                        return "response", None, active_elapsed
                    return (
                        "missing_finish",
                        f"DeepSeek Harness stopped without accepted finish: {recorder.turn_reason}",
                        active_elapsed,
                    )
                if (
                    interactive
                    and recorder.idle
                    and recorder.turn_reason.get("kind") == "error"
                ):
                    return (
                        "runtime_error",
                        f"DeepSeek Harness failed: {recorder.turn_reason}",
                        active_elapsed,
                    )
                if interaction is not None and interaction.planner_activity == "ended":
                    return (
                        "task_replaced"
                        if interaction.task_replacement_requested
                        else "interrupted",
                        None,
                        active_elapsed,
                    )
                if dashboard_task is not None and dashboard_task.done():
                    dashboard_task.result()
                active = not (interactive and state.get("quiescent"))
                if input_queue is not None:
                    if not active and not waiting and not driver.paused:
                        logger.info(
                            "dsh waiting for input; /interrupt pauses, /quit ends the run"
                        )
                    waiting = not active
                if input_queue is not None and "receipt" in outcome:
                    while not input_queue.empty():
                        line = input_queue.get_nowait()
                        if line is None or line.strip().lower() in QUIT_TOKENS:
                            return "interrupted", None, active_elapsed
                        line = line.strip()
                        if not line:
                            continue
                        if line.lower() == "/interrupt":
                            await cancel_and_wait(boundary.cancel_active_and_wait)
                            await driver.interrupt()
                            logger.info("dsh paused; enter a message to continue")
                        else:
                            await driver.submit(
                                DashboardMessage(uuid.uuid4().hex, line, "sending")
                            )
                            emit_user(line)
                            recorder.idle = False
                        active = True
                if active_elapsed >= self.timeout_s:
                    return (
                        "timeout",
                        f"DeepSeek Harness timed out after {self.timeout_s}s of execution",
                        active_elapsed,
                    )
                await asyncio.sleep(0.02)
        finally:
            if dashboard is not None:
                dashboard.end()
            if dashboard_task is not None:
                # Sealing wakes the condition waiter; retain controller cleanup
                # rather than orphaning its thread or cancellation operation.
                await dashboard_task

    def _write_patch(
        self,
        work,
        control_path,
        url,
        boundary,
        system_prompt,
        max_steps,
        attachment_home,
    ):
        # JSON is a YAML subset: prompts, URLs, and paths cannot alter patch syntax.
        disabled = (
            "tool-plugin-manager",
            "plugin-manager",
            "hmr",
            "tool-working-directory",
            "tool-bash",
            "tool-pwsh",
            "tool-jobs",
            "tool-fs",
            "tool-fs-search",
            "agent-instructions",
            "skill",
            "skill-filesystem",
            "tool-skill",
            "skill-office",
            "workspace-dependencies",
            "goal",
            "goal-round-driver",
            "command-goal",
            "plan-mode",
            "subagent",
            "subagent-spawn-in-process",
            "subagent-fork-in-process",
            "tool-subagent-control",
            "tool-subagent-list-agents",
            "tool-subagent",
            "tool-subagent-fork",
            "ptc-runtime",
            "workflow-ptc",
            "tool-workflow",
            "tool-todo",
            "tool-goal",
            "tool-web",
            "mcp-resources",
            "session-telemetry-otel",
            "session-title-llm",
        )
        rows = [{"id": name, "disabled": True} for name in disabled]
        rows += [
            {"id": "attachment-local", "config": {"dshHome": str(attachment_home)}},
            {"id": "tools", "config": {"mode": "native"}},
            {"id": "agent-loop", "config": {"agents": [], "maxParallelToolCalls": 1}},
            {
                "id": "llm-pi-ai",
                "config": {
                    "providers": {
                        PROVIDER: {"retryPolicy": {"mode": "normal", "maxRetries": 2}}
                    }
                },
            },
            {
                "id": "sdk-jsonrpc-server",
                "inject": ["sdkAppStartup", "loader", "rpentBridgeReady"],
            },
            {
                "insert": [
                    {
                        "id": "mcp-rpent",
                        "name": "@deepseek-ai/dsh-mcp-client",
                        "config": {
                            "serverName": "rpent",
                            "transport": "streamable-http",
                            "url": url,
                            "toolCallTimeoutMs": int(self.timeout_s * 1000),
                            "failOnStartupError": True,
                            "reconnect": {"enabled": False},
                        },
                    },
                    {
                        "id": "rpent-bridge",
                        "name": str(Path(__file__).with_name("deepseek_bridge.mjs")),
                        "config": {
                            "toolNames": [
                                add_mcp_prefix(t.name) for t in boundary.list_tools()
                            ],
                            "systemPrompt": system_prompt,
                            "maxSteps": max_steps,
                            "controlPath": str(control_path),
                            "toolLedgerPath": str(boundary.ledger_path),
                            "hostControl": {
                                "socketPath": str(attachment_home / "host.sock"),
                                "runId": work.name,
                            },
                            "expectedSelection": {
                                "provider": PROVIDER,
                                "model": self.model,
                                "reasoningEffort": self.effort,
                            },
                        },
                    },
                ]
            },
        ]
        path = work / "rpent.patch.json"
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        return path


def run_probe_turn(
    *,
    prompt: str,
    model: str,
    dsh_home: str | Path | None,
    reasoning_effort: str,
    timeout_s: float,
) -> str:
    """Probe the same composition and credential route with no robot tools."""
    from rpent.dashboard.events import NullDashboardEventSink

    class EmptyToolkit:
        def list_tools(self):
            return ()

        def cancel_active_and_wait(self):
            pass

    with tempfile.TemporaryDirectory(prefix="rpent-dsh-check-") as output:
        planner = DeepSeekHarnessPlanner(
            output_dir=output,
            recipe_tag="check",
            model=model,
            dsh_home=dsh_home,
            reasoning_effort=reasoning_effort,
            timeout_s=timeout_s,
            dashboard_events=NullDashboardEventSink(),
        )
        result = planner._solve(
            "Answer the user's request.", prompt, EmptyToolkit(), 1, probe=True
        )
        if result.error:
            if result.stats.get("stop_reason") == "timeout":
                raise TimeoutError(result.error)
            raise RuntimeError(result.error)
        return "".join(
            b.get("text", "")
            for m in result.messages
            if m.get("role") == "assistant"
            for b in m.get("content", [])
            if b.get("type") == "text"
        )
