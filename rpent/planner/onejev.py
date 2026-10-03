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
"""Finite-action planner driven by an external OneJev choice-scoring API."""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import jsonschema

from rpent.dashboard.events import DashboardEventSink, TranscriptEvent, UsageEvent
from rpent.planner.base import Planner, PlannerResult
from rpent.planner.onejev_client import OneJevClient
from rpent.planner.onejev_types import DecisionAdapter
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger

logger = get_logger("onejev")


class _DecisionTrace:
    """Write the three readable decision artifacts through RPent's EnvState."""

    def __init__(self, toolkit: Toolkit) -> None:
        self._state = toolkit.state
        self.records: dict[str, list[dict[str, Any]]] = {
            "onejev_state.json": [],
            "onejev_question.json": [],
            "onejev_decision.json": [],
        }
        self.entry: dict[str, Any] = {}
        self.started = 0.0
        self.active = False
        for name in self.records:
            self.save(name)

    def save(self, name: str = "onejev_decision.json") -> None:
        """Persist one trace as an indented JSON array."""
        if (
            self._state.save(
                name, self.records[name], step=None, indent=4, ensure_ascii=False
            )
            is None
        ):
            raise RuntimeError(f"Could not save OneJev trace: {name}")

    def start(self, turn: int) -> dict[str, Any]:
        """Begin a decision before the HTTP request is sent."""
        self.entry = {"turn": turn, "status": "requested"}
        self.started = time.monotonic()
        self.active = True
        return self.entry

    def record_request(self, request: dict[str, Any]) -> None:
        """Record exactly what the client is about to send."""
        turn = self.entry["turn"]
        self.records["onejev_state.json"].append(
            {
                "turn": turn,
                "state": request["state"],
                "media_count": len(request["media"]),
            }
        )
        self.records["onejev_question.json"].append(
            {
                "turn": turn,
                "model": request["model"],
                "questions": request["questions"],
            }
        )
        self.records["onejev_decision.json"].append(self.entry)
        for name in self.records:
            self.save(name)

    def record_response(self, response: dict[str, Any]) -> None:
        """Keep the raw response even when subsequent validation fails."""
        self.entry.update(response)
        self.entry["status"] = "responded"
        self.entry["latency_s"] = round(time.monotonic() - self.started, 3)
        self.save()


class OneJevPlanner(Planner):
    """Select computed candidates, execute through Toolkit, and record evidence."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        adapter_factory: Callable[[Toolkit], DecisionAdapter],
        timeout_s: float,
        dashboard_events: DashboardEventSink,
    ) -> None:
        """Remember the external endpoint and the robot's adapter factory."""
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("OneJev planner timeout must be finite and positive")
        self._base_url = base_url
        self._model = model
        self._adapter_factory = adapter_factory
        self._timeout_s = timeout_s
        self._events = dashboard_events

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue=None,
        dashboard_interaction=None,
    ) -> PlannerResult:
        """Run bounded choice decisions using only robot-owned observations."""
        if input_queue is not None or dashboard_interaction is not None:
            return PlannerResult(
                error="OneJev currently supports non-interactive evaluation"
            )
        started = time.monotonic()
        deadline = started + self._timeout_s
        messages: list[dict[str, Any]] = []
        trace = None
        stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "turns_used": 0,
            "tool_calls": 0,
        }
        finish_result = None
        error = None
        client = None

        def emit(payload: dict[str, Any]) -> None:
            messages.append(payload)
            self._events.emit(TranscriptEvent(payload=payload))

        def finish(outcome: dict[str, str]) -> dict[str, Any]:
            result = toolkit.execute_tool("finish", outcome)
            stats["tool_calls"] += 1
            emit({"role": "tool", "name": "finish", "content": result.result})
            if not result.is_finish:
                raise RuntimeError(f"Toolkit refused finish: {result.result}")
            return {
                key: value for key, value in result.result.items() if key != "_finish"
            }

        try:
            if max_turns <= 0:
                raise ValueError("OneJev --max-turns must be positive")
            trace = _DecisionTrace(toolkit)
            adapter = self._adapter_factory(toolkit)
            schemas = {
                item["name"]: item["input_schema"] for item in toolkit.get_tools_spec()
            }
            client = OneJevClient(self._base_url, model=self._model)
            while stats["turns_used"] < max_turns:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    finish_result = finish(
                        {"status": "stuck", "summary": "OneJev planner timed out"}
                    )
                    error = "OneJev planner timed out"
                    break
                toolkit.raise_if_cancelled()
                context = adapter.prepare()
                if context.outcome is not None:
                    finish_result = finish(context.outcome)
                    break
                if not context.candidates:
                    finish_result = finish(
                        {
                            "status": "stuck",
                            "summary": "No admissible action candidates",
                        }
                    )
                    break
                for candidate in context.candidates:
                    if candidate.tool_name not in schemas:
                        raise ValueError(
                            f"Unknown candidate tool: {candidate.tool_name}"
                        )
                    jsonschema.validate(
                        candidate.arguments, schemas[candidate.tool_name]
                    )
                    json.dumps(candidate.arguments, allow_nan=False)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("OneJev planner timed out before scoring")
                entry = trace.start(stats["turns_used"] + 1)

                scores = client.score(
                    context,
                    timeout_s=min(120.0, remaining),
                    on_request=trace.record_request,
                    on_response=trace.record_response,
                )
                stats["turns_used"] += 1
                stats["total_input_tokens"] += scores.usage["input_tokens"]
                stats["total_output_tokens"] += scores.usage["output_tokens"]
                selected = max(
                    context.candidates, key=lambda item: scores.probabilities[item.id]
                )
                entry.update(
                    {
                        "status": "scored",
                        "selected": selected.id,
                        "selected_tool_name": selected.tool_name,
                        "selected_action": selected.as_dict(),
                    }
                )
                trace.save()
                emit(
                    {
                        "role": "assistant",
                        "content": {
                            "selected": selected.as_dict(),
                            "probabilities": scores.probabilities,
                        },
                    }
                )
                logger.info(
                    "decision %d: %s %s",
                    stats["turns_used"],
                    selected.tool_name,
                    selected.arguments,
                )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "OneJev planner timed out before action execution"
                    )
                toolkit.raise_if_cancelled()
                timer = threading.Timer(
                    max(0.0, deadline - time.monotonic()),
                    toolkit.cancel_active_and_wait,
                )
                timer.daemon = True
                timer.start()
                try:
                    result = toolkit.execute_tool(
                        selected.tool_name, selected.arguments
                    )
                finally:
                    timer.cancel()
                stats["tool_calls"] += 1
                record = toolkit.state.latest_record()
                entry["execution"] = {
                    "step_idx": record.step_idx
                    if record and not result.is_finish
                    else None,
                    "result": (
                        record.result
                        if record and not result.is_finish
                        else result.result
                    ),
                }
                entry["status"] = "executed"
                trace.save()
                adapter.observe(selected, result)
                # Images already belong to EnvState; do not duplicate base64 in messages.
                emit(
                    {
                        "role": "tool",
                        "name": selected.tool_name,
                        "content": entry["execution"],
                    }
                )
                self._events.emit(
                    UsageEvent(
                        inp=stats["total_input_tokens"],
                        out=stats["total_output_tokens"],
                        tool_calls=stats["tool_calls"],
                    )
                )
                trace.active = False
                if result.is_finish:
                    finish_result = {
                        key: value
                        for key, value in result.result.items()
                        if key != "_finish"
                    }
                    break
            else:
                # Last motion may have achieved native success on the last allowed turn.
                context = adapter.prepare()
                finish_result = finish(
                    context.outcome
                    or {
                        "status": "stuck",
                        "summary": "OneJev decision budget exhausted",
                    }
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.error("OneJev planner stopped: %s", error)
            if trace is not None and trace.active:
                trace.entry.update({"status": "error", "error": error})
                try:
                    trace.save()
                except Exception as trace_exc:
                    logger.warning("could not record decision error: %s", trace_exc)
            try:
                finish_result = finish({"status": "failure", "summary": error})
            except Exception as finish_exc:
                logger.warning("could not record finish: %s", finish_exc)
        finally:
            if client is not None:
                client.close()
        stats["elapsed_s"] = round(time.monotonic() - started, 3)
        toolkit.state.save(
            "onejev_outcome.json",
            {"finish": finish_result, "error": error, "stats": stats},
            step=None,
        )
        return PlannerResult(
            finish_result=finish_result, messages=messages, stats=stats, error=error
        )
