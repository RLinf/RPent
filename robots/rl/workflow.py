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

"""RL conversation boundaries and response gates on the native API agent loop."""

from __future__ import annotations

import time

from pydantic_ai import ModelRetry, ToolCallPart
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UnexpectedModelBehavior

from robots.rl.contracts import argument_hint, validate_relations
from rpent.planner.api_loop import _InvalidToolArguments


class WorkflowCapability(AbstractCapability):
    """Validate complete replies before executing any tool."""

    def __init__(self, toolkit) -> None:
        self.toolkit = toolkit
        self.allowed = {item.name: item for item in toolkit.list_tools()}
        self.stats = toolkit.workflow_stats
        self.stats.update(invalid_tool_responses=0, model_seconds=0.0)

    async def before_model_request(self, ctx, request_context):
        """Allow a final reply or a single validated tool call."""
        request_context.model_settings = {
            **(request_context.model_settings or {}),
            "parallel_tool_calls": False,
        }
        from rpent.planner.qwen import is_qwen_model

        prepared = request_context.model_request_parameters
        definitions = [*prepared.function_tools, *prepared.output_tools]
        bounded = is_qwen_model(
            getattr(request_context.model, "model_name", "")
        ) and any(
            t.parameters_json_schema.get("type") == "object"
            and not t.parameters_json_schema.get("required")
            and t.parameters_json_schema.get("additionalProperties") is False
            for t in definitions
        )
        if bounded:
            from robots.rl.qwen_grammar import bounded_tool_format

            settings = request_context.model_settings
            settings["extra_body"] = {
                **(settings.get("extra_body") or {}),
                "tool_choice": "auto",
                "response_format": bounded_tool_format(
                    definitions,
                    parallel=False,
                    require_tool=False,
                    fractional_bounds_at_execution=True,
                ),
            }
        self.request_started = time.perf_counter()
        return request_context

    async def after_model_request(self, ctx, *, request_context, response):
        """Reject the whole reply before any sibling tool can change state."""
        elapsed = time.perf_counter() - self.request_started
        self.stats["model_seconds"] += elapsed
        trace = getattr(request_context.model, "request_trace", None)
        if trace is not None:
            trace.response(response)
        calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
        if response.state != "complete" or response.finish_reason == "length":
            raise UnexpectedModelBehavior(
                "Token limit reached before any response was generated"
            )
        if len(calls) > 1:
            self.stats["invalid_tool_responses"] += 1
            raise _InvalidToolArguments(
                "Use one tool call per response. No tool executed."
            )
        for call in calls:
            if call.tool_name not in self.allowed:
                self.stats["invalid_tool_responses"] += 1
                raise ModelRetry(
                    f"Unknown tool name: {call.tool_name}. No tool executed."
                )
            try:
                arguments = call.args_as_dict()
                self.allowed[call.tool_name].validate_arguments(arguments)
                validate_relations(call.tool_name, arguments)
            except (AssertionError, ValueError, TypeError) as exc:
                self.stats["invalid_tool_responses"] += 1
                hint = argument_hint(call.tool_name, self.toolkit.environment)
                raise _InvalidToolArguments(
                    f"Invalid tool arguments: {exc}. {hint} No tool executed."
                ) from exc
        return response
