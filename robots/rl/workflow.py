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

import asyncio
import time
from dataclasses import replace

from pydantic_ai import BinaryContent, ModelRetry, ToolCallPart
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart

from rpent.planner.request_trace import write_json


class WorkflowCapability(AbstractCapability):
    """Keep one global request budget while replacing only visible context."""

    def __init__(
        self,
        toolkit,
        system_prompt: str,
        user_message: str,
        max_turns: int,
        *,
        timeout_s: float,
    ) -> None:
        self.toolkit = toolkit
        self.system_prompt, self.user_message = system_prompt, user_message
        self.max_turns, self.timeout_s = max_turns, timeout_s
        self.started = time.perf_counter()
        self.revision = 0
        self.context_id = 0
        self.required: set[str] = set()
        self.allowed = {item.name: item for item in toolkit.list_tools()}
        self.stats = toolkit.workflow_stats
        self.stats.update(context_resets=0, invalid_tool_responses=0, model_seconds=0.0)

    async def before_model_request(self, ctx, request_context):
        """Prepare fresh snapshots or retain the current episode/subtask history."""
        toolkit, snapshot = self.toolkit, self.toolkit.snapshot
        turn = ctx.usage.requests + 1
        reset = toolkit.workflow == "zap_turn" or (
            toolkit.workflow == "zap_subtask"
            and snapshot.todo_revision != self.revision
        )
        if reset:
            observation = await asyncio.to_thread(toolkit.observe_snapshot)
            observation.data["budget"] = {
                "request": turn,
                "requests_remaining_including_current": self.max_turns - turn + 1,
                "seconds_remaining": max(
                    0, self.timeout_s - (time.perf_counter() - self.started)
                ),
            }
            toolkit._count_images(observation)
            text = f"{self.user_message}\n\n{observation.to_text()}"
            request_context.messages = [
                ModelRequest(
                    parts=[
                        SystemPromptPart(self.system_prompt),
                        UserPromptPart(
                            [
                                text,
                                *[
                                    BinaryContent(data=image, media_type="image/png")
                                    for image in observation.images
                                ],
                            ]
                        ),
                    ]
                )
            ]
            if turn > 1:
                self.context_id += 1
                self.stats["context_resets"] += 1
            self.revision = snapshot.todo_revision
        self.required = set()
        if snapshot is not None:
            if toolkit.workflow == "zap_subtask" and not snapshot.todo:
                self.required = {"change_query", "update_todo"}
            elif not snapshot.queries:
                self.required = {"change_query"}
            elif toolkit.workflow == "zap_turn" and not snapshot.todo:
                self.required = {"update_todo"}
        parameters = request_context.model_request_parameters
        request_context.model_request_parameters = replace(
            parameters,
            function_tools=[
                tool
                for tool in parameters.function_tools
                if not self.required or tool.name in self.required
            ],
            output_tools=[] if self.required else parameters.output_tools,
            allow_text_output=False,
        )
        request_context.model_settings = {
            **(request_context.model_settings or {}),
            "parallel_tool_calls": len(self.required) == 2,
        }
        from rpent.planner.qwen import is_qwen_model

        prepared = request_context.model_request_parameters
        definitions = [*prepared.function_tools, *prepared.output_tools]
        bounded = is_qwen_model(
            getattr(request_context.model, "model_name", "")
        ) and any(
            t.parameters_json_schema.get("type") == "object"
            and not t.parameters_json_schema.get("properties")
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
                    definitions, parallel=len(self.required) == 2
                ),
            }
        self.turn = turn
        self.request_started = time.perf_counter()
        write_json(
            toolkit.output_dir / "workflow_requests" / f"{turn:04d}.json",
            {
                "request": turn,
                "context_id": self.context_id,
                "reset": reset,
                "todo_revision": self.revision,
                "required_tools": sorted(self.required),
                "workflow": toolkit.workflow,
                "image_size": toolkit.image_size,
                "bounded_qwen_xml": bounded,
                "history_messages_before_compaction": len(request_context.messages),
            },
        )
        return request_context

    async def after_model_request(self, ctx, *, request_context, response):
        """Reject the whole reply before any sibling tool can change state."""
        elapsed = time.perf_counter() - self.request_started
        self.stats["model_seconds"] += elapsed
        trace = getattr(request_context.model, "request_trace", None)
        if trace is not None:
            trace.response(response)
        calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
        error = None
        if response.state != "complete" or response.finish_reason == "length":
            raise UnexpectedModelBehavior(
                "Token limit reached before any response was generated"
            )
        names = [call.tool_name for call in calls]
        if self.required:
            if len(names) != len(self.required) or set(names) != self.required:
                error = f"This request must call exactly: {', '.join(sorted(self.required))}."
        elif len(calls) != 1:
            error = "Emit exactly one available tool call."
        if error is None:
            for call in calls:
                try:
                    if call.tool_name not in self.allowed:
                        raise ValueError(f"unknown tool {call.tool_name}")
                    arguments = call.args_as_dict()
                    parameters = self.allowed[call.tool_name].validate_arguments(
                        arguments
                    )
                    if call.tool_name == "observe":
                        self.toolkit.observer.compile(parameters.calls)
                    from robots.rl.contracts import validate_relations

                    validate_relations(call.tool_name, arguments)
                except (ValueError, TypeError) as exc:
                    from robots.rl.contracts import argument_hint

                    error = f"Invalid tool arguments: {exc}. {argument_hint(call.tool_name, getattr(self.toolkit, 'environment', 'libero'))}"
                    break
        path = self.toolkit.output_dir / "workflow_responses" / f"{self.turn:04d}.json"
        write_json(
            path,
            {
                "request": self.turn,
                "context_id": self.context_id,
                "model_seconds": elapsed,
                "tools": names,
                "error": error,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )
        if error:
            self.stats["invalid_tool_responses"] += 1
            feedback = error + " No tool executed."
            if self.toolkit.snapshot is not None:
                self.toolkit.snapshot.last_result = {"error": feedback}
            raise ModelRetry(feedback)
        return response
