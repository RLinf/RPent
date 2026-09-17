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

"""API planner composed from Pydantic AI and Harness capabilities.

The SDK owns model requests, tool scheduling, retries, queued-message delivery,
and cancellation. Harness owns compaction; clai owns terminal interaction.
RPent's runner saves the returned transcript through its existing output flow.
"""

import asyncio
import base64
import threading
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from functools import partial
from typing import Any
from uuid import uuid4

from jsonschema import Draft202012Validator
from pydantic_ai import (
    Agent,
    BinaryContent,
    CallToolsNode,
    CancellationToken,
    EnqueuedMessagesEvent,
    FunctionToolset,
    ModelRequestNode,
    ModelRetry,
    ModelSettings,
    PartEndEvent,
    RunCancelled,
    RunContext,
    RunUsage,
    StructuredDict,
    TextPart,
    ThinkingPart,
    Tool,
    ToolCallPart,
    ToolOutput,
    ToolReturn,
    UsageLimits,
)
from pydantic_ai.agent import WrapperAgent
from pydantic_ai.capabilities import AbstractCapability, Thinking, on_event
from pydantic_ai.capabilities.abstract import AgentNode, NodeResult, WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, UserContent
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.run import AgentRun, AgentRunResult
from pydantic_ai_harness.compaction import SlidingWindowCompaction

from rpent.dashboard.events import DashboardEventSink, TranscriptEvent, UsageEvent
from rpent.dashboard.interaction import DashboardInteractionPort, DashboardMessage
from rpent.dashboard.planner_control import DashboardPlannerControl
from rpent.planner.base import REASONING_EFFORTS, PlannerResult
from rpent.tools.toolkit import Toolkit, ToolResult
from rpent.utils.logging import get_logger

logger = get_logger("api")


class ApiAgentLoop:
    """Implement ``Planner`` with a native Agent and reusable Harness capabilities."""

    def __init__(
        self,
        *,
        model: Model,
        dashboard_events: DashboardEventSink,
        max_tokens: int = 8192,
        reasoning_effort: str = "none",
        no_images: bool = False,
        timeout_s: float = 1200,
        interactive: bool = False,
    ) -> None:
        if reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"unsupported reasoning effort: {reasoning_effort}")
        if max_tokens < 1 or timeout_s <= 0:
            raise ValueError("max_tokens and timeout_s must be positive")
        self.model = model
        self.dashboard_events = dashboard_events
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.no_images = no_images
        self.timeout_s = timeout_s
        self.interactive = interactive

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue=None,
        dashboard_interaction: DashboardInteractionPort | None = None,
    ) -> PlannerResult:
        """Run a conversation and return its transcript for RPent to save."""
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        if input_queue is not None:
            raise ValueError("api does not use RPent's terminal input queue")
        if self.interactive and dashboard_interaction is not None:
            raise ValueError("clai and Dashboard cannot run together")
        return asyncio.run(
            self._solve(
                system_prompt,
                user_message,
                toolkit,
                max_turns,
                dashboard_interaction,
            )
        )

    async def _solve(
        self,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        interaction: DashboardInteractionPort | None,
    ) -> PlannerResult:
        conversation_id = uuid4().hex
        adapter = _HarnessToolkit(
            toolkit, dashboard_events=self.dashboard_events, no_images=self.no_images
        )
        session = _Session(
            adapter, conversation_id, max_turns, interactive=self.interactive
        )
        agent = Agent(
            self.model,
            name="rpent_api",
            system_prompt=system_prompt,
            output_type=adapter.output_types,
            toolsets=[adapter],
            # Accepted finish skips sibling tool calls in the same response.
            end_strategy="early",
            # Finish refusals may span attempts; the request budget still applies.
            retries={"output": max_turns},
            model_settings=_build_model_settings(self.model, self.max_tokens),
            capabilities=[
                SlidingWindowCompaction(
                    max_messages=80, max_fraction=0.8, keep_messages=40
                ),
                Thinking(
                    False if self.reasoning_effort == "none" else self.reasoning_effort
                ),
                session,
            ],
        )
        if interaction is not None:
            session.control = DashboardPlannerControl(
                interaction=interaction,
                cancel_active_and_wait=adapter.cancel_active_and_wait,
                emit_user=session.emit_user,
                emit_initial_user=lambda: session.emit_user(user_message),
                defer_message_ack=True,
                submit_while_busy=True,
            )
        try:
            await asyncio.wait_for(
                session.run(_ConversationAgent(agent), user_message),
                timeout=None if self.interactive else self.timeout_s,
            )
        except asyncio.TimeoutError:
            session.error = f"planner timed out after {self.timeout_s:g}s"
        except Exception as exc:
            session.error = f"{type(exc).__name__}: {exc}"
            logger.exception("Harness planner failed")
        finally:
            session.emit_usage()
        return PlannerResult(
            finish_result=adapter.finish_result,
            messages=adapter.messages,
            stats={
                "total_input_tokens": session.usage.input_tokens,
                "total_output_tokens": session.usage.output_tokens,
                "total_cached_input_tokens": session.usage.cache_read_tokens,
                "turns_used": session.usage.requests,
                "tool_calls": adapter.tool_calls,
            },
            error=session.error,
        )


def _build_model_settings(model: Model, max_tokens: int) -> ModelSettings:
    """Configure sequential calls and preserve Anthropic prompt caching."""
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

    if isinstance(model, AnthropicModel):
        return AnthropicModelSettings(
            max_tokens=max_tokens,
            parallel_tool_calls=False,
            anthropic_cache_instructions=True,
            anthropic_cache_tool_definitions=True,
            anthropic_cache_messages=True,
        )
    return ModelSettings(max_tokens=max_tokens, parallel_tool_calls=False)


class _ConversationAgent(WrapperAgent):
    """Preserve completed operations across failed and cancelled runs.

    This checkpoint takes precedence over the CLI's last successful history.
    """

    def __init__(self, agent: Agent) -> None:
        super().__init__(agent)
        self.history: list[ModelMessage] | None = None

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[UserContent] | None = None,
        *,
        message_history: Sequence[ModelMessage] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentRun]:
        run: AgentRun | None = None
        try:
            async with self.wrapped.iter(
                user_prompt,
                message_history=(
                    self.history if self.history is not None else message_history
                ),
                **kwargs,
            ) as run:
                yield run
        finally:
            # Read after the SDK exits so cancellation cleanup is included.
            if run is not None:
                self.history = run.all_messages()


class _HarnessToolkit(FunctionToolset):
    """Translate schemas and multimodal results without bypassing Toolkit."""

    def __init__(
        self,
        toolkit: Toolkit,
        *,
        dashboard_events: DashboardEventSink,
        no_images: bool,
    ) -> None:
        super().__init__(id="rpent")
        self.toolkit = toolkit
        self.dashboard_events = dashboard_events
        self.no_images = no_images
        self.finish_result: dict[str, Any] | None = None
        self.tool_calls = 0
        self.messages: list[dict[str, Any]] = []
        self.stopping = threading.Event()
        self.validators: dict[str, Draft202012Validator] = {}
        self.output_types: list[Any] = [str]
        for spec in toolkit.get_tools_spec():
            name = spec["name"]
            self.validators[name] = Draft202012Validator(spec["input_schema"])
            if name == "finish":
                # Eager annotations bind this toolkit's schema to the output
                # function. Keep this module free of postponed annotations.
                FinishArguments = StructuredDict(
                    spec["input_schema"], name="FinishArguments"
                )

                async def finish(
                    ctx: RunContext, arguments: FinishArguments
                ) -> dict[str, Any]:
                    result = await self.execute("finish", arguments, ctx)
                    if not result.is_finish:
                        raise ModelRetry(self._text(result))
                    self.finish_result = result.result
                    return result.result

                self.output_types.append(
                    ToolOutput(
                        finish, name="finish", description=spec.get("description")
                    )
                )
            else:
                self.add_tool(
                    Tool.from_schema(
                        partial(self.call, name),
                        name=name,
                        description=spec.get("description"),
                        json_schema=spec["input_schema"],
                        takes_ctx=True,
                        sequential=True,
                    )
                )

    async def execute(
        self, name: str, arguments: dict[str, Any], ctx: RunContext
    ) -> ToolResult:
        """Validate and run one physical operation, retaining cancellation ownership."""
        error = next(self.validators[name].iter_errors(arguments), None)
        if error is not None:
            raise ModelRetry(f"Invalid arguments for {name}: {error.message}")
        if self.stopping.is_set():
            raise asyncio.CancelledError
        self.dashboard_events.emit(
            TranscriptEvent({"type": "tool_call", "tool": name, "args": arguments})
        )
        self.tool_calls += 1
        operation = asyncio.create_task(
            asyncio.to_thread(self.toolkit.execute_tool, name, arguments)
        )
        try:
            result = await asyncio.shield(operation)
        except asyncio.CancelledError:
            self.stopping.set()
            await asyncio.to_thread(self.cancel_active_and_wait)
            # A cancelled asyncio task does not stop the physical worker.
            # Drain it before another run may use the same toolkit.
            await operation
            raise
        text = self._text(result)
        self.messages.append(
            {
                "role": "tool",
                "name": name,
                "tool_call_id": ctx.tool_call_id,
                "content": text,
            }
        )
        self.dashboard_events.emit(
            TranscriptEvent({"type": "tool_result", "tool": name, "result": text})
        )
        return result

    async def call(self, name: str, ctx: RunContext, /, **arguments: Any) -> ToolReturn:
        """Return native multimodal content for an ordinary tool."""
        result = await self.execute(name, arguments, ctx)
        content: list[str | BinaryContent] = []
        for block in result.content_blocks:
            if block["type"] == "text":
                content.append(block["text"])
            elif block["type"] == "image" and self.no_images:
                content.append("[Image omitted: --no-images is enabled.]")
            elif block["type"] == "image":
                source = block["source"]
                content.append(
                    BinaryContent(
                        data=base64.b64decode(source["data"]),
                        media_type=source["media_type"],
                    )
                )
        return ToolReturn(return_value=content)

    def cancel_active_and_wait(self) -> None:
        """Stop admitting tools before requesting physical cancellation."""
        self.stopping.set()
        self.toolkit.cancel_active_and_wait()

    @staticmethod
    def _text(result: ToolResult) -> str:
        return "\n".join(
            block["text"] for block in result.content_blocks if block["type"] == "text"
        )


class _Session(AbstractCapability):
    """Track native runs and bridge Dashboard input and events.

    This capability observes public lifecycle boundaries; it never replaces a
    graph node or mutates message history. The conversation driver starts a new
    native run only for a follow-up after completion or interruption.
    """

    def __init__(
        self,
        toolkit: _HarnessToolkit,
        conversation_id: str,
        max_turns: int,
        *,
        interactive: bool = False,
    ) -> None:
        self.toolkit = toolkit
        self.conversation_id = conversation_id
        self.limits = UsageLimits(request_limit=max_turns)
        self.usage = RunUsage()
        self.interactive = interactive
        self.error: str | None = None
        self.control: DashboardPlannerControl | None = None
        self.inbox: asyncio.Queue[DashboardMessage] = asyncio.Queue()
        self.enqueued: dict[str, DashboardMessage] = {}
        self.run_done = asyncio.Event()
        self.run_done.set()
        self.cancellation: CancellationToken | None = None
        self.completions = 0

    async def wrap_run(
        self, ctx: RunContext, *, handler: WrapRunHandler
    ) -> AgentRunResult:
        if self.toolkit.finish_result is not None:
            raise UserError("The task has finished. Use /exit to close the session.")
        self.toolkit.stopping.clear()
        if self.interactive and ctx.prompt:
            self.emit_user(ctx.prompt)
        try:
            result = await handler()
            self.error = None
            return result
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            # clai also keeps its own per-run usage for /usage. Observe it without
            # replacing the SDK's counters, including failed/interrupted runs.
            self.usage.incr(ctx.usage)
            self.emit_usage()

    async def before_model_request(
        self, ctx: RunContext, request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.limits.check_before_request(self.usage + ctx.usage)
        return request_context

    async def before_node_run(self, ctx: RunContext, *, node: AgentNode) -> AgentNode:
        if self.toolkit.stopping.is_set():
            # Dashboard drains the physical operation before awaiting the SDK
            # interrupt. Do not deliver queued input during that drain.
            ctx.cancel()
            return node
        if not isinstance(node, ModelRequestNode):
            return node
        while not self.inbox.empty():
            submission = self.inbox.get_nowait()
            enqueue_id = ctx.enqueue(submission.text)
            if enqueue_id is not None:
                self.enqueued[enqueue_id] = submission
                self.completions += 1
        return node

    async def after_node_run(
        self, ctx: RunContext, *, node: AgentNode, result: NodeResult
    ) -> NodeResult:
        if isinstance(node, ModelRequestNode) and isinstance(result, CallToolsNode):
            # Retain completed turns even when request history is later compacted.
            content: list[dict[str, Any]] = []
            for part in result.model_response.parts:
                if isinstance(part, TextPart) and part.content:
                    content.append({"type": "text", "text": part.content})
                elif isinstance(part, ThinkingPart) and part.content:
                    content.append({"type": "thinking", "thinking": part.content})
                elif isinstance(part, ToolCallPart):
                    content.append(
                        {
                            "type": "tool_use",
                            "id": part.tool_call_id,
                            "name": part.tool_name,
                            "input": part.args_as_dict(),
                        }
                    )
            self.toolkit.messages.append({"role": "assistant", "content": content})
        if isinstance(node, (ModelRequestNode, CallToolsNode)):
            self.emit_usage(self.usage + ctx.usage)
        return result

    @on_event(EnqueuedMessagesEvent)
    async def acknowledge_enqueued(
        self, ctx: RunContext, event: EnqueuedMessagesEvent
    ) -> None:
        submission = self.enqueued.pop(event.enqueue_id, None)
        if submission is not None and self.control is not None:
            self.control.message_started(submission.message_id, submission.text)

    @on_event(PartEndEvent)
    async def emit_part(self, ctx: RunContext, event: PartEndEvent) -> None:
        part = event.part
        if not isinstance(part, (TextPart, ThinkingPart)) or not part.content:
            return
        kind = "thinking" if isinstance(part, ThinkingPart) else "text"
        self.toolkit.dashboard_events.emit(
            TranscriptEvent({"type": kind, "text": part.content})
        )
        if not self.interactive:
            logger.info("[%s] %s", kind, part.content)

    def emit_user(self, text: str) -> None:
        self.toolkit.messages.append({"role": "user", "content": text})
        self.toolkit.dashboard_events.emit(
            TranscriptEvent({"type": "user", "text": text})
        )

    def emit_usage(self, usage: RunUsage | None = None) -> None:
        if usage is None:
            usage = self.usage
        self.toolkit.dashboard_events.emit(
            UsageEvent(
                inp=usage.input_tokens,
                out=usage.output_tokens,
                tool_calls=self.toolkit.tool_calls,
            )
        )

    async def submit_dashboard_message(self, message: DashboardMessage) -> int:
        self.inbox.put_nowait(message)
        return 1

    async def interrupt(self) -> int:
        """Discard unstarted submissions, then cancel and drain the native run."""
        discarded = self.inbox.qsize()
        while not self.inbox.empty():
            submission = self.inbox.get_nowait()
            if self.control is not None:
                self.control.message_discarded(submission.message_id)
        if self.cancellation is not None:
            self.cancellation.cancel()
            await self.run_done.wait()
        # The run accounts for its own completion and SDK-enqueued messages.
        # Only submissions that never entered it need to be subtracted here.
        return discarded

    async def run_cli(self, agent: _ConversationAgent, prompt: str) -> None:
        """Run the preset task, then hand its history to the native CLI."""
        from rich.console import Console

        console = Console()
        console.print("Task context:\n" + prompt, markup=False, highlight=False)
        console.print(
            "Starting the task. You can enter follow-up instructions after it "
            "responds, or use /exit to close.",
            markup=False,
        )
        with console.status("Running task…"):
            result = await agent.run(
                prompt,
                conversation_id=self.conversation_id,
                usage_limits=self.limits,
            )
        console.print(result.output, markup=False, highlight=False)
        await agent.to_cli(
            prog_name="rpent",
            message_history=result.all_messages(),
            usage_limits=self.limits,
        )

    async def run(self, agent: _ConversationAgent, prompt: str) -> None:
        tasks: list[asyncio.Task] = []
        try:
            if self.interactive:
                await self.run_cli(agent, prompt)
                return
            if self.control is None:
                self.emit_user(prompt)
                await self._run_conversation(agent, prompt)
                return
            await self.control.start()
            tasks = [
                asyncio.create_task(self.control.run(self)),
                asyncio.create_task(self._run_conversation(agent, prompt)),
            ]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task
        finally:
            if self.control is not None:
                self.control.end()
            # Stop new physical work before cancelling the SDK task. Toolkit
            # cancellation is cooperative, and must finish before solve returns.
            self.toolkit.stopping.set()
            if self.cancellation is not None:
                self.cancellation.cancel()
            await asyncio.to_thread(self.toolkit.cancel_active_and_wait)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_conversation(self, agent: _ConversationAgent, prompt: str) -> None:
        """Run the SDK, then wait for Dashboard follow-ups in the same conversation."""
        while True:
            self.cancellation = CancellationToken()
            self.run_done.clear()
            self.completions = 1
            try:
                await agent.run(
                    prompt,
                    usage_limits=self.limits,
                    conversation_id=self.conversation_id,
                    cancellation_token=self.cancellation,
                )
            except RunCancelled:
                self.error = None
            finally:
                self.cancellation = None
                self.run_done.set()
                if self.control is not None:
                    for pending in self.enqueued.values():
                        self.control.message_discarded(pending.message_id)
                self.enqueued.clear()
            if self.toolkit.finish_result is not None or self.control is None:
                return
            if self.usage.requests >= self.limits.request_limit:
                return
            for _ in range(self.completions):
                await self.control.complete(self)
            submission = await self.inbox.get()
            self.control.message_started(submission.message_id, submission.text)
            prompt = submission.text
