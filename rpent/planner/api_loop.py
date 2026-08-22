"""Provider-independent tool-use agent loop built on pydantic-ai.

The loop wraps the agent's :class:`~rpent.tools.toolkit.Toolkit` as
pydantic-ai function tools and drives :class:`pydantic_ai.Agent` runs,
streaming each turn so progress is logged in real time. Task completion is
signalled by the env-provided ``finish`` tool, whose result carries ``_finish``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import json
import queue
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, BinaryContent, ModelSettings, Tool, ToolReturn
from pydantic_ai.capabilities import ProcessHistory, Thinking
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

from rpent.cli.tui import QUIT_TOKENS
from rpent.dashboard.events import (
    DashboardEventSink,
    TranscriptEvent,
    UsageEvent,
)
from rpent.dashboard.interaction import DashboardInteractionPort, DashboardMessage
from rpent.dashboard.planner_control import DashboardPlannerControl
from rpent.planner.base import PlannerResult
from rpent.tools.state import EnvState
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger

logger = get_logger("api_loop")

#: Console-log truncation limits (characters).
_TEXT_LOG_LIMIT = 500
_ARGS_LOG_LIMIT = 250
_TOOL_LOG_LIMIT = 350
#: Cap on cumulative decoded image bytes kept in the resent request history.
_MAX_HISTORY_IMAGE_BYTES = 4 * 1024 * 1024

#: Always retain at least this many of the most recent images, even if a single
#: frame exceeds the byte budget, so the model never loses its current view.
_MIN_RECENT_IMAGES = 2


class ApiAgentLoop:
    """Planner that runs the tool-calling loop via a pydantic-ai ``Agent``."""

    def __init__(
        self,
        model: Model,
        max_tokens: int = 8192,
        no_images: bool = False,
        *,
        dashboard_events: DashboardEventSink,
        timeout_s: int | None = None,
    ):
        """Store the pydantic-ai model and the output-token cap."""
        self._model = model
        self._max_tokens = max_tokens
        self._dashboard_events = dashboard_events
        self._no_images = no_images
        self._timeout_s = timeout_s

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
        """Run the tool-calling loop until finish, normal stop, or budget."""
        if input_queue is not None and dashboard_interaction is not None:
            raise ValueError(
                "input_queue and dashboard_interaction cannot be used together"
            )
        if dashboard_interaction is not None:
            return asyncio.run(
                self._solve_dashboard(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    toolkit=toolkit,
                    max_turns=max_turns,
                    interaction=dashboard_interaction,
                )
            )
        solve = self._solve(
            system_prompt=system_prompt,
            user_message=user_message,
            toolkit=toolkit,
            max_turns=max_turns,
            input_queue=input_queue,
        )
        if input_queue is not None:
            return asyncio.run(solve)
        try:
            return asyncio.run(asyncio.wait_for(solve, timeout=self._timeout_s))
        except asyncio.TimeoutError:
            toolkit.cancel_active_and_wait()
            return PlannerResult(
                finish_result=None,
                messages=[{"role": "user", "content": user_message}],
                stats={},
                error=f"API planner timed out after {self._timeout_s}s",
            )

    async def _solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
    ) -> PlannerResult:
        agent = self._build_agent(system_prompt, toolkit)

        interactive = input_queue is not None
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        observer = _ApiRunObserver(
            dashboard_events=self._dashboard_events,
            messages=messages,
            max_turns=max_turns,
        )
        last_error: str | None = None
        usage: RunUsage | None = None
        quit_requested = False

        def _inject_pending(run: Any) -> bool:
            """Drain queued user lines into the live run; True => end session.

            Each line is enqueued ``asap`` so it lands in the next model request
            (the next turn boundary). This runs on the event-loop thread, so
            mutating the run's pending-message queue here is race-free.
            """
            while True:
                try:
                    line = input_queue.get_nowait()  # type: ignore[union-attr]
                except queue.Empty:
                    return False
                if line is None:
                    return True
                line = line.strip()
                if line.lower() in QUIT_TOKENS:
                    return True
                if not line:
                    continue
                run.enqueue(line, priority="asap")
                messages.append({"role": "user", "content": line})
                logger.info("[user] %s", _clip(line, _ARGS_LOG_LIMIT))

        async def _await_next() -> str | None:
            """Block off-loop for the next user line between runs (None => end)."""
            logger.info("awaiting input — type a message to continue, /quit to end")
            while True:
                line = await asyncio.to_thread(input_queue.get)  # type: ignore[union-attr]
                if line is None:
                    return None
                line = line.strip()
                if line.lower() in QUIT_TOKENS:
                    return None
                if line:
                    logger.info("[user] %s", _clip(line, _ARGS_LOG_LIMIT))
                    return line

        seed = user_message
        history: list[ModelMessage] | None = None
        try:
            while True:
                run_turns = 0
                # request_limit overrides pydantic-ai's default (50) so the
                # manual max_turns break below is what bounds each run.
                async with agent.iter(
                    seed,
                    message_history=history,
                    usage_limits=UsageLimits(request_limit=max_turns + 1),
                ) as run:
                    async for node in run:
                        if interactive and _inject_pending(run):
                            quit_requested = True
                            break
                        if Agent.is_call_tools_node(node):
                            run_turns += 1
                            observer.observe_response(
                                node.model_response,
                                run.usage,
                                log_turn=run_turns,
                            )

                            async with node.stream(run.ctx) as stream:
                                async for event in stream:
                                    observer.observe_tool(event, run.usage)

                            if observer.finish_result is not None:
                                logger.info("FINISH called: %s", observer.finish_result)
                                break
                            if observer.turns >= max_turns:
                                logger.info(
                                    "reached max_turns=%d. Stopping.", max_turns
                                )
                                break
                        elif Agent.is_end_node(node):
                            if interactive:
                                logger.info(
                                    "model ended turn without a tool call "
                                    "— awaiting your input."
                                )
                            else:
                                logger.info(
                                    "model ended turn without a tool call. Stopping."
                                )
                            break

                    usage = run.usage
                    if interactive:
                        history = run.all_messages()

                # finish, quit, non-interactive, or the cumulative turn budget is
                # spent => end the whole session so max_turns is enforced across
                # every run, not per run.
                if (
                    observer.finish_result is not None
                    or quit_requested
                    or not interactive
                    or observer.turns >= max_turns
                ):
                    break
                nxt = await _await_next()
                if nxt is None:
                    break
                seed = nxt
                messages.append({"role": "user", "content": seed})
        except UsageLimitExceeded as e:
            logger.info("usage limit reached: %s", e)
        except Exception as e:  # noqa: BLE001 - surfaced via PlannerResult.error
            last_error = _api_error_text(e, no_images=self._no_images)
            logger.error("agent run failed: %s", last_error)

        return PlannerResult(
            finish_result=observer.finish_result,
            messages=messages,
            stats=_build_stats(usage, observer.turns, observer.tool_calls),
            error=last_error,
        )

    async def _solve_dashboard(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        interaction: DashboardInteractionPort,
    ) -> PlannerResult:
        """Drive cancellable PydanticAI runs from complete history checkpoints."""
        agent = self._build_agent(system_prompt, toolkit)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

        def emit_user(text: str, *, initial: bool = False) -> None:
            if not initial:
                messages.append({"role": "user", "content": text})
            self._dashboard_events.emit(
                TranscriptEvent(
                    {"type": "initial_prompt"}
                    if initial
                    else {"type": "user", "text": text}
                )
            )

        control = DashboardPlannerControl(
            interaction=interaction,
            cancel_active_and_wait=toolkit.cancel_active_and_wait,
            emit_user=emit_user,
            emit_initial_user=lambda: emit_user(user_message, initial=True),
            defer_message_ack=True,
        )
        observer = _ApiRunObserver(
            dashboard_events=self._dashboard_events,
            messages=messages,
            max_turns=max_turns,
        )
        session = _ApiDashboardSession(
            agent=agent,
            control=control,
            observer=observer,
            max_turns=max_turns,
            no_images=self._no_images,
        )
        error: str | None = None
        try:
            await asyncio.wait_for(
                session.run(user_message),
                timeout=self._timeout_s,
            )
        except asyncio.TimeoutError:
            error = f"API planner timed out after {self._timeout_s}s"
            control.end()
        except Exception as exc:
            error = _api_error_text(exc, no_images=self._no_images)
            control.end()
        finally:
            try:
                await control.cancel_active_toolkit()
            except Exception as exc:
                cleanup_error = (
                    f"API toolkit cancellation failed: {type(exc).__name__}: {exc}"
                )
                logger.warning(cleanup_error)
                error = error or cleanup_error
            await session.close()

        return PlannerResult(
            finish_result=observer.finish_result,
            messages=messages,
            stats={
                "backend": "api",
                **_build_stats(session.usage, observer.turns, observer.tool_calls),
            },
            error=error or session.error,
        )

    def _build_agent(self, system_prompt: str, toolkit: Toolkit) -> Agent:
        """Build an Agent for terminal or Dashboard execution."""
        return Agent(
            self._model,
            instructions=system_prompt or None,
            tools=_build_tools(toolkit, no_images=self._no_images),
            model_settings=_build_model_settings(self._model, self._max_tokens),
            capabilities=[
                Thinking(effort="high"),
                ProcessHistory(processor=_prune_history_images),
            ],
        )


@dataclasses.dataclass
class _ApiRunObserver:
    """Record model/tool events shared by terminal and Dashboard runs."""

    dashboard_events: DashboardEventSink
    messages: list[dict[str, Any]]
    max_turns: int
    turns: int = 0
    tool_calls: int = 0
    finish_result: dict[str, Any] | None = None
    pending_finish: dict[str, Any] | None = None

    def observe_response(
        self,
        response: ModelResponse,
        usage: RunUsage,
        *,
        log_turn: int | None = None,
    ) -> None:
        self.turns += 1
        message = _serialize_response(response)
        self.messages.append(message)
        _log_response(
            response,
            usage,
            self.turns if log_turn is None else log_turn,
            self.max_turns,
        )
        for block in message["content"]:
            if block["type"] == "text":
                payload = {"type": "text", "text": block["text"]}
            elif block["type"] == "thinking":
                payload = {"type": "thinking", "text": block["thinking"]}
            else:
                continue
            self.dashboard_events.emit(TranscriptEvent(payload))
        self.emit_usage(usage)

    def observe_tool(self, event: Any, usage: RunUsage) -> bool:
        completed = False
        if isinstance(event, FunctionToolCallEvent):
            self.tool_calls += 1
            part = event.part
            args = part.args_as_dict()
            self.dashboard_events.emit(
                TranscriptEvent(
                    {"type": "tool_call", "tool": part.tool_name, "args": args}
                )
            )
            if part.tool_name == "finish":
                self.pending_finish = {"_finish": True, **args}
        elif isinstance(event, FunctionToolResultEvent):
            completed = True
            message = _serialize_tool_result(event)
            self.messages.append(message)
            _log_tool_result(message)
            part = event.part
            is_error = bool(getattr(part, "is_error", False))
            if self.pending_finish is not None:
                if not is_error and "finish refused" not in str(message):
                    self.finish_result = self.pending_finish
                self.pending_finish = None
            self.dashboard_events.emit(
                TranscriptEvent(
                    {
                        "type": "tool_result",
                        "tool": message.get("name") or "tool_result",
                        "result": {
                            "is_error": is_error,
                            "size": len(message["content"]),
                        },
                    }
                )
            )
        self.emit_usage(usage)
        return completed

    def emit_usage(self, usage: RunUsage) -> None:
        self.dashboard_events.emit(
            UsageEvent(
                inp=int(usage.input_tokens or 0),
                out=int(usage.output_tokens or 0),
                tool_calls=self.tool_calls,
            )
        )


class _ApiDashboardSession:
    """Own serial, independent PydanticAI runs for one Dashboard TaskRun."""

    def __init__(
        self,
        *,
        agent: Agent,
        control: DashboardPlannerControl,
        observer: _ApiRunObserver,
        max_turns: int,
        no_images: bool,
    ) -> None:
        self._agent = agent
        self._control = control
        self._max_turns = max_turns
        self._no_images = no_images
        self._observer = observer
        self._history: list[ModelMessage] = []
        self.usage = RunUsage()
        self._pending_prompts: deque[tuple[str | None, str]] = deque()
        self._run_task: asyncio.Task[Any] | None = None
        self._active_prompt = False
        self._closing = False
        self.error: str | None = None

    async def run(self, prompt: str) -> None:
        await self.submit(prompt)
        await self._control.start()
        await self._control.run(self)

    async def submit(self, text: str) -> int:
        """Queue Dashboard input as a new independent API run."""
        return self._queue_prompt(text)

    async def submit_dashboard_message(self, message: DashboardMessage) -> int:
        """Queue Dashboard input and defer acknowledgement until it starts."""
        return self._queue_prompt(message.text, message_id=message.message_id)

    def _queue_prompt(self, text: str, *, message_id: str | None = None) -> int:
        if self._closing:
            raise RuntimeError("API conversation is closed")
        if self.error is not None:
            raise RuntimeError(self.error)
        self._pending_prompts.append((message_id, text))
        if self._run_task is None:
            self._run_task = asyncio.create_task(self._run_pending_prompts())
        return 1

    async def interrupt(self) -> int:
        run_task = self._run_task
        interrupted = int(self._active_prompt) + len(self._pending_prompts)
        discarded_message_ids = tuple(
            message_id
            for message_id, _ in self._pending_prompts
            if message_id is not None
        )
        self._pending_prompts.clear()
        for message_id in discarded_message_ids:
            self._control.message_discarded(message_id)
        if run_task is None or run_task.done():
            return interrupted
        run_task.cancel()
        try:
            with contextlib.suppress(asyncio.CancelledError):
                await run_task
        finally:
            if self._run_task is run_task:
                self._run_task = None
        return interrupted

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        await self.interrupt()

    async def _run_pending_prompts(self) -> None:
        task = asyncio.current_task()
        try:
            while self._pending_prompts and not self._closing:
                message_id, seed = self._pending_prompts.popleft()
                self._active_prompt = True
                try:
                    if message_id is not None:
                        self._control.message_started(message_id, seed)
                    if not await self._run_agent(seed):
                        self._pending_prompts.clear()
                        return
                    await self._control.complete(self)
                finally:
                    self._active_prompt = False
        finally:
            if self._run_task is task:
                self._run_task = None

    async def _run_agent(self, seed: str) -> bool:
        run_completed = False
        run: Any | None = None
        node: Any | None = None
        try:
            async with self._agent.iter(
                seed,
                message_history=list(self._history),
                usage=self.usage,
                usage_limits=UsageLimits(request_limit=self._max_turns + 1),
            ) as run:
                node = run.next_node
                while not Agent.is_end_node(node):
                    if Agent.is_call_tools_node(node):
                        await self._process_tool_node(run, node)
                    if (
                        self._observer.finish_result is not None
                        or self._observer.turns >= self._max_turns
                    ):
                        self._control.end()
                        return False
                    if self._pending_prompts:
                        # Dashboard input accepted at this tool boundary starts
                        # a fresh run from the checkpoint captured below.
                        node = await run.next(node)
                        break
                    node = await run.next(node)

                run_completed = True
        except Exception as exc:
            self.error = _api_error_text(exc, no_images=self._no_images)
            if not self._closing:
                self._control.end()
        finally:
            # Preserve interrupted tool results for PydanticAI to repair on the
            # next run. Older supported releases can leave a bare tool-call
            # response when cancellation wins before any tool returns; remove
            # only that unusable frontier.
            if run is not None:
                history = list(run.all_messages())
                if (
                    history
                    and isinstance(history[-1], ModelResponse)
                    and history[-1].tool_calls
                ):
                    if request := getattr(node, "request", None):
                        history.append(request)
                    else:
                        history.pop()
                self._history = history
        return run_completed and not self._closing

    async def _process_tool_node(self, run: Any, node: Any) -> None:
        self._observer.observe_response(node.model_response, run.usage)

        async with node.stream(run.ctx) as stream:
            async for event in stream:
                tool_completed = self._observer.observe_tool(event, run.usage)
                if (
                    tool_completed
                    and self._observer.finish_result is None
                    and self._observer.turns < self._max_turns
                ):
                    await self._control.tool_completed(self)


def _build_model_settings(model: Model, max_tokens: int) -> ModelSettings:
    """Build model settings, enabling prompt caching for Anthropic models."""
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

    if isinstance(model, AnthropicModel):
        return AnthropicModelSettings(
            max_tokens=max_tokens,
            anthropic_cache_instructions=True,
            anthropic_cache_tool_definitions=True,
            anthropic_cache_messages=True,
        )
    return ModelSettings(max_tokens=max_tokens)


def _prune_history_images(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Drop old camera images so the resent request body stays bounded."""
    # Every image in history, oldest -> newest: (msg_idx, part_idx, item_idx, nbytes).
    located: list[tuple[int, int, int, int]] = []
    for mi, message in enumerate(messages):
        for pi, part in enumerate(getattr(message, "parts", ()) or ()):
            if not isinstance(part, UserPromptPart) or not isinstance(
                part.content, list
            ):
                continue
            for ii, item in enumerate(part.content):
                if isinstance(item, BinaryContent) and item.media_type.startswith(
                    "image/"
                ):
                    located.append((mi, pi, ii, len(item.data)))

    if not located:
        return messages

    # Walk newest -> oldest, keeping images while under the byte budget.
    keep: set[tuple[int, int, int]] = set()
    total = 0
    for rank, (mi, pi, ii, nbytes) in enumerate(reversed(located)):
        if rank < _MIN_RECENT_IMAGES or total + nbytes <= _MAX_HISTORY_IMAGE_BYTES:
            keep.add((mi, pi, ii))
            total += nbytes

    if len(keep) == len(located):
        return messages

    drop_items_by_part: dict[tuple[int, int], set[int]] = {}
    for mi, pi, ii, _ in located:
        if (mi, pi, ii) not in keep:
            drop_items_by_part.setdefault((mi, pi), set()).add(ii)

    new_messages = list(messages)
    for (mi, pi), drop_items in drop_items_by_part.items():
        message = new_messages[mi]
        part = message.parts[pi]
        new_content = [
            "[earlier camera image omitted to bound request size]"
            if ci in drop_items
            else item
            for ci, item in enumerate(part.content)
        ]
        new_parts = list(message.parts)
        new_parts[pi] = dataclasses.replace(part, content=new_content)
        new_messages[mi] = dataclasses.replace(message, parts=new_parts)

    return new_messages


def _is_image_rejection(e: Exception) -> bool:
    """True when the provider returned a 4xx complaining about image input.

    Matches errors like ``400 {'code': 10007, 'msg': "Bad Request: [message
    type 'image_url' is not supported]"}`` from OpenAI-compatible endpoints
    serving text-only models.
    """
    if not isinstance(e, ModelHTTPError):
        return False
    if not 400 <= e.status_code < 500:
        return False
    return "image" in str(e).lower()


def _api_error_text(error: Exception, *, no_images: bool) -> str:
    text = f"{type(error).__name__}: {error}"
    if not no_images and _is_image_rejection(error):
        text += (
            "\n\nThe model rejected image input — it is likely a text-only "
            "model (no vision support). Re-run with --no-images: RPent will "
            "then keep every visual observation as a file-path text notice "
            "instead of sending image bytes."
        )
    return text


def _build_tools(toolkit: Toolkit, *, no_images: bool = False) -> list[Tool]:
    """Build the API-only image reader plus pydantic-ai toolkit wrappers."""
    image_reader = _make_image_reader(toolkit.state, no_images=no_images)
    tools: list[Tool] = [Tool(image_reader, name="read_image")]
    for spec in toolkit.get_tools_spec():
        name = spec["name"]
        tools.append(
            Tool.from_schema(
                function=_make_tool_function(toolkit, name, no_images=no_images),
                name=name,
                description=spec.get("description", ""),
                json_schema=spec.get("input_schema")
                or {"type": "object", "properties": {}},
                takes_ctx=False,
            )
        )
    return tools


def _make_image_reader(
    state: EnvState,
    *,
    no_images: bool,
) -> Callable[[str, int], ToolReturn | dict[str, str] | str]:
    if no_images:

        def read_image_tool(name: str, step: int = -1) -> str:
            return read_image_text_only(name, step, state=state)

        read_image_tool.__name__ = "read_image"
        read_image_tool.__doc__ = read_image_text_only.__doc__
        return read_image_tool

    def read_image_tool(
        name: str, step: int = -1
    ) -> ToolReturn | dict[str, str]:
        return read_image(name, step, state=state)

    read_image_tool.__name__ = "read_image"
    read_image_tool.__doc__ = read_image.__doc__
    return read_image_tool


def read_image(
    name: str, step: int = -1, *, state: EnvState
) -> ToolReturn | dict[str, str]:
    """Read a step-scoped image artifact as visual input.

    Artifact failures are returned as structured tool errors so a bad
    model-supplied name or step does not abort the agent run.
    """
    try:
        resolved_step, path = _resolve_image_artifact(state, name, step)
        content = BinaryContent(
            data=state.load_bytes(name, step=resolved_step),
            media_type=_image_media_type(path),
        )
    except Exception as e:
        return {"error": str(e)}
    return ToolReturn(
        return_value={"artifact": name, "step": resolved_step},
        content=[content],
    )


def read_image_text_only(
    name: str, step: int = -1, *, state: EnvState
) -> str | dict[str, str]:
    """Acknowledge an image artifact without sending bytes to the model."""
    try:
        resolved_step, _ = _resolve_image_artifact(state, name, step)
    except Exception as e:
        return {"error": str(e)}
    return (
        f"Image artifact {name!r} exists at step {resolved_step}, but image "
        "input is disabled (--no-images, text-only model). Reason from textual "
        "state instead: view_env_state, back_project, and numeric tool results."
    )


def _resolve_image_artifact(
    state: EnvState,
    name: str,
    step: int,
) -> tuple[int, Path]:
    record = state.get(step)
    path = state.artifact_path(name, step=record.step_idx)
    if name not in record.artifacts or not path.is_file():
        raise FileNotFoundError(
            f"image artifact {name!r} is not available at step {step}"
        )
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"artifact {name!r} is not an image")
    return record.step_idx, path


def _image_media_type(path: Path) -> str:
    return "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"


def _make_tool_function(toolkit: Toolkit, name: str, *, no_images: bool = False):
    """Return a callable that dispatches one tool call to the toolkit."""

    def _call(**kwargs: Any) -> Any:
        result = toolkit.execute_tool(name, kwargs)
        text, images = _content_blocks_to_pydantic(result.content_blocks)
        if images and not no_images:
            return ToolReturn(return_value=text, content=images)
        return text

    _call.__name__ = name
    return _call


def _content_blocks_to_pydantic(
    blocks: list[dict[str, Any]],
) -> tuple[str, list[BinaryContent]]:
    """Split Anthropic-shaped content blocks into text and image content."""
    text_parts: list[str] = []
    images: list[BinaryContent] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "image":
            source = block.get("source") or {}
            data = source.get("data")
            if source.get("type") == "base64" and data:
                images.append(
                    BinaryContent(
                        data=base64.b64decode(data),
                        media_type=source.get("media_type", "image/png"),
                    )
                )
    text = "\n\n".join(part for part in text_parts if part) or "{}"
    return text, images


def _serialize_response(response: ModelResponse) -> dict[str, Any]:
    """Render one assistant turn as a serialisable transcript message."""
    content: list[dict[str, Any]] = []
    for part in response.parts:
        if isinstance(part, TextPart):
            if part.content:
                content.append({"type": "text", "text": part.content})
        elif isinstance(part, ThinkingPart):
            if part.content:
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
    return {"role": "assistant", "content": content}


def _serialize_tool_result(event: FunctionToolResultEvent) -> dict[str, Any]:
    """Render one tool result as a serialisable transcript message (no images)."""
    part = event.part
    content = getattr(part, "content", None)
    if not isinstance(content, str):
        content = json.dumps(content, default=str)
    return {
        "role": "tool",
        "name": getattr(part, "tool_name", None),
        "tool_call_id": getattr(part, "tool_call_id", None),
        "content": content,
    }


def _build_stats(
    usage: RunUsage | None, turns: int, n_tool_calls: int
) -> dict[str, Any]:
    """Assemble the run stats dict from accumulated usage and counters."""
    stats: dict[str, Any] = {"turns_used": turns, "tool_calls": n_tool_calls}
    if usage is not None:
        stats.update(
            {
                "total_input_tokens": int(usage.input_tokens or 0),
                "total_output_tokens": int(usage.output_tokens or 0),
                "cache_read_tokens": int(usage.cache_read_tokens or 0),
                "cache_write_tokens": int(usage.cache_write_tokens or 0),
                "requests": int(usage.requests or 0),
            }
        )
    return stats


def _log_response(
    response: ModelResponse, usage: RunUsage, turn: int, max_turns: int
) -> None:
    """Log model text, thinking, tool calls, and cumulative usage for a turn."""
    logger.info("=== turn %d/%d ===", turn, max_turns)
    for part in response.parts:
        if isinstance(part, TextPart):
            text = (part.content or "").strip()
            if text:
                logger.info("[model] %s", text)
        elif isinstance(part, ThinkingPart):
            text = (part.content or "").strip()
            if text:
                logger.info("[think] %s", _clip(text, _TEXT_LOG_LIMIT))
        elif isinstance(part, ToolCallPart):
            args = json.dumps(part.args_as_dict(), default=str)
            logger.info("[tool>] %s(%s)", part.tool_name, _clip(args, _ARGS_LOG_LIMIT))
    logger.info(
        "[usage] in=%s out=%s cache_read=%s cache_write=%s requests=%s",
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
        usage.requests,
    )


def _log_tool_result(message: dict[str, Any]) -> None:
    """Log a one-line summary of a tool result."""
    content = " ".join((message.get("content") or "").split())
    logger.info("[tool<] %s: %s", message.get("name"), _clip(content, _TOOL_LOG_LIMIT))


def _clip(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` characters with an overflow marker."""
    if len(text) <= limit:
        return text
    return text[:limit] + "...(+%d)" % (len(text) - limit)
