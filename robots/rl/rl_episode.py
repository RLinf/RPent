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

"""Run one shared-workflow simulator episode for an RL trainer.

An RL trainer such as RLinf starts this module once per episode on the node
that hosts the simulator, with the OpenAI-compatible endpoint that serves its
policy. The end-effector planner solves the task with a fresh simulator
environment and the running SAM3 service, with adaptive observations.
The outcome is written to ``outcome.json`` in ``--output-dir``.
RPent keeps per-process state such as the output directory, so every episode
runs in its own process.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import signal
import sys
from collections.abc import Callable
from pathlib import Path

from openai import AsyncOpenAI
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_core import to_jsonable_python

from robots.rl.robot_spec import get_robot_spec, get_toolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner.api_loop import ApiAgentLoop
from rpent.planner.base import PlannerResult
from rpent.planner.compaction import build_history_compaction
from rpent.planner.qwen import QwenChatModel, qwen_model_settings
from rpent.robots.robot_spec import RunConfig
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("rl_episode")
# Exit status when the environment or a shared service did not start; the
# trainer may run the episode again.
EXIT_SETUP_FAILED = 75
# Planner errors with a task outcome, including exhaustion of episode budgets.
_TERMINAL_ERRORS = {
    "tool_arguments",
    "tool_name",
    "response_truncated",
    "context_limit",
    "episode_timeout",
}


def _token_counter(tokenizer_path: Path) -> Callable[[str], int]:
    """Count text tokens with the served model's tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # Segments carry their own text only; the chat template already added the
    # special tokens that the real prompt pays for.
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def start_environment(
    args: argparse.Namespace, task: dict, release: contextlib.ExitStack
) -> tuple[Toolkit, RunConfig]:
    """Start the task's scene and return its toolkit and config.

    ``release`` stops the started services and closes the toolkit.
    """
    args.env = task.get("environment", getattr(args, "env", "libero"))
    args.workflow = task.get("workflow", getattr(args, "workflow", "episode"))
    args.image_size = task.get("image_size", getattr(args, "image_size", 1024))
    for name, default in (
        ("perception_mode", "single"),
        ("max_observe_calls", 8),
        ("observe_concurrency", 8),
        ("action_mode", "single"),
        ("max_actions_per_call", 3),
        ("reasoning_history", -1),
        ("image_history", -1),
    ):
        setattr(args, name, task.get(name, getattr(args, name, default)))
    args.suite, args.task, args.seed = task.get("suite"), task.get("task"), task["seed"]
    args.task_name = task.get("task_name", getattr(args, "task_name", None))
    args.split = task.get("split", getattr(args, "split", "target"))
    args.task_config = task.get(
        "task_config", getattr(args, "task_config", "demo_randomized")
    )
    args.planner = "api"
    args.memory_profile, args.memory_dir, args.memory_repo = "none", None, None
    spec = get_robot_spec()
    config = spec.parse_config(args)
    events = NullDashboardEventSink()
    daemons, runtime = spec.init_runtime(args, args.output_dir, events, {"env", "sam3"})
    for daemon in daemons:
        release.callback(daemon.stop)
    toolkit = get_toolkit(
        runtime_kwargs=runtime,
        dashboard_events=events,
        config=config,
    )
    release.callback(toolkit.close)
    return toolkit, config


def render_prompts(config: RunConfig) -> tuple[str, str]:
    """Return the adaptive system prompt and the opening user message."""
    prompts = get_robot_spec().prompts
    variables = config.prompt_vars
    return prompts.render("system", variables=variables), prompts.render(
        "user", variables=variables
    )


def build_planner(
    *,
    base_url: str,
    model_name: str,
    max_tokens: int,
    request_timeout_s: float,
    episode_timeout_s: float,
    observation_mode: str,
    history_compaction: dict | None,
    count_tokens: Callable[[str], int] | None,
    reasoning_effort: str = "none",
    thinking: bool = True,
    tokenizer_path: Path | None = None,
    sampling_params: dict | None = None,
    request_trace_dir: Path | None = None,
) -> tuple[ApiAgentLoop, AsyncOpenAI]:
    """Build the native planner on a policy endpoint and its model client.

    Each model request is sent once, without client retries.
    The history_compaction option holds the tiered compaction settings.
    Qwen requests carry thinking-mode sampling, an explicit thinking switch,
    and an effort tier only when requested.
    """
    if observation_mode != "adaptive":
        raise ValueError("rl supports only adaptive observations")
    trace, http_client = None, None
    if request_trace_dir is not None:
        import httpx

        from rpent.planner.request_trace import RequestTrace, TraceTransport

        trace = RequestTrace(request_trace_dir)
        http_client = httpx.AsyncClient(
            transport=TraceTransport(trace), timeout=request_timeout_s
        )
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="unused",
        max_retries=0,
        timeout=request_timeout_s,
        http_client=http_client,
    )
    # The planner calls this capability factory once per run, so it must be a
    # callable whenever compaction is enabled and absent otherwise.
    compaction = (
        (lambda: build_history_compaction(history_compaction, tokenizer=count_tokens))
        if history_compaction is not None
        else None
    )
    planner = ApiAgentLoop(
        model=QwenChatModel(
            model_name,
            provider=OpenAIProvider(openai_client=client),
            # SGLang constrains the whole tool set once any tool is strict.
            profile={"openai_supports_strict_tool_definition": False},
        ),
        dashboard_events=NullDashboardEventSink(),
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        thinking=thinking,
        no_images=False,
        timeout_s=episode_timeout_s,
        compaction=compaction,
        model_settings=qwen_model_settings(
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            thinking=thinking,
            parallel_tool_calls=False,
            **(sampling_params or {}),
        ),
        sequential_tools=True,
    )
    planner.model.tokenizer_path = tokenizer_path
    planner.model.request_trace = trace
    return planner, client


async def solve(
    planner: ApiAgentLoop,
    client: AsyncOpenAI,
    toolkit: Toolkit,
    *,
    system_prompt: str,
    user_message: str,
    max_turns: int,
) -> PlannerResult:
    """Run the planner and close its model client with the episode."""
    async with client:
        return await planner.solve_async(
            system_prompt=system_prompt,
            user_message=user_message,
            toolkit=toolkit,
            max_turns=max_turns,
        )


def episode_outcome(result: PlannerResult, toolkit: Toolkit, max_turns: int) -> dict:
    """Describe a finished episode; success always comes from the environment."""
    success = bool(toolkit.solved())
    service_errors = []
    for record in toolkit.state.records():
        for name in sorted(record.artifacts):
            if name.startswith("sam3_detection_") and name.endswith(".json"):
                detection = toolkit.state.load(name, step=record.step_idx)
                if detection["category"] == "service_error":
                    service_errors.append(
                        {
                            "step": record.step_idx,
                            "artifact": name,
                            "error": detection["error"],
                        }
                    )
    if service_errors:
        return {
            "success": success,
            "truncated": False,
            "terminal": False,
            "termination_reason": "infrastructure_error",
            "error_kind": "tool_transport",
            "error": "SAM3 service failure; retain this attempt and retry the episode at most once.",
            "service_errors": service_errors,
        }
    truncated = toolkit.state.latest_record().truncated
    terminal_error = result.error_kind in _TERMINAL_ERRORS
    if success:
        reason = "success"
    elif result.error:
        reason = result.error_kind if terminal_error else "error"
    elif result.stats["turns_used"] >= max_turns:
        reason = "request_limit"
    elif truncated:
        reason = "environment_step_limit"
    else:
        reason = "planner_completed"
    return {
        "success": success,
        "truncated": truncated,
        "terminal": success or truncated or not result.error or terminal_error,
        "termination_reason": reason,
        "error": result.error,
        "error_kind": result.error_kind,
    }


def write_transcript(output: Path, result: PlannerResult) -> None:
    transcript = {
        "messages": result.messages,
        "stats": result.stats,
        "finish_result": result.finish_result,
        "error": result.error,
        "error_kind": result.error_kind,
    }
    (output / "planner.json").write_text(
        json.dumps(to_jsonable_python(transcript, bytes_mode="base64"), indent=2) + "\n"
    )


def run(args: argparse.Namespace) -> int:
    """Run the requested episode, write ``outcome.json`` and return the exit status."""
    request = json.loads(args.request.read_text())
    task, options = request["task"], dict(request["planner"])
    max_turns = options.pop("max_turns")
    history_compaction = options.pop("history_compaction", None)
    if options.pop("no_images", False):
        raise ValueError("rl requires camera images")
    args.observation_mode = options.pop("observation_mode", args.observation_mode)
    args.prompt = options.pop("prompt", args.prompt)
    args.image_size = options.pop("image_size", args.image_size)
    init_output_dir(args.output_dir)
    with contextlib.ExitStack() as release:
        try:
            toolkit, config = start_environment(args, task, release)
        except Exception:
            logger.exception("the episode environment did not start")
            return EXIT_SETUP_FAILED
        system_prompt, user_message = render_prompts(config)
        count_tokens = (
            _token_counter(args.tokenizer_path) if args.tokenizer_path else None
        )
        planner, client = build_planner(
            **options,
            observation_mode=args.observation_mode,
            history_compaction=history_compaction,
            count_tokens=count_tokens,
            tokenizer_path=args.tokenizer_path,
            request_trace_dir=(
                args.output_dir / "api_trace"
                if getattr(args, "record_api_trace", False)
                else None
            ),
        )
        planner = get_robot_spec().configure_planner(planner, config)
        result = asyncio.run(
            solve(
                planner,
                client,
                toolkit,
                system_prompt=system_prompt,
                user_message=user_message,
                max_turns=max_turns,
            )
        )
        write_transcript(args.output_dir, result)
        outcome = {
            **episode_outcome(result, toolkit, max_turns),
            "observation_mode": args.observation_mode,
            "prompt": args.prompt,
            "init_state_id": task.get("init_state_id"),
            "memory_entries": [],
        }
    # Written once the simulator is released, so an outcome means a clean end.
    (args.output_dir / "outcome.json").write_text(json.dumps(outcome) + "\n")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one trainer-owned episode without starting models or services."""
    parser = argparse.ArgumentParser(description=__doc__)
    get_robot_spec().add_cli_args(parser, use_dashboard=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--record-api-trace",
        action="store_true",
        help="Record full HTTP requests, responses and compaction inputs for debugging.",
    )
    parser.add_argument(
        "--request",
        type=Path,
        required=True,
        help="JSON with the episode's task and planner options.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="Tokenizer of the served planner model, used to count tokens for "
        "history compaction.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    # A stopped episode still releases its simulator and model client.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    sys.exit(run(args))


if __name__ == "__main__":
    main()
