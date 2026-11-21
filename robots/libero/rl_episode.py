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

"""Run one LIBERO episode for an RL trainer.

An RL trainer such as RLinf starts this module once per episode, on the node
that hosts the simulator, with the OpenAI-compatible endpoint that serves its
policy. The native Evaluation planner solves the task with a fresh LIBERO
environment and the running Pi0.5 and SAM3 services, and the outcome is written
to ``outcome.json`` in ``--output-dir``. RPent keeps per-process state such as
the output directory, so every episode runs in its own process.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import signal
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_harness.compaction import SlidingWindowCompaction
from pydantic_core import to_jsonable_python

from robots.libero.memory import render_frozen_entries
from robots.libero.robot_spec import get_robot_spec, get_toolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner.api_loop import ApiAgentLoop
from rpent.planner.base import PlannerResult
from rpent.robots.robot_spec import RunConfig
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("libero_rl_episode")
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
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def start_environment(
    args: argparse.Namespace, task: dict, release: contextlib.ExitStack
) -> tuple[Toolkit, RunConfig, str | None, list[dict]]:
    """Start the task's scene and return its toolkit, config and frozen memory.

    ``release`` stops the started services and closes the toolkit. With a
    frozen memory directory, the task's ``memory_entries`` are rendered for the
    first task message and a receipt of their hashes is returned.
    """
    output, memory_dir = args.output_dir, args.frozen_memory_dir
    memory_prompt, receipt = None, []
    if memory_dir is not None:
        memory_prompt, receipt = render_frozen_entries(
            memory_dir, task["memory_entries"]
        )
        (output / "memory-injection.json").write_text(
            json.dumps({"entries": receipt}, indent=2) + "\n"
        )
        (output / "files").mkdir(exist_ok=True)
    args.suite, args.task, args.seed = task["suite"], task["task"], task["seed"]
    args.planner = "api"
    args.memory_profile = "local" if memory_dir else "none"
    args.memory_dir, args.memory_repo = memory_dir, None
    spec = get_robot_spec()
    config = spec.parse_config(args)
    events = NullDashboardEventSink()
    daemons, runtime = spec.init_runtime(args, output, events, {"env", "vla", "sam3"})
    for daemon in daemons:
        release.callback(daemon.stop)
    runtime["vla_seed"] = task.get("vla_seed")
    toolkit = get_toolkit(
        runtime_kwargs=runtime,
        dashboard_events=events,
        config=config,
        state_output_dir=output,
        memory_file_root=output / "files" if memory_dir is not None else None,
    )
    release.callback(toolkit.close)
    return toolkit, config, memory_prompt, receipt


def render_prompts(
    config: RunConfig, output: Path, memory_prompt: str | None
) -> tuple[str, str]:
    """Return the Evaluation system prompt and the first task message."""
    frozen_memory = memory_prompt is not None
    variables = {
        **config.prompt_vars,
        "output_dir": output / "files" if frozen_memory else output,
        "memory_in_task": frozen_memory,
    }
    prompts = get_robot_spec().prompts
    user_message = prompts.render("user", variables=variables)
    if frozen_memory:
        user_message += "\n\n## Frozen task memory\n\n" + memory_prompt
    return prompts.render("system", variables=variables), user_message


def build_planner(
    *,
    base_url: str,
    model_name: str,
    max_tokens: int,
    request_timeout_s: float,
    episode_timeout_s: float,
    no_images: bool,
    history_compaction: dict | None,
    count_tokens: Callable[[str], int] | None,
) -> tuple[ApiAgentLoop, AsyncOpenAI]:
    """Build the native planner on a policy endpoint and its model client.

    Each model request is sent once, without client retries.
    ``history_compaction`` holds ``SlidingWindowCompaction`` options. With
    ``no_images`` the model receives tool results as text only; image outputs
    are replaced by a notice, and `segment` still locates objects.
    """
    client = AsyncOpenAI(
        base_url=base_url, api_key="unused", max_retries=0, timeout=request_timeout_s
    )
    compaction = (
        partial(SlidingWindowCompaction, **history_compaction, tokenizer=count_tokens)
        if history_compaction is not None
        else None
    )
    planner = ApiAgentLoop(
        model=OpenAIChatModel(
            model_name, provider=OpenAIProvider(openai_client=client)
        ),
        dashboard_events=NullDashboardEventSink(),
        no_images=no_images,
        timeout_s=episode_timeout_s,
        compaction=compaction,
        model_settings={"max_tokens": max_tokens, "parallel_tool_calls": False},
        sequential_tools=True,
    )
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
    truncated = bool(toolkit.state.get().truncated)
    terminal_error = result.error_kind in _TERMINAL_ERRORS
    if success:
        reason = "success"
    elif result.error:
        reason = result.error_kind if terminal_error else "error"
    elif truncated:
        reason = "environment_limit"
    elif result.stats["turns_used"] >= max_turns:
        reason = "request_limit"
    else:
        reason = "planner_completed"
    return {
        "success": success,
        "truncated": truncated,
        "terminal": success or not result.error or terminal_error,
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
    if bool(task.get("memory_entries")) != (args.frozen_memory_dir is not None):
        raise ValueError(
            "memory_entries and --frozen-memory-dir must be given together, got "
            f"{len(task.get('memory_entries') or [])} entries and "
            f"{args.frozen_memory_dir}"
        )
    max_turns = options.pop("max_turns")
    init_output_dir(args.output_dir)
    with contextlib.ExitStack() as release:
        try:
            toolkit, config, memory_prompt, receipt = start_environment(
                args, task, release
            )
        except Exception:
            logger.exception("the episode environment did not start")
            return EXIT_SETUP_FAILED
        system_prompt, user_message = render_prompts(
            config, args.output_dir, memory_prompt
        )
        count_tokens = (
            _token_counter(args.tokenizer_path) if args.tokenizer_path else None
        )
        planner, client = build_planner(**options, count_tokens=count_tokens)
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
            "init_state_id": toolkit.primitives.reset_info["init_state_id"],
            "memory_entries": receipt,
        }
    # Written once the simulator is released, so an outcome means a clean end.
    (args.output_dir / "outcome.json").write_text(json.dumps(outcome) + "\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    get_robot_spec().add_cli_args(parser, use_dashboard=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--request",
        type=Path,
        required=True,
        help="JSON with the episode's task and planner options.",
    )
    parser.add_argument(
        "--frozen-memory-dir",
        type=Path,
        help="Use a prevalidated local memory corpus in read-only Evaluation mode.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="Tokenizer of the served planner model, used to count tokens for "
        "history compaction.",
    )
    args = parser.parse_args()
    if args.vla_endpoint is None or args.sam3_endpoint is None:
        parser.error("--vla-endpoint and --sam3-endpoint must name running services")
    # A stopped episode still releases its simulator and model client.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    sys.exit(run(args))


if __name__ == "__main__":
    main()
