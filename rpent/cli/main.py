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

"""Physical agent main CLI entrypoint."""

# `rpent/cli/`
#
# CLI entrypoints for RPent (currently just `main.py`).
#
# ## Run
#
# `main()` is exposed as the `rpent` console script (see `[project.scripts]`
# in `pyproject.toml`):
#
# ```bash
# rpent --robot libero --suite libero_object_task --task 0 --seed 0 [...]
# ```
#
# ## Note
#
# Do not import `rpent.cli` from other `rpent` modules. `main.py` pulls in
# `rpent.planner`, `rpent.robots`, `rpent.utils`, `rpent.dashboard`, and
# `rpent.tools`, so importing the CLI back into any of them would create an
# import cycle. Nothing else should depend on this package.
from __future__ import annotations

import argparse
import json
import queue
import shlex
import sys
import time
from collections.abc import Callable
from pathlib import Path

from rpent.cli.tui import (
    start_first_prompt_resolver,
    start_interactive_reader,
)
from rpent.dashboard.events import (
    NullDashboardEventSink,
    RunStartedEvent,
)
from rpent.memory import MemoryManager
from rpent.planner.base import REASONING_EFFORTS, build_planner
from rpent.robots import enumerate_robots, get_robot_spec, get_toolkit
from rpent.utils.logging import get_logger, init_output_dir
from rpent.utils.resources import ensure_resources

logger = get_logger("agent")


# ---------------------------------------------------------------------------
# API agent transcript serialization
# ---------------------------------------------------------------------------


def _strip_images(value):
    """Return a copy of ``value`` with inline image payloads omitted.

    SDK objects are left untouched; ``json.dump(..., default=str)`` handles
    them at write time. Only the bulky base64 image blocks are replaced.
    """
    if isinstance(value, list):
        return [_strip_images(v) for v in value]
    if isinstance(value, dict):
        if value.get("type") == "image":
            return {"type": "image", "source": {"_omitted_for_transcript": True}}
        if value.get("type") == "image_url":
            return {"type": "image_url", "image_url": {"_omitted_for_transcript": True}}
        return {k: _strip_images(v) for k, v in value.items()}
    return value


def _serialize_messages(messages: list[dict]) -> list[dict]:
    """Strip inline image payloads from messages before writing the transcript."""
    return [
        {
            **{k: v for k, v in m.items() if k != "content"},
            "content": _strip_images(m.get("content")),
        }
        for m in messages
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    known_robots = enumerate_robots()
    known_robots_text = ", ".join(known_robots) if known_robots else "none"
    ap = argparse.ArgumentParser(
        description="RPent: Agentic Infrastructure for the Physical World",
        add_help=False,
    )

    ap.add_argument(
        "--robot",
        dest="robot_name",
        required=False,
        choices=known_robots,
        help=f"Robot backend. Known robots: {known_robots_text}.",
    )
    ap.add_argument(
        "--env",
        dest="env_name",
        required=False,
        choices=known_robots,
        help="Deprecated alias for --robot; use --robot instead.",
    )

    # models
    ap.add_argument(
        "--planner",
        default="api",
        choices=["api", "claude_code", "codex"],
        help="LLM backend: api | claude_code | codex.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Model id. For the 'api' planner, prefix the provider "
        "(e.g. anthropic:claude-opus-4-8, openai:gpt-5.5, "
        "openai-chat:glm-5.2). For claude_code/codex this "
        "overrides the backend default model.",
    )
    ap.add_argument(
        "--base-url",
        default=None,
        help="API base URL. Defaults to the selected backend's base URL env var.",
    )
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default="none",
        help="Planner reasoning effort for api, claude_code, and "
        "codex. Higher effort may improve task success rate "
        "but increases runtime. Defaults to none.",
    )
    ap.add_argument(
        "--no-images",
        action="store_true",
        help="Never send image bytes to the model (api planner only). "
        "Use for text-only models that reject image input "
        "(e.g. 400 \"message type 'image_url' is not supported\"); "
        "read_image then returns the image name instead, with a notice.",
    )
    ap.add_argument(
        "--planner-timeout-s",
        type=int,
        default=None,
        help="Wall-clock cap for api/claude_code/codex planner runs. "
        "Terminal interactive API/Claude sessions are exempt. "
        "Defaults to CODEX_TIMEOUT_S (codex only), "
        "CELL_TIMEOUT_S, or 1200.",
    )
    ap.add_argument(
        "--claude-code-max-budget-usd",
        type=float,
        default=None,
        help="Budget passed to claude -p --max-budget-usd. "
        "Defaults to MAX_BUDGET_USD env or 10.",
    )

    # other config
    ap.add_argument("--output-dir", default=None)
    ap.add_argument(
        "--memory-profile",
        choices=["hf", "local"],
        default=None,
        help="Memory profile (default: hf for evaluation, local for exploration).",
    )
    ap.add_argument(
        "--memory-dir",
        default=None,
        help="Local memory root (environment default when omitted).",
    )
    ap.add_argument(
        "--explore",
        action="store_true",
        help="Enable exploration and memory distillation.",
    )
    ap.add_argument(
        "--dashboard",
        action="store_true",
        help="Start a local dashboard server for this single run.",
    )
    ap.add_argument(
        "--dashboard-host",
        default="127.0.0.1",
        help="Dashboard bind host. Defaults to 127.0.0.1.",
    )
    ap.add_argument(
        "--dashboard-port",
        type=int,
        default=0,
        help="Dashboard port. 0 asks the OS for a free port.",
    )
    ap.add_argument(
        "--dashboard-language",
        choices=["en", "zh-cn"],
        default="en",
        help="Dashboard UI language. 'zh-cn' serves the Chinese "
        "translation; defaults to English.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging for stdout and the run.log "
        "file. Defaults to INFO when not set.",
    )
    ap.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode: opens an interactive cli session.",
    )

    return ap


def _handoff_message(output_dir, session_number: int, session_max: int) -> str:
    """Build the opening message for a continuation session."""
    attempts_dir = Path(output_dir) / "attempts"
    prior = (
        sorted(p.name for p in attempts_dir.glob("attempt_*_failed.json"))
        if attempts_dir.is_dir()
        else []
    )
    return (
        f"You are agent {session_number} of up to {session_max} on this cell. "
        f"{len(prior)} attempt(s) by earlier agents are archived in "
        f"{attempts_dir}/ ({', '.join(prior) if prior else 'none yet'}), and their "
        "working notes are in the memory inbox under wip/.\n\n"
        "Read every archive and the working notes before acting. Do not repeat "
        "failed approaches. A fresh toolkit has already restored a clean scene; "
        "inspect it before acting."
    )


def _start_continuation_session(
    args,
    *,
    output_dir,
    recipe_tag,
    dashboard_events,
    prompt_bundle,
    prompt_vars,
    session_number: int,
    session_max: int,
):
    """Build a fresh planner and prompts for an exploration handoff."""
    logger.info("=== handing off to agent %d/%d ===", session_number, session_max)
    planner = build_planner(
        args.planner,
        output_dir=output_dir,
        recipe_tag=recipe_tag,
        robot_name=args.robot_name,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        planner_timeout_s=args.planner_timeout_s,
        reasoning_effort=args.reasoning_effort,
        claude_code_max_budget_usd=args.claude_code_max_budget_usd,
        dashboard_events=dashboard_events,
        no_images=args.no_images,
    )
    system_prompt = prompt_bundle.render(
        "system",
        variables={
            **prompt_vars,
            "session_number": session_number,
            "session_max": session_max,
        },
    )
    session_message = _handoff_message(output_dir, session_number, session_max)
    return planner, system_prompt, session_message


def main() -> int:
    parser = _build_argparser()
    # Two-phase argparse: first grab --robot / --env / --dashboard so we know
    # which robot's flags to add and whether to make its required flags optional.
    early, _ = parser.parse_known_args()

    # --env is a deprecated alias for --robot; resolve it before loading the spec.
    if early.env_name is not None:
        if early.robot_name is not None:
            parser.error("--robot and --env are aliases; provide only one of them")
        logger.warning("--env is deprecated and will be removed; use --robot instead")
        early.robot_name = early.env_name
    if early.robot_name is None:
        if "-h" not in sys.argv and "--help" not in sys.argv:
            parser.error("--robot is required")
    else:
        robot_spec = get_robot_spec(early.robot_name)
        robot_spec.add_cli_args(parser, use_dashboard=early.dashboard)
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="show this help message and exit",
    )
    args = parser.parse_args()
    args.robot_name = early.robot_name
    if args.dashboard and args.interactive:
        parser.error("--dashboard and --interactive cannot be used together")
    if args.explore and args.robot_name != "libero":
        parser.error("--explore is currently supported only for LIBERO")
    if args.explore and args.memory_profile == "hf":
        parser.error("--explore cannot be used with --memory-profile hf")
    if args.explore and getattr(args, "explore_sessions", 1) <= 0:
        parser.error("--explore-sessions must be greater than 0")
    args.memory_profile = args.memory_profile or ("local" if args.explore else "hf")
    if args.memory_profile == "hf" and args.memory_dir is not None:
        parser.error("--memory-dir requires --memory-profile local or --explore")
    if args.dashboard:
        from rpent.cli.dashboard import run_dashboard_session

        return run_dashboard_session(args, robot_spec, parser=parser)

    run_config = robot_spec.parse_config(args)
    recipe_tag = run_config.recipe_tag
    output_dir = run_config.output_dir
    prompt_vars = run_config.prompt_vars
    task_desc = run_config.task_desc

    robot_name = args.robot_name

    # mkdir + logging wiring (robot-side already picked the path).
    output_dir = init_output_dir(output_dir, verbose=args.verbose)
    logger.info("physical agent cmd: %s", shlex.join([sys.executable, *sys.argv]))

    # Preserve the original HF-backed evaluation behavior by default.
    memory_profile = getattr(args, "memory_profile", "hf")
    if not getattr(args, "explore", False) and memory_profile == "hf":
        ensure_resources(robot_spec)
    else:
        logger.info("resources: using local %s memory profile", memory_profile)

    dashboard_events = NullDashboardEventSink()

    planner = build_planner(
        args.planner,
        output_dir=output_dir,
        recipe_tag=recipe_tag,
        robot_name=robot_name,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        planner_timeout_s=args.planner_timeout_s,
        reasoning_effort=args.reasoning_effort,
        claude_code_max_budget_usd=args.claude_code_max_budget_usd,
        dashboard_events=dashboard_events,
        no_images=args.no_images,
    )
    prompt_bundle = robot_spec.prompts
    prompt_vars = {**prompt_vars, "output_dir": output_dir}
    system_prompt = prompt_bundle.render(
        "system",
        variables=prompt_vars,
    )
    user_msg = prompt_bundle.render(
        "user",
        variables=prompt_vars,
    )

    input_queue: "queue.Queue[str | None] | None" = None
    await_first_prompt: "Callable[[], str | None] | None" = None
    if args.interactive:
        input_queue = queue.Queue()
        # Pre-fill the first prompt with the rendered default task (editable
        # preset);
        start_interactive_reader(input_queue, first_prompt_default=user_msg)
        logger.info(
            "interactive mode on: the built-in task is pre-filled — "
            "edit it and press Enter, submit it as-is, or clear it to "
            "type your own. Once running, type to steer the agent. "
            "/help for commands."
        )
        # Resolve the opening prompt on a background thread so the user can type
        # it while the (slow) env/VLA servers boot below.
        await_first_prompt = start_first_prompt_resolver(input_queue)

    # --- initialise robot runtime --------------------------------------------
    daemons, primitives_kwargs = robot_spec.init_runtime(
        args,
        output_dir,
        dashboard_events,
        None,
    )

    # --- agent loop --------------------------------------------------------
    t0 = time.time()
    finish_result, messages, agent_error = None, [], None
    stats: dict = {}
    first_user_msg: str | None = user_msg
    if await_first_prompt is not None:
        # Block until the opening prompt typed during startup is ready.
        first_user_msg = await_first_prompt()
        if first_user_msg is None:
            logger.info("no task entered; ending session before start.")
    # Exploration may hand off between independent planner contexts.
    sessions = max(1, int(getattr(args, "explore_sessions", 1) or 1))
    if not getattr(args, "explore", False):
        sessions = 1
    recipe_path = ""
    solved = False
    memory_manager: MemoryManager | None = None
    try:
        if first_user_msg is not None:
            dashboard_events.emit(RunStartedEvent())
        session_msg = first_user_msg
        for session_number in range(1, sessions + 1):
            if session_msg is None:
                break
            if session_number > 1:
                planner, system_prompt, session_msg = _start_continuation_session(
                    args,
                    output_dir=output_dir,
                    recipe_tag=recipe_tag,
                    dashboard_events=dashboard_events,
                    prompt_bundle=prompt_bundle,
                    prompt_vars=prompt_vars,
                    session_number=session_number,
                    session_max=sessions,
                )
            state_output_dir = output_dir
            if getattr(args, "explore", False):
                state_output_dir = (
                    output_dir / "sessions" / f"session_{session_number:03d}"
                )
            if robot_name == "libero":
                toolkit = get_toolkit(
                    robot_name,
                    primitives_kwargs=primitives_kwargs,
                    dashboard_events=dashboard_events,
                    config=run_config,
                    mode="exploration" if args.explore else "evaluation",
                    attempts_per_session=getattr(
                        args, "explore_attempts_per_session", 0
                    ),
                    state_output_dir=state_output_dir,
                )
            else:
                toolkit = get_toolkit(
                    robot_name,
                    primitives_kwargs=primitives_kwargs,
                    dashboard_events=dashboard_events,
                    config=run_config,
                )
            memory_manager = toolkit.memory
            try:
                result = planner.solve(
                    system_prompt=system_prompt,
                    user_message=session_msg,
                    toolkit=toolkit,
                    max_turns=args.max_turns,
                    input_queue=input_queue,
                )
                finish_result = result.finish_result
                messages += result.messages
                stats = result.stats
                agent_error = result.error
                if robot_name == "libero":
                    solved = toolkit.solved()
                    if solved:
                        recipe_path = toolkit.write_recipe(recipe_tag)
            finally:
                toolkit.close()
            if solved:
                break
            if agent_error:
                if (
                    getattr(args, "explore", False)
                    and session_number < sessions
                    and "timed out" in agent_error.lower()
                ):
                    logger.warning(
                        "session %d/%d timed out; continuing with a fresh handoff",
                        session_number,
                        sessions,
                    )
                    continue
                break
    except Exception as exc:
        agent_error = f"{type(exc).__name__}: {exc}"
        logger.error("EXCEPTION in agent loop: %s", agent_error)
    finally:
        if recipe_path:
            logger.info("recipe: %s", recipe_path)
        else:
            logger.info("recipe: not written (cell unsolved)")
        for d in daemons:
            d.stop()

    elapsed = time.time() - t0

    transcript_path = Path(output_dir) / f"transcript_{recipe_tag}.json"
    record = {
        **task_desc,
        "model": args.model,
        "elapsed_s": round(elapsed, 1),
        "finish": finish_result,
        "stats": stats,
        "messages": _serialize_messages(messages),
    }
    with open(transcript_path, "a") as f:
        json.dump(record, f, indent=2, default=str)

    logger.info("elapsed: %.1fs", elapsed)
    logger.info(
        "usage: in=%s out=%s tool_calls=%s",
        stats.get("total_input_tokens", "?"),
        stats.get("total_output_tokens", "?"),
        stats.get("tool_calls", "?"),
    )
    logger.info("transcript: %s", transcript_path)

    # Publish exploration artifacts into the corpus after the session loop.
    if (
        getattr(args, "explore", False)
        and getattr(args, "auto_merge_memory", False)
        and not agent_error
        and memory_manager is not None
    ):
        try:
            merge_result = memory_manager.merge_memory(
                cell_tag=run_config.recipe_tag,
                run_state_dir=run_config.output_dir,
                solved=solved,
            )
            if merge_result:
                logger.info("memory merged: %s", merge_result)
        except Exception as exc:
            agent_error = f"memory finalization failed: {type(exc).__name__}: {exc}"
            logger.error("%s", agent_error)

    return 1 if agent_error else 0


if __name__ == "__main__":
    sys.exit(main())
