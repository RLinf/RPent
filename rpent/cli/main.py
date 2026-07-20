"""Physical agent main CLI entrypoint."""

# `rpent/cli/`
#
# CLI entrypoints for RPent (currently just `main.py`).
#
# ## Rules
#
# - **No `__init__.py`.** This directory is not a Python subpackage of
#   `rpent`. Do not add one.
# - **Never reference `rpent.cli` as a dotted import path** anywhere in
#   the codebase. Without `__init__.py` it isn't one, and treating it as
#   one would risk import cycles — `main.py` already pulls in
#   `rpent.cerebrum`, `rpent.envs`, `rpent.utils`, `rpent.dashboard`, and
#   `rpent.tools`, so exposing it as an importable submodule invites a
#   cycle back to the CLI.
# - **Setuptools skips it.** `pyproject.toml`'s `packages.find` only picks
#   up dirs with `__init__.py`, so the CLI ships as source-tree scripts,
#   not as an installed submodule.
#
# ## Run
#
# ```bash
# python rpent/cli/main.py --suite libero_object_task --task 0 --seed 0 [...]
# ```
#
# Do not use `python -m rpent.cli.main`.
from __future__ import annotations

import argparse
import json
import shlex
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from rpent.cerebrum.base import build_cerebrum  # noqa: E402
from rpent.envs import get_env_spec, get_runtime  # noqa: E402
from rpent.utils.config import (
    get_repo_root,
)
from rpent.utils.logging import get_logger, init_output_dir  # noqa: E402

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
    ap = argparse.ArgumentParser(
        description="Standalone hybrid LLM-in-the-loop agent for LIBERO PRO",
    )

    # models
    ap.add_argument(
        "--cerebrum",
        default="api",
        choices=["api", "claude_code", "codex"],
        help="LLM backend: api | claude_code | codex.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Model id. For the 'api' cerebrum you need to prefix provider to the model id "
        "(e.g. anthropic:claude-opus-4-8, openai:gpt-5.5, "
        "openai-chat:glm-5.2).",
    )
    ap.add_argument(
        "--base-url",
        default=None,
        help="API base URL. Defaults to the selected backend's base URL env var.",
    )
    ap.add_argument(
        "--api-key",
        default=None,
        help="API key. Defaults to the selected backend's API key env var.",
    )
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument(
        "--cerebrum-timeout-s",
        type=int,
        default=None,
        help="Wall-clock cap for the claude_code/codex cerebrum "
        "subprocess. Defaults to CODEX_TIMEOUT_S (codex only), "
        "CELL_TIMEOUT_S, or 1200.",
    )
    ap.add_argument(
        "--claude-code-max-budget-usd",
        type=float,
        default=None,
        help="Budget passed to claude -p --max-budget-usd. "
        "Defaults to MAX_BUDGET_USD env or 10.",
    )

    # env_server / vla_server / transport
    ap.add_argument(
        "--no-driver",
        action="store_true",
        help="Don't spawn driver; attach to existing output dir",
    )
    ap.add_argument(
        "--env-endpoint",
        default="127.0.0.1",
        help="Host of an existing env server to connect to; required with --no-driver.",
    )
    ap.add_argument(
        "--env-port",
        type=int,
        default=0,
        help="Port of an existing env server to connect to; required with --no-driver.",
    )
    ap.add_argument(
        "--vla-endpoint",
        default=None,
        help="Base URL of an existing vla_server (e.g. http://host:8000). "
        "If omitted with a spawned driver, a local vla_server is started; "
        "required with --no-driver.",
    )
    ap.add_argument(
        "--cuda-device",
        default=None,
        help="GPU device(s) to expose via CUDA_VISIBLE_DEVICES.",
    )

    # other config
    ap.add_argument("--output-dir", default=None)
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
        "variant (index.zh-cn.html); defaults to English.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging for stdout and the run.log "
        "file. Defaults to INFO when not set.",
    )

    # environments
    ap.add_argument(
        "--env",
        dest="env_name",
        default="libero",
        help="Environment backend. Defaults to libero.",
    )
    ap.add_argument(
        "--instruction",
        default=None,
        help="Natural-language task for physical robot environments.",
    )
    ap.add_argument(
        "--env-config", default=None, help="Environment-specific configuration file."
    )
    ap.add_argument("--max-episode-steps", type=int, default=10000)

    ap.add_argument(
        "--libero-type",
        default=None,
        choices=["standard", "pro", "plus"],
        help="LIBERO variant (auto-routed from suite suffix if not set).",
    )
    ap.add_argument(
        "--suite", default=None, help="e.g. libero_object_task, libero_spatial_swap"
    )
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)

    return ap


def main() -> int:
    """Run one RPent agent session for the selected environment."""
    parser = _build_argparser()
    args = parser.parse_args()

    # With --dashboard, open the launcher first: serve the start screen, then
    # block until the user clicks Run and overlay their choices onto args.
    # Everything downstream (output_dir, State, run loop) then sees final args.
    dashboard_server = None
    dashboard_url = None
    if args.dashboard:
        from rpent.dashboard import DashboardServer
        from rpent.dashboard.launcher import apply_to_args, defaults_from_args

        dashboard_server = DashboardServer(
            host=args.dashboard_host,
            port=args.dashboard_port,
            language=args.dashboard_language,
        )
        dashboard_url = dashboard_server.start()
        print(
            f"Dashboard: {dashboard_url}. Open it, adjust the run config, and click Run to start.",
            flush=True,
        )
        launch_config = dashboard_server.wait_for_launch(
            defaults=defaults_from_args(args)
        )
        apply_to_args(args, launch_config)
        logger.info("launcher config applied: %s", launch_config)

    env_name = args.env_name
    suite = args.suite or env_name
    task = args.task if args.task is not None else 0
    seed = args.seed
    env_spec = get_env_spec(env_name)
    prompt_bundle = env_spec.prompts

    # resolve output directory
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = get_repo_root() / "logs" / f"{timestamp}_{suite}_t{task}_s{seed}"
    output_dir = init_output_dir(output_dir, verbose=args.verbose)
    logger.info("physical agent cmd: %s", shlex.join([sys.executable, *sys.argv]))

    recipe_tag = f"{suite.replace('libero_', '')}_t{task}_s{seed}"

    dashboard_state = None
    if args.dashboard and dashboard_server is not None:
        from rpent.dashboard.state import State

        dashboard_state = State(
            run_id=f"{suite}/{output_dir.name}",
            name=recipe_tag,
            suite=suite,
            task=task,
            seed=seed,
            output_dir=str(output_dir),
            video_path=str(Path(output_dir) / "episode.mp4"),
        )
        # Server is already serving the launcher; register the run so the
        # frontend can switch from the start screen to the live monitor.
        dashboard_server.register(dashboard_state)

    cerebrum = build_cerebrum(
        args.cerebrum,
        output_dir=output_dir,
        recipe_tag=recipe_tag,
        env_name=env_name,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        cerebrum_timeout_s=args.cerebrum_timeout_s,
        claude_code_max_budget_usd=args.claude_code_max_budget_usd,
        dashboard=dashboard_state,
    )

    prompt_vars = {
        "suite": suite,
        "task": task,
        "seed": seed,
        "instruction": args.instruction
        or "Inspect the robot and wait for a safe instruction.",
        "output_dir": output_dir,
        "recipe_tag": recipe_tag,
    }
    system_prompt = prompt_bundle.render(
        "system",
        variables=prompt_vars,
    )
    user_msg = prompt_bundle.render(
        "user",
        variables=prompt_vars,
    )

    runtime = get_runtime(
        env_name,
        args=args,
        output_dir=str(output_dir),
        dashboard=dashboard_state,
    )
    toolkit = None

    t0 = time.time()
    finish_result, messages, agent_error = None, [], None
    stats: dict = {}
    try:
        toolkit = runtime.start()
        result = cerebrum.solve(
            system_prompt=system_prompt,
            user_message=user_msg,
            toolkit=toolkit,
            max_turns=args.max_turns,
        )
        finish_result = result.finish_result
        messages = result.messages
        stats = result.stats
        agent_error = result.error
    except Exception as e:
        logger.error("EXCEPTION in agent loop: %s", e)
    finally:
        if toolkit is not None:
            recipe_path = toolkit.write_recipe(recipe_tag)
            logger.info("recipe: %s", recipe_path)
            toolkit.close()
        runtime.stop()

    elapsed = time.time() - t0

    transcript_path = Path(output_dir) / f"transcript_{recipe_tag}.json"
    record = {
        "suite": suite,
        "task": task,
        "seed": seed,
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
    if agent_error:
        logger.error("error: %s", agent_error)

    if args.dashboard and dashboard_state is not None:
        dashboard_state.mark_done()
        logger.info(
            "Run finished. Dashboard still serving at %s. Press Ctrl+C to stop.",
            dashboard_url,
        )
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
