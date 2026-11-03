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

"""Run the existing Codex planner inside an EFE agent container."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from rpent.benchmark.efe import prompts
from rpent.benchmark.efe.toolkit import EfeToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory.manager import MemoryManager
from rpent.planner.base import REASONING_EFFORTS, is_recoverable_agent_error
from rpent.utils.logging import get_logger, init_output_dir

if TYPE_CHECKING:
    from rpent.planner.base import PlannerResult

logger = get_logger("efe")

_USAGE_KEYS = (
    "total_input_tokens",
    "total_cached_input_tokens",
    "total_output_tokens",
    "total_reasoning_output_tokens",
)


@contextmanager
def _trial_codex_home(output_dir: Path) -> Iterator[None]:
    """Point the Codex child at trial-local state and restore the host value."""
    codex_home = output_dir / "codex-home"
    codex_home.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("CODEX_HOME")
    os.environ["CODEX_HOME"] = str(codex_home)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CODEX_HOME", None)
        else:
            os.environ["CODEX_HOME"] = previous


@contextmanager
def _trial_repo_root(workdir: Path) -> Iterator[None]:
    """Resolve memory tool relative paths under the task workspace.

    ``MemoryManager`` resolves relative paths with ``get_repo_root()``. The
    installed package resolves that to its site-packages directory, so the
    task workspace must be the explicit root unless the caller set one.
    """
    previous = os.environ.get("RPENT_REPO_ROOT")
    if previous is None:
        os.environ["RPENT_REPO_ROOT"] = str(workdir)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("RPENT_REPO_ROOT", None)
        else:
            os.environ["RPENT_REPO_ROOT"] = previous


def _aggregate_stats(sessions: list[dict[str, object]]) -> dict[str, object]:
    """Sum session counters; a single session keeps its full planner stats.

    ``usage_observed`` stays true only when every session reported usage.
    Otherwise the zero counters mean "unknown", not zero tokens.
    """
    if len(sessions) == 1:
        return sessions[0]["stats"]
    totals: dict[str, object] = dict.fromkeys(_USAGE_KEYS, 0)
    totals["turns_used"] = 0
    totals["tool_calls"] = 0
    observed = bool(sessions)
    for session in sessions:
        stats = session["stats"]
        observed = observed and stats.get("usage_observed") is True
        for key in ("turns_used", "tool_calls", *_USAGE_KEYS):
            totals[key] += int(stats.get(key) or 0)
    return {"backend": "codex_sdk", "usage_observed": observed, **totals}


def run_efe(
    instruction: str,
    *,
    workdir: Path,
    output_dir: Path,
    model: str,
    timeout_s: int = 600,
    max_turns: int = 200,
    reasoning_effort: str = "none",
    memory_dir: Path | None = None,
    explore_sessions: int = 1,
) -> dict[str, object]:
    """Run an EFE coding session and save its configuration, usage, and error.

    The session uses RPent's prompt, memory corpus, and shared file tools.
    It may hand off across sessions. Without ``memory_dir`` the corpus starts
    empty, so the run measures RPent without prior experience.

    Args:
        instruction: The task instruction supplied by Harbor.
        workdir: The public agent workspace inside the task container.
        output_dir: A directory for RPent artifacts, separate from verifier output.
        model: The Codex model identifier.
        timeout_s: Maximum planner wall time in seconds across all sessions.
        max_turns: Maximum model responses per session.
        reasoning_effort: A reasoning level supported by the planner.
        memory_dir: An existing RPent memory corpus.
        explore_sessions: Number of independent planner sessions.

    Returns:
        The session report. This report contains no verifier reward.
    """
    if not instruction.strip() or not model.strip():
        raise ValueError("The task instruction and model must not be empty.")
    if timeout_s <= 0 or max_turns <= 0:
        raise ValueError("timeout_s and max_turns must be positive.")
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"Unsupported reasoning effort: {reasoning_effort}")
    if explore_sessions <= 0:
        raise ValueError("explore_sessions must be positive.")
    workdir = workdir.resolve(strict=True)
    if not workdir.is_dir():
        raise ValueError(f"The workspace is not a directory: {workdir}")
    output_dir = output_dir.resolve()
    if (output_dir / "session.json").exists():
        raise FileExistsError(f"The session report already exists: {output_dir}")
    init_output_dir(output_dir)
    if memory_dir is not None:
        memory_root = Path(memory_dir).expanduser().resolve(strict=True)
        if not memory_root.is_dir():
            raise ValueError(f"The memory corpus is not a directory: {memory_root}")
        memory_mode = "loaded"
    else:
        memory_root = output_dir / "memory"
        memory_root.mkdir(parents=True, exist_ok=True)
        memory_mode = "empty"
    explore = explore_sessions > 1
    memory = MemoryManager(
        memory_root,
        memory_access="inbox_write" if explore else "read_only",
        inbox_cell_tag="efe" if explore else None,
    )
    prompt_vars: dict[str, object] = {
        "workdir": str(workdir),
        "output_dir": str(output_dir),
        "memory_dir": str(memory_root),
        "memory_inbox": str(memory_root / "_internal" / "inbox" / "efe"),
        "session_max": explore_sessions,
    }
    report: dict[str, object] = {
        "compatibility": "EFE-compatible / non-official",
        "planner": "codex",
        "model": model,
        "workdir": str(workdir),
        "timeout_s": timeout_s,
        "max_turns": max_turns,
        "reasoning_effort": reasoning_effort,
        "memory_root": str(memory_root),
        "memory_mode": memory_mode,
        "explore_sessions": explore_sessions,
        "codex_home": str(output_dir / "codex-home"),
        "versions": {
            "rpent": version("rpent"),
            "openai-codex": version("openai-codex"),
            "openai-codex-cli-bin": version("openai-codex-cli-bin"),
        },
        "codex_binary_override": os.environ.get("CODEX_BIN"),
        "instruction": instruction,
        "agent_finish": None,
        "sessions": [],
        "stats": {},
        "error": None,
    }
    sessions: list[dict[str, object]] = report["sessions"]  # type: ignore[assignment]
    started = time.monotonic()
    deadline = started + timeout_s
    try:
        from rpent.planner.codex import CodexPlanner

        with _trial_codex_home(output_dir), _trial_repo_root(workdir):
            for number in range(1, explore_sessions + 1):
                remaining = math.ceil(deadline - time.monotonic())
                if remaining <= 0:
                    if not sessions:
                        report["error"] = (
                            "The planner timeout expired before the session started."
                        )
                    break
                session_dir = (
                    output_dir
                    if explore_sessions == 1
                    else output_dir / "sessions" / f"session_{number:03d}"
                )
                session_dir.mkdir(parents=True, exist_ok=True)
                toolkit = EfeToolkit(memory=memory)
                try:
                    planner = CodexPlanner(
                        output_dir=str(session_dir),
                        repo_root=workdir,
                        output_path=session_dir
                        / ("codex.txt" if number == 1 else f"codex_{number:02d}.txt"),
                        model=model,
                        timeout_s=remaining,
                        reasoning_effort=reasoning_effort,
                        dashboard_events=NullDashboardEventSink(),
                    )
                    session_vars = {**prompt_vars, "session_number": number}
                    system_prompt = prompts.system_prompt(
                        session_vars, explore=explore, handoff=number > 1
                    )
                    user_message = prompts.user_prompt(instruction, session_vars)
                    result: PlannerResult = planner.solve(
                        system_prompt=system_prompt,
                        user_message=user_message,
                        toolkit=toolkit,
                        max_turns=max_turns,
                    )
                finally:
                    toolkit.close()
                sessions.append(
                    {
                        "session": number,
                        "stats": result.stats,
                        "finish": result.finish_result,
                        "error": result.error,
                    }
                )
                report["agent_finish"] = result.finish_result
                if result.error is not None:
                    report["error"] = result.error
                    if (
                        explore
                        and number < explore_sessions
                        and is_recoverable_agent_error(result.error)
                    ):
                        logger.warning(
                            "session %d/%d ended with %s; continuing with a fresh handoff",
                            number,
                            explore_sessions,
                            result.error,
                        )
                        continue
                    break
                # A later successful session supersedes earlier recoverable errors.
                report["error"] = None
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("EFE coding session failed")
    finally:
        report["stats"] = _aggregate_stats(sessions)
        report["elapsed_s"] = round(time.monotonic() - started, 3)
        (output_dir / "session.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report
