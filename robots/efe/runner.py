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

"""Run the existing Codex planner as the EFE trial's only agent process."""

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

from robots.efe import prompt_bundle, session_vars
from robots.efe.constants import SANDBOX_CHOICES
from robots.efe.env_client import EfeEnvClient
from robots.efe.toolkit import EfeToolkit, tool_capabilities
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory.manager import MemoryManager
from rpent.planner.base import REASONING_EFFORTS, is_recoverable_agent_error
from rpent.robots.prompt_bundle import PromptBundle
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
# The planner reports an expired budget as one of these strings. Match them
# narrowly: ``Connection timed out`` and ``Request timed out`` are transient
# failures, and a real failure must not be reported as a clean stop.
# Anchored on the subject so a transient ``Request timed out after 30s`` stays a
# real failure instead of being reported as a clean budget stop.
_BUDGET_STOP_MARKERS = (
    "sdk timed out after",
    "planner timed out after",
    "planner timeout expired",
)


def _is_budget_stop(agent_error: str | None) -> bool:
    """Whether an error is the planner reaching its time budget.

    Harbor reads a non-zero agent exit as a failed trial, rescans the whole
    ``run.log`` for transient messages, and can mark a graded episode's reward
    invalid. An episode that used its budget is a clean stop, so this path
    exits zero.
    """
    lowered = (agent_error or "").lower()
    return any(marker in lowered for marker in _BUDGET_STOP_MARKERS)


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
    skills: bool = True,
    sandbox: str = "read-only",
    efe_base_url: str | None = None,
    connect_timeout_s: float = 3600.0,
    client: EfeEnvClient | None = None,
) -> dict[str, object]:
    """Run one EFE trial with RPent's planner and the EFE robot tools.

    The session uses RPent's prompt, memory corpus, and the EFE toolkit. It may
    hand off across sessions; all sessions share the one episode. Without
    ``memory_dir`` the corpus starts empty, so the run measures RPent without
    prior experience.

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
        skills: Replace the joint-level submit tool with the high-level skills.
        sandbox: Codex shell sandbox. ``read-only`` keeps shell commands away
            from the simulator, so only RPent tools act on the episode;
            ``full-access`` lets the planner script the service directly.
        efe_base_url: Simulator service URL. Default: let the client probe.
        connect_timeout_s: How long to wait for the simulator to answer.
        client: An existing client, for tests and for callers that already
            connected. Default: connect to the service.

    Returns:
        The session report. This report contains no verifier reward.
    """
    if not instruction.strip() or not model.strip():
        raise ValueError("The task instruction and model must not be empty.")
    if timeout_s <= 0 or max_turns <= 0:
        raise ValueError("timeout_s and max_turns must be positive.")
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"Unsupported reasoning effort: {reasoning_effort}")
    if sandbox not in SANDBOX_CHOICES:
        raise ValueError(f"Unsupported Codex sandbox: {sandbox}")
    if explore_sessions <= 0:
        raise ValueError("explore_sessions must be positive.")
    if not math.isfinite(connect_timeout_s) or connect_timeout_s <= 0:
        raise ValueError("connect_timeout_s must be positive.")
    # The connect wait counts against the trial's wall clock, so it cannot
    # outlast the planner budget: a 3600 s default would consume a 600 s trial
    # before the first observation.
    connect_timeout_s = min(connect_timeout_s, float(timeout_s))
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
    prompt_vars: dict[str, object] = session_vars.prompt_vars(
        workdir=workdir,
        output_dir=output_dir,
        memory_dir=memory_root,
        inbox_tag=session_vars.DEFAULT_INBOX_TAG,
        session_max=explore_sessions,
        skills=skills,
    )
    # The generic CLI renders the same factories, so this path cannot drift.
    prompts = PromptBundle(
        system=prompt_bundle.system_prompt, user=prompt_bundle.user_prompt
    )
    report: dict[str, object] = {
        "compatibility": "EFE-compatible / non-official",
        "planner": "codex",
        "model": model,
        "workdir": str(workdir),
        "timeout_s": timeout_s,
        "connect_timeout_s": connect_timeout_s,
        "max_turns": max_turns,
        "reasoning_effort": reasoning_effort,
        "codex_sandbox": sandbox,
        "memory_root": str(memory_root),
        "memory_mode": memory_mode,
        "explore_sessions": explore_sessions,
        "skills": skills,
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
        "budget_stop": None,
    }
    sessions: list[dict[str, object]] = report["sessions"]  # type: ignore[assignment]
    started = time.monotonic()
    deadline = started + timeout_s
    try:
        from rpent.planner.codex import CodexPlanner

        client = client or EfeEnvClient(
            base_url=efe_base_url,
            connect_timeout_s=connect_timeout_s,
        )
        report["service"] = {
            "protocol": client.spec.get("protocol"),
            "robot": client.spec.get("robot"),
            "action_types": sorted(client.spec["action_types"]),
        }
        with _trial_codex_home(output_dir), _trial_repo_root(workdir):
            for number in range(1, explore_sessions + 1):
                if client.done:
                    # The service refuses a reset only once the trial is
                    # settled. An attempt that ended early on a failure can
                    # still be abandoned while the reset budget lasts, so the
                    # next session starts a fresh attempt rather than skipping
                    # the rest of the budget.
                    if client.reset_calls >= client.reset_budget:
                        logger.info(
                            "the episode ended and the reset budget is spent; "
                            "no further session starts"
                        )
                        break
                    logger.info(
                        "the attempt ended (%s); resetting for session %d/%d",
                        client.done_reason or "no reason reported",
                        number,
                        explore_sessions,
                    )
                    try:
                        client.reset()
                    except (RuntimeError, ValueError) as exc:
                        logger.info("the service refused the reset: %s", exc)
                        break
                remaining = math.ceil(deadline - time.monotonic())
                if remaining <= 0:
                    if not sessions:
                        report["budget_stop"] = (
                            "The planner timeout expired before the session "
                            f"started ({timeout_s}s)."
                        )
                    break
                session_dir = (
                    output_dir
                    if explore_sessions == 1
                    else output_dir / "sessions" / f"session_{number:03d}"
                )
                session_dir.mkdir(parents=True, exist_ok=True)
                toolkit = EfeToolkit(
                    client=client,
                    dashboard_events=NullDashboardEventSink(),
                    memory=memory,
                    state_output_dir=session_dir,
                    enable_skills=skills,
                )
                solved = False
                try:
                    planner = CodexPlanner(
                        output_dir=str(session_dir),
                        repo_root=workdir,
                        output_path=session_dir
                        / ("codex.txt" if number == 1 else f"codex_{number:02d}.txt"),
                        model=model,
                        timeout_s=remaining,
                        reasoning_effort=reasoning_effort,
                        sandbox=sandbox,
                        dashboard_events=NullDashboardEventSink(),
                    )
                    per_session_vars = {
                        **prompt_vars,
                        "session_number": number,
                        "handoff": number > 1,
                        "write_notes": explore,
                        "restricted_shell": sandbox != "full-access",
                        **tool_capabilities(client, enable_skills=skills),
                    }
                    system_prompt = prompts.render("system", variables=per_session_vars)
                    user_message = prompts.render(
                        "user",
                        variables={
                            **per_session_vars,
                            "instruction": instruction.strip(),
                        },
                    )
                    result: PlannerResult = planner.solve(
                        system_prompt=system_prompt,
                        user_message=user_message,
                        toolkit=toolkit,
                        max_turns=max_turns,
                    )
                finally:
                    solved = toolkit.solved()
                    toolkit.close()
                sessions.append(
                    {
                        "session": number,
                        "solved": solved,
                        "stats": result.stats,
                        "finish": result.finish_result,
                        "error": result.error,
                        "metrics": toolkit.metrics(),
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
        logger.exception("EFE session failed")
    finally:
        # A terminal planner timeout is the budget running out, not a crash.
        # Report it separately so the CLI exits zero: Harbor treats a non-zero
        # agent exit as a failed trial, rescans the whole run.log for transient
        # messages, and can mark the reward of a validly graded episode invalid.
        if report["error"] is not None and _is_budget_stop(str(report["error"])):
            report["budget_stop"] = report["error"]
            report["error"] = None
            logger.info("the planner used its time budget: %s", report["budget_stop"])
        report["stats"] = _aggregate_stats(sessions)
        if client is not None:
            report["efe"] = {
                "submit_calls": client.submit_calls,
                "observe_calls": client.observe_calls,
                "control_steps": client.control_steps,
                "done": client.done,
                "done_reason": client.done_reason,
                "solved": any(bool(session.get("solved")) for session in sessions),
            }
        report["elapsed_s"] = round(time.monotonic() - started, 3)
        (output_dir / "session.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report
