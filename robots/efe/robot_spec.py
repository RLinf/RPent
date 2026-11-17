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

"""EFE robot extension hooks.

The task container provides one episode through ``rcb_client``. RPent connects
to that existing service and starts no simulator. It sends an explicit reset
only when the task declares a reset budget.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.efe import session_vars
from robots.efe.prompt_bundle import system_prompt, user_prompt
from rpent.dashboard.events import DashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.utils.config import get_memory_dir, get_repo_root

if TYPE_CHECKING:
    from rpent.utils.daemon import ProcessDaemon


def get_robot_spec() -> RobotSpec:
    """Return the EFE robot descriptor."""
    return RobotSpec(
        name="efe",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        prepare_prompt_vars=_prepare_prompt_vars,
        validate_args=_validate_args,
        prepare_memory=_prepare_memory,
    )


def get_toolkit(
    *,
    runtime_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
    mode: str = "evaluation",
    attempts_per_session: int = 0,
    state_output_dir: Path | str | None = None,
    **_: Any,
):
    """Build the EFE toolkit over the connected service client."""
    from robots.efe.toolkit import EfeToolkit

    memory = MemoryManager(
        root=config.prompt_vars.get("memory_dir") or get_memory_dir("efe"),
        memory_access="inbox_write" if mode == "exploration" else "read_only",
        inbox_cell_tag=config.recipe_tag if mode == "exploration" else None,
    )
    return EfeToolkit(
        client=runtime_kwargs["client"],
        dashboard_events=dashboard_events,
        memory=memory,
        state_output_dir=state_output_dir or config.output_dir,
        enable_skills=bool(config.prompt_vars.get("skills", True)),
    )


def _prepare_prompt_vars(
    args: argparse.Namespace,
    config: RunConfig,
    runtime_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Report the tools the connected task registers.

    The generic CLI renders the system prompt after this hook, so the prompt
    lists the tools the session actually registers: an unsupported joint
    layout keeps the raw submit tool, and the service spec decides reset and
    hidden success.
    """
    from robots.efe.toolkit import tool_capabilities

    return tool_capabilities(
        runtime_kwargs["client"],
        enable_skills=bool(config.prompt_vars.get("skills", True)),
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    """Add the EFE service and task arguments.

    EFE has no Dashboard entry point; ``use_dashboard`` keeps
    ``--instruction-file`` optional so the Dashboard path reports that instead
    of a missing argument. ``_parse_config`` still requires it.
    """
    parser.add_argument(
        "--instruction-file",
        type=Path,
        required=not use_dashboard,
        help="File holding the task instruction supplied by EFE.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Public agent workspace. Default: the current directory.",
    )
    parser.add_argument(
        "--efe-base-url",
        default=None,
        help="Simulator service URL. Default: probe sim:8000, then localhost:8000.",
    )
    parser.add_argument(
        "--efe-connect-timeout-s",
        type=float,
        default=3600.0,
        help="How long to wait for the simulator to answer.",
    )
    parser.add_argument(
        "--efe-no-skills",
        dest="efe_skills",
        action="store_false",
        help="Register only the raw EFE tools, without high-level skills.",
    )
    parser.set_defaults(efe_skills=True)


def _validate_args(args: argparse.Namespace) -> None:
    # EFE does not declare ``supports_exploration``, so the generic CLI rejects
    # ``--explore`` before robot hooks run. Keep the guard for callers that
    # invoke this hook directly. This CLI has no ``--explore-sessions`` option:
    # the Harbor adapter and ``run_efe`` own the session count.
    if getattr(args, "explore", False):
        raise ValueError(
            "The EFE runner starts its own sessions; set explore_sessions on "
            "RPentEfeAgent or pass explore_sessions to run_efe"
        )
    if getattr(args, "memory_repo", None) is not None:
        raise ValueError(
            "--memory-repo is not supported for EFE; use --memory-profile local "
            "with --memory-dir, or pass memory_source to RPentEfeAgent"
        )
    if args.instruction_file is not None and not args.instruction_file.is_file():
        raise ValueError(
            f"The instruction file does not exist: {args.instruction_file}"
        )
    if args.efe_connect_timeout_s <= 0:
        raise ValueError("--efe-connect-timeout-s must be positive")


def _parse_config(args: argparse.Namespace) -> RunConfig:
    """Derive the run identity and prompts from the EFE instruction."""
    if args.instruction_file is None:
        raise ValueError("--instruction-file is required")
    instruction_path = Path(args.instruction_file).expanduser().resolve(strict=True)
    instruction = instruction_path.read_text(encoding="utf-8")
    task_name = instruction_path.parent.name or "efe"
    tag = session_vars.recipe_tag(task_name, instruction)
    workdir = Path(args.workdir or Path.cwd()).expanduser().resolve()
    memory_dir = (
        Path(args.memory_dir).expanduser().resolve()
        if getattr(args, "memory_dir", None)
        else get_memory_dir("efe")
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else get_repo_root() / "logs" / f"{datetime.now():%Y%m%d-%H:%M:%S}_{tag}"
    )
    # ``build_planner`` and ``MemoryManager`` resolve their root through
    # ``get_repo_root()``. The planner is built before ``init_runtime`` runs, so
    # bind the trial workspace here rather than in that hook; otherwise a manual
    # run resolves relative paths and the Codex working directory against the
    # installed package instead of the task workspace.
    if not os.environ.get("RPENT_REPO_ROOT"):
        os.environ["RPENT_REPO_ROOT"] = str(workdir)
    prompt_vars = {
        **session_vars.prompt_vars(
            workdir=workdir,
            output_dir=output_dir,
            memory_dir=memory_dir,
            inbox_tag=tag,
            session_max=1,
            skills=bool(getattr(args, "efe_skills", True)),
        ),
        "recipe_tag": tag,
        "mode": "eval",
        "task_name": task_name,
        "instruction": instruction,
        "session_number": 1,
    }
    return RunConfig(
        recipe_tag=tag,
        output_dir=output_dir,
        prompt_vars=prompt_vars,
        task_desc={"task": task_name, "instruction": instruction},
    )


def _prepare_memory(args: argparse.Namespace, config: RunConfig) -> None:
    """Create the run's memory root; the runner never downloads a corpus."""
    memory_dir = Path(config.prompt_vars["memory_dir"])
    memory_dir.mkdir(parents=True, exist_ok=True)


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Connect to the existing simulator service. This starts no subprocess."""
    from robots.efe.env_client import EfeEnvClient

    client = EfeEnvClient(
        base_url=args.efe_base_url,
        connect_timeout_s=args.efe_connect_timeout_s,
    )
    return [], {"client": client}
