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

"""Static robot-extension descriptor.

Lives in :mod:`rpent.robots` alongside
:class:`~rpent.robots.prompt_bundle.PromptBundle` so robots
and planners can both import it without pulling in
:mod:`rpent.tools` or the RPC transport layer. Tool schemas,
handlers, server lifecycle, and the MCP allowlist live on
:class:`rpent.tools.toolkit.Toolkit` and its robot subclasses —
``RobotSpec`` carries the robot identity, the prompt bundle, and runner hooks
that keep CLI orchestration independent of concrete robot implementations.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rpent.dashboard.events import DashboardEventSink
from rpent.dashboard.spec import DashboardSpec
from rpent.evaluation import RunFinalizer
from rpent.robots.prompt_bundle import PromptBundle

if TYPE_CHECKING:
    from rpent.utils.daemon import ProcessDaemon


@dataclass(frozen=True)
class RunConfig:
    """Derived per-run identifiers produced by :attr:`RobotSpec.parse_config`."""

    recipe_tag: str
    output_dir: Path
    prompt_vars: dict[str, Any]
    task_desc: dict[str, Any]


@dataclass(frozen=True)
class RobotSpec:
    """Robot-level (non-tool) extension points for RPent."""

    name: str
    prompts: PromptBundle
    add_cli_args: Callable[[argparse.ArgumentParser, bool], None]
    parse_config: Callable[[argparse.Namespace], RunConfig]
    init_runtime: Callable[
        [argparse.Namespace, Path, DashboardEventSink, set[str] | None],
        tuple[list["ProcessDaemon"], dict[str, Any]],
    ]
    dashboard: DashboardSpec | None = None
    #: Physical scene restoration requires operator involvement, not simulator reset.
    #: Real-robot runs require an exclusive operator terminal.
    is_real_robot: bool = False
    #: Whether the robot implements the exploration-time toolkit contract.
    #: The CLI uses this instead of hard-coding robot names so real-robot
    #: extensions can opt into exploration with their own reset semantics.
    supports_exploration: bool = False
    #: Supports human-interactive exploration via the shared HIL input broker.
    #: Requires supports_exploration, get_toolkit(operator_input=...), and toolkit
    #: request_direct_verdict/finalize_direct_verdict/direct_verdict_requested.
    #: Operator success gates automatic memory publication for this capability.
    supports_human_interactive_exploration: bool = False
    memory_repo_id: str = "RLinf/RPent-memory"
    finalize_run: RunFinalizer | None = None
    #: Replay this robot's recorded plan for one cell, in place of a planner.
    #: Takes the toolkit, the cell tag, and a note sink; returns at least
    #: ``{"done": bool}``. Left unset by robots without Flash Mode.
    run_flash: Callable[[Any, str, Callable[[str], None]], dict[str, Any]] | None = None
