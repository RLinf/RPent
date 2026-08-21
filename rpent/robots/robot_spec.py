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
    init_shared_runtime: Callable[
        [argparse.Namespace, Path, DashboardEventSink],
        tuple[list["ProcessDaemon"], dict[str, Any]],
    ]
    init_task_runtime: Callable[
        [argparse.Namespace, Path, DashboardEventSink],
        tuple[list["ProcessDaemon"], dict[str, Any]],
    ]
    init_runtime: Callable[
        [argparse.Namespace, Path, DashboardEventSink],
        tuple[list["ProcessDaemon"], dict[str, Any]],
    ]
    dashboard: dict[str, Any] | None = None
    finalize_run: (
        Callable[[argparse.Namespace, RunConfig], dict[str, Any] | None] | None
    ) = None
