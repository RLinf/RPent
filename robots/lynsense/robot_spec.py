# Copyright 2026 The RPent Authors.
# SPDX-License-Identifier: Apache-2.0

"""RPent native read-only backend; importing this module never imports ROS."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.lynsense.ros2_adapter import Ros2StateAdapter
from rpent.dashboard.events import DashboardEventSink
from rpent.dashboard.spec import DashboardSpec
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.utils.config import get_memory_dir, get_repo_root

if TYPE_CHECKING:
    from robots.lynsense.toolkit import LynsenseToolkit
    from rpent.utils.daemon import ProcessDaemon


LYNSENSE_DASHBOARD_SPEC: DashboardSpec = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task",
        "fields": (),
        "display": "Lynsense read-only state",
        "output_slug": "lynsense_readonly",
    },
    "runtime_components": ({"name": "env", "label": "LYNSENSE", "scope": "unique"},),
    "frame_channels": (),
    "primitives": (),
}


def _system_prompt(variables=None) -> str:
    return (
        "You observe the Lynsense robot's right arm in read-only mode. "
        "Your only tool is read_robot_state, which takes no arguments. "
        "Report the returned status and reason; only status ok confirms readable "
        "data. Never claim an observation without calling the tool. Non-ok data "
        "is not a successful observation. Receipt freshness is not sensor time "
        "synchronization. No status proves robot health, safety, or readiness to "
        "move. Do not perform or invent motion, gripper, enable, reset, shell, "
        "file, or image operations. End with an ordinary text summary; there "
        "is no finish tool."
    )


def _user_prompt(variables=None) -> str:
    return "Call read_robot_state and summarize the right-arm observation and its limitations."


def get_robot_spec() -> RobotSpec:
    return RobotSpec(
        name="lynsense",
        prompts=PromptBundle(system=_system_prompt, user=_user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        dashboard=LYNSENSE_DASHBOARD_SPEC,
        is_real_robot=True,
        supports_exploration=False,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    parser.add_argument("--ros-domain-id", type=int, default=3)
    parser.add_argument("--joint-state-topic", default="/right_xarm/joint_states")
    parser.add_argument("--robot-state-topic", default="/right_xarm/robot_states")
    parser.add_argument("--state-timeout", type=float, default=5.0)
    parser.add_argument("--state-max-age", type=float, default=2.0)


def _make_adapter(args: argparse.Namespace) -> Ros2StateAdapter:
    return Ros2StateAdapter(
        expected_domain_id=args.ros_domain_id,
        joint_topic=args.joint_state_topic,
        robot_topic=args.robot_state_topic,
        connect_timeout_s=args.state_timeout,
        stale_after_s=args.state_max_age,
    )


def _parse_config(args: argparse.Namespace) -> RunConfig:
    if getattr(args, "planner", None) != "api":
        raise ValueError("lynsense read-only requires --planner api")
    if getattr(args, "memory_profile", None) != "local":
        raise ValueError("lynsense read-only requires --memory-profile local")
    if any(getattr(args, mode, False) for mode in ("dashboard", "interactive", "explore")):
        raise ValueError("lynsense read-only does not support dashboard, interactive, or exploration")
    _make_adapter(args)  # Validate configuration without ROS imports or resources.
    requested_output = getattr(args, "output_dir", None)
    output_dir = (
        Path(requested_output).expanduser().resolve()
        if requested_output
        else get_repo_root() / "logs" / (
            "lynsense_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        )
    )
    memory_dir = getattr(args, "memory_dir", None)
    return RunConfig(
        recipe_tag="lynsense_readonly",
        output_dir=output_dir,
        prompt_vars={"memory_dir": str(Path(memory_dir).expanduser().resolve())} if memory_dir else {},
        task_desc={"operation": "read_robot_state", "arm": "right"},
    )


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    selected = {"env"} if components is None else components
    if selected - {"env"}:
        raise ValueError(f"unsupported lynsense runtime components: {sorted(selected)}")
    return ([], {"adapter": _make_adapter(args)}) if selected else ([], {})


def get_toolkit(
    *,
    runtime_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
) -> LynsenseToolkit:
    """Consume the inert adapter; release it on any construction/start failure."""
    adapter = runtime_kwargs.pop("adapter")
    # A rejected claim must not close another toolkit's live resources.
    adapter.claim_toolkit()
    try:
        from robots.lynsense.toolkit import LynsenseToolkit

        toolkit = LynsenseToolkit(
            adapter=adapter,
            dashboard_events=dashboard_events,
            memory=MemoryManager(config.prompt_vars.get("memory_dir") or get_memory_dir("lynsense")),
        )
        connection = adapter.connect()
        if connection["status"] != "ok":
            raise RuntimeError(connection["reason"])
        return toolkit
    except BaseException:
        adapter.close()
        raise
