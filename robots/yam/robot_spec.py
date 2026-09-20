# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""YAM robot extension runner hooks."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from robots.yam.contracts import (
    MODEL_SPEC,
    env_runtime_contract,
    vla_runtime_contract,
)
from robots.yam.evaluation import finalize_run
from robots.yam.prompt_bundle import system_prompt, user_prompt
from robots.yam.tasks import DIAGNOSTIC_TASKS, TASK_INSTRUCTIONS
from rpent.dashboard.events import DashboardEventSink
from rpent.dashboard.spec import DashboardSpec
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.robots.runtime import try_wait_server
from rpent.utils.config import get_memory_dir, get_repo_root
from rpent.utils.rpc import make_rpc_client

if TYPE_CHECKING:
    from rpent.utils.daemon import ProcessDaemon


YAM_DASHBOARD_SPEC: DashboardSpec = {
    "external_env": True,
    "operator_commands": {
        **{
            command: "Use the operator terminal: python -m robots.yam.operator_control "
            f"--config <site.json> --endpoint <env> --command {command} "
            "--episode-id <current-id> --note <evidence>"
            for command in ("/done", "/success", "/failure", "/abort")
        },
        "/continue": None,
    },
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <task_name> <seed>",
        "fields": (
            {"name": "task_name", "choices": tuple(TASK_INSTRUCTIONS)},
            {"name": "seed", "kind": "integer", "minimum": 0},
        ),
        "display": "{task_name} / seed {seed}",
        "output_slug": "{task_name}_s{seed}",
    },
    "runtime_components": (
        {"name": "env", "label": "ENV", "scope": "unique"},
        {"name": "vla", "label": "VLA", "scope": "shared"},
    ),
    "primitives": (
        "status",
        "pi05_act",
        "move_to",
        "rotate_wrist",
        "set_gripper",
        "release",
    ),
}


def get_robot_spec() -> RobotSpec:
    return RobotSpec(
        name="yam",
        dashboard=YAM_DASHBOARD_SPEC,
        supports_exploration=True,
        is_real_robot=True,
        finalize_run=finalize_run,
        run_diagnostic=_run_diagnostic,
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
    mode: str = "evaluation",
    attempts_per_session: int = 0,
    state_output_dir: Path | str | None = None,
):
    from robots.yam.toolkit import YamToolkit

    explore = mode == "exploration"
    memory = MemoryManager(
        root=config.prompt_vars.get("memory_dir") or get_memory_dir("yam"),
        memory_access="inbox_write" if explore else "read_only",
        inbox_cell_tag=config.recipe_tag if explore else None,
    )
    return YamToolkit(
        primitives_kwargs=primitives_kwargs,
        dashboard_events=dashboard_events,
        memory=memory,
        mode=mode,
        attempts_per_session=attempts_per_session,
        state_output_dir=state_output_dir,
        run_output_dir=config.output_dir,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    parser.set_defaults(memory_profile="local")
    required = not use_dashboard
    parser.add_argument(
        "--task-name",
        required=required,
        default=None,
    )
    parser.add_argument(
        "--task-id",
        type=int,
        choices=tuple(DIAGNOSTIC_TASKS),
        help="103 manual_primitive_test; 104 vla_deployment_test (no planner)",
    )
    parser.add_argument("--task-language", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--explore-attempts-per-session", type=int, default=5)
    parser.add_argument("--explore-sessions", type=int, default=1)
    parser.add_argument(
        "--auto-merge-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--env-endpoint",
        required=required,
        help="YAM env_server endpoint. Use socket://host:port for pickle TCP.",
    )
    parser.add_argument(
        "--vla-endpoint",
        help="Optional trained Pi0.5 YAM endpoint; omit for primitives-only control.",
    )
    parser.add_argument(
        "--without-vla",
        action="store_true",
        help="Run primitives-only wiring without starting or connecting a VLA server.",
    )


def _run_diagnostic(args: argparse.Namespace) -> int | None:
    if args.task_id not in DIAGNOSTIC_TASKS:
        return None
    if args.dashboard or args.explore:
        raise ValueError(
            "YAM diagnostic tasks are operator consoles; omit --dashboard and --explore"
        )
    from robots.yam.manual import run_session

    return run_session(args)


def _parse_config(args: argparse.Namespace) -> RunConfig:
    if not args.task_name:
        raise ValueError("--task-name is required")
    explore = bool(getattr(args, "explore", False))
    memory_profile = getattr(args, "memory_profile", None) or "local"
    args.memory_profile = memory_profile
    memory_dir_arg = getattr(args, "memory_dir", None)
    memory_dir = (
        Path(memory_dir_arg).expanduser().resolve()
        if memory_dir_arg
        else get_memory_dir("yam")
    )
    output_dir = getattr(args, "output_dir", None)
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S.%f")
        output_dir = (
            get_repo_root() / "logs" / f"{timestamp}_yam_{args.task_name}_s{args.seed}"
        )
    recipe_tag = f"yam_{args.task_name}_s{args.seed}_{uuid4().hex[:12]}"
    if getattr(args, "dashboard", False) and args.task_language is not None:
        raise ValueError(
            "Dashboard A/B instructions are selected by task_name; omit --task-language"
        )
    instruction = args.task_language or TASK_INSTRUCTIONS.get(
        args.task_name, args.task_name.replace("_", " ")
    )
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=Path(output_dir),
        prompt_vars={
            "recipe_tag": recipe_tag,
            "task_name": args.task_name,
            "seed": args.seed,
            "instruction": instruction,
            "mode": "explore" if explore else "eval",
            "vla_enabled": bool(args.vla_endpoint and not args.without_vla),
            "memory_profile": memory_profile,
            "memory_dir": str(memory_dir),
            "memory_inbox": str(memory_dir / "_internal" / "inbox" / recipe_tag),
            "session_number": 1,
            "session_max": max(1, int(getattr(args, "explore_sessions", 1) or 1)),
        },
        task_desc={
            "env": "yam",
            "task_name": args.task_name,
            "requested_seed": args.seed,
            "instruction": instruction,
            "policy_name": MODEL_SPEC.policy_name,
            "action_layout": MODEL_SPEC.action_layout,
            "camera_order": list(MODEL_SPEC.camera_order),
        },
    )


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list["ProcessDaemon"], dict[str, Any]]:
    del output_dir
    available = {"env", "vla"}
    selected = (
        ({"env", "vla"} if args.vla_endpoint else {"env"})
        if components is None
        else set(components)
    )
    if getattr(args, "without_vla", False):
        selected = set(selected) - {"vla"}
    unknown = selected.difference(available)
    if unknown:
        raise ValueError(f"unknown YAM runtime components: {sorted(unknown)}")
    if "env" in selected and not args.env_endpoint:
        raise ValueError(
            "--env-endpoint is required for YAM; start env_server on the control machine first"
        )
    if "vla" in selected and not args.vla_endpoint:
        raise ValueError("--vla-endpoint or --without-vla is required for YAM")

    owned_daemons: dict[str, ProcessDaemon] = {}
    primitives_kwargs: dict[str, Any] = {}
    if "env" in selected:
        env_rpc = make_rpc_client(args.env_endpoint)
        primitives_kwargs.update(
            try_wait_server(
                owned_daemons,
                dashboard_events,
                "env",
                env_rpc,
                None,
                5.0 if getattr(args, "dashboard", False) else 300.0,
                post_fn=lambda: _build_env_runtime_kwargs(args, env_rpc),
            )
        )
    if "vla" in selected:
        vla_rpc = make_rpc_client(args.vla_endpoint)
        primitives_kwargs.update(
            try_wait_server(
                owned_daemons,
                dashboard_events,
                "vla",
                vla_rpc,
                None,
                300.0,
                post_fn=lambda: _build_vla_runtime_kwargs(vla_rpc),
            )
        )
    return list(owned_daemons.values()), primitives_kwargs


def _build_env_runtime_kwargs(args: argparse.Namespace, env_rpc: Any) -> dict[str, Any]:
    from robots.yam.env_client import YamEnvClient

    if getattr(args, "dashboard", False):
        if env_rpc.call("env.is_started", timeout_s=5) is not True:
            raise RuntimeError(
                "YAM is not started. In the operator terminal, use the selected "
                "site config and operator_control --event status to initialize "
                "hardware, then reset_pose and ready before submitting a task."
            )
        language = env_rpc.call("env.get_task_language", timeout_s=5)
        if language != TASK_INSTRUCTIONS[args.task_name]:
            raise ValueError(
                f"YAM task language mismatch. Stop the current run, remove held "
                f"objects, home/shutdown, then start env_server with "
                f"the site config for {args.task_name!r}."
            )

    return {
        "env": YamEnvClient(
            env_rpc,
            expected_meta=env_runtime_contract(
                task_name=args.task_name,
                seed=int(args.seed),
                max_episode_steps=int(args.max_episode_steps),
            ),
        ),
    }


def _build_vla_runtime_kwargs(vla_rpc: Any) -> dict[str, Any]:
    from rpent.robots.components.vla_client_base import BaseVLAClient

    expected_meta = vla_runtime_contract()
    actual_meta = vla_rpc.call("healthz", timeout_s=30.0)
    if actual_meta != expected_meta:
        raise RuntimeError(
            "YAM VLA metadata mismatch: "
            f"expected={expected_meta!r} actual={actual_meta!r}. "
            "Connect to the Pi0.5 YAM qpos14 VLA server."
        )
    return {"model": BaseVLAClient(vla_rpc)}
