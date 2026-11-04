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

"""BEHAVIOR configuration and borrowed-component wiring.

The standard CLI connects to separately launched RPent Env/VLA facades. The
Env service owns the official policy bridge; the official evaluator owns the
simulation. Its WebSocket URL is distinct from the internal Env RPC endpoint.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from robots.behavior.prompt_bundle import system_prompt, user_prompt
from robots.behavior.tasks import TASK_LANGUAGE, TASK_NAMES
from rpent.dashboard.events import DashboardEventSink
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.utils.config import get_memory_dir

if TYPE_CHECKING:
    from rpent.tools.toolkit import Toolkit
    from rpent.utils.daemon import ProcessDaemon

_SETTING_KEYS = {
    "task_id",
    "task_language",
    "execution_horizon",
    "env_endpoint",
    "vla_endpoint",
    "sam3_endpoint",
    "control_mode",
    "max_chunks_per_call",
    "instruction_mode",
    "tool_allowlist",
    "env_rpc_timeout_s",
}
_DEFAULT_BEHAVIOR_TOOLS = {"view_env_state", "vla_act", "finish"}
_AGENT_BEHAVIOR_TOOLS = _DEFAULT_BEHAVIOR_TOOLS | {"pixel_to_robot_base"}
_BEHAVIOR_TOOLS = _AGENT_BEHAVIOR_TOOLS | {
    "read_robot_state",
    "move_eef_delta",
    "segment",
    "set_gripper",
    "release",
    "move_to",
    "rotate_wrist",
    "rotate_wrist_joint",
    "move_base",
}


def get_robot_spec() -> RobotSpec:
    """Return the standard descriptor without importing model or simulator code."""
    return RobotSpec(
        name="behavior",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        validate_args=_validate_args,
    )


def get_toolkit(
    *,
    runtime_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
) -> Toolkit:
    """Build the BEHAVIOR tool collection using RPent's artifact and memory owners."""
    from robots.behavior.toolkit import BehaviorToolkit
    from rpent.memory import MemoryManager

    memory = MemoryManager(
        root=config.prompt_vars["memory_dir"], memory_access="read_only"
    )
    return BehaviorToolkit(
        runtime_kwargs=runtime_kwargs,
        dashboard_events=dashboard_events,
        memory=memory,
        state_output_dir=config.output_dir,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    parser.set_defaults(memory_profile="local")
    parser.add_argument(
        "--behavior-config",
        type=Path,
        help="BEHAVIOR tool/runtime YAML; set task identity and instruction via CLI.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        choices=tuple(TASK_NAMES),
        metavar="0..99",
        help="Official BEHAVIOR 100-task ID.",
    )
    parser.add_argument(
        "--task-language",
        help=(
            "Task instruction for the selected VLA. Defaults to the official "
            "instruction text for --task-id when omitted."
        ),
    )
    parser.add_argument(
        "--execution-horizon", type=int, help="Native actions per VLA call."
    )
    parser.add_argument(
        "--env-endpoint", help="Borrowed RPent BEHAVIOR bridge RPC endpoint."
    )
    parser.add_argument(
        "--vla-endpoint", help="Borrowed RPent VLA facade RPC endpoint."
    )
    parser.add_argument(
        "--sam3-endpoint", help="Borrowed native RPent SAM3 RPC endpoint."
    )
    parser.add_argument(
        "--control-mode",
        choices=("scripted", "agent"),
        help="Agent mode returns control after each bounded VLA tool call.",
    )
    parser.add_argument(
        "--max-chunks-per-call", type=int, help="Maximum VLA chunks in one tool call."
    )
    parser.add_argument(
        "--instruction-mode",
        choices=("main", "override"),
        help="Use the full main-task instruction or allow policy-compatible overrides.",
    )
    parser.add_argument(
        "--tool-allowlist",
        nargs="+",
        help="BEHAVIOR tools to expose; view_env_state and finish are required.",
    )
    parser.add_argument(
        "--env-rpc-timeout-s",
        type=float,
        help="Env RPC timeout, longer than the bridge's total chunk timeout.",
    )


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    config_path = getattr(args, "behavior_config", None)
    if config_path is not None:
        try:
            with Path(config_path).open() as source:
                loaded = yaml.safe_load(source)
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(
                f"cannot read BEHAVIOR config {config_path}: {error}"
            ) from error
        if not isinstance(loaded, dict) or set(loaded) - _SETTING_KEYS:
            raise ValueError(
                "BEHAVIOR config must contain only documented task settings"
            )
        settings.update(loaded)
    for key in _SETTING_KEYS:
        override = getattr(args, key, None)
        if override is not None:
            settings[key] = override
    task_id = settings.get("task_id")
    if type(task_id) is not int or task_id not in TASK_NAMES:
        raise ValueError("task_id must be an official BEHAVIOR task ID from 0 to 99")
    language = settings.get("task_language")
    if language is None:
        language = TASK_LANGUAGE.get(task_id)
    if not isinstance(language, str) or not language.strip():
        raise ValueError("--task-language or task_language in the config is required")
    settings["task_language"] = language.strip()
    horizon = settings.get("execution_horizon")
    if horizon is not None and (type(horizon) is not int or horizon < 1):
        raise ValueError("execution_horizon must be a positive integer")
    control_mode = settings.setdefault("control_mode", "scripted")
    if not isinstance(control_mode, str) or control_mode not in {"scripted", "agent"}:
        raise ValueError("control_mode must be scripted or agent")
    instruction_mode = settings.setdefault(
        "instruction_mode", "main" if control_mode == "agent" else "override"
    )
    if not isinstance(instruction_mode, str) or instruction_mode not in {
        "main",
        "override",
    }:
        raise ValueError("instruction_mode must be main or override")
    max_chunks = settings.setdefault(
        "max_chunks_per_call", 1 if control_mode == "agent" else None
    )
    if max_chunks is not None and (type(max_chunks) is not int or max_chunks < 1):
        raise ValueError("max_chunks_per_call must be a positive integer")
    if control_mode == "agent" and max_chunks is None:
        raise ValueError("agent mode requires max_chunks_per_call")
    allowlist = settings.setdefault(
        "tool_allowlist",
        sorted(
            _AGENT_BEHAVIOR_TOOLS
            if control_mode == "agent"
            else _DEFAULT_BEHAVIOR_TOOLS
        ),
    )
    if (
        not isinstance(allowlist, list)
        or any(not isinstance(name, str) for name in allowlist)
        or len(set(allowlist)) != len(allowlist)
        or set(allowlist) - _BEHAVIOR_TOOLS
        or not {"finish", "view_env_state"}.issubset(allowlist)
    ):
        raise ValueError(
            "tool_allowlist must contain unique BEHAVIOR tool names, including "
            "view_env_state and finish"
        )
    timeout = settings.get("env_rpc_timeout_s")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("env_rpc_timeout_s must be a finite positive number")
    for key in ("env_endpoint", "vla_endpoint", "sam3_endpoint"):
        endpoint = settings.get(key)
        if endpoint is not None:
            from rpent.utils.rpc.client_utils import parse_endpoint

            if not isinstance(endpoint, str):
                raise ValueError(f"{key} must be an RPC endpoint string")
            protocol, _, port = parse_endpoint(endpoint)
            if protocol not in {"http", "socket"} or not 0 < port < 65536:
                raise ValueError(
                    f"{key} must be an internal RPent http/socket RPC endpoint; "
                    "the official evaluator WebSocket is a different interface"
                )
    return settings


def _validate_args(args: argparse.Namespace) -> None:
    settings = _settings(args)
    _validate_runtime_settings(args, settings, _required_components(settings))


def _required_components(settings: dict[str, Any]) -> set[str]:
    return {"env", "vla"} | (
        {"sam3"} if "segment" in settings["tool_allowlist"] else set()
    )


def _validate_runtime_settings(
    args: argparse.Namespace, settings: dict[str, Any], components: set[str]
) -> None:
    if getattr(args, "dashboard", False):
        raise ValueError("BEHAVIOR Dashboard lifecycle is not connected yet")
    if getattr(args, "memory_profile", "local") not in (None, "local"):
        raise ValueError("BEHAVIOR evaluation uses an explicit local memory only")
    if getattr(args, "memory_repo", None) is not None:
        raise ValueError(
            "BEHAVIOR evaluation does not synchronize a remote memory corpus"
        )
    if getattr(args, "output_dir", None) is None:
        raise ValueError(
            "--output-dir is required; keep experiment artifacts outside the repo"
        )
    required = ["execution_horizon"] + [
        f"{component}_endpoint" for component in sorted(components)
    ]
    missing = [key for key in required if settings.get(key) is None]
    if missing:
        raise ValueError(
            "BEHAVIOR runtime is not configured: "
            + ", ".join(missing)
            + ". Start the required RPent services first and configure "
            "their internal RPent endpoints."
        )


def _parse_config(args: argparse.Namespace) -> RunConfig:
    settings = _settings(args)
    if getattr(args, "output_dir", None) is None:
        raise ValueError("--output-dir is required")
    output_dir = Path(args.output_dir).expanduser()
    memory_dir = (
        Path(args.memory_dir).expanduser().resolve()
        if getattr(args, "memory_dir", None)
        else get_memory_dir("behavior")
    )
    task_id = settings["task_id"]
    recipe_tag = f"behavior_{TASK_NAMES[task_id]}"
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=output_dir,
        prompt_vars={
            "task_id": task_id,
            "task_name": TASK_NAMES[task_id],
            "task_language": settings["task_language"],
            "execution_horizon": settings.get("execution_horizon"),
            "control_mode": settings["control_mode"],
            "max_chunks_per_call": settings["max_chunks_per_call"],
            "instruction_mode": settings["instruction_mode"],
            "tool_allowlist": settings["tool_allowlist"],
            "memory_dir": str(memory_dir),
            "recipe_tag": recipe_tag,
        },
        task_desc={
            "env": "behavior",
            "task_name": TASK_NAMES[task_id],
            "runtime_kind": "internal_facades",
            **settings,
        },
    )


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    from robots.behavior.env_client import BehaviorEnvClient
    from robots.behavior.vla_client import BehaviorVLAClient
    from rpent.robots.runtime import try_wait_server
    from rpent.utils.rpc import make_rpc_client

    settings = _settings(args)
    selected = _required_components(settings) if components is None else components
    if selected - {"env", "vla", "sam3"}:
        raise ValueError(f"unsupported BEHAVIOR components: {sorted(selected)}")
    _validate_runtime_settings(args, settings, selected)
    runtime_kwargs: dict[str, Any] = {
        "task_id": settings["task_id"],
        "task_language": settings["task_language"],
        "execution_horizon": settings["execution_horizon"],
        "max_chunks_per_call": settings["max_chunks_per_call"],
        "instruction_mode": settings["instruction_mode"],
        "tool_allowlist": settings["tool_allowlist"],
    }
    for name in ("env", "vla"):
        if name not in selected:
            continue
        rpc = make_rpc_client(settings[f"{name}_endpoint"])
        factory = (
            (
                lambda rpc=rpc: BehaviorEnvClient(
                    rpc,
                    task_id=settings["task_id"],
                    timeout_s=settings.get("env_rpc_timeout_s"),
                )
            )
            if name == "env"
            else (lambda rpc=rpc: BehaviorVLAClient(rpc))
        )
        client = try_wait_server(
            {}, dashboard_events, name, rpc, None, 5.0, post_fn=factory
        )
        runtime_kwargs["env" if name == "env" else "model"] = client
    if "sam3" in selected:
        from rpent.robots.components.sam3_client import Sam3Client

        rpc = make_rpc_client(settings["sam3_endpoint"])
        runtime_kwargs["sam3_client"] = try_wait_server(
            {},
            dashboard_events,
            "sam3",
            rpc,
            None,
            5.0,
            post_fn=lambda: Sam3Client(rpc),
        )
    # Endpoints are borrowed. The CLI must not stop their owner on exit.
    return [], runtime_kwargs
