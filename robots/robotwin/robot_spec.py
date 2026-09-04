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

"""RoboTwin robot extension — runtime contracts and runtime hooks."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from robots.robotwin.prompt_bundle import system_prompt, user_prompt
from rpent.dashboard.events import DashboardEventSink, RuntimeStatusEvent
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.robots.runtime import (
    stop_owned_daemons,
    try_spawn_server,
    try_wait_server,
)
from rpent.utils.config import get_memory_dir, get_repo_root

if TYPE_CHECKING:
    from rpent.utils.daemon import ProcessDaemon

# Native RoboTwin task YAMLs exposed to the dashboard, the CLI, and the env
# server -- defined once so every consumer shows the same choices.
ROBOTWIN_TASK_CONFIGS = (
    "demo_clean",
    "demo_randomized",
)

#: Env-side camera names exposed by the RoboTwin EnvServer, in fixed order.
#: Shared across the env client, primitives, and toolkit.
ROBOTWIN_CAMERA_NAMES = (
    "head",
    "left_wrist",
    "right_wrist",
)

#: Supported native action representations for the RoboTwin agent runtime.
RoboTwinActionType = Literal["qpos", "ee"]

#: Episode-status keys the RoboTwin env client requires in every status mapping.
ROBOTWIN_STATUS_KEYS = (
    "eval_success",
    "take_action_cnt",
    "step_lim",
    "actual_seed",
)

#: Per-call RPC read timeout (seconds) for idempotent env queries.
ROBOTWIN_READ_TIMEOUT_S = 120.0

#: Per-call RPC timeout (seconds) for env calls that mutate episode state.
ROBOTWIN_STATE_CHANGE_TIMEOUT_S = 600.0


@dataclass(frozen=True)
class RoboTwinModelSpec:
    """Values required by the LingBot EEF runtime."""

    policy_name: str
    robot_config_relpath: str
    norm_stats: str
    qwen_base: str
    camera_order: tuple[str, ...]
    state_layout: str
    action_layout: str
    use_length: int


MODEL_SPEC = RoboTwinModelSpec(
    policy_name="robotwin_eef",
    robot_config_relpath="configs/robot_configs/robotwin_eef.yaml",
    norm_stats="norm_stats/robotwin_eef.json",
    qwen_base="qwen_base",
    camera_order=("cam_high", "cam_left_wrist", "cam_right_wrist"),
    state_layout="eef16",
    action_layout="eef16",
    use_length=50,
)


ROBOTWIN_DASHBOARD_SPEC = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <task_name> <seed>",
        "fields": (
            {"name": "task_name"},
            {"name": "seed", "kind": "integer", "minimum": 0},
        ),
        "display": "{task_name} / seed {seed}",
        "output_slug": "{task_name}_s{seed}",
    },
    "runtime_components": (
        {"name": "env", "label": "ENV", "scope": "unique"},
        {"name": "vla", "label": "VLA", "scope": "shared"},
    ),
    "frame_channels": (
        {"name": "camera", "label": "head camera"},
        {"name": "left_wrist", "label": "left wrist"},
        {"name": "right_wrist", "label": "right wrist"},
    ),
}


def env_runtime_contract(
    *,
    task_name: str,
    task_config: str,
    seed: int,
    max_episode_steps: int = 10000,
) -> dict[str, object]:
    """Return the identity required from a RoboTwin EnvServer."""
    return {
        "runtime": "rlinf_robotwin_env",
        "task_name": task_name,
        "task_config": task_config,
        "seed": int(seed),
        "seed_mode": "exact",
        "action_layouts": ["qpos14", MODEL_SPEC.action_layout],
        "execution": {
            "reset": True,
            "step": True,
            "chunk_step": True,
            "action_layouts": ["qpos14", MODEL_SPEC.action_layout],
            "chunk_step_all_frames": True,
            "step_limit": int(max_episode_steps),
        },
        "extensions": {
            "render_camera": {
                "camera_names": list(ROBOTWIN_CAMERA_NAMES),
                "metric_depth": True,
            },
            "get_camera_meta": True,
            "get_task_language": True,
            "plan_arm_path": True,
        },
    }


def vla_runtime_contract() -> dict[str, object]:
    """Return the identity required from a LingBot RoboTwin server.

    Only the hard contract fields that break inference input/output when
    mismatched are kept; descriptive behaviour fields live with the
    implementation (facade/transport), not in the runtime-equality check.
    """
    return {
        "runtime": "lingbotvla",
        "policy_name": MODEL_SPEC.policy_name,
        "camera_order": list(MODEL_SPEC.camera_order),
        "state_layout": MODEL_SPEC.state_layout,
        "action_layout": MODEL_SPEC.action_layout,
        "use_length": MODEL_SPEC.use_length,
    }


def get_robot_spec() -> RobotSpec:
    return RobotSpec(
        name="robotwin",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        dashboard=ROBOTWIN_DASHBOARD_SPEC,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
):
    """Return the RoboTwin toolkit for the current session."""
    from robots.robotwin.toolkit import RoboTwinToolkit

    memory = MemoryManager(
        root=config.prompt_vars.get("memory_dir") or get_memory_dir("robotwin"),
    )
    return RoboTwinToolkit(
        primitives_kwargs=primitives_kwargs,
        dashboard_events=dashboard_events,
        memory=memory,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    required = not use_dashboard
    parser.add_argument("--task-name", required=required)
    parser.add_argument(
        "--seed",
        type=int,
        required=required,
        help=(
            "Exact RoboTwin scene seed. For the standard demo_randomized "
            "evaluation, use a per-task seed from "
            "robots/robotwin/eval/demo_randomized.json."
        ),
    )
    parser.add_argument(
        "--task-config",
        choices=ROBOTWIN_TASK_CONFIGS,
        default="demo_randomized",
        help="Native RoboTwin task YAML.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=10000,
        help=(
            "Episode action budget for the RPent RoboTwin agent runtime. "
            "Overrides the native per-task eval step limit so long agent "
            "rollouts are not capped at the baseline budget."
        ),
    )
    parser.add_argument(
        "--robotwin-assets-path",
        default=os.environ.get("ROBOTWIN_ASSETS_PATH"),
        help=("Path to the RoboTwin asset snapshot. Defaults to ROBOTWIN_ASSETS_PATH."),
    )
    parser.add_argument("--env-endpoint", default=None)
    parser.add_argument("--vla-endpoint", default=None)
    parser.add_argument(
        "--vla-model-path",
        default=os.environ.get("LINGBOT_MODEL_PATH"),
        help=("LingBot checkpoint. Defaults to LINGBOT_MODEL_PATH when set."),
    )
    parser.add_argument(
        "--lingbot-robot-config",
        default=os.environ.get("LINGBOT_ROBOT_CONFIG"),
        help=(
            "Path to the LingBot FeatureTransform robot config. Defaults to "
            "the model snapshot's configs/robot_configs/robotwin_eef.yaml."
        ),
    )
    parser.add_argument("--cuda-device", default=None)
    parser.add_argument(
        "--env-cuda-device",
        default=None,
        help="CUDA_VISIBLE_DEVICES value for the RoboTwin EnvServer.",
    )
    parser.add_argument(
        "--vla-cuda-device",
        default=None,
        help="CUDA_VISIBLE_DEVICES value for the LingBot VLA server.",
    )


def _parse_config(args: argparse.Namespace) -> RunConfig:
    if not args.task_name:
        raise ValueError("--task-name is required")
    env_cuda_device, vla_cuda_device = _resolve_cuda_devices(args)
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = (
            get_repo_root()
            / "logs"
            / f"{timestamp}_robotwin_{args.task_name}_s{args.seed}"
        )
    output_dir = Path(output_dir)
    recipe_tag = f"robotwin_{args.task_name}_s{args.seed}"
    task_config = getattr(args, "task_config", "demo_randomized")
    initial_seed = int(args.seed)
    memory_dir = (
        Path(args.memory_dir).expanduser().resolve()
        if args.memory_dir
        else get_memory_dir("robotwin")
    )
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=output_dir,
        prompt_vars={
            "task_name": args.task_name,
            "seed": args.seed,
            "task_config": task_config,
            "instruction": "<native task_language from state_00>",
            "memory_dir": str(memory_dir),
            "reference_tag": f"{args.task_name}_s0",
        },
        task_desc={
            "env": "robotwin",
            "task_name": args.task_name,
            "requested_seed": args.seed,
            "initial_native_seed": initial_seed,
            "seed_mode": "exact",
            "task_config": task_config,
            "instruction": None,
            "policy_name": MODEL_SPEC.policy_name,
            "action_layout": MODEL_SPEC.action_layout,
            "env_cuda_device": env_cuda_device,
            "vla_cuda_device": vla_cuda_device,
        },
    )


def _wait_for_tcp(
    host: str,
    port: int,
    daemon: ProcessDaemon | None,
    timeout_s: float = 900.0,
) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        if daemon is not None and daemon.poll() is not None:
            raise RuntimeError(
                f"{daemon.name} exited before listening; inspect its log"
            )
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as error:
            last_error = error
            time.sleep(0.5)
    raise TimeoutError(f"LingBot server not ready: {last_error}")


def _parse_vla_endpoint(endpoint: str) -> tuple[str, int]:
    value = endpoint.split("://", 1)[-1]
    host, separator, port_text = value.rpartition(":")
    if not separator or not host or not port_text:
        raise ValueError("--vla-endpoint must be [ws://]host:port")
    return host, int(port_text)


def _require_directory(path: Path, option: str) -> None:
    if not path.is_dir():
        raise ValueError(f"{option} directory not found: {path}")


def _resolve_env_runtime_path(args: argparse.Namespace) -> Path | None:
    assets_path: Path | None = None
    if args.env_endpoint is None:
        configured_assets = getattr(args, "robotwin_assets_path", None)
        if not configured_assets:
            raise ValueError(
                "--robotwin-assets-path is required when launching the local "
                "env server; set ROBOTWIN_ASSETS_PATH or pass the option explicitly"
            )
        assets_path = Path(configured_assets).expanduser().resolve()
        from robotwin.assets import validate_root

        validate_root(assets_path)
    return assets_path


def _resolve_vla_runtime_path(args: argparse.Namespace) -> Path | None:
    model_path: Path | None = None
    if args.vla_endpoint is None:
        configured_model = getattr(args, "vla_model_path", None)
        if not configured_model:
            raise ValueError(
                "--vla-model-path is required when launching the local VLA "
                "server; set LINGBOT_MODEL_PATH or pass the option explicitly"
            )
        model_path = Path(configured_model).expanduser().resolve()
        _validate_local_model_files(model_path)
        _resolve_lingbot_robot_config(args, model_path)
    return model_path


def _validate_local_model_files(model_path: Path) -> None:
    required = [
        model_path / "config.json",
        model_path / "lingbotvla_cli.yaml",
        model_path / MODEL_SPEC.norm_stats,
        model_path / MODEL_SPEC.qwen_base,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError(f"LingBot model snapshot is incomplete: {missing}")


def _resolve_lingbot_robot_config(
    args: argparse.Namespace,
    model_path: Path,
) -> Path:
    configured = getattr(args, "lingbot_robot_config", None)
    path = (
        Path(configured).expanduser()
        if configured
        else model_path / MODEL_SPEC.robot_config_relpath
    ).resolve()
    if not path.is_file():
        raise ValueError(
            "LingBot robot config not found: "
            f"{path}. Pass --lingbot-robot-config or download the complete "
            "pinned model snapshot."
        )
    return path


def _resolve_cuda_devices(
    args: argparse.Namespace,
) -> tuple[str | None, str | None]:
    shared = getattr(args, "cuda_device", None)
    env_device = getattr(args, "env_cuda_device", None)
    vla_device = getattr(args, "vla_cuda_device", None)
    if shared is not None and (env_device is not None or vla_device is not None):
        raise ValueError(
            "--cuda-device cannot be combined with --env-cuda-device or "
            "--vla-cuda-device"
        )
    if shared is not None:
        value = str(shared)
        return value, value
    return (
        str(env_device) if env_device is not None else None,
        str(vla_device) if vla_device is not None else None,
    )


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list["ProcessDaemon"], dict[str, Any]]:
    """Initialize every RoboTwin component, or only ``components`` when given."""
    available = {"env", "vla"}
    selected = available if components is None else components
    unknown = selected.difference(available)
    if unknown:
        raise ValueError(f"unknown RoboTwin runtime components: {sorted(unknown)}")

    owned_daemons: dict[str, ProcessDaemon] = {}
    env_pending: tuple[ProcessDaemon | None, Any] | None = None
    vla_pending: tuple[ProcessDaemon | None, tuple[str, int]] | None = None

    if "env" in selected:
        env_pending = try_spawn_server(
            owned_daemons,
            dashboard_events,
            "env",
            lambda: _spawn_env_server(args, output_dir),
        )

    if "vla" in selected:
        dashboard_events.emit(RuntimeStatusEvent("vla", "starting"))
        try:
            vla_pending = _spawn_vla_server(args, output_dir)
            vla_daemon, _ = vla_pending
            if vla_daemon is not None:
                owned_daemons["vla"] = vla_daemon
        except Exception as exc:
            stop_owned_daemons(owned_daemons, dashboard_events)
            dashboard_events.emit(RuntimeStatusEvent("vla", "failed", error=exc))
            raise RuntimeError(f"[vla] spawn failed: {exc}") from exc

    primitives_kwargs: dict[str, Any] = {}

    if env_pending is not None:
        env_daemon, env_rpc = env_pending
        env_kwargs = try_wait_server(
            owned_daemons,
            dashboard_events,
            "env",
            env_rpc,
            env_daemon,
            900.0 if env_daemon is not None else 300.0,
            post_fn=lambda: _build_env_runtime_kwargs(args, env_rpc),
        )
        primitives_kwargs.update(env_kwargs)

    if vla_pending is not None:
        vla_daemon, endpoint = vla_pending
        host, port = endpoint
        try:
            _wait_for_tcp(
                host,
                port,
                vla_daemon,
                timeout_s=900.0 if vla_daemon is not None else 300.0,
            )
            vla_kwargs = _build_vla_runtime_kwargs(endpoint)
        except Exception as exc:
            stop_owned_daemons(owned_daemons, dashboard_events)
            dashboard_events.emit(RuntimeStatusEvent("vla", "failed", error=exc))
            raise RuntimeError(f"[vla] wait / client connect failed: {exc}") from exc
        dashboard_events.emit(RuntimeStatusEvent("vla", "ready"))
        primitives_kwargs.update(vla_kwargs)

    return list(owned_daemons.values()), primitives_kwargs


def _spawn_env_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple["ProcessDaemon | None", Any]:
    from rpent.utils.daemon import ProcessDaemon, pick_free_port

    env_cuda_device, _ = _resolve_cuda_devices(args)
    resolved_assets = _resolve_env_runtime_path(args)
    assets_path = str(resolved_assets) if resolved_assets else None
    initial_seed = int(args.seed)

    if args.env_endpoint is None:
        if assets_path is None:
            raise ValueError(
                "--robotwin-assets-path is required to launch the env server"
            )
        host, env_port = "127.0.0.1", pick_free_port()
        env_daemon = ProcessDaemon(
            "robotwin_env_server",
            [
                sys.executable,
                str(get_repo_root() / "robots" / "robotwin" / "env_server.py"),
                "--task-name",
                args.task_name,
                "--task-config",
                args.task_config,
                "--seed",
                str(initial_seed),
                "--max-episode-steps",
                str(args.max_episode_steps),
                "--assets-path",
                assets_path,
                "--transport",
                "http",
                "--host",
                host,
                "--port",
                str(env_port),
                "--parent-watch",
            ],
            env_overrides={
                "ROBOTWIN_ASSETS_PATH": assets_path,
                **(
                    {"CUDA_VISIBLE_DEVICES": env_cuda_device}
                    if env_cuda_device is not None
                    else {}
                ),
            },
            log_path=str(output_dir / "robotwin_env_server.log"),
        )
        from rpent.utils.rpc.http_rpc import HttpRpcClient

        env_rpc = HttpRpcClient(f"http://{host}:{env_port}")
        env_daemon.start()
        return env_daemon, env_rpc
    else:
        from rpent.utils.rpc import make_rpc_client

        return None, make_rpc_client(args.env_endpoint)


def _spawn_vla_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, tuple[str, int]]:
    from rpent.utils.daemon import ProcessDaemon, pick_free_port

    if args.vla_endpoint is not None:
        return None, _parse_vla_endpoint(args.vla_endpoint)

    _, vla_cuda_device = _resolve_cuda_devices(args)
    model_path = _resolve_vla_runtime_path(args)
    assert model_path is not None
    robot_config = _resolve_lingbot_robot_config(args, model_path)
    host, vla_port = "127.0.0.1", pick_free_port()
    norm_path = model_path / MODEL_SPEC.norm_stats
    qwen_path = model_path / MODEL_SPEC.qwen_base
    daemon = ProcessDaemon(
        "lingbot_vla_server",
        [
            sys.executable,
            str(get_repo_root() / "robots" / "robotwin" / "vla_server.py"),
            "--model-path",
            str(model_path),
            "--use-length",
            str(MODEL_SPEC.use_length),
            "--port",
            str(vla_port),
            "--norm-path",
            str(norm_path),
            "--lingbot-robot-config",
            str(robot_config),
            "--parent-watch",
        ],
        env_overrides={
            "QWEN25_PATH": str(qwen_path),
            **(
                {"CUDA_VISIBLE_DEVICES": vla_cuda_device}
                if vla_cuda_device is not None
                else {}
            ),
        },
        log_path=str(output_dir / "lingbot_vla_server.log"),
    )
    daemon.start()
    return daemon, (host, vla_port)


def _build_env_runtime_kwargs(
    args: argparse.Namespace,
    env_rpc: Any,
) -> dict[str, Any]:
    from robots.robotwin.env_client import RoboTwinEnvClient

    initial_seed = int(args.seed)

    return {
        "env": RoboTwinEnvClient(
            env_rpc,
            expected_meta=env_runtime_contract(
                task_name=args.task_name,
                task_config=args.task_config,
                seed=initial_seed,
                max_episode_steps=int(args.max_episode_steps),
            ),
        ),
        "seed": initial_seed,
        "seed_mode": "exact",
    }


def _build_vla_runtime_kwargs(endpoint: tuple[str, int]) -> dict[str, Any]:
    from robots.robotwin.vla_client import LingBotVLAClient

    host, port = endpoint
    model = LingBotVLAClient(host=host, port=port)
    model.validate_contract(vla_runtime_contract())
    return {"model": model}
