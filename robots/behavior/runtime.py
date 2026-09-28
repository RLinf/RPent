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

"""Standard RPent runtime hooks for BEHAVIOR."""

from __future__ import annotations

import argparse
import os
import re
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.behavior.policy_checkpoint import (
    POLICY_CHECKPOINT_ENV,
    SHARED_POLICY_CHECKPOINT_PATH,
    SHARED_POLICY_PROFILE_ID,
    write_policy_checkpoint_manifest,
)
from robots.behavior.schemas import (
    ACTION_DIM,
    DEFAULT_ACTION_CHUNK,
    behavior_tool_specs_for_task,
)
from robots.behavior.task_specs import (
    BehaviorTaskSpec,
    get_task_spec,
    get_task_spec_by_index,
)
from rpent.dashboard.events import DashboardEventSink
from rpent.robots.robot_spec import RunConfig
from rpent.robots.runtime import try_spawn_server, try_wait_server
from rpent.utils.config import get_memory_dir, get_repo_root, get_rlinf_repo_path
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.rpc import make_rpc_client
from rpent.utils.rpc.http_rpc import HttpRpcClient

if TYPE_CHECKING:
    from rpent.utils.rpc import RpcClient

BEHAVIOR_MODES = ("eval", "explore")
BEHAVIOR_COMPONENTS = {"env", "vla", "dino"}
DEFAULT_EVAL_COMPONENTS = {"env", "vla", "dino"}
DEFAULT_MAX_EPISODE_STEPS = 43_200
DEFAULT_PLANNER_TIMEOUT_S = 7_200
RLINF_ROOT_ENV = "RPENT_RLINF_ROOT"
BEHAVIOR_PYTHON_ENV = "RPENT_BEHAVIOR_PYTHON"


def _default_behavior_python(behavior_repo: Path) -> str:
    configured = os.environ.get(BEHAVIOR_PYTHON_ENV)
    if configured:
        return str(Path(configured).expanduser())
    return str(behavior_repo / ".venv-behavior" / "bin" / "python")


def _single_cuda_device(value: Any) -> str | None:
    if value in (None, ""):
        return None
    device = str(value)
    if re.fullmatch(r"[0-9]+", device) is None:
        raise ValueError("CUDA device must be a single physical GPU ordinal")
    return device


def _component_cuda_device(
    args: argparse.Namespace,
    component: str,
) -> str | None:
    if component == "env":
        specific = getattr(args, "behavior_env_cuda_device", None)
    elif component in {"vla", "dino"}:
        specific = getattr(args, "behavior_model_cuda_device", None)
    else:
        raise ValueError(f"unsupported CUDA component: {component}")
    return _single_cuda_device(
        specific if specific not in (None, "") else getattr(args, "cuda_device", None)
    )


def _task_from_args(args: argparse.Namespace) -> BehaviorTaskSpec:
    task_name = getattr(args, "task_name", None)
    task_value = getattr(args, "task", None)
    if task_name:
        spec = get_task_spec(str(task_name))
        if task_value is not None and str(task_value).strip().isdigit():
            by_index = get_task_spec_by_index(int(task_value))
            if by_index is not spec:
                raise ValueError(
                    f"BEHAVIOR task identity mismatch: {task_name!r} != {task_value!r}"
                )
        return spec
    if task_value is None:
        raise ValueError("--task-name is required")
    if isinstance(task_value, str) and not task_value.strip().isdigit():
        return get_task_spec(task_value.strip())
    return get_task_spec_by_index(int(task_value))


def _public_seed_from_args(args: argparse.Namespace) -> int:
    public_seed = getattr(args, "public_seed", None)
    seed = getattr(args, "seed", None)
    if public_seed is not None and seed is not None and int(public_seed) != int(seed):
        raise ValueError("--public-seed and --seed disagree")
    value = public_seed if public_seed is not None else seed
    if value is None:
        raise ValueError("--public-seed is required")
    if isinstance(value, bool):
        raise ValueError("public seed must be an integer")
    return int(value)


def add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    required = not use_dashboard
    parser.set_defaults(planner="codex", planner_timeout_s=DEFAULT_PLANNER_TIMEOUT_S)
    parser.add_argument(
        "--task-name",
        required=required,
        choices=("turning_on_radio", "picking_up_trash"),
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task name or task index alias; task-name is preferred.",
    )
    parser.add_argument("--public-seed", type=int, required=required)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Alias for --public-seed for dashboard and legacy launchers.",
    )
    parser.add_argument(
        "--auto-merge-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge exploration output into layered memory (default: enabled).",
    )
    parser.add_argument(
        "--explore-attempts-per-session",
        type=int,
        default=0,
        help="Unsupported for BEHAVIOR; use --explore-sessions instead.",
    )
    parser.add_argument(
        "--explore-sessions",
        type=int,
        default=1,
        help="Fresh one-attempt BEHAVIOR sessions per exploration run.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=DEFAULT_MAX_EPISODE_STEPS,
    )
    parser.add_argument("--env-endpoint", default=None)
    parser.add_argument("--vla-endpoint", default=None)
    parser.add_argument("--dino-endpoint", default=None)
    default_behavior_repo = get_rlinf_repo_path()
    parser.add_argument(
        "--behavior-repo",
        default=str(default_behavior_repo),
        help=(
            "Source checkout containing the RLinf BEHAVIOR integration. "
            f"Can also be set with {RLINF_ROOT_ENV}."
        ),
    )
    parser.add_argument(
        "--behavior-python",
        default=_default_behavior_python(default_behavior_repo),
        help=(
            "Python executable for the official BEHAVIOR/OmniGibson env process. "
            f"Can also be set with {BEHAVIOR_PYTHON_ENV}."
        ),
    )
    parser.add_argument(
        "--activity-instance-dir",
        default=None,
        help="Optional explicit official BEHAVIOR task-instance JSON directory.",
    )
    parser.add_argument(
        "--env-config-path",
        default=None,
        help="Optional explicit RLinf BEHAVIOR env YAML; the bundled adapter otherwise resolves the pinned template.",
    )
    parser.add_argument(
        "--policy-checkpoint",
        default=str(SHARED_POLICY_CHECKPOINT_PATH),
        help=(
            "Path to your Pi05-Behavior model checkpoint. "
            f"Can also be set with {POLICY_CHECKPOINT_ENV}."
        ),
    )
    parser.add_argument(
        "--cuda-device",
        default=None,
        help=(
            "Shared fallback physical GPU ordinal. Component-specific BEHAVIOR "
            "GPU flags take precedence."
        ),
    )
    parser.add_argument(
        "--behavior-env-cuda-device",
        default=None,
        help="Physical GPU ordinal exposed only to the BEHAVIOR env process.",
    )
    parser.add_argument(
        "--behavior-model-cuda-device",
        default=None,
        help="Physical GPU ordinal shared by the BEHAVIOR VLA and DINO processes.",
    )
    parser.add_argument("--dino-source-archive", default=None)
    parser.add_argument("--dino-weights", default=None)
    parser.add_argument("--dino-cache-dir", default=None)
    parser.add_argument("--vla-ready-timeout-s", type=float, default=900.0)


def parse_config(args: argparse.Namespace) -> RunConfig:
    mode = "explore" if getattr(args, "explore", False) else "eval"
    if getattr(args, "explore", False):
        if getattr(args, "env_endpoint", None) is not None:
            raise ValueError(
                "BEHAVIOR explore requires an owned env sidecar; omit --env-endpoint"
            )
        if getattr(args, "explore_attempts_per_session", 0) > 0:
            raise ValueError(
                "BEHAVIOR explore runs one attempt per session; use --explore-sessions"
            )
    spec = _task_from_args(args)
    public_seed = _public_seed_from_args(args)
    activity_instance_id = spec.instance_for_public_seed(public_seed, phase=mode)
    if getattr(args, "activity_instance_id", None) is not None:
        requested = int(args.activity_instance_id)
        if requested != activity_instance_id:
            raise ValueError(
                "activity_instance_id must match the task public-seed mapping: "
                f"expected {activity_instance_id}, got {requested}"
            )
    cuda_device = _single_cuda_device(getattr(args, "cuda_device", None))
    env_cuda_device = _component_cuda_device(args, "env")
    model_cuda_device = _component_cuda_device(args, "vla")
    if int(getattr(args, "max_episode_steps", 0) or 0) <= 0:
        raise ValueError("--max-episode-steps must be positive")

    args.task_name = spec.task_name
    args.task = spec.task_index
    args.public_seed = public_seed
    args.seed = public_seed
    args.behavior_phase = mode
    args.activity_definition_id = spec.activity_definition_id
    args.activity_instance_id = activity_instance_id
    args.scene_model = spec.scene_model
    args.cuda_device = cuda_device
    args.behavior_env_cuda_device = env_cuda_device
    args.behavior_model_cuda_device = model_cuda_device

    recipe_tag = spec.tag(public_seed)
    output_dir = getattr(args, "output_dir", None)
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output_dir = get_repo_root() / "logs" / f"{timestamp}_behavior_{recipe_tag}"
    output_dir = Path(output_dir).expanduser().resolve()
    requested_memory_profile = getattr(args, "memory_profile", None)
    memory_profile = requested_memory_profile or (
        "local" if mode == "explore" else "hf"
    )
    configured_memory_dir = getattr(args, "memory_dir", None)
    memory_dir = (
        Path(configured_memory_dir).expanduser().resolve()
        if configured_memory_dir
        else get_memory_dir("behavior").resolve()
    )
    args.memory_profile = memory_profile
    args.memory_dir = str(memory_dir)
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=output_dir,
        prompt_vars={
            "task_name": spec.task_name,
            "task": spec.task_index,
            "task_language": spec.task_language,
            "instruction": spec.task_language,
            "task_instruction": spec.task_language,
            "public_seed": public_seed,
            "activity_definition_id": spec.activity_definition_id,
            "activity_instance_id": activity_instance_id,
            "scene_model": spec.scene_model,
            "mode": mode,
            "behavior_phase": mode,
            "max_episode_steps": int(args.max_episode_steps),
            "wall_clock_seconds": int(getattr(args, "planner_timeout_s", 7200) or 7200),
            "public_capabilities": [
                item["name"] for item in behavior_tool_specs_for_task(spec)
            ],
            "memory_dir": str(memory_dir),
            "memory_profile": memory_profile,
            "memory_inbox": str(memory_dir / "_internal" / "inbox" / recipe_tag),
            "recipe_tag": recipe_tag,
            "output_dir": str(output_dir),
        },
        task_desc={
            "env": "behavior",
            "task_name": spec.task_name,
            "task": spec.task_index,
            "public_seed": public_seed,
            "activity_definition_id": spec.activity_definition_id,
            "activity_instance_id": activity_instance_id,
            "scene_model": spec.scene_model,
            "mapping_version": spec.mapping_version,
            "behavior_mode": mode,
            "policy_profile_id": SHARED_POLICY_PROFILE_ID,
            "action_dim": ACTION_DIM,
            "action_horizon": DEFAULT_ACTION_CHUNK,
            "cuda_device": cuda_device,
            "behavior_env_cuda_device": env_cuda_device,
            "behavior_model_cuda_device": model_cuda_device,
            "memory_profile": memory_profile,
            "memory_dir": str(memory_dir),
        },
    )


def env_runtime_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "runtime": "behavior_env",
        "task_name": args.task_name,
        "task": int(args.task),
        "task_language": get_task_spec(args.task_name).task_language,
        "activity_definition_id": int(args.activity_definition_id),
        "activity_instance_id": int(args.activity_instance_id),
        "public_seed": int(args.public_seed),
        "scene_model": str(args.scene_model),
        "max_episode_steps": int(args.max_episode_steps),
        "action_dim": ACTION_DIM,
        "action_horizon": DEFAULT_ACTION_CHUNK,
        "official_success_path": ["info", "done", "success"],
        "behavior_repo": str(Path(args.behavior_repo).expanduser().resolve()),
        "activity_instance_dir": (
            None
            if not getattr(args, "activity_instance_dir", None)
            else str(Path(args.activity_instance_dir).expanduser().resolve())
        ),
        "rlinf_env_config_path": (
            None
            if not getattr(args, "env_config_path", None)
            else str(Path(args.env_config_path).expanduser().resolve())
        ),
    }


def _behavior_python_path(value: str | Path) -> Path:
    """Return an absolute executable path without dereferencing venv symlinks."""

    return Path(value).expanduser().absolute()


def _behavior_subprocess_env(
    *,
    cuda_device: str | None = None,
    **overrides: str,
) -> dict[str, str]:
    return {
        **overrides,
        "RPENT_REPO_ROOT": str(get_repo_root()),
        **({"CUDA_VISIBLE_DEVICES": cuda_device} if cuda_device is not None else {}),
    }


def vla_runtime_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "runtime": "pi05_vla",
        "embodiment": "behavior",
        "config_name": "pi05_behavior",
        "action_dim": ACTION_DIM,
        "action_horizon": DEFAULT_ACTION_CHUNK,
        "policy_profile_id": SHARED_POLICY_PROFILE_ID,
        "checkpoint": str(Path(args.policy_checkpoint).expanduser()),
    }


def _wait_for_server_endpoint(
    daemon: ProcessDaemon,
    *,
    log_offset: int = 0,
    timeout_s: float = 1800.0,
) -> str:
    """Read this launch's bound RPC address, never a preselected free port.

    ProcessDaemon appends logs, so callers supply the pre-launch byte offset.
    Application readiness is still checked by the shared RPC health probe.
    """
    if daemon.log_path is None:
        raise ValueError("endpoint discovery requires a daemon log_path")
    deadline = time.monotonic() + timeout_s
    with Path(daemon.log_path).open("rb") as log:
        log.seek(log_offset)
        while time.monotonic() < deadline:
            exit_code = daemon.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"{daemon.name} exited with code {exit_code}; see {daemon.log_path}"
                )
            position = log.tell()
            line = log.readline()
            if not line.endswith(b"\n"):
                log.seek(position)
                time.sleep(0.1)
                continue
            match = re.fullmatch(
                rb"RPC server listening on (http://[^\s:]+:[1-9][0-9]*)\r?\n",
                line,
            )
            if match is not None:
                return match.group(1).decode("ascii")
    raise TimeoutError(
        f"{daemon.name} did not announce an RPC endpoint within {timeout_s}s; "
        f"see {daemon.log_path}"
    )


def _spawn_env_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, "RpcClient"]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.env_endpoint is not None:
        return None, make_rpc_client(args.env_endpoint)
    # Bind only after Ray/OmniGibson initialization; the OS chooses the port.
    host, port = "127.0.0.1", 0
    cuda_device = _component_cuda_device(args, "env")
    # Keep the virtualenv launcher path intact.  Resolving ``bin/python``
    # follows its symlink to the system interpreter and silently drops the
    # virtualenv's site-packages (OmniGibson, OpenPI, OmegaConf, ...).
    behavior_python = _behavior_python_path(args.behavior_python)
    if not behavior_python.is_file():
        raise RuntimeError(f"BEHAVIOR Python executable is missing: {behavior_python}")
    cmd = [
        str(behavior_python),
        str(get_repo_root() / "robots" / "behavior" / "env_server.py"),
        "--task-name",
        str(args.task_name),
        "--public-seed",
        str(args.public_seed),
        "--task-index",
        str(args.task),
        "--activity-definition-id",
        str(args.activity_definition_id),
        "--activity-instance-id",
        str(args.activity_instance_id),
        "--scene-model",
        str(args.scene_model),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--output-dir",
        str(output_dir),
        "--behavior-repo",
        str(Path(args.behavior_repo).expanduser().resolve()),
        "--host",
        host,
        "--port",
        str(port),
        "--parent-watch",
    ]
    if getattr(args, "activity_instance_dir", None):
        cmd.extend(
            [
                "--activity-instance-dir",
                str(Path(args.activity_instance_dir).expanduser().resolve()),
            ]
        )
    if getattr(args, "env_config_path", None):
        cmd.extend(
            [
                "--env-config-path",
                str(Path(args.env_config_path).expanduser().resolve()),
            ]
        )
    if cuda_device is not None:
        cmd.extend(["--cuda-device", cuda_device])
    daemon = ProcessDaemon(
        name="behavior_env_server",
        cmd=cmd,
        env_overrides=_behavior_subprocess_env(
            cuda_device=cuda_device,
            **{
                "ROBOT_PLATFORM": "BEHAVIOR",
                "OMNIGIBSON_HEADLESS": "1",
                # Ray otherwise clears CUDA_VISIBLE_DEVICES for the zero-GPU actor
                # that owns the single OmniGibson subprocess.
                "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
            },
        ),
        log_path=str(output_dir / "behavior_env_server.log"),
    )
    log_path = Path(daemon.log_path)
    log_offset = log_path.stat().st_size if log_path.exists() else 0
    daemon.start()
    try:
        endpoint = _wait_for_server_endpoint(daemon, log_offset=log_offset)
        return daemon, HttpRpcClient(endpoint)
    except BaseException:
        # try_spawn_server cannot own this daemon until this function returns.
        daemon.stop()
        raise


def _spawn_vla_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, "RpcClient"]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.vla_endpoint is not None:
        return None, make_rpc_client(args.vla_endpoint)
    host, port = "127.0.0.1", pick_free_port()
    cuda_device = _component_cuda_device(args, "vla")
    # See _spawn_env_server: do not dereference a virtualenv's python symlink.
    behavior_python = _behavior_python_path(args.behavior_python)
    if not behavior_python.is_file():
        raise RuntimeError(f"BEHAVIOR Python executable is missing: {behavior_python}")
    checkpoint_manifest = output_dir / "policy_checkpoint_manifest.json"
    write_policy_checkpoint_manifest(
        checkpoint_manifest,
        Path(args.policy_checkpoint).expanduser(),
    )
    cmd = [
        str(behavior_python),
        str(get_repo_root() / "robots" / "behavior" / "vla_server.py"),
        "--host",
        host,
        "--port",
        str(port),
        "--model-path",
        str(Path(args.policy_checkpoint).expanduser()),
        "--parent-watch",
    ]
    if cuda_device is not None:
        cmd.extend(["--cuda-device", cuda_device])
    daemon = ProcessDaemon(
        name="behavior_vla_server",
        cmd=cmd,
        env_overrides=_behavior_subprocess_env(cuda_device=cuda_device),
        log_path=str(output_dir / "behavior_vla_server.log"),
    )
    daemon.start()
    return daemon, make_rpc_client(f"http://{host}:{port}")


def _spawn_dino_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, "RpcClient"]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dino_endpoint is not None:
        return None, make_rpc_client(args.dino_endpoint)
    host, port = "127.0.0.1", pick_free_port()
    cuda_device = _component_cuda_device(args, "dino")
    behavior_python = _behavior_python_path(args.behavior_python)
    if not behavior_python.is_file():
        raise RuntimeError(f"BEHAVIOR Python executable is missing: {behavior_python}")
    cmd = [
        str(behavior_python),
        str(get_repo_root() / "robots" / "behavior" / "dino_v2" / "server.py"),
        "--host",
        host,
        "--port",
        str(port),
        "--parent-watch",
    ]
    if getattr(args, "dino_source_archive", None):
        cmd.extend(
            [
                "--source-archive",
                str(Path(args.dino_source_archive).expanduser().resolve()),
            ]
        )
    if getattr(args, "dino_weights", None):
        cmd.extend(["--weights", str(Path(args.dino_weights).expanduser().resolve())])
    if getattr(args, "dino_cache_dir", None):
        cmd.extend(
            [
                "--cache-dir",
                str(Path(args.dino_cache_dir).expanduser().resolve()),
            ]
        )
    if cuda_device is not None:
        cmd.extend(["--cuda-device", cuda_device])
    daemon = ProcessDaemon(
        name="behavior_dino_server",
        cmd=cmd,
        env_overrides=_behavior_subprocess_env(cuda_device=cuda_device),
        log_path=str(output_dir / "behavior_dino_server.log"),
    )
    daemon.start()
    return daemon, HttpRpcClient(f"http://{host}:{port}")


def _connect_env(
    args: argparse.Namespace,
    rpc: "RpcClient",
    output_dir: Path,
) -> dict[str, Any]:
    from robots.behavior.env_client import BehaviorEnvClient

    env = BehaviorEnvClient(rpc, expected_meta=env_runtime_contract(args))
    initial_observation, initial_info = env.reset()
    task_language = initial_observation.get("task_descriptions")
    if isinstance(task_language, (list, tuple)):
        task_language = next(
            (item for item in task_language if isinstance(item, str)), None
        )
    if task_language is not None and str(task_language).strip():
        expected = get_task_spec(args.task_name).task_language
        if str(task_language).strip() != expected:
            raise RuntimeError("environment task language does not match TaskSpec")
    return {
        "env": env,
        "task_name": args.task_name,
        "behavior_phase": args.behavior_phase,
        "public_seed": int(args.public_seed),
        "max_episode_steps": int(args.max_episode_steps),
        "action_horizon": DEFAULT_ACTION_CHUNK,
        "initial_observation": initial_observation,
        "initial_info": initial_info,
        "output_dir": Path(output_dir),
    }


def _connect_vla(
    args: argparse.Namespace, rpc: "RpcClient", output_dir: Path
) -> dict[str, Any]:
    from robots.behavior.vla_client import BehaviorPi05VLAClient

    return {
        "model": BehaviorPi05VLAClient(rpc, embodiment="behavior"),
    }


def _connect_dino(args: argparse.Namespace, rpc: "RpcClient") -> dict[str, Any]:
    from robots.behavior.dino_v2.client import BehaviorDinoClient

    client = BehaviorDinoClient(rpc, expected_meta={"runtime": "behavior_dino"})
    return {
        "dino_component": client,
    }


def init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Initialize requested BEHAVIOR components under the RobotSpec contract."""

    os.environ[RLINF_ROOT_ENV] = str(Path(args.behavior_repo).expanduser().resolve())
    selected = set(DEFAULT_EVAL_COMPONENTS if components is None else components)
    session_env = "env" in selected and getattr(args, "explore", False)
    if session_env:
        # Each toolkit owns a fresh ENV; the runner retains shared models.
        selected.discard("env")
    unknown = selected.difference(BEHAVIOR_COMPONENTS)
    if unknown:
        raise ValueError(f"unknown BEHAVIOR runtime components: {sorted(unknown)}")

    starters = {
        "env": lambda: _spawn_env_server(args, output_dir),
        "vla": lambda: _spawn_vla_server(args, output_dir),
        "dino": lambda: _spawn_dino_server(args, output_dir),
    }
    connectors = {
        "env": lambda rpc: _connect_env(args, rpc, output_dir),
        "vla": lambda rpc: _connect_vla(args, rpc, output_dir),
        "dino": lambda rpc: _connect_dino(args, rpc),
    }
    timeouts = {
        "env": 1800.0,
        "vla": float(getattr(args, "vla_ready_timeout_s", 900.0)),
        "dino": 600.0,
    }
    pending: dict[str, tuple[ProcessDaemon | None, RpcClient]] = {}
    owned_daemons: dict[str, ProcessDaemon] = {}
    for component, starter in starters.items():
        if component in selected:
            pending[component] = try_spawn_server(
                owned_daemons, dashboard_events, component, starter
            )
    runtime_kwargs: dict[str, Any] = {}
    if session_env:
        runtime_kwargs["env_args"] = argparse.Namespace(
            **{**vars(args), "explore": False}
        )
    for component, (daemon, rpc) in pending.items():
        runtime_kwargs.update(
            try_wait_server(
                owned_daemons,
                dashboard_events,
                component,
                rpc,
                daemon,
                timeouts[component] if daemon is not None else 120.0,
                post_fn=partial(connectors[component], rpc),
            )
        )
    return list(owned_daemons.values()), runtime_kwargs


__all__ = [
    "BEHAVIOR_COMPONENTS",
    "BEHAVIOR_MODES",
    "DEFAULT_EVAL_COMPONENTS",
    "DEFAULT_MAX_EPISODE_STEPS",
    "DEFAULT_PLANNER_TIMEOUT_S",
    "add_cli_args",
    "env_runtime_contract",
    "init_runtime",
    "parse_config",
    "vla_runtime_contract",
]
