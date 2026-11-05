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

"""Action-model selection and deployment shared by VLA and WAM integrations."""

import argparse
import importlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from rpent.utils.config import get_repo_root, resolve_cuda_device
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.rpc import RpcClient, make_rpc_client

WAM_BACKENDS = {
    "cosmos-policy": "cosmos_policy",
    "cosmos3": "cosmos3",
    "fast-wam": "fast_wam",
}
BACKEND_ALIASES = {"cosmos": "cosmos-policy", "fast_wam": "fast-wam", "rlinf": "pi05"}
PLATFORM_VLA_BACKENDS = {
    "libero": ("pi05",),
    "robotwin": ("lingbot",),
    "robocasa": ("rldx",),
    "robodojo": ("pi05", "xpolicylab"),
    "dual_franka": ("pi05",),
    "franka": ("pi05",),
    "yam": ("pi05",),
}
REMOTE_ONLY_PLATFORMS = {"franka", "yam"}


def _runtime_config(backend: str) -> ModuleType:
    return importlib.import_module(
        f"rpent.robots.components.{WAM_BACKENDS[backend]}.runtime_config"
    )


def _supported_backends(platform: str) -> tuple[str, ...]:
    return PLATFORM_VLA_BACKENDS[platform] + tuple(
        name for name in WAM_BACKENDS if platform in _runtime_config(name).ADAPTERS
    )


def _show_model_options(parser: argparse.ArgumentParser, backend: str) -> None:
    for group in parser._action_groups:
        if not hasattr(group, "_am_backend"):
            continue
        visible = group._am_backend == backend
        group.description = group._am_description if visible else None
        for action in group._group_actions:
            action.help = action._am_help if visible else argparse.SUPPRESS


class _BackendAction(argparse.Action):
    """Select one model and show its options in subsequent --help output."""

    def __call__(self, parser, namespace, values, option_string=None):
        backend = BACKEND_ALIASES.get(values, values)
        setattr(namespace, self.dest, backend)
        _show_model_options(parser, backend)


def add_am_args(parser: argparse.ArgumentParser, *, platform: str) -> None:
    """Register supported backends, common deployment fields and model options."""
    backends = _supported_backends(platform)
    selection = parser.add_argument_group("Action model")
    selection.add_argument(
        "--am-backend",
        choices=backends,
        default=backends[0],
        type=lambda value: BACKEND_ALIASES.get(value, value),
        action=_BackendAction,
        help=f"Action model (default: {backends[0]}). Select before --help to see model options.",
    )
    for alias in ("--vla-backend", "--wam-backend") + (
        ("--policy-backend",) if platform == "robodojo" else ()
    ):
        selection.add_argument(
            alias,
            dest="am_backend",
            choices=backends,
            type=lambda value: BACKEND_ALIASES.get(value, value),
            action=_BackendAction,
            default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )
    connection = parser.add_argument_group(
        "Action-model connection (repeated tasks)",
        "Reuse an existing worker: --am-backend <backend> --am-endpoint <URL>. "
        "Configure its checkpoint and devices when deploying that worker.",
    )
    connection.add_argument(
        "--am-endpoint", metavar="URL", help="Existing model service"
    )
    for alias in ("--vla-endpoint", "--wam-endpoint"):
        connection.add_argument(
            alias, dest="am_endpoint", default=argparse.SUPPRESS, help=argparse.SUPPRESS
        )
    deployment = parser.add_argument_group(
        "Action-model local deployment (per run)",
        "Start and stop a model worker for this run; excludes --am-endpoint."
        if platform not in REMOTE_ONLY_PLATFORMS
        else "This platform connects to externally deployed model services.",
    )
    fields = {
        "checkpoint": (
            "MODEL",
            "Model checkpoint or service model identity, as required by the backend",
        ),
        "python": (
            "PYTHON",
            "Model environment Python; VLA defaults to the current interpreter",
        ),
        "root": (
            "DIR",
            "Model working directory; defaults to the RPent checkout unless required by the model",
        ),
        "cuda-device": (
            "DEVICE",
            "CUDA_VISIBLE_DEVICES for the local action model; may share GPUs with other services; defaults to --cuda-device, then inherits the environment",
        ),
    }
    aliases = {
        "checkpoint": ("--wam-checkpoint", "--vla-model-path"),
        "python": ("--wam-python", "--pi05-python"),
        "root": ("--wam-root",),
        "cuda-device": ("--wam-cuda-device", "--vla-cuda-device"),
    }
    for field, (metavar, help_text) in fields.items():
        deployment.add_argument(
            f"--am-{field}",
            metavar=metavar,
            help=help_text
            if platform not in REMOTE_ONLY_PLATFORMS
            else argparse.SUPPRESS,
        )
        for alias in aliases[field]:
            deployment.add_argument(
                alias,
                dest="am_" + field.replace("-", "_"),
                default=argparse.SUPPRESS,
                help=argparse.SUPPRESS,
            )
    for backend in backends:
        before = len(parser._action_groups)
        if backend in WAM_BACKENDS:
            _runtime_config(backend).add_arguments(parser)
        elif backend == "lingbot":
            group = parser.add_argument_group("LingBot local options")
            group.add_argument(
                "--lingbot-robot-config",
                help="FeatureTransform robot YAML; defaults to LINGBOT_ROBOT_CONFIG or the checkpoint config",
            )
        elif backend == "pi05" and platform not in REMOTE_ONLY_PLATFORMS:
            group = parser.add_argument_group("Pi0.5 local options")
            group.add_argument(
                "--pi05-repo-id",
                help="Dataset normalization statistics ID; dual-Franka defaults to DUAL_FRANKA_REPO_ID",
            )
            group.add_argument(
                "--pi05-norm-stats-path",
                help="Normalization statistics directory; defaults to PI05_NORM_STATS_PATH",
            )
            group.add_argument(
                "--vla-repo-id",
                dest="pi05_repo_id",
                default=argparse.SUPPRESS,
                help=argparse.SUPPRESS,
            )
        for group in parser._action_groups[before:]:
            group._am_backend = backend
            group._am_description = group.description
            for action in group._group_actions:
                action._am_help = action.help
    _show_model_options(parser, backends[0])
    parser.epilog = (parser.epilog or "") + (
        "\n\nAction-model deployment and task examples:\n"
        "https://rpent.readthedocs.io/en/latest/rst_source/development/add_action_model.html#action-model-deployment"
    )


@dataclass(frozen=True)
class AMConfig:
    """Resolved model deployment, independent of the platform's transport."""

    backend: str
    platform: str
    endpoint: str | None
    checkpoint: str | None
    python: str
    root: Path
    cuda_device: str | None
    model_arguments: tuple[str, ...] = ()

    @property
    def is_wam(self) -> bool:
        return self.backend in WAM_BACKENDS

    def spawn_worker(
        self,
        command: list[str],
        output_dir: Path,
        *,
        name: str,
        env_overrides: dict[str, str] | None = None,
    ) -> ProcessDaemon:
        """Start Python worker arguments with this deployment's cwd and devices."""
        overrides = {
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(get_repo_root()), os.getenv("PYTHONPATH")))
            ),
            **(env_overrides or {}),
        }
        if self.cuda_device is not None:
            overrides["CUDA_VISIBLE_DEVICES"] = self.cuda_device
        daemon = ProcessDaemon(
            name=name,
            cmd=[self.python, *command],
            cwd=str(self.root),
            env_overrides=overrides,
            log_path=str(output_dir / f"{name}.log"),
        )
        daemon.start()
        return daemon


def select_am(args: argparse.Namespace, platform: str) -> AMConfig:
    """Resolve the chosen model and reject inapplicable explicit launch options."""
    backend = getattr(args, "am_backend", None) or PLATFORM_VLA_BACKENDS[platform][0]
    backend = BACKEND_ALIASES.get(backend, backend)
    if backend not in _supported_backends(platform):
        raise ValueError(
            f"{backend} is not supported on {platform}; choose from {_supported_backends(platform)}"
        )
    endpoint = getattr(args, "am_endpoint", None)
    explicit = {
        field: getattr(args, "am_" + field, None)
        for field in ("checkpoint", "python", "root", "cuda_device")
    }
    model_fields = {
        **{name: _runtime_config(name).OPTION_FIELDS for name in WAM_BACKENDS},
        "lingbot": ("lingbot_robot_config",),
        "pi05": ("pi05_repo_id", "pi05_norm_stats_path"),
    }
    for name, fields in model_fields.items():
        for field in fields:
            if (
                getattr(args, field, None) is not None
                and getattr(args, field) is not False
            ):
                if name != backend:
                    raise ValueError(
                        f"--{field.replace('_', '-')} requires --am-backend {name}"
                    )
                if endpoint:
                    raise ValueError(
                        f"--{field.replace('_', '-')} configures local deployment; remove --am-endpoint"
                    )
    if endpoint or platform in REMOTE_ONLY_PLATFORMS:
        for field, value in explicit.items():
            if value is not None:
                raise ValueError(
                    f"--am-{field.replace('_', '-')} configures local deployment; this run uses an external model service"
                )
    checkpoint, python, root = (explicit[k] for k in ("checkpoint", "python", "root"))
    model_arguments: tuple[str, ...] = ()
    if not endpoint and platform not in REMOTE_ONLY_PLATFORMS:
        if backend in WAM_BACKENDS:
            if not checkpoint:
                raise ValueError(
                    "--am-backend requires --am-endpoint or --am-checkpoint"
                )
            # Model defaults are resolved on a copy, so they never become explicit
            # options when a Dashboard reuses the session namespace.
            model_args = argparse.Namespace(**vars(args))
            _runtime_config(backend).validate_options(model_args)
            checkpoint = model_args.am_checkpoint
            python, root = model_args.am_python, model_args.am_root
            if not python:
                raise ValueError(
                    "--am-python must reference the provisioned model environment"
                )
            model_arguments = tuple(
                _runtime_config(backend).worker_arguments(model_args)
            )
        elif backend == "pi05":
            checkpoint = checkpoint or os.getenv("PI05_CHECKPOINT_PATH")
            options = []
            repo_id = getattr(args, "pi05_repo_id", None) or (
                os.getenv("DUAL_FRANKA_REPO_ID") if platform == "dual_franka" else None
            )
            norm_stats = getattr(args, "pi05_norm_stats_path", None) or os.getenv(
                "PI05_NORM_STATS_PATH"
            )
            if repo_id:
                options += ["--repo-id", repo_id]
            if norm_stats:
                options += [
                    "--norm-stats-path",
                    str(Path(norm_stats).expanduser().absolute()),
                ]
            model_arguments = tuple(options)
        elif backend == "lingbot":
            checkpoint = checkpoint or os.getenv("LINGBOT_MODEL_PATH")
        elif backend == "xpolicylab":
            checkpoint = checkpoint or "RoboDojo-sim-arx_x5-joint-0"
    python = python or sys.executable
    executable = shutil.which(str(Path(python).expanduser()))
    if not executable:
        raise ValueError("--am-python must reference a Python executable")
    root_path = Path(root).expanduser().resolve() if root else get_repo_root()
    if not root_path.is_dir():
        raise ValueError("--am-root must reference a directory")
    device = resolve_cuda_device(args, "am")
    return AMConfig(
        backend=backend,
        platform=platform,
        endpoint=endpoint,
        checkpoint=str(Path(checkpoint).expanduser().absolute())
        if checkpoint and backend not in WAM_BACKENDS and backend != "xpolicylab"
        else checkpoint,
        python=str(Path(executable).absolute()),
        root=root_path,
        cuda_device=str(device) if device is not None else None,
        model_arguments=model_arguments,
    )


def start_wam_service(
    config: AMConfig, output_dir: Path
) -> tuple[ProcessDaemon | None, RpcClient]:
    """Connect a WAM endpoint or launch its model-owned RPC entry point."""
    runtime_config = _runtime_config(config.backend)
    if config.endpoint:
        return None, make_rpc_client(
            config.endpoint, enable_sessions=runtime_config.USES_SESSIONS
        )
    port = pick_free_port()
    command = [
        "-m",
        f"rpent.robots.components.{WAM_BACKENDS[config.backend]}.server",
        "--platform",
        config.platform,
        "--checkpoint",
        config.checkpoint,
        *config.model_arguments,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--parent-watch",
    ]
    daemon = config.spawn_worker(command, output_dir, name="wam_server")
    return daemon, make_rpc_client(
        f"http://127.0.0.1:{port}", enable_sessions=runtime_config.USES_SESSIONS
    )
