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

"""Shared action-model service startup and category-specific configuration."""

import argparse
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

from rpent.utils.config import get_repo_root, resolve_cuda_device
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.rpc import RpcClient, make_rpc_client

# --- Shared model GPU option registration ---


def add_am_device_arg(
    parser: argparse.ArgumentParser | argparse._ArgumentGroup,
    kind: Literal["vla", "wam"],
) -> None:
    """Register a model service's GPU option on the owning parser or group."""
    parser.add_argument(
        f"--{kind}-cuda-device",
        metavar="DEVICE",
        help=f"CUDA_VISIBLE_DEVICES for the local {kind.upper()} worker; "
        "may share GPUs with other services; "
        "defaults to --cuda-device, then inherits the environment. "
        "Configure external workers at their own startup.",
    )


# --- WAM backend selection and worker configuration ---

WAM_BACKENDS = {
    "cosmos-policy": "cosmos_policy",
    "fast-wam": "fast_wam",
}
WAM_ALIASES = {"cosmos": "cosmos-policy", "fast_wam": "fast-wam"}


def _runtime_config(backend: str) -> ModuleType:
    return importlib.import_module(
        f"rpent.robots.components.{WAM_BACKENDS[backend]}.runtime_config"
    )


def add_am_endpoint_arg(
    parser: argparse.ArgumentParser | argparse._ArgumentGroup,
) -> None:
    """Register one action-model endpoint, retaining the original VLA spelling."""
    parser.add_argument(
        "--am-endpoint", metavar="URL", help="Existing action-model worker endpoint"
    )
    parser.add_argument("--vla-endpoint", dest="am_endpoint", help=argparse.SUPPRESS)


def add_am_args(parser: argparse.ArgumentParser, *, default_backend: str) -> None:
    """Register shared options and lightweight model-owned launch options."""
    selection = parser.add_argument_group("Action model")
    selection.add_argument(
        "--am-backend",
        choices=(default_backend, *WAM_BACKENDS, *WAM_ALIASES),
        default=default_backend,
        help=f"Action model (default: {default_backend}); use --am-endpoint to reuse a worker",
    )
    connection = parser.add_argument_group(
        "Action-model connection (repeated tasks)",
        "Start a worker once, then reuse its endpoint across task runs.\n"
        "Configure model options at worker startup. RPent leaves it running.\n"
        "Append to your platform/task/planner command:\n"
        "  --am-backend <backend> --am-endpoint http://127.0.0.1:<port>",
    )
    add_am_endpoint_arg(connection)
    selection.add_argument(
        "--vla-backend",
        dest="am_backend",
        choices=(default_backend,),
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    deployment = parser.add_argument_group(
        "WAM local deployment (per run)",
        "RPent starts and stops the worker for this run. Requires a checkpoint\n"
        "and a provisioned model Python environment; excludes --am-endpoint.",
    )
    deployment.add_argument(
        "--wam-checkpoint",
        metavar="PATH",
        help="Local model checkpoint (required for local deployment)",
    )
    deployment.add_argument(
        "--wam-python",
        metavar="PYTHON",
        help="Model environment Python (required unless the model supplies a default)",
    )
    deployment.add_argument(
        "--wam-root",
        metavar="DIR",
        help="Worker working directory (default: RPent checkout, unless required by the model)",
    )
    add_am_device_arg(deployment, "wam")
    for backend in WAM_BACKENDS:
        _runtime_config(backend).add_arguments(parser)
    parser.epilog = (parser.epilog or "") + (
        "\n\nWAM deployment and task examples:\n"
        "https://rpent.readthedocs.io/en/latest/rst_source/development/wam_backends.html"
    )


@dataclass(frozen=True)
class WAMConfig:
    """Selected backend and platform, independent of robot toolkit behavior."""

    backend: str
    platform: str
    endpoint: str | None

    @property
    def rpc_backend(self) -> str:
        return WAM_BACKENDS[self.backend]

    def start_service(
        self, args: argparse.Namespace, output_dir: Path
    ) -> tuple[ProcessDaemon | None, RpcClient]:
        """Borrow an endpoint or launch the registered model in its own environment."""
        runtime_config = _runtime_config(self.backend)
        if self.endpoint:
            return None, make_rpc_client(
                self.endpoint, enable_sessions=runtime_config.USES_SESSIONS
            )
        port = pick_free_port()
        cmd = [
            str(Path(args.wam_python).expanduser().absolute()),
            "-m",
            f"rpent.robots.components.{self.rpc_backend}.server",
            "--platform",
            self.platform,
            "--checkpoint",
            str(Path(args.wam_checkpoint).expanduser().resolve()),
            *runtime_config.worker_arguments(args),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--parent-watch",
        ]
        overrides = {
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(get_repo_root()), os.getenv("PYTHONPATH")))
            )
        }
        device = resolve_cuda_device(args, "wam")
        if device is not None:
            overrides["CUDA_VISIBLE_DEVICES"] = str(device)
        daemon = ProcessDaemon(
            name="wam",
            cmd=cmd,
            cwd=str(Path(args.wam_root).expanduser().resolve())
            if args.wam_root
            else str(get_repo_root()),
            env_overrides=overrides,
            log_path=str(output_dir / "wam_server.log"),
        )
        daemon.start()
        return daemon, make_rpc_client(
            f"http://127.0.0.1:{port}",
            enable_sessions=runtime_config.USES_SESSIONS,
        )


def select_wam(args: argparse.Namespace, platform: str) -> WAMConfig | None:
    """Validate backend/platform support and exclusive local/remote configuration."""
    backend = getattr(args, "am_backend", None)
    endpoint = getattr(args, "am_endpoint", None)
    checkpoint = getattr(args, "wam_checkpoint", None)
    backend = WAM_ALIASES.get(backend, backend)
    for name in WAM_BACKENDS:
        if name != backend and any(
            getattr(args, field, None) for field in _runtime_config(name).OPTION_FIELDS
        ):
            raise ValueError(f"{name} worker options require --am-backend {name}")
    if backend not in WAM_BACKENDS:
        if checkpoint or any(
            getattr(args, field, None)
            for field in ("wam_python", "wam_root", "wam_cuda_device")
        ):
            raise ValueError("WAM deployment options require a WAM --am-backend")
        return None
    runtime_config = _runtime_config(backend)
    if platform not in runtime_config.ADAPTERS:
        raise ValueError(
            f"{backend} has no adapter for {platform}; supported: {tuple(runtime_config.ADAPTERS)}"
        )
    if bool(endpoint) == bool(checkpoint):
        raise ValueError(
            "--am-backend requires --am-endpoint or --wam-checkpoint, mutually exclusive"
        )
    if endpoint and any(
        getattr(args, field, None) for field in runtime_config.OPTION_FIELDS
    ):
        raise ValueError(
            "WAM worker options require --wam-checkpoint; configure external workers at their launch"
        )
    if checkpoint:
        runtime_config.validate_options(args)
        if not Path(checkpoint).expanduser().exists():
            raise ValueError("--wam-checkpoint must reference a local checkpoint")
        if not args.wam_python or not Path(args.wam_python).expanduser().is_file():
            raise ValueError(
                "--wam-python must reference the provisioned model environment"
            )
        if args.wam_root and not Path(args.wam_root).expanduser().is_dir():
            raise ValueError("--wam-root must reference a directory")
    return WAMConfig(backend, platform, endpoint or None)
