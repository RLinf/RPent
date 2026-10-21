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

from rpent.utils.config import get_repo_root
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
        "inherits the environment when omitted. Configure external workers at their own startup.",
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


def add_wam_args(parser: argparse.ArgumentParser) -> None:
    """Register shared options and lightweight model-owned launch options."""
    selection = parser.add_argument_group("WAM backend")
    selection.add_argument(
        "--wam-backend",
        choices=tuple(WAM_BACKENDS) + tuple(WAM_ALIASES),
        help="Model backend; select either an existing endpoint or a local checkpoint",
    )
    connection = parser.add_argument_group(
        "WAM connection (repeated tasks)",
        "Start a worker once, then reuse its endpoint across task runs.\n"
        "Configure model options at worker startup. RPent leaves it running.\n"
        "Append to your platform/task/planner command:\n"
        "  --wam-backend <backend> --wam-endpoint http://127.0.0.1:<port>",
    )
    connection.add_argument(
        "--wam-endpoint",
        metavar="URL",
        help="Existing HTTP/socket worker; excludes --wam-checkpoint",
    )
    deployment = parser.add_argument_group(
        "WAM local deployment (per run)",
        "RPent starts and stops the worker for this run. Requires a checkpoint\n"
        "and a provisioned model Python environment; excludes --wam-endpoint.",
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
        if args.wam_cuda_device is not None:
            overrides["CUDA_VISIBLE_DEVICES"] = str(args.wam_cuda_device)
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
    backend = getattr(args, "wam_backend", None)
    endpoint = getattr(args, "wam_endpoint", None)
    checkpoint = getattr(args, "wam_checkpoint", None)
    backend = WAM_ALIASES.get(backend, backend)
    for name in WAM_BACKENDS:
        if name != backend and any(
            getattr(args, field, None) for field in _runtime_config(name).OPTION_FIELDS
        ):
            raise ValueError(f"{name} worker options require --wam-backend {name}")
    if backend is None:
        if endpoint or checkpoint:
            raise ValueError("--wam-endpoint/--wam-checkpoint requires --wam-backend")
        return None
    if backend not in WAM_BACKENDS:
        raise ValueError(f"Unsupported WAM backend: {backend!r}")
    if any(
        getattr(args, key, None)
        for key in ("vla_backend", "vla_endpoint", "vla_model_path")
    ):
        raise ValueError("--wam-backend cannot be combined with VLA options")
    runtime_config = _runtime_config(backend)
    if platform not in runtime_config.ADAPTERS:
        raise ValueError(
            f"{backend} has no adapter for {platform}; supported: {tuple(runtime_config.ADAPTERS)}"
        )
    if bool(endpoint) == bool(checkpoint):
        raise ValueError(
            "--wam-backend requires --wam-endpoint or --wam-checkpoint, mutually exclusive"
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
