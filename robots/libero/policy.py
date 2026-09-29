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

"""LIBERO action-model selection, capabilities and service connection."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rpent.utils.config import get_repo_root
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.rpc import make_rpc_client
from rpent.utils.rpc.http_rpc import HttpRpcClient

if TYPE_CHECKING:
    from rpent.robots.components.policy_client_base import BasePolicyClient
    from rpent.utils.rpc import RpcClient


def add_policy_args(parser: argparse.ArgumentParser) -> None:
    """Expose mutually exclusive VLA and WAM backends with separate endpoints."""
    backends = parser.add_mutually_exclusive_group()
    backends.add_argument(
        "--vla-backend",
        choices=("pi05",),
        default=None,
        help="VLA action backend (default: pi05 when no WAM is selected).",
    )
    backends.add_argument(
        "--wam-backend",
        choices=("cosmos-policy", "cosmos"),
        default=None,
        help="WAM action backend; cosmos is an alias for cosmos-policy.",
    )
    parser.add_argument(
        "--vla-endpoint",
        default=None,
        help="[http|socket://]host:port of a VLA service; otherwise start Pi0.5 locally.",
    )
    parser.add_argument(
        "--wam-endpoint",
        default=None,
        help="[http|socket://]host:port of a separately started WAM service.",
    )
    parser.add_argument(
        "--wam-checkpoint",
        default=None,
        help="Local Cosmos checkpoint file or directory; starts an owned worker.",
    )
    parser.add_argument(
        "--wam-python",
        default=os.getenv("COSMOS_POLICY_PYTHON"),
        help="Python executable in the isolated Cosmos environment.",
    )
    parser.add_argument(
        "--wam-root",
        default=os.getenv("COSMOS_POLICY_ROOT"),
        help="Official Cosmos source checkout used as the worker directory.",
    )
    parser.add_argument(
        "--wam-text-embeddings",
        default=None,
        help="Writable T5 embeddings cache for an owned worker.",
    )
    parser.add_argument(
        "--wam-predict-future",
        action="store_true",
        help="Enable optional future-state/value generation in an owned worker.",
    )
    parser.add_argument(
        "--wam-cached-instructions-only",
        action="store_true",
        help="Reject uncached instructions instead of loading T5 in an owned worker.",
    )


@dataclass(frozen=True)
class PolicyConfig:
    """Resolved model category and backend shared by CLI and Dashboard runs."""

    kind: str
    backend: str
    endpoint: str | None
    checkpoint: str | None = None

    @property
    def memory_enabled(self) -> bool:
        """Whether the current integration provides an experience corpus."""
        return self.backend == "pi05"

    def validate_run(self, args: argparse.Namespace) -> None:
        """Validate model-specific capabilities without restricting task suites."""
        if self.backend != "cosmos-policy":
            return
        if args.planner == "flash" or getattr(args, "explore", False):
            raise ValueError(
                "Cosmos Policy supports evaluation without Flash Mode only"
            )
        if args.libero_type not in ("standard", "pro"):
            raise ValueError(
                "Cosmos Policy supports standard and pro environments only"
            )
        if getattr(args, "memory_profile", None) == "hf":
            raise ValueError(
                "Cosmos Policy requires --memory-profile local; "
                "this skips HF synchronization; Cosmos memory is not supported"
            )
        if args.memory_dir is not None:
            raise ValueError("Cosmos Policy does not support --memory-dir")

    def make_client(self, rpc: RpcClient) -> BasePolicyClient:
        """Create the backend's observation/action adapter with lazy imports."""
        if self.backend == "cosmos-policy":
            from rpent.robots.components.cosmos_policy_client import CosmosPolicyClient

            client = CosmosPolicyClient(rpc)
            try:
                client.validate_libero()
            except Exception:
                client.close()
                raise
            return client
        from rpent.robots.components.pi05_vla_client import Pi05VLAClient

        return Pi05VLAClient(rpc, embodiment="libero")

    def start_service(
        self,
        args: argparse.Namespace,
        output_dir: Path,
    ) -> tuple[ProcessDaemon | None, RpcClient]:
        """Borrow a model endpoint or own the selected backend's worker."""
        if self.endpoint is not None:
            return None, make_rpc_client(self.endpoint)
        if self.kind == "wam":
            return self._start_cosmos(args, output_dir)
        host, port = "127.0.0.1", pick_free_port()
        cuda_args = (
            ["--cuda-device", str(args.cuda_device)]
            if args.cuda_device is not None
            else []
        )
        daemon = ProcessDaemon(
            name="vla_server",
            cmd=[
                sys.executable,
                str(get_repo_root() / "rpent/robots/components/pi05_vla_server.py"),
                "--embodiment",
                "libero",
                "--transport",
                "http",
                "--host",
                host,
                "--port",
                str(port),
                "--parent-watch",
                *cuda_args,
            ],
            log_path=str(output_dir / "vla_server.log"),
        )
        daemon.start()
        return daemon, HttpRpcClient(f"http://{host}:{port}")

    def _start_cosmos(
        self, args: argparse.Namespace, output_dir: Path
    ) -> tuple[ProcessDaemon, RpcClient]:
        """Start the canonical model server using an explicitly provisioned environment."""
        host, port = "127.0.0.1", pick_free_port()
        cmd = [
            str(Path(args.wam_python).expanduser().absolute()),
            "-m",
            "rpent.robots.components.cosmos_policy_server",
            "--checkpoint",
            str(Path(self.checkpoint).expanduser().resolve()),
            "--host",
            host,
            "--port",
            str(port),
            "--parent-watch",
        ]
        for enabled, option in (
            (args.wam_predict_future, "--predict-future"),
            (args.wam_cached_instructions_only, "--cached-instructions-only"),
        ):
            if enabled:
                cmd.append(option)
        if args.cuda_device is not None:
            cmd.extend(["--cuda-device", str(args.cuda_device)])
        if args.wam_text_embeddings:
            cmd.extend(
                [
                    "--text-embeddings",
                    str(Path(args.wam_text_embeddings).expanduser().absolute()),
                ]
            )
        pythonpath = os.pathsep.join(
            filter(None, (str(get_repo_root()), os.environ.get("PYTHONPATH")))
        )
        daemon = ProcessDaemon(
            name="wam",
            cmd=cmd,
            cwd=str(Path(args.wam_root).expanduser().resolve()),
            env_overrides={"PYTHONPATH": pythonpath},
            log_path=str(output_dir / "wam_server.log"),
        )
        daemon.start()
        return daemon, HttpRpcClient(f"http://{host}:{port}")


def select_policy(args: argparse.Namespace) -> PolicyConfig:
    """Resolve category and reject mixed VLA/WAM endpoint configuration."""
    checkpoint = getattr(args, "wam_checkpoint", None)
    worker_options = any(
        getattr(args, name, None)
        for name in (
            "wam_predict_future",
            "wam_cached_instructions_only",
            "wam_text_embeddings",
        )
    )
    if worker_options and not checkpoint:
        raise ValueError(
            "WAM worker options require --wam-checkpoint; configure external workers at their launch"
        )
    if args.wam_backend is not None:
        if args.wam_backend not in ("cosmos-policy", "cosmos"):
            raise ValueError(f"Unsupported LIBERO WAM backend: {args.wam_backend!r}")
        if args.vla_backend is not None or args.vla_endpoint is not None:
            raise ValueError("--wam-backend cannot be combined with VLA options")
        if bool(args.wam_endpoint) == bool(checkpoint):
            raise ValueError(
                "--wam-backend requires --wam-endpoint or --wam-checkpoint, mutually exclusive"
            )
        if checkpoint:
            if not Path(checkpoint).expanduser().exists():
                raise ValueError("--wam-checkpoint must reference a local checkpoint")
            for name, is_directory in (("wam_python", False), ("wam_root", True)):
                value = getattr(args, name, None)
                path = Path(value).expanduser() if value else None
                if path is None or not (
                    path.is_dir() if is_directory else path.is_file()
                ):
                    raise ValueError(
                        f"--{name.replace('_', '-')} must reference the provisioned Cosmos environment"
                    )
        return PolicyConfig(
            "wam", "cosmos-policy", args.wam_endpoint or None, checkpoint
        )
    if args.wam_endpoint is not None or checkpoint is not None:
        raise ValueError("--wam-endpoint/--wam-checkpoint requires --wam-backend")
    return PolicyConfig("vla", args.vla_backend or "pi05", args.vla_endpoint)
