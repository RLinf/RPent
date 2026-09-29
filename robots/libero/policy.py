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
        choices=("cosmos-policy",),
        default=None,
        help="WAM action backend; requires --wam-endpoint.",
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


@dataclass(frozen=True)
class PolicyConfig:
    """Resolved model category and backend shared by CLI and Dashboard runs."""

    kind: str
    backend: str
    endpoint: str | None

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

            return CosmosPolicyClient(rpc)
        from rpent.robots.components.pi05_vla_client import Pi05VLAClient

        return Pi05VLAClient(rpc, embodiment="libero")

    def start_service(
        self,
        args: argparse.Namespace,
        output_dir: Path,
    ) -> tuple[ProcessDaemon | None, RpcClient]:
        """Borrow an external model service or start the local Pi0.5 worker."""
        if self.endpoint is not None:
            return None, make_rpc_client(self.endpoint)
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


def select_policy(args: argparse.Namespace) -> PolicyConfig:
    """Resolve category and reject mixed VLA/WAM endpoint configuration."""
    if args.wam_backend is not None:
        if args.vla_backend is not None or args.vla_endpoint is not None:
            raise ValueError("--wam-backend cannot be combined with VLA options")
        if not args.wam_endpoint:
            raise ValueError("--wam-backend requires --wam-endpoint")
        return PolicyConfig("wam", args.wam_backend, args.wam_endpoint)
    if args.wam_endpoint is not None:
        raise ValueError("--wam-endpoint requires --wam-backend")
    return PolicyConfig("vla", args.vla_backend or "pi05", args.vla_endpoint)
