# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Model registration and process ownership shared by robot WAM runtimes."""

import argparse
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from rpent.utils.config import get_repo_root
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.rpc import RpcClient, make_rpc_client

BACKENDS = {
    "cosmos-policy": "cosmos_policy",
    "fast-wam": "fast_wam",
}
ALIASES = {"cosmos": "cosmos-policy", "fast_wam": "fast-wam"}


def _runtime(backend: str) -> ModuleType:
    return importlib.import_module(
        f"rpent.robots.components.{BACKENDS[backend]}.runtime"
    )


def add_wam_args(parser: argparse.ArgumentParser) -> None:
    """Register shared options and lightweight model-owned launch options."""
    parser.add_argument("--wam-backend", choices=tuple(BACKENDS) + tuple(ALIASES))
    parser.add_argument("--wam-endpoint", help="HTTP/socket endpoint of a WAM worker")
    parser.add_argument("--wam-checkpoint", help="Local checkpoint for an owned worker")
    parser.add_argument(
        "--wam-python", help="Python executable in the model environment"
    )
    parser.add_argument("--wam-root", help="Working directory for the model worker")
    for backend in BACKENDS:
        _runtime(backend).add_arguments(parser)


@dataclass(frozen=True)
class WAMConfig:
    """Selected backend and platform, independent of robot toolkit behavior."""

    backend: str
    platform: str
    endpoint: str | None

    @property
    def rpc_backend(self) -> str:
        return BACKENDS[self.backend]

    def start_service(
        self, args: argparse.Namespace, output_dir: Path
    ) -> tuple[ProcessDaemon | None, RpcClient]:
        """Borrow an endpoint or launch the registered model in its own environment."""
        runtime = _runtime(self.backend)
        if self.endpoint:
            return None, make_rpc_client(
                self.endpoint, enable_sessions=runtime.USES_SESSIONS
            )
        port = pick_free_port()
        cmd = [
            str(Path(args.wam_python).expanduser().absolute()),
            "-m",
            f"rpent.robots.components.{self.rpc_backend}.server",
            *runtime.worker_arguments(args, self.platform),
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
        device = getattr(args, "wam_cuda_device", None)
        if device is None:
            device = getattr(args, "cuda_device", None)
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
        rpc = make_rpc_client(
            f"http://127.0.0.1:{port}", enable_sessions=runtime.USES_SESSIONS
        )
        try:
            daemon.start()
        except Exception:
            rpc.close()
            daemon.stop()
            raise
        return daemon, rpc


def select_wam(args: argparse.Namespace, platform: str) -> WAMConfig | None:
    """Validate backend/platform support and exclusive local/remote configuration."""
    backend = getattr(args, "wam_backend", None)
    endpoint = getattr(args, "wam_endpoint", None)
    checkpoint = getattr(args, "wam_checkpoint", None)
    backend = ALIASES.get(backend, backend)
    for name in BACKENDS:
        if name != backend and any(
            getattr(args, field, None) for field in _runtime(name).OPTION_FIELDS
        ):
            raise ValueError(f"{name} worker options require --wam-backend {name}")
    if backend is None:
        if endpoint or checkpoint:
            raise ValueError("--wam-endpoint/--wam-checkpoint requires --wam-backend")
        return None
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported WAM backend: {backend!r}")
    if any(
        getattr(args, key, None)
        for key in ("vla_backend", "vla_endpoint", "vla_model_path")
    ):
        raise ValueError("--wam-backend cannot be combined with VLA options")
    runtime = _runtime(backend)
    if platform not in runtime.ADAPTERS:
        raise ValueError(
            f"{backend} has no adapter for {platform}; supported: {tuple(runtime.ADAPTERS)}"
        )
    if bool(endpoint) == bool(checkpoint):
        raise ValueError(
            "--wam-backend requires --wam-endpoint or --wam-checkpoint, mutually exclusive"
        )
    # Each model validates its own options; external services own their deployment.
    runtime.validate_options(args, owned=bool(checkpoint))
    if checkpoint:
        if not Path(checkpoint).expanduser().exists():
            raise ValueError("--wam-checkpoint must reference a local checkpoint")
        if not args.wam_python or not Path(args.wam_python).expanduser().is_file():
            raise ValueError(
                "--wam-python must reference the provisioned model environment"
            )
        if args.wam_root and not Path(args.wam_root).expanduser().is_dir():
            raise ValueError("--wam-root must reference a directory")
    return WAMConfig(backend, platform, endpoint or None)
