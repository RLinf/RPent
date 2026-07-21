"""LIBERO process lifecycle adapter for the generic RPent runner."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from robots.libero.env_client import LiberoEnvClient
from robots.libero.toolkit import LiberoToolkit
from rpent.envs.process import start_socket_server_process, stop_socket_server_process
from rpent.envs.runtime import EnvRuntime
from rpent.utils.config import get_libero_type, get_repo_root
from rpent.utils.rpc import create_rpc_client, set_socket_endpoint
from rpent.utils.vla_client import VLAClient


class LiberoRuntime(EnvRuntime):
    """Preserve the existing LIBERO env/VLA lifecycle behind EnvRuntime."""

    def __init__(
        self,
        *,
        args: Any,
        output_dir: str | Path,
        dashboard: Any = None,
    ) -> None:
        self.args = args
        self.output_dir = Path(output_dir)
        self.dashboard = dashboard
        self._env_proc: subprocess.Popen | None = None
        self._vla_proc: subprocess.Popen | None = None

    def start(self) -> LiberoToolkit:
        args = self.args
        if not args.suite:
            raise ValueError("the LIBERO environment requires --suite")
        if args.task is None:
            raise ValueError("the LIBERO environment requires --task")

        vla_endpoint = args.vla_endpoint
        if not args.no_driver:
            env = os.environ.copy()
            env["LIBERO_TYPE"] = args.libero_type or get_libero_type()
            if args.cuda_device is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
            env.setdefault("MUJOCO_GL", "egl")
            env.setdefault("ROBOT_PLATFORM", "LIBERO")
            command = [
                sys.executable,
                str(get_repo_root() / "robots" / "libero" / "env_server.py"),
                "--suite",
                args.suite,
                "--task",
                str(args.task),
                "--seed",
                str(args.seed),
                "--max-episode-steps",
                str(args.max_episode_steps),
                "--output-dir",
                str(self.output_dir),
            ]
            self._env_proc = start_socket_server_process(
                command,
                output_dir=self.output_dir,
                log_name="env_server.log",
                env=env,
                cwd=get_repo_root(),
            )
            if vla_endpoint is None:
                vla_endpoint, self._vla_proc = _start_vla_server(
                    cuda_device=args.cuda_device,
                    log_path=self.output_dir / "vla_server.log",
                )
        else:
            if args.env_port <= 0:
                raise ValueError(
                    "--no-driver requires --env-port pointing at an existing env_server"
                )
            if vla_endpoint is None:
                raise ValueError(
                    "--no-driver requires --vla-endpoint pointing at an existing vla_server"
                )
            set_socket_endpoint(self.output_dir, args.env_endpoint, args.env_port)

        expected_meta = {
            "suite": args.suite,
            "task": args.task,
            "seed": args.seed,
            "max_episode_steps": args.max_episode_steps,
        }
        env_client = LiberoEnvClient(
            create_rpc_client(self.output_dir), expected_meta=expected_meta
        )
        return LiberoToolkit(
            primitives_kwargs={
                "env": env_client,
                "model": VLAClient(vla_endpoint),
            },
            video_path=str(self.output_dir / "episode.mp4"),
            dashboard=self.dashboard,
        )

    def stop(self) -> None:
        stop_socket_server_process(
            self._env_proc,
            output_dir=self.output_dir,
        )
        _stop_vla_server(self._vla_proc)
        self._env_proc = None
        self._vla_proc = None


def _start_vla_server(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    cuda_device: str | None = None,
    log_path: str | Path | None = None,
) -> tuple[str, subprocess.Popen]:
    if port == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            port = int(sock.getsockname()[1])

    env = os.environ.copy()
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    command = [
        sys.executable,
        str(get_repo_root() / "robots" / "libero" / "vla_server.py"),
        "--host",
        host,
        "--port",
        str(port),
    ]
    log_file = Path(log_path).open("a", encoding="utf-8") if log_path else None
    proc = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT if log_file else None,
        env=env,
    )
    base_url = f"http://{host}:{port}"
    client = VLAClient(base_url)
    deadline = time.monotonic() + 300.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("vla server exited before becoming ready")
        try:
            if client.healthz():
                return base_url, proc
        except Exception:
            pass
        time.sleep(2.0)
    proc.terminate()
    raise RuntimeError("vla server not ready after 300s")


def _stop_vla_server(proc: subprocess.Popen | None, timeout_s: float = 10.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
