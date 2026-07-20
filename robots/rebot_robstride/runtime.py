"""Process lifecycle adapter for the reBot DevArm RobStride environment."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from robots.rebot_robstride.config import load_config
from robots.rebot_robstride.env_client import RebotRobstrideEnvClient
from robots.rebot_robstride.toolkit import RebotRobstrideToolkit
from rpent.envs.process import start_socket_server_process, stop_socket_server_process
from rpent.envs.runtime import EnvRuntime
from rpent.utils.config import get_repo_root
from rpent.utils.rpc import create_rpc_client, set_socket_endpoint


class RebotRobstrideRuntime(EnvRuntime):
    """Start or attach to the single-owner RobStride hardware server."""

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
        self._process: subprocess.Popen | None = None
        self._client: RebotRobstrideEnvClient | None = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    def start(self) -> RebotRobstrideToolkit:
        if not self.args.no_driver:
            command = [
                sys.executable,
                str(get_repo_root() / "robots" / "rebot_robstride" / "env_server.py"),
            ]
            env_config = getattr(self.args, "env_config", None)
            if env_config:
                command.extend(["--config", str(env_config)])
            self._process = start_socket_server_process(
                command,
                output_dir=self.output_dir,
                log_name="env_server.log",
                cwd=get_repo_root(),
                ready_timeout_s=30.0,
            )
        else:
            if self.args.env_port <= 0:
                raise ValueError(
                    "--no-driver requires --env-port pointing at an existing "
                    "reBot RobStride server"
                )
            set_socket_endpoint(
                self.output_dir, self.args.env_endpoint, self.args.env_port
            )

        client = RebotRobstrideEnvClient(create_rpc_client(self.output_dir))
        self._client = client
        config = load_config(getattr(self.args, "env_config", None))
        interval_s = min(0.5, config.heartbeat_timeout_s / 3.0)
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(interval_s,),
            daemon=True,
        )
        self._heartbeat_thread.start()
        return RebotRobstrideToolkit(env=client, dashboard=self.dashboard)

    def _heartbeat_loop(self, interval_s: float) -> None:
        while not self._heartbeat_stop.is_set():
            client = self._client
            if client is None:
                return
            try:
                client.heartbeat()
            except Exception:
                if self._heartbeat_stop.is_set():
                    return
            self._heartbeat_stop.wait(interval_s)

    def stop(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
        self._heartbeat_thread = None
        self._client = None
        stop_socket_server_process(
            self._process,
            output_dir=self.output_dir,
        )
        self._process = None
