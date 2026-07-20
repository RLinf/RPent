#!/usr/bin/env python3
"""Single-owner SocketCAN RPC server for the reBot DevArm RobStride arm."""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any

from robots.rebot_robstride.config import load_config
from robots.rebot_robstride.driver import RebotRobstrideDriver
from rpent.utils.socket_rpc import SocketRpcServer


def make_dispatch(driver: Any, shutdown_event: threading.Event):
    """Return the strict RPC dispatcher used by the hardware server."""
    handlers = {
        "robot.state": driver.state,
        "robot.enable": driver.enable,
        "robot.move_joints": driver.move_joints,
        "robot.set_gripper": driver.set_gripper,
        "robot.stop_motion": driver.stop_motion,
        "robot.reset_stop": driver.reset_stop,
        "robot.emergency_stop": driver.emergency_stop,
    }

    def dispatch(method: str, args: tuple, kwargs: dict):
        if method == "shutdown":
            shutdown_event.set()
            return {"ok": True}
        handler = handlers.get(method)
        if handler is None:
            raise ValueError(f"unknown RPC method: {method!r}")
        return handler(*args, **kwargs)

    return dispatch


def validate_socketcan(channel: str, bitrate: int) -> None:
    """Fail with an actionable message unless SocketCAN is up at the expected rate."""
    result = subprocess.run(
        ["ip", "-details", "link", "show", channel],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"SocketCAN interface {channel!r} does not exist; connect a CAN adapter first"
        )
    output = result.stdout
    first_line = output.splitlines()[0] if output.splitlines() else ""
    flags = first_line.partition("<")[2].partition(">")[0].split(",")
    if "UP" not in flags:
        raise RuntimeError(
            f"SocketCAN interface {channel!r} is down; run: "
            f"sudo ip link set {channel} up type can bitrate {bitrate}"
        )
    match = re.search(r"\bbitrate\s+(\d+)", output)
    if match is None or int(match.group(1)) != bitrate:
        actual = match.group(1) if match else "unknown"
        raise RuntimeError(
            f"SocketCAN interface {channel!r} bitrate is {actual}, expected {bitrate}; "
            f"reconfigure it before starting RPent"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--config", type=Path, default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = load_config(args.config)
    validate_socketcan(config.channel, config.bitrate)

    driver = RebotRobstrideDriver(config)
    shutdown_event = threading.Event()
    server = None
    try:
        initial_state = driver.connect()
        server = SocketRpcServer(
            (args.host, args.port), make_dispatch(driver, shutdown_event)
        )
        host, port = server.server_address
        threading.Thread(target=server.serve_forever, daemon=True).start()

        def request_shutdown(_signum, _frame) -> None:
            shutdown_event.set()

        signal.signal(signal.SIGINT, request_shutdown)
        signal.signal(signal.SIGTERM, request_shutdown)
        print(
            json.dumps(
                {
                    "event": "transport_ready",
                    "kind": "socket",
                    "host": host,
                    "port": port,
                    "environment": "rebot_robstride",
                    "motors_seen": len(initial_state["joint_positions"]) + 1,
                }
            ),
            flush=True,
        )
        shutdown_event.wait()
        return 0
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
