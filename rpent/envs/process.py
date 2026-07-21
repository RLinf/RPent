"""Shared subprocess lifecycle helpers for socket-based environment servers."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import TextIO

from rpent.utils.rpc import create_rpc_client, set_socket_endpoint


def _pipe_output(
    proc: subprocess.Popen,
    log_file: TextIO,
    ready_events: "queue.Queue[dict]",
) -> None:
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            log_file.write(line)
            log_file.flush()
            try:
                event = json.loads(line)
            except Exception:
                continue
            if isinstance(event, dict) and event.get("event") == "transport_ready":
                ready_events.put(event)
    finally:
        log_file.close()


def start_socket_server_process(
    command: list[str],
    *,
    output_dir: str | Path,
    log_name: str,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    ready_timeout_s: float = 300.0,
) -> subprocess.Popen:
    """Start a server and register the socket endpoint from its ready event."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = (out_dir / log_name).open("a", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env or os.environ.copy(),
        cwd=cwd,
    )
    ready_events: queue.Queue[dict] = queue.Queue()
    threading.Thread(
        target=_pipe_output,
        args=(proc, log_file, ready_events),
        daemon=True,
    ).start()

    deadline = time.monotonic() + ready_timeout_s
    while True:
        try:
            event = ready_events.get(timeout=0.2)
        except queue.Empty:
            event = None
        if (
            event is not None
            and event.get("kind") == "socket"
            and event.get("host")
            and event.get("port")
        ):
            set_socket_endpoint(out_dir, event["host"], int(event["port"]))
            return proc
        if proc.poll() is not None:
            tail = (out_dir / log_name).read_text(errors="replace")[-3000:]
            raise RuntimeError(
                f"environment server exited before becoming ready:\n{tail}"
            )
        if time.monotonic() >= deadline:
            proc.terminate()
            raise RuntimeError(
                f"environment server not ready after {ready_timeout_s:.1f}s"
            )


def stop_socket_server_process(
    proc: subprocess.Popen | None,
    *,
    output_dir: str | Path,
    timeout_s: float = 15.0,
) -> None:
    """Request graceful shutdown, then terminate a stuck server."""
    if proc is None or proc.poll() is not None:
        return
    try:
        create_rpc_client(output_dir).call("shutdown", timeout_s=timeout_s)
    except Exception:
        pass
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
