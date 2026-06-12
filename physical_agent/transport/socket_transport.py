"""JSON-over-TCP transport for the interactive driver."""
from __future__ import annotations

import json
import socket
import socketserver
import threading
import uuid
import base64
from collections.abc import Callable
from pathlib import Path
from typing import Any


DEFAULT_CONNECT_TIMEOUT_S = 10.0
DEFAULT_REQUEST_TIMEOUT_S = 30.0


class SocketDriverClient:
    """One-request-per-connection JSON socket client."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
    ):
        self.host = host
        self.port = int(port)
        self.connect_timeout_s = connect_timeout_s

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict:
        req_id = str(uuid.uuid4())
        payload = {"id": req_id, "method": method, "params": params or {}}
        request_timeout_s = timeout_s or DEFAULT_REQUEST_TIMEOUT_S
        try:
            with socket.create_connection(
                (self.host, self.port),
                timeout=self.connect_timeout_s,
            ) as sock:
                sock.settimeout(request_timeout_s)
                with sock.makefile("rwb") as f:
                    f.write(json.dumps(payload).encode("utf-8") + b"\n")
                    f.flush()
                    line = f.readline()
        except Exception as exc:
            return {"error": f"socket transport error: {exc}"}

        if not line:
            return {"error": "socket transport error: empty response"}
        try:
            response = json.loads(line.decode("utf-8"))
        except Exception as exc:
            return {"error": f"socket transport error: invalid JSON response: {exc}"}
        if response.get("id") != req_id:
            return {
                "error": (
                    "socket transport error: response id mismatch "
                    f"{response.get('id')} != {req_id}"
                )
            }
        if not response.get("ok"):
            out = {"error": response.get("error", "socket transport request failed")}
            if response.get("traceback"):
                out["traceback"] = response["traceback"]
            return out
        result = response.get("result")
        if isinstance(result, dict):
            return result
        return {"result": result}

    def send_command(
        self,
        command: dict,
        *,
        current_step: int | None = None,
        timeout_s: float = 600.0,
    ) -> dict:
        params: dict[str, Any] = {"command": command}
        if current_step is not None:
            params["current_step"] = int(current_step)
        return self.request("send_command", params, timeout_s=timeout_s)

    def load_states(self) -> list:
        result = self.request("get_states")
        states = result.get("states")
        return states if isinstance(states, list) else []

    def latest_step(self) -> int | None:
        result = self.request("get_latest_step")
        step = result.get("step")
        return int(step) if step is not None else None

    def load_step(self, step: int | None = None) -> dict:
        params = {} if step is None else {"step": int(step)}
        result = self.request("get_step", params)
        if result.get("error"):
            raise IndexError(result["error"])
        entry = result.get("step_data")
        if not isinstance(entry, dict):
            raise ValueError("socket transport returned invalid step data")
        return entry

    def get_image_paths(self, step: int) -> dict[str, Path | str]:
        result = self.request("get_image_paths", {"step": int(step)})
        if result.get("error"):
            return {}
        return {
            key: value
            for key, value in result.items()
            if key in {"image", "image_cam"} and value
        }

    def load_image(self, step: int, kind: str = "agent") -> bytes | None:
        result = self.request("get_image", {"step": int(step), "kind": kind})
        if result.get("error"):
            return None
        data = result.get("data")
        if data is None:
            return None
        if not isinstance(data, str):
            raise ValueError("socket transport returned invalid image data")
        return base64.b64decode(data.encode("ascii"))

    def load_camera_meta(self) -> dict[str, Any]:
        result = self.request("get_camera_meta")
        if result.get("error"):
            raise FileNotFoundError(result["error"])
        meta = result.get("camera_meta")
        if not isinstance(meta, dict):
            raise ValueError("socket transport returned invalid camera metadata")
        return meta

    def load_depth(self, step: int) -> Any:
        import numpy as np

        result = self.request("get_depth", {"step": int(step)})
        if result.get("error"):
            raise FileNotFoundError(result["error"])
        if "depth" not in result:
            raise ValueError("socket transport returned no depth data")
        return np.array(result["depth"])

    def close(self) -> None:
        return None


# Compatibility name from the command-only abstraction.
SocketTransportClient = SocketDriverClient


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        line = self.rfile.readline(10_000_000)
        if not line:
            return
        req_id = None
        try:
            request = json.loads(line.decode("utf-8"))
            req_id = request.get("id")
            result = self.server.dispatch(request)  # type: ignore[attr-defined]
            response = {"id": req_id, "ok": True, "result": result}
        except Exception as exc:
            import traceback

            response = {
                "id": req_id,
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        self.wfile.write(json.dumps(response, default=str).encode("utf-8") + b"\n")


class TransportTCPServer(socketserver.ThreadingTCPServer):
    """Small TCP server that dispatches one JSON request per connection."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        dispatch: Callable[[dict], dict],
    ):
        super().__init__(server_address, _RequestHandler)
        self._dispatch = dispatch
        self._dispatch_lock = threading.Lock()

    def dispatch(self, request: dict) -> dict:
        method = request.get("method")
        if method == "send_command":
            with self._dispatch_lock:
                return self._dispatch(request)
        return self._dispatch(request)
