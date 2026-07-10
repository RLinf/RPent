"""Pickle-framed TCP transport for the env + model RPC boundary.

The agent process ships ``(method, args, kwargs)`` tuples to the driver
process and receives the method's return value. Numpy arrays, dicts of
arrays, and other LIBERO/OpenPI payloads ride the wire as pickle frames
(length-prefixed, one frame per request/response).

Both processes are spawned by the same user on the same host, so we use
pickle rather than a more defensive codec.
"""
from __future__ import annotations

import pickle
import socket
import socketserver
import struct
import threading
import uuid
from collections.abc import Callable
from typing import Any


DEFAULT_CONNECT_TIMEOUT_S = 10.0
DEFAULT_REQUEST_TIMEOUT_S = 30.0

_LEN_PREFIX = struct.Struct(">I")


def _read_exact(reader, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = reader.read(n - len(buf))
        if not chunk:
            raise ConnectionError(
                f"socket closed mid-frame (read {len(buf)}/{n} bytes)"
            )
        buf.extend(chunk)
    return bytes(buf)


def _read_frame(reader) -> Any:
    (length,) = _LEN_PREFIX.unpack(_read_exact(reader, _LEN_PREFIX.size))
    return pickle.loads(_read_exact(reader, length))


def _write_frame(writer, obj: Any) -> None:
    body = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    writer.write(_LEN_PREFIX.pack(len(body)) + body)
    writer.flush()


class RpcError(RuntimeError):
    """Raised when a remote method call returns an error."""

    def __init__(self, method: str, message: str, *, traceback: str | None = None):
        super().__init__(f"{method}: {message}")
        self.method = method
        self.server_traceback = traceback


class SocketRpcClient:
    """One-request-per-connection pickle-framed RPC client."""

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

    def call(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        req_id = str(uuid.uuid4())
        payload = {
            "id": req_id,
            "method": method,
            "args": tuple(args),
            "kwargs": dict(kwargs or {}),
        }
        request_timeout_s = timeout_s or DEFAULT_REQUEST_TIMEOUT_S
        with socket.create_connection(
            (self.host, self.port), timeout=self.connect_timeout_s
        ) as sock:
            sock.settimeout(request_timeout_s)
            with sock.makefile("rwb") as f:
                _write_frame(f, payload)
                response = _read_frame(f)
        if not isinstance(response, dict):
            raise RpcError(method, f"bad response type: {type(response).__name__}")
        if response.get("id") != req_id:
            raise RpcError(
                method, f"id mismatch ({response.get('id')!r} != {req_id!r})"
            )
        if not response.get("ok"):
            raise RpcError(
                method,
                str(response.get("error", "<no error message>")),
                traceback=response.get("traceback"),
            )
        return response.get("result")

    def close(self) -> None:
        return None


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            payload = _read_frame(self.rfile)
        except Exception:
            return
        req_id = None
        try:
            req_id = payload.get("id") if isinstance(payload, dict) else None
            method = payload["method"]
            args = payload.get("args") or ()
            kwargs = payload.get("kwargs") or {}
            result = self.server.dispatch(method, args, kwargs)  # type: ignore[attr-defined]
            response: dict = {"id": req_id, "ok": True, "result": result}
        except Exception as exc:
            import traceback as _tb
            response = {
                "id": req_id,
                "ok": False,
                "error": str(exc),
                "traceback": _tb.format_exc(),
            }
        try:
            _write_frame(self.wfile, response)
        except Exception:
            pass


class SocketRpcServer(socketserver.ThreadingTCPServer):
    """TCP server that dispatches pickle-framed RPC calls."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        dispatch: Callable[[str, tuple, dict], Any],
    ):
        super().__init__(server_address, _RequestHandler)
        self._dispatch = dispatch
        self._dispatch_lock = threading.Lock()

    def dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        with self._dispatch_lock:
            return self._dispatch(method, args, kwargs)
