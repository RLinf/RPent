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

"""Pickle-framed TCP transport for the env + model RPC boundary.

The client process ships ``(method, args, kwargs)`` tuples to the server
process and receives the method's return value. Numpy arrays, dicts of
arrays, and any other pickle-serializable payloads ride the wire as pickle
frames (length-prefixed, one frame per request/response).

This transport trusts both endpoints. Use a private network or SSH tunnel;
unpickling data from an untrusted client permits arbitrary code execution.
"""

from __future__ import annotations

import pickle
import shutil
import socket
import socketserver
import struct
import tempfile
from typing import Any, Callable

from rpent.utils.logging import get_logger
from rpent.utils.rpc.rpc_client import RpcClient, check_response
from rpent.utils.rpc.rpc_facade import make_error_response

logger = get_logger("socket_rpc")

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
    # Protocol 5 uses numpy._core.numeric for NumPy 2 arrays, which the
    # deployed NumPy 1.26 model runtime cannot import. Protocol 4 works on both.
    # Spill large RGBD chunks instead of keeping serialized bytes plus a second
    # concatenated length-prefixed copy in RAM. This is not a real-time guarantee.
    with tempfile.SpooledTemporaryFile(max_size=1024 * 1024) as body:
        pickle.dump(obj, body, protocol=4)
        writer.write(_LEN_PREFIX.pack(body.tell()))
        body.seek(0)
        shutil.copyfileobj(body, writer, length=1024 * 1024)
    writer.flush()


class SocketRpcClient(RpcClient):
    """One-request-per-connection pickle-framed RPC client."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        enable_sessions: bool = False,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
    ):
        super().__init__(enable_sessions=enable_sessions)
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
        payload = {
            "method": method,
            "args": tuple(args),
            "kwargs": dict(kwargs or {}),
            "session_id": self._session_id,
        }
        request_timeout_s = (
            DEFAULT_REQUEST_TIMEOUT_S if timeout_s is None else timeout_s
        )
        with socket.create_connection(
            (self.host, self.port), timeout=self.connect_timeout_s
        ) as sock:
            sock.settimeout(request_timeout_s)
            with sock.makefile("rwb") as f:
                _write_frame(f, payload)
                response = _read_frame(f)
        return check_response(response, method)


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            payload = _read_frame(self.rfile)
        except Exception as exc:
            logger.debug("rpc read failed: %s", exc)
            return
        try:
            method = payload["method"]
            args = payload.get("args") or ()
            kwargs = payload.get("kwargs") or {}
            session_id = payload.get("session_id")
            if session_id is None:
                result = self.server.dispatch(method, args, kwargs)
            else:
                result = self.server.dispatch(  # type: ignore[attr-defined]
                    method, args, kwargs, session_id=session_id
                )
            response: dict = {"ok": True, "result": result}
        except Exception as exc:
            response = make_error_response(exc)
        try:
            _write_frame(self.wfile, response)
        except Exception as exc:
            logger.debug("rpc write failed: %s", exc)


class SocketRpcServer(socketserver.ThreadingTCPServer):
    """TCP server that dispatches pickle-framed RPC calls."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        dispatch: Callable[..., Any],
    ):
        super().__init__(server_address, _RequestHandler)
        self.dispatch = dispatch
