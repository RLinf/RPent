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

"""Execute persistent Python cells with a generic proxy to registered tools."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import traceback
from typing import Any

MAX_MESSAGE_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_BYTES = 24 * 1024


class _Output(io.TextIOBase):
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.remaining = MAX_OUTPUT_BYTES
        self.truncated = False

    def write(self, text: str) -> int:
        encoded = text.encode("utf-8", errors="replace")
        if len(encoded) > self.remaining:
            self.truncated = True
        if self.remaining:
            self.parts.append(
                encoded[: self.remaining].decode("utf-8", errors="ignore")
            )
            self.remaining -= min(self.remaining, len(encoded))
        return len(text)

    def getvalue(self) -> str:
        return "".join(self.parts) + ("\n[output truncated]" if self.truncated else "")


class _Channel:
    def __init__(self) -> None:
        self.input = sys.stdin.buffer
        # Native libraries may write to fd 1. Keep those writes off the protocol.
        self.output = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
        with open(os.devnull, "wb") as sink:
            os.dup2(sink.fileno(), sys.stdout.fileno())

    def send(self, message: dict[str, Any]) -> None:
        data = (
            json.dumps(message, allow_nan=False, default=_json_value).encode("utf-8")
            + b"\n"
        )
        if len(data) > MAX_MESSAGE_BYTES:
            raise ValueError("Python message exceeds the 16 MiB limit")
        self.output.write(data)
        self.output.flush()

    def receive(self) -> dict[str, Any]:
        data = self.input.readline(MAX_MESSAGE_BYTES + 1)
        if not data:
            raise EOFError
        if len(data) > MAX_MESSAGE_BYTES or not data.endswith(b"\n"):
            raise ValueError("Invalid or oversized host message")
        result = json.loads(data)
        if not isinstance(result, dict):
            raise ValueError("Host message must be an object")
        return result

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.send({"kind": "call", "name": name, "arguments": arguments})
        response = self.receive()
        if response.get("kind") != "return":
            raise RuntimeError("Unexpected host response")
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["data"]


def _json_value(value: Any) -> Any:
    if type(value).__module__.split(".", 1)[0] == "numpy":
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    raise TypeError(f"Tool arguments cannot serialize {type(value).__name__}")


class _Finished(BaseException):
    """Stop the cell after finish; user ``except Exception`` cannot swallow it."""


class _Robot:
    """Resolve registered tool names without robot-specific wrappers."""

    def __init__(self, channel: _Channel, names: list[str]) -> None:
        self._channel = channel
        self._names = frozenset(names)

    def __getattr__(self, name: str) -> Any:
        if name not in self._names or name.startswith("_"):
            raise AttributeError(f"No Python robot tool {name!r}")

        def call(**arguments: Any) -> dict[str, Any]:
            result = self._channel.call(name, arguments)
            if result.get("_finish") is True and "error" not in result:
                raise _Finished
            return result

        return call

    def __dir__(self) -> list[str]:
        return sorted(self._names)


def main() -> None:
    """Serve sequential cells until the host closes its pipe or kills the worker."""
    channel = _Channel()
    namespace: dict[str, Any] = {"__name__": "__policy__"}
    while True:
        try:
            message = channel.receive()
        except EOFError:
            return
        if message.get("kind") != "execute":
            raise ValueError("Expected a Python execution request")
        stdout, stderr = _Output(), _Output()
        namespace["robo"] = _Robot(channel, message["tools"])
        result: dict[str, Any] = {"exception": None}
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                filename = message["filename"]
                exec(compile(message["code"], filename, "exec"), namespace)
            except _Finished:
                pass
            except BaseException as exc:
                result["exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc)[:8000],
                    "traceback": traceback.format_exc(limit=12)[-16000:],
                }
        result.update(stdout=stdout.getvalue(), stderr=stderr.getvalue())
        channel.send({"kind": "result", "result": result})


if __name__ == "__main__":
    main()
