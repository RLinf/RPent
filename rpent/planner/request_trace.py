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


"""Lossless API request/response recording for offline multimodal supervision."""

from __future__ import annotations

import base64
import contextvars
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from rpent.utils.serialization import write_json

_NATIVE_ID = contextvars.ContextVar("rpent_native_request_id", default=None)


class RequestTrace:
    """Store exact wire payloads with content-addressed binary attachments."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.native_counter = len(list((self.directory / "native").glob("*.json")))
        self.http_counter = len(
            list((self.directory / "requests").glob("*/request.json"))
        )
        self.latest_native: str | None = None

    def asset(self, data: bytes, media_type: str = "application/octet-stream") -> dict:
        digest = hashlib.sha256(data).hexdigest()
        if data.startswith(b"\x89PNG"):
            media_type, extension = "image/png", "png"
        elif data.startswith(b"\xff\xd8"):
            media_type, extension = "image/jpeg", "jpg"
        else:
            extension = "bin"
        relative = Path("assets") / f"{digest}.{extension}"
        destination = self.directory / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        return {"$asset": str(relative), "sha256": digest, "media_type": media_type}

    def externalize(self, value: Any) -> Any:
        """Replace only binary bytes/data URIs, preserving the remaining payload."""
        if isinstance(value, bytes):
            return {**self.asset(value), "$encoding": "bytes"}
        if isinstance(value, str) and value.startswith("data:") and ";base64," in value:
            prefix, encoded = value.split(";base64,", 1)
            return {
                **self.asset(base64.b64decode(encoded, validate=True), prefix[5:]),
                "$encoding": "data_uri",
                "data_uri_prefix": prefix + ";base64,",
            }
        if isinstance(value, list):
            return [self.externalize(item) for item in value]
        if isinstance(value, dict):
            return {key: self.externalize(item) for key, item in value.items()}
        return value

    def native(self, messages: list, settings: dict, parameters: Any):
        """Capture the SDK messages after all history processors have run."""
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        self.native_counter += 1
        self.latest_native = f"{self.native_counter:06d}"
        serialized = self.externalize(
            ModelMessagesTypeAdapter.dump_python(messages, mode="python")
        )
        write_json(
            self.directory / "native" / f"{self.latest_native}.json",
            {
                "messages": serialized,
                "settings": settings,
                "parameters": self.externalize(vars(parameters)),
                "time_ns": time.time_ns(),
            },
        )
        return _NATIVE_ID.set(self.latest_native)

    def response(self, response: Any) -> None:
        """Keep provider metadata that the human-readable transcript omits."""
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        write_json(
            self.directory / "responses" / f"{self.latest_native}.json",
            self.externalize(
                ModelMessagesTypeAdapter.dump_python([response], mode="python")
            ),
        )

    def compaction(self, before: list, after: list) -> None:
        """Record both sides of a real compaction event, including native parts."""
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        if before == after:
            return
        value = self.externalize(
            ModelMessagesTypeAdapter.dump_python(before, mode="python")
        )
        retained = self.externalize(
            ModelMessagesTypeAdapter.dump_python(after, mode="python")
        )
        path = self.directory / "compactions.jsonl"
        with path.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "before": value,
                        "after": retained,
                        "before_count": len(before),
                        "after_count": len(after),
                        "next_native_id": f"{self.native_counter + 1:06d}",
                    },
                    default=str,
                )
                + "\n"
            )


class _ResponseStream(httpx.AsyncByteStream):
    def __init__(self, wrapped, directory: Path, metadata: dict) -> None:
        self.wrapped, self.directory, self.metadata = wrapped, directory, metadata
        self.started = False

    async def __aiter__(self):
        # The SDK's legacy-httpx adapter may iterate an exhausted stream again
        # during cleanup. Preserve the captured first pass without reopening it.
        if self.started:
            async for chunk in self.wrapped:
                yield chunk
            return
        self.started = True
        complete = False
        tail = b""
        error = None
        try:
            with (self.directory / "response.body").open("wb") as stream:
                async for chunk in self.wrapped:
                    stream.write(chunk)
                    tail = (tail + chunk)[-1000:]
                    if b"data: [DONE]" in tail:
                        complete = True
                        stream.flush()
                        self.metadata.update(
                            complete=True, error=None, finished_ns=time.time_ns()
                        )
                        write_json(self.directory / "metadata.json", self.metadata)
                    yield chunk
                complete = True
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.metadata.update(
                complete=complete,
                error=None if complete else error,
                finished_ns=time.time_ns(),
            )
            write_json(self.directory / "metadata.json", self.metadata)

    async def aclose(self) -> None:
        await self.wrapped.aclose()


class TraceTransport(httpx.AsyncBaseTransport):
    """Observe the real HTTP boundary without buffering or altering responses."""

    def __init__(self, trace: RequestTrace, wrapped=None) -> None:
        self.trace = trace
        self.wrapped = wrapped or httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.trace.http_counter += 1
        directory = self.trace.directory / "requests" / f"{self.trace.http_counter:06d}"
        directory.mkdir(parents=True, exist_ok=False)
        raw = await request.aread()
        payload = json.loads(raw)
        write_json(directory / "request.json", self.trace.externalize(payload))
        metadata = {
            "native_id": _NATIVE_ID.get(),
            "started_ns": time.time_ns(),
            "url": str(request.url),
            "complete": False,
            "utc": datetime.now(timezone.utc).isoformat(),
            "request_sha256": hashlib.sha256(raw).hexdigest(),
        }
        write_json(directory / "metadata.json", metadata)
        try:
            response = await self.wrapped.handle_async_request(request)
        except BaseException as exc:
            metadata.update(
                error=f"{type(exc).__name__}: {exc}", finished_ns=time.time_ns()
            )
            write_json(directory / "metadata.json", metadata)
            raise
        metadata.update(
            status_code=response.status_code,
            content_type=response.headers.get("content-type", ""),
        )
        if response.is_stream_consumed:
            (directory / "response.body").write_bytes(response.content)
            metadata.update(complete=True, error=None, finished_ns=time.time_ns())
            write_json(directory / "metadata.json", metadata)
            return response
        response.stream = _ResponseStream(response.stream, directory, metadata)
        return response

    async def aclose(self) -> None:
        await self.wrapped.aclose()


def reset_native(token) -> None:
    _NATIVE_ID.reset(token)


def traced_compaction(factory, trace: RequestTrace):
    """Attach auditing to the existing compactor without changing its decisions."""
    compactor = factory()
    original = compactor.before_model_request

    async def observed(ctx, request_context):
        before = list(request_context.messages)
        result = await original(ctx, request_context)
        trace.compaction(before, list(result.messages))
        return result

    compactor.before_model_request = observed
    return compactor
