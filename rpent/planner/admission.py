# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Exclude explicit serving-admission waits from an evaluation's task clock."""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

import httpx

from rpent.utils.serialization import write_json_atomic as write_json

PREFIX = b": rpent-admission "
HEADER = "x-rpent-admission-protocol"
WAIT_HEARTBEAT = b": waiting for model admission"
ADMISSION_RECORD = "admission-clock.json"


def recorded_queue_seconds(trace_dir: Path) -> float:
    """Read the latest wait confirmed by the planner's admission stream."""
    try:
        record = json.loads((trace_dir / ADMISSION_RECORD).read_text())
        seconds = float(record["confirmed_queue_seconds"])
    except (OSError, KeyError, TypeError, ValueError):
        return 0.0
    return seconds if math.isfinite(seconds) and seconds > 0 else 0.0


class AdmissionClock:
    """Account only waits reported by the admission proxy, never token generation."""

    def __init__(self, trace_dir: str | Path | None = None) -> None:
        self.seconds = 0.0
        self.waiting_since: float | None = None
        self._record_path = (
            Path(trace_dir) / ADMISSION_RECORD if trace_dir is not None else None
        )
        self._last_record = 0.0

    def excluded(self) -> float:
        return self.seconds + (
            time.monotonic() - self.waiting_since
            if self.waiting_since is not None
            else 0.0
        )

    def event(self, event: dict) -> None:
        if event["state"] == "waiting" and self.waiting_since is None:
            self.waiting_since = time.monotonic()
        elif event["state"] == "admitted" and self.waiting_since is not None:
            seconds = float(event["queue_seconds"])
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError("Invalid model admission duration")
            self.seconds += seconds
            self.waiting_since = None
        else:
            raise ValueError("Invalid model admission event order")
        self._record()

    def _record(self) -> None:
        if self._record_path is not None:
            self._last_record = time.monotonic()
            write_json(self._record_path, {"confirmed_queue_seconds": self.excluded()})

    def heartbeat(self) -> None:
        """Extend recorded queue time only while the proxy is responding."""
        if (
            self.waiting_since is not None
            and time.monotonic() - self._last_record >= 30
        ):
            self._record()

    def interrupted(self) -> None:
        if self.waiting_since is not None:
            self.seconds += time.monotonic() - self.waiting_since
            self.waiting_since = None
            self._record()


class _AdmissionStream(httpx.AsyncByteStream):
    def __init__(self, wrapped, clock: AdmissionClock) -> None:
        self.wrapped, self.clock = wrapped, clock
        self.pending = b""
        self.admitted = False

    async def __aiter__(self):
        # The SDK drains the same exhausted response again after [DONE].
        try:
            async for chunk in self.wrapped:
                if not self.admitted:
                    self.pending += chunk
                    while b"\n" in self.pending:
                        line, self.pending = self.pending.split(b"\n", 1)
                        line = line.rstrip(b"\r")
                        if line.startswith(PREFIX):
                            event = json.loads(line[len(PREFIX) :])
                            self.clock.event(event)
                            self.admitted = event["state"] == "admitted"
                        elif line == WAIT_HEARTBEAT:
                            self.clock.heartbeat()
                        elif line.startswith(b"data:"):
                            raise ValueError("Model replied before admission")
                        if self.admitted:
                            self.pending = b""
                            break
                    if len(self.pending) > 4096:
                        raise ValueError("Invalid model admission preamble")
                yield chunk
            if not self.admitted:
                raise ValueError("Model admission stream ended before admission")
        finally:
            self.clock.interrupted()

    async def aclose(self) -> None:
        await self.wrapped.aclose()


class AdmissionTransport(httpx.AsyncBaseTransport):
    """Observe proxy SSE comments without changing any model response bytes."""

    def __init__(self, clock: AdmissionClock, wrapped=None) -> None:
        self.clock = clock
        self.wrapped = wrapped or httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self.wrapped.handle_async_request(request)
        if response.status_code == 200:
            if response.headers.get(HEADER) != "1":
                await response.aclose()
                raise ValueError("Queue exclusion requires the RPent admission proxy")
            response.stream = _AdmissionStream(response.stream, self.clock)
        return response

    async def aclose(self) -> None:
        await self.wrapped.aclose()


async def await_task_budget(run: asyncio.Task, seconds: float, clock: AdmissionClock):
    """Cancel and drain the same task when its non-queue budget is exhausted."""
    started, excluded = time.monotonic(), clock.excluded()
    try:
        while not run.done():
            remaining = seconds - (
                time.monotonic() - started - (clock.excluded() - excluded)
            )
            if remaining <= 0:
                return await asyncio.wait_for(run, timeout=0)
            await asyncio.wait({run}, timeout=min(0.1, remaining))
        return await run
    except BaseException:
        if not run.done():
            run.cancel()
            await asyncio.gather(run, return_exceptions=True)
        raise
