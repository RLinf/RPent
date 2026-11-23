# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Serving waits preserve response bytes and do not consume robot task budgets."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIError, AsyncOpenAI

from rpent.planner.admission import (
    HEADER,
    PREFIX,
    AdmissionClock,
    AdmissionTransport,
    await_task_budget,
    recorded_queue_seconds,
)
from tools.rl.admission_proxy import create_app


class Bytes(httpx.AsyncByteStream):
    def __init__(self, chunks, gate=None):
        self.chunks, self.gate, self.closed = chunks, gate, False

    async def __aiter__(self):
        if self.gate is not None:
            await self.gate.wait()
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


def test_model_factory_keeps_queue_clock_and_http_trace(monkeypatch, tmp_path):
    from rpent.planner.base import build_api_model

    monkeypatch.setenv("RPENT_EXCLUDE_MODEL_QUEUE", "1")
    model = build_api_model(
        "openai-chat:qwen-test",
        "http://localhost:31000/replica/0/v1",
        strict_tool_definitions=False,
        request_trace_dir=tmp_path,
    )
    assert isinstance(model.admission_clock, AdmissionClock)
    assert model.request_trace.directory == tmp_path


def test_watchdog_only_credits_reported_model_wait(tmp_path, monkeypatch):
    from rpent.planner import admission

    now = 100.0
    monkeypatch.setattr(admission, "time", SimpleNamespace(monotonic=lambda: now))
    clock = AdmissionClock(tmp_path)
    clock.event({"state": "waiting"})
    now = 131.0
    clock.heartbeat()
    assert recorded_queue_seconds(tmp_path) == 31.0

    now = 160.0
    assert recorded_queue_seconds(tmp_path) == 31.0
    clock.event({"state": "admitted", "queue_seconds": 60.0})
    now = 190.0
    clock.heartbeat()
    assert recorded_queue_seconds(tmp_path) == 60.0


def test_direct_endpoint_cannot_silently_exclude_unknown_queue():
    async def run():
        stream = Bytes([b"data: [DONE]\n\n"])
        transport = AdmissionTransport(
            AdmissionClock(),
            httpx.MockTransport(lambda request: httpx.Response(200, stream=stream)),
        )
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError, match="requires the RPent admission proxy"):
                await client.post("http://direct/v1/chat/completions")
        assert stream.closed

    asyncio.run(run())


def test_admission_transport_keeps_sse_bytes_and_closes_stream():
    async def run():
        wire = (
            PREFIX
            + b'{"state":"waiting"}\n\n'
            + b": waiting for model admission\n\n"
            + PREFIX
            + b'{"state":"admitted","queue_seconds":2.5}\n\n'
            + b'data: {"choices":[]}\n\ndata: [DONE]\n\n'
        )
        stream = Bytes([wire[:5], wire[5:37], wire[37:]])
        clock = AdmissionClock()
        transport = AdmissionTransport(
            clock,
            httpx.MockTransport(
                lambda request: httpx.Response(
                    200, headers={HEADER: "1"}, stream=stream
                )
            ),
        )
        async with httpx.AsyncClient(transport=transport) as client:
            response = await client.post("http://model/v1/chat/completions")
        assert response.content == wire and stream.closed
        assert clock.seconds == 2.5 and clock.waiting_since is None

    asyncio.run(run())


def test_sdk_cleanup_can_drain_the_exhausted_admitted_stream():
    async def run():
        wire = (
            PREFIX
            + b'{"state":"waiting"}\n\n'
            + PREFIX
            + b'{"state":"admitted","queue_seconds":1.25}\n\n'
            + b"data: [DONE]\n\n"
        )

        class Once(httpx.AsyncByteStream):
            def __init__(self):
                self.chunks = iter([wire])

            async def __aiter__(self):
                for chunk in self.chunks:
                    yield chunk

        clock = AdmissionClock()
        transport = AdmissionTransport(
            clock,
            httpx.MockTransport(
                lambda request: httpx.Response(
                    200, headers={HEADER: "1"}, stream=Once()
                )
            ),
        )
        async with httpx.AsyncClient(transport=transport) as client:
            async with client.stream(
                "POST", "http://model/v1/chat/completions"
            ) as response:
                assert (
                    b"".join([chunk async for chunk in response.aiter_bytes()]) == wire
                )
                assert [chunk async for chunk in response.stream] == []
        assert clock.seconds == 1.25 and clock.waiting_since is None

    asyncio.run(run())


@pytest.mark.parametrize(
    "wire",
    [
        b"data: {}\n\n",
        PREFIX + b'{"state":"waiting"}\n\n',
        PREFIX
        + b'{"state":"waiting"}\n\n'
        + PREFIX
        + b'{"state":"admitted","queue_seconds":-1}\n\n',
    ],
)
def test_admission_requires_ordered_valid_accounting(wire):
    async def run():
        clock = AdmissionClock()
        transport = AdmissionTransport(
            clock,
            httpx.MockTransport(
                lambda request: httpx.Response(
                    200, headers={HEADER: "1"}, stream=Bytes([wire])
                )
            ),
        )
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError):
                await client.post("http://model/v1/chat/completions")
        assert clock.waiting_since is None

    asyncio.run(run())


def test_task_budget_excludes_queue_and_cancels_overlong_work():
    async def run():
        clock = AdmissionClock()

        async def queued():
            clock.event({"state": "waiting"})
            await asyncio.sleep(0.2)
            clock.event({"state": "admitted", "queue_seconds": 0.2})
            await asyncio.sleep(0.01)
            return "done"

        assert (
            await await_task_budget(asyncio.create_task(queued()), 0.1, clock) == "done"
        )
        closed = asyncio.Event()

        async def slow():
            try:
                await asyncio.sleep(2)
            finally:
                closed.set()

        task = asyncio.create_task(slow())
        with pytest.raises(TimeoutError):
            await await_task_budget(task, 0.03, clock)
        assert task.cancelled() and closed.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("max_in_flight", [1, 32])
def test_proxy_limits_replica_preserves_payload_and_releases_waiter(max_in_flight):
    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()
        requests = []

        async def upstream(request):
            requests.append(json.loads(request.content))
            if len(requests) == max_in_flight:
                entered.set()
            return httpx.Response(200, stream=Bytes([b"data: [DONE]\n\n"], release))

        app = create_app(
            ["http://native/v1"],
            httpx.MockTransport(upstream),
            max_in_flight=max_in_flight,
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://proxy"
            ) as client:
                body = {
                    "stream": True,
                    "model": "qwen",
                    "messages": [{"role": "user", "content": "same"}],
                    "temperature": 1,
                }
                active = [
                    asyncio.create_task(
                        client.post("/replica/0/v1/chat/completions", json=body)
                    )
                    for _ in range(max_in_flight)
                ]
                await asyncio.wait_for(entered.wait(), 2)
                waiting = asyncio.create_task(
                    client.post("/replica/0/v1/chat/completions", json=body)
                )
                await asyncio.sleep(0.02)
                assert requests == [body] * max_in_flight
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
                release.set()
                for result in await asyncio.gather(*active):
                    assert result.content.endswith(b"data: [DONE]\n\n")
                    assert result.headers[HEADER] == "1"
                await asyncio.wait_for(
                    client.post("/replica/0/v1/chat/completions", json=body), 1
                )
                assert requests == [body] * (max_in_flight + 1)

    asyncio.run(run())


def test_proxy_upstream_errors_remain_sdk_errors():
    async def run():
        app = create_app(
            ["http://native/v1"],
            httpx.MockTransport(
                lambda request: httpx.Response(503, text="unavailable")
            ),
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app)) as http:
                client = AsyncOpenAI(
                    api_key="local",
                    base_url="http://proxy/replica/0/v1",
                    http_client=http,
                    max_retries=0,
                )
                stream = await client.chat.completions.create(
                    model="qwen",
                    messages=[{"role": "user", "content": "hello"}],
                    stream=True,
                )
                with pytest.raises(APIError, match="503 Service Unavailable"):
                    async for _ in stream:
                        pass

    asyncio.run(run())
