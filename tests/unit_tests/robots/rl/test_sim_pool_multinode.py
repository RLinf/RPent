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

"""Unit tests for SimPoolRouter, --sleepable pools, and SAM3 2048 backlog."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request, Response

from robots.rl.server import (
    BINARY_EXECUTE_MEDIA_TYPE,
    SimPoolManager,
    SimPoolRouter,
    _WorkerHandle,
    pack_execute_response,
    unpack_execute_response,
)
from rpent.robots.components.sam3_pool import Sam3Pool
from rpent.utils.rpc.http_rpc import HttpRpcServer


def test_http_rpc_and_sam3_pool_support_2048_queue() -> None:
    assert HttpRpcServer.request_queue_size >= 2048
    dummy = SimpleNamespace(call=lambda *a, **kw: {}, close=lambda: None)
    pool = Sam3Pool([dummy], max_pending=2048, timeout_s=600.0)
    assert pool.max_pending == 2048
    assert pool.timeout_s == 600.0
    pool.close()


def test_sim_pool_manager_sleep_and_wake() -> None:
    async def _run() -> None:
        mgr = SimPoolManager(
            environment="libero",
            gpus=[0, 1],
            num_workers=2,
            sim_python="python",
            sam3_endpoint=None,
            sleepable=True,
        )
        killed: list[int] = []
        spawned: list[int | None] = []

        class _FakeProc:
            def poll(self):
                return None

            def terminate(self):
                killed.append(1)

            def wait(self, timeout=None):
                return 0

            def kill(self):
                killed.append(1)

        class _FakeWriter:
            def close(self):
                pass

        async def _fake_spawn(gpu):
            spawned.append(gpu)
            h = _WorkerHandle(
                worker_id=len(spawned),
                gpu=gpu,
                proc=_FakeProc(),
                reader=None,
                writer=_FakeWriter(),
            )
            mgr._workers.add(h)
            await mgr._idle.put(h)
            return h

        mgr._spawn_worker = _fake_spawn  # type: ignore[method-assign]
        await mgr._spawn_worker(0)
        await mgr._spawn_worker(1)
        assert mgr.health()["workers_total"] == 2
        assert mgr.health()["sleeping"] is False

        slept = await mgr.sleep()
        assert slept["sleeping"] is True
        assert slept["workers_total"] == 0
        assert slept["workers_idle"] == 0
        assert len(killed) == 2

        with pytest.raises(RuntimeError, match="sleeping"):
            await mgr.open_session({"task": {}})

        woke = await mgr.wake()
        assert woke["sleeping"] is False
        assert woke["workers_total"] == 2
        assert woke["workers_idle"] == 2
        assert spawned == [0, 1, 0, 1]

        # Non-sleepable pool is a no-op
        dedicated = SimPoolManager(
            environment="robotwin",
            gpus=[0],
            num_workers=1,
            sim_python="python",
            sam3_endpoint=None,
            sleepable=False,
        )
        res = await dedicated.sleep()
        assert res["sleepable"] is False
        assert res["sleeping"] is False

    asyncio.run(_run())


def _make_fake_backend(name: str, capacity: int, *, sleepable: bool) -> FastAPI:
    app = FastAPI()
    state = {
        "sleeping": False,
        "sessions": {},
        "seq": 0,
        "cohorts_closed": [],
    }

    @app.get("/health")
    async def health():
        total = 0 if state["sleeping"] else capacity
        active = len(state["sessions"])
        return {
            "ok": True,
            "environment": "robotwin",
            "workers_target": capacity,
            "workers_total": total,
            "workers_idle": max(0, total - active),
            "active_sessions": active,
            "sleepable": sleepable,
            "sleeping": state["sleeping"],
        }

    @app.post("/sessions/open")
    async def open_session(req: Request):
        body = await req.json()
        state["seq"] += 1
        sid = f"{name}-{state['seq']}"
        state["sessions"][sid] = body.get("cohort_id")
        return {"session_id": sid, "backend": name}

    @app.post("/sessions/{sid}/execute")
    async def execute(sid: str, req: Request):
        body = json.loads(await req.body())
        meta = {"text": f"{name}:{body['name']}", "done": False}
        img = b"\x89PNG-raw-" + name.encode()
        return Response(
            content=pack_execute_response(meta, [img]),
            media_type=BINARY_EXECUTE_MEDIA_TYPE,
        )

    @app.post("/sessions/{sid}/close")
    async def close_session(sid: str):
        state["sessions"].pop(sid, None)
        return {"closed": True}

    @app.post("/cohorts/{cid}/close")
    async def close_cohort(cid: str):
        sids = [s for s, c in list(state["sessions"].items()) if c == cid]
        for s in sids:
            state["sessions"].pop(s, None)
        state["cohorts_closed"].append(cid)
        return {"cohort_id": cid, "closed": len(sids)}

    @app.post("/pool/sleep")
    async def sleep():
        if sleepable:
            state["sleeping"] = True
            state["sessions"].clear()
        return await health()

    @app.post("/pool/wake")
    async def wake():
        if sleepable:
            state["sleeping"] = False
        return await health()

    return app


def test_sim_pool_router_balances_and_proxies_binary_and_lifecycle() -> None:
    app_a = _make_fake_backend("h100", capacity=2, sleepable=False)
    app_b = _make_fake_backend("a100", capacity=4, sleepable=True)

    transport_a = httpx.ASGITransport(app=app_a)
    transport_b = httpx.ASGITransport(app=app_b)

    class _MuxTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.host == "backend-a":
                return await transport_a.handle_async_request(request)
            return await transport_b.handle_async_request(request)

    router = SimPoolRouter(
        environment="robotwin",
        backends=["http://backend-a:28552", "http://backend-b:28552"],
    )

    async def _run() -> None:
        router._client = httpx.AsyncClient(transport=_MuxTransport())
        try:
            h = await router.health()
            assert h["ok"] is True
            assert h["workers_target"] == 6
            assert h["workers_total"] == 6
            assert h["workers_idle"] == 6

            opened = [
                await router.open_session({"task": {}, "cohort_id": "c1"})
                for _ in range(6)
            ]
            backends_hit = [row["backend"] for row in opened]
            assert backends_hit.count("h100") == 2
            assert backends_hit.count("a100") == 4

            # Zero-Base64 binary /execute passthrough
            sid0 = opened[0]["session_id"]
            status, payload, media = await router.execute(
                sid0,
                json.dumps({"name": "observe", "arguments": {}}).encode(),
                BINARY_EXECUTE_MEDIA_TYPE,
            )
            assert status == 200
            assert media == BINARY_EXECUTE_MEDIA_TYPE
            meta, images = unpack_execute_response(payload)
            assert meta["text"].endswith(":observe")
            assert images == [b"\x89PNG-raw-" + opened[0]["backend"].encode()]

            # Close one session, then close cohort for the rest
            await router.close_session(sid0)
            closed = await router.close_cohort("c1")
            assert closed["closed"] == 5

            # Sleep sleeps only sleepable backend-b; backend-a stays warm
            slept = await router.lifecycle("sleep")
            assert slept["workers_target"] == 6
            assert slept["workers_total"] == 2
            assert slept["sleeping"] is True

            woke = await router.lifecycle("wake")
            assert woke["workers_total"] == 6
            assert woke["sleeping"] is False
        finally:
            await router.shutdown()

    asyncio.run(_run())


def test_sim_pool_router_transport_error_returns_503() -> None:
    import pytest
    from fastapi import HTTPException

    class _DisconnectTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.fail_open = False

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"ok": True, "workers_target": 2})
            if request.url.path == "/sessions/open":
                if self.fail_open:
                    raise httpx.RemoteProtocolError(
                        "Server disconnected without sending a response."
                    )
                return httpx.Response(200, json={"session_id": "s-1"})
            raise httpx.RemoteProtocolError(
                "Server disconnected without sending a response."
            )

    transport = _DisconnectTransport()
    router = SimPoolRouter(environment="robotwin", backends=["http://backend-a:28552"])

    async def _run() -> None:
        router._client = httpx.AsyncClient(transport=transport)
        try:
            await router.health()
            opened = await router.open_session({"task": {}, "cohort_id": "c1"})
            assert opened["session_id"] == "s-1"
            assert router._reserved["http://backend-a:28552"] == 1

            with pytest.raises(HTTPException) as exc_info:
                await router.execute("s-1", b"{}", "application/json")
            assert exc_info.value.status_code == 503

            transport.fail_open = True
            with pytest.raises(HTTPException) as exc_info2:
                await router.open_session({"task": {}, "cohort_id": "c1"})
            assert exc_info2.value.status_code == 503
            assert router._reserved["http://backend-a:28552"] == 1
        finally:
            await router.shutdown()

    asyncio.run(_run())


def test_sim_pool_sleep_prevents_replacement_respawns() -> None:
    """Verify sleep() sets _sleeping=True before closing active sessions so no workers respawn."""
    from unittest.mock import MagicMock

    from robots.rl.server import SimPoolManager, _WorkerHandle

    mgr = SimPoolManager(
        environment="robotwin",
        num_workers=2,
        sim_python="python",
        gpus=[0],
        sam3_endpoint=None,
        sleepable=True,
        max_episodes_per_worker=1,
    )
    spawn_calls = 0

    async def _fake_spawn(gpu):
        nonlocal spawn_calls
        spawn_calls += 1
        raise AssertionError("sleep() must not trigger worker respawns")

    mgr._spawn_worker = _fake_spawn  # type: ignore[method-assign]

    proc = MagicMock()
    proc.poll.return_value = None
    writer = MagicMock()
    worker = _WorkerHandle(0, 0, proc, MagicMock(), writer)
    mgr._workers.add(worker)
    mgr._sessions["s-1"] = worker
    mgr._session_cohorts["s-1"] = "c-1"

    async def _run() -> None:
        health = await mgr.sleep()
        await asyncio.sleep(0.05)
        assert health["sleeping"] is True
        assert health["workers_total"] == 0
        assert health["active_sessions"] == 0
        assert spawn_calls == 0

    asyncio.run(_run())
