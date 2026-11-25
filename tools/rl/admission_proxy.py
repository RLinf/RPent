#!/usr/bin/env python3
# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Limit concurrent requests per evaluation replica and report admission waits in SSE."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from rpent.planner.admission import HEADER, PREFIX, WAIT_HEARTBEAT

logger = logging.getLogger(__name__)


def create_app(
    upstreams: list[str], transport=None, *, max_in_flight: int = 1
) -> FastAPI:
    """Own an independent request limit for each upstream replica."""
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be positive")
    gates = [asyncio.Semaphore(max_in_flight) for _ in upstreams]
    chat_connections = len(upstreams) * max_in_flight
    limits = httpx.Limits(
        max_connections=chat_connections + len(upstreams),
        max_keepalive_connections=chat_connections,
    )

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(
            transport=transport, timeout=14400, limits=limits
        ) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    def upstream(index: int) -> str:
        if not 0 <= index < len(upstreams):
            raise HTTPException(404, "Unknown replica")
        return upstreams[index].rstrip("/")

    @app.get("/replica/{index}/v1/models")
    async def models(index: int):
        response = await app.state.client.get(upstream(index) + "/models")
        return Response(
            response.content,
            status_code=response.status_code,
            media_type="application/json",
        )

    @app.post("/replica/{index}/v1/chat/completions")
    async def chat(index: int, request: Request):
        url = upstream(index) + "/chat/completions"
        body = await request.body()
        if json.loads(body).get("stream") is not True:
            raise HTTPException(400, "Admission accounting requires streamed requests")

        async def stream():
            started = time.monotonic()
            yield PREFIX + b'{"state":"waiting"}\n\n'
            acquire = asyncio.create_task(gates[index].acquire())
            try:
                while not acquire.done():
                    done, _ = await asyncio.wait({acquire}, timeout=10)
                    if not done:
                        yield WAIT_HEARTBEAT + b"\n\n"
                await acquire
                wait = time.monotonic() - started
                yield (
                    PREFIX
                    + json.dumps({"state": "admitted", "queue_seconds": wait}).encode()
                    + b"\n\n"
                )
                try:
                    async with app.state.client.stream(
                        "POST",
                        url,
                        content=body,
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        if response.is_error:
                            raw = await response.aread()
                            try:
                                error = json.loads(raw)
                            except ValueError:
                                error = {
                                    "message": f"{response.status_code} {response.reason_phrase}: {raw.decode(errors='replace')}"
                                }
                            if isinstance(error, dict):
                                error = error.get("error", error)
                            if not isinstance(error, dict):
                                error = {"message": str(error)}
                            error["status_code"] = response.status_code
                            yield (
                                b"data: "
                                + json.dumps({"error": error}).encode()
                                + b"\n\n"
                            )
                            return
                        async for chunk in response.aiter_bytes():
                            yield chunk
                except Exception as error:
                    logger.exception("Replica %s model request failed", index)
                    payload = {
                        "error": {"type": "model_service_error", "message": str(error)}
                    }
                    yield b"data: " + json.dumps(payload).encode() + b"\n\n"
            finally:
                if not acquire.done():
                    acquire.cancel()
                elif not acquire.cancelled() and acquire.exception() is None:
                    gates[index].release()

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={HEADER: "1"}
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream",
        action="append",
        required=True,
        help="Replica /v1 URL; repeat in stable order",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=1,
        help="Maximum concurrent requests sent to each replica",
    )
    args = parser.parse_args()
    import uvicorn

    app = create_app(args.upstream, max_in_flight=args.max_in_flight)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
