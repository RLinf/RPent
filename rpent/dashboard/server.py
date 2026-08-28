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

"""Dashboard HTTP server for the live monitor.

The FastAPI app runs on an OS-picked free port inside a daemon thread, so it can
sit alongside the agent run loop and keep serving after the run finishes, until
the process is stopped.

Routes mirror the fixed frontend contract in ``rpent/dashboard/index.html``.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from rpent.dashboard.interaction import (
    DashboardMessageConflictError,
    InteractionUnavailableError,
    UnknownDashboardMessageError,
)
from rpent.dashboard.state import DashboardState


class DashboardServer:
    """Threaded FastAPI server exposing the dashboard API for registered runs."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        runs_dir: str = "",
        language: str = "en",
        dashboard_spec: dict[str, Any],
    ) -> None:
        self.host = host
        self.port = int(port)
        self.runs_dir = runs_dir
        self._dashboard_spec = dashboard_spec
        dashboard_dir = Path(__file__).parent
        self._language = "zh-cn" if language == "zh-cn" else "en"
        self._index_html = (dashboard_dir / "index.html").read_text(encoding="utf-8")
        self._static_dir = dashboard_dir / "static"
        self._state: DashboardState | None = None
        self._app = self._build_app()
        self._server: uvicorn.Server | None = None

        # -- launcher state: when armed, the frontend shows a start screen and
        # the run loop blocks in wait_for_launch() until the user clicks Run.
        self._launch_enabled = False
        self._launch_defaults: dict[str, Any] = {}
        self._launch_config: dict[str, Any] | None = None
        self._launch_event = threading.Event()

    def register(self, state: DashboardState) -> None:
        if self._state is not None and self._state is not state:
            raise ValueError("Dashboard Session is already registered")
        self._state = state
        if not self.runs_dir:
            self.runs_dir = str(state.output_dir.parent)

    def _resolve(self, state_id: str) -> DashboardState | None:
        state = self._state
        if state is None or state.run_id != state_id:
            return None
        return state

    def wait_for_launch(self, defaults: dict[str, Any]) -> dict[str, Any]:
        """Arm the launcher and block until the user submits the start form.

        ``defaults`` pre-fills the launcher form (typically ``vars(args)``).
        Returns the config the user confirmed via ``POST /api/launch/run``.
        """
        self._launch_defaults = dict(defaults)
        self._launch_enabled = True
        self._launch_event.wait()
        return self._launch_config or {}

    def start(self) -> str:
        """Launch uvicorn in a daemon thread; return the base URL once serving."""
        port = self.port or _free_port(self.host)
        config = uvicorn.Config(
            self._app, host=self.host, port=port, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        threading.Thread(target=self._server.run, daemon=True).start()

        t0 = time.time()
        while not self._server.started and time.time() - t0 < 10:
            time.sleep(0.05)
        if not self._server.started:
            raise RuntimeError(f"dashboard server did not start on {self.host}:{port}")
        self.port = port
        return f"http://{self.host}:{port}"

    # -- routes ------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="RPent dashboard")
        app.mount(
            "/static",
            StaticFiles(directory=self._static_dir),
            name="dashboard-static",
        )

        @app.get("/")
        def index() -> HTMLResponse:
            html = self._index_html.replace(
                "__DASHBOARD_LANGUAGE__",
                self._language,
            )
            return HTMLResponse(html)

        @app.get("/healthz")
        def healthz() -> JSONResponse:
            return JSONResponse({"ok": True})

        @app.get("/api/commands")
        def api_commands() -> JSONResponse:
            return JSONResponse(self._dashboard_spec)

        @app.get("/api/launch/state")
        def api_launch_state() -> JSONResponse:
            return JSONResponse(
                {
                    "enabled": self._launch_enabled,
                    "pending": self._launch_enabled and not self._launch_event.is_set(),
                    "defaults": self._launch_defaults,
                }
            )

        @app.post("/api/launch/run")
        def api_launch_run(payload: dict[str, Any] = Body(default={})) -> JSONResponse:
            if not self._launch_enabled:
                return JSONResponse({"error": "launcher not armed"}, status_code=409)
            if self._launch_event.is_set():
                return JSONResponse({"error": "already launched"}, status_code=409)
            self._launch_config = {
                **self._launch_defaults,
                **payload,
            }
            self._launch_event.set()
            return JSONResponse({"ok": True})

        @app.get("/api/runs")
        def api_runs() -> JSONResponse:
            state = self._state
            return JSONResponse(
                {
                    "runs_dir": self.runs_dir,
                    "runs": [] if state is None else [state.run_info()],
                }
            )

        @app.post("/api/sessions/{session_id:path}/messages")
        def api_submit_message(
            session_id: str,
            payload: dict[str, Any] = Body(default={}),
        ) -> JSONResponse:
            live = self._resolve(session_id)
            if live is None:
                return JSONResponse({"error": "unknown session"}, status_code=404)
            try:
                live.submit_input(payload.get("text"))
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=422)
            except InteractionUnavailableError as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)
            return JSONResponse({"ok": True}, status_code=202)

        @app.delete(
            "/api/sessions/{session_id:path}/messages/{message_id}",
        )
        def api_withdraw_message(
            session_id: str,
            message_id: str,
        ) -> JSONResponse:
            live = self._resolve(session_id)
            if live is None:
                return JSONResponse({"error": "unknown session"}, status_code=404)
            try:
                live.withdraw_message(message_id)
            except UnknownDashboardMessageError as exc:
                return JSONResponse({"error": str(exc)}, status_code=404)
            except (DashboardMessageConflictError, InteractionUnavailableError) as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)
            return JSONResponse({"ok": True})

        @app.post("/api/sessions/{session_id:path}/interrupt")
        def api_interrupt(session_id: str) -> JSONResponse:
            live = self._resolve(session_id)
            if live is None:
                return JSONResponse({"error": "unknown session"}, status_code=404)
            try:
                result = live.request_interrupt()
            except InteractionUnavailableError as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)
            if result == "noop":
                return JSONResponse(
                    {
                        "status": "noop",
                        "interrupt_requested": False,
                    }
                )
            return JSONResponse(
                {
                    "status": "requested",
                    "interrupt_requested": True,
                    "deduplicated": result == "duplicate",
                },
                status_code=202,
            )

        @app.get("/api/run")
        def api_run(run: str) -> JSONResponse:
            live = self._resolve(run)
            if live is None:
                return JSONResponse({"error": "unknown run"}, status_code=404)
            return JSONResponse(live.run_detail())

        @app.get("/api/run/transcript")
        def api_transcript(run: str, since: int = 0) -> JSONResponse:
            live = self._resolve(run)
            events = live.events_since(since) if live else []
            return JSONResponse({"events": events})

        @app.get("/api/run/frame")
        def api_frame(
            run: str,
            kind: str = self._dashboard_spec["frame_channels"][0]["name"],
            t: str = "",
        ) -> Response:
            live = self._resolve(run)
            try:
                png = live.frame(kind) if live else None
            except ValueError as exc:
                return JSONResponse({"detail": str(exc)}, status_code=422)
            if png is None:
                return Response(status_code=404)
            return Response(png, media_type="image/png")

        @app.get("/api/run/video")
        def api_video(run: str) -> Response:
            live = self._resolve(run)
            if live is None or not live.has_video():
                return Response(status_code=404)
            return FileResponse(
                live.video_path,
                media_type="video/mp4",
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        @app.get("/api/run/action-video")
        def api_action_video(run: str, step: int) -> Response:
            live = self._resolve(run)
            path = live.action_video_path(step) if live else None
            if path is None:
                return Response(status_code=404)
            return FileResponse(
                path,
                media_type="video/mp4",
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        @app.get("/api/stream")
        def api_stream(run: str) -> StreamingResponse:
            async def gen():
                while True:
                    live = self._resolve(run)
                    if live is not None:
                        yield f"data: {json.dumps(live.snapshot())}\n\n"
                    else:
                        yield ": keepalive\n\n"
                    await asyncio.sleep(0.1)

            return StreamingResponse(gen(), media_type="text/event-stream")

        return app


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])
