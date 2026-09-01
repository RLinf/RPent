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
import mimetypes
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
from rpent.dashboard.state import (
    DashboardState,
    PrimitiveArgumentError,
    PrimitiveConfigError,
)
from rpent.utils.daemon import pick_free_port
from rpent.utils.logging import get_logger

logger = get_logger("dashboard_server")


def _infer_frame_media_type(artifact: str) -> str:
    """Infer a Dashboard frame response type from its artifact filename."""
    media_type, _ = mimetypes.guess_type(artifact)
    if media_type is None or not media_type.startswith("image/"):
        raise ValueError(
            f"Dashboard frame artifact must have a recognized image suffix: {artifact!r}"
        )
    return media_type


class DashboardServer:
    """Threaded FastAPI server exposing one Dashboard Session."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        language: str = "en",
        state: DashboardState,
    ) -> None:
        self.host = host
        self.port = int(port)
        self._state = state
        dashboard_spec = state.dashboard_spec
        self._frame_media_types = {
            channel["name"]: _infer_frame_media_type(channel["artifact"])
            for channel in dashboard_spec["frame_channels"]
        }
        dashboard_dir = Path(__file__).parent
        self._language = "zh-cn" if language == "zh-cn" else "en"
        self._index_html = (dashboard_dir / "index.html").read_text(encoding="utf-8")
        self._static_dir = dashboard_dir / "static"
        self._app = self._build_app()
        self._server: uvicorn.Server | None = None

    def start(self) -> str:
        """Launch uvicorn in a daemon thread; return the base URL once serving."""
        port = self.port or pick_free_port(self.host)
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
            return JSONResponse(self._state.dashboard_spec)

        @app.post("/api/session/messages")
        def api_submit_message(
            payload: dict[str, Any] = Body(default={}),
        ) -> JSONResponse:
            try:
                self._state.submit_input(payload.get("text"))
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=422)
            except InteractionUnavailableError as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)
            return JSONResponse({"ok": True}, status_code=202)

        @app.delete(
            "/api/session/messages/{message_id}",
        )
        def api_withdraw_message(
            message_id: str,
        ) -> JSONResponse:
            try:
                self._state.withdraw_message(message_id)
            except UnknownDashboardMessageError as exc:
                return JSONResponse({"error": str(exc)}, status_code=404)
            except (DashboardMessageConflictError, InteractionUnavailableError) as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)
            return JSONResponse({"ok": True})

        @app.post("/api/session/interrupt")
        def api_interrupt() -> JSONResponse:
            try:
                result = self._state.request_interrupt()
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

        @app.get("/api/session/state")
        def api_session_state(timeline_since: int = 0) -> JSONResponse:
            return JSONResponse(
                self._state.session_detail(timeline_since=timeline_since)
            )

        @app.get("/api/session/primitives")
        def api_primitives() -> JSONResponse:
            try:
                primitives = self._state.primitive_specs()
            except InteractionUnavailableError as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)
            return JSONResponse({"primitives": primitives})

        @app.post("/api/session/primitive")
        def api_execute_primitive(
            payload: Any = Body(default=None),
        ) -> JSONResponse:
            if not isinstance(payload, dict):
                return JSONResponse(
                    {"error": "request body must be a JSON object"},
                    status_code=422,
                )
            name = payload.get("name")
            arguments = payload.get("arguments")
            if not isinstance(name, str) or not name.strip():
                return JSONResponse(
                    {"error": "name must be a non-empty string"},
                    status_code=422,
                )
            if not isinstance(arguments, dict):
                return JSONResponse(
                    {"error": "arguments must be a JSON object"},
                    status_code=422,
                )
            try:
                tool_result = self._state.execute_primitive(name, arguments)
            except InteractionUnavailableError as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)
            except PrimitiveArgumentError as exc:
                return JSONResponse({"error": str(exc)}, status_code=422)
            except PrimitiveConfigError as exc:
                logger.error("Dashboard primitive configuration error: %s", exc)
                return JSONResponse({"error": str(exc)}, status_code=500)
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=403)
            result = tool_result.result
            if isinstance(result, dict) and result.get("error") is not None:
                error = " ".join(str(result["error"]).split())[:500]
                return JSONResponse(
                    {"error": error or "primitive execution failed"},
                    status_code=422,
                )
            return JSONResponse({"ok": True})

        @app.get("/api/session/transcript")
        def api_transcript(since: int = 0) -> JSONResponse:
            return JSONResponse({"events": self._state.events_since(since)})

        @app.get("/api/session/frame")
        def api_frame(
            kind: str = self._state.dashboard_spec["frame_channels"][0]["name"],
        ) -> Response:
            try:
                frame = self._state.frame(kind)
            except ValueError as exc:
                return JSONResponse({"detail": str(exc)}, status_code=422)
            if frame is None:
                return Response(status_code=404)
            return Response(frame, media_type=self._frame_media_types[kind])

        @app.get("/api/session/video")
        def api_video() -> Response:
            if not self._state.has_video():
                return Response(status_code=404)
            return FileResponse(
                self._state.video_path,
                media_type="video/mp4",
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        @app.get("/api/session/action-video")
        def api_action_video(step: int) -> Response:
            path = self._state.action_video_path(step)
            if path is None:
                return Response(status_code=404)
            return FileResponse(
                path,
                media_type="video/mp4",
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        @app.get("/api/session/stream")
        def api_stream() -> StreamingResponse:
            async def gen():
                version = -1
                while True:
                    version, snapshot = await asyncio.to_thread(
                        self._state.wait_for_snapshot,
                        version,
                        timeout=15.0,
                    )
                    if snapshot is None:
                        yield ": keepalive\n\n"
                    else:
                        yield f"data: {json.dumps(snapshot)}\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        return app
