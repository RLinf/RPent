from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from rpent.dashboard.events import (
    RunFinishedEvent,
    RunStartedEvent,
    RuntimeStatusEvent,
    ToolResultEvent,
)
from rpent.dashboard.server import DashboardServer
from rpent.dashboard.state import DashboardState


class DashboardStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.state = DashboardState(
            run_id="test-run",
            name="test",
            suite="suite",
            task=1,
            seed=2,
            output_dir=str(self.output_dir),
            video_path=str(self.output_dir / "episode.mp4"),
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _frame_file(self, name: str, data: bytes) -> Path:
        path = self.output_dir / name
        path.write_bytes(data)
        return path

    def test_camera_and_wrist_frames_are_stored(self) -> None:
        camera = self._frame_file("camera.png", b"camera")
        wrist = self._frame_file("wrist.png", b"wrist")

        self.state.emit(
            ToolResultEvent(
                "view_driver_state",
                {
                    "step": 3,
                    "image_cam_path": str(camera),
                    "image_wrist_path": str(wrist),
                },
            )
        )

        self.assertEqual(self.state.frame("camera"), b"camera")
        self.assertEqual(self.state.frame("wrist"), b"wrist")
        self.assertEqual(self.state.snapshot()["frame_idx"], 3)

    def test_unrelated_image_fields_do_not_replace_realtime_frames(self) -> None:
        camera = self._frame_file("camera.png", b"camera")
        wrist = self._frame_file("wrist.png", b"wrist")
        overlay = self._frame_file("overlay.png", b"overlay")
        self.state.emit(
            ToolResultEvent(
                "view_driver_state",
                {
                    "step": 1,
                    "image_cam_path": str(camera),
                    "image_wrist_path": str(wrist),
                },
            )
        )

        self.state.emit(
            ToolResultEvent(
                "segment",
                {
                    "step": 1,
                    "camera": "wrist",
                    "image_path": str(wrist),
                    "overlay_path": str(overlay),
                },
            )
        )

        self.assertEqual(self.state.frame("camera"), b"camera")
        self.assertEqual(self.state.frame("wrist"), b"wrist")

    def test_new_snapshot_clears_missing_source_and_old_step_is_ignored(self) -> None:
        camera_1 = self._frame_file("camera-1.png", b"camera-1")
        wrist_1 = self._frame_file("wrist-1.png", b"wrist-1")
        camera_2 = self._frame_file("camera-2.png", b"camera-2")
        self.state.emit(
            ToolResultEvent(
                "view_driver_state",
                {
                    "step": 1,
                    "image_cam_path": str(camera_1),
                    "image_wrist_path": str(wrist_1),
                },
            )
        )
        self.state.emit(
            ToolResultEvent(
                "view_driver_state",
                {"step": 2, "image_cam_path": str(camera_2)},
            )
        )
        self.state.emit(
            ToolResultEvent(
                "view_driver_state",
                {
                    "step": 1,
                    "image_cam_path": str(camera_1),
                    "image_wrist_path": str(wrist_1),
                },
            )
        )

        self.assertEqual(self.state.frame("camera"), b"camera-2")
        self.assertIsNone(self.state.frame("wrist"))
        self.assertEqual(
            self.state.snapshot()["frame_available"],
            {"camera": True, "wrist": False},
        )

    def test_run_lifecycle_and_terminal_state_is_idempotent(self) -> None:
        self.assertEqual(self.state.snapshot()["state"], "starting")

        self.state.emit(RuntimeStatusEvent("env", "ready"))
        self.assertEqual(self.state.snapshot()["state"], "starting")

        self.state.emit(RunStartedEvent())
        self.assertEqual(self.state.snapshot()["state"], "running")

        self.state.emit(
            RunFinishedEvent(
                state="succeeded",
                reason="completed",
                terminated=False,
            )
        )
        detail = self.state.run_detail()
        self.assertEqual(detail["state"], "succeeded")
        self.assertFalse(detail["terminated"])

        self.state.emit(
            RunFinishedEvent(
                state="failed",
                reason="late_error",
                error="must not replace terminal state",
            )
        )
        self.assertEqual(self.state.snapshot()["state"], "succeeded")

    def test_failed_start_serializes_reason_and_error(self) -> None:
        self.state.emit(
            RunFinishedEvent(
                state="failed",
                reason="runtime_initialization",
                error=RuntimeError("VLA failed"),
            )
        )

        snapshot = self.state.snapshot()
        self.assertEqual(snapshot["state"], "failed")
        self.assertEqual(snapshot["finish_reason"], "runtime_initialization")
        self.assertEqual(snapshot["error"], "VLA failed")

    def test_invalid_frame_kind_and_terminal_state_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.state.frame("agent")
        with self.assertRaises(ValueError):
            self.state.emit(RunFinishedEvent(state="done"))


class DashboardServerTest(unittest.TestCase):
    def test_frame_api_validates_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            camera = output_dir / "camera.png"
            wrist = output_dir / "wrist.png"
            camera.write_bytes(b"camera")
            wrist.write_bytes(b"wrist")
            state = DashboardState(
                run_id="api-run",
                name="api",
                suite="suite",
                task=0,
                seed=0,
                output_dir=str(output_dir),
                video_path=str(output_dir / "episode.mp4"),
            )
            state.emit(
                ToolResultEvent(
                    "view_driver_state",
                    {
                        "step": 1,
                        "image_cam_path": str(camera),
                        "image_wrist_path": str(wrist),
                    },
                )
            )
            server = DashboardServer()
            server.register(state)

            with TestClient(server._app) as client:
                self.assertEqual(
                    client.get(
                        "/api/run/frame",
                        params={"run": "api-run", "kind": "camera"},
                    ).content,
                    b"camera",
                )
                self.assertEqual(
                    client.get(
                        "/api/run/frame",
                        params={"run": "api-run", "kind": "wrist"},
                    ).content,
                    b"wrist",
                )
                self.assertEqual(
                    client.get(
                        "/api/run/frame",
                        params={"run": "api-run", "kind": "agent"},
                    ).status_code,
                    422,
                )
                detail = client.get(
                    "/api/run",
                    params={"run": "api-run"},
                ).json()
                self.assertEqual(
                    detail["frame_available"],
                    {"camera": True, "wrist": True},
                )


if __name__ == "__main__":
    unittest.main()
