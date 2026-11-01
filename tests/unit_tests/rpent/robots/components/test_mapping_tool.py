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

"""Offline tests for the robot-agnostic ``build_map`` tool flow."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.robots.components.lingbotmap_server import (
    FrameArchiveBackend,
    LingBotMapFacade,
)
from rpent.session import EnvState
from rpent.tools.mapping import MappingMixin
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import init_output_dir


class _LocalRpcClient:
    """Dispatch calls directly into a facade's registered RPC methods."""

    def __init__(self, facade: Any) -> None:
        self._facade = facade

    def call(
        self, method: str, args: tuple = (), kwargs: dict | None = None, **_: Any
    ) -> Any:
        handler = getattr(self._facade, "_rpc")[method]
        return handler(*(args or ()), **(kwargs or {}))


class _FakePerception:
    """One fixed detection per label, for the labeling path."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def detect_segment(self, image: Any, *, labels: list[str], min_score: float = 0.3):
        from rpent.robots.components.gdino_sam2_client import (
            Detection,
            DetectSegmentResult,
        )

        self.calls.append(list(labels))
        if isinstance(image, np.ndarray):
            frame = np.asarray(image)
        else:
            frame = np.asarray(Image.open(io.BytesIO(bytes(image))).convert("RGB"))
        detections = tuple(
            Detection(
                label=label,
                score=0.8,
                box=[1.0, 2.0, 30.0, 20.0],
                mask=frame[..., 0] > 0,
            )
            for label in labels
        )
        return DetectSegmentResult(
            found=True,
            detections=detections,
            image_shape=(frame.shape[0], frame.shape[1]),
        )


class _MappingToolkit(MappingMixin, Toolkit):
    """Minimal toolkit exercising the mixin without a robot."""

    def __init__(
        self,
        output_dir: Path,
        *,
        operator_reply: str | None = "done",
        frame_count: int = 5,
        perception: Any = None,
        localize: Any = None,
    ) -> None:
        init_output_dir(output_dir / "log")
        state = EnvState(output_dir / "state")
        super().__init__(
            dashboard_events=NullDashboardEventSink(),
            state=state,
            memory=MemoryManager(output_dir / "memory"),
        )
        self._lingbotmap_client = self._make_map_client(output_dir)
        self._gdino_sam2_client = perception
        self._localize_client = localize
        self._operator_reply = operator_reply
        self._frame_count = frame_count
        self._frame_interval_s = 0.05
        self._asked: list[str] = []
        self.register_mapping_tools()

    def _make_map_client(self, output_dir: Path) -> Any:
        from rpent.robots.components.lingbotmap_client import LingBotMapClient

        facade = LingBotMapFacade(FrameArchiveBackend(str(output_dir / "maps")))
        return LingBotMapClient(_LocalRpcClient(facade))

    def _map_frame(self) -> tuple[Any, list[float] | None, float]:
        index = getattr(self, "_emitted", 0)
        self._emitted = index + 1
        frame = np.full((12, 16, 3), 40 + index * 10, dtype=np.uint8)
        if index >= self._frame_count:
            time.sleep(0.05)  # let the poller idle after frames run out
        return frame, [0.0, 0.0, 0.05 * index, 0, 0, 0], 100.0 + index

    def _ask_operator(self, prompt: str, *, kind: str | None = None) -> str | None:
        self._asked.append(prompt)
        return self._operator_reply

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ):
        from rpent.tools import ToolResult

        return ToolResult(data={"command": command, **result})


class TestBuildMapTool:
    def test_build_map_runs_full_lifecycle(self, tmp_path: Path) -> None:
        perception = _FakePerception()
        toolkit = _MappingToolkit(tmp_path, perception=perception)
        result = toolkit.execute_tool(
            "build_map", {"map_name": "lab", "labels": ["desk"]}
        )
        data = result.to_dict()
        assert data.get("success"), data
        assert data["map_name"] == "lab"
        assert data["frames_pushed"] >= 1
        assert (Path(data["map_path"]) / "map.json").is_file()
        assert data["labels"]["desk"]["score"] == pytest.approx(0.8)
        assert perception.calls, "labeling never ran"
        assert any("Teleoperate" in prompt for prompt in toolkit._asked)

    def test_build_map_without_labels_skips_perception(self, tmp_path: Path) -> None:
        perception = _FakePerception()
        toolkit = _MappingToolkit(tmp_path, perception=perception)
        result = toolkit.execute_tool("build_map", {"map_name": "lab"})
        data = result.to_dict()
        assert data.get("success"), data
        assert data["labels"] == {}
        assert perception.calls == []

    def test_labels_refused_without_perception_service(self, tmp_path: Path) -> None:
        toolkit = _MappingToolkit(tmp_path, perception=None)
        result = toolkit.execute_tool(
            "build_map", {"map_name": "lab", "labels": ["desk"]}
        )
        assert "error" in result.to_dict()
        assert "Grounding DINO" in result.to_dict()["error"]

    def test_empty_map_name_is_refused(self, tmp_path: Path) -> None:
        toolkit = _MappingToolkit(tmp_path)
        result = toolkit.execute_tool("build_map", {"map_name": "  "})
        assert "error" in result.to_dict()

    def test_zero_pushed_frames_reports_error(self, tmp_path: Path) -> None:
        toolkit = _MappingToolkit(tmp_path, frame_count=0)

        # A source that always raises produces no pushed frames.
        def broken_frame() -> tuple[Any, list[float] | None, float]:
            raise RuntimeError("camera offline")

        toolkit._map_frame = broken_frame  # type: ignore[method-assign]
        result = toolkit.execute_tool("build_map", {"map_name": "lab"})
        data = result.to_dict()
        assert "error" in data
        assert data["frames_pushed"] == 0


class TestDetectObjectsTool:
    @staticmethod
    def _record_with_frame(toolkit: _MappingToolkit, *, with_frame: bool) -> None:
        with toolkit.state.record_step(state={"step_idx": 0}) as step:
            if with_frame:
                toolkit.state.save(
                    "front.png",
                    np.full((12, 16, 3), 200, dtype=np.uint8),
                    step=step,
                )

    def test_detect_objects_reads_persisted_frame(self, tmp_path: Path) -> None:
        perception = _FakePerception()
        toolkit = _MappingToolkit(tmp_path, perception=perception)
        self._record_with_frame(toolkit, with_frame=True)
        result = toolkit.execute_tool("detect_objects", {"labels": ["mug"], "step": -1})
        data = result.to_dict()
        assert data.get("found"), data
        assert data["detections"][0]["label"] == "mug"

    def test_detect_objects_requires_a_frame(self, tmp_path: Path) -> None:
        perception = _FakePerception()
        toolkit = _MappingToolkit(tmp_path, perception=perception)
        self._record_with_frame(toolkit, with_frame=False)
        result = toolkit.execute_tool("detect_objects", {"labels": ["mug"]})
        assert "error" in result.to_dict()


class TestMappingRegistration:
    def test_detect_objects_absent_without_perception_client(
        self, tmp_path: Path
    ) -> None:
        toolkit = _MappingToolkit(tmp_path, perception=None)
        names = {tool.name for tool in toolkit.list_tools()}
        assert "build_map" in names
        assert "detect_objects" not in names

    def test_build_map_schema_is_object_typed(self, tmp_path: Path) -> None:
        toolkit = _MappingToolkit(tmp_path)
        build_map = next(t for t in toolkit.list_tools() if t.name == "build_map")
        schema = build_map.input_schema
        assert schema["type"] == "object"
        assert set(schema.get("required", [])) == {"map_name"}


class _FakeLocalize:
    """Return a scripted clearance result and record each call."""

    def __init__(
        self,
        *,
        status: str = "clear",
        nearest: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> None:
        self.status = status
        self.nearest = nearest
        self.reason = reason
        self.calls: list[dict[str, Any]] = []

    def check_path(self, image: Any, **kwargs: Any) -> Any:
        from rpent.tools.clearance import ClearanceResult

        self.calls.append({"image": image, **kwargs})
        located = self.status not in {"unlocalized", "unusable"}
        pose = np.eye(4) if located else None
        if pose is not None:
            pose[:3, :3] = np.asarray(
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            )
            pose[0, 3] = 0.5
        return ClearanceResult(
            status=self.status,
            unit="camera_heights",
            length=2.0,
            half_width=0.75,
            heading_deg=0.0,
            min_height=0.05,
            max_height=4.0,
            frame="model",
            pose=pose,
            keyframe=2 if located else None,
            camera_height=0.2,
            meters_per_unit=None,
            unit_scale=5.0 if located else None,
            nearest=self.nearest,
            observed_distance=None,
            coverage=1.0 if located else None,
            point_count=100,
            blocked_points=3 if self.status == "blocked" else 0,
            coverage_bins=[True] * 12 if located else [],
            latency_ms=4.0,
            reason=self.reason,
            localization={"status": "lost", "reason": self.reason},
        )


class TestCheckClearanceTool:
    def test_tool_needs_the_localize_client(self, tmp_path: Path) -> None:
        toolkit = _MappingToolkit(tmp_path, localize=None)
        names = {tool.name for tool in toolkit.list_tools()}

        assert "check_clearance" not in names

        toolkit = _MappingToolkit(tmp_path, localize=_FakeLocalize())
        names = {tool.name for tool in toolkit.list_tools()}

        assert "check_clearance" in names

    def test_forwards_the_live_frame_and_reuses_the_prior(self, tmp_path: Path) -> None:
        localize = _FakeLocalize()
        toolkit = _MappingToolkit(tmp_path, localize=localize)

        first = toolkit.execute_tool(
            "check_clearance",
            {
                "heading_deg": 15.0,
                "length": 3.0,
                "half_width": 1.0,
                "unit": "map_units",
            },
        ).to_dict()
        second = toolkit.execute_tool("check_clearance", {}).to_dict()

        assert first["success"] is True
        assert first["status"] == "clear"
        assert first["used_prior"] is False
        assert second["used_prior"] is True

        first_call, second_call = localize.calls
        assert first_call["image"].shape == (12, 16, 3)
        assert first_call["heading_deg"] == pytest.approx(15.0)
        assert first_call["length"] == pytest.approx(3.0)
        assert first_call["half_width"] == pytest.approx(1.0)
        assert first_call["unit"] == "map_units"
        assert first_call["exhaustive"] is True
        assert second_call["exhaustive"] is False
        assert second_call["prior_index"] == 2
        assert np.allclose(second_call["prior_pose"], np.asarray(first["pose"]))

    def test_blocked_result_is_a_successful_check(self, tmp_path: Path) -> None:
        localize = _FakeLocalize(
            status="blocked",
            nearest={
                "distance": 1.5,
                "lateral": 0.2,
                "height": 0.4,
                "label": "chair",
                "object_id": "obj_0001",
            },
        )
        toolkit = _MappingToolkit(tmp_path, localize=localize)

        data = toolkit.execute_tool("check_clearance", {}).to_dict()

        assert data["success"] is True
        assert data["status"] == "blocked"
        assert data["nearest"]["label"] == "chair"

    def test_unlocalized_result_tells_the_model_to_turn(self, tmp_path: Path) -> None:
        localize = _FakeLocalize(status="unlocalized", reason="too few inliers")
        toolkit = _MappingToolkit(tmp_path, localize=localize)
        toolkit._clearance_prior_index = 2
        toolkit._clearance_prior_pose = np.eye(4)

        data = toolkit.execute_tool("check_clearance", {}).to_dict()

        assert data["success"] is False
        assert "too few inliers" in data["error"]
        assert "rotate" in data["error"]
        assert toolkit._clearance_prior_index is None
        assert toolkit._clearance_prior_pose is None

    def test_unusable_result_reports_the_map_reason(self, tmp_path: Path) -> None:
        localize = _FakeLocalize(
            status="unusable", reason="the map records no floor estimate"
        )
        toolkit = _MappingToolkit(tmp_path, localize=localize)

        data = toolkit.execute_tool("check_clearance", {}).to_dict()

        assert data["success"] is False
        assert data["error"] == "the map records no floor estimate"

    def test_service_failure_becomes_a_tool_error(self, tmp_path: Path) -> None:
        class _BrokenLocalize:
            def check_path(self, image: Any, **kwargs: Any) -> Any:
                raise RuntimeError("service offline")

        toolkit = _MappingToolkit(tmp_path, localize=_BrokenLocalize())

        data = toolkit.execute_tool("check_clearance", {}).to_dict()

        assert "clearance check failed" in data["error"]
        assert "service offline" in data["error"]
