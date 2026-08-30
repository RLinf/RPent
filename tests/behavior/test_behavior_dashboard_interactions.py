from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROLS_JS = (
    REPO_ROOT / "robots" / "behavior" / "dashboard" / "static" / "behavior_controls.js"
)
CONTROLS_CSS = (
    REPO_ROOT / "robots" / "behavior" / "dashboard" / "static" / "behavior_controls.css"
)


class _ObserveOnlyBackend:
    def dashboard_control_capabilities(self):
        return {
            "motion_available": False,
            "observe_available": True,
            "unavailable_reason": "manual_motion_unavailable",
        }

    def dashboard_safe_stop(self, *, reason: str, stop_mode: str):
        return {
            "status": "ok",
            "stopped": True,
            "reason": reason,
            "stop_mode": stop_mode,
            "primitive_success": True,
            "task_success": False,
            "official_success_source": 'info["done"]["success"]',
            "official_success_receipt": None,
            "motion_command_issued": False,
            "total_env_steps": 0,
        }


class _PreparedBackend(_ObserveOnlyBackend):
    def __init__(self):
        self.discarded: list[dict[str, str]] = []

    def dashboard_control_capabilities(self):
        return {
            "motion_available": True,
            "observe_available": True,
            "unavailable_reason": "",
        }

    def dashboard_prepare_manual_command(self, **kwargs):
        return {"status": "ok", "plan_id": "prepared-plan"}

    def dashboard_discard_prepared_command(self, **kwargs):
        self.discarded.append(
            {
                "command_id": str(kwargs.get("command_id") or ""),
                "plan_id": str(kwargs.get("plan_id") or ""),
            }
        )
        return {"status": "discarded"}


def _request(url: str, *, payload: dict[str, object] | None = None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        return response.status, response.read()


def test_behavior_local_controls_keep_keyboard_pointer_and_release_safety() -> None:
    source = CONTROLS_JS.read_text("utf-8")

    for marker in (
        'window.addEventListener("keydown", handleKeyDown)',
        'window.addEventListener("keyup", handleKeyUp)',
        "event.repeat",
        "isEditableTarget(event.target)",
        'button.addEventListener("pointercancel"',
        'button.addEventListener("lostpointercapture"',
        'window.addEventListener("blur"',
        'window.addEventListener("pagehide"',
        'document.addEventListener("visibilitychange"',
        'requestInteractionStop("visibility_hidden")',
        'event.key === "Escape"',
        "setControlsExpanded",
        "controlTooltip",
        "updateControlTooltips",
        "motion_unavailable_reason",
        "observe_unavailable_reason",
        "setButtonTooltip",
        'button.removeAttribute("title")',
        "Refresh the currently selected camera view.",
        "Move the chassis forward by 5 cm. Hold to continue.",
        'requestInteractionStop("controls_collapsed")',
        "postCameraSelection(camera)",
        'fetch("/api/run/control/camera"',
        "captureViews();",
        "safe-stop receipt: task_success=",
        "requestPlannerInterrupt",
        "/interrupt",
        'terminal.task_success === true ? "true" : "false"',
        'terminal.command_id || terminal.kind || "terminal"',
    ):
        assert marker in source
    assert "button.dataset.tooltip = text" in source

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable for JavaScript syntax validation")
    subprocess.run([node, "--check", str(CONTROLS_JS)], check=True)


def test_behavior_controls_toggle_font_does_not_change_when_collapsed() -> None:
    css = CONTROLS_CSS.read_text("utf-8")
    match = re.search(r"\.controls-toggle\s*\{(?P<body>.*?)\n\}", css, re.DOTALL)

    assert match is not None
    declarations = match.group("body")
    assert "font-size: 12px;" in declarations
    assert "line-height: 1.15;" in declarations


def test_behavior_dashboard_http_keeps_three_cameras_buttons_and_stop_receipt(
    tmp_path: Path,
) -> None:
    from robots.behavior.dashboard import create_server

    run_id = "behavior-dashboard/http-contract"
    server, _state = create_server(
        host="127.0.0.1",
        port=0,
        output_dir=tmp_path,
        run_id=run_id,
        control_backend=_ObserveOnlyBackend(),
    )
    base_url = server.start()
    try:
        status, html_bytes = _request(base_url + "/")
        assert status == 200
        html = html_bytes.decode("utf-8")
        assert '<div class="col right behavior-dashboard">' in html
        for marker in (
            '<div class="framewrap behavior-mode" id="framewrap">',
            'class="control-rail control-left" id="interactiveControls"',
            'class="control-rail control-left"',
            '<div class="frame-stage">',
            'class="control-rail control-right"',
            'class="controls-toggle collapsed-toggle"',
            '<div class="frame-tabs behavior-frame-tabs" aria-label="Camera view">',
            'data-kind="head" data-camera="head" class="active"',
            'data-kind="left_wrist" data-camera="left_wrist"',
            'data-kind="right_wrist" data-camera="right_wrist"',
            "Interactive Controls",
            'data-target="chassis"',
            'data-target="left_arm"',
            'data-target="right_arm"',
            'data-action="forward"',
            'data-action="turn_left"',
            'data-action="turn_right"',
            'data-action="backward"',
            'data-action="observe"',
            'data-action="up"',
            'data-action="down"',
            'data-action="rotate_left"',
            'data-action="rotate_right"',
            'data-action="open"',
            'data-action="close"',
            "/behavior-static/behavior_controls.js",
            "/behavior-static/behavior_controls.css",
        ):
            assert marker in html
        assert html.count('class="frame-tabs behavior-frame-tabs"') == 1
        assert html.count('class="frame-tabs legacy-frame-tabs"') == 1
        tooltip_control_buttons = re.findall(
            r'<button class="(?:control-button|target-button)[^"]*"[^>]*>',
            html,
        )
        assert len(tooltip_control_buttons) == 14
        assert all(
            re.search(r'data-tooltip="[^"]+"', item) for item in tooltip_control_buttons
        )
        assert all("title=" not in item for item in tooltip_control_buttons)
        for hidden_pipeline_label in (
            ">Prepare</button>",
            ">Execute</button>",
            ">Discard</button>",
            ">Capture</button>",
            ">Safe stop</button>",
        ):
            assert hidden_pipeline_label not in html

        status, js_bytes = _request(base_url + "/behavior-static/behavior_controls.js")
        assert status == 200
        assert b"handleKeyDown" in js_bytes

        status, camera_bytes = _request(
            base_url + "/api/run/control/camera",
            payload={"run": run_id, "camera": "left_wrist"},
        )
        assert status == 200
        assert json.loads(camera_bytes)["selected_camera"] == "left_wrist"

        status, css_bytes = _request(
            base_url + "/behavior-static/behavior_controls.css"
        )
        assert status == 200
        css = css_bytes.decode("utf-8")
        for marker in (
            ".framewrap.behavior-mode",
            "grid-template-columns: minmax(168px, .52fr) minmax(260px, 1fr) minmax(168px, .5fr)",
            ".control-left",
            ".control-right",
            ".dpad::before",
            ".round-button",
            ".behavior-frame-tabs",
            ".function-grid",
            ".observe-wrap",
            ".controls-collapsed",
        ):
            assert marker in css
        assert ".control-button::after" in css
        assert ".target-button::after" in css
        assert "content: attr(data-tooltip)" in css
        assert '.control-button[data-tooltip=""]::after' in css

        status, receipt_bytes = _request(
            base_url + "/api/run/control/stop",
            payload={
                "run": run_id,
                "lease_id": "http-contract",
                "reason": "test_complete",
                "stop_mode": "safe_stop",
            },
        )
        assert status == 200
        result = json.loads(receipt_bytes)
        receipt = result["terminal_receipt"]
        assert receipt["motion_command_issued"] is False
        assert receipt["task_success"] is False
        assert receipt["raw_success_observed"] is False

        status, state_bytes = _request(
            base_url + "/api/run/control/state?run=" + run_id.replace("/", "%2F")
        )
        assert status == 200
        snapshot = json.loads(state_bytes)
        assert snapshot["last_terminal"] == receipt
        assert snapshot["last_terminal"]["task_success"] is False
    finally:
        server.stop(timeout_s=5)


def test_standard_dashboard_entry_uses_behavior_control_server_and_state() -> None:
    from robots.behavior.dashboard import (
        BehaviorDashboardServer,
        BehaviorDashboardState,
    )
    from robots.behavior.robot_spec import get_robot_spec
    from rpent.cli.dashboard import _dashboard_server_and_state_classes

    spec = get_robot_spec()
    server_cls, state_cls = _dashboard_server_and_state_classes(spec, spec.dashboard)

    assert server_cls is BehaviorDashboardServer
    assert state_cls is BehaviorDashboardState
    assert "classes" in spec.dashboard


def test_behavior_dashboard_state_ingests_frame_paths_from_observe(
    tmp_path: Path,
) -> None:
    from robots.behavior.dashboard import BehaviorDashboardState
    from rpent.dashboard.events import ToolResultEvent

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ"
        "/pLvAAAAAElFTkSuQmCC"
    )
    frame_dir = tmp_path / "dashboard_captures"
    frame_dir.mkdir()
    frame_paths = {}
    for camera in ("head", "left_wrist", "right_wrist"):
        path = frame_dir / f"observe_1_{camera}.png"
        path.write_bytes(png)
        frame_paths[camera] = str(path)

    state = BehaviorDashboardState(
        run_id="behavior-dashboard/frames", output_dir=tmp_path
    )
    state.emit(
        ToolResultEvent(
            name="observe",
            result={
                "name": "observe",
                "status": "ok",
                "step": 12,
                "frames": frame_paths,
            },
        )
    )

    assert state.frame("head") == png
    assert state.frame("left_wrist") == png
    assert state.frame("right_wrist") == png


def test_standard_dashboard_cli_selects_behavior_control_state(tmp_path: Path) -> None:
    from robots.behavior.dashboard import BehaviorDashboardState
    from robots.behavior.robot_spec import BEHAVIOR_DASHBOARD_SPEC
    from rpent.cli.dashboard import (
        _bind_robot_dashboard_backend,
        _unbind_robot_dashboard_backend,
    )

    state = BehaviorDashboardState(
        run_id="behavior-dashboard/bind-contract",
        output_dir=tmp_path,
        dashboard_spec=BEHAVIOR_DASHBOARD_SPEC,
    )
    _bind_robot_dashboard_backend(state, {"env": _ObserveOnlyBackend()})

    controller = state.control_controller()
    assert controller is not None
    snapshot = controller.state()
    assert snapshot["available"] is True
    assert snapshot["observe_available"] is True

    _unbind_robot_dashboard_backend(state)
    controller = state.control_controller()
    assert controller is None
    assert state.run_detail()["control"]["unavailable_reason"] == "controller_not_bound"


def test_behavior_dashboard_unbind_discards_prepared_command(tmp_path: Path) -> None:
    from robots.behavior.dashboard import (
        BehaviorControlController,
        BehaviorDashboardState,
    )
    from rpent.dashboard.events import RunStartedEvent

    backend = _PreparedBackend()
    state = BehaviorDashboardState(
        run_id="behavior-dashboard/unbind", output_dir=tmp_path
    )
    state.emit(RunStartedEvent())
    controller = BehaviorControlController(state=state, backend=backend)
    state.bind_controller(controller)

    prepared = controller.prepare(
        lease_id="unbind-test",
        sequence=1,
        target="chassis",
        action="forward",
        camera="head",
    )
    controller.unbind_backend()

    assert backend.discarded == [
        {"command_id": prepared["command_id"], "plan_id": "prepared-plan"}
    ]
    snapshot = controller.state()
    assert snapshot["available"] is False
    assert snapshot["unavailable_reason"] == "backend_not_bound"
