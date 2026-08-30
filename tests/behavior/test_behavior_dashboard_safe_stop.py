from __future__ import annotations

import json
from pathlib import Path


class _SafeStopBackend:
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


def test_safe_stop_seals_non_success_receipt_without_motion(tmp_path: Path) -> None:
    from robots.behavior.dashboard import (
        BehaviorControlController,
        BehaviorDashboardState,
    )

    state = BehaviorDashboardState(run_id="radio-dev-smoke", output_dir=tmp_path)
    controller = BehaviorControlController(state=state, backend=_SafeStopBackend())
    state.bind_controller(controller)

    result = controller.stop(
        lease_id="bounded-smoke",
        reason="authorized_live_smoke_complete",
        stop_mode="safe_stop",
    )

    receipt = result["terminal_receipt"]
    assert receipt["kind"] == "behavior_dashboard_safe_stop_terminal_receipt"
    assert receipt["primitive_success"] is True
    assert receipt["motion_command_issued"] is False
    assert receipt["task_success"] is False
    assert receipt["raw_success_observed"] is False
    assert receipt["official_success_receipt"] is None
    assert receipt["total_env_steps"] == 0

    receipt_path = Path(result["terminal_receipt_path"])
    assert receipt_path == tmp_path / "terminal_receipt.json"
    assert json.loads(receipt_path.read_text("utf-8")) == receipt

    snapshot = state.snapshot()
    assert snapshot["progress"]["terminal_receipt_complete"] is True
    assert snapshot["progress"]["official_task_success"] is False
    assert snapshot["control"]["phase"] == "stopped"
    assert snapshot["control"]["available"] is False
    assert snapshot["control"]["last_terminal"] == receipt
