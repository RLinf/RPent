from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import pytest

from rpent.tools.common import finish

REPO_ROOT = Path(__file__).resolve().parents[2]
BEHAVIOR_ROOT = REPO_ROOT / "robots" / "behavior"
pytestmark = pytest.mark.skipif(
    not BEHAVIOR_ROOT.is_dir(),
    reason="BEHAVIOR robot plugin has not landed in this worktree yet",
)


def _module(name: str):
    return importlib.import_module(f"robots.behavior.{name}")


def test_common_finish_cannot_forge_behavior_official_success() -> None:
    result = finish(status="success", summary="operator text")

    assert result == {
        "_finish": True,
        "status": "success",
        "summary": "operator text",
    }
    for forbidden in (
        "task_success",
        "official_success_source",
        "official_success_receipt",
        "info_done",
    ):
        assert forbidden not in result


def test_explore_harness_rejects_core_owned_rpent_flags() -> None:
    harness = _module("harness")

    with pytest.raises(ValueError, match="outer harness owns"):
        harness._normalize_passthrough(["--robot", "behavior"])
    with pytest.raises(ValueError, match="outer harness owns"):
        harness._normalize_passthrough(["--explore"])
    with pytest.raises(ValueError, match="outer harness owns"):
        harness._normalize_passthrough(["--output-dir=/tmp/x"])


def test_explore_harness_attempt_argv_uses_standard_behavior_mode(tmp_path) -> None:
    harness = _module("harness")

    argv = harness._attempt_argv(
        rpent_executable="rpent",
        attempt_dir=tmp_path / "attempt_001",
        passthrough=["--task-name", "picking_up_trash", "--public-seed", "0"],
    )

    assert argv[:6] == [
        "rpent",
        "--robot",
        "behavior",
        "--behavior-mode",
        "explore",
        "--output-dir",
    ]
    assert "--explore" not in argv
    assert argv[-4:] == ["--task-name", "picking_up_trash", "--public-seed", "0"]


def test_explore_harness_dry_run_creates_one_fresh_invocation_per_attempt(tmp_path):
    harness = _module("harness")
    args = argparse.Namespace(
        attempts=2,
        output_dir=tmp_path / "outer",
        rpent_executable="rpent",
        cwd=None,
        timeout_s=None,
        stop_on_explicit_success=True,
        dry_run=True,
    )

    assert harness.run_explore(args, ["--task-name", "picking_up_trash"]) == 0
    summary = (tmp_path / "outer" / "explore_harness_summary.json").read_text("utf-8")

    assert '"attempts_run": 2' in summary
    assert "attempt_001" in summary
    assert "attempt_002" in summary
    assert "--behavior-mode" in summary
    assert "--explore" not in summary


def test_explore_harness_success_detection_uses_explicit_terminal_receipts() -> None:
    harness = _module("harness")

    assert harness._explicit_success([{"task_success": True}]) is True
    assert harness._explicit_success([{"official_success": True}]) is True
    assert harness._explicit_success([{"primitive_success": True}]) is False
    assert harness._explicit_success([{"status": "success"}]) is False


def test_shared_component_exports_stay_lightweight() -> None:
    exported = importlib.import_module("rpent.robots.components")

    assert "BaseEnvClient" in exported.__all__
    assert "BaseVLAFacade" in exported.__all__
    assert "Sam3Engine" not in exported.__all__
    assert "Pi05VLAFacade" not in exported.__all__
