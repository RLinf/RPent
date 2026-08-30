from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from rpent.dashboard.state import DashboardState

REPO_ROOT = Path(__file__).resolve().parents[2]
BEHAVIOR_ROOT = REPO_ROOT / "robots" / "behavior"
pytestmark = pytest.mark.skipif(
    not BEHAVIOR_ROOT.is_dir(),
    reason="BEHAVIOR robot plugin has not landed in this worktree yet",
)


def _source(relative: str) -> str:
    return (REPO_ROOT / relative).read_text("utf-8")


def _dashboard_spec():
    return {
        "task": {
            "command": "/rpent-behavior-task",
            "usage": "/rpent-behavior-task <task_name> <public_seed>",
            "fields": (
                {"name": "task_name", "suggestions": ("picking_up_trash",)},
                {"name": "public_seed", "kind": "integer", "minimum": 0},
            ),
            "display": "{task_name} / seed {public_seed}",
            "output_slug": "{task_name}_s{public_seed}",
        },
        "runtime_components": (
            {"name": "env", "label": "ENV", "scope": "unique"},
            {"name": "vla", "label": "VLA", "scope": "shared"},
        ),
        "frame_channels": (
            {"name": "head", "label": "head", "legacy_path_key": "head_path"},
            {
                "name": "left_wrist",
                "label": "left wrist",
                "legacy_path_key": "left_wrist_path",
            },
            {
                "name": "right_wrist",
                "label": "right wrist",
                "legacy_path_key": "right_wrist_path",
            },
        ),
    }


def test_harness_source_documents_no_main_explore_and_fresh_attempt_invocations():
    source = _source("robots/behavior/harness.py")

    assert "Each attempt is a separate standard RPent process" in source
    assert "rpent --robot behavior --behavior-mode explore --output-dir" in source
    assert "never passes main ``--explore``" in source
    assert '"--explore"' in source
    assert "attempt_dir" in source
    assert "subprocess.run(" in source
    assert "shell=False" in source


def test_dashboard_state_accepts_behavior_three_camera_spec_and_task_commands(tmp_path):
    state = DashboardState(
        run_id="behavior-dashboard-test",
        output_dir=tmp_path,
        dashboard_spec=_dashboard_spec(),
    )
    state.shared_services_ready()

    request = state.submit_input("/rpent-behavior-task picking_up_trash 3")

    assert request == {"task_name": "picking_up_trash", "public_seed": 3}
    claimed = state.wait_for_task(timeout=0)
    assert claimed is not None
    assert claimed.request == request
    assert claimed.output_dir.name == "0001_picking_up_trash_s3"


def test_dashboard_static_js_keeps_keyboard_and_frame_channel_markers() -> None:
    source = _source("rpent/dashboard/static/dashboard.js")

    for marker in (
        "Enter to send",
        "Shift+Enter",
        "Esc to interrupt",
        "frameChannelLabel",
        "renderFrameTabs",
        "mediaState.unavailableKind !== mediaState.kind",
    ):
        assert marker in source


def test_behavior_robot_spec_exposes_manual_control_dashboard_contract() -> None:
    from robots.behavior.robot_spec import get_robot_spec

    spec = get_robot_spec().dashboard

    assert spec is not None
    assert spec["behavior_control"]["targets"] == ("chassis", "left_arm", "right_arm")
    assert spec["behavior_control"]["pipeline"] == (
        "prepare",
        "execute",
        "discard",
        "capture",
        "stop",
    )
    assert spec["behavior_control"]["cameras"] == (
        "head",
        "left_wrist",
        "right_wrist",
    )


def test_behavior_add_cli_args_sets_two_hour_dashboard_defaults() -> None:
    import argparse

    from robots.behavior import runtime

    parser = argparse.ArgumentParser()
    parser.add_argument("--planner-timeout-s", type=int, default=None)
    runtime.add_cli_args(parser, use_dashboard=True)
    args = parser.parse_args([])

    assert args.max_episode_steps == 43200
    assert args.planner_timeout_s == 7200


def test_behavior_importable_modules_are_current_new_standard_only() -> None:
    for module_name in (
        "episode_memory_index",
        "episode_memory_merge",
        "harness",
        "prompt_bundle",
    ):
        assert importlib.import_module(f"robots.behavior.{module_name}")

    for removed in (
        "serial_explore",
        "candidate_explore",
        "legacy_dino_episode_memory",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"robots.behavior.{removed}")
