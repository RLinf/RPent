# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.yam.tasks import classify_episode
from robots.yam.toolkit import YamToolkit


def ready():
    return {
        "episode_id": "a",
        "take_action_cnt": 3,
        "step_lim": 100,
        "stop_requested": False,
        "ready_for_motion": True,
        "eval_success": False,
    }


@pytest.mark.parametrize(
    "change,reason",
    [
        ({}, "ready"),
        ({"stop_requested": True}, "stopped"),
        ({"ready_for_motion": False}, "awaiting_ready"),
        ({"take_action_cnt": 100}, "budget_exhausted"),
        ({"terminal_event": "abort"}, "abort"),
        ({"terminal_event": "failure"}, "failure"),
        ({"eval_success": True}, "success"),
    ],
)
def test_rules(change, reason):
    s = {**ready(), **change}
    result = classify_episode(s)
    assert result["reason"] == reason
    assert result["can_continue"] == (reason == "ready")
    assert all(result[k] == v for k, v in s.items())


def test_continuation_bound_and_episode_isolation():
    t = object.__new__(YamToolkit)
    t._mode = "exploration"
    t._state = SimpleNamespace(latest_record=lambda: None)
    t._continuation_key = None
    t._continuation_count = 0
    s = ready()
    t.status = lambda: classify_episode(s)
    assert all(t.exploration_continuation() for _ in range(3))
    assert t.exploration_continuation() is None
    s["take_action_cnt"] += 1
    assert t.exploration_continuation()
    s["stop_requested"] = True
    assert t.exploration_continuation() is None
    s["stop_requested"] = False
    t._state = SimpleNamespace(
        latest_record=lambda: SimpleNamespace(command={"action": "finish"})
    )
    assert t.exploration_continuation() is None


@pytest.mark.parametrize("requested_status", ["success", "SUCCESS", " success "])
def test_pending_reset_and_finish_do_not_end_or_spend_attempt(
    toolkit_factory, receipt, clock, requested_status
):
    toolkit = toolkit_factory()
    before = clock.now
    pending = toolkit.execute_tool("reset", {})
    assert pending.result["status"] == "pending" and not pending.is_finish
    assert toolkit._session_attempt == 0 and clock.now - before == pytest.approx(20)
    receipt("ready")
    reset = toolkit.execute_tool("reset", {})
    assert reset.result["log"]["result"]["attempt"] == 1
    before = clock.now
    pending = toolkit.execute_tool(
        "finish", {"status": requested_status, "summary": "unverified claim"}
    )
    assert pending.result["status"] == "pending" and not pending.is_finish
    assert not toolkit.solved() and toolkit._session_attempt == 1
    assert clock.now - before == pytest.approx(20)


@pytest.mark.parametrize("stopped", [False, True])
def test_diagnostic_handoff_counts_only_a_continuable_episode(
    toolkit_factory, ready_client, receipt, clock, stopped
):
    if stopped:
        ready_client.request_stop()
    toolkit = toolkit_factory()
    assert toolkit._session_attempt == (0 if stopped else 1)
    if stopped:
        pending = toolkit.execute_tool("reset", {})
        assert pending.result["status"] == "pending"
        assert toolkit._session_attempt == 0
        receipt("ready")
        reset = toolkit.execute_tool("reset", {})
        assert reset.result["log"]["result"]["attempt"] == 1
        assert toolkit._session_attempt == 1


@pytest.mark.parametrize(
    "event,finished,status",
    [
        ("failure", False, "retry"),
        ("abort", True, "failure"),
        ("success", True, "success"),
    ],
)
def test_finish_uses_current_operator_result(
    toolkit_factory, ready_client, receipt, event, finished, status
):
    toolkit = toolkit_factory()
    receipt(event)
    result = toolkit.execute_tool(
        "finish", {"status": "success", "summary": "agent claim"}
    )
    assert result.is_finish is finished
    assert result.result["status"] == status
    assert toolkit.solved() is (event == "success")
    if event != "success":
        assert toolkit.write_recipe("unverified") == ""


def test_stale_receipt_not_reused_after_reset(ready_client, env, receipt):
    old = env._episode_id
    receipt("success")
    assert ready_client.read_control_state()[1]["episode_status"]["eval_success"]
    receipt("ready")
    ready_client.reset()
    receipt("success", episode_id=old)
    current = ready_client.read_control_state()[1]["episode_status"]
    assert current["episode_id"] != old and not current["eval_success"]
    receipt("abort")
    assert ready_client.read_control_state()[1]["episode_status"]["stop_requested"]


def test_current_episode_recipe_memory_merge_and_next_read(
    toolkit_factory, ready_client, env, receipt, tmp_path
):
    toolkit = toolkit_factory()
    assert "pi05_act" not in toolkit._tools
    assert {
        "move_to",
        "rotate_wrist",
        "set_gripper",
        "release",
        "reset",
    } <= toolkit._tools.keys()
    toolkit.execute_tool("set_gripper", {"arm": "left", "val": 0.25, "steps": 1})
    old_episode = env._episode_id
    receipt("ready")
    ready_client.reset()
    toolkit.execute_tool("release", {"arm": "right", "val": 1.0, "steps": 1})
    assert toolkit.write_recipe("unverified") == ""
    receipt("success")
    toolkit.execute_tool("render", {})
    tag = "yam_place_cube_s12"
    note = """---
scope: suite
suite: yam
regime: real
task_id: place_cube
task_language: place the cube
confidence: single-shot
evidence:
  cells: [yam_place_cube_s12]
---
Use the left gripper to stabilize the object before opening the right gripper.
"""
    inbox = tmp_path / "memory" / "_internal" / "inbox" / tag / "suite_technique.md"
    written = toolkit.execute_tool(
        "write_text_file", {"path": str(inbox), "content": note}
    )
    assert "error" not in written.result
    recipe = Path(toolkit.write_recipe(tag)).read_text()
    assert '"action": "release"' in recipe and '"action": "set_gripper"' not in recipe
    audit = json.loads((tmp_path / "run" / f"{tag}.json").read_text())
    assert old_episode not in recipe
    assert audit["solved"] and audit["episode_status"]["episode_id"] == env._episode_id
    toolkit.close()
    merged = toolkit.memory.merge_memory(
        cell_tag=tag, run_state_dir=tmp_path / "run", solved=True
    )
    assert merged["suite"] == 1 and merged["task"] == 1
    assert (tmp_path / "memory" / "MEMORY.md").exists()
    reader = toolkit_factory(mode="evaluation")
    published = tmp_path / "memory" / "suite" / "suite_yam_real_tplace_cube.md"
    recalled = reader.execute_tool("read_text_file", {"path": str(published)})
    assert "stabilize the object" in recalled.result["content"]
    recipe_read = reader.execute_tool(
        "read_text_file",
        {"path": str(tmp_path / "memory" / "task_only" / f"{tag}_recipe.jsonl")},
    )
    assert '"action": "release"' in recipe_read.result["content"]
    assert (
        "error"
        in reader.execute_tool(
            "write_text_file", {"path": str(inbox), "content": "overwrite"}
        ).result
    )


@pytest.mark.parametrize(
    "recoverable,steps,included", [(True, 2, True), (False, 2, False), (True, 0, False)]
)
def test_recipe_filters_rejected_waypoints(
    toolkit_factory, ready_client, receipt, monkeypatch, recoverable, steps, included
):
    toolkit = toolkit_factory()
    monkeypatch.setattr(
        toolkit._primitives,
        "move_to",
        lambda **kwargs: {
            "success": False,
            "recoverable": recoverable,
            "executed_steps": steps,
        },
    )
    toolkit.execute_tool("move_to", {"arm": "right", "xyz": [0, 0, 0]})
    receipt("success")
    toolkit.execute_tool("render", {})
    text = Path(toolkit.write_recipe("solved")).read_text()
    assert ('"action": "move_to"' in text) == included
