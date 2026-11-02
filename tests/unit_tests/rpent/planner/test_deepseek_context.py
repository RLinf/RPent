# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Operator provenance, consumption evidence, and fresh-session handoff."""

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.dashboard.interaction import DashboardMessage
from rpent.planner.deepseek_context import (
    DeepSeekHandoffContext,
    consumed_operator_inputs,
)
from rpent.planner.deepseek_harness import _Recorder
from rpent.planner.deepseek_session import DeepSeekSessionDriver


def observe_user(recorder, receipt, text, sequence):
    recorder.observe(
        SimpleNamespace(
            method="session.event",
            payload={
                "event": {
                    "type": "user/message",
                    "seq": sequence,
                    "data": {
                        "id": receipt,
                        "role": "user",
                        "content": [{"type": "text", "text": text}],
                    },
                }
            },
        ),
        "opening-prompt",
    )


def test_consumed_inputs_require_positive_evidence_and_preserve_event_order():
    recorder = _Recorder(NullDashboardEventSink())
    observe_user(recorder, "opening-prompt", "generated handoff", 1)
    observe_user(recorder, "r1", "stay away from red cup", 4)
    observe_user(recorder, "r1", "stay away from red cup", 4)
    candidates = {
        "m1": {"receipt": "r1", "text": "stay away from red cup"},
        "m2": {"text": "earlier constraint, ACK lost"},
        "pending": {"receipt": "r3", "text": "never consumed"},
        "withdrawn": {"receipt": "r4", "text": "withdrawn"},
    }
    records = consumed_operator_inputs(
        candidates,
        recorder.consumed_user_receipts,
        {
            "r2": {"messageId": "m2", "status": "consumed", "eventSeq": 3},
            "r3": {"messageId": "pending", "status": "queued"},
            "r4": {"messageId": "withdrawn", "status": "discarded"},
            "foreign": {"messageId": "unknown", "status": "consumed"},
        },
    )
    assert [record["message_id"] for record in records] == ["m2", "m1"]
    assert len(recorder.messages) == 1
    assert "generated handoff" not in json.dumps(records)


def test_driver_keeps_original_operator_text_and_excludes_generated_context(tmp_path):
    candidates = {}
    host = Mock(request=AsyncMock())
    host.request.side_effect = [
        {"receipt": f"r{index}", "newTurn": False} for index in range(3)
    ]
    driver = DeepSeekSessionDriver(host, Mock(), operator_inputs=candidates)
    messages = [
        DashboardMessage("typed", "Move left once.", "sending"),
        DashboardMessage(
            "manual",
            "[manual execution records and observation]\nKeep away from the red cup.",
            "sending",
            source="dashboard_continuation",
            operator_text="Keep away from the red cup.",
        ),
        DashboardMessage(
            "automatic",
            "Continue the generated exploration workflow.",
            "sending",
            source="exploration_continuation",
        ),
    ]

    async def submit():
        for message in messages:
            await driver.submit(message)

    asyncio.run(submit())
    assert candidates == {
        "typed": {"text": "Move left once.", "receipt": "r0"},
        "manual": {"text": "Keep away from the red cup.", "receipt": "r1"},
    }
    context = DeepSeekHandoffContext(tmp_path)
    records = consumed_operator_inputs(candidates, {"r0": 1, "r1": 2}, {})
    context.merge({"consumed_operator_inputs": records}, 1)
    context.merge({"consumed_operator_inputs": records}, 2)
    for _ in range(2):
        prompt = context.append_to("Fresh initial task")
        assert prompt.count("Move left once.") == 1
        assert "not new commands to replay" in prompt
        assert "manual execution records" not in prompt
        assert "generated exploration workflow" not in prompt
    # Rendering historical text never submits another SDK command.
    assert host.request.await_count == 3
    assert len(json.loads(context.path.read_text())["consumed_operator_inputs"]) == 2
    assert not context.path.with_suffix(".json.tmp").exists()
    assert DeepSeekHandoffContext(tmp_path / "new-task").append_to("fresh") == "fresh"


@pytest.mark.parametrize("entry", ["cli", "dashboard"])
@pytest.mark.parametrize(
    "planner_name,persistence_failure",
    [
        ("deepseek_harness", None),
        ("api", None),
        ("deepseek_harness", "safe"),
        ("deepseek_harness", "unsafe"),
    ],
)
def test_host_handoff_preserves_confirmed_context_and_fails_closed(
    tmp_path, monkeypatch, entry, planner_name, persistence_failure
):
    from rpent.cli import dashboard as dashboard_cli
    from rpent.cli import main as main_cli
    from rpent.dashboard.state import DashboardState
    from rpent.planner.base import PlannerResult
    from rpent.robots import PromptBundle, RobotSpec, RunConfig

    constraint = "Never approach the red cup again."
    action = "Move left once."
    correction = "Keep the red cup restriction and stay at least 20 cm away."
    requests, builds = [], []
    output = tmp_path / "run"
    if persistence_failure:

        def fail_persistence(*args):
            raise OSError("operator context persistence failed")

        monkeypatch.setattr(DeepSeekHandoffContext, "merge", fail_persistence)

    class Toolkit:
        memory = None

        def solved(self):
            return False

        def close(self):
            pass

        def cancel_active_and_wait(self):
            pass

    class Planner:
        def solve(self, **kwargs):
            requests.append(kwargs)
            number = len(requests)
            if planner_name == "deepseek_harness" and number > 1:
                assert (output / "dsh-operator-context.json").exists()
            recorder = _Recorder(NullDashboardEventSink())
            observe_user(recorder, "opening-prompt", kwargs["user_message"], 1)
            texts = (
                [constraint, action]
                if number == 1
                else [correction]
                if number == 2
                else []
            )
            candidates = {"pending": {"receipt": "pending", "text": "UNCONSUMED INPUT"}}
            for index, text in enumerate(texts, 2):
                receipt = f"session-{number}-receipt-{index}"
                candidates[receipt] = {"receipt": receipt, "text": text}
                observe_user(recorder, receipt, text, index)
            return PlannerResult(
                messages=recorder.messages,
                stats={
                    "consumed_operator_inputs": consumed_operator_inputs(
                        candidates, recorder.consumed_user_receipts, {}
                    )
                },
                stop_info={
                    "reason": "budget",
                    "execution_certain": persistence_failure != "unsafe",
                    "result_delivery_complete": True,
                    "cleanup_complete": True,
                    "allow_new_session": True,
                },
            )

    def build(*args, **kwargs):
        builds.append(kwargs)
        return Planner()

    def arguments(parser, use_dashboard):
        parser.add_argument("--explore-sessions", type=int, default=3)
        parser.add_argument("--explore-attempts-per-session", type=int, default=2)

    spec = RobotSpec(
        name="libero",
        prompts=PromptBundle(
            system=lambda _: "system policy", user=lambda _: "initial task"
        ),
        add_cli_args=arguments,
        parse_config=lambda args: RunConfig(
            recipe_tag="handoff",
            output_dir=output,
            prompt_vars={},
            task_desc={"robot": "libero"},
        ),
        init_runtime=lambda *args: ([], {}),
        supports_exploration=True,
    )
    monkeypatch.setattr(main_cli, "enumerate_robots", lambda: ("libero",))
    monkeypatch.setattr(main_cli, "get_robot_spec", lambda name: spec)
    monkeypatch.setattr("rpent.memory.loading.prepare_run_memory", lambda *args: None)
    for module in (main_cli, dashboard_cli):
        monkeypatch.setattr(module, "get_toolkit", lambda *args, **kwargs: Toolkit())
        monkeypatch.setattr(module, "build_planner", build)
        monkeypatch.setattr(module, "_preflight_deepseek", lambda args: None)
    argv = [
        "--robot",
        "libero",
        "--planner",
        planner_name,
        "--model",
        "gpt-6.1-sol",
        "--reasoning-effort",
        "xhigh",
        "--output-dir",
        str(output),
        "--explore",
        "--max-turns",
        "17",
        "--planner-timeout-s",
        "123",
        "--memory-profile",
        "local",
    ]
    if entry == "cli":
        monkeypatch.setattr(
            main_cli,
            "start_interactive_reader",
            lambda inputs, **kw: inputs.put(kw["first_prompt_default"]),
        )
        monkeypatch.setattr(sys, "argv", ["rpent", *argv, "--tui"])
        assert main_cli.main() == (1 if persistence_failure else 0)
    else:
        state = DashboardState(
            output_dir=tmp_path / "dashboard",
            dashboard_spec={
                "task": {
                    "command": "/rpent-task",
                    "usage": "/rpent-task <seed>",
                    "fields": ({"name": "seed", "kind": "integer", "minimum": 0},),
                    "display": "seed {seed}",
                    "output_slug": "s{seed}",
                },
                "runtime_components": (),
                "primitives": (),
            },
        )
        state.shared_services_ready()
        state.submit_input("/rpent-task 0")
        claimed = state.wait_for_task(timeout=0)
        args = main_cli._build_argparser().parse_args(argv)
        args.robot_name, args.explore_sessions, args.explore_attempts_per_session = (
            "libero",
            3,
            2,
        )
        error = dashboard_cli._run_dashboard_task(
            args=args,
            robot_spec=spec,
            state=state,
            claimed=claimed,
            shared_runtime_kwargs={},
            unique_components=set(),
            session_root=tmp_path,
        )
        if persistence_failure:
            assert "operator context persistence failed" in error
            if persistence_failure == "unsafe":
                assert state._shutdown_requested
        else:
            assert error is None
    if persistence_failure:
        assert len(requests) == len(builds) == 1
        transcript = json.loads((output / "transcript_handoff.json").read_text())
        assert "operator context persistence failed" in transcript["error"]
        assert transcript["stop_info"]["execution_certain"] is (
            persistence_failure != "unsafe"
        )
        assert len(transcript["planner_sessions"]) == 1
        assert not (output / "dsh-operator-context.json").exists()
        return
    assert len(requests) == len(builds) == 3
    for index, request in enumerate(requests):
        prompt = request["user_message"]
        expected = int(planner_name == "deepseek_harness" and index > 0)
        assert prompt.count(constraint) == expected
        assert prompt.count(action) == expected
        assert prompt.count(correction) == int(expected and index == 2)
        assert "UNCONSUMED INPUT" not in prompt
        assert prompt.count("Previously consumed operator instructions") == expected
    assert all(request["max_turns"] == 17 for request in requests)
    assert all(build["planner_timeout_s"] == 123 for build in builds)
    assert all(
        build["model"] == "gpt-6.1-sol" and build["reasoning_effort"] == "xhigh"
        for build in builds
    )
    if planner_name == "deepseek_harness":
        saved = json.loads((output / "dsh-operator-context.json").read_text())
        assert [row["text"] for row in saved["consumed_operator_inputs"]] == [
            constraint,
            action,
            correction,
        ]
    else:
        assert not (output / "dsh-operator-context.json").exists()
