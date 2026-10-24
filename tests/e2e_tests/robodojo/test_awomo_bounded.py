# Copyright 2026 The RPent Authors.
"""Real Awomo inference and bounded public CLI mixed-control verification."""

import json
import os
import sys
from pathlib import Path

import pytest

from robots.robodojo.robot_spec import get_robot_spec
from robots.robodojo.toolkit import RoboDojoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from tests.e2e_tests.common import (
    ScriptedToolCall,
    parse_runtime_args,
    publish_check,
    run_scripted_policy_chain,
    runtime_phase,
)

pytestmark = [
    pytest.mark.timeout(1800),
    pytest.mark.skipif(
        os.environ.get("RPENT_RUN_AWOMO05_INTEGRATION") != "1",
        reason="requires real Awomo weights, RoboDojo and reserved GPUs",
    ),
]


def runtime_argv():
    return [
        "--task",
        os.environ.get("RPENT_ROBODOJO_TASK", "general_pickup"),
        "--vla-backend",
        "awomo05",
        "--policy-server-url",
        os.environ["RPENT_AWOMO05_POLICY_SERVER_URL"],
        "--xpolicylab-root",
        os.environ["RPENT_XPOLICYLAB_ROOT"],
        "--policy-python",
        sys.executable,
        "--max-episode-steps",
        "40",
    ]


def test_awomo_real_inference_continuation_takeover_and_reset(monkeypatch):
    output = Path(os.environ["RPENT_E2E_OUTPUT_DIR"]) / "awomo-component"
    spec = get_robot_spec()
    args = parse_runtime_args(spec, runtime_argv())
    events = []
    with runtime_phase(spec, args, output / "runtime", {"env", "vla"}) as runtime:
        policy, native = runtime["vla_client"], runtime["env"]
        observe, infer = policy.update_obs, policy.get_action

        def record_obs(obs, step):
            assert obs["instruction"] == native.get_task_language()
            events.append(("observe", step))
            return observe(obs, step)

        def record_infer():
            events.append(("infer", native.get_status()["step"]))
            return infer()

        monkeypatch.setattr(policy, "update_obs", record_obs)
        monkeypatch.setattr(policy, "get_action", record_infer)
        toolkit = None
        try:
            toolkit = RoboDojoToolkit(
                primitives_kwargs=runtime,
                dashboard_events=NullDashboardEventSink(),
                memory=MemoryManager(output / "memory"),
                state_output_dir=output / "states",
            )
            # The continuation/takeover case needs a remainder after two
            # single-action calls; its size comes from the actual model reply.
            first = toolkit.execute_tool("run_policy", {"max_steps": 1})
            assert not first.is_error, first.data
            chunk_size = toolkit.state.get().result["inferences"][0]["action_count"]
            if chunk_size < 3:
                pytest.skip(
                    "continuation/takeover probe requires a chunk of at least 3 actions"
                )
            assert first.data["log"]["result"]["executed_steps"] == 1
            second = toolkit.execute_tool("run_policy", {"max_steps": 1})
            assert not second.is_error, second.data
            assert second.data["log"]["result"]["executed_steps"] == 1
            assert second.data["log"]["result"]["inference_count"] == 0
            assert second.data["log"]["result"]["pending_actions"] == chunk_size - 2
            native.hold_step()
            assert not toolkit._primitives.env._pending
            result = toolkit.execute_tool("run_policy", {"max_steps": 1})
            assert not result.is_error, result.data
            assert [step for kind, step in events if kind == "infer"] == [0, 3]
            assert [step for kind, step in events if kind == "observe"] == list(
                range(5)
            )
            assert native.get_status()["step"] == 4
            native_success = native.is_success()
            toolkit._primitives.env.reset()
            result = toolkit.execute_tool("run_policy", {"max_steps": 1})
            assert not result.is_error, result.data
            assert native.get_status()["step"] == 1
            toolkit.execute_tool(
                "finish", {"status": "stuck", "summary": "bounded inference verified"}
            )
            publish_check(
                "robodojo_awomo_component",
                {
                    "events": events,
                    "executed_actions": 5,
                    "action_chunk_size": chunk_size,
                    "native_success": native_success,
                    "model_meta": policy.model_meta,
                },
            )
        finally:
            if toolkit is not None:
                toolkit.close()
            policy.close()


def test_awomo_public_cli_policy_primitive_policy_finish():
    output = Path(os.environ["RPENT_E2E_OUTPUT_DIR"]) / "awomo-cli"
    result = run_scripted_policy_chain(
        robot="robodojo",
        robot_argv=[
            *runtime_argv(),
            "--vla-agent",
        ],
        output_dir=output,
        before_action_calls=(
            ScriptedToolCall("view_env_state", {}),
            ScriptedToolCall("run_policy", {"max_steps": 2}),
            ScriptedToolCall("hold", {"steps": 1}),
        ),
        action=ScriptedToolCall("run_policy", {"max_steps": 2}),
        action_count_field="executed_steps",
        include_images=True,
    )
    assert result["post_action_new_image_count"] > 0, result
    manifest = json.loads((output / "states.json").read_text())
    controls = [
        step
        for step in manifest["steps"]
        if (step.get("command") or {}).get("action") in {"run_policy", "hold"}
    ]
    assert [step["command"]["action"] for step in controls] == [
        "run_policy",
        "hold",
        "run_policy",
    ]
    assert [step["result"]["executed_steps"] for step in controls] == [2, 1, 2]
    assert all(not step["result"].get("error") for step in controls)
    assert [step["state"]["control"]["step"] for step in controls] == [2, 3, 5]
    policies = [
        step["result"] for step in controls if step["command"]["action"] == "run_policy"
    ]
    # Each call starts inference at the confirmed step. Extra inferences
    # depend on the returned chunk size, not a fixed deploy.yml replan_steps.
    for start, report in zip((0, 3), policies, strict=True):
        expected = start
        for inference in report["inferences"]:
            assert inference["observation_step"] == expected
            assert inference["action_count"] > 0
            assert inference["info"]["upstream"]["latency_ms"] > 0
            expected += inference["action_count"]
        assert expected >= start + report["executed_steps"]
        assert report["pending_actions"] == expected - start - report["executed_steps"]
    transcript = json.loads((output / result["transcript"]).read_text())
    feedback = [
        json.loads(message["content"])
        for message in transcript["messages"]
        if message["role"] == "tool" and message["name"] in {"run_policy", "hold"}
    ]
    assert [item["execution"]["confirmed_actions"] for item in feedback] == [2, 1, 2]
    for item in feedback:
        assert "inferences" not in item and "inferences" not in item["log"]["result"]
    assert [
        item["log"]["result"]["inference_count"]
        for item in feedback
        if item["log"]["command"]["action"] == "run_policy"
    ] == [len(report["inferences"]) for report in policies]
    result["saved_inferences"] = sum(len(report["inferences"]) for report in policies)
    native_result = json.loads((output / "result.json").read_text())
    result["native_success"] = native_result["success"]
    assert not native_result["agent_error_present"], native_result
    result["definition"] = (
        "public CLI -> real Awomo -> primitive -> real Awomo -> finish"
    )
    publish_check("robodojo_awomo_public_cli", result)
