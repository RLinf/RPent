# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""RGB-only, one-action turns and native automatic completion on CPU."""

import copy
import json
from types import SimpleNamespace

import pytest
from pydantic_ai.models.function import DeltaThinkingPart, DeltaToolCall, FunctionModel

from robots.rl.prompt_bundle import system_prompt
from robots.rl.rl_episode import episode_outcome, parse_args
from robots.rl.robot_spec import configure_planner
from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner.api_loop import ApiAgentLoop
from tests.unit_tests.robots.rl.test_coordinates import (  # noqa: F401
    CalibratedEnv,
    make,
    offline,
)


def solve(toolkit, replies, *, max_turns=10):
    requests = []

    async def stream(messages, info):
        requests.append(copy.deepcopy(messages))
        assert {t.name for t in info.function_tools} == {"move_to", "set_gripper"}
        yield replies[len(requests) - 1]

    planner = ApiAgentLoop(
        model=FunctionModel(stream_function=stream),
        dashboard_events=NullDashboardEventSink(),
        max_tokens=100,
        timeout_s=10,
        compaction=None,
        sequential_tools=True,
    )
    configure_planner(
        planner,
        SimpleNamespace(
            prompt_vars={"workflow": "pear", "reasoning": toolkit.reasoning}
        ),
    )
    result = planner.solve(
        system_prompt=system_prompt(
            {
                "workflow": "pear",
                "world_cord": toolkit.world_cord,
                "reasoning": toolkit.reasoning,
            }
        ),
        user_message="task",
        toolkit=toolkit,
        max_turns=max_turns,
    )
    return result, requests


def call(name="move_to", **kwargs):
    return {
        0: DeltaToolCall(
            name=name, json_args=json.dumps(kwargs or {"xyz": [0.1, -0.2, 0.95]})
        )
    }


@pytest.mark.parametrize("ending", ["success", "steps", "initial"])
def test_native_end_stops_without_another_request_and_saves_rgb(tmp_path, ending):
    env = CalibratedEnv(success_below_z=0.99 if ending == "success" else None)
    if ending == "initial":
        env.terminated = True
    if ending == "steps":
        original = env.step

        def step(action):
            result = original(action)
            env.truncated = True
            return result

        env.step = step
    toolkit, _ = make(tmp_path, env=env, workflow="pear", world_cord="base")
    assert {t.name for t in toolkit.list_tools()} == {"move_to", "set_gripper"}
    result, requests = solve(toolkit, [call()])
    assert result.error is None
    assert len(requests) == (0 if ending == "initial" else 1)
    assert result.finish_result["terminated"] == (ending != "steps")
    assert result.finish_result["truncated"] == (ending == "steps")
    assert env.depth_calls == 0
    assert not list(toolkit.output_dir.rglob("*_world.npz"))
    for record in toolkit.state.records():
        assert "agentview.png" in record.artifacts and "wrist.png" in record.artifacts
    assert episode_outcome(result, toolkit, 10)["terminal"]


@pytest.mark.parametrize(
    "reply",
    [
        "done",
        {0: DeltaToolCall(name="finish", json_args="{}")},
        {0: DeltaToolCall(name="move_to", json_args='{"xyz":[1,2]}')},
        {
            0: DeltaToolCall(name="move_to", json_args='{"xyz":[0,0,1]}'),
            1: DeltaToolCall(name="set_gripper", json_args='{"opening":0}'),
        },
    ],
)
def test_invalid_reply_is_one_failed_episode_and_executes_nothing(tmp_path, reply):
    toolkit, env = make(tmp_path, workflow="pear")
    result, requests = solve(toolkit, [reply])
    assert len(requests) == 1 and not env.actions
    assert result.error_kind == "invalid_action"
    assert result.stats["tool_calls"] == 0
    outcome = episode_outcome(result, toolkit, 10)
    assert outcome["terminal"] and outcome["termination_reason"] == "invalid_action"


def test_reasoning_off_rejects_thinking_and_prompt_requests_direct_call(tmp_path):
    toolkit, env = make(tmp_path, workflow="pear", reasoning=False)
    reply = {
        0: DeltaThinkingPart(content="reason"),
        1: DeltaToolCall(name="set_gripper", json_args='{"opening":0}'),
    }
    result, requests = solve(toolkit, [reply])
    assert len(requests) == 1 and not env.actions
    assert result.error_kind == "invalid_action"
    assert "directly" in str(requests[0])


def test_request_budget_and_partial_motion_need_no_finish(tmp_path):
    toolkit, _ = make(tmp_path, workflow="pear", reasoning=False)
    result, requests = solve(toolkit, [call("set_gripper", opening=1)] * 2, max_turns=2)
    assert result.error is None and len(requests) == 2
    assert result.stats["tool_calls"] == 2
    assert episode_outcome(result, toolkit, 2)["termination_reason"] == "request_limit"


def test_reasoning_aliases_parse_without_models_or_services(tmp_path):
    args = ["--output-dir", str(tmp_path), "--request", str(tmp_path / "request.json")]
    assert parse_args(args + ["--reasoning"]).no_thinking is False
    assert parse_args(args + ["--no-reasoning"]).no_thinking is True
    assert parse_args(args).no_thinking is None


def test_pear_runtime_connects_only_env_even_when_sam_is_requested(
    tmp_path, monkeypatch
):
    import threading

    from robots.rl.robot_spec import _init_runtime
    from rpent.utils.rpc import RpcFacade
    from rpent.utils.rpc.http_rpc import HttpRpcServer

    env = CalibratedEnv()
    calls = []
    facade = RpcFacade()
    meta = {"suite": "libero_goal", "task": 0, "seed": 0, "max_episode_steps": 10000}
    facade._rpc.update(
        {"env.get_env_meta": lambda: meta, "env.reset": lambda: (env.last_obs, {})}
    )

    def dispatch(method, args, kwargs):
        calls.append(method)
        return facade._dispatch(method, args, kwargs)

    server = HttpRpcServer(("127.0.0.1", 0), dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--request",
            str(tmp_path / "request.json"),
            "--workflow",
            "pear",
            "--world-cord",
            "general",
        ]
    )
    args.suite, args.task, args.seed = "libero_goal", 0, 0
    args.env_endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    args.sam3_endpoint = "http://127.0.0.1:1"
    monkeypatch.setattr(
        "robots.rl.robot_spec._start_sam3",
        lambda *args: pytest.fail("pear must not start or connect SAM3"),
    )
    try:
        owned, runtime = _init_runtime(
            args, tmp_path, NullDashboardEventSink(), {"env", "sam3"}
        )
        assert not owned and set(runtime) == {"env"}
        assert "env.reset" in calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


@pytest.mark.parametrize("reasoning", [False, True])
def test_reasoning_switch_reaches_actual_qwen_http_body(tmp_path, reasoning):
    import asyncio

    import httpx
    from openai import AsyncOpenAI
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.providers.openai import OpenAIProvider

    from rpent.planner.qwen import QwenChatModel, qwen_model_settings

    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen3.5-4b",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            client = AsyncOpenAI(
                api_key="unused", base_url="http://offline/v1", http_client=http
            )
            model = QwenChatModel(
                "qwen3.5-4b", provider=OpenAIProvider(openai_client=client)
            )
            planner = ApiAgentLoop(
                model=model,
                dashboard_events=NullDashboardEventSink(),
                max_tokens=100,
                reasoning_effort="high",
                thinking=True,
                model_settings=qwen_model_settings(
                    max_tokens=100, reasoning_effort="high", thinking=True
                ),
            )
            configure_planner(
                planner,
                SimpleNamespace(
                    prompt_vars={"workflow": "pear", "reasoning": reasoning}
                ),
            )
            await model.request(
                [ModelRequest(parts=[UserPromptPart("test")])],
                planner.model_settings,
                ModelRequestParameters(),
            )
            await client.close()

    asyncio.run(run())
    assert bodies[0]["chat_template_kwargs"]["enable_thinking"] is reasoning
    if not reasoning:
        assert "reasoning_effort" not in bodies[0]
