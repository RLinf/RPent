"""Lynsense registration and actual API/CLI paths with offline model replies."""

import argparse
import importlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from robots.lynsense import get_robot_spec, get_toolkit
from robots.lynsense import robot_spec
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.planner.api_loop import ApiAgentLoop, _build_tools
from rpent.robots import enumerate_robots


@pytest.fixture(autouse=True)
def output_directory(tmp_path, monkeypatch):
    from rpent.utils import logging

    monkeypatch.setattr(logging, "_output_dir", tmp_path)


def args_for(tmp_path, **changes):
    parser = argparse.ArgumentParser()
    get_robot_spec().add_cli_args(parser, False)
    args = parser.parse_args([])
    defaults = dict(planner="api", memory_profile="local", memory_dir=None,
                    output_dir=str(tmp_path), dashboard=False, interactive=False, explore=False)
    for name, value in (defaults | changes).items():
        setattr(args, name, value)
    return args


def make_toolkit(tmp_path, **changes):
    spec = get_robot_spec()
    args = args_for(tmp_path, **changes)
    config = spec.parse_config(args)
    daemons, kwargs = spec.init_runtime(args, tmp_path, NullDashboardEventSink(), None)
    assert daemons == []
    return get_toolkit(runtime_kwargs=kwargs, config=config, dashboard_events=NullDashboardEventSink())


def test_registry_import_is_lazy_without_ros(monkeypatch):
    for name in ("rclpy", "sensor_msgs", "xarm_msgs"):
        monkeypatch.setitem(sys.modules, name, None)
    importlib.reload(robot_spec)
    assert "lynsense" in enumerate_robots()
    spec = get_robot_spec()
    assert spec.name == "lynsense"
    assert spec.is_real_robot and not spec.supports_exploration
    assert "read_robot_state" in spec.prompts.render("system")


def test_runtime_returns_inert_resources_and_honors_component_selection(tmp_path, fake_ros):
    spec = get_robot_spec()
    args = args_for(tmp_path)
    assert spec.init_runtime(args, tmp_path, NullDashboardEventSink(), set()) == ([], {})
    with pytest.raises(ValueError, match="component"):
        spec.init_runtime(args, tmp_path, NullDashboardEventSink(), {"vla"})
    daemons, kwargs = spec.init_runtime(args, tmp_path, NullDashboardEventSink(), {"env"})
    assert daemons == []
    assert kwargs["adapter"].get_snapshot()["status"] == "unavailable"
    assert fake_ros.events == []
    kwargs["adapter"].close()


@pytest.mark.parametrize("change", [
    {"planner": "codex"}, {"planner": "claude_code"}, {"planner": "task_card"},
    {"memory_profile": "hf"}, {"memory_profile": None},
    {"dashboard": True}, {"interactive": True}, {"explore": True},
    {"state_timeout": -1}, {"state_max_age": float("inf")},
    {"ros_domain_id": 233}, {"joint_state_topic": "relative"},
    {"robot_state_topic": "/right_xarm/joint_states"},
])
def test_parse_config_rejects_unsupported_or_invalid_run_before_ros(tmp_path, fake_ros, change):
    with pytest.raises(ValueError):
        get_robot_spec().parse_config(args_for(tmp_path, **change))
    assert fake_ros.events == []


def test_unique_tool_reaches_actual_api_builder_without_envstate(tmp_path, fake_ros):
    toolkit = make_toolkit(tmp_path)
    try:
        assert [s["name"] for s in toolkit.get_tools_spec()] == ["read_robot_state"]
        assert toolkit.get_tools_spec()[0]["input_schema"]["additionalProperties"] is False
        for no_images in (False, True):
            assert [tool.name for tool in _build_tools(toolkit, no_images=no_images)] == ["read_robot_state"]
        result = toolkit.execute_tool("read_robot_state", {}).result
        assert result["status"] == "ok"
        assert "state_capture_error" not in result
        assert "error" in toolkit.execute_tool("read_robot_state", {"move": True}).result
        for name in ("read_image", "read_text_file", "write_text_file", "finish", "set_position"):
            assert "error" in toolkit.execute_tool(name, {}).result
        assert toolkit.solved() is False
        assert toolkit.write_recipe("anything") is None
    finally:
        toolkit.close()


def test_toolkit_factory_failure_closes_adapter(tmp_path, fake_ros, monkeypatch):
    import robots.lynsense.toolkit as toolkit_module
    spec = get_robot_spec()
    args = args_for(tmp_path)
    _, kwargs = spec.init_runtime(args, tmp_path, NullDashboardEventSink(), None)
    adapter = kwargs["adapter"]

    def fail_constructor(*args, **kwargs):
        raise RuntimeError("toolkit constructor failed")

    monkeypatch.setattr(toolkit_module, "LynsenseToolkit", fail_constructor)
    with pytest.raises(RuntimeError, match="constructor failed"):
        get_toolkit(runtime_kwargs=kwargs, config=spec.parse_config(args), dashboard_events=NullDashboardEventSink())
    assert adapter.get_snapshot()["status"] == "closed"
    assert fake_ros.events == []


def test_connection_failure_propagates_and_cleans_up(tmp_path, fake_ros):
    fake_ros.auto_messages = False
    with pytest.raises(RuntimeError, match="timed out"):
        make_toolkit(tmp_path, state_timeout=0.025)
    assert not fake_ros.active


def test_runtime_adapter_is_consumed_once(tmp_path, fake_ros):
    spec = get_robot_spec()
    args = args_for(tmp_path)
    _, kwargs = spec.init_runtime(args, tmp_path, NullDashboardEventSink(), None)
    toolkit = get_toolkit(runtime_kwargs=kwargs, config=spec.parse_config(args), dashboard_events=NullDashboardEventSink())
    try:
        assert kwargs == {}
        with pytest.raises(KeyError, match="adapter"):
            get_toolkit(runtime_kwargs=kwargs, config=spec.parse_config(args), dashboard_events=NullDashboardEventSink())
        assert toolkit.read_robot_state()["status"] == "ok"
    finally:
        toolkit.close()


@pytest.mark.parametrize("concurrent", [False, True])
def test_copied_runtime_kwargs_cannot_create_two_owners(tmp_path, fake_ros, concurrent):
    spec = get_robot_spec()
    args = args_for(tmp_path)
    _, kwargs = spec.init_runtime(args, tmp_path, NullDashboardEventSink(), None)
    copies = [dict(kwargs), dict(kwargs)]

    def construct(copy):
        try:
            return get_toolkit(runtime_kwargs=copy, config=spec.parse_config(args), dashboard_events=NullDashboardEventSink())
        except RuntimeError as exc:
            return exc

    if concurrent:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(construct, copies))
    else:
        results = list(map(construct, copies))
    owners = [result for result in results if not isinstance(result, Exception)]
    try:
        assert len(owners) == 1
        assert "claimed" in str(next(result for result in results if isinstance(result, Exception)))
        assert owners[0].read_robot_state()["status"] == "ok"
        assert "context_shutdown" not in fake_ros.events
    finally:
        for owner in owners:
            owner.close()


@pytest.mark.parametrize("stage", ["constructor", "connect"])
def test_factory_interruption_closes_owned_adapter(tmp_path, fake_ros, monkeypatch, stage):
    import robots.lynsense.toolkit as toolkit_module

    spec = get_robot_spec()
    args = args_for(tmp_path)
    _, kwargs = spec.init_runtime(args, tmp_path, NullDashboardEventSink(), None)
    adapter = kwargs["adapter"]

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    if stage == "constructor":
        monkeypatch.setattr(toolkit_module, "LynsenseToolkit", interrupt)
    else:
        monkeypatch.setattr(adapter, "connect", interrupt)
    with pytest.raises(KeyboardInterrupt):
        get_toolkit(runtime_kwargs=kwargs, config=spec.parse_config(args), dashboard_events=NullDashboardEventSink())
    assert adapter.get_snapshot()["status"] == "closed"
    assert not fake_ros.active


def test_custom_memory_and_environment_snapshot(tmp_path, fake_ros):
    toolkit = make_toolkit(tmp_path, memory_dir=str(tmp_path / "local-memory"))
    try:
        assert toolkit.memory.root == tmp_path / "local-memory"
        snapshot = toolkit.get_env_state(command={}, result={}, elapsed_s=0)
        assert snapshot["positions"] == toolkit.read_robot_state()["positions"]
        assert snapshot["status"] == "ok"
    finally:
        toolkit.close()


@pytest.mark.parametrize("behavior", ["read", "no_tool", "error"])
def test_actual_planner_can_read_or_stop_or_fail(tmp_path, fake_ros, behavior):
    toolkit = make_toolkit(tmp_path)
    results = []

    def model(messages, info):
        assert [tool.name for tool in info.function_tools] == ["read_robot_state"]
        if behavior == "error":
            raise RuntimeError("offline model failure")
        if behavior == "no_tool":
            return ModelResponse(parts=[TextPart("No observation made.")])
        returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        if returns:
            results.append(returns[-1].content)
            return ModelResponse(parts=[TextPart("Read-only state received.")])
        return ModelResponse(parts=[ToolCallPart("read_robot_state", {}, "state-1")])

    try:
        planner = ApiAgentLoop(FunctionModel(model), dashboard_events=NullDashboardEventSink(), timeout_s=5)
        result = planner.solve(system_prompt="Read state only.", user_message="Read state.", toolkit=toolkit, max_turns=3)
        assert result.finish_result is None
        if behavior == "error":
            assert "offline model failure" in result.error
        else:
            assert result.error is None
        if behavior == "read":
            assert result.stats["tool_calls"] == 1
            assert '"status": "ok"' in str(results)
    finally:
        toolkit.close()
    assert fake_ros.events.count("context_shutdown") == 1


@pytest.mark.parametrize("model_present", [True, False])
def test_real_cli_preserves_model_validation_and_local_memory(tmp_path, fake_ros, monkeypatch, model_present):
    from rpent.cli import main as cli
    import pydantic_ai.models as models

    model_tool_lists = []

    def response(messages, info):
        model_tool_lists.append([tool.name for tool in info.function_tools])
        if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            return ModelResponse(parts=[TextPart("State read offline.")])
        return ModelResponse(parts=[ToolCallPart("read_robot_state", {}, "state-cli")])

    original_infer = models.infer_model

    def infer(model, **kwargs):
        if not isinstance(model, str):
            return original_infer(model, **kwargs)
        assert model == "openai:offline-test"
        return FunctionModel(response)

    def reject_sync(*args, **kwargs):
        pytest.fail("unexpected remote memory sync")

    monkeypatch.setattr(models, "infer_model", infer)
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    monkeypatch.setattr(MemoryManager, "sync", reject_sync)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    argv = ["rpent", "--robot", "lynsense", "--planner", "api", "--memory-profile", "local",
            "--memory-dir", str(tmp_path / "memory"), "--output-dir", str(tmp_path), "--max-turns", "3"]
    if model_present:
        argv += ["--model", "openai:offline-test"]
    monkeypatch.setattr(sys, "argv", argv)
    if not model_present:
        with pytest.raises(ValueError, match="model id"):
            cli.main()
        assert fake_ros.events == []
    else:
        assert cli.main() == 0
        assert model_tool_lists == [["read_robot_state"], ["read_robot_state"]]
        assert fake_ros.events.count("context_shutdown") == 1
        transcript = json.loads((tmp_path / "transcript_lynsense_readonly.json").read_text())
        assert transcript["stats"]["tool_calls"] == 1
        assert transcript["finish"] is None
