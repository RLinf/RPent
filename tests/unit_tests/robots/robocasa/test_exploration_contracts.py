"""CPU-only exploration contracts; no simulator or policy is constructed."""

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robocasa import robot_spec, toolkit, tools
from robots.robocasa.env_client import RoboCasaEnvClient
from robots.robocasa.env_server import RoboCasaEnvFacade
from robots.robocasa.primitives import RoboCasaPrimitives
from robots.robocasa.rldx_skill import RLDXSkill
from robots.robocasa.vla_server import RoboCasaVLAFacade
from rpent.dashboard.events import NullDashboardEventSink
from rpent.prompt.utils import format_prompt
from rpent.utils import templates


@pytest.fixture
def exploration(monkeypatch, tmp_path, fake_single_arm_primitives):
    class Env:
        success = False
        actions = 0

        def __init__(self):
            self.seeds = []

        def reset_exploration(self):
            self.seeds.append(7)
            self.success = False
            self.actions = 0
            return {"seed": 7, "notice": "按配置 seed 重新初始化，完整物理布局确定性仍需真实仿真验证"}

    env = Env()
    vla = SimpleNamespace(reset_session=lambda: None)

    class Primitives(fake_single_arm_primitives):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.env = env
            self._pos_jac = self._fwd_offset = "stale"
            self._cam_meta_cache = {"agentview": "stale"}
            self._rldx = RLDXSkill(env, vla_client=vla)
            self._rldx._hist = deque(["old"], maxlen=3)

        reset_exploration = RoboCasaPrimitives.reset_exploration

        def move_to(self, win=False):
            env.actions += 1
            env.success = win
            return {"ok": True}

    def dump(primitives, state, log):
        log = log or {}
        with state.record_step(
            state={}, terminated=env.success, truncated=False,
            extras={"success": env.success}, **log,
        ) as idx:
            pass
        return state.get(idx)

    monkeypatch.setattr("robots.robocasa.primitives.RoboCasaPrimitives", Primitives)
    monkeypatch.setattr(tools, "dump_state", dump)
    monkeypatch.setattr(toolkit, "get_output_dir", lambda: tmp_path)
    monkeypatch.setattr(templates, "default_variables", lambda: {"output_dir": str(tmp_path)})
    config = robot_spec._parse_config(argparse.Namespace(
        task_name="OpenDrawer", split="target", seed=7,
        output_dir=tmp_path, memory_dir=tmp_path / "memory",
    ))

    def make(budget=3, session=1, mode="exploration"):
        return robot_spec.get_toolkit(
            primitives_kwargs={"env_client": env, "vla_client": vla},
            dashboard_events=NullDashboardEventSink(), config=config,
            mode=mode, attempts_per_session=budget,
            state_output_dir=tmp_path / "sessions" / f"session_{session:03d}",
        )

    return make, env, vla, config


@pytest.mark.parametrize("budget", [0, 1, 3])
def test_attempt_budget_and_guarded_finish(exploration, budget):
    make, env, _, _ = exploration
    robot = make(budget)
    finish = lambda: robot.execute_tool("finish", {"status": "success", "summary": "planner claim"})
    assert not robot.solved()
    assert finish().is_finish is (budget in (0, 1))
    for attempt in range(2, (budget or 5) + 1):
        result = robot.execute_tool("reset", {"reason": "changed approach"})
        assert result.result["log"]["result"]["attempt"] == attempt
    assert finish().is_finish
    assert not robot.solved()  # finish claims and successful tools are not native success
    if budget:
        count = len(env.seeds)
        robot.execute_tool("reset", {"reason": "over budget"})
        assert len(env.seeds) == count
    robot.execute_tool("move_to", {"win": True})
    assert robot.solved()
    count = len(env.seeds)
    robot.execute_tool("reset", {"reason": "after success"})
    assert len(env.seeds) == count


def test_failed_reset_consumes_attempt_and_blocks_actions(exploration):
    make, env, vla, _ = exploration
    robot = make(3)

    def fail():
        raise RuntimeError("private session reset RPC failed")

    vla.reset_session = fail
    result = robot.execute_tool("reset", {"reason": "retry"})
    assert "RPC failed" in result.result["error"]
    assert robot._session_attempt == 2
    robot.execute_tool("move_to", {"win": True})
    assert env.actions == 0
    assert not robot.solved()
    assert robot.write_recipe("cell") == ""
    vla.reset_session = lambda: None
    robot.execute_tool("reset", {"reason": "recover"})
    assert robot._session_attempt == 3
    assert env.seeds == [7, 7]
    robot.execute_tool("move_to", {"win": True})
    assert robot.solved()


def test_sessions_initialize_once_and_preserve_trace(exploration):
    make, env, _, _ = exploration
    first = make()
    first.execute_tool("move_to", {})
    first.execute_tool("reset", {"reason": "retry"})
    assert env.actions == 0
    primitive = first._primitives
    assert primitive._pos_jac is primitive._fwd_offset is None
    assert primitive._cam_meta_cache == {}
    assert not primitive._rldx._hist
    assert primitive._vla_desync
    second = make(session=2)
    assert env.seeds == [7, 7, 7]
    assert primitive.reset_calls == second._primitives.reset_calls == 0
    assert len(first.state.records()) == 3
    assert len(second.state.records()) == 1
    assert second._session_attempt == 1


def test_recipe_only_winning_attempt_at_run_root(exploration, tmp_path):
    make, env, _, config = exploration
    robot = make()
    robot.execute_tool("move_to", {})
    assert robot.write_recipe(config.recipe_tag) == ""
    robot.execute_tool("reset", {"reason": "new strategy"})
    robot.execute_tool("move_to", {"win": True})
    robot.write_recipe(config.recipe_tag)
    recipe = tmp_path / f"{config.recipe_tag}_recipe.jsonl"
    assert [json.loads(line) for line in recipe.read_text().splitlines()] == [
        {"action": "move_to", "win": True}
    ]
    assert not robot.state.artifact_path(recipe.name, step=None).exists()
    env.success = False
    robot.get_env_state(command={"action": "observe"}, result={}, elapsed_s=0)
    assert not robot.solved()  # latest state, not cumulative OR


def test_evaluation_keeps_legacy_initialization(exploration):
    make, env, _, _ = exploration
    robot = make(mode="evaluation")
    assert env.seeds == []
    assert robot._primitives.reset_calls == 1
    assert "exploration" not in robot._primitives.kwargs
    assert robot._tools["finish"][1] is tools.finish


def test_memory_permissions_and_merge_names(exploration, tmp_path):
    make, _, _, config = exploration
    robot = make()
    root = robot.memory.root
    bindings = robot.memory.get_common_tool_bindings()
    write = bindings["write_text_file"][1]
    read = bindings["read_text_file"][1]
    inbox = root / "_internal" / "inbox" / config.recipe_tag
    write(path=str(inbox / "wip" / "notes.md"), content="observations")
    with pytest.raises(PermissionError):
        write(path=str(root / "_internal" / "inbox" / "other" / "note.md"), content="no")
    with pytest.raises(PermissionError):
        write(path=str(root / "task_only" / "bad.json"), content="no")
    robot.execute_tool("move_to", {"win": True})
    robot.write_recipe(config.recipe_tag)
    (tmp_path / f"{config.recipe_tag}.json").write_text('{"success": true}')
    robot.memory.merge_memory(cell_tag=config.recipe_tag, run_state_dir=tmp_path, solved=True)
    audit = root / "task_only" / f"{config.recipe_tag}.json"
    assert json.loads(read(path=str(audit))["content"])["success"]
    from robots.robocasa.prompt_bundle import system_prompt, user_prompt
    variables = {**config.prompt_vars, "mode": "explore", "memory_profile": "local",
                 "memory_inbox": str(inbox), "output_dir": str(tmp_path),
                 "session_number": 1, "session_max": 3, "attempts_per_session": 3}
    prompt = format_prompt(system_prompt(variables), variables=variables)
    assert "OpenDrawer_target_s<seed>.json" in prompt
    assert "完整物理布局确定性仍需真实仿真验证" in prompt
    assert "_check_success" in prompt
    assert "{{" not in prompt
    assert "no-reset mode" not in format_prompt(user_prompt(variables), variables=variables)


def test_client_defers_only_exploration_initial_reset():
    meta = {"seed": 7, "camera_h": 256, "camera_w": 256}
    calls = []

    def call(method, **kwargs):
        calls.append(method)
        if method == "env.get_env_meta":
            return meta
        if method == "env.reset_exploration":
            return {"observation": {"fresh": True}, "seed": 7,
                    "reset_contract": "configured_seed_reinitialization"}
        return {}

    client = RoboCasaEnvClient(SimpleNamespace(call=call), expected_meta=meta, defer_reset=True)
    assert calls == ["env.get_env_meta"]
    client.reset_exploration()
    assert client.last_obs == {"fresh": True}
    calls.clear()
    RoboCasaEnvClient(SimpleNamespace(call=call), expected_meta=meta)
    assert calls == ["env.get_env_meta", "env.reset"]


def test_server_rebuilds_same_configuration_instead_of_advancing_rng(monkeypatch):
    made = []

    class Env:
        def __init__(self, **kwargs):
            self.seed = kwargs["seed"]
            self.closed = False
            self.count = 99
            made.append(kwargs)

        def close(self):
            self.closed = True

        def reset(self):
            self.count = 0
            return {"sample": np.random.random()}

    monkeypatch.setitem(sys.modules, "robosuite", SimpleNamespace(make=Env))
    facade = RoboCasaEnvFacade.__new__(RoboCasaEnvFacade)
    facade.seed = 7
    facade._env_kwargs = {"seed": 7, "env_name": "OpenDrawer"}
    facade.env = Env(seed=7)
    old = facade.env
    first = facade.reset_exploration()
    second = facade.reset_exploration()
    assert old.closed
    assert first == second  # fake scene only; not a real simulator assertion
    assert made[1:] == [facade._env_kwargs, facade._env_kwargs]
    assert facade.env.count == 0


def test_rldx_reset_is_private_and_failure_propagates():
    calls = []
    facade = RoboCasaVLAFacade.__new__(RoboCasaVLAFacade)
    facade.policy = SimpleNamespace(reset=lambda options: calls.append(options))
    assert facade.reset_session(session_id="private-A") == {"ok": True}
    assert calls == [{"session_ids": ["private-A"]}]
    skill = RLDXSkill(object(), vla_client=SimpleNamespace(
        reset_session=lambda: facade.reset_session(session_id="private-A")))
    skill._hist = deque(["old frame"])
    skill._last_prompt = "old instruction"
    skill.reset_session()
    assert not skill._hist and skill._last_prompt is None
    assert skill._vla_client is not None

    def fail(options):
        raise RuntimeError("RPC failure")

    facade.policy.reset = fail
    with pytest.raises(RuntimeError, match="RPC failure"):
        skill.reset_session()


def test_real_spec_accepts_exploration_flags():
    spec = robot_spec.get_robot_spec()
    assert spec.supports_exploration
    parser = argparse.ArgumentParser()
    spec.add_cli_args(parser, False)
    args = parser.parse_args(["--task-name", "OpenDrawer", "--explore-sessions", "3",
                              "--explore-attempts-per-session", "0"])
    assert args.explore_sessions == 3
    assert args.explore_attempts_per_session == 0


def test_real_primitives_skip_exploration_constructor_reset(tmp_path):
    calls = []
    env = SimpleNamespace(reset=lambda: calls.append("legacy reset"))
    vla = SimpleNamespace(reset_session=lambda: None)
    explored = RoboCasaPrimitives(env, str(tmp_path), None, vla, exploration=True)
    assert calls == []
    explored._frames.append("archived frame")
    env.reset_exploration = lambda: {"seed": 7}
    explored.reset_exploration()
    assert explored._frames == ["archived frame"]
    RoboCasaPrimitives(env, str(tmp_path), None, vla)
    assert calls == ["legacy reset"]


def test_parse_config_renders_actual_exploration_prompt(tmp_path):
    args = argparse.Namespace(
        task_name="OpenDrawer", split="target", seed=7, output_dir=tmp_path,
        memory_dir=None, explore=True, explore_sessions=3,
        explore_attempts_per_session=1,
    )
    config = robot_spec._parse_config(args)
    assert config.prompt_vars["mode"] == "explore"
    assert config.prompt_vars["memory_profile"] == "local"
    assert config.prompt_vars["memory_inbox"] == str(
        Path(config.prompt_vars["memory_dir"])
        / "_internal"
        / "inbox"
        / config.recipe_tag
    )
    assert config.prompt_vars["session_number"] == 1
    assert config.prompt_vars["session_max"] == 3
    assert config.prompt_vars["attempts_per_session"] == 1
    variables = {**config.prompt_vars, "output_dir": str(tmp_path)}
    prompt = robot_spec.get_robot_spec().prompts.render("system", variables=variables)
    assert "完整物理布局确定性仍需真实仿真验证" in prompt
    assert "1 attempts INCLUDING" in prompt
    assert "{{" not in prompt


def test_handoff_does_not_claim_physical_restoration(tmp_path):
    from rpent.cli.main import _handoff_message
    message = _handoff_message(tmp_path, 2, 3, robot_name="robocasa")
    assert "完整物理布局确定性仍需真实仿真验证" in message
    assert "restored a clean scene" not in message


def test_environment_reset_failure_exhausts_budget(exploration):
    make, env, _, _ = exploration
    robot = make(3)

    def fail():
        raise RuntimeError("environment reconstruction failed")

    env.reset_exploration = fail
    for attempt in (2, 3):
        result = robot.execute_tool("reset", {"reason": "retry"})
        assert "reconstruction failed" in result.result["error"]
        assert robot._session_attempt == attempt
    result = robot.execute_tool("finish", {"status": "failure", "summary": "handoff"})
    assert result.is_finish
    assert not robot.solved()


def test_vla_client_rejects_negative_reset_acknowledgement():
    from robots.robocasa.vla_client import RoboCasaVLAClient
    client = RoboCasaVLAClient.__new__(RoboCasaVLAClient)
    calls = []

    def call(method, **kwargs):
        calls.append((method, kwargs))
        return {"ok": False}

    client._client = SimpleNamespace(call=call)
    with pytest.raises(RuntimeError, match="not acknowledged"):
        client.reset_session()
    assert calls == [("vla.reset_session", {"timeout_s": client._TIMEOUT_S["default"]})]
