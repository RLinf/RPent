# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.behavior import robot_spec as behavior_robot_spec
from robots.behavior import toolkit as behavior_toolkit
from robots.behavior.dino_v2.client import BehaviorDinoClient
from robots.behavior.dino_v2.encoder import DINOV2_DIMENSION
from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.robot_spec import get_toolkit
from robots.behavior.tools import BehaviorPrimitives
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager, loading
from rpent.robots import RunConfig, get_robot_spec
from rpent.tools.common import CommonTools


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--memory-profile", choices=["hf", "local"], default=None)
    parser.add_argument("--memory-dir", default=None)
    get_robot_spec("behavior").add_cli_args(parser, use_dashboard=False)
    return parser


def test_behavior_registry_supports_exploration_without_manual_primitives() -> None:
    spec = get_robot_spec("behavior")
    assert spec.supports_exploration
    assert spec.dashboard["primitives"] == ()


def test_behavior_uses_shared_memory_profiles(tmp_path: Path) -> None:
    memory_dir = tmp_path / "behavior-memory"
    args = _parser().parse_args(
        [
            "--task-name",
            "turning_on_radio",
            "--public-seed",
            "1",
            "--memory-dir",
            str(memory_dir),
            "--memory-profile",
            "local",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    config = get_robot_spec("behavior").parse_config(args)

    assert args.memory_profile == "local"
    assert args.memory_dir == str(memory_dir.resolve())
    assert config.prompt_vars["memory_profile"] == "local"
    assert config.prompt_vars["memory_dir"] == str(memory_dir.resolve())
    assert config.prompt_vars["memory_inbox"].endswith(
        "_internal/inbox/turning_on_radio_s1"
    )

    evaluation = _parser().parse_args(
        [
            "--task-name",
            "turning_on_radio",
            "--public-seed",
            "1",
            "--memory-profile",
            "hf",
        ]
    )
    config = get_robot_spec("behavior").parse_config(evaluation)
    assert config.prompt_vars["memory_profile"] == "hf"
    assert config.prompt_vars["mode"] == "eval"
    exploration = _parser().parse_args(
        [
            "--task-name",
            "turning_on_radio",
            "--public-seed",
            "0",
            "--explore",
        ]
    )
    config = get_robot_spec("behavior").parse_config(exploration)
    assert config.prompt_vars["memory_profile"] == "local"
    assert config.prompt_vars["mode"] == "explore"


def test_behavior_explicit_memory_source_does_not_record_stale_local_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _parser().parse_args(
        ["--task-name", "turning_on_radio", "--public-seed", "1"]
    )
    args.memory_repo = "team/memory@" + "a" * 40
    args.model = "gpt-6-astra"
    config = get_robot_spec("behavior").parse_config(args)
    selected = tmp_path / "selected"
    monkeypatch.setattr(loading, "prepare_explicit_source", lambda *_: selected)

    loading.prepare_run_memory(args, get_robot_spec("behavior"), config)

    assert args.memory_dir == config.prompt_vars["memory_dir"] == str(selected)
    assert "memory_dir" not in config.task_desc


@pytest.mark.parametrize(
    ("mode", "write_allowed"),
    [("eval", False), ("explore", True)],
)
def test_behavior_toolkit_uses_one_official_memory_manager(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    write_allowed: bool,
) -> None:
    captured: dict[str, Any] = {}

    def fake_toolkit(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(behavior_toolkit, "BehaviorToolkit", fake_toolkit)
    memory_dir = tmp_path / "memory"
    recipe_tag = "turning_on_radio_s1"
    config = RunConfig(
        recipe_tag=recipe_tag,
        output_dir=tmp_path / "run",
        prompt_vars={
            "mode": mode,
            "memory_dir": str(memory_dir),
        },
        task_desc={},
    )

    toolkit = behavior_robot_spec.get_toolkit(
        runtime_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        mode="exploration" if mode == "explore" else "evaluation",
    )

    assert toolkit.memory is captured["memory"]
    assert toolkit.memory.root == memory_dir.resolve()
    write = CommonTools(memory=toolkit.memory).write_text_file
    destination = memory_dir / "_internal" / "inbox" / recipe_tag / "wip" / "notes.md"
    if write_allowed:
        write(str(destination), "evidence")
        assert destination.read_text() == "evidence"
    else:
        with pytest.raises(PermissionError, match="writing to memory is denied"):
            write(str(destination), "evidence")

    component_names = {
        item["name"]
        for item in behavior_robot_spec.BEHAVIOR_DASHBOARD_SPEC["runtime_components"]
    }
    assert component_names == {"env", "vla", "dino"}


class _FakeEnv:
    total_env_steps = 0
    official_success_latched = False
    official_success_receipt = None

    def __init__(self) -> None:
        self.last_move: dict[str, Any] | None = None
        self.gripper_calls: list[tuple[str, dict[str, Any]]] = []

    def move_to(self, **kwargs: Any) -> dict[str, Any]:
        self.last_move = kwargs
        return {"status": "failed", "stop_reason": "motion_unavailable"}

    def close_gripper(self, **kwargs: Any) -> dict[str, Any]:
        self.gripper_calls.append(("close_gripper", kwargs))
        return {"status": "success"}

    def open_gripper(self, **kwargs: Any) -> dict[str, Any]:
        self.gripper_calls.append(("open_gripper", kwargs))
        return {"status": "success"}


class _FakeRpcClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(
        self,
        method: str,
        *,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        self.calls.append((method, kwargs or {}))
        if method == "env.get_env_meta":
            return {}
        if method == "dino.get_meta":
            return {
                "runtime": "behavior_dino",
                "dimension": DINOV2_DIMENSION,
            }
        return {"status": "success"}


def test_behavior_toolkit_factory_maps_shared_modes(tmp_path: Path) -> None:
    config = RunConfig(
        recipe_tag="turning_on_radio_s0",
        output_dir=tmp_path / "run",
        prompt_vars={
            "mode": "eval",
            "memory_dir": str(tmp_path / "memory"),
            "task_name": "turning_on_radio",
            "public_seed": 0,
            "max_episode_steps": 64,
        },
        task_desc={},
    )
    dashboard_events = NullDashboardEventSink()

    evaluation = get_toolkit(
        runtime_kwargs={},
        dashboard_events=dashboard_events,
        config=config,
        mode="evaluation",
    )
    exploration = get_toolkit(
        runtime_kwargs={"output_dir": tmp_path / "session"},
        dashboard_events=dashboard_events,
        config=config,
        mode="exploration",
    )
    config.prompt_vars["mode"] = "explore"
    inherited = get_toolkit(
        runtime_kwargs={},
        dashboard_events=dashboard_events,
        config=config,
    )

    assert evaluation.primitives.behavior_phase == "eval"
    assert exploration.primitives.behavior_phase == "explore"
    assert inherited.primitives.behavior_phase == "eval"
    assert evaluation.memory._memory_access == "read_only"
    assert exploration.memory._memory_access == "inbox_write"
    assert exploration.primitives.output_dir == tmp_path / "session"
    with pytest.raises(ValueError, match="one attempt per session"):
        get_toolkit(
            runtime_kwargs={},
            dashboard_events=dashboard_events,
            config=config,
            mode="exploration",
            attempts_per_session=1,
        )
    with pytest.raises(ValueError, match="unsupported BEHAVIOR toolkit mode"):
        get_toolkit(
            runtime_kwargs={},
            dashboard_events=dashboard_events,
            config=config,
            mode="unsupported",
        )


@pytest.mark.parametrize("construction_fails", [False, True])
def test_behavior_toolkit_owns_session_env_not_shared_models(
    tmp_path, monkeypatch, construction_fails
):
    from robots.behavior import runtime

    stopped = []
    launched = []
    model = object()
    dino = object()
    args = SimpleNamespace()
    config = RunConfig(
        recipe_tag="turning_on_radio_s0",
        output_dir=tmp_path,
        prompt_vars={
            "memory_dir": str(tmp_path / "memory"),
            "task_name": "turning_on_radio",
            "public_seed": 0,
        },
        task_desc={},
    )

    def init_runtime(received_args, output_dir, dashboard_events, components):
        assert received_args is args
        assert components == {"env"}
        launched.append(output_dir)
        return [SimpleNamespace(stop=lambda: stopped.append(output_dir))], {
            "env": _FakeEnv()
        }

    monkeypatch.setattr(runtime, "init_runtime", init_runtime)
    if construction_fails:

        def fail(**kwargs):
            raise RuntimeError("toolkit construction failed")

        monkeypatch.setattr(behavior_toolkit, "BehaviorToolkit", fail)
    kwargs = {"env_args": args, "model": model, "dino_component": dino}
    for index in range(2):
        output = tmp_path / f"session_{index}"
        if construction_fails:
            with pytest.raises(RuntimeError, match="toolkit construction failed"):
                get_toolkit(
                    runtime_kwargs=kwargs,
                    dashboard_events=NullDashboardEventSink(),
                    config=config,
                    mode="exploration",
                    state_output_dir=output,
                )
        else:
            toolkit = get_toolkit(
                runtime_kwargs=kwargs,
                dashboard_events=NullDashboardEventSink(),
                config=config,
                mode="exploration",
                state_output_dir=output,
            )
            assert toolkit.primitives.model is model
            assert toolkit.primitives.dino_component is dino
            toolkit.close()
            toolkit.close()
        assert stopped == launched
    assert len(stopped) == 2
    assert kwargs == {"env_args": args, "model": model, "dino_component": dino}


def test_behavior_clients_and_tools_use_explicit_component_rpc_names() -> None:
    rpc = _FakeRpcClient()
    client = BehaviorEnvClient(rpc, expected_meta={})
    dino_client = BehaviorDinoClient(rpc, expected_meta={"runtime": "behavior_dino"})
    env = _FakeEnv()
    primitives = BehaviorPrimitives(env=env, task_name="turning_on_radio")

    client.close_gripper(hand="right")
    client.open_gripper(hand="right")
    primitives.close(hand="right")
    primitives.open(hand="right")

    assert rpc.calls == [
        ("env.get_env_meta", {}),
        ("dino.get_meta", {}),
        ("env.close_gripper", {"hand": "right"}),
        ("env.open_gripper", {"hand": "right"}),
    ]
    assert dino_client.server_meta["dimension"] == DINOV2_DIMENSION
    assert env.gripper_calls == [
        ("close_gripper", {"hand": "right"}),
        ("open_gripper", {"hand": "right"}),
    ]
    assert "env.close_gripper" in BehaviorEnvClient._TIMEOUT_S
    assert "env.open_gripper" in BehaviorEnvClient._TIMEOUT_S
    assert "env.close" not in BehaviorEnvClient._TIMEOUT_S
    assert "env.open" not in BehaviorEnvClient._TIMEOUT_S


@pytest.mark.parametrize(
    ("pause_after", "expected_steps"),
    [("predict", 0), ("chunk", 32)],
)
def test_pi0_nav_pick_stops_at_chunk_cancellation_boundary(
    tmp_path: Path, pause_after: str, expected_steps: int
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class Model:
        calls = 0

        def predict(self, observation, *, options):
            self.calls += 1
            if pause_after == "predict":
                entered.set()
                assert release.wait(5)
            return np.zeros((32, 23), dtype=np.float32)

    class Env:
        total_env_steps = 0
        official_success_latched = False
        official_success_receipt = None
        calls = 0

        def chunk_step(self, actions, *, return_all_frames=False):
            self.calls += 1
            self.total_env_steps += len(actions)
            if pause_after == "chunk":
                entered.set()
                assert release.wait(5)
            return (
                {"states": np.zeros(256, dtype=np.float32)},
                0.0,
                False,
                False,
                {
                    "executed_steps": len(actions),
                    "_rpent": {"total_env_steps": self.total_env_steps},
                },
            )

    model = Model()
    env = Env()
    toolkit = behavior_toolkit.BehaviorToolkit(
        runtime_kwargs={
            "env": env,
            "model": model,
            "initial_observation": {"states": np.zeros(256, dtype=np.float32)},
            "output_dir": tmp_path / "run",
            "max_episode_steps": 64,
        },
        memory=MemoryManager(tmp_path / "memory"),
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            operation = pool.submit(
                toolkit.execute_tool,
                "pi0_nav_pick",
                {"instruction": "turn on the radio", "chunks": 2},
            )
            assert entered.wait(5)
            cancellation = pool.submit(toolkit.cancel_active_and_wait)
            try:
                with toolkit._scheduler.condition:
                    assert toolkit._scheduler.condition.wait_for(
                        lambda: toolkit._scheduler.paused, timeout=5
                    )
            finally:
                release.set()
            result = operation.result(timeout=5)
            cancellation.result(timeout=5)

        assert model.calls == 1
        assert env.calls == int(pause_after == "chunk")
        assert env.total_env_steps == expected_steps
        assert result.data["code"] == "tool_cancelled"
        assert result.data["interrupted"] is True
        assert result.data["task_success"] is False
        assert result.data["total_env_steps"] == expected_steps
    finally:
        toolkit.close()


@pytest.mark.parametrize(
    ("mode", "task_name", "public_seed"),
    [
        ("eval", "turning_on_radio", 1),
        ("explore", "picking_up_trash", 0),
    ],
)
def test_behavior_prompts_render_real_run_config_semantics(
    tmp_path: Path,
    mode: str,
    task_name: str,
    public_seed: int,
) -> None:
    spec = get_robot_spec("behavior")
    parser = _parser()
    cli_args = [
        "--task-name",
        task_name,
        "--public-seed",
        str(public_seed),
        "--output-dir",
        str(tmp_path / "output"),
        "--memory-dir",
        str(tmp_path / "memory"),
    ]
    if mode == "explore":
        cli_args.insert(0, "--explore")
    args = parser.parse_args(cli_args)
    variables = spec.parse_config(args).prompt_vars

    system = spec.prompts.render("system", variables=variables)
    user = spec.prompts.render("user", variables=variables)

    assert "{{" not in system
    assert "{{" not in user
    for value in (
        task_name,
        variables["task_instruction"],
        str(public_seed),
        mode,
        variables["recipe_tag"],
        variables["output_dir"],
        str(variables["max_episode_steps"]),
        str(variables["wall_clock_seconds"]),
        str(variables["public_capabilities"]),
        variables["memory_profile"],
        variables["memory_dir"],
    ):
        assert value in system or value in user

    if mode == "explore":
        assert variables["memory_inbox"] in system
        assert f"{variables['output_dir']}/{variables['recipe_tag']}.json" in system
        assert f"{variables['memory_inbox']}/wip/" in system
        assert "official success" in system.lower()
        assert "merge" in system.lower()
    for tool in (
        "pi0_nav_pick",
        "observe",
        "pixel_to_world",
        "navigate_to",
        "move_to",
        "rotate_wrist",
        "close",
        "open",
        "press",
    ):
        assert f"`{tool}`" in system
    assert variables["task_instruction"] in user
    assert variables["recipe_tag"] in user

    required = {
        "mode",
        "task_name",
        "task_instruction",
        "public_seed",
        "recipe_tag",
        "output_dir",
        "max_episode_steps",
        "wall_clock_seconds",
        "public_capabilities",
        "memory_profile",
        "memory_dir",
    }
    if mode == "explore":
        required.add("memory_inbox")
    for key in required:
        incomplete = {name: value for name, value in variables.items() if name != key}
        with pytest.raises(KeyError, match=key):
            spec.prompts.render("system", variables=incomplete)

    literal = {**variables, "task_instruction": "literal {{do_not_expand}}"}
    assert "literal {{do_not_expand}}" in spec.prompts.render(
        "system", variables=literal
    )

    with pytest.raises(ValueError, match="unsupported BEHAVIOR prompt mode"):
        spec.prompts.render("system", variables={**variables, "mode": "unknown"})


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (
            ["--explore-attempts-per-session", "1"],
            "BEHAVIOR explore runs one attempt per session; use --explore-sessions",
        ),
        (
            ["--env-endpoint", "127.0.0.1:1"],
            "BEHAVIOR explore requires an owned env sidecar; omit --env-endpoint",
        ),
    ],
)
def test_behavior_explore_rejects_incompatible_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
    message: str,
) -> None:
    from rpent.cli import main as cli

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent",
            "--robot",
            "behavior",
            "--task-name",
            "turning_on_radio",
            "--public-seed",
            "0",
            "--explore",
            *extra_args,
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err
