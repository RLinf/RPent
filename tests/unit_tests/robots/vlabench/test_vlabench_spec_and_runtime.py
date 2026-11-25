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

"""Offline contracts for VLABench spec, CLI config, finalization, runtime, and RPC client."""

from __future__ import annotations

import argparse
import ast
import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.vlabench import adapter, get_robot_spec, get_toolkit
from robots.vlabench.agent_toolkit import VLABenchToolkit
from robots.vlabench.config import (
    add_cli_args,
    finalize_run,
    parse_config,
    prepare_memory,
    validate_args,
)
from robots.vlabench.sim.contracts import env_identity
from robots.vlabench.sim.env_server import VLABenchEnvFacade
from rpent.dashboard.events import NullDashboardEventSink
from rpent.evaluation import RunFinalizationContext
from rpent.robots.robot_spec import RobotSpec
from rpent.utils import templates
from rpent.utils.config import get_repo_root
from rpent.utils.daemon import pick_free_port
from rpent.utils.rpc.http_rpc import HttpRpcClient

IDENTITY = {
    "task_name": "select_painting",
    "track": "track_1_in_distribution",
    "episode_index": 0,
    "seed": 42,
}


def _cli_args(tmp_path: Path, **overrides: Any) -> argparse.Namespace:
    defaults: dict[str, Any] = {
        "task_name": "select_painting",
        "track": "track_1_in_distribution",
        "episode_index": 0,
        "seed": 42,
        "eval_unseen": False,
        "max_episode_steps": None,
        "vlabench_root": str(tmp_path / "VLABench"),
        "vlabench_asset_root": str(tmp_path / "assets"),
        "vlabench_python": None,
        "env_endpoint": None,
        "env_cuda_device": None,
        "cuda_device": None,
        "output_dir": str(tmp_path / "out"),
        "memory_dir": str(tmp_path / "memory"),
        "memory_repo": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_spec_and_dashboard_contracts() -> None:
    spec = get_robot_spec()
    assert isinstance(spec, RobotSpec)
    assert spec.name == "vlabench"
    assert spec.supports_exploration is False
    assert spec.finalize_run is finalize_run
    dashboard = spec.resolve_dashboard(argparse.Namespace())
    assert dashboard is not None
    assert dashboard["runtime_components"] == (
        {"name": "env", "label": "ENV", "scope": "unique"},
    )
    assert dashboard["primitives"] == ("move_pose", "move_delta", "set_gripper")


def test_cli_args_and_validation_and_config(tmp_path: Path) -> None:
    cli_parser = argparse.ArgumentParser()
    add_cli_args(cli_parser, use_dashboard=False)
    with pytest.raises(SystemExit):
        cli_parser.parse_args([])

    dash_parser = argparse.ArgumentParser()
    add_cli_args(dash_parser, use_dashboard=True)
    parsed = dash_parser.parse_args([])
    assert parsed.task_name is None

    validate_args(_cli_args(tmp_path))
    validate_args(
        _cli_args(
            tmp_path,
            vlabench_root=None,
            vlabench_asset_root=None,
            env_endpoint="http://127.0.0.1:1",
        )
    )

    with pytest.raises(ValueError, match="--vlabench-root"):
        validate_args(_cli_args(tmp_path, vlabench_root=None))
    with pytest.raises(ValueError, match="--vlabench-asset-root"):
        validate_args(_cli_args(tmp_path, vlabench_asset_root=None))
    with pytest.raises(ValueError, match="--max-episode-steps"):
        validate_args(_cli_args(tmp_path, max_episode_steps=0))
    with pytest.raises(ValueError, match="--episode-index"):
        validate_args(_cli_args(tmp_path, episode_index=-1))
    with pytest.raises(ValueError, match="--seed"):
        validate_args(_cli_args(tmp_path, seed=-1))
    with pytest.raises(ValueError, match="--memory-repo"):
        validate_args(_cli_args(tmp_path, memory_repo="org/repo"))

    config = parse_config(_cli_args(tmp_path))
    assert config.recipe_tag == "vlabench_select_painting_e0_s42"
    assert config.task_desc["env"] == "vlabench"
    assert config.prompt_vars["mode"] == "eval"
    prepare_memory(_cli_args(tmp_path), config)
    assert Path(config.prompt_vars["memory_dir"]).is_dir()


def test_get_toolkit_rejects_exploration_and_builds_readonly_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": str(tmp_path / "out")}
    )

    class _MinimalEnv:
        def robot_model_state(self) -> dict[str, Any]:
            return {"world_from_eef": np.eye(4)}

        def sensor_snapshot(
            self, *, cameras: list[str], height: int = 512, width: int = 512
        ) -> dict[str, Any]:
            del height, width
            calib = {
                "intrinsic_K": np.eye(3),
                "extrinsic_cam2world": np.eye(4),
                "height": 4,
                "width": 4,
            }
            return {
                "observation_id": "obs_0",
                "cameras": [
                    {
                        "camera": cam,
                        "rgb_top_down": np.zeros((4, 4, 3), dtype=np.uint8),
                        "depth_meters": np.ones((4, 4), dtype=np.float32),
                        "calibration": calib,
                    }
                    for cam in cameras
                ],
                "robot": {"model": self.robot_model_state()},
            }

        def task_status(self) -> dict[str, Any]:
            return {"task_success": False, "terminated": False, "truncated": False}

        def get_task_language(self) -> str:
            return "pick the painting"

    config = parse_config(_cli_args(tmp_path))
    toolkit = get_toolkit(
        runtime_kwargs={"env": _MinimalEnv()},
        dashboard_events=NullDashboardEventSink(),
        config=config,
    )
    assert isinstance(toolkit, VLABenchToolkit)
    assert toolkit.memory.root == Path(config.prompt_vars["memory_dir"]).resolve()
    with pytest.raises(ValueError, match="evaluation mode only"):
        get_toolkit(
            runtime_kwargs={"env": _MinimalEnv()},
            dashboard_events=NullDashboardEventSink(),
            config=config,
            mode="exploration",
        )


def test_finalize_run_writes_result_json(tmp_path: Path) -> None:
    context = RunFinalizationContext(
        output_dir=tmp_path,
        robot_name="vlabench",
        task_desc={"env": "vlabench", "task_name": "select_painting"},
        environment_success=True,
        agent_error=None,
        elapsed_s=12.34,
        planner="api",
        model="openai:gpt-5.4",
        reasoning_effort="high",
        max_turns=20,
        planner_timeout_s=600,
        finish_result={"status": "success", "success": True},
        stats={"tool_calls": 5},
        environment_metrics={"native_actions": 10},
    )
    path = finalize_run(context)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "result.json"
    assert payload["robot"] == "vlabench"
    assert payload["environment_success"] is True
    assert payload["environment_result_available"] is True
    assert payload["elapsed_s"] == 12.3
    assert payload["metrics"] == {"native_actions": 10}


def test_init_runtime_borrows_endpoint_and_cleans_failed_daemon(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _cli_args(tmp_path, env_endpoint="http://127.0.0.1:9999")
    with pytest.raises(ValueError, match="unknown VLABench runtime components"):
        adapter.init_runtime(args, tmp_path, NullDashboardEventSink(), {"vla"})

    class _FakeRpc:
        def call(self, method: str, args=(), kwargs=None, *, timeout_s=None):
            del args, kwargs, timeout_s
            if method == "env.get_env_meta":
                return env_identity(**IDENTITY)
            if method == "env.reset":
                return {"episode_id": "ep_1", "task_name": "select_painting"}
            raise AssertionError(method)

    fake_rpc = _FakeRpc()
    monkeypatch.setattr("rpent.utils.rpc.make_rpc_client", lambda endpoint: fake_rpc)
    monkeypatch.setattr("rpent.robots.runtime.wait_for_ready", lambda *a, **kw: None)
    daemons, runtime_kwargs = adapter.init_runtime(
        args, tmp_path, NullDashboardEventSink(), None
    )
    assert daemons == []
    assert isinstance(runtime_kwargs["env"], adapter.VLABenchEnvClient)

    stopped = {"called": False}

    class _FakeDaemon:
        def stop(self) -> None:
            stopped["called"] = True

    monkeypatch.setattr(
        adapter, "spawn_env_server", lambda a, o: (_FakeDaemon(), fake_rpc)
    )
    monkeypatch.setattr(
        "rpent.robots.runtime.wait_for_ready",
        lambda *a, **kw: (_ for _ in ()).throw(TimeoutError("never ready")),
    )
    with pytest.raises(RuntimeError, match="wait / client connect failed"):
        adapter.init_runtime(
            _cli_args(tmp_path), tmp_path, NullDashboardEventSink(), None
        )
    assert stopped["called"] is True


class _LoopbackTimestep:
    def last(self) -> bool:
        return False


class _LoopbackTaskEnv:
    def __init__(self) -> None:
        self.resets = 0
        self.steps: list[np.ndarray] = []
        self.cancel_gate = threading.Event()
        self.step_started = threading.Event()
        self.physics = object()
        self.robot = SimpleNamespace(get_qpos=lambda physics: np.zeros(7))
        self.task = SimpleNamespace(
            get_instruction=lambda: "pick the painting",
            should_terminate_episode=lambda physics: False,
        )

    def reset(self) -> None:
        self.resets += 1
        self.steps.clear()

    def step(self, action: np.ndarray) -> _LoopbackTimestep:
        self.steps.append(np.asarray(action, dtype=float))
        self.step_started.set()
        self.cancel_gate.wait(timeout=2.0)
        return _LoopbackTimestep()

    def close(self) -> None:
        pass


def test_env_client_loopback_reset_cancel_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _LoopbackTaskEnv()
    facade = VLABenchEnvFacade(
        env, metadata=env_identity(**IDENTITY), max_episode_steps=20
    )
    facade._gravity_compensated_action = lambda action: np.asarray(action, dtype=float)
    facade.robot_model_state = lambda: {"gripper_qpos": np.full(2, 0.04)}

    port = pick_free_port()
    server_thread = threading.Thread(
        target=lambda: facade.serve(
            transport="http", host="127.0.0.1", port=port, parent_watch=False
        ),
        daemon=True,
    )
    server_thread.start()
    deadline = time.monotonic() + 10.0
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError("loopback server did not start")
            time.sleep(0.01)

    rpc = HttpRpcClient(f"http://127.0.0.1:{port}")
    try:
        client = adapter.VLABenchEnvClient(rpc, **IDENTITY)
        assert env.resets == 1
        assert client.get_task_language() == "pick the painting"

        cancelled = {"flag": False}

        def check_cancelled() -> None:
            if cancelled["flag"]:
                raise RuntimeError("stop requested")

        def trigger_cancel() -> None:
            env.step_started.wait(timeout=2.0)
            cancelled["flag"] = True
            assert facade._motion_cancelled.wait(timeout=2.0)
            env.cancel_gate.set()

        canceller = threading.Thread(target=trigger_cancel, daemon=True)
        canceller.start()
        receipt = client.execute_gripper(
            gripper=1.0, steps=5, check_cancelled=check_cancelled
        )
        canceller.join(timeout=2.0)
        assert receipt["interrupted"] is True
        assert receipt["stop_reason"] == "cancelled"
        assert receipt["steps_used"] < 5

        calls: list[str] = []
        real_call = rpc.call

        def corrupt_pose_receipt(method: str, args=(), kwargs=None, *, timeout_s=None):
            calls.append(method)
            if method == "env.execute_gripper":
                return {"ok": True}
            return real_call(method, args, kwargs, timeout_s=timeout_s)

        monkeypatch.setattr(rpc, "call", corrupt_pose_receipt)
        with pytest.raises(RuntimeError, match="invalid VLABench motion receipt"):
            client.execute_gripper(gripper=1.0, steps=2)
        # Dispatched motion must not be retried after a malformed receipt.
        assert calls.count("env.execute_gripper") == 1
    finally:
        try:
            rpc.call("shutdown", timeout_s=2.0)
        except Exception:
            pass
        server_thread.join(timeout=5.0)


def test_spawn_env_server_targets_sim_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    class _FakeProcessDaemon:
        def __init__(
            self,
            name: str,
            command: list[str],
            *,
            env_overrides: dict[str, str],
            log_path: str,
        ) -> None:
            captured["name"] = name
            captured["command"] = command
            captured["env_overrides"] = env_overrides
            captured["log_path"] = log_path
            captured["started"] = False

        def start(self) -> None:
            captured["started"] = True

    monkeypatch.setattr("rpent.utils.daemon.ProcessDaemon", _FakeProcessDaemon)
    monkeypatch.setattr("rpent.utils.daemon.pick_free_port", lambda: 43210)
    daemon, _rpc = adapter.spawn_env_server(_cli_args(tmp_path), tmp_path)
    assert daemon is not None
    assert captured["started"] is True
    expected_script = str(
        get_repo_root() / "robots" / "vlabench" / "sim" / "env_server.py"
    )
    assert captured["command"][1] == expected_script
    assert "--parent-watch" in captured["command"]


def test_sim_subpackage_has_no_upward_robot_imports() -> None:
    sim_dir = Path(adapter.__file__).resolve().parent / "sim"
    py_files = sorted(sim_dir.glob("*.py"))
    assert py_files
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "robots.vlabench"
                    if alias.name.startswith("robots.vlabench."):
                        assert (
                            alias.name == "robots.vlabench.sim"
                            or alias.name.startswith("robots.vlabench.sim.")
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    assert node.level <= 1
                module = node.module or ""
                assert module != "robots.vlabench"
                if module.startswith("robots.vlabench."):
                    assert module == "robots.vlabench.sim" or module.startswith(
                        "robots.vlabench.sim."
                    )
