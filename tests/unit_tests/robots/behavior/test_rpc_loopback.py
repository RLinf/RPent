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
import socket
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import pytest

from robots.behavior.dino_v2.encoder import DINOV2_DIMENSION
from robots.behavior.dino_v2.server import BehaviorDinoFacade
from robots.behavior.env_server import BehaviorEnvFacade
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.rpc import wait_for_ready
from rpent.utils.rpc.http_rpc import HttpRpcClient


class _ThreadRecordingBehaviorEnvFacade(BehaviorEnvFacade):
    def __init__(self) -> None:
        super().__init__(backend=object(), meta={"task_language": "test"})
        self.serve_thread_id: int | None = None
        self.business_thread_id: int | None = None

    def serve(self, **kwargs: Any) -> None:
        self.serve_thread_id = threading.get_ident()
        super().serve(**kwargs)

    def get_env_meta(self) -> dict[str, Any]:
        self.business_thread_id = threading.get_ident()
        return super().get_env_meta()


def test_behavior_facades_use_default_healthz_and_registered_metadata() -> None:
    facade = BehaviorEnvFacade(backend=object(), meta={"task_language": "test"})
    dino = BehaviorDinoFacade(
        encoder=object(),
        meta={"runtime": "behavior_dino", "dimension": DINOV2_DIMENSION},
    )

    assert facade._dispatch("healthz", (), {}) == {"status": "ok"}
    assert facade._dispatch("env.get_env_meta", (), {}) == {"task_language": "test"}
    assert "env.close_gripper" in facade._rpc
    assert "env.open_gripper" in facade._rpc
    assert "env.close" not in facade._rpc
    assert "env.open" not in facade._rpc
    with pytest.raises(ValueError, match="requires primitive arguments"):
        facade.close_gripper()
    assert dino._dispatch("healthz", (), {}) == {"status": "ok"}
    dino_meta = dino._dispatch("dino.get_meta", (), {})
    assert dino_meta["runtime"] == "behavior_dino"
    assert dino_meta["dimension"] == DINOV2_DIMENSION
    assert isinstance(dino_meta["pid"], int)


def test_behavior_env_facade_serve_dispatches_business_calls_on_serving_thread() -> (
    None
):
    facade = _ThreadRecordingBehaviorEnvFacade()
    port = pick_free_port()
    thread = threading.Thread(
        target=facade.serve,
        kwargs={
            "transport": "http",
            "host": "127.0.0.1",
            "port": port,
        },
        daemon=True,
    )
    thread.start()
    client = HttpRpcClient(f"http://127.0.0.1:{port}")

    try:
        deadline = time.monotonic() + 3.0
        while True:
            try:
                assert client.call("healthz", timeout_s=0.5) == {"status": "ok"}
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)

        assert client.call("env.get_env_meta", timeout_s=1.0) == {
            "task_language": "test"
        }
        assert facade.business_thread_id == facade.serve_thread_id
        assert facade.business_thread_id != threading.get_ident()
        assert client.call("shutdown", timeout_s=1.0) == {"ok": True}
    finally:
        client.close()
        facade._shutdown_event.set()
        thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert facade._closed
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.1)
        assert probe.connect_ex(("127.0.0.1", port)) != 0


def test_env_main_sigterm_uses_owning_thread_cleanup(tmp_path):
    port = pick_free_port()
    child_args = [
        "--task-name",
        "turning_on_radio",
        "--public-seed",
        "0",
        "--task-index",
        "0",
        "--activity-definition-id",
        "0",
        "--activity-instance-id",
        "242",
        "--scene-model",
        "test",
        "--max-episode-steps",
        "32",
        "--output-dir",
        str(tmp_path),
        "--behavior-repo",
        str(tmp_path),
        "--port",
        str(port),
        "--parent-watch",
    ]
    child_code = """
import signal
import threading
from robots.behavior import env_server, rlinf_env

closed_on = []
class Backend:
    def __init__(self, **kwargs):
        pass
    def close(self):
        closed_on.append(threading.get_ident())

rlinf_env.OfficialBehaviorBackend = Backend
previous = signal.getsignal(signal.SIGTERM)
env_server.main()
assert closed_on == [threading.get_ident()]
assert signal.getsignal(signal.SIGTERM) == previous
print("OWNING_THREAD_CLEANUP_OK", flush=True)
"""
    log_path = tmp_path / "parent_watch.log"
    daemon = ProcessDaemon(
        name="behavior_parent_watch",
        cmd=[sys.executable, "-c", child_code, *child_args],
        log_path=str(log_path),
    )
    client = HttpRpcClient(f"http://127.0.0.1:{port}")
    try:
        daemon.start()
        wait_for_ready(client, timeout_s=10.0, poll_interval_s=0.01, daemon=daemon)
        meta = client.call("env.get_env_meta", timeout_s=1.0)
        assert meta["task"] == 0
        assert meta["activity_instance_id"] == 242
        daemon.stop()
        log = log_path.read_text()
        assert daemon.poll() == 0, log
        assert "OWNING_THREAD_CLEANUP_OK" in log
        assert "Fatal Python error" not in log
    finally:
        client.close()
        daemon.stop()


def test_behavior_sidecars_import_without_planner_dependencies() -> None:
    script = """
import builtins

original_import = builtins.__import__

def block_planner(name, *args, **kwargs):
    if name.startswith("pydantic_ai"):
        raise ModuleNotFoundError("simulator runtime does not install pydantic_ai")
    return original_import(name, *args, **kwargs)

builtins.__import__ = block_planner
import robots.behavior.env_server
import robots.behavior.vla_server
import robots.behavior.dino_v2.server
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "override",
    [{"norm_stats_path": "/other/stats.json"}, {"repo_id": "other/dataset"}],
)
def test_behavior_vla_rejects_normalization_overrides(override: dict[str, str]) -> None:
    from robots.behavior.vla_server import BehaviorPi05VLAFacade

    with pytest.raises(ValueError, match="checkpoint-bundled normalization"):
        BehaviorPi05VLAFacade(
            model_path="/unused/checkpoint", embodiment="behavior", **override
        )


def test_behavior_uses_the_shared_pi05_registry_and_wire_contract() -> None:
    from robots.behavior.vla_client import BehaviorPi05VLAClient

    class FakeRpcClient:
        def call(self, method, *, args, timeout_s):
            assert method == "vla.predict"
            observation, options = args
            assert observation["main_images"].shape == (1, 720, 720, 3)
            assert observation["wrist_images"].shape == (1, 2, 480, 480, 3)
            assert observation["states"].shape == (1, 256)
            assert observation["task_descriptions"] == ["turn on the radio"]
            assert options == {"mode": "eval"}
            assert timeout_s == 600.0
            return np.zeros((1, 32, 23), dtype=np.float32)

    model = BehaviorPi05VLAClient(FakeRpcClient(), embodiment="behavior")
    action = model.predict(
        {
            "main_images": np.zeros((720, 720, 3), dtype=np.uint8),
            "wrist_images": np.zeros((2, 480, 480, 3), dtype=np.uint8),
            "states": np.zeros(256, dtype=np.float32),
            "task_descriptions": "turn on the radio",
        },
        options={"mode": "eval"},
    )
    assert action.shape == (32, 23)
    assert action.dtype == np.float32


@pytest.mark.parametrize(
    ("task_name", "mode", "seed", "instruction"),
    [
        (
            "turning_on_radio",
            "explore",
            0,
            "Turn on the radio receiver that's on the table in the living room.",
        ),
        (
            "picking_up_trash",
            "explore",
            0,
            "Put the three can of soda from the living room inside the tash can "
            "in the kitchen.",
        ),
        (
            "picking_up_trash",
            "eval",
            10,
            "Put the three can of soda from the living room inside the tash can "
            "in the kitchen.",
        ),
    ],
)
@pytest.mark.parametrize("batched", [False, True])
def test_runtime_preserves_rlinf_task_language(
    tmp_path,
    task_name: str,
    mode: str,
    seed: int,
    instruction: str,
    batched: bool,
) -> None:
    from robots.behavior import runtime

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--explore", action="store_true")
    runtime.add_cli_args(parser, use_dashboard=False)
    cli_args = [
        "--task-name",
        task_name,
        "--public-seed",
        str(seed),
        "--behavior-repo",
        str(tmp_path / "RLinf"),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    if mode == "explore":
        cli_args.append("--explore")
    args = parser.parse_args(cli_args)
    config = runtime.parse_config(args)
    meta = runtime.env_runtime_contract(args)
    assert meta["task_language"] == instruction
    for key in ("task_language", "task_instruction", "instruction"):
        assert config.prompt_vars[key] == instruction

    class Rpc:
        language = instruction

        def call(self, method, **kwargs):
            if method == "env.get_env_meta":
                return meta
            assert method == "env.reset"
            text = [self.language] if batched else self.language
            return {"task_descriptions": text}, {"done": {"success": False}}

    rpc = Rpc()
    connected = runtime._connect_env(args, rpc, config.output_dir)
    expected = [instruction] if batched else instruction
    assert connected["initial_observation"]["task_descriptions"] == expected
    assert connected["env"].official_success_latched is False

    rpc.language = "Put the three soda cans from the living room inside the trash can in the kitchen."
    with pytest.raises(RuntimeError, match="task language does not match TaskSpec"):
        runtime._connect_env(args, rpc, config.output_dir)


def test_env_endpoint_discovery_uses_actual_bind_and_ignores_old_log(tmp_path):
    from robots.behavior.runtime import _wait_for_server_endpoint

    log = tmp_path / "env.log"
    log.write_text("RPC server listening on http://127.0.0.1:1\n")
    offset = log.stat().st_size
    daemon = ProcessDaemon(
        "test_env",
        [
            sys.executable,
            "-c",
            "from types import SimpleNamespace; "
            "from robots.behavior.env_server import BehaviorEnvFacade; "
            "print('Ray started; no application endpoint yet', flush=True); "
            "BehaviorEnvFacade(backend=SimpleNamespace(close=lambda: None), "
            "meta={'task_language': 'test'}).serve("
            "transport='http', host='127.0.0.1', port=0, parent_watch=True)",
        ],
        log_path=str(log),
    )
    daemon.start()
    rpc = None
    try:
        endpoint = _wait_for_server_endpoint(daemon, log_offset=offset)
        assert endpoint != "http://127.0.0.1:1"
        rpc = HttpRpcClient(endpoint)
        assert rpc.call("healthz") == {"status": "ok"}
        assert rpc.call("env.get_env_meta") == {"task_language": "test"}
        assert rpc.call("shutdown") == {"ok": True}
    finally:
        if rpc is not None:
            rpc.close()
        daemon.stop()


def test_env_endpoint_discovery_reports_early_exit(tmp_path):
    from types import SimpleNamespace

    from robots.behavior.runtime import _wait_for_server_endpoint

    log = tmp_path / "env.log"
    log.write_text("initialization failed\n")
    daemon = SimpleNamespace(log_path=str(log), name="test_env", poll=lambda: 2)
    with pytest.raises(RuntimeError, match="exited with code 2"):
        _wait_for_server_endpoint(daemon)


def test_behavior_external_sidecar_python_gets_source_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robots.behavior import runtime

    monkeypatch.setenv("PYTHONPATH", "/tmp/existing-pythonpath")

    env = runtime._behavior_subprocess_env(
        cuda_device="2",
        ROBOT_PLATFORM="BEHAVIOR",
    )

    assert env["ROBOT_PLATFORM"] == "BEHAVIOR"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert "PYTHONPATH" not in env
    assert env["RPENT_REPO_ROOT"] == str(runtime.get_repo_root())


@pytest.mark.parametrize("action_chunk", [32, 16])
def test_behavior_vla_connect_checks_shared_status(action_chunk: int, tmp_path) -> None:
    from robots.behavior import runtime
    from robots.behavior.vla_client import BehaviorPi05VLAClient

    class Rpc:
        def call(self, method, *, timeout_s):
            assert method == "vla.status"
            return {"action_chunk": action_chunk}

    if action_chunk != 32:
        with pytest.raises(RuntimeError, match="action_chunk|32"):
            runtime._connect_vla(argparse.Namespace(), Rpc(), tmp_path)
    else:
        model = runtime._connect_vla(argparse.Namespace(), Rpc(), tmp_path)["model"]
        assert isinstance(model, BehaviorPi05VLAClient)


def test_behavior_vla_connect_rejects_dino_healthz_endpoint(tmp_path) -> None:
    from robots.behavior import runtime

    dino = BehaviorDinoFacade(
        encoder=object(), meta={"runtime": "behavior_dino", "dimension": 384}
    )

    class Rpc:
        def call(self, method, args=(), kwargs=None, *, timeout_s=None):
            return dino._dispatch(method, args, kwargs or {})

    rpc = Rpc()
    assert rpc.call("healthz") == {"status": "ok"}
    with pytest.raises((ValueError, RuntimeError), match="vla.status"):
        runtime._connect_vla(argparse.Namespace(), rpc, tmp_path)


@pytest.mark.parametrize(
    ("action_chunk", "message"), [(31, "must return"), (33, "exceeds")]
)
def test_behavior_vla_wrong_horizon_never_reaches_env(
    action_chunk: int, message: str
) -> None:
    from robots.behavior.tools import BehaviorPrimitives
    from robots.behavior.vla_client import BehaviorPi05VLAClient

    class Rpc:
        def call(self, method, *, args, timeout_s):
            assert method == "vla.predict"
            return np.zeros((1, action_chunk, 23), dtype=np.float32)

    class Env:
        total_env_steps = 0
        official_success_latched = False

        def __init__(self):
            self.calls = 0

        def chunk_step(self, actions, *, return_all_frames=False):
            self.calls += 1
            raise AssertionError("invalid VLA actions reached ENV")

    env = Env()
    primitives = BehaviorPrimitives(
        env=env,
        model=BehaviorPi05VLAClient(Rpc(), embodiment="behavior"),
        max_episode_steps=100,
        initial_observation={
            "main_images": np.zeros((2, 2, 3), dtype=np.uint8),
            "wrist_images": np.zeros((2, 2, 2, 3), dtype=np.uint8),
            "states": np.zeros(256, dtype=np.float32),
            "task_descriptions": "turn on the radio",
        },
        task_name="turning_on_radio",
        public_seed=0,
    )
    with pytest.raises(ValueError, match=message):
        primitives.pi0_nav_pick(instruction="turn on the radio", chunks=1)
    assert env.calls == 0
