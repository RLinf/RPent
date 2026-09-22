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

import numpy as np
import pytest


@pytest.mark.parametrize("mask", [None, "", "2,3", "GPU-example"])
def test_policy_without_gpu_flag_preserves_visibility(monkeypatch, tmp_path, mask):
    import argparse
    import json
    import os
    import sys
    from types import SimpleNamespace

    from rpent.robots.components import xpolicylab_vla_server as server

    root = tmp_path / "XPolicyLab"
    policy = root / "policy/Pi_05"
    policy.mkdir(parents=True)
    (policy / "setup_eval_policy_server.sh").touch()
    info = root / "utils/robot"
    info.mkdir(parents=True)
    (info / "_robot_info.json").write_text(
        json.dumps({"arx_x5": {"arm_dim": [6, 6], "ee_dim": [1, 1]}})
    )
    parser = argparse.ArgumentParser()
    server.add_backend_args(parser)
    args = parser.parse_args(
        [
            "--policy-root",
            str(policy),
            "--env-cfg-type",
            "arx_x5",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    recorded = {}

    def daemon(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(start=lambda: None)

    monkeypatch.setattr(server, "ProcessDaemon", daemon)
    monkeypatch.setattr(server, "_wait_for_port", lambda *args: None)
    server._spawn_policy_server(args)
    assert recorded["cmd"][:3] == [
        sys.executable,
        "-u",
        str(root / "setup_policy_server.py"),
    ]
    assert "action_dim=14" in recorded["cmd"]
    assert "CUDA_VISIBLE_DEVICES" not in recorded.get("env_overrides", {})
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == mask


def test_xpolicylab_cli_without_rlinf_dependencies(monkeypatch):
    import importlib.util
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "omegaconf", None)
    monkeypatch.setitem(sys.modules, "rlinf", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    spec = importlib.util.find_spec("rpent.robots.components.xpolicylab_vla_server")
    ws = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ws)
    calls = []
    monkeypatch.setattr(
        ws,
        "_connect_policy",
        lambda url, args: SimpleNamespace(close=lambda: calls.append("close")),
    )
    monkeypatch.setattr(
        ws.XPolicyLabVLAFacade,
        "serve",
        lambda self, **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server",
            "--task",
            "pick",
            "--policy-server-url",
            "ws://policy",
            "--port",
            "6000",
        ],
    )
    ws.main()
    assert calls == [
        {"transport": "http", "host": "127.0.0.1", "port": 6000, "parent_watch": False},
        "close",
    ]


def test_cli_forwards_configuration_and_restores_sigterm(monkeypatch):
    import os
    import signal
    import sys
    from types import SimpleNamespace

    from rpent.robots.components import xpolicylab_vla_server as server

    received = {}
    options = {
        "policy-root": "/policy",
        "bench": "RoboDojo",
        "evaluation-id": "run-1",
        "task": "pick",
        "env-cfg-type": "arx_x5",
        "action-type": "joint",
        "ckpt": "weights",
        "policy-gpu": "2",
        "policy-port": "1234",
        "policy-server-url": "ws://policy",
        "output-dir": "run-output",
        "cuda-device": "3",
        "transport": "socket",
        "host": "localhost",
        "port": "6000",
    }
    argv = ["xpolicylab_vla_server", "--parent-watch"]
    for name, value in options.items():
        argv.extend([f"--{name}", value])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    previous = signal.getsignal(signal.SIGTERM)

    def facade(args):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
        received.update(vars(args))
        return SimpleNamespace(
            serve=lambda **kwargs: received.update(serve=kwargs),
            close=lambda: received.update(closed=True),
        )

    monkeypatch.setattr(server, "XPolicyLabVLAFacade", facade)
    server.main()
    for name, value in options.items():
        assert str(received[name.replace("-", "_")]) == value
    assert received["serve"] == {
        "transport": "socket",
        "host": "localhost",
        "port": 6000,
        "parent_watch": True,
    }
    assert received["closed"] is True
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.parametrize("wrapped", [True, False])
def test_xpolicylab_rpc_preserves_observations_actions_and_reset(monkeypatch, wrapped):
    import argparse
    from types import SimpleNamespace

    from rpent.robots.components import xpolicylab_vla_server as ws

    parser = argparse.ArgumentParser()
    ws.add_backend_args(parser)
    args = parser.parse_args(["--task", "pick", "--policy-server-url", "ws://policy"])
    calls = []
    actions = np.zeros((5, 14), dtype=np.float64)

    def call(**kwargs):
        calls.append(kwargs)
        return {"actions": actions} if wrapped else actions

    monkeypatch.setattr(
        ws,
        "_connect_policy",
        lambda *a: SimpleNamespace(call=call, close=lambda: calls.append("close")),
    )
    monkeypatch.setattr(ws, "_spawn_policy_server", lambda *a: pytest.fail("borrowed"))
    facade = ws.XPolicyLabVLAFacade(args)
    obs = {
        "vision": {
            name: {"color": np.zeros((2, 2, 3))}
            for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")
        },
        "state": {"joints": np.zeros(14)},
        "instruction": "pick",
    }
    try:
        assert facade._dispatch("vla.predict", (obs,), {}) is actions
        assert calls[0]["obs"] is obs
        assert [item["func_name"] for item in calls] == ["update_obs", "get_action"]
        assert facade._dispatch("reset", (), {}) == {"ok": True}
        assert calls[-1] == {"func_name": "reset"}
    finally:
        facade.close()
    assert calls[-1] == "close"


def test_xpolicylab_metadata(monkeypatch):
    import argparse
    import sys
    from types import ModuleType

    from rpent.robots.components import xpolicylab_vla_server as ws

    module = ModuleType("client_server.ws.model_client")
    module.WsModelClient = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, module.__name__, module)
    args = argparse.Namespace(task="pick", evaluation_id="run-1")
    assert ws._connect_policy("ws://policy", args) == {
        "url": "ws://policy",
        "evaluation_id": "run-1",
        "trial_id": "pick-vla",
        "action_case_id": "pick_case",
        "repeat_index": None,
    }


@pytest.mark.parametrize("exit_mode", ["normal", "sigterm", "error", "atexit"])
def test_owned_policy_exits_with_server(tmp_path, exit_mode):
    import os
    import subprocess
    import sys
    import textwrap

    pid_file = tmp_path / "policy.pid"
    script = textwrap.dedent("""
        import os, signal, sys
        from pathlib import Path
        from types import SimpleNamespace
        from rpent.utils.daemon import ProcessDaemon
        from rpent.robots.components import xpolicylab_vla_server as ws

        mode, pid_file = sys.argv[1:]
        def spawn(args):
            daemon = ProcessDaemon("fake_policy", [sys.executable, "-c", "import time; time.sleep(60)"])
            daemon.start()
            Path(pid_file).write_text(str(daemon._proc.pid))
            return daemon
        ws._spawn_policy_server = spawn
        ws._connect_policy = lambda *a: SimpleNamespace(close=lambda: None)
        def serve(self, **kwargs):
            if mode == "sigterm":
                os.kill(os.getpid(), signal.SIGTERM)
            if mode == "error":
                raise RuntimeError("serve failed")
        ws.XPolicyLabVLAFacade.serve = serve
        sys.argv = ["server", "--task", "pick",
                    "--bench", "bench", "--ckpt", "weights", "--env-cfg-type", "arms",
                    "--action-type", "joint", "--policy-port", "1234"]
        if mode == "atexit":
            import argparse
            parser = argparse.ArgumentParser()
            ws.add_backend_args(parser)
            ws.XPolicyLabVLAFacade(parser.parse_args(sys.argv[1:]))
        else:
            ws.main()
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, exit_mode, str(pid_file)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert (
        result.returncode
        == {"normal": 0, "sigterm": 143, "error": 1, "atexit": 0}[exit_mode]
    ), result.stderr
    pid = int(pid_file.read_text())
    try:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


@pytest.mark.parametrize("failure", [None, "wait", "connect", "base", "close"])
@pytest.mark.parametrize("explicit_output", [False, True])
def test_xpolicylab_owned_launcher_cleanup(
    monkeypatch, tmp_path, failure, explicit_output
):
    import argparse
    from types import SimpleNamespace

    from rpent.robots.components import xpolicylab_vla_server as ws

    (tmp_path / "setup_eval_policy_server.sh").touch()
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "run" if explicit_output else tmp_path
    parser = argparse.ArgumentParser()
    ws.add_backend_args(parser)
    args = parser.parse_args(
        [
            "--policy-root",
            str(tmp_path),
            "--bench",
            "RoboDojo",
            "--task",
            "pick",
            "--ckpt",
            "weights",
            "--env-cfg-type",
            "arx_x5",
            "--action-type",
            "joint",
            "--policy-port",
            "1234",
            "--policy-gpu",
            "2",
        ]
        + (["--output-dir", str(output_dir)] if explicit_output else [])
    )
    calls = []
    recorded = {}

    def daemon(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(
            start=lambda: calls.append("start"), stop=lambda **kw: calls.append("stop")
        )

    def wait(*a):
        if failure == "wait":
            raise RuntimeError("wait failed")

    def connect(*a):
        if failure == "connect":
            raise RuntimeError("connect failed")

        def close():
            calls.append("close")
            if failure == "close":
                raise RuntimeError("client close failed")

        return SimpleNamespace(close=close)

    monkeypatch.setattr(ws, "ProcessDaemon", daemon)
    monkeypatch.setattr(ws, "_wait_for_port", wait)
    monkeypatch.setattr(ws, "_connect_policy", connect)
    if failure == "base":

        def fail_init(self):
            raise RuntimeError("base failed")

        monkeypatch.setattr(ws.BaseVLAFacade, "__init__", fail_init)
    if failure in {"wait", "connect", "base"}:
        with pytest.raises(RuntimeError, match=failure):
            ws.XPolicyLabVLAFacade(args)
        assert calls == ["start", "stop"] + (["close"] if failure == "base" else [])
    else:
        facade = ws.XPolicyLabVLAFacade(args)
        facade.close()
        facade.close()
        assert calls == ["start", "stop", "close"]
    assert recorded["log_path"] == str(output_dir / "vla_server.log")
    assert output_dir.is_dir()
    assert recorded["cmd"] == [
        "bash",
        str(tmp_path / "setup_eval_policy_server.sh"),
        "RoboDojo",
        "pick",
        "weights",
        "arx_x5",
        "joint",
        "0",
        "2",
        "uv",
        "1234",
        "localhost",
    ]
