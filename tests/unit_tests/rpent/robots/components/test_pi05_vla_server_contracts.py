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


def test_xpolicylab_cli_without_rlinf_dependencies(monkeypatch):
    import importlib.util
    import sys
    from types import SimpleNamespace

    from rpent.robots.components import xpolicylab_vla_server as ws

    monkeypatch.setitem(sys.modules, "omegaconf", None)
    monkeypatch.setitem(sys.modules, "rlinf", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    spec = importlib.util.find_spec("rpent.robots.components.pi05_vla_server")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    calls = []
    monkeypatch.setattr(
        ws,
        "_connect_policy",
        lambda url, args: SimpleNamespace(close=lambda: calls.append("close")),
    )
    monkeypatch.setattr(
        ws.XPolicyLabVLAFacade,
        "serve",
        lambda self, **kwargs: (calls.append(kwargs), self.close()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server",
            "--policy-backend",
            "xpolicylab",
            "--task",
            "pick",
            "--policy-server-url",
            "ws://policy",
            "--port",
            "6000",
        ],
    )
    server.main()
    assert calls == [
        {"transport": "http", "host": "127.0.0.1", "port": 6000, "parent_watch": False},
        "close",
    ]


def test_rlinf_missing_omegaconf_reports_policy_environment(monkeypatch):
    import sys
    from types import SimpleNamespace

    from rpent.robots.components.pi05_vla_server import create_backend

    monkeypatch.setitem(sys.modules, "omegaconf", None)
    monkeypatch.setitem(sys.modules, "rlinf", None)
    with pytest.raises(
        RuntimeError,
        match="RLinf policy backend requires omegaconf.*Python environment",
    ) as error:
        create_backend(
            SimpleNamespace(
                policy_backend="rlinf",
                model_path="/checkpoint",
                embodiment="libero",
                model_backend="openpi_pytorch",
                norm_stats_path=None,
                repo_id=None,
            )
        )
    assert isinstance(error.value.__cause__, ImportError)


def test_dual_vla_prediction_follows_shared_component_rpc_contract(monkeypatch):
    import sys
    from types import ModuleType

    import torch

    from rpent.robots.components.pi05_vla_server import Pi05VLAFacade

    calls = []

    class Model:
        def cuda(self):
            return self

        def eval(self):
            return self

        def predict_action_batch(self, obs, mode):
            calls.append(mode)
            return torch.ones((20, 20)), None

    def get_model(cfg, torch_dtype):
        assert cfg.action_dim == 20 and cfg.openpi.num_images_in_input == 3
        assert cfg.openpi_data.repo_id == "test/dataset"
        assert cfg.openpi.config_name == "pi05_dualfranka_tcp_rot6d"
        assert cfg.num_action_chunks == cfg.openpi.action_chunk == 20
        assert cfg.openpi.train_expert_only is False
        assert cfg.openpi.detach_critic_input is True
        return Model()

    loader = ModuleType("rlinf.models.embodiment.openpi")
    loader.get_model = get_model
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    facade = Pi05VLAFacade(
        model_path="/unused/checkpoint",
        embodiment="dual_franka",
        repo_id="test/dataset",
    )
    try:
        assert calls == []
        actions = facade._dispatch("vla.predict", ({},), {"options": {"mode": "eval"}})
        assert actions.shape == (20, 20) and actions.dtype == np.float32
        assert calls == ["eval"]
    finally:
        facade.close()


@pytest.mark.parametrize("embodiment", ["libero", "dual_franka"])
def test_cli_forwards_model_configuration(monkeypatch, embodiment):
    import sys
    from types import SimpleNamespace

    from rpent.robots.components import pi05_vla_server as server

    received = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pi05_vla_server",
            "--embodiment",
            embodiment,
            "--model-path",
            "/checkpoint",
            "--repo-id",
            "test/dataset",
            "--norm-stats-path",
            "/stats",
            "--port",
            "6000",
        ],
    )

    def facade(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(serve=lambda **options: received.update(options))

    monkeypatch.setattr(server, "Pi05VLAFacade", facade)
    server.main()
    assert received["embodiment"] == embodiment
    assert received["repo_id"] == "test/dataset"
    assert received["norm_stats_path"] == "/stats"
    assert received["model_backend"] == "openpi_pytorch"
    assert received["port"] == 6000


def test_libero_preset_keeps_existing_defaults():
    from rpent.robots.components.pi05_vla_server import (
        PI05_EMBODIMENTS,
        build_model_cfg,
    )

    cfg = build_model_cfg("/checkpoint", PI05_EMBODIMENTS["libero"])
    assert cfg.openpi.config_name == "pi05_libero"
    assert cfg.action_dim == 7
    assert cfg.num_action_chunks == cfg.openpi.action_chunk == 5
    assert cfg.openpi.num_images_in_input == 2
    assert cfg.openpi.train_expert_only is True
    assert cfg.openpi.detach_critic_input is None
    assert "openpi_data" not in cfg


@pytest.mark.parametrize("backend", ["rlinf", "xpolicylab"])
def test_cli_selects_policy_backend(monkeypatch, backend):
    import sys
    from types import SimpleNamespace

    from rpent.robots.components import pi05_vla_server as server
    from rpent.robots.components import xpolicylab_vla_server as ws

    received = {}

    def rlinf(**kwargs):
        received["backend"] = "rlinf"
        return SimpleNamespace(serve=lambda **kw: received.update(kw))

    def xpolicy(args):
        received.update(
            backend="xpolicylab", task=args.task, output_dir=args.output_dir
        )
        return SimpleNamespace(
            serve=lambda **kw: received.update(kw), close=lambda: None
        )

    monkeypatch.setattr(server, "Pi05VLAFacade", rlinf)
    monkeypatch.setattr(ws, "XPolicyLabVLAFacade", xpolicy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "server",
            "--policy-backend",
            backend,
            "--output-dir",
            "run-output",
            "--model-path",
            "/checkpoint",
            "--task",
            "pick",
            "--port",
            "6000",
            "--parent-watch",
        ],
    )
    server.main()
    assert received["backend"] == backend
    if backend == "xpolicylab":
        assert received["output_dir"] == "run-output"
    assert received["port"] == 6000
    assert received["parent_watch"] is True


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
        from rpent.robots.components import pi05_vla_server as server
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
        sys.argv = ["server", "--policy-backend", "xpolicylab", "--task", "pick",
                    "--bench", "bench", "--ckpt", "weights", "--env-cfg-type", "arms",
                    "--action-type", "joint", "--policy-port", "1234"]
        if mode == "atexit":
            import argparse
            parser = argparse.ArgumentParser()
            ws.add_backend_args(parser)
            ws.XPolicyLabVLAFacade(parser.parse_args(sys.argv[3:]))
        else:
            server.main()
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
