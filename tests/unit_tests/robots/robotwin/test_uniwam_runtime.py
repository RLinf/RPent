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


"""UniWAM runtime selection and resource ownership contracts."""

from argparse import ArgumentParser
from types import SimpleNamespace

import pytest

from robots.robotwin import robot_spec
from robots.robotwin.uniwam_client import UniWAMClient
from rpent.dashboard.events import NullDashboardEventSink
from rpent.utils import daemon as daemon_module
from rpent.utils import rpc as rpc_module


def arguments(*extra):
    parser = ArgumentParser()
    robot_spec._add_cli_args(parser, False)
    return parser.parse_args(
        ["--task-name", "beat_block_hammer", "--seed", "7", *extra]
    )


def test_backend_default_and_checkpoint_environment_are_separate(monkeypatch, tmp_path):
    args = arguments()
    assert args.vla_backend == "lingbot"
    monkeypatch.setenv("LINGBOT_MODEL_PATH", str(tmp_path))
    monkeypatch.setattr(robot_spec, "_validate_local_model_files", lambda path: None)
    monkeypatch.setattr(robot_spec, "_resolve_lingbot_robot_config", lambda *args: None)
    assert robot_spec._resolve_vla_runtime_path(args) == tmp_path


def test_uniwam_external_endpoint_needs_no_local_model_paths(monkeypatch, tmp_path):
    rpc = object()
    monkeypatch.setattr(rpc_module, "make_rpc_client", lambda endpoint: rpc)
    monkeypatch.setattr(robot_spec, "try_wait_server", lambda *args, post_fn: post_fn())
    owned, runtime = robot_spec._init_runtime(
        arguments(
            "--vla-backend", "uniwam", "--vla-endpoint", "socket://localhost:1234"
        ),
        tmp_path,
        NullDashboardEventSink(),
        {"vla"},
    )
    assert owned == []
    assert isinstance(runtime["model"], UniWAMClient)
    assert runtime["vla_backend"] == "uniwam"


def test_uniwam_launcher_uses_selected_interpreter_and_cuda(monkeypatch, tmp_path):
    created = []

    class Daemon:
        def __init__(self, name, command, **kwargs):
            self.command = command
            self.env = kwargs["env_overrides"]
            self.started = False
            created.append(self)

        def start(self):
            self.started = True

    monkeypatch.setattr(daemon_module, "ProcessDaemon", Daemon)
    monkeypatch.setattr(daemon_module, "pick_free_port", lambda: 12345)
    monkeypatch.setattr(rpc_module, "make_rpc_client", lambda endpoint: endpoint)
    monkeypatch.setenv("LINGBOT_MODEL_PATH", "/not-the-uniwam-checkpoint")
    monkeypatch.setenv("UNIWAM_MODEL_PATH", str(tmp_path))
    args = arguments(
        "--vla-backend",
        "uniwam",
        "--uniwam-python",
        "/model-env/bin/python",
        "--uniwam-policy-dir",
        str(tmp_path),
        "--uniwam-wan-path",
        str(tmp_path),
        "--uniwam-vlm-path",
        str(tmp_path),
        "--vla-cuda-device",
        "2",
    )
    owned, endpoint = robot_spec._spawn_uniwam_server(args, tmp_path)
    assert owned is created[0] and owned.started
    assert owned.command[0] == "/model-env/bin/python"
    assert owned.command[owned.command.index("--model-path") + 1] == str(tmp_path)
    assert owned.env["CUDA_VISIBLE_DEVICES"] == "2"
    assert endpoint == "socket://127.0.0.1:12345"


def test_failed_uniwam_start_cleans_up_owned_environment(monkeypatch, tmp_path):
    stopped = []
    env = SimpleNamespace(stop=lambda: stopped.append("env"))
    monkeypatch.setattr(robot_spec, "_spawn_env_server", lambda *a: (env, object()))

    def fail(*args):
        raise ValueError("missing weights")

    monkeypatch.setattr(robot_spec, "_spawn_uniwam_server", fail)
    with pytest.raises(RuntimeError, match="spawn failed"):
        robot_spec._init_runtime(
            arguments("--vla-backend", "uniwam"),
            tmp_path,
            NullDashboardEventSink(),
            None,
        )
    assert stopped == ["env"]
