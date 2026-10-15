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


@pytest.mark.parametrize("use_env", [True, False])
def test_uniwam_launcher_uses_resource_root_interpreter_and_cuda(
    monkeypatch, tmp_path, use_env
):
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
    root = tmp_path / "bundle"
    root.mkdir()
    for name in ("policy", "checkpoint", "wan", "vlm"):
        resource = tmp_path / name
        resource.mkdir()
        (root / name).symlink_to(resource, target_is_directory=True)
    (tmp_path / "checkpoint" / "pytorch_model").mkdir()
    monkeypatch.setenv("UNIWAM_MODEL_PATH", str(root) if use_env else "/unused")
    args = arguments(
        "--vla-backend",
        "uniwam",
        "--uniwam-python",
        "/model-env/bin/python",
        "--vla-cuda-device",
        "2",
        *([] if use_env else ["--vla-model-path", str(root)]),
    )
    owned, endpoint = robot_spec._spawn_uniwam_server(args, tmp_path)
    assert owned is created[0] and owned.started
    assert owned.command[0] == "/model-env/bin/python"
    for flag, relative in (
        ("--model-path", "checkpoint/pytorch_model"),
        ("--policy-dir", "policy"),
        ("--wan-path", "wan"),
        ("--vlm-path", "vlm"),
    ):
        assert owned.command[owned.command.index(flag) + 1] == str(tmp_path / relative)
    assert owned.env["CUDA_VISIBLE_DEVICES"] == "2"
    assert endpoint == "socket://127.0.0.1:12345"


def test_failed_uniwam_start_cleans_up_owned_environment(monkeypatch, tmp_path):
    stopped = []
    env = SimpleNamespace(stop=lambda: stopped.append("env"))
    monkeypatch.setattr(robot_spec, "_spawn_env_server", lambda *a: (env, object()))
    monkeypatch.delenv("UNIWAM_MODEL_PATH", raising=False)
    with pytest.raises(RuntimeError, match="UNIWAM_MODEL_PATH"):
        robot_spec._init_runtime(
            arguments("--vla-backend", "uniwam"),
            tmp_path,
            NullDashboardEventSink(),
            None,
        )
    assert stopped == ["env"]
