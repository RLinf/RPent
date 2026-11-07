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


"""UniWAM selection and worker ownership through the shared AM runtime."""

import sys
from argparse import ArgumentParser
from types import SimpleNamespace

import pytest

from robots.robotwin import robot_spec
from robots.robotwin.wam_client import RoboTwinWAMClient
from rpent.dashboard.events import NullDashboardEventSink
from rpent.robots.components import am_runtime
from rpent.robots.components.uniwam.adapter import make_adapter


def arguments(*extra):
    parser = ArgumentParser()
    robot_spec._add_cli_args(parser, False)
    return parser.parse_args(
        ["--task-name", "beat_block_hammer", "--seed", "7", *extra]
    )


def bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    for name in ("policy", "checkpoint", "wan", "vlm"):
        resource = tmp_path / name
        resource.mkdir()
        (root / name).symlink_to(resource, target_is_directory=True)
    (tmp_path / "checkpoint/pytorch_model").mkdir()
    (tmp_path / "checkpoint/config.json").write_text("{}")
    (tmp_path / "policy/deploy_policy.py").touch()
    return root


def test_backend_default_and_checkpoint_environment_are_separate(monkeypatch):
    monkeypatch.setenv("LINGBOT_MODEL_PATH", "/lingbot")
    args = arguments()
    assert args.am_backend == "lingbot"
    assert am_runtime.select_am(args, "robotwin").checkpoint == "/lingbot"
    with pytest.raises(ValueError, match="am-checkpoint"):
        am_runtime.select_am(arguments("--am-backend", "uniwam"), "robotwin")


def test_uniwam_external_endpoint_needs_no_local_model_paths(monkeypatch, tmp_path):
    meta, _ = make_adapter("robotwin", "external")
    rpc = SimpleNamespace(call=lambda *a, **kw: meta, close=lambda: None)
    monkeypatch.setattr(am_runtime, "make_rpc_client", lambda *a, **kw: rpc)
    monkeypatch.setattr(robot_spec, "try_wait_server", lambda *args, post_fn: post_fn())
    owned, runtime = robot_spec._init_runtime(
        arguments("--vla-backend", "uniwam", "--vla-endpoint", "http://localhost:1234"),
        tmp_path,
        NullDashboardEventSink(),
        {"wam"},
    )
    assert owned == []
    assert isinstance(runtime["model"], RoboTwinWAMClient)
    assert runtime["policy_backend"] == "uniwam"
    assert runtime["policy_kind"] == "wam"


@pytest.mark.parametrize("legacy", [True, False])
def test_uniwam_launcher_uses_resource_root_interpreter_and_cuda(
    monkeypatch, tmp_path, legacy
):
    created = []

    class Daemon:
        def __init__(self, name, cmd, **kwargs):
            self.command = cmd
            self.env = kwargs["env_overrides"]
            self.started = False
            created.append(self)

        def start(self):
            self.started = True

    monkeypatch.setattr(am_runtime, "ProcessDaemon", Daemon)
    monkeypatch.setattr(am_runtime, "pick_free_port", lambda: 12345)
    monkeypatch.setattr(am_runtime, "make_rpc_client", lambda endpoint, **kw: endpoint)
    root = bundle(tmp_path)
    args = arguments(
        "--vla-backend" if legacy else "--am-backend",
        "uniwam",
        "--vla-model-path" if legacy else "--am-checkpoint",
        str(root),
        "--am-python",
        sys.executable,
        "--am-cuda-device",
        "2",
    )
    config = am_runtime.select_am(args, "robotwin")
    owned, endpoint = am_runtime.start_wam_service(config, tmp_path)
    assert owned is created[0] and owned.started
    assert owned.command[0] == sys.executable
    assert owned.command[1:3] == ["-m", "rpent.robots.components.uniwam.server"]
    assert owned.command[owned.command.index("--checkpoint") + 1] == str(root)
    assert owned.env["CUDA_VISIBLE_DEVICES"] == "2"
    assert endpoint == "http://127.0.0.1:12345"


def test_failed_uniwam_start_cleans_up_owned_environment(monkeypatch, tmp_path):
    stopped = []
    env = SimpleNamespace(stop=lambda: stopped.append("env"))
    monkeypatch.setattr(robot_spec, "_spawn_env_server", lambda *a: (env, object()))

    def fail(*args):
        raise RuntimeError("worker start failed")

    monkeypatch.setattr(robot_spec, "start_wam_service", fail)
    with pytest.raises(RuntimeError, match="worker start failed"):
        robot_spec._init_runtime(
            arguments(
                "--am-backend", "uniwam", "--am-endpoint", "http://localhost:1234"
            ),
            tmp_path,
            NullDashboardEventSink(),
            None,
        )
    assert stopped == ["env"]


@pytest.mark.parametrize("explore", [False, True])
def test_uniwam_preserves_exploration_and_memory(explore):
    args = arguments("--am-backend", "uniwam", "--am-endpoint", "http://localhost:1234")
    args.explore = explore
    args.output_dir = None
    args.memory_dir = None
    args.memory_profile = "local" if explore else "hf"
    config = robot_spec._parse_config(args)
    assert config.prompt_vars["policy_backend"] == "uniwam"
    assert config.prompt_vars["mode"] == ("explore" if explore else "eval")
    assert config.prompt_vars["memory_profile"] == args.memory_profile
