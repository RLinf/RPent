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

"""MetaWorld memory defaults and renderer launch contracts."""

import argparse
import sys
from dataclasses import replace
from unittest.mock import Mock

import pytest

from robots.metaworld import env_server, robot_spec


@pytest.mark.parametrize("dashboard", [False, True])
@pytest.mark.parametrize("profile", [None, "hf"])
def test_memory_defaults_and_explicit_hf_use_cli_validation(
    monkeypatch, capsys, dashboard, profile
):
    from rpent.cli import dashboard as dashboard_cli
    from rpent.cli import main as cli

    class Captured(Exception):
        pass

    def capture(args, *a, **kw):
        assert args.memory_profile == "local"
        raise Captured

    spec = robot_spec.get_robot_spec()
    monkeypatch.setattr(
        cli, "get_robot_spec", lambda name: replace(spec, parse_config=capture)
    )
    monkeypatch.setattr(dashboard_cli, "run_dashboard_session", capture)
    argv = [
        "rpent",
        "--robot",
        "metaworld",
        "--planner",
        "api",
        "--model",
        "openai-chat:test",
    ]
    if dashboard:
        argv += ["--dashboard"]
    if profile:
        argv += ["--memory-profile", profile]
    monkeypatch.setattr(sys, "argv", argv)
    if profile:
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 2
        assert (
            "MetaWorld supports only --memory-profile local" in capsys.readouterr().err
        )
    else:
        with pytest.raises(Captured):
            cli.main()


@pytest.mark.parametrize("backend", [None, "osmesa", "egl"])
@pytest.mark.parametrize("device", [None, 3])
def test_spawn_preserves_backend_and_forwards_physical_device(
    monkeypatch, tmp_path, backend, device
):
    if backend is None:
        monkeypatch.delenv("MUJOCO_GL", raising=False)
    else:
        monkeypatch.setenv("MUJOCO_GL", backend)
    parser = argparse.ArgumentParser()
    robot_spec._add_cli_args(parser)
    args = parser.parse_args([] if device is None else ["--cuda-device", str(device)])
    daemon = Mock()
    monkeypatch.setattr(robot_spec, "ProcessDaemon", daemon)
    robot_spec._spawn(args, tmp_path)
    command = daemon.call_args.args[1]
    assert daemon.call_args.kwargs["env_overrides"]["MUJOCO_GL"] == (backend or "egl")
    if device is None:
        assert "--cuda-device" not in command
    else:
        assert command[command.index("--cuda-device") + 1] == str(device)
    daemon.return_value.start.assert_called_once()


@pytest.mark.parametrize("backend", [None, "egl", "osmesa"])
@pytest.mark.parametrize("device", [None, 3])
def test_server_maps_device_before_environment_construction(
    monkeypatch, backend, device
):
    if backend is None:
        monkeypatch.delenv("MUJOCO_GL", raising=False)
    else:
        monkeypatch.setenv("MUJOCO_GL", backend)
    events = []
    monkeypatch.setattr(
        "rpent.utils.egl.configure_egl_device", lambda d: events.append(("egl", d))
    )
    env = Mock()

    def construct(*args):
        events.append(("construct",))
        return env

    monkeypatch.setattr(env_server, "MetaWorldEnvFacade", construct)
    monkeypatch.setattr(
        sys,
        "argv",
        ["env"] + ([] if device is None else ["--cuda-device", str(device)]),
    )
    env_server.main()
    expected = (
        [("egl", device)] if device is not None and backend in (None, "egl") else []
    )
    assert events == [*expected, ("construct",)]
    env.serve.assert_called_once()
    env.close.assert_called_once()
