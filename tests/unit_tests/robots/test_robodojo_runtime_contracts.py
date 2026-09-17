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

"""Offline RoboDojo configuration and read-only dispatch contracts."""

import argparse
import os
import sys
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest

from robots.robodojo import robot_spec, tools
from rpent.tools.toolkit import _is_readonly


def _args(*flags):
    parser = argparse.ArgumentParser()
    robot_spec._add_cli_args(parser, False)
    return parser.parse_args(["--task", "put_bottles_into_dustbin", *flags])


def test_explicit_paths_do_not_read_workspace_or_mutate_parent(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBODOJO_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", "inherited")
    args = _args(
        "--source-root",
        str(tmp_path / "source"),
        "--xpolicylab-root",
        str(tmp_path / "policy"),
    )
    before = dict(os.environ)
    env = robot_spec._runtime_overrides(args)
    assert str(tmp_path / "source") in env["PYTHONPATH"].split(os.pathsep)
    assert str(tmp_path / "policy") in env["PYTHONPATH"].split(os.pathsep)
    assert env["ROBODOJO_PI05_POLICY_ROOT"] == str(tmp_path / "policy/policy/Pi_05")
    assert dict(os.environ) == before
    assert args.sim_python == args.pi05_python == sys.executable


def test_local_spawn_requires_source_root():
    with pytest.raises(ValueError, match="--source-root"):
        robot_spec._runtime_overrides(_args())


def test_task_inventory_uses_explicit_checkout(tmp_path, monkeypatch):
    from robots.robodojo import tasks

    configs = tmp_path / "task/RoboDojo/config"
    configs.mkdir(parents=True)
    (configs / "example.yml").write_text("{}")
    (configs / "_task.yml").write_text("{}")
    monkeypatch.setenv("ROBODOJO_SOURCE_ROOT", "/ignored")
    assert tasks.list_tasks(tmp_path) == ["example"]
    assert tasks.validate_task("example", tmp_path) is None
    assert "unknown RoboDojo task" in tasks.validate_task("other", tmp_path)
    assert tasks.list_tasks() == []
    assert tasks.task_summary("remote") == {"task": "remote"}


@pytest.mark.parametrize("component", ["env", "vla"])
def test_borrowed_endpoint_needs_no_local_paths(monkeypatch, tmp_path, component):
    client = object()
    monkeypatch.setattr(robot_spec, "make_rpc_client", lambda endpoint: client)
    args = _args(f"--{component}-endpoint", "http://example:1234")
    assert getattr(robot_spec, f"_spawn_{component}_server")(args, tmp_path) == (
        None,
        client,
    )


@pytest.mark.parametrize(
    "component,flag", [("env", "--sim-python"), ("vla", "--pi05-python")]
)
def test_spawn_uses_explicit_interpreter_and_child_paths(
    monkeypatch, tmp_path, component, flag
):
    recorded = {}

    class Daemon:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

        def start(self):
            recorded["started"] = True

    monkeypatch.setattr(robot_spec, "ProcessDaemon", Daemon)
    args = _args("--source-root", str(tmp_path), flag, sys.executable)
    getattr(robot_spec, f"_spawn_{component}_server")(args, tmp_path)
    assert recorded["cmd"][0] == sys.executable
    assert recorded["started"]
    assert recorded["env_overrides"]["ROBODOJO_PI05_POLICY_ROOT"] == str(
        tmp_path / "XPolicyLab/policy/Pi_05"
    )


def test_readers_are_readonly_and_do_not_act():
    obs = {
        "vision": {
            "cam_head": {
                "depth": np.ones((2, 2)),
                "color": np.zeros((2, 2, 3)),
                "intrinsic_matrix": np.eye(3),
                "extrinsic_matrix": np.eye(4),
            }
        }
    }
    primitives = SimpleNamespace(
        _last_obs=obs,
        sam3_client=SimpleNamespace(
            segment=lambda *a, **kw: SimpleNamespace(found=False)
        ),
    )
    for name in [
        "back_project",
        "segment",
        "view_env_state",
        "get_reward_details",
        "get_safety_status",
    ]:
        assert _is_readonly(partial(getattr(tools, name)))
    assert not _is_readonly(tools.move_to)
    assert tools.back_project(primitives, None, 0, 0)["world_xyz"] == [0, 0, -1]
    assert tools.segment(primitives, None, "object")["found"] is False
