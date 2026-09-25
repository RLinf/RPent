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

"""Dashboard startup memory requirements before shared robot services."""

import argparse
from types import SimpleNamespace

import pytest

from rpent.cli import dashboard
from rpent.cli.main import _build_argparser
from rpent.dashboard.server import DashboardServer
from rpent.dashboard.state import DashboardState
from rpent.memory import MemoryManager


@pytest.mark.parametrize(
    ("robot", "profile", "sync_before_shared"),
    [
        ("robocasa", "hf", True),
        ("robotwin", "hf", True),
        ("robocasa", "local", False),
        ("libero", "hf", False),
        ("libero", "local", False),
    ],
)
def test_shared_runtime_gets_memory_before_startup(
    robot, profile, sync_before_shared, tmp_path, monkeypatch
):
    root = tmp_path / "memory" / robot
    events = []

    def sync(manager, *, remote_repo):
        assert manager.root == root
        assert remote_repo == "test/memory"
        global_file = root / "global" / "GLOBAL_MEMORY.md"
        global_file.parent.mkdir(parents=True)
        global_file.write_text("Live task observations take precedence.")
        events.append("sync")
        return root

    def init_runtime(args, output_dir, state, components):
        assert components == {"vla"}
        if sync_before_shared:
            assert (root / "global" / "GLOBAL_MEMORY.md").is_file()
        events.append("shared-runtime")
        return [], {}

    def start(server):
        # Exercise actual session orchestration, then exit after service startup.
        server._state.request_shutdown()
        return "http://127.0.0.1:12345"

    def fail_session(state, error):
        pytest.fail(f"Shared service startup failed: {error}")

    monkeypatch.setattr(dashboard, "get_memory_dir", lambda name: root)
    monkeypatch.setattr(MemoryManager, "sync", sync)
    monkeypatch.setattr(DashboardServer, "start", start)
    monkeypatch.setattr(DashboardState, "fail_session", fail_session)
    spec = SimpleNamespace(
        name=robot,
        is_real_robot=False,
        memory_repo_id="test/memory",
        prepare_memory=(lambda args, config: None) if robot == "libero" else None,
        dashboard={
            "task": {
                "command": "/rpent-task",
                "usage": "/rpent-task <seed>",
                "fields": ({"name": "seed", "kind": "integer", "minimum": 0},),
                "display": "seed {seed}",
                "output_slug": "s{seed}",
            },
            "runtime_components": ({"name": "vla", "scope": "shared"},),
            "primitives": (),
        },
        init_runtime=init_runtime,
    )
    args = _build_argparser().parse_args(
        [
            "--robot",
            robot,
            "--dashboard",
            "--planner",
            "codex",
            "--memory-profile",
            profile,
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    assert (
        dashboard.run_dashboard_session(args, spec, parser=argparse.ArgumentParser())
        == 0
    )
    assert events == (["sync"] if sync_before_shared else []) + ["shared-runtime"]
