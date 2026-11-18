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
    ("robot", "profile", "sync_before_shared", "explicit_repo"),
    [
        ("robocasa", "hf", True, False),
        ("robocasa", "hf", True, True),
        ("robotwin", "hf", True, False),
        ("robotwin", "hf", True, True),
        ("robocasa", "local", False, False),
        ("libero", "hf", False, False),
        ("libero", "local", False, False),
        ("dual_franka", "local", False, False),
    ],
)
@pytest.mark.parametrize("planner", ["api", "claude_code", "codex"])
def test_shared_runtime_gets_memory_before_startup(
    robot, profile, sync_before_shared, planner, tmp_path, monkeypatch, explicit_repo
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
    if explicit_repo:
        from rpent.memory import source

        def prepare(source_url, name):
            assert name == robot and source_url == "team/memory@release"
            sync(MemoryManager(root), remote_repo="test/memory")
            return root

        monkeypatch.setattr(source, "prepare_explicit_source", prepare)
    monkeypatch.setattr(DashboardServer, "start", start)
    monkeypatch.setattr(DashboardState, "fail_session", fail_session)
    spec = SimpleNamespace(
        name=robot,
        is_real_robot=robot == "dual_franka",
        supports_human_interactive_exploration=robot == "dual_franka",
        memory_repo_id="test/memory",
        prepare_memory=(lambda args, config: None) if robot == "libero" else None,
        resolve_dashboard=lambda args: {
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
            planner,
            "--model",
            "test",
            "--memory-profile",
            profile,
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    if explicit_repo:
        args.memory_repo = "team/memory@release"
    assert (
        dashboard.run_dashboard_session(args, spec, parser=argparse.ArgumentParser())
        == 0
    )
    assert events == (["sync"] if sync_before_shared else []) + ["shared-runtime"]
    if explicit_repo:
        assert args.memory_dir == str(root)


def test_task_and_exploration_sessions_bind_fresh_prompts(tmp_path, monkeypatch):
    from pathlib import Path

    from robots.dual_franka import prompt_bundle
    from rpent.cli import main as main_cli
    from rpent.dashboard.state import ClaimedTask
    from rpent.memory import loading
    from rpent.robots.prompt_bundle import PromptBundle
    from rpent.robots.robot_spec import RunConfig

    runs = []
    prepared = []
    planner_contexts = []
    session_paths = []

    def parse_config(args):
        return RunConfig(
            recipe_tag=f"task_{args.task_id}",
            output_dir=Path(args.output_dir),
            prompt_vars={
                "mode": "explore",
                "task_id": args.task_id,
                "task_name": f"task_{args.task_id}",
                "instruction": f"Move task-{args.task_id} object {{{{literal}}}}",
                "setup": f"setup-{args.task_id}",
                "constraints": f"constraint-{args.task_id}",
                "success_criteria": f"success-{args.task_id}",
                "recipe_tag": f"task_{args.task_id}",
                "memory_dir": "unprepared-memory",
                "memory_inbox": "unprepared-inbox",
                "explore_attempts_per_session": args.explore_attempts_per_session,
            },
            task_desc={"task_id": args.task_id},
        )

    def prepare_memory(args, spec, config):
        prepared.append(args.task_id)
        config.prompt_vars.update(
            memory_dir=str(tmp_path / f"memory-{args.task_id}"),
            memory_inbox=str(tmp_path / f"inbox-{args.task_id}"),
        )

    class State:
        task_replacement_requested = False
        task_finalization_cancelled = False

        def emit(self, event):
            pass

        def begin_planner_session(self, *, video_path):
            session_paths.append(video_path)

        def bind_toolkit(self, toolkit):
            pass

        def unbind_toolkit(self, toolkit):
            pass

    class Toolkit:
        memory = None

        def solved(self):
            return False

        def close(self):
            pass

    class Planner:
        def solve(self, *, system_prompt, user_message, **kwargs):
            runs.append((system_prompt, user_message))
            return SimpleNamespace(
                finish_result=None, messages=[], stats={}, error=None
            )

    def build_planner(*args, **kwargs):
        planner_contexts.append(kwargs)
        return Planner()

    spec = SimpleNamespace(
        name="dual_franka",
        is_real_robot=True,
        supports_human_interactive_exploration=False,
        supports_exploration=True,
        finalize_run=None,
        parse_config=parse_config,
        init_runtime=lambda *args: ([], {}),
        prompts=PromptBundle(prompt_bundle.system_prompt, prompt_bundle.user_prompt),
    )
    monkeypatch.setattr(loading, "prepare_run_memory", prepare_memory)
    monkeypatch.setattr(main_cli, "get_robot_spec", lambda name: spec)
    monkeypatch.setattr(dashboard, "get_toolkit", lambda *args, **kwargs: Toolkit())
    monkeypatch.setattr(dashboard, "build_planner", build_planner)
    args = _build_argparser().parse_args(["--dashboard", "--explore"])
    args.robot_name = "dual_franka"
    args.explore_sessions = 2
    args.explore_attempts_per_session = 4
    args.auto_merge_memory = False
    state = State()
    for task_id in (1, 2):
        assert (
            dashboard._run_dashboard_task(
                args=args,
                robot_spec=spec,
                state=state,
                claimed=ClaimedTask(
                    number=task_id,
                    request={"task_id": task_id},
                    output_dir=tmp_path / f"run-{task_id}",
                ),
                shared_runtime_kwargs={},
                unique_components=set(),
                session_root=tmp_path / "session",
            )
            is None
        )

    assert prepared == [1, 2]
    assert len(runs) == 4
    for index, (system, user) in enumerate(runs):
        task_id = index // 2 + 1
        session = index % 2 + 1
        other_task = 3 - task_id
        assert f"session {session} of 2" in system
        assert "Attempt limit for this session: 4." in system
        assert f"task_{task_id}" in system
        assert f"Move task-{task_id} object {{{{literal}}}}" in user
        assert f"constraint-{task_id}" in user
        assert f"success-{task_id}" in user
        assert str(tmp_path / f"memory-{task_id}") in system
        assert str(tmp_path / f"inbox-{task_id}") in system
        assert str(tmp_path / f"run-{task_id}") in system
        assert str(tmp_path / f"run-{task_id}") in user
        for stale in (
            f"task-{other_task}",
            f"memory-{other_task}",
            f"run-{other_task}",
        ):
            assert stale not in system + user
        assert "unprepared" not in system + user
        assert planner_contexts[index]["memory_dir"] == str(
            tmp_path / f"memory-{task_id}"
        )
        assert session_paths[index] == (
            tmp_path
            / f"run-{task_id}"
            / "sessions"
            / f"session_{session:03d}"
            / "episode.mp4"
        )
        if session == 2:
            assert runs[index - 1][1] in user
            assert "Current task instruction:" in user
            assert "physical scene is not automatically reset" in user
