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

"""Contracts for RoboCasa configuration, prompts, and resources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from robots.robocasa.prompt_bundle import system_prompt
from robots.robocasa.robot_spec import _parse_config
from rpent.memory import MemoryManager
from rpent.prompt.utils import format_prompt


def _args(
    tmp_path: Path,
    *,
    memory_dir: Path | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        task_name="OpenDrawer",
        split="target",
        seed=1,
        output_dir=tmp_path / "run",
        memory_dir=memory_dir,
        memory_profile="local",
        planner="codex",
        model="offline",
    )


def test_parse_config_uses_default_memory_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))

    config = _parse_config(
        _args(tmp_path, memory_dir=None),
    )

    assert config.prompt_vars["memory_dir"] == str(tmp_path / "memory" / "robocasa")


def test_default_memory_sync_uses_robocasa_subtree(monkeypatch, tmp_path):
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)

    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("RPENT_MEMORY_HF_REPO", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    root = MemoryManager(tmp_path / "memory" / "robocasa").sync(
        remote_repo="RLinf/RPent-memory"
    )

    assert root == (tmp_path / "memory" / "robocasa")
    assert calls == {
        "repo_id": "RLinf/RPent-memory",
        "repo_type": "dataset",
        "local_dir": str(tmp_path / "memory"),
        "allow_patterns": ["robocasa/**"],
    }


def test_results_corpus_is_readable_through_memory_tool(monkeypatch, tmp_path):
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    memory_root = tmp_path / "memory" / "robocasa"
    results = memory_root / "results"
    results.mkdir(parents=True)
    audit = results / "OpenDrawer_s0.json"
    audit.write_text('{"success": true}\n')

    manager = MemoryManager(root=memory_root)
    bindings = manager.get_common_tool_bindings()
    read_text_file = bindings["read_text_file"][1]
    write_text_file = bindings["write_text_file"][1]

    assert read_text_file(path=str(audit))["content"] == '{"success": true}\n'
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        write_text_file(path=str(audit), content="{}\n")


def test_parse_config_resolves_local_memory_dir(tmp_path):
    memory_dir = tmp_path / "local-memory"

    config = _parse_config(
        _args(tmp_path, memory_dir=memory_dir),
    )

    assert config.prompt_vars["memory_dir"] == str(memory_dir.resolve())


def test_prompt_names_only_current_task_memory(tmp_path):
    memory_dir = tmp_path / "memory"
    rendered = format_prompt(
        system_prompt(),
        variables={
            "task_name": "OpenDrawer",
            "memory_dir": str(memory_dir),
        },
    )

    assert str(memory_dir / "task-specific" / "OpenDrawer_s0.json") in rendered
    assert str(memory_dir / "task-specific" / "OpenDrawer_s0_recipe.jsonl") in rendered
    assert str(memory_dir / "task-specific" / "OpenDrawer.md") in rendered
    assert "as needed for the live task" in rendered
    assert "do not require reading every file" in rendered
    assert "ArrangeTea_s0" not in rendered
    assert str(memory_dir / "global" / "GLOBAL_MEMORY.md") in rendered
    assert "{{" not in rendered


def test_all_50_tasks_share_prompt_and_tool_selection(tmp_path, make_corpus):
    from robots.robocasa.memory import GLOBAL_FILE, RoboCasaMemoryManager, TaskMemory
    from robots.robocasa.robot_spec import get_robot_spec

    manifest = json.loads(
        (
            Path(__file__).parents[4] / "robots/robocasa/eval/target50_v2.json"
        ).read_text()
    )
    tasks = [task for split in manifest["splits"].values() for task in split["tasks"]]
    provided = tasks[::2]
    root = make_corpus(
        tmp_path / "robocasa",
        tasks=provided,
        markdown=manifest["splits"]["composite_seen"]["tasks"],
    )
    for task in tasks:
        selection = TaskMemory.load(root, task)
        prompt = get_robot_spec().prompts.render(
            "system",
            variables={
                "task_name": task,
                "memory_dir": str(root),
            },
        )
        for name in selection.selected:
            assert str(root / name) in prompt
        assert str(root / GLOBAL_FILE) in prompt
        assert "{{" not in prompt
        manager = RoboCasaMemoryManager(selection)
        read = manager.get_common_tool_bindings()["read_text_file"][1]
        for name in selection.selected:
            assert "content" in read(path=str(root / name))
        assert not manager.unread_files
        other_task = "OpenDrawer" if task != "OpenDrawer" else "StirVegetables"
        with pytest.raises(PermissionError, match="current task/global selection"):
            read(path=str(root / "task-specific" / f"{other_task}_s0.json"))


def test_optional_layers_and_stale_files_are_not_substituted(tmp_path, make_corpus):
    from robots.robocasa.memory import RoboCasaMemoryManager, TaskMemory

    root = make_corpus(tmp_path / "robocasa", tasks=(), global_memory=True)
    stale = root / "results/ArrangeTea.md"
    stale.parent.mkdir()
    stale.write_text("Old snapshot memory must not be read")
    selection = TaskMemory.load(root, "ArrangeTea")
    assert selection.selected == ("global/GLOBAL_MEMORY.md",)
    manager = RoboCasaMemoryManager(selection)
    bindings = manager.get_common_tool_bindings()
    with pytest.raises(PermissionError):
        bindings["read_text_file"][1](path=str(stale))
    assert bindings["list_dir"][1](path=str(root))["files"] == ["global"]
    with pytest.raises(PermissionError):
        bindings["list_dir"][1](path=str(root / "task-specific"))
    empty = make_corpus(tmp_path / "empty", tasks=(), global_memory=False)
    with pytest.raises(ValueError, match="task-global requires"):
        TaskMemory.load(empty, "ArrangeTea")


@pytest.mark.parametrize(
    "corruption", ["half_pair", "directory", "invalid_text", "path_escape"]
)
def test_corrupt_corpus_fails_before_prompt_or_robot_start(
    tmp_path, make_corpus, corruption
):
    from robots.robocasa.memory import TaskMemory

    root = make_corpus(tmp_path / "robocasa")
    path = root / "task-specific/OpenDrawer_s0.json"
    if corruption == "half_pair":
        (root / "task-specific/OpenDrawer_s0_recipe.jsonl").unlink()
    elif corruption == "directory":
        path.unlink()
        path.mkdir()
    elif corruption == "invalid_text":
        path.write_bytes(b"\xff")
    else:
        foreign = tmp_path / "foreign.json"
        foreign.write_text("{}")
        path.unlink()
        path.symlink_to(foreign)
    with pytest.raises(ValueError):
        TaskMemory.load(root, "OpenDrawer")


def test_reads_are_optional_and_files_are_rechecked(tmp_path, make_corpus):
    from robots.robocasa.memory import RoboCasaMemoryManager, TaskMemory
    from robots.robocasa.toolkit import RoboCasaToolkit
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.tools.toolkit import Toolkit

    root = make_corpus(tmp_path / "robocasa")
    manager = RoboCasaMemoryManager(TaskMemory.load(root, "OpenDrawer"))
    toolkit = RoboCasaToolkit.__new__(RoboCasaToolkit)
    Toolkit.__init__(toolkit, dashboard_events=NullDashboardEventSink(), memory=manager)
    read = manager.get_common_tool_bindings()["read_text_file"][1]
    path = root / manager.selection.selected[0]
    read(path=str(path), max_chars=1)
    assert manager.unread_files
    assert toolkit.execute_tool(
        "finish", {"status": "stuck", "summary": "test"}
    ).is_finish
    for name in manager.selection.selected:
        read(path=str(root / name))
    assert toolkit.execute_tool(
        "finish", {"status": "stuck", "summary": "test"}
    ).is_finish
    path.write_text("changed after initial validation")
    with pytest.raises(ValueError, match="changed during this run"):
        read(path=str(path))
    # A subsequent run uses updated memory without editing a manifest or code.
    updated = RoboCasaMemoryManager(TaskMemory.load(root, "OpenDrawer"))
    assert (
        updated.get_common_tool_bindings()["read_text_file"][1](path=str(path))[
            "content"
        ]
        == "changed after initial validation"
    )


@pytest.mark.parametrize("reading", ["none", "partial", "complete"])
def test_on_demand_reads_do_not_block_actions_or_finish(
    tmp_path, monkeypatch, make_corpus, reading
):
    from robots.robocasa.memory import RoboCasaMemoryManager, TaskMemory
    from robots.robocasa.toolkit import RoboCasaToolkit
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.session import EnvState
    from rpent.tools.toolkit import Toolkit

    root = make_corpus(tmp_path / "robocasa")
    manager = RoboCasaMemoryManager(TaskMemory.load(root, "OpenDrawer"))
    toolkit = RoboCasaToolkit.__new__(RoboCasaToolkit)
    Toolkit.__init__(
        toolkit,
        dashboard_events=NullDashboardEventSink(),
        memory=manager,
        state=EnvState(tmp_path / "run"),
    )
    monkeypatch.setattr(toolkit, "get_env_state", lambda **kwargs: kwargs["result"])
    read = manager.get_common_tool_bindings()["read_text_file"][1]
    if reading == "partial":
        read(path=str(root / manager.selection.selected[0]), max_chars=1)
    elif reading == "complete":
        for name in manager.selection.selected:
            read(path=str(root / name))

    for name, arguments in (
        ("move_to", {"xyz": [0.1, 0.2, 0.9]}),
        ("rldx_skill", {"prompt": "Open the drawer"}),
    ):
        handler = Mock(return_value={"executed": name})
        handler._readonly = False
        toolkit.add_tool(name, {"name": name}, handler)
        assert toolkit.execute_tool(name, arguments).result == {"executed": name}
        handler.assert_called_once_with(**arguments)
    assert toolkit.execute_tool(
        "finish", {"status": "stuck", "summary": "test"}
    ).is_finish


def test_memory_policy_flag_is_not_available(tmp_path):
    from robots.robocasa.robot_spec import get_robot_spec

    parser = argparse.ArgumentParser()
    get_robot_spec().add_cli_args(parser, use_dashboard=False)
    args = parser.parse_args(["--task-name", "OpenDrawer"])
    assert not hasattr(args, "memory_policy")
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--task-name", "OpenDrawer", "--memory-policy", "task-specific"]
        )
    config = _parse_config(_args(tmp_path, memory_dir=tmp_path / "memory"))
    assert "memory_policy" not in config.prompt_vars
    assert "memory_revision" not in config.prompt_vars


@pytest.mark.parametrize("dashboard", [False, True])
def test_cli_and_dashboard_sync_memory_without_a_revision(
    tmp_path, monkeypatch, dashboard
):
    from rpent.cli import main as cli

    class SyncCaptured(Exception):
        pass

    captured = {}

    def capture_sync(self, **kwargs):
        captured.update(kwargs)
        raise SyncCaptured

    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(MemoryManager, "sync", capture_sync)
    monkeypatch.setattr(
        "rpent.dashboard.server.DashboardServer.start",
        lambda self: "http://127.0.0.1:0",
    )
    argv = [
        "rpent",
        "--robot",
        "robocasa",
        "--task-name",
        "OpenDrawer",
        "--planner",
        "codex",
        "--output-dir",
        str(tmp_path / "run"),
    ]
    if dashboard:
        argv.append("--dashboard")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SyncCaptured):
        cli.main()
    assert captured == {
        "remote_repo": "RLinf/RPent-memory",
    }


def test_updated_optional_markdown_is_discovered_without_manifest(
    tmp_path, make_corpus
):
    from robots.robocasa.memory import TaskMemory

    root = make_corpus(tmp_path / "robocasa", markdown=())
    path = root / "task-specific/OpenDrawer.md"
    assert "task-specific/OpenDrawer.md" in TaskMemory.load(root, "OpenDrawer").missing
    path.write_text("New task note")
    selection = TaskMemory.load(root, "OpenDrawer")
    assert "task-specific/OpenDrawer.md" in selection.selected
    assert not (root / "CORPUS.json").exists()


def test_legacy_task_directory_cannot_be_mistaken_for_missing_memory(
    tmp_path, make_corpus
):
    from robots.robocasa.memory import TaskMemory

    root = make_corpus(tmp_path / "robocasa")
    (root / "task-specific").rename(root / "task_only")
    with pytest.raises(ValueError, match="legacy memory directory"):
        TaskMemory.load(root, "OpenDrawer")


@pytest.mark.parametrize(
    ("components", "task_name", "missing"),
    [
        (None, "OpenDrawer", "global/GLOBAL_MEMORY.md"),
        (None, "OpenDrawer", "task-specific/OpenDrawer_s0.json"),
        ({"vla"}, None, "global/GLOBAL_MEMORY.md"),
    ],
)
def test_invalid_memory_prevents_runtime_start(
    tmp_path, monkeypatch, make_corpus, components, task_name, missing
):
    from robots.robocasa import robot_spec
    from rpent.dashboard.events import NullDashboardEventSink

    root = make_corpus(tmp_path / "robocasa")
    (root / missing).unlink()
    args = _args(tmp_path, memory_dir=root)
    args.task_name = task_name
    spawn = Mock()
    monkeypatch.setattr(robot_spec, "try_spawn_server", spawn)

    with pytest.raises(ValueError, match="task-global requires|incomplete seed-0"):
        robot_spec.get_robot_spec().init_runtime(
            args, tmp_path / "run", NullDashboardEventSink(), components
        )
    spawn.assert_not_called()


def test_component_diagnostics_do_not_require_planner_memory(tmp_path, monkeypatch):
    from robots.robocasa import robot_spec
    from rpent.dashboard.events import NullDashboardEventSink

    parser = argparse.ArgumentParser()
    spec = robot_spec.get_robot_spec()
    spec.add_cli_args(parser, use_dashboard=False)
    args = parser.parse_args(["--task-name", "OpenDrawer"])
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    spawn = Mock(return_value=(None, object()))
    monkeypatch.setattr(robot_spec, "try_spawn_server", spawn)
    monkeypatch.setattr(robot_spec, "try_wait_server", Mock(return_value={}))

    spec.init_runtime(args, tmp_path / "run", NullDashboardEventSink(), {"env"})
    assert [call.args[2] for call in spawn.call_args_list] == ["env"]


@pytest.mark.parametrize("missing", ["json", "jsonl", "global"])
def test_dashboard_checks_each_task_before_starting_its_env(
    tmp_path, monkeypatch, make_corpus, missing
):
    from robots.robocasa import robot_spec
    from rpent.cli import dashboard as dashboard_cli
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.dashboard.state import ClaimedTask

    root = make_corpus(tmp_path / "robocasa")
    args = _args(tmp_path, memory_dir=root)
    args.task_name = None  # The shared VLA starts before /rpent-task.
    args.verbose = False
    args.explore = False
    spec = robot_spec.get_robot_spec()
    spawn = Mock(return_value=(None, object()))
    shared = {"vla_client": object()}
    monkeypatch.setattr(robot_spec, "try_spawn_server", spawn)
    monkeypatch.setattr(robot_spec, "try_wait_server", Mock(return_value=shared))

    # Missing task files cannot be checked until the task is selected.
    path = (
        root
        / {
            "json": "task-specific/OpenDrawer_s0.json",
            "jsonl": "task-specific/OpenDrawer_s0_recipe.jsonl",
            "global": "global/GLOBAL_MEMORY.md",
        }[missing]
    )
    content = path.read_bytes()
    if missing != "global":
        path.unlink()
    daemons, runtime = spec.init_runtime(
        args, tmp_path / "session", NullDashboardEventSink(), {"vla"}
    )
    assert daemons == []
    assert runtime == shared
    assert [call.args[2] for call in spawn.call_args_list] == ["vla"]
    spawn.reset_mock()

    if missing == "global":
        path.unlink()  # Also catch a file removed after shared startup.
    state = SimpleNamespace(task_replacement_requested=False)
    claimed = ClaimedTask(
        number=1,
        request={"task_name": "OpenDrawer", "split": "target", "seed": 1},
        output_dir=tmp_path / "run",
    )
    error = dashboard_cli._run_dashboard_task(
        args=args,
        robot_spec=spec,
        state=state,
        claimed=claimed,
        shared_runtime_kwargs=runtime,
        unique_components={"env"},
        session_root=tmp_path / "session",
    )
    assert "task-global requires" in error or "incomplete seed-0" in error
    spawn.assert_not_called()

    # Repairing memory lets a later task use the existing shared VLA.
    path.write_bytes(content)
    args.task_name = "OpenDrawer"
    spec.init_runtime(args, claimed.output_dir, NullDashboardEventSink(), {"env"})
    assert [call.args[2] for call in spawn.call_args_list] == ["env"]


@pytest.mark.parametrize("with_index", [False, True])
def test_native_merge_output_shares_prompt_tool_and_audit_selection(
    tmp_path, make_native_corpus, with_index
):
    from robots.robocasa.memory import (
        RoboCasaMemoryManager,
        TaskMemory,
        selection_errors,
    )
    from robots.robocasa.robot_spec import get_robot_spec

    root = make_native_corpus(tmp_path / "native")
    other = make_native_corpus(tmp_path / "other", task="StirVegetables")
    wrong_split = make_native_corpus(tmp_path / "source", split="pretrain")
    for source in (other, wrong_split):
        for layer in ("task-specific", "task-family"):
            for path in (source / layer).iterdir():
                (root / layer / path.name).write_bytes(path.read_bytes())
    if not with_index:
        (root / "MEMORY.md").unlink()
    selection = TaskMemory.load(root, "OpenDrawer", profile="local", split="target")
    expected = {
        "task-specific/OpenDrawer_target_s0.json",
        "task-specific/OpenDrawer_target_s0_recipe.jsonl",
        "task-family/task-family_robocasa_target_tOpenDrawer.md",
        "global/global.md",
    }
    assert set(selection.selected) == expected
    assert selection.missing == ("task-specific/OpenDrawer.md",)
    assert selection_errors(selection.metadata(), "OpenDrawer", "target") == []
    prompt = get_robot_spec().prompts.render(
        "system",
        variables={
            "task_name": "OpenDrawer",
            "split": "target",
            "memory_dir": str(root),
            "memory_profile": "local",
            "mode": "eval",
        },
    )
    for name in expected:
        assert str(root / name) in prompt
    assert "StirVegetables" not in prompt
    assert "OpenDrawer_pretrain" not in prompt
    manager = RoboCasaMemoryManager(selection, output_dir=tmp_path / "run")
    bindings = manager.get_common_tool_bindings()
    for name in expected:
        assert (
            bindings["read_text_file"][1](path=str(root / name))["content"]
            == (root / name).read_text()
        )
    assert not manager.unread_files
    for path in (
        root / "MEMORY.md",
        root / "_internal",
        root / "task-specific/StirVegetables_target_s0.json",
        root / "task-specific/OpenDrawer_pretrain_s0.json",
        root / "task-family/task-family_robocasa_pretrain_tOpenDrawer.md",
        root / "task-family/task-family_robocasa_target_tStirVegetables.md",
    ):
        with pytest.raises(PermissionError):
            bindings["read_text_file"][1](path=str(path))
    assert set(bindings["list_dir"][1](path=str(root))["files"]) == {
        "global",
        "task-family",
        "task-specific",
    }
    audit = [
        json.loads(line)
        for line in (tmp_path / "run/memory_reads.jsonl").read_text().splitlines()
    ]
    assert {entry["path"] for entry in audit} == expected


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("missing_member", [0, 1])
def test_local_half_pair_is_rejected_for_both_layouts(
    tmp_path, make_corpus, make_native_corpus, native, missing_member
):
    from robots.robocasa.memory import TaskMemory, local_task_files, task_files

    root = (make_native_corpus if native else make_corpus)(tmp_path / "corpus")
    names = (
        local_task_files("OpenDrawer", "target") if native else task_files("OpenDrawer")
    )
    (root / names[missing_member]).unlink()
    with pytest.raises(ValueError, match="incomplete seed-0"):
        TaskMemory.load(root, "OpenDrawer", profile="local")


def test_local_mixed_current_task_pairs_require_separate_directories(
    tmp_path, make_corpus, make_native_corpus
):
    from robots.robocasa.memory import TaskMemory

    root = make_native_corpus(tmp_path / "mixed")
    make_corpus(root)
    with pytest.raises(ValueError, match="separate --memory-dir"):
        TaskMemory.load(root, "OpenDrawer", profile="local")
    # HF profile deliberately uses only the published task/global selection.
    selection = TaskMemory.load(root, "OpenDrawer", profile="hf")
    assert "task-specific/OpenDrawer_s0.json" in selection.selected
    assert "task-specific/OpenDrawer_target_s0.json" not in selection.selected


def test_local_global_only_and_multiple_plain_global_leaves(tmp_path, make_corpus):
    from robots.robocasa.memory import TaskMemory, selection_errors

    root = make_corpus(tmp_path / "corpus", tasks=())
    (root / "global/second.md").write_text("Plain global Markdown is supported.")
    selection = TaskMemory.load(root, "OpenDrawer", profile="local")
    assert set(selection.selected) == {"global/GLOBAL_MEMORY.md", "global/second.md"}
    assert len(selection.missing) == 3
    assert selection_errors(selection.metadata(), "OpenDrawer", "target") == []
    (root / "global/GLOBAL_MEMORY.md").unlink()
    assert TaskMemory.load(root, "OpenDrawer", profile="local").selected == (
        "global/second.md",
    )
    with pytest.raises(ValueError, match="task-global requires"):
        TaskMemory.load(root, "OpenDrawer", profile="hf")
    (root / "global/second.md").unlink()
    with pytest.raises(ValueError, match="task-global requires"):
        TaskMemory.load(root, "OpenDrawer", profile="local")


def test_exploration_can_start_empty_and_keeps_inbox_access(tmp_path, monkeypatch):
    from robots.robocasa import robot_spec, toolkit
    from rpent.dashboard.events import NullDashboardEventSink

    root = tmp_path / "empty"
    args = _args(tmp_path, memory_dir=root)
    args.explore = True
    args.explore_sessions = 3
    args.explore_attempts_per_session = 5
    spawn = Mock(return_value=(None, object()))
    monkeypatch.setattr(robot_spec, "try_spawn_server", spawn)
    monkeypatch.setattr(robot_spec, "try_wait_server", Mock(return_value={}))
    robot_spec.get_robot_spec().init_runtime(
        args, tmp_path / "run", NullDashboardEventSink(), {"env"}
    )
    assert spawn.call_count == 1
    config = _parse_config(args)
    monkeypatch.setattr(
        toolkit, "RoboCasaToolkit", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    instance = robot_spec.get_toolkit(
        runtime_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        mode="exploration",
        attempts_per_session=5,
        state_output_dir=tmp_path / "attempt",
    )
    assert instance.mode == "exploration"
    assert instance.attempts_per_session == 5
    assert instance.state_output_dir == tmp_path / "attempt"
    bindings = instance.memory.get_common_tool_bindings()
    inbox_file = root / "_internal/inbox/OpenDrawer_target_s1/wip/notes.md"
    bindings["write_text_file"][1](path=str(inbox_file), content="Offline note")
    assert (
        bindings["read_text_file"][1](path=str(inbox_file))["content"] == "Offline note"
    )
    with pytest.raises(PermissionError):
        bindings["write_text_file"][1](
            path=str(root / "global/direct.md"), content="Denied"
        )
    prompt = robot_spec.get_robot_spec().prompts.render(
        "system", variables={**config.prompt_vars, "output_dir": str(config.output_dir)}
    )
    assert "scope: task-family" in prompt
    assert "suite: robocasa" in prompt
    assert "scope: suite" not in prompt
    assert "MULTI-ATTEMPT EXPLORE" in prompt
