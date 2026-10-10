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

"""Model-selected RoboCasa corpora across shared memory preparation."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pytest

from robots.robocasa import memory
from robots.robocasa.robot_spec import get_robot_spec
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory.loading import prepare_run_memory


@pytest.mark.parametrize(
    ("model", "planner", "version", "expected"),
    [
        ("gpt-5.5", "codex", "auto", memory.GPT5),
        ("gpt-6-astra", "codex", "auto", memory.ASTRA),
        ("openai:gpt-6-astra", "api", "auto", memory.ASTRA),
        ("openai-chat:gpt-5.5", "api", "auto", memory.GPT5),
        ("gpt-6-astra", "codex", memory.GPT5, memory.GPT5),
        ("unknown", "api", "auto", memory.GPT5),
    ],
)
def test_model_selection(monkeypatch, model, planner, version, expected):
    monkeypatch.delenv("CODEX_MODEL", raising=False)
    assert memory.select_version(version, model=model, planner=planner) == expected


def test_codex_default_model(monkeypatch):
    monkeypatch.setenv("CODEX_MODEL", "gpt-6-astra")
    assert memory.select_version(planner="codex") == memory.ASTRA
    assert memory.select_version(planner="codex", model="gpt-5.5") == memory.GPT5


@pytest.fixture
def hub(monkeypatch, make_corpus):
    monkeypatch.delenv("RPENT_MEMORY_HF_REPO", raising=False)
    info = Mock(return_value=SimpleNamespace(sha="commit-one"))
    monkeypatch.setattr(
        huggingface_hub, "HfApi", lambda: SimpleNamespace(repo_info=info)
    )

    def download(**kwargs):
        from pathlib import Path

        prefix = kwargs["allow_patterns"][0].removesuffix("/**")
        make_corpus(Path(kwargs["local_dir"]) / prefix)
        return kwargs["local_dir"]

    sync = Mock(side_effect=download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", sync)
    return info, sync


def test_download_isolates_models_repositories_and_commits(tmp_path, hub, monkeypatch):
    info, sync = hub

    def run(version=memory.GPT5):
        return memory.sync_version(version=version, cache_dir=tmp_path)

    first = run()
    second = run(memory.ASTRA)
    assert first != second
    assert sync.call_args.kwargs["allow_patterns"] == ["robocasa/GPT_6_astra_high/**"]
    assert sync.call_args.kwargs["revision"] == "commit-one"
    info.return_value = SimpleNamespace(sha="commit-two")
    third = run()
    assert third != first
    assert info.call_args.kwargs["revision"] == "release/v0.1"
    monkeypatch.setenv("RPENT_MEMORY_HF_REPO", "owner/custom")
    assert run() != third
    assert sync.call_args.kwargs["repo_id"] == "owner/custom"


def test_missing_selected_corpus_does_not_use_legacy_root(tmp_path, hub, make_corpus):
    _, sync = hub
    sync.side_effect = lambda **kw: make_corpus(tmp_path / "robocasa")
    with pytest.raises(ValueError, match="directory not found"):
        memory.sync_version(version=memory.ASTRA, cache_dir=tmp_path)


def test_resolution_failure_does_not_use_another_revision(tmp_path, hub):
    info, sync = hub
    memory.sync_version(version=memory.GPT5, cache_dir=tmp_path)
    sync.reset_mock()
    info.side_effect = ConnectionError("Hub unavailable")
    with pytest.raises(ConnectionError):
        memory.sync_version(version=memory.GPT5, cache_dir=tmp_path)
    sync.assert_not_called()


def args_for(tmp_path, **overrides):
    values = {
        "task_name": "OpenDrawer",
        "split": "target",
        "seed": 1,
        "output_dir": tmp_path / "run",
        "memory_profile": "hf",
        "memory_dir": None,
        "memory_version": "auto",
        "planner": "codex",
        "model": "gpt-6-astra",
        "explore": False,
    }
    return Namespace(**(values | overrides))


def test_shared_preparation_binds_runtime_prompts_and_tools(tmp_path, monkeypatch, hub):
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    spec = get_robot_spec()
    args = args_for(tmp_path, model=None)
    monkeypatch.setenv("CODEX_MODEL", "gpt-6-astra")
    config = spec.parse_config(args)
    prepare_run_memory(args, spec, config)
    assert args.model == "gpt-6-astra"
    assert args.memory_dir == config.prompt_vars["memory_dir"]
    assert config.prompt_vars["memory_version"] == memory.ASTRA
    selection = memory.memory_from_variables(config.prompt_vars)
    assert selection.root.name == memory.ASTRA
    assert selection.profile == "hf"
    manager = memory.RoboCasaMemoryManager(selection, output_dir=tmp_path / "run")
    assert manager.root == selection.root


@pytest.mark.parametrize("explore", [False, True])
def test_local_and_exploration_do_not_download(tmp_path, monkeypatch, explore):
    args = args_for(
        tmp_path, explore=explore, memory_profile="local", memory_dir=tmp_path
    )
    config = (
        get_robot_spec().parse_config(args)
        if not explore
        else SimpleNamespace(prompt_vars={"memory_dir": str(tmp_path)})
    )
    sync = Mock()
    monkeypatch.setattr(memory, "sync_version", sync)
    memory.prepare_memory(args, config)
    sync.assert_not_called()
    assert config.prompt_vars["memory_dir"] == str(tmp_path)


@pytest.mark.parametrize("profile,explore", [("local", False), (None, True)])
def test_hf_selectors_reject_local_modes(tmp_path, profile, explore):
    args = args_for(
        tmp_path, memory_profile=profile, explore=explore, memory_version=memory.ASTRA
    )
    with pytest.raises(ValueError, match="requires --memory-profile hf"):
        memory.validate_options(args)


def test_dashboard_shared_vla_defers_hf_selection(tmp_path, monkeypatch):
    from robots.robocasa import robot_spec

    args = args_for(tmp_path, task_name=None)
    spawn = Mock(return_value=(None, object()))
    monkeypatch.setattr(robot_spec, "try_spawn_server", spawn)
    monkeypatch.setattr(robot_spec, "try_wait_server", Mock(return_value={}))
    get_robot_spec().init_runtime(args, tmp_path, NullDashboardEventSink(), {"vla"})
    assert [c.args[2] for c in spawn.call_args_list] == ["vla"]


def test_dashboard_uses_each_claimed_model(tmp_path, monkeypatch, hub):
    from dataclasses import replace
    from pathlib import Path

    from rpent.cli import dashboard
    from rpent.dashboard.state import ClaimedTask

    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    args = args_for(tmp_path, model="gpt-5.5", task_name=None)
    args.verbose = False
    spec = get_robot_spec()

    def init_runtime(task_args, *unused):
        assert task_args.model == "gpt-6-astra"
        assert Path(task_args.memory_dir).name == memory.ASTRA
        raise RuntimeError("verified selected corpus before task services")

    error = dashboard._run_dashboard_task(
        args=args,
        robot_spec=replace(spec, init_runtime=init_runtime),
        state=SimpleNamespace(task_replacement_requested=False),
        claimed=ClaimedTask(
            number=1,
            request={
                "model": "gpt-6-astra",
                "task_name": "OpenDrawer",
                "split": "target",
                "seed": 1,
            },
            output_dir=tmp_path / "run",
        ),
        shared_runtime_kwargs={},
        unique_components={"env"},
        session_root=tmp_path,
    )
    assert "verified selected corpus" in error
    assert args.model == "gpt-5.5"
    assert args.memory_dir is None
