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

"""Version selection, download boundaries and cache failure behavior."""

import hashlib
import json
import sys
from argparse import Namespace
from types import SimpleNamespace

import huggingface_hub
import pytest

from rpent.memory import loading
from rpent.memory.versions import (
    ASTRA,
    GPT5,
    replay_directory,
    select_version,
    sync_version,
)


@pytest.mark.parametrize(
    ("model", "planner", "version", "expected"),
    [
        ("gpt-5.5", "codex", "auto", GPT5),
        ("gpt-6-astra", "codex", "auto", ASTRA),
        ("openai:gpt-6-astra", "api", "auto", ASTRA),
        ("openai-chat:gpt-5.5", "api", "auto", GPT5),
        ("gpt-6-astra", "codex", GPT5, GPT5),
        ("opus", "claude_code", "auto", GPT5),
        (None, "api", "auto", GPT5),
        ("gpt-6-astra", "flash", "auto", GPT5),
    ],
)
def test_selection(model, planner, version, expected, monkeypatch):
    monkeypatch.delenv("CODEX_MODEL", raising=False)
    assert select_version(version, model=model, planner=planner) == expected


def test_codex_environment_precedence(monkeypatch):
    monkeypatch.setenv("CODEX_MODEL", "gpt-6-astra")
    assert select_version(planner="codex") == ASTRA
    assert select_version(planner="codex", model="gpt-5.5") == GPT5
    assert select_version(planner="api") == GPT5


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.delenv("RPENT_MEMORY_HF_REPO", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    snapshot = tmp_path / "hub"
    files = {}
    for version in (GPT5, ASTRA):
        root = snapshot / "libero" / version
        root.mkdir(parents=True)
        contents = {
            "MEMORY.md": version,
            "global/note.md": f"{version} global",
            "task-family/task-family_object_task_t0.md": f"{version} family",
            "task-specific/object_task_t0_s0.json": "{}",
            "task-specific/object_task_t0_s0_recipe.jsonl": "{}\n",
        }
        if version == GPT5:
            contents["task_card/object_task_t0_plan.json"] = "{}"
            contents["task_card/object_task_t0_anchors.json"] = "{}"
        for name, content in contents.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        files[version] = {
            "files": {
                name: hashlib.sha256(content.encode()).hexdigest()
                for name, content in contents.items()
            },
            "source_files": {"old/original.md": "0" * 64},
        }
    manifest = snapshot / "libero/manifest.json"
    manifest.write_text(json.dumps({"versions": files}))
    state = SimpleNamespace(
        sha="a" * 40, calls=[], fail=False, download_fail=False, snapshot=snapshot
    )

    def info(*args, **kwargs):
        if state.fail:
            raise ConnectionError("offline")
        return SimpleNamespace(
            sha=state.sha,
            siblings=[
                SimpleNamespace(rfilename=str(p.relative_to(snapshot)))
                for p in snapshot.rglob("*")
                if p.is_file()
            ],
        )

    def download(*args, **kwargs):
        state.calls.append(kwargs)
        if state.download_fail:
            raise ConnectionError("interrupted transfer")
        return str(snapshot)

    monkeypatch.setattr(
        huggingface_hub, "HfApi", lambda: SimpleNamespace(repo_info=info)
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **k: str(manifest)
    )
    return state


def test_only_selected_subtree_and_complete_offline_cache(hub, tmp_path):
    cache = tmp_path / "cache"
    root = sync_version(version=ASTRA, cache_dir=cache)
    assert (root / "MEMORY.md").read_text() == ASTRA
    downloaded = hub.calls[0]["allow_patterns"]
    assert all(name.startswith(f"libero/{ASTRA}/") for name in downloaded)
    assert f"libero/{ASTRA}/task-specific/object_task_t0_s0.json" in downloaded
    assert f"libero/{ASTRA}/task-family/task-family_object_task_t0.md" in downloaded
    assert len(downloaded) == 5
    assert not (root / "task_card").exists()
    hub.fail = True
    assert sync_version(version=ASTRA, cache_dir=cache) == root
    with pytest.raises(RuntimeError, match="no complete cache"):
        sync_version(version=GPT5, cache_dir=cache)
    (root / "MEMORY.md").write_text("partial")
    with pytest.raises(RuntimeError, match="no complete cache"):
        sync_version(version=ASTRA, cache_dir=cache)


def test_pinned_revisions_and_repositories_never_share_fallback(hub, tmp_path):
    cache = tmp_path / "cache"
    first = sync_version(version=ASTRA, revision="a" * 40, cache_dir=cache)
    hub.sha = "b" * 40
    second = sync_version(version=ASTRA, revision=hub.sha, cache_dir=cache)
    assert first != second
    hub.fail = True
    with pytest.raises(RuntimeError):
        sync_version(version=ASTRA, revision="c" * 40, cache_dir=cache)
    with pytest.raises(RuntimeError):
        sync_version(
            version=ASTRA, revision=hub.sha, repo_id="other/repo", cache_dir=cache
        )


def test_download_failure_cannot_leave_a_complete_cache(hub, tmp_path):
    hub.download_fail = True
    with pytest.raises(ConnectionError):
        sync_version(version=ASTRA, cache_dir=tmp_path / "cache")
    hub.fail = True
    with pytest.raises(RuntimeError):
        sync_version(version=ASTRA, cache_dir=tmp_path / "cache")


def test_manifest_hashes_prevent_partial_or_wrong_corpus(hub, tmp_path):
    (hub.snapshot / "libero" / ASTRA / "MEMORY.md").write_text("wrong")
    with pytest.raises(ValueError, match="checksum"):
        sync_version(version=ASTRA, cache_dir=tmp_path / "cache")


@pytest.mark.parametrize("version", [GPT5, ASTRA])
def test_unversioned_layout_requires_its_historical_client(hub, tmp_path, version):
    import shutil

    shutil.rmtree(hub.snapshot / "libero")
    root = hub.snapshot / "libero"
    root.mkdir()
    (root / "MEMORY.md").write_text("legacy")
    with pytest.raises(ValueError, match="historical client"):
        sync_version(version=version, cache_dir=tmp_path / "cache")
    assert not hub.calls


def test_output_copy_and_versions_do_not_overwrite_each_other(hub, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as executor:
        roots = list(
            executor.map(
                lambda v: sync_version(version=v, cache_dir=tmp_path / "cache"),
                [GPT5, ASTRA],
            )
        )
    assert [(p / "MEMORY.md").read_text() for p in roots] == [GPT5, ASTRA]
    out = tmp_path / "export"
    assert (
        sync_version(version=ASTRA, cache_dir=tmp_path / "cache", output_dir=out) == out
    )
    assert (out / "MEMORY.md").read_text() == ASTRA
    with pytest.raises(ValueError, match="already exists"):
        sync_version(version=GPT5, cache_dir=tmp_path / "cache", output_dir=out)


def test_local_explore_conflicts_and_replay(tmp_path, monkeypatch):
    for profile, explore in [("local", False), ("local", True), ("hf", True)]:
        with pytest.raises(ValueError, match="requires"):
            loading.validate_memory_options(
                Namespace(memory_version=ASTRA, memory_profile=profile, explore=explore)
            )
    with pytest.raises(ValueError, match="no Flash"):
        replay_directory(tmp_path)
    task_card = tmp_path / "task_card"
    task_card.mkdir()
    (task_card / "object_task_t0_plan.json").write_text("{}")
    assert replay_directory(tmp_path) == task_card


def test_task_model_changes_resolve_root_again_without_changing_effort(
    hub, tmp_path, monkeypatch
):
    from robots.libero import robot_spec, toolkit
    from rpent.cli import main as run_cli
    from rpent.dashboard.events import NullDashboardEventSink

    monkeypatch.setattr(loading, "get_memory_dir", lambda _: tmp_path / "cache")
    monkeypatch.setattr(toolkit, "LiberoToolkit", lambda **kw: SimpleNamespace(**kw))
    spec = robot_spec.get_robot_spec()
    parser = run_cli._build_argparser()
    spec.add_cli_args(parser, use_dashboard=False)
    args = parser.parse_args(
        [
            "--robot",
            "libero",
            "--suite",
            "libero_object_task",
            "--task",
            "0",
            "--planner",
            "codex",
            "--reasoning-effort",
            "low",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    for model, version in [("gpt-5.5", GPT5), ("gpt-6-astra", ASTRA)]:
        args.model = model
        config = spec.parse_config(args)
        loading.prepare_run_memory(args, spec, config)
        root = config.prompt_vars["memory_dir"]
        assert root.name == version
        assert args.reasoning_effort == "low"
        prompt = spec.prompts.render(
            "system", variables={**config.prompt_vars, "output_dir": config.output_dir}
        )
        assert str(root / "MEMORY.md") in prompt
        assert "MULTI-ATTEMPT EXPLORE MODE" not in prompt
        bound = robot_spec.get_toolkit(
            runtime_kwargs={},
            dashboard_events=NullDashboardEventSink(),
            config=config,
        )
        assert bound.memory.root == root.resolve()
        assert (bound.memory.root / "MEMORY.md").read_text() == version

    downloaded = len(hub.calls)
    args.memory_profile = "local"
    args.memory_dir = str(root)
    for explore in (False, True):
        args.explore = explore
        config = spec.parse_config(args)
        loading.prepare_run_memory(args, spec, config)
        assert config.prompt_vars["memory_dir"] == str(root)
        prompt = spec.prompts.render(
            "system", variables={**config.prompt_vars, "output_dir": config.output_dir}
        )
        assert str(root / "MEMORY.md") in prompt
        assert ("MULTI-ATTEMPT EXPLORE MODE" in prompt) is explore
    assert len(hub.calls) == downloaded


def test_flash_prepares_the_version_with_published_replay_assets(
    hub, tmp_path, monkeypatch
):
    monkeypatch.setattr(loading, "get_memory_dir", lambda _: tmp_path / "cache")
    spec = SimpleNamespace(name="libero", memory_repo_id="test/repo")
    args = Namespace(planner="flash", model="gpt-6-astra", memory_version="auto")
    config = SimpleNamespace(prompt_vars={})
    loading.prepare_run_memory(args, spec, config)
    root = config.prompt_vars["memory_dir"]
    assert root.name == GPT5
    assert replay_directory(root) == root / "task_card"
    assert (root / "task_card/object_task_t0_anchors.json").is_file()

    args.memory_version = ASTRA
    with pytest.raises(ValueError, match="has no Flash/Task Card replay assets"):
        loading.prepare_run_memory(args, spec, config)
    assert len(hub.calls) == 1


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ([], GPT5),
        (["--planner", "codex"], ASTRA),
        (["--planner", "codex", "--model", "gpt-5.5"], GPT5),
        (["--planner", "codex", "--memory-version", GPT5], GPT5),
        (["--planner", "api", "--model", "openai:gpt-6-astra"], ASTRA),
    ],
)
def test_sync_and_run_use_same_model_selection(
    options, expected, monkeypatch, tmp_path
):
    from rpent.cli import main as run_cli
    from rpent.cli import memory as memory_cli

    monkeypatch.setenv("CODEX_MODEL", "gpt-6-astra")
    selected = []

    def sync(**kwargs):
        selected.append(kwargs["version"])
        return tmp_path / kwargs["version"]

    monkeypatch.setattr(memory_cli, "sync_version", sync)
    monkeypatch.setattr(loading, "sync_version", sync)
    monkeypatch.setattr(sys, "argv", ["rpent-memory", "sync", *options])
    assert memory_cli.main() == 0

    args = run_cli._build_argparser().parse_args(["--robot", "libero", *options])
    config = SimpleNamespace(prompt_vars={})
    spec = SimpleNamespace(name="libero", memory_repo_id="test/repo")
    loading.prepare_run_memory(args, spec, config)
    assert selected == [expected, expected]
    assert config.prompt_vars["memory_dir"] == tmp_path / expected
    if args.planner == "codex":
        assert args.model == ("gpt-5.5" if "--model" in options else "gpt-6-astra")


def test_manifest_published_file_set_is_required(hub, tmp_path):
    extra = hub.snapshot / "libero" / ASTRA / "global" / "unlisted.md"
    extra.write_text("not in the published manifest")
    with pytest.raises(ValueError, match="selected file list"):
        sync_version(version=ASTRA, cache_dir=tmp_path / "cache")
    assert not hub.calls


def test_legacy_cache_cannot_bypass_versioned_source_requirement(hub, tmp_path):
    cache = tmp_path / "cache"
    root = sync_version(version=GPT5, cache_dir=cache)
    receipt = root.parent / f"{GPT5}.receipt.json"
    files = json.loads(receipt.read_text())["files"]
    # Pre-version receipts do not identify a validated versioned Hub subtree.
    receipt.write_text(json.dumps(files))
    hub.fail = True
    with pytest.raises(RuntimeError, match="no complete cache"):
        sync_version(version=GPT5, cache_dir=cache)
    hub.fail = False
    assert sync_version(version=GPT5, cache_dir=cache) == root
    assert len(hub.calls) == 2
