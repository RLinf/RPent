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
        (root / "MEMORY.md").write_text(version)
        files[version] = {
            "files": {"MEMORY.md": hashlib.sha256(version.encode()).hexdigest()}
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
    assert hub.calls[0]["allow_patterns"] == [f"libero/{ASTRA}/MEMORY.md"]
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


def test_legacy_layout_only_serves_gpt5(hub, tmp_path):
    import shutil

    shutil.rmtree(hub.snapshot / "libero")
    root = hub.snapshot / "libero"
    root.mkdir()
    (root / "MEMORY.md").write_text("legacy")
    result = sync_version(version=GPT5, cache_dir=tmp_path / "cache")
    assert (result / "MEMORY.md").read_text() == "legacy"
    with pytest.raises(ValueError, match="has no"):
        sync_version(version=ASTRA, cache_dir=tmp_path / "cache")


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
    legacy = tmp_path / "task_card"
    legacy.mkdir()
    (legacy / "object_task_t0_plan.json").write_text("{}")
    assert replay_directory(tmp_path) == legacy


def test_task_model_changes_resolve_root_again_without_changing_effort(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(loading, "sync_version", lambda **kw: tmp_path / kw["version"])
    spec = SimpleNamespace(name="libero", memory_repo_id="test/repo")
    args = Namespace(
        planner="codex",
        model="gpt-5.5",
        memory_profile="hf",
        memory_version="auto",
        reasoning_effort="low",
    )
    for model, version in [("gpt-5.5", GPT5), ("gpt-6-astra", ASTRA)]:
        args.model = model
        config = SimpleNamespace(prompt_vars={})
        loading.prepare_run_memory(args, spec, config)
        assert config.prompt_vars["memory_dir"] == tmp_path / version
        assert args.reasoning_effort == "low"
    args.memory_profile = "local"
    config = SimpleNamespace(prompt_vars={"memory_dir": tmp_path})
    loading.prepare_run_memory(args, spec, config)
    assert config.prompt_vars["memory_dir"] == tmp_path
