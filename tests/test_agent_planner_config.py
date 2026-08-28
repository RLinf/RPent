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

# ruff: noqa: E402, I001

import hashlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest


if importlib.util.find_spec("openai_codex") is None:
    sdk = types.ModuleType("openai_codex")

    class _ApprovalMode:
        deny_all = "deny_all"

    class _Sandbox:
        read_only = "read_only"
        workspace_write = "workspace_write"
        full_access = "full_access"

    class _ReasoningEffort:
        none = "none"
        low = "low"
        medium = "medium"
        high = "high"
        xhigh = "xhigh"

    sdk.ApprovalMode = _ApprovalMode
    sdk.Sandbox = _Sandbox
    sdk.CodexConfig = lambda **kwargs: kwargs
    generated = types.ModuleType("openai_codex.generated")
    generated_v2 = types.ModuleType("openai_codex.generated.v2_all")
    generated_v2.ReasoningEffort = _ReasoningEffort
    sys.modules["openai_codex"] = sdk
    sys.modules["openai_codex.generated"] = generated
    sys.modules["openai_codex.generated.v2_all"] = generated_v2

from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner import claude_code as claude_code_module
from rpent.planner import codex as codex_module
from rpent.planner.base import AgentRuntimeConfig, build_planner
from rpent.planner.claude_code import ClaudeCodePlanner
from rpent.planner.codex import CodexPlanner


@pytest.fixture(autouse=True)
def _clean_planner_environment(monkeypatch):
    for name in (
        "ANTHROPIC_BASE_URL",
        "CODEX_API_KEY",
        "CODEX_API_KEY_FILE",
        "CODEX_BASE_URL",
        "CODEX_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def _secure_key_file(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "codex.key"
    path.write_text(content)
    path.chmod(0o600)
    return path


def _planner(tmp_path: Path, **kwargs) -> CodexPlanner:
    return CodexPlanner(
        output_dir=str(tmp_path / "output"),
        dashboard_events=NullDashboardEventSink(),
        **kwargs,
    )


def test_key_file_is_injected_by_env_name_not_config_value(
    tmp_path, monkeypatch, caplog
):
    key = os.urandom(24).hex()
    key_path = _secure_key_file(tmp_path, key + "\n")
    monkeypatch.setenv("CODEX_API_KEY_FILE", str(key_path))
    monkeypatch.setenv("CODEX_API_KEY", "ignored-inline-value")
    monkeypatch.setenv("CODEX_BASE_URL", "https://provider.example/v1")
    monkeypatch.setattr(codex_module.openai_codex, "CodexConfig", lambda **kw: kw)

    config = _planner(
        tmp_path, model="gpt-5.5", reasoning_effort="xhigh"
    )._build_config("http://127.0.0.1:1234/mcp/")

    injected = config["env"][codex_module.PROVIDER_ENV_KEY]
    assert (
        hashlib.sha256(injected.encode()).digest()
        == hashlib.sha256(key.encode()).digest()
    )
    assert all(key not in override for override in config["config_overrides"])
    assert (
        'model_providers.rpent_proxy.base_url="https://provider.example/v1"'
        in config["config_overrides"]
    )
    assert (
        'model_providers.rpent_proxy.wire_api="responses"' in config["config_overrides"]
    )
    leaked = key in caplog.text
    assert leaked is False


@pytest.mark.parametrize("mode", [0o400, 0o640, 0o644])
def test_key_file_requires_exact_0600(tmp_path, monkeypatch, mode):
    key_path = _secure_key_file(tmp_path, "placeholder")
    key_path.chmod(mode)
    monkeypatch.setenv("CODEX_API_KEY_FILE", str(key_path))
    monkeypatch.setenv("CODEX_BASE_URL", "https://provider.example/v1")

    with pytest.raises(ValueError, match="permissions must be 0600"):
        _planner(tmp_path)


def test_key_file_rejects_symlink(tmp_path, monkeypatch):
    target = _secure_key_file(tmp_path, "placeholder")
    link = tmp_path / "linked.key"
    link.symlink_to(target)
    monkeypatch.setenv("CODEX_API_KEY_FILE", str(link))
    monkeypatch.setenv("CODEX_BASE_URL", "https://provider.example/v1")

    with pytest.raises(ValueError, match="must not be a symlink"):
        _planner(tmp_path)


def test_custom_base_url_requires_a_key(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_BASE_URL", "https://provider.example/v1")

    with pytest.raises(ValueError, match="requires CODEX_API_KEY"):
        _planner(tmp_path)


def test_responses_url_and_formal_runtime_options(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_API_KEY", "placeholder")
    workdir = tmp_path / "work"
    workdir.mkdir()

    planner = _planner(
        tmp_path,
        repo_root=tmp_path,
        workdir=workdir,
        sandbox="workspace_write",
        base_url="https://provider.example",
        model="gpt-5.5",
        reasoning_effort="xhigh",
    )

    assert planner._base_url == "https://provider.example/v1"
    assert planner._turn_options["cwd"] == str(workdir)
    assert (
        planner._turn_options["sandbox"]
        == codex_module.openai_codex.Sandbox.workspace_write
    )
    assert planner._turn_options["model"] == "gpt-5.5"
    assert "effort" not in planner._thread_options
    assert planner._turn_options["effort"] == codex_module.ReasoningEffort.xhigh


def test_operation_url_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_API_KEY", "placeholder")

    with pytest.raises(ValueError, match="API base"):
        _planner(tmp_path, base_url="https://provider.example/v1/responses")


def test_build_planner_forwards_agent_runtime_to_codex(tmp_path, monkeypatch):
    captured = {}

    class _FakeCodexPlanner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(codex_module, "CodexPlanner", _FakeCodexPlanner)
    workdir = tmp_path / "work"

    build_planner(
        "codex",
        output_dir=tmp_path / "output",
        recipe_tag="formal",
        robot_name="libero",
        base_url="https://provider.example/v1",
        model="gpt-5.5",
        reasoning_effort="xhigh",
        dashboard_events=NullDashboardEventSink(),
        agent_runtime=AgentRuntimeConfig(repo_root=tmp_path, workdir=workdir),
    )

    assert captured["repo_root"] == tmp_path
    assert captured["workdir"] == workdir
    assert "sandbox" not in captured
    assert captured["base_url"] == "https://provider.example/v1"


def test_build_planner_forwards_agent_runtime_to_claude_code(tmp_path, monkeypatch):
    captured = {}

    class _FakeClaudeCodePlanner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(claude_code_module, "ClaudeCodePlanner", _FakeClaudeCodePlanner)
    workdir = tmp_path / "claude-work"

    build_planner(
        "claude_code",
        output_dir=tmp_path / "output",
        recipe_tag="formal",
        robot_name="libero",
        base_url="https://anthropic-provider.example",
        model="claude-opus-4-8",
        reasoning_effort="high",
        dashboard_events=NullDashboardEventSink(),
        agent_runtime=AgentRuntimeConfig(repo_root=tmp_path, workdir=workdir),
    )

    assert captured["repo_root"] == tmp_path
    assert captured["workdir"] == workdir
    assert captured["base_url"] == "https://anthropic-provider.example"
    assert captured["model"] == "claude-opus-4-8"
    assert captured["reasoning_effort"] == "high"


def test_claude_code_options_use_runtime_and_endpoint(tmp_path, monkeypatch):
    class _FakeSdk:
        @staticmethod
        def ClaudeAgentOptions(**kwargs):
            return kwargs

    toolkit = types.SimpleNamespace(get_tools_spec=lambda: [])
    monkeypatch.setattr(
        claude_code_module,
        "_build_rpent_server",
        lambda _sdk, *, toolkit: "rpent-server",
    )
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://ignored.example")
    workdir = tmp_path / "claude-work"
    planner = ClaudeCodePlanner(
        output_dir=str(tmp_path / "output"),
        dashboard_events=NullDashboardEventSink(),
        repo_root=tmp_path,
        workdir=workdir,
        base_url="https://anthropic-provider.example/",
        model="claude-opus-4-8",
        reasoning_effort="high",
    )

    options = planner._build_options(_FakeSdk, toolkit=toolkit, max_turns=7)

    assert options["cwd"] == str(workdir)
    assert options["env"] == {
        "ANTHROPIC_BASE_URL": "https://anthropic-provider.example"
    }
    assert options["model"] == "claude-opus-4-8"
    assert options["effort"] == "high"
    assert options["max_turns"] == 7
