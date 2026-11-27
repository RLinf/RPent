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

import json

import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner.base import build_planner
from rpent.planner.check import LlmCheckRequest, check_llm
from rpent.planner.opencode import OpenCodePlanner
from rpent.planner.utils.acp import AcpSession


@pytest.mark.parametrize(
    ("config", "detail"),
    [
        ("{bad", "JSONDecodeError"),
        ("null", "AttributeError"),
        ('{"provider": null}', "Invalid provider configuration"),
    ],
)
def test_check_reports_invalid_inline_config(monkeypatch, config, detail):
    monkeypatch.setenv("OPENCODE_BIN", "/configured/opencode")
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", config)

    async def reject_config(self, prompt):
        raise RuntimeError("Invalid provider configuration")

    monkeypatch.setattr(AcpSession, "run", reject_config)
    result = check_llm(LlmCheckRequest(planner="opencode"))
    assert result.status == "sdk_error"
    assert detail in result.detail


@pytest.mark.parametrize("probe", [False, True])
def test_opencode_preserves_provider_and_scopes_tools(tmp_path, monkeypatch, probe):
    monkeypatch.setenv("OPENCODE_BIN", "/configured/opencode")
    monkeypatch.setenv("OPENCODE_DISABLE_EXTERNAL_SKILLS", "0")
    original = json.dumps(
        {"provider": {"local": {"options": {"baseURL": "https://example.invalid"}}}}
    )
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", original)
    planner = build_planner(
        "opencode",
        output_dir=tmp_path,
        recipe_tag="test",
        robot_name="libero",
        model="local/exact-model",
        dashboard_events=NullDashboardEventSink(),
    )
    session = planner._create_session(
        tmp_path,
        None,
        lambda event: None,
        tmp_path / "stderr.log",
        mcp_url=None if probe else "http://127.0.0.1:1234/mcp/",
        tool_names=() if probe else ("observe", "finish"),
        redact=planner._load_secret_redactor(),
    )
    config = json.loads(session._env["OPENCODE_CONFIG_CONTENT"])
    assert config["provider"] == json.loads(original)["provider"]
    assert config["share"] == "disabled" and config["autoupdate"] is False
    agent = config["agent"][config["default_agent"]]
    assert "steps" not in agent
    assert agent["permission"] == (
        {"*": "deny"}
        if probe
        else {"*": "deny", "rpent_observe": "allow", "rpent_finish": "allow"}
    )
    assert session._command == ["/configured/opencode", "acp", "--pure"]
    assert session._model == "local/exact-model"
    assert session._env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    assert session._env["OPENCODE_DISABLE_CLAUDE_CODE"] == "1"
    assert session._env["OPENCODE_DISABLE_EXTERNAL_SKILLS"] == "1"
    assert "127.0.0.1" in session._env["NO_PROXY"]
    assert not list(tmp_path.iterdir())


def test_missing_opencode_is_reported_without_guessing_credentials(monkeypatch):
    monkeypatch.delenv("OPENCODE_BIN", raising=False)
    monkeypatch.setattr("rpent.planner.opencode.shutil.which", lambda name: None)
    result = check_llm(LlmCheckRequest(planner="opencode"))
    assert result.status == "sdk_error"
    assert "OPENCODE_BIN" in result.detail
    assert result.credential_present is None


def test_redaction_snapshots_environment_and_inline_provider_keys(monkeypatch):
    env_key = "opencode-env-secret"
    inline_key = "opencode-inline-secret"
    monkeypatch.setenv("TEST_API_KEY", env_key)
    monkeypatch.setenv(
        "OPENCODE_CONFIG_CONTENT",
        json.dumps({"provider": {"local": {"options": {"apiKey": inline_key}}}}),
    )
    redact = OpenCodePlanner._load_secret_redactor()
    monkeypatch.setenv("TEST_API_KEY", "replacement-env-secret")
    monkeypatch.delenv("OPENCODE_CONFIG_CONTENT")
    detail = redact(f"Request failed: {env_key} {inline_key}")
    assert detail == "Request failed: <TEST_API_KEY> <provider.local.options.apiKey>"
    assert redact("replacement-env-secret") == "replacement-env-secret"


def test_check_redacts_inline_provider_key(monkeypatch):
    secret = "opencode-inline-secret"
    monkeypatch.setenv(
        "OPENCODE_CONFIG_CONTENT",
        json.dumps({"provider": {"local": {"options": {"apiKey": secret}}}}),
    )

    def fail(self, prompt):
        raise RuntimeError(f"API key rejected: {secret}")

    monkeypatch.setattr(OpenCodePlanner, "probe", fail)
    result = check_llm(LlmCheckRequest(planner="opencode"))
    assert result.status == "auth_failed"
    assert "<provider.local.options.apiKey>" in result.detail
    assert secret not in result.detail
