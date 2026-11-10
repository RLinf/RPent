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


@pytest.mark.parametrize("probe", [False, True])
def test_opencode_preserves_provider_and_scopes_tools(tmp_path, monkeypatch, probe):
    monkeypatch.setenv("OPENCODE_BIN", "/configured/opencode")
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
        max_turns=3,
        tool_names=() if probe else ("observe", "finish"),
    )
    config = json.loads(session._env["OPENCODE_CONFIG_CONTENT"])
    assert config["provider"] == json.loads(original)["provider"]
    assert config["share"] == "disabled" and config["autoupdate"] is False
    agent = config["agent"][config["default_agent"]]
    assert agent["steps"] == 3
    assert agent["permission"] == (
        {"*": "deny"}
        if probe
        else {"*": "deny", "rpent_observe": "allow", "rpent_finish": "allow"}
    )
    assert session._command == ["/configured/opencode", "acp", "--pure"]
    assert session._model == "local/exact-model"
    assert "127.0.0.1" in session._env["NO_PROXY"]
    assert not list(tmp_path.iterdir())


def test_missing_opencode_is_reported_without_guessing_credentials(monkeypatch):
    monkeypatch.delenv("OPENCODE_BIN", raising=False)
    monkeypatch.setattr("rpent.planner.opencode.shutil.which", lambda name: None)
    result = check_llm(LlmCheckRequest(planner="opencode"))
    assert result.status == "sdk_error"
    assert "OPENCODE_BIN" in result.detail
    assert result.credential_present is None
