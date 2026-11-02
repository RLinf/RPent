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

"""Configuration preflight neither starts sessions nor exposes secret metadata."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rpent.planner import deepseek_config


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(
        clients=[],
        result={"status": "ready", "provider": "openai-codex", "model": "test-model"},
    )

    class Client:
        def __init__(self, config):
            self.config = config
            self.closed = False
            state.clients.append(self)

        def start(self):
            patch = json.loads(Path(self.config.patches[0]).read_text())
            config = patch[-1]["insert"][0]["config"]
            if state.result is not None:
                Path(config["resultPath"]).write_text(json.dumps(state.result))

        def close(self):
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "deepseek_harness",
        SimpleNamespace(HarnessClient=Client, HarnessConfig=SimpleNamespace),
    )
    monkeypatch.setattr(deepseek_config, "version", lambda name: "0.2.1rc2")
    return state


def check(home, **kwargs):
    return deepseek_config.preflight_configuration(
        dsh_home=home, model="test-model", reasoning_effort="xhigh", **kwargs
    )


def test_metadata_preflight_closes_without_initializing_session(runtime, tmp_path):
    # The fake client deliberately has no initialize/prompt/request method.
    runtime.result["debug"] = "SYNTHETIC_PRIVATE_CREDENTIAL"
    result = check(tmp_path / "home")
    assert result["provider"] == "openai-codex"
    assert result["dsh_home"] == str(tmp_path / "home")
    assert runtime.clients[0].closed
    assert not Path(runtime.clients[0].config.cwd).exists()
    assert "SYNTHETIC_PRIVATE_CREDENTIAL" not in json.dumps(result)


@pytest.mark.parametrize(
    "status,error,match",
    [
        ("missing_credential", RuntimeError, "rpent-dsh-login"),
        ("unsupported_capability", ValueError, "reasoning or image"),
        ("inspection_failed", RuntimeError, "could not inspect"),
    ],
)
def test_preflight_failures_are_stable_and_do_not_echo_raw_diagnostics(
    runtime, tmp_path, status, error, match, capsys
):
    runtime.result = {"status": status, "detail": "SYNTHETIC_PRIVATE_CREDENTIAL"}
    with pytest.raises(error, match=match) as caught:
        check(tmp_path / "home")
    assert "SYNTHETIC_PRIVATE_CREDENTIAL" not in str(caught.value)
    assert capsys.readouterr().out == ""
    assert runtime.clients[0].closed


def test_preflight_timeout_closes_runtime(runtime, tmp_path):
    runtime.result = None
    with pytest.raises(TimeoutError, match="timed out"):
        check(tmp_path / "home", timeout_s=0.01)
    assert runtime.clients[0].closed


def test_preflight_rejects_changed_model_identity(runtime, tmp_path):
    runtime.result["model"] = "different-model"
    with pytest.raises(RuntimeError, match="different model"):
        check(tmp_path / "home")


def test_preflight_missing_home_and_wrong_version_fail_before_runtime(
    runtime, monkeypatch
):
    monkeypatch.delenv("DSH_HOME", raising=False)
    with pytest.raises(ValueError, match="dsh-home"):
        check(None)
    monkeypatch.setattr(deepseek_config, "version", lambda name: "0.2.0")
    with pytest.raises(RuntimeError, match="0.2.1rc2"):
        check("unused-home")
    assert runtime.clients == []
