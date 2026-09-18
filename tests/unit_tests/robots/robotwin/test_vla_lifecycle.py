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

import sys
from types import ModuleType, SimpleNamespace

import pytest

from robots.robotwin import vla_server


@pytest.mark.parametrize("fail", [False, True])
def test_robotwin_closes_watcher_on_model_failure_and_normal_return(
    tmp_path, monkeypatch, fail
):
    config = tmp_path / "robot.yaml"
    config.write_text("{}")
    calls = []
    monkeypatch.setattr(
        vla_server,
        "watch_parent_death",
        lambda callback: SimpleNamespace(close=lambda: calls.append("close")),
    )
    policy_module = ModuleType("deploy.lingbot_vla_policy")
    server_module = ModuleType("deploy.websocket_policy_server")

    def policy(*args, **kwargs):
        calls.append("model")
        if fail:
            raise RuntimeError("model failed")
        return SimpleNamespace(infer=lambda obs: obs)

    policy_module.LingbotVLAServer = policy
    server_module.WebsocketPolicyServer = lambda *args, **kwargs: SimpleNamespace(
        serve_forever=lambda: calls.append("serve")
    )
    monkeypatch.setitem(sys.modules, "deploy", ModuleType("deploy"))
    monkeypatch.setitem(sys.modules, policy_module.__name__, policy_module)
    monkeypatch.setitem(sys.modules, server_module.__name__, server_module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vla",
            "--model-path",
            "/model",
            "--norm-path",
            "/norm",
            "--port",
            "1234",
            "--parent-watch",
            "--lingbot-robot-config",
            str(config),
        ],
    )
    if fail:
        with pytest.raises(RuntimeError, match="model failed"):
            vla_server.main()
        assert calls == ["model", "close"]
    else:
        vla_server.main()
        assert calls == ["model", "serve", "close"]
