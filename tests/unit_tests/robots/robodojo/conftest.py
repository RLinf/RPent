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

"""Offline agent adapter fixtures, including environments without RLinf."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def agent_module(monkeypatch):
    name = "rlinf.envs.sim.robodojo.robodojo_env"
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        # The simulator adapter is optional in RPent's base test environment.
        stub = ModuleType(name)

        class RoboDojoEnv:
            def close(self, clear_cache=True):
                if not self._closed:
                    self.venv.close(clear_cache)
                    self._closed = True

        stub.RoboDojoEnv = RoboDojoEnv
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.delitem(sys.modules, "robots.robodojo.rlinf_env", raising=False)
    module = importlib.import_module("robots.robodojo.rlinf_env")
    yield module
    sys.modules.pop("robots.robodojo.rlinf_env", None)


@pytest.fixture
def make_agent(agent_module):
    def create(native, *, meta=None, slot=None, recorder=None):
        agent = object.__new__(agent_module.RoboDojoAgentEnv)
        agent.meta = meta or {}
        agent.eval_fair = agent.meta.get("mode") == "eval-fair"
        agent._closed = False
        agent.recorder = recorder or SimpleNamespace(
            record=lambda obs: None, close=lambda: []
        )
        agent.bottle_mon = agent_module._SafetyMonitor()
        agent.venv = SimpleNamespace(envs=[slot or SimpleNamespace(env=native)])
        return agent

    return create
