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

"""Offline tests for ``make_env`` init-state handling (RLinf/RPent#246).

Shipped perturbation suites can contain tasks whose ``pruned_init`` file
holds zero states. ``make_env`` must refuse to build such an env with a
diagnostic naming every affected task, instead of dying on a modulo by
zero or silently shrinking the evaluation denominator.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

import robots.libero.env_server as env_server


class FakeSuite:
    """Mimics the LIBERO benchmark API surface ``make_env`` consumes."""

    def __init__(self, counts: list[int]):
        self._counts = counts

    def get_num_tasks(self) -> int:
        return len(self._counts)

    @property
    def n_tasks(self) -> int:
        return len(self._counts)

    def get_task_init_states(self, task_id: int) -> list[object]:
        return [object()] * self._counts[task_id]


def _install_fake_rlinf(monkeypatch: pytest.MonkeyPatch, suite: FakeSuite):
    """Stub the ``rlinf`` benchmark/env modules ``make_env`` imports.

    Returns ``(env_cls, captured_cfgs)`` so tests can inspect the env cfg
    that a successful ``make_env`` hands to ``LiberoEnv``.
    """
    captured_cfgs: list = []

    class FakeLiberoEnv:
        def __init__(self, *, cfg, **kwargs):
            del kwargs
            captured_cfgs.append(cfg)

    names = (
        "rlinf",
        "rlinf.envs",
        "rlinf.envs.sim",
        "rlinf.envs.sim.libero",
        "rlinf.envs.sim.libero.libero_env",
        "rlinf.envs.sim.libero.utils",
        "rlinf.envs.sim.libero.utils.benchmark",
    )
    modules: dict[str, ModuleType] = {}
    for name in names:
        module = ModuleType(name)
        if name not in (
            "rlinf.envs.sim.libero.libero_env",
            "rlinf.envs.sim.libero.utils.benchmark",
        ):
            module.__path__ = []  # mark intermediate nodes as packages
        modules[name] = module
    modules["rlinf.envs.sim.libero.libero_env"].LiberoEnv = FakeLiberoEnv
    modules["rlinf.envs.sim.libero.utils.benchmark"].get_benchmark = (
        lambda _suite_name: lambda: suite
    )
    modules["rlinf.envs.sim.libero.utils"].benchmark = modules[
        "rlinf.envs.sim.libero.utils.benchmark"
    ]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return FakeLiberoEnv, captured_cfgs


def test_make_env_fails_loudly_on_empty_init_states(monkeypatch):
    counts = [50] * 10
    counts[2] = 0  # libero_10_task#2 ships 0 states (RLinf/RPent#246)
    _install_fake_rlinf(monkeypatch, FakeSuite(counts))

    with pytest.raises(RuntimeError, match="libero_10_task") as excinfo:
        env_server.make_env(task_id=2, seed=0, suite_name="libero_10_task")

    message = str(excinfo.value)
    assert "task 2" in message
    assert "0 init states" in message


def test_make_env_error_lists_every_empty_task_in_suite(monkeypatch):
    counts = [50] * 10
    counts[3] = 0
    counts[7] = 0  # libero_spatial_task#3/#7 ship 0 states (RLinf/RPent#246)
    _install_fake_rlinf(monkeypatch, FakeSuite(counts))

    with pytest.raises(RuntimeError, match=r"\[3, 7\]"):
        env_server.make_env(task_id=3, seed=0, suite_name="libero_spatial_task")


def test_make_env_keeps_offset_reset_id_for_healthy_tasks(monkeypatch):
    counts = [10, 20, 30]
    env_cls, captured_cfgs = _install_fake_rlinf(monkeypatch, FakeSuite(counts))

    env = env_server.make_env(task_id=2, seed=5, suite_name="libero_spatial")

    assert isinstance(env, env_cls)
    # first_id = 10 + 20, rid = first_id + seed % trials = 30 + 5 % 30.
    assert captured_cfgs[0].specific_reset_id == 35
