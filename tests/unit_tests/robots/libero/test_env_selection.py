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

"""Task selection must preserve the requested benchmark identity."""

import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from robots.libero.env_server import make_env


@pytest.fixture
def backend(monkeypatch):
    counts = (2, 0, 4)
    queried = []

    def initial_states(task):
        queried.append(task)
        return list(range(counts[task]))

    suite = SimpleNamespace(n_tasks=len(counts), get_task_init_states=initial_states)
    constructor = Mock(side_effect=lambda **kwargs: kwargs["cfg"])
    for name in ("rlinf", "rlinf.envs", "rlinf.envs.libero"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    module = types.ModuleType("rlinf.envs.libero.libero_env")
    module.LiberoEnv = constructor
    monkeypatch.setitem(sys.modules, module.__name__, module)
    utils = types.ModuleType("rlinf.envs.libero.utils")
    utils.benchmark = SimpleNamespace(get_benchmark=lambda _: lambda: suite)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    return constructor, queried


@pytest.mark.parametrize("task", [-1, -3, 3, 10])
def test_invalid_task_cannot_alias_another_task(backend, task):
    constructor, queried = backend
    with pytest.raises(ValueError, match="task_id.*demo"):
        make_env(task_id=task, seed=0, suite_name="demo")
    assert not queried
    constructor.assert_not_called()


def test_empty_selected_task_is_rejected_before_construction(backend):
    constructor, _ = backend
    with pytest.raises(ValueError, match="demo.*task 1.*no initial states"):
        make_env(task_id=1, seed=0, suite_name="demo")
    constructor.assert_not_called()


@pytest.mark.parametrize(
    ("task", "seed", "reset_id"), [(0, 3, 1), (2, 0, 2), (2, 5, 3), (2, 7, 5)]
)
def test_valid_task_offsets_and_seed_wrapping_are_preserved(
    backend, task, seed, reset_id
):
    constructor, _ = backend
    config = make_env(task_id=task, seed=seed, suite_name="demo")
    assert config.specific_reset_id == reset_id
    assert config.seed == seed
    assert config.task_suite_name == "demo"
    constructor.assert_called_once()
