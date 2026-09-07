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

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.robots import RunConfig


def _run_config(memory_dir: Path, *, recipe_tag: str = "cell-s0") -> RunConfig:
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=memory_dir.parent / "run",
        prompt_vars={"memory_dir": str(memory_dir)},
        task_desc={},
    )


@pytest.mark.parametrize("robot_name", ["robocasa", "robotwin"])
def test_evaluation_toolkit_factories_use_configured_read_only_memory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    robot_name: str,
) -> None:
    robot_spec = import_module(f"robots.{robot_name}.robot_spec")
    toolkit_module = import_module(f"robots.{robot_name}.toolkit")
    toolkit_name = {"robocasa": "RoboCasaToolkit", "robotwin": "RoboTwinToolkit"}[
        robot_name
    ]
    captured: dict[str, Any] = {}

    def fake_toolkit(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(toolkit_module, toolkit_name, fake_toolkit)
    memory_dir = tmp_path / robot_name / "memory"
    kwargs_name = "primitives_kwargs" if robot_name == "robotwin" else "runtime_kwargs"
    toolkit = robot_spec.get_toolkit(
        **{kwargs_name: {"env": "offline"}},
        dashboard_events=NullDashboardEventSink(),
        config=_run_config(memory_dir),
    )

    assert toolkit.memory.root == memory_dir.resolve()
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        toolkit.memory.authorize_write(memory_dir / "global" / "strategy.md")
    assert captured[kwargs_name] == {"env": "offline"}


@pytest.mark.parametrize("robot_name", ["libero", "robocasa", "robotwin"])
def test_toolkit_factories_fall_back_to_each_robot_memory_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    robot_name: str,
) -> None:
    robot_spec = import_module(f"robots.{robot_name}.robot_spec")
    toolkit_module = import_module(f"robots.{robot_name}.toolkit")
    toolkit_name = {
        "libero": "LiberoToolkit",
        "robocasa": "RoboCasaToolkit",
        "robotwin": "RoboTwinToolkit",
    }[robot_name]
    default_memory = tmp_path / robot_name / "memory"
    monkeypatch.setattr(robot_spec, "get_memory_dir", lambda _: default_memory)
    monkeypatch.setattr(
        toolkit_module, toolkit_name, lambda **kwargs: SimpleNamespace(**kwargs)
    )
    config = RunConfig(
        recipe_tag="cell-s0",
        output_dir=tmp_path / "run",
        prompt_vars={},
        task_desc={},
    )
    kwargs_name = "primitives_kwargs" if robot_name == "robotwin" else "runtime_kwargs"
    toolkit = robot_spec.get_toolkit(
        **{kwargs_name: {}},
        dashboard_events=NullDashboardEventSink(),
        config=config,
    )
    assert toolkit.memory.root == default_memory.resolve()
