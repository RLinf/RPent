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

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robots.libero import robot_spec as libero_robot_spec
from robots.libero import toolkit as libero_toolkit
from robots.robocasa import robot_spec as robocasa_robot_spec
from robots.robocasa import toolkit as robocasa_toolkit
from robots.robotwin import robot_spec as robotwin_robot_spec
from robots.robotwin import toolkit as robotwin_toolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.robots import RunConfig


def _run_config(memory_dir: Path, *, recipe_tag: str = "cell-s0") -> RunConfig:
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=memory_dir.parent / "run",
        prompt_vars={"memory_dir": str(memory_dir)},
        task_desc={},
    )


@pytest.mark.parametrize(
    ("robot_spec", "toolkit_module", "toolkit_name", "configured_leaf"),
    [
        (robocasa_robot_spec, robocasa_toolkit, "RoboCasaToolkit", "memory"),
        (robotwin_robot_spec, robotwin_toolkit, "RoboTwinToolkit", "memory"),
    ],
)
def test_evaluation_toolkit_factories_use_configured_read_only_memory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    robot_spec: Any,
    toolkit_module: Any,
    toolkit_name: str,
    configured_leaf: str,
) -> None:
    captured: dict[str, Any] = {}

    def fake_toolkit(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(toolkit_module, toolkit_name, fake_toolkit)
    resources_dir = tmp_path / robot_spec.__name__
    configured_dir = resources_dir / configured_leaf
    memory_dir = resources_dir / "memory"

    toolkit = robot_spec.get_toolkit(
        primitives_kwargs={"env": "offline"},
        dashboard_events=NullDashboardEventSink(),
        config=_run_config(configured_dir),
    )

    assert toolkit.memory.root == memory_dir.resolve()
    write = toolkit.memory.get_common_tool_bindings()["write_text_file"][1]
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        write(str(memory_dir / "global" / "strategy.md"), "changed")
    assert captured["primitives_kwargs"] == {"env": "offline"}


@pytest.mark.parametrize(
    ("robot_name", "robot_spec", "toolkit_module", "toolkit_name"),
    [
        ("libero", libero_robot_spec, libero_toolkit, "LiberoToolkit"),
        ("robotwin", robotwin_robot_spec, robotwin_toolkit, "RoboTwinToolkit"),
    ],
)
def test_toolkit_factories_fall_back_to_each_robot_memory_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    robot_name: str,
    robot_spec: Any,
    toolkit_module: Any,
    toolkit_name: str,
) -> None:
    default_memory = tmp_path / robot_name / "memory"
    monkeypatch.setattr(
        robot_spec,
        "get_memory_dir",
        lambda requested_robot: (
            default_memory if requested_robot == robot_name else None
        ),
    )
    monkeypatch.setattr(
        toolkit_module,
        toolkit_name,
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    config = RunConfig(
        recipe_tag="cell-s0",
        output_dir=tmp_path / "run",
        prompt_vars={},
        task_desc={},
    )

    toolkit = robot_spec.get_toolkit(
        primitives_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        config=config,
    )

    assert toolkit.memory.root == default_memory.resolve()
