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

"""Contracts for shared per-toolkit memory construction."""

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
from rpent.robots.memory import ToolkitMemoryConfig, create_toolkit_memory


def _write(manager: Any):
    return manager.get_common_tool_bindings()["write_text_file"][1]


def _read(manager: Any):
    return manager.get_common_tool_bindings()["read_text_file"][1]


def test_exploration_memory_is_fresh_lazy_and_stable_per_cell(tmp_path: Path) -> None:
    root = tmp_path / "configured-memory"
    cell_tag = "stable-task_s7"
    config = ToolkitMemoryConfig(
        root=root,
        mode="exploration",
        cell_tag=cell_tag,
    )

    first = create_toolkit_memory(config)
    second = create_toolkit_memory(config)
    inbox = root / "_internal" / "inbox" / cell_tag

    assert first is not second
    assert first.root == second.root == root.resolve()
    assert not root.exists()
    assert not (inbox / "wip").exists()

    note = inbox / "wip" / "notes.md"
    assert _write(first)(str(note), "shared observations")["bytes_written"] == 19
    assert _read(second)(str(note))["content"] == "shared observations"
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        _write(second)(
            str(root / "_internal" / "inbox" / "other-cell" / "notes.md"),
            "forbidden",
        )


@pytest.mark.parametrize("mode", ["evaluation", "replay"])
def test_non_exploration_memory_is_read_only_and_lazy(
    mode: str,
    tmp_path: Path,
) -> None:
    root = tmp_path / mode
    manager = create_toolkit_memory(
        ToolkitMemoryConfig(root=root, mode=mode, cell_tag="unused-cell")
    )

    with pytest.raises(PermissionError, match="writing to memory is denied"):
        _write(manager)(str(root / "global" / "note.md"), "forbidden")
    assert not root.exists()


ADAPTERS = (
    ("libero", libero_robot_spec, libero_toolkit, "LiberoToolkit"),
    ("robocasa", robocasa_robot_spec, robocasa_toolkit, "RoboCasaToolkit"),
    ("robotwin", robotwin_robot_spec, robotwin_toolkit, "RoboTwinToolkit"),
)


@pytest.mark.parametrize("memory_profile", ["local", "hf"])
@pytest.mark.parametrize(
    ("robot_name", "robot_spec", "toolkit_module", "toolkit_name"),
    ADAPTERS,
)
def test_adapter_factories_preserve_configured_profile_root_and_read_only_access(
    memory_profile: str,
    robot_name: str,
    robot_spec: Any,
    toolkit_module: Any,
    toolkit_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configured_root = tmp_path / robot_name / memory_profile
    monkeypatch.setattr(
        toolkit_module,
        toolkit_name,
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    config = RunConfig(
        recipe_tag=f"{robot_name}-cell",
        output_dir=tmp_path / "output",
        prompt_vars={
            "memory_dir": str(configured_root),
            "memory_profile": memory_profile,
        },
        task_desc={},
    )

    instance = robot_spec.get_toolkit(
        primitives_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        config=config,
    )

    assert instance.memory.root == configured_root.resolve()
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        _write(instance.memory)(
            str(configured_root / "global" / "note.md"),
            "forbidden",
        )
    assert not configured_root.exists()


@pytest.mark.parametrize(
    ("robot_name", "robot_spec", "toolkit_module", "toolkit_name"),
    ADAPTERS,
)
def test_adapter_factories_share_exploration_constructor_contract(
    robot_name: str,
    robot_spec: Any,
    toolkit_module: Any,
    toolkit_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    default_root = tmp_path / robot_name / "default-memory"
    cell_tag = f"{robot_name}-stable-cell"
    monkeypatch.setattr(robot_spec, "get_memory_dir", lambda name: default_root)
    monkeypatch.setattr(
        toolkit_module,
        toolkit_name,
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    config = RunConfig(
        recipe_tag=cell_tag,
        output_dir=tmp_path / "output",
        prompt_vars={},
        task_desc={},
    )

    first = robot_spec.get_toolkit(
        primitives_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        mode="exploration",
    )
    second = robot_spec.get_toolkit(
        primitives_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        mode="exploration",
    )

    assert first.memory is not second.memory
    assert first.memory.root == second.memory.root == default_root.resolve()
    assert not default_root.exists()
    own_note = default_root / "_internal" / "inbox" / cell_tag / "wip" / "notes.md"
    assert _write(first.memory)(str(own_note), robot_name)["bytes_written"] == len(
        robot_name
    )
    assert _read(second.memory)(str(own_note))["content"] == robot_name
