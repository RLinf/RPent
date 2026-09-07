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

import sys
from pathlib import Path
from types import ModuleType

import pytest

from rpent.dashboard.events import NullDashboardEventSink
from rpent.planner.base import PlannerBuildConfig, build_planner


class StubPlanner:
    """Minimal object satisfying the runtime planner check."""

    def solve(self, **kwargs):
        raise AssertionError(f"not exercised by factory tests: {kwargs!r}")


def _build(reference: str, tmp_path: Path, sink: NullDashboardEventSink):
    return build_planner(
        reference,
        output_dir=tmp_path,
        recipe_tag="pick_s0",
        robot_name="testbot",
        base_url="https://planner.example/v1",
        model="research-model",
        max_tokens=1234,
        planner_timeout_s=56,
        reasoning_effort="high",
        claude_code_max_budget_usd=None,
        dashboard_events=sink,
        no_images=True,
    )


def test_external_planner_factory_receives_shared_build_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("test_external_planner")
    expected = StubPlanner()
    received: list[PlannerBuildConfig] = []

    def create_planner(config: PlannerBuildConfig) -> StubPlanner:
        received.append(config)
        return expected

    module.create_planner = create_planner  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    sink = NullDashboardEventSink()

    planner = _build("test_external_planner:create_planner", tmp_path, sink)

    assert planner is expected
    assert received == [
        PlannerBuildConfig(
            output_dir=tmp_path,
            recipe_tag="pick_s0",
            robot_name="testbot",
            base_url="https://planner.example/v1",
            model="research-model",
            max_tokens=1234,
            planner_timeout_s=56,
            reasoning_effort="high",
            dashboard_events=sink,
            no_images=True,
        )
    ]


def test_external_planner_factory_must_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("test_missing_factory")
    monkeypatch.setitem(sys.modules, module.__name__, module)

    with pytest.raises(ValueError, match="has no factory 'create_planner'"):
        _build(
            "test_missing_factory:create_planner",
            tmp_path,
            NullDashboardEventSink(),
        )


def test_external_planner_module_must_be_importable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="could not import custom planner module"):
        _build(
            "rpent_missing_test_module:create_planner",
            tmp_path,
            NullDashboardEventSink(),
        )


@pytest.mark.parametrize("factory_value", [None, object()])
def test_external_planner_factory_must_be_callable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_value: object,
) -> None:
    module = ModuleType("test_noncallable_factory")
    module.create_planner = factory_value  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    with pytest.raises(TypeError, match="factory .* is not callable"):
        _build(
            "test_noncallable_factory:create_planner",
            tmp_path,
            NullDashboardEventSink(),
        )


def test_external_planner_factory_must_return_a_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("test_invalid_planner")
    module.create_planner = lambda config: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)

    with pytest.raises(TypeError, match="without a callable solve method"):
        _build(
            "test_invalid_planner:create_planner",
            tmp_path,
            NullDashboardEventSink(),
        )
