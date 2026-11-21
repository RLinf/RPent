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

"""``rpent-flywheel`` delegates to the ``tool-flywheel`` component."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rpent.cli import flywheel as flywheel_cli
from rpent.install.require import MissingComponentError


def test_flywheel_entry_runs_the_component_cli(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple] = []

    def component_main(*args):
        calls.append(args)
        return 3

    monkeypatch.setattr(
        flywheel_cli,
        "require",
        lambda component, module: SimpleNamespace(main=component_main),
    )

    assert flywheel_cli.main() == 3
    # The component parses ``sys.argv`` itself.
    assert calls == [()]


def test_flywheel_entry_reports_a_missing_component(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def missing(component, module):
        raise MissingComponentError(
            f"component {component!r} requires {module!r}, which is not importable"
        )

    monkeypatch.setattr(flywheel_cli, "require", missing)

    assert flywheel_cli.main() == 1
    assert "component 'tool-flywheel' requires" in capsys.readouterr().err
