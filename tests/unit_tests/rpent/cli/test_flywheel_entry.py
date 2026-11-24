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

import sys
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


def test_help_works_without_the_component_environment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(sys, "argv", ["rpent-flywheel", "--help"])

    def unexpected(component, module):
        raise AssertionError("help must not consult the environment")

    monkeypatch.setattr(flywheel_cli, "require", unexpected)

    with pytest.raises(SystemExit) as exit_info:
        flywheel_cli.main()

    assert exit_info.value.code == 0
    assert "usage: rpent-flywheel" in capsys.readouterr().out


def test_help_survives_a_wheel_without_the_component(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    import builtins

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] == "components":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(sys, "argv", ["rpent-flywheel", "--help"])

    assert flywheel_cli.main() == 0

    output = capsys.readouterr().out
    assert "usage: rpent-flywheel" in output
    # A wheel has no manifests, so the command the help names must be one that
    # works from a checkout instead of the `rpent-install` script whose
    # checkout guard rejects a wheel's root.
    assert "./rpent-install.sh install tool-flywheel" in output


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
