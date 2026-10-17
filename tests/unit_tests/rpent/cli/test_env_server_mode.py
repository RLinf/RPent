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

"""Offline contract tests for the ``--env-server`` CLI mode (issue #57)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


class _ExecReplaced(Exception):
    """Stop the CLI at the ``os.execvp`` point in ``--env-server`` tests."""


def _cli_module():
    from rpent.cli import main as cli

    return cli


def _install_fake_spec(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Register a minimal fake robot spec, mirroring test_main_contracts.

    Like every real robot spec, ``add_cli_args`` makes ``--suite`` required
    unless the caller relaxes it (the dashboard does; ``--env-server`` must
    too, since the standalone server takes task selection from its own args).
    """
    cli = _cli_module()

    def add_cli_args(parser, use_dashboard: bool) -> None:
        parser.add_argument(
            "--suite",
            default=None,
            required=not use_dashboard,
        )
        parser.add_argument("--explore-sessions", type=int, default=1)

    def parse_config(args) -> None:
        del args
        raise AssertionError("parse_config must not run in --env-server mode")

    def get_robot_spec(robot: str):
        return SimpleNamespace(
            name=robot,
            add_cli_args=add_cli_args,
            parse_config=parse_config,
            supports_exploration=False,
            supports_human_interactive_exploration=False,
            is_real_robot=False,
            validate_args=None,
        )

    monkeypatch.setattr(cli, "enumerate_robots", lambda: (name,))
    monkeypatch.setattr(cli, "get_robot_spec", get_robot_spec)


def _run_env_server_mode(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> list[str]:
    """Run the CLI until it execs the server; return the captured argv."""
    cli = _cli_module()
    _install_fake_spec(monkeypatch, "libero")
    captured: dict[str, list[str]] = {}

    def fake_execvp(program, args) -> None:
        del program
        captured["argv"] = list(args)
        raise _ExecReplaced

    monkeypatch.setattr(cli.os, "execvp", fake_execvp)
    monkeypatch.setattr(sys, "argv", ["rpent", *argv])

    with pytest.raises(_ExecReplaced):
        cli.main()
    return captured["argv"]


def test_env_server_mode_execs_the_robot_server_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = _run_env_server_mode(monkeypatch, ["--robot", "libero", "--env-server"])

    assert argv == [
        sys.executable,
        str(REPO_ROOT / "robots" / "libero" / "env_server.py"),
    ]


def test_env_server_mode_forwards_extra_args_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv = _run_env_server_mode(
        monkeypatch,
        [
            "--robot",
            "libero",
            "--env-server",
            "--env-server-args",
            "--port",
            "5555",
            "--seed",
            "2",
        ],
    )

    assert argv == [
        sys.executable,
        str(REPO_ROOT / "robots" / "libero" / "env_server.py"),
        "--port",
        "5555",
        "--seed",
        "2",
    ]


def test_env_server_mode_errors_for_robot_without_server_script(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _cli_module()
    _install_fake_spec(monkeypatch, "mockbot")
    monkeypatch.setattr(sys, "argv", ["rpent", "--robot", "mockbot", "--env-server"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert (
        "--env-server is not supported for robot 'mockbot'" in capsys.readouterr().err
    )


def test_env_server_args_require_env_server(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _cli_module()
    _install_fake_spec(monkeypatch, "libero")
    monkeypatch.setattr(
        sys,
        "argv",
        ["rpent", "--robot", "libero", "--env-server-args", "--port", "5555"],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert "--env-server-args requires --env-server" in capsys.readouterr().err
