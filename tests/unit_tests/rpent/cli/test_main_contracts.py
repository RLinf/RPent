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

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
ROBOT_HELP_CASES = [
    ("libero", ("--suite", "--task", "--libero-type")),
    ("robocasa", ("--task-name", "--split", "--hi-res")),
    ("robotwin", ("--task-name", "--task-config", "--robotwin-assets-path")),
]


class ConfigCaptured(Exception):
    """Stop the CLI immediately after argument validation in routing tests."""


def _cli_module():
    from rpent.cli import main as cli

    return cli


def _run_cli_help(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-m", "rpent.cli.main", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def _capture_validated_args(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> tuple[str, Any]:
    cli = _cli_module()
    captured: dict[str, Any] = {}

    def add_cli_args(parser, use_dashboard: bool) -> None:
        del use_dashboard
        parser.add_argument("--explore-sessions", type=int, default=1)

    def parse_config(args) -> None:
        captured["args"] = args
        raise ConfigCaptured

    def get_robot_spec(name: str):
        captured["robot_name"] = name
        return SimpleNamespace(
            add_cli_args=add_cli_args,
            parse_config=parse_config,
        )

    monkeypatch.setattr(
        cli, "enumerate_robots", lambda: ("libero", "robocasa", "robotwin")
    )
    monkeypatch.setattr(cli, "get_robot_spec", get_robot_spec)
    monkeypatch.setattr(sys, "argv", ["rpent", *argv])

    with pytest.raises(ConfigCaptured):
        cli.main()
    return captured["robot_name"], captured["args"]


def test_public_cli_help_runs_from_source_checkout_without_a_robot_runtime() -> None:
    result = _run_cli_help("--help")

    assert result.returncode == 0, result.stderr
    assert "RPent: Agentic Infrastructure for the Physical World" in result.stdout
    for option in (
        "--robot",
        "--env",
        "--planner",
        "--output-dir",
        "--dashboard",
        "--interactive",
    ):
        assert option in result.stdout


@pytest.mark.parametrize(
    ("robot_name", "robot_options"),
    ROBOT_HELP_CASES,
)
def test_public_robot_specific_help_exits_cleanly_and_includes_extension_options(
    robot_name: str,
    robot_options: tuple[str, ...],
) -> None:
    result = _run_cli_help("--robot", robot_name, "--help")

    assert result.returncode == 0, result.stderr
    assert all(option in result.stdout for option in robot_options)


def test_shared_cli_defaults_reach_robot_config_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot_name, args = _capture_validated_args(
        monkeypatch,
        ["--robot", "libero"],
    )

    assert robot_name == "libero"
    assert args.robot_name == "libero"
    assert args.planner == "api"
    assert args.model is None
    assert args.max_turns == 100
    assert args.max_tokens == 8192
    assert args.dashboard is False
    assert args.dashboard_host == "127.0.0.1"
    assert args.dashboard_port == 0
    assert args.dashboard_language == "en"
    assert args.memory_profile == "hf"


def test_deprecated_env_alias_routes_to_the_same_robot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot_name, args = _capture_validated_args(
        monkeypatch,
        ["--env", "robocasa"],
    )

    assert robot_name == "robocasa"
    assert args.robot_name == "robocasa"
    assert args.env_name == "robocasa"


def test_robot_and_env_aliases_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _cli_module()
    monkeypatch.setattr(
        cli, "enumerate_robots", lambda: ("libero", "robocasa", "robotwin")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["rpent", "--robot", "libero", "--env", "libero"],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert "provide only one" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["--robot", "libero", "--dashboard", "--interactive"],
            "cannot be used together",
        ),
        (["--robot", "robocasa", "--explore"], "supported only for LIBERO"),
        (
            ["--robot", "libero", "--explore", "--memory-profile", "hf"],
            "cannot be used with --memory-profile hf",
        ),
        (
            ["--robot", "libero", "--explore", "--explore-sessions", "0"],
            "greater than 0",
        ),
        (
            [
                "--robot",
                "libero",
                "--memory-profile",
                "hf",
                "--memory-dir",
                "/offline/memory",
            ],
            "requires --memory-profile local or --explore",
        ),
    ],
)
def test_shared_cli_validation_stops_before_robot_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    message: str,
) -> None:
    cli = _cli_module()
    parse_called = False

    def add_cli_args(parser, use_dashboard: bool) -> None:
        del use_dashboard
        parser.add_argument("--explore-sessions", type=int, default=1)

    def parse_config(args) -> None:
        nonlocal parse_called
        del args
        parse_called = True

    monkeypatch.setattr(
        cli, "enumerate_robots", lambda: ("libero", "robocasa", "robotwin")
    )
    monkeypatch.setattr(
        cli,
        "get_robot_spec",
        lambda name: SimpleNamespace(
            name=name,
            add_cli_args=add_cli_args,
            parse_config=parse_config,
        ),
    )
    monkeypatch.setattr(sys, "argv", ["rpent", *argv])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err
    assert parse_called is False


def test_transcript_serialization_strips_nested_images_without_mutating_input() -> None:
    cli = _cli_module()
    messages = [
        {
            "role": "user",
            "metadata": {"turn": 1},
            "content": [
                {"type": "text", "text": "inspect"},
                {
                    "type": "image",
                    "source": {"type": "base64", "data": "sensitive-image"},
                },
                {
                    "nested": {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,sensitive"},
                    }
                },
            ],
        }
    ]
    before = copy.deepcopy(messages)

    serialized = cli._serialize_messages(messages)

    assert messages == before
    assert serialized[0]["metadata"] == {"turn": 1}
    assert serialized[0]["content"][0] == {"type": "text", "text": "inspect"}
    assert serialized[0]["content"][1] == {
        "type": "image",
        "source": {"_omitted_for_transcript": True},
    }
    assert serialized[0]["content"][2]["nested"] == {
        "type": "image_url",
        "image_url": {"_omitted_for_transcript": True},
    }
    assert "sensitive" not in repr(serialized)


def test_handoff_message_lists_prior_attempts_deterministically(tmp_path: Path) -> None:
    cli = _cli_module()
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    (attempts / "attempt_10_failed.json").write_text("{}")
    (attempts / "attempt_02_failed.json").write_text("{}")
    (attempts / "unrelated.json").write_text("{}")

    message = cli._handoff_message(tmp_path, session_number=2, session_max=4)

    assert "agent 2 of up to 4" in message
    assert "2 attempt(s)" in message
    assert message.index("attempt_02_failed.json") < message.index(
        "attempt_10_failed.json"
    )
    assert "unrelated.json" not in message
    assert "memory inbox under wip/" in message


def test_full_cli_exploration_finalizes_memory_without_starting_gpu_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _cli_module()
    from rpent.planner.base import PlannerResult
    from rpent.robots import PromptBundle, RobotSpec, RunConfig
    from rpent.tools.toolkit import ToolResult

    calls: dict[str, Any] = {}

    class FakeMemoryManager:
        def merge_memory(self, **kwargs: Any) -> dict[str, int]:
            calls["merge_memory"] = kwargs
            return {"suite": 1}

    class FakeDaemon:
        stopped = False

        def stop(self) -> None:
            self.stopped = True

    class FakeToolkit:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []
            self.closed = False
            self.memory = FakeMemoryManager()

        def execute_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
            self.calls.append((name, args))
            return ToolResult(
                name,
                {
                    "_finish": True,
                    "status": args["status"],
                    "summary": args["summary"],
                },
            )

        def close(self) -> None:
            self.closed = True

        def solved(self) -> bool:
            return True

        def write_recipe(self, recipe_tag: str) -> str:
            calls["write_recipe"] = recipe_tag
            return str(tmp_path / f"recipe_{recipe_tag}.jsonl")

    class ScriptedPlanner:
        def solve(
            self,
            *,
            system_prompt: str,
            user_message: str,
            toolkit: FakeToolkit,
            max_turns: int,
            input_queue: Any = None,
            dashboard_interaction: Any = None,
        ) -> PlannerResult:
            calls["solve"] = {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "max_turns": max_turns,
                "input_queue": input_queue,
                "dashboard_interaction": dashboard_interaction,
            }
            finish = toolkit.execute_tool(
                "finish",
                {"status": "success", "summary": "simulated task complete"},
            )
            return PlannerResult(
                finish_result=finish.result,
                messages=[{"role": "assistant", "content": "finished offline"}],
                stats={
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "tool_calls": 1,
                },
            )

    daemon = FakeDaemon()
    toolkit = FakeToolkit()
    planner = ScriptedPlanner()

    def add_cli_args(parser: Any, use_dashboard: bool) -> None:
        del use_dashboard
        parser.add_argument("--auto-merge-memory", action="store_true")
        parser.add_argument("--explore-sessions", type=int, default=1)
        parser.add_argument("--explore-attempts-per-session", type=int, default=2)

    def parse_config(args: Any) -> RunConfig:
        return RunConfig(
            recipe_tag="libero_s0",
            output_dir=Path(args.output_dir),
            prompt_vars={"memory_dir": args.memory_dir},
            task_desc={"robot": "libero"},
        )

    def init_runtime(*args: Any) -> tuple[list[FakeDaemon], dict[str, str]]:
        calls["init_runtime"] = args
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        return [daemon], {"runtime": "simulated"}

    robot_spec = RobotSpec(
        name="libero",
        prompts=PromptBundle(
            system=lambda variables: "simulated system prompt",
            user=lambda variables: "simulated user task",
        ),
        add_cli_args=add_cli_args,
        parse_config=parse_config,
        init_runtime=init_runtime,
    )

    def build_planner(*args: Any, **kwargs: Any) -> ScriptedPlanner:
        calls["build_planner"] = (args, kwargs)
        return planner

    def get_toolkit(*args: Any, **kwargs: Any) -> FakeToolkit:
        calls["get_toolkit"] = (args, kwargs)
        return toolkit

    def reject_resource_download(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            f"CPU-only smoke test tried to download resources: {args!r}"
        )

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(cli, "enumerate_robots", lambda: ("libero",))
    monkeypatch.setattr(cli, "get_robot_spec", lambda name: robot_spec)
    monkeypatch.setattr(cli, "build_planner", build_planner)
    monkeypatch.setattr(cli, "get_toolkit", get_toolkit)
    monkeypatch.setattr(cli, "ensure_resources", reject_resource_download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent",
            "--robot",
            "libero",
            "--explore",
            "--auto-merge-memory",
            "--memory-profile",
            "local",
            "--memory-dir",
            str(tmp_path / "memory"),
            "--output-dir",
            str(tmp_path),
            "--max-turns",
            "4",
        ],
    )

    assert cli.main() == 0

    assert calls["solve"] == {
        "system_prompt": "simulated system prompt\n",
        "user_message": "simulated user task\n",
        "max_turns": 4,
        "input_queue": None,
        "dashboard_interaction": None,
    }
    assert toolkit.calls == [
        ("finish", {"status": "success", "summary": "simulated task complete"})
    ]
    assert toolkit.closed is True
    assert daemon.stopped is True
    assert calls["get_toolkit"][1]["primitives_kwargs"] == {"runtime": "simulated"}
    assert calls["get_toolkit"][1]["mode"] == "exploration"
    assert calls["get_toolkit"][1]["attempts_per_session"] == 2
    assert calls["write_recipe"] == "libero_s0"
    assert calls["merge_memory"] == {
        "cell_tag": "libero_s0",
        "run_state_dir": tmp_path,
        "solved": True,
    }

    transcript = json.loads((tmp_path / "transcript_libero_s0.json").read_text())
    assert transcript["robot"] == "libero"
    assert transcript["finish"] == {
        "_finish": True,
        "status": "success",
        "summary": "simulated task complete",
    }
    assert transcript["stats"]["tool_calls"] == 1
    assert transcript["messages"] == [
        {"role": "assistant", "content": "finished offline"}
    ]
