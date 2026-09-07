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

import argparse
from pathlib import Path

import pytest

from rpent.robots import get_robot_spec
from rpent.utils.config import get_memory_dir


def _parser(robot_name: str, *, dashboard: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--memory-profile", choices=["hf", "local"], default=None)
    parser.add_argument("--memory-dir", default=None)
    get_robot_spec(robot_name).add_cli_args(parser, use_dashboard=dashboard)
    return parser


@pytest.mark.parametrize(
    ("robot_name", "required_args", "identity_fields"),
    [
        ("libero", ["--suite", "libero_object_task", "--task", "2"], ("suite", "task")),
        ("robocasa", ["--task-name", "OpenDrawer"], ("task_name",)),
        (
            "robotwin",
            ["--task-name", "block_hammer_beat", "--seed", "7"],
            ("task_name", "seed"),
        ),
    ],
)
def test_robot_arguments_are_required_on_cli_but_deferred_for_dashboard(
    robot_name: str,
    required_args: list[str],
    identity_fields: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit):
        _parser(robot_name).parse_args([])
    cli_args = _parser(robot_name).parse_args(required_args)
    assert all(getattr(cli_args, field) is not None for field in identity_fields)

    dashboard_args = _parser(robot_name, dashboard=True).parse_args([])
    assert all(getattr(dashboard_args, field) is None for field in identity_fields)


@pytest.mark.parametrize(
    ("robot_name", "message"),
    [
        ("libero", "--suite is required"),
        ("robocasa", "--task-name is required"),
        ("robotwin", "--task-name is required"),
    ],
)
def test_dashboard_identity_must_be_filled_before_config_parsing(
    robot_name: str,
    message: str,
) -> None:
    args = _parser(robot_name, dashboard=True).parse_args([])

    with pytest.raises(ValueError, match=message):
        get_robot_spec(robot_name).parse_config(args)


def test_libero_default_evaluation_config(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    args = _parser("libero").parse_args(
        [
            "--suite",
            "libero_object_task",
            "--task",
            "2",
            "--seed",
            "7",
            "--output-dir",
            str(output_dir),
        ]
    )

    config = get_robot_spec("libero").parse_config(args)

    assert config.recipe_tag == "object_task_t2_s7"
    assert config.output_dir == output_dir
    assert config.prompt_vars["mode"] == "eval"
    assert config.prompt_vars["memory_profile"] == "hf"
    assert config.prompt_vars["reference_tag"] == "object_task_t2_s0"
    assert config.task_desc == {
        "suite": "libero_object_task",
        "task": 2,
        "seed": 7,
    }


def test_libero_exploration_uses_local_memory_and_session_metadata(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "memory"
    args = _parser("libero").parse_args(
        [
            "--suite",
            "libero_spatial",
            "--task",
            "1",
            "--explore",
            "--explore-sessions",
            "4",
            "--memory-dir",
            str(memory_dir),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    config = get_robot_spec("libero").parse_config(args)

    assert args.memory_profile == "local"
    assert config.prompt_vars["mode"] == "explore"
    assert config.prompt_vars["session_number"] == 1
    assert config.prompt_vars["session_max"] == 4
    assert config.prompt_vars["memory_dir"] == str(memory_dir.resolve())
    assert config.prompt_vars["memory_inbox"].endswith("_internal/inbox/spatial_t1_s0")


def test_libero_local_evaluation_requires_an_existing_corpus(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    args = _parser("libero").parse_args(
        [
            "--suite",
            "libero_goal",
            "--task",
            "0",
            "--memory-profile",
            "local",
            "--memory-dir",
            str(memory_dir),
        ]
    )

    with pytest.raises(ValueError, match="local memory corpus not found"):
        get_robot_spec("libero").parse_config(args)

    memory_dir.mkdir()
    with pytest.raises(ValueError, match="local memory corpus not found"):
        get_robot_spec("libero").parse_config(args)

    task_only = memory_dir / "task_only"
    task_only.mkdir()
    (task_only / "goal_t0_s0.json").write_text("{}")
    config = get_robot_spec("libero").parse_config(args)
    assert config.prompt_vars["memory_profile"] == "local"


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--explore", "--memory-profile", "hf"], "cannot be used"),
        (["--explore", "--explore-sessions", "0"], "greater than 0"),
        (["--memory-profile", "hf", "--memory-dir", "/tmp/memory"], "requires"),
    ],
)
def test_libero_rejects_invalid_mode_and_memory_combinations(
    extra_args: list[str],
    message: str,
) -> None:
    args = _parser("libero").parse_args(
        ["--suite", "libero_goal", "--task", "0", *extra_args]
    )
    with pytest.raises(ValueError, match=message):
        get_robot_spec("libero").parse_config(args)


def test_robocasa_config_defaults_and_valid_override(tmp_path: Path) -> None:
    args = _parser("robocasa").parse_args(
        [
            "--task-name",
            "PnPCounterToCab",
            "--split",
            "pretrain",
            "--seed",
            "11",
            "--hi-res",
            "512",
            "--output-dir",
            str(tmp_path),
        ]
    )

    config = get_robot_spec("robocasa").parse_config(args)

    assert config.recipe_tag == "PnPCounterToCab_pretrain_s11"
    assert config.output_dir == tmp_path
    assert config.prompt_vars == {
        "task_name": "PnPCounterToCab",
        "split": "pretrain",
        "seed": 11,
        "recipe_tag": "PnPCounterToCab_pretrain_s11",
        "memory_dir": str(get_memory_dir("robocasa")),
    }
    assert config.task_desc == {
        "task_name": "PnPCounterToCab",
        "split": "pretrain",
        "seed": 11,
    }


def test_robotwin_external_runtime_config_needs_no_local_assets(tmp_path: Path) -> None:
    args = _parser("robotwin").parse_args(
        [
            "--task-name",
            "block_hammer_beat",
            "--seed",
            "13",
            "--task-config",
            "demo_clean",
            "--env-endpoint",
            "http://offline.invalid:1",
            "--vla-endpoint",
            "ws://offline.invalid:2",
            "--env-cuda-device",
            "2",
            "--vla-cuda-device",
            "3",
            "--output-dir",
            str(tmp_path),
        ]
    )

    config = get_robot_spec("robotwin").parse_config(args)

    assert config.recipe_tag == "robotwin_block_hammer_beat_s13"
    assert config.output_dir == tmp_path
    assert config.prompt_vars["instruction"].startswith("<native")
    assert config.task_desc["seed_mode"] == "exact"
    assert config.task_desc["env_cuda_device"] == "2"
    assert config.task_desc["vla_cuda_device"] == "3"
    assert args.env_endpoint == "http://offline.invalid:1"
    assert args.vla_endpoint == "ws://offline.invalid:2"


def test_robotwin_cli_defaults_can_come_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROBOTWIN_ASSETS_PATH", "/offline/assets")
    monkeypatch.setenv("LINGBOT_MODEL_PATH", "/offline/model")
    monkeypatch.setenv("LINGBOT_ROBOT_CONFIG", "/offline/robot.yaml")

    args = _parser("robotwin").parse_args(
        ["--task-name", "block_hammer_beat", "--seed", "1"]
    )

    assert args.robotwin_assets_path == "/offline/assets"
    assert args.vla_model_path == "/offline/model"
    assert args.lingbot_robot_config == "/offline/robot.yaml"


def test_robotwin_rejects_conflicting_cuda_routes() -> None:
    args = _parser("robotwin").parse_args(
        [
            "--task-name",
            "block_hammer_beat",
            "--seed",
            "1",
            "--env-endpoint",
            "http://offline.invalid:1",
            "--vla-endpoint",
            "ws://offline.invalid:2",
            "--cuda-device",
            "0",
            "--env-cuda-device",
            "1",
        ]
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        get_robot_spec("robotwin").parse_config(args)


def test_robotwin_requires_assets_when_initializing_a_local_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ROBOTWIN_ASSETS_PATH", raising=False)
    args = _parser("robotwin").parse_args(
        [
            "--task-name",
            "block_hammer_beat",
            "--seed",
            "1",
            "--vla-endpoint",
            "ws://offline.invalid:2",
        ]
    )
    spec = get_robot_spec("robotwin")

    # Parsing remains a pure configuration step; optional runtime dependencies
    # and local asset paths are checked only for the selected component.
    spec.parse_config(args)
    with pytest.raises(RuntimeError, match="--robotwin-assets-path is required"):
        spec.init_runtime(args, tmp_path, _NullDashboardEvents(), {"env"})


class _NullDashboardEvents:
    @property
    def enabled(self) -> bool:
        return False

    def emit(self, event: object) -> None:
        del event
