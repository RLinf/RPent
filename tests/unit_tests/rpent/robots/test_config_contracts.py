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
import importlib
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from robots.dual_franka.runtime_config import (
    DEFAULT_CONFIG as DUAL_FRANKA_DEFAULT_CONFIG,
)
from robots.franka.runtime_config import (
    DEFAULT_CONFIG as FRANKA_CONFIG,
)
from robots.franka.runtime_config import (
    get_robot_config_path,
    set_robot_config_path,
)
from rpent.robots import get_robot_spec
from rpent.robots.components import am_runtime
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
    "robot_name,backend",
    [
        ("libero", "pi05"),
        ("libero", "cosmos-policy"),
        ("libero", "cosmos3"),
        ("libero", "fast-wam"),
        ("robotwin", "lingbot"),
        ("robotwin", "fast-wam"),
    ],
)
def test_am_help_distinguishes_connection_deployment_and_models(
    robot_name, backend, capsys
):
    with pytest.raises(SystemExit) as exc:
        _parser(robot_name).parse_args(["--am-backend", backend, "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--env-cuda-device" in help_text
    assert "--cuda-device" in help_text
    assert "--sim-cuda-device" not in help_text
    for group in (
        "Action model:",
        "Action-model connection (repeated tasks):",
        "Action-model local deployment (per run):",
    ):
        assert group in help_text
    assert "add_action_model.html#action-model-deployment" in help_text
    for option, model in (
        ("cosmos-text-embeddings", "cosmos-policy"),
        ("predict-future", "cosmos-policy"),
        ("cosmos-cached-instructions-only", "cosmos-policy"),
        ("cosmos3-config", "cosmos3"),
        ("cosmos3-sglang-endpoint", "cosmos3"),
        ("fast-wam-config", "fast-wam"),
        ("fast-wam-dataset-stats", "fast-wam"),
        ("lingbot-robot-config", "lingbot"),
        ("pi05-repo-id", "pi05"),
    ):
        assert (f"--{option}" in help_text) == (backend == model)
        legacy = option.replace("cosmos-", "wam-").replace("fast-wam-", "wam-")
        if legacy != option:
            assert f"--{legacy}" not in help_text


@pytest.mark.parametrize(
    "robot_name", ["libero", "robotwin", "robocasa", "robodojo", "dual_franka"]
)
@pytest.mark.parametrize(
    "device,shared_device",
    [
        (None, None),
        (None, "1"),
        ("0", None),
        ("0", "1"),
        ("2,3", "1"),
        ("GPU-example", "1"),
    ],
)
def test_local_vla_gpu_override_reaches_worker(
    monkeypatch, tmp_path, robot_name, device, shared_device
):
    from rpent.utils import daemon

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "inherited")

    module = importlib.import_module(f"robots.{robot_name}.robot_spec")
    factory = Mock()
    monkeypatch.setattr(am_runtime, "ProcessDaemon", factory)
    monkeypatch.setattr(daemon, "ProcessDaemon", factory)
    monkeypatch.setattr(daemon, "pick_free_port", lambda: 8116)
    if hasattr(module, "ProcessDaemon"):
        monkeypatch.setattr(module, "ProcessDaemon", factory)
        monkeypatch.setattr(module, "pick_free_port", lambda: 8116)
    interpreter = tmp_path / "python"
    interpreter.touch(mode=0o755)
    args = _parser(robot_name, dashboard=True).parse_args(
        (["--vla-cuda-device", device] if device is not None else [])
        + (["--env-cuda-device", "2"] if robot_name != "dual_franka" else [])
        + (["--cuda-device", shared_device] if shared_device is not None else [])
        + [
            "--am-python",
            str(interpreter),
            "--am-root",
            str(tmp_path),
            "--am-checkpoint",
            str(tmp_path),
        ]
    )
    device = device if device is not None else shared_device
    args.am_checkpoint = str(tmp_path)
    if args.am_backend == "pi05" and robot_name != "robodojo":
        args.pi05_repo_id = "test/data"
    if robot_name == "robotwin":
        monkeypatch.setattr(module, "_resolve_vla_runtime_path", lambda args: tmp_path)
        monkeypatch.setattr(
            module,
            "_resolve_lingbot_robot_config",
            lambda *args: tmp_path / "robot.yaml",
        )
        assert module._resolve_cuda_devices(args) == ("2", device)
    elif robot_name == "robodojo":
        args.source_root = str(tmp_path)
        args.am_backend = "xpolicylab"
        args.task = "put_bottles_into_dustbin"
    module._spawn_vla_server(args, tmp_path)
    command = factory.call_args.kwargs["cmd"]
    assert command[0] == str(interpreter)
    assert factory.call_args.kwargs["cwd"] == str(tmp_path)
    checkpoint_flag = "--ckpt" if robot_name == "robodojo" else "--model-path"
    assert command[command.index(checkpoint_flag) + 1] == str(tmp_path)
    overrides = factory.call_args.kwargs["env_overrides"]
    assert overrides.get("CUDA_VISIBLE_DEVICES") == device
    assert ("CUDA_VISIBLE_DEVICES" in overrides) == (device is not None)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "inherited"
    assert (
        str(args.cuda_device) if args.cuda_device is not None else None
    ) == shared_device
    if robot_name == "robodojo":
        from rpent.robots.components.xpolicylab_vla_server import add_backend_args

        parser = argparse.ArgumentParser()
        add_backend_args(parser)
        child_args, _ = parser.parse_known_args(factory.call_args.kwargs["cmd"])
        if device is not None:
            assert child_args.policy_gpu == device
        else:
            assert "--policy-gpu" not in factory.call_args.kwargs["cmd"]
    if robot_name != "dual_franka":
        args.seed = 0
        if robot_name == "libero":
            args.suite, args.task = "libero_spatial", 0
        elif robot_name == "robotwin":
            monkeypatch.setattr(
                module, "_resolve_env_runtime_path", lambda args: tmp_path
            )
        _, rpc = module._spawn_env_server(args, tmp_path)
        try:
            if robot_name in ("robotwin", "robodojo"):
                assert (
                    factory.call_args.kwargs["env_overrides"]["CUDA_VISIBLE_DEVICES"]
                    == "2"
                )
            if robot_name != "robotwin":
                command = factory.call_args.kwargs["cmd"]
                assert command[command.index("--cuda-device") + 1] == "2"
        finally:
            rpc.close()


@pytest.mark.parametrize("robot_name", ["libero", "robodojo", "dual_franka"])
@pytest.mark.parametrize(
    "device,shared_device", [(None, None), (None, 1), (0, None), (0, 1)]
)
def test_sam3_gpu_override_reaches_worker(
    monkeypatch, tmp_path, robot_name, device, shared_device
):
    module = importlib.import_module(f"robots.{robot_name}.robot_spec")
    factory = Mock()
    monkeypatch.setattr(module, "ProcessDaemon", factory)
    args = _parser(robot_name, dashboard=True).parse_args(
        ([] if device is None else ["--sam3-cuda-device", str(device)])
        + ([] if shared_device is None else ["--cuda-device", str(shared_device)])
    )
    device = device if device is not None else shared_device
    args.am_cuda_device = str(device) if device is not None else None
    args.env_cuda_device = device
    if robot_name == "dual_franka":
        monkeypatch.setenv("SAM3_CHECKPOINT_PATH", str(tmp_path / "sam3.pt"))
    _, rpc = module._spawn_sam3_server(args, tmp_path)
    try:
        command = factory.call_args.kwargs["cmd"]
        assert ("--cuda-device" in command) == (device is not None)
        if device is not None:
            assert command[command.index("--cuda-device") + 1] == str(device)
    finally:
        rpc.close()


@pytest.mark.parametrize(
    "robot_name", ["libero", "robotwin", "robocasa", "robodojo", "dual_franka"]
)
def test_gpu_help_explains_shared_devices(robot_name):
    parser = _parser(robot_name, dashboard=True)
    for action in parser._actions:
        if action.dest.endswith("_cuda_device") and action.help != argparse.SUPPRESS:
            assert "may share GPUs" in action.help
            assert "only" not in action.help


@pytest.mark.parametrize(
    "alias,canonical,value",
    [
        ("--vla-endpoint", "--am-endpoint", "http://host:8116"),
        ("--wam-endpoint", "--am-endpoint", "http://host:8116"),
        ("--vla-model-path", "--am-checkpoint", "/models/checkpoint"),
        ("--wam-checkpoint", "--am-checkpoint", "/models/checkpoint"),
        ("--wam-python", "--am-python", "/env/bin/python"),
        ("--pi05-python", "--am-python", "/env/bin/python"),
        ("--wam-root", "--am-root", "/model/source"),
        ("--vla-cuda-device", "--am-cuda-device", "0,1"),
        ("--wam-cuda-device", "--am-cuda-device", "0,1"),
    ],
)
def test_legacy_am_options_use_one_destination(alias, canonical, value):
    parser = _parser("libero", dashboard=True)
    old, new = parser.parse_args([alias, value]), parser.parse_args([canonical, value])
    assert vars(old) == vars(new)
    assert not hasattr(old, alias[2:].replace("-", "_"))
    assert alias not in parser.format_help()


@pytest.mark.parametrize(
    "platform,options,message",
    [
        ("libero", ["--predict-future"], "requires --am-backend cosmos-policy"),
        (
            "libero",
            ["--am-backend", "fast-wam", "--predict-future"],
            "requires --am-backend cosmos-policy",
        ),
        (
            "libero",
            [
                "--am-backend",
                "cosmos-policy",
                "--am-endpoint",
                "http://host:8116",
                "--predict-future",
            ],
            "configures local deployment",
        ),
        (
            "libero",
            ["--fast-wam-config", "config.yaml"],
            "requires --am-backend fast-wam",
        ),
        (
            "robotwin",
            ["--fast-wam-config", "config.yaml"],
            "requires --am-backend fast-wam",
        ),
        (
            "libero",
            ["--am-backend", "fast-wam", "--pi05-repo-id", "data"],
            "requires --am-backend pi05",
        ),
        (
            "robotwin",
            ["--am-backend", "fast-wam", "--lingbot-robot-config", "robot.yaml"],
            "requires --am-backend lingbot",
        ),
        (
            "robocasa",
            ["--am-endpoint", "http://host:8000", "--am-checkpoint", "/model"],
            "configures local deployment",
        ),
        (
            "robotwin",
            ["--am-endpoint", "ws://host:8000", "--am-cuda-device", "0"],
            "configures local deployment",
        ),
        (
            "libero",
            ["--am-endpoint", "http://host:8000", "--am-python", "python"],
            "configures local deployment",
        ),
        (
            "libero",
            ["--am-endpoint", "http://host:8000", "--am-root", "/model"],
            "configures local deployment",
        ),
        ("franka", ["--am-checkpoint", "/model"], "external model service"),
        ("yam", ["--am-cuda-device", "0"], "external model service"),
    ],
)
def test_inapplicable_am_options_fail_before_startup(platform, options, message):
    args = _parser(platform, dashboard=True).parse_args(options)
    with pytest.raises(ValueError, match=message):
        am_runtime.select_am(args, platform)


def test_backend_choices_are_scoped_to_platform():
    parser = _parser("robotwin", dashboard=True)
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--am-backend", "cosmos-policy"])
    assert exc.value.code == 2


def test_environment_defaults_do_not_become_explicit_launch_options(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("PI05_CHECKPOINT_PATH", str(tmp_path))
    parser = _parser("libero", dashboard=True)
    args = parser.parse_args([])
    selected = am_runtime.select_am(args, "libero")
    assert selected.checkpoint == str(tmp_path)
    assert args.am_checkpoint is None
    args.am_endpoint = "http://host:8000"
    assert am_runtime.select_am(args, "libero").checkpoint is None
    assert selected.endpoint is None


def test_relative_model_python_is_resolved_before_changing_worker_directory(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    interpreter = tmp_path / "venv" / "python"
    interpreter.parent.mkdir()
    interpreter.touch(mode=0o755)
    worker_root = tmp_path / "model"
    worker_root.mkdir()
    args = _parser("libero", dashboard=True).parse_args(
        [
            "--am-python",
            "venv/python",
            "--am-root",
            "model",
        ]
    )
    selected = am_runtime.select_am(args, "libero")
    assert selected.python == str(interpreter)
    assert selected.root == worker_root


@pytest.mark.parametrize(
    "robot_name,default_backend", [("libero", "pi05"), ("robotwin", "lingbot")]
)
def test_am_backend_selects_one_model_category(robot_name, default_backend):
    from rpent.robots.components.am_runtime import select_am

    parser = _parser(robot_name, dashboard=True)
    args = parser.parse_args(["--am-endpoint", "http://localhost:8116"])
    assert args.am_backend == default_backend
    assert (
        parser.parse_args(["--vla-backend", default_backend]).am_backend
        == default_backend
    )
    assert not select_am(args, robot_name).is_wam
    args = parser.parse_args(
        ["--am-backend", "fast-wam", "--am-endpoint", "http://localhost:8117"]
    )
    assert select_am(args, robot_name).endpoint == args.am_endpoint
    # The Dashboard descriptor does not require a complete deployment yet.
    args.am_endpoint = None
    assert "wam" in {
        item["name"]
        for item in get_robot_spec(robot_name).resolve_dashboard(args)[
            "runtime_components"
        ]
    }
    assert "--vla-backend" not in parser.format_help()
    assert "--wam-backend" not in parser.format_help()


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
        "policy_kind": "vla",
        "policy_backend": "pi05",
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

    task_specific = memory_dir / "task-specific"
    task_specific.mkdir()
    (task_specific / "goal_t0_s0.json").write_text("{}")
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
        "mode": "eval",
        "memory_profile": "hf",
        "reference_tag": "PnPCounterToCab_s0",
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
            "--am-endpoint",
            "ws://offline.invalid:2",
            "--env-cuda-device",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )

    config = get_robot_spec("robotwin").parse_config(args)

    assert config.recipe_tag == "robotwin_block_hammer_beat_demo_clean_s13"
    assert config.output_dir == tmp_path
    assert config.prompt_vars["mode"] == "eval"
    assert config.prompt_vars["memory_profile"] == "hf"
    assert config.prompt_vars["reference_tag"] == "block_hammer_beat_s0"
    assert config.prompt_vars["instruction"].startswith("<native")
    assert config.task_desc["seed_mode"] == "exact"
    assert config.task_desc["env_cuda_device"] == "2"
    assert config.task_desc["vla_cuda_device"] is None
    assert args.env_endpoint == "http://offline.invalid:1"
    assert args.am_endpoint == "ws://offline.invalid:2"


def test_robotwin_exploration_uses_config_qualified_local_reference(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / "robotwin-memory"
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
            "--am-endpoint",
            "ws://offline.invalid:2",
            "--output-dir",
            str(tmp_path / "run"),
            "--explore",
            "--memory-profile",
            "local",
            "--memory-dir",
            str(memory_dir),
        ]
    )

    config = get_robot_spec("robotwin").parse_config(args)

    assert config.recipe_tag == "robotwin_block_hammer_beat_demo_clean_s13"
    assert config.prompt_vars["mode"] == "explore"
    assert config.prompt_vars["memory_profile"] == "local"
    assert (
        config.prompt_vars["reference_tag"]
        == "robotwin_block_hammer_beat_demo_clean_s0"
    )
    assert config.prompt_vars["memory_inbox"] == str(
        memory_dir.resolve()
        / "_internal"
        / "inbox"
        / "robotwin_block_hammer_beat_demo_clean_s13"
    )


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
    assert am_runtime.select_am(args, "robotwin").checkpoint == "/offline/model"
    from robots.robotwin.robot_spec import _resolve_lingbot_robot_config

    monkeypatch.setattr(Path, "is_file", lambda self: True)
    assert _resolve_lingbot_robot_config(args, Path("/offline/model")) == Path(
        "/offline/robot.yaml"
    )


@pytest.mark.parametrize("robot_name", ["libero", "robotwin", "robocasa", "robodojo"])
@pytest.mark.parametrize("option", ["--sim-cuda-device"])
def test_removed_gpu_options_are_rejected(robot_name, option) -> None:
    with pytest.raises(SystemExit) as exc:
        _parser(robot_name, dashboard=True).parse_args([option, "0"])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("robot_name", "packaged_default"),
    [
        ("franka", FRANKA_CONFIG),
        ("dual_franka", DUAL_FRANKA_DEFAULT_CONFIG),
    ],
)
def test_robot_config_flag_is_recorded_at_parse_time(
    robot_name: str,
    packaged_default: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Isolate `~` so the packaged default's easy_handeye YAML paths are
    # deterministically missing and the fail-fast check kicks in.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    calib = tmp_path / "calib.yaml"
    calib.write_text("parameters: {}\ntransformation: {}\n")
    custom = tmp_path / "robot.yaml"
    custom.write_text(f"perception:\n  calibration:\n    external: {calib}\n")
    args = _parser(robot_name).parse_args(["--robot-config", str(custom)])
    try:
        get_robot_spec(robot_name).parse_config(args)
        assert get_robot_config_path() == custom

        # The packaged defaults reference on-robot easy_handeye YAMLs: parse
        # fails fast, but only after the default config path is recorded.
        args = _parser(robot_name).parse_args([])
        with pytest.raises(ValueError, match="easy_handeye"):
            get_robot_spec(robot_name).parse_config(args)
        assert get_robot_config_path() == packaged_default
    finally:
        set_robot_config_path(None)
