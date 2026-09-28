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
import sys
from pathlib import Path

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
from rpent.utils.config import get_memory_dir
from rpent.utils.daemon import ProcessDaemon
from rpent.utils.rpc.http_rpc import HttpRpcClient


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
        (
            "behavior",
            ["--task-name", "turning_on_radio", "--public-seed", "1"],
            ("task_name", "public_seed"),
        ),
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
        ("behavior", "--task-name is required"),
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


def test_behavior_uses_shared_memory_profiles(tmp_path: Path) -> None:
    memory_dir = tmp_path / "behavior-memory"
    args = _parser("behavior").parse_args(
        [
            "--task-name",
            "turning_on_radio",
            "--public-seed",
            "1",
            "--memory-dir",
            str(memory_dir),
            "--memory-profile",
            "local",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    config = get_robot_spec("behavior").parse_config(args)

    assert args.memory_profile == "local"
    assert args.memory_dir == str(memory_dir.resolve())
    assert config.prompt_vars["memory_profile"] == "local"
    assert config.prompt_vars["memory_dir"] == str(memory_dir.resolve())
    assert config.prompt_vars["memory_inbox"].endswith(
        "_internal/inbox/turning_on_radio_s1"
    )

    evaluation = _parser("behavior").parse_args(
        [
            "--task-name",
            "turning_on_radio",
            "--public-seed",
            "1",
            "--memory-profile",
            "hf",
        ]
    )
    config = get_robot_spec("behavior").parse_config(evaluation)
    assert config.prompt_vars["memory_profile"] == "hf"
    assert config.prompt_vars["mode"] == "eval"
    exploration = _parser("behavior").parse_args(
        [
            "--task-name",
            "turning_on_radio",
            "--public-seed",
            "0",
            "--explore",
        ]
    )
    config = get_robot_spec("behavior").parse_config(exploration)
    assert config.prompt_vars["memory_profile"] == "local"
    assert config.prompt_vars["mode"] == "explore"


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
        [
            "--suite",
            "libero_goal",
            "--task",
            "0",
            *extra_args,
        ]
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

    assert config.recipe_tag == "robotwin_block_hammer_beat_demo_clean_s13"
    assert config.output_dir == tmp_path
    assert config.prompt_vars["mode"] == "eval"
    assert config.prompt_vars["memory_profile"] == "hf"
    assert config.prompt_vars["reference_tag"] == "block_hammer_beat_s0"
    assert config.prompt_vars["instruction"].startswith("<native")
    assert config.task_desc["seed_mode"] == "exact"
    assert config.task_desc["env_cuda_device"] == "2"
    assert config.task_desc["vla_cuda_device"] == "3"
    assert args.env_endpoint == "http://offline.invalid:1"
    assert args.vla_endpoint == "ws://offline.invalid:2"


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
            "--vla-endpoint",
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
        [
            "--task-name",
            "block_hammer_beat",
            "--seed",
            "1",
        ]
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


@pytest.mark.parametrize(
    ("task_name", "mode", "seed", "instruction"),
    [
        (
            "turning_on_radio",
            "explore",
            0,
            "Turn on the radio receiver that's on the table in the living room.",
        ),
        (
            "picking_up_trash",
            "explore",
            0,
            "Put the three can of soda from the living room inside the tash can "
            "in the kitchen.",
        ),
        (
            "picking_up_trash",
            "eval",
            10,
            "Put the three can of soda from the living room inside the tash can "
            "in the kitchen.",
        ),
    ],
)
@pytest.mark.parametrize("batched", [False, True])
def test_runtime_preserves_rlinf_task_language(
    tmp_path: Path,
    task_name: str,
    mode: str,
    seed: int,
    instruction: str,
    batched: bool,
) -> None:
    import argparse

    from robots.behavior import runtime

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--explore", action="store_true")
    runtime.add_cli_args(parser, use_dashboard=False)
    cli_args = [
        "--task-name",
        task_name,
        "--public-seed",
        str(seed),
        "--behavior-repo",
        str(tmp_path / "RLinf"),
        "--output-dir",
        str(tmp_path / "output"),
    ]
    if mode == "explore":
        cli_args.append("--explore")
    args = parser.parse_args(cli_args)
    config = runtime.parse_config(args)
    meta = runtime.env_runtime_contract(args)
    assert meta["task_language"] == instruction
    for key in ("task_language", "task_instruction", "instruction"):
        assert config.prompt_vars[key] == instruction

    class Rpc:
        language = instruction

        def call(self, method, **kwargs):
            if method == "env.get_env_meta":
                return meta
            assert method == "env.reset"
            text = [self.language] if batched else self.language
            return {"task_descriptions": text}, {"done": {"success": False}}

    rpc = Rpc()
    connected = runtime._connect_env(args, rpc, config.output_dir)
    expected = [instruction] if batched else instruction
    assert connected["initial_observation"]["task_descriptions"] == expected
    assert connected["env"].official_success_latched is False

    rpc.language = "Put the three soda cans from the living room inside the trash can in the kitchen."
    with pytest.raises(RuntimeError, match="task language does not match TaskSpec"):
        runtime._connect_env(args, rpc, config.output_dir)


def test_env_endpoint_discovery_uses_actual_bind_and_ignores_old_log(tmp_path: Path):
    from robots.behavior.runtime import _wait_for_server_endpoint

    log = tmp_path / "env.log"
    log.write_text("RPC server listening on http://127.0.0.1:1\n")
    offset = log.stat().st_size
    daemon = ProcessDaemon(
        "test_env",
        [
            sys.executable,
            "-c",
            "from types import SimpleNamespace; "
            "from robots.behavior.env_server import BehaviorEnvFacade; "
            "print('Ray started; no application endpoint yet', flush=True); "
            "BehaviorEnvFacade(backend=SimpleNamespace(close=lambda: None), "
            "meta={'task_language': 'test'}).serve("
            "transport='http', host='127.0.0.1', port=0, parent_watch=True)",
        ],
        log_path=str(log),
    )
    daemon.start()
    rpc = None
    try:
        endpoint = _wait_for_server_endpoint(daemon, log_offset=offset)
        assert endpoint != "http://127.0.0.1:1"
        rpc = HttpRpcClient(endpoint)
        assert rpc.call("healthz") == {"status": "ok"}
        assert rpc.call("env.get_env_meta") == {"task_language": "test"}
        assert rpc.call("shutdown") == {"ok": True}
    finally:
        if rpc is not None:
            rpc.close()
        daemon.stop()


def test_env_endpoint_discovery_reports_early_exit(tmp_path: Path):
    from types import SimpleNamespace

    from robots.behavior.runtime import _wait_for_server_endpoint

    log = tmp_path / "env.log"
    log.write_text("initialization failed\n")
    daemon = SimpleNamespace(log_path=str(log), name="test_env", poll=lambda: 2)
    with pytest.raises(RuntimeError, match="exited with code 2"):
        _wait_for_server_endpoint(daemon)


def test_robot_config_separates_world_and_self_collision_padding(tmp_path):
    import yaml

    from robots.behavior.motion import build_robot_config

    names = [f"{h}_arm_joint{i}" for h in ("left", "right") for i in range(1, 8)]
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<robot name="test">'
        + "".join(f'<joint name="{n}" type="revolute"/>' for n in names)
        + "</robot>"
    )
    source = {
        "collision_link_names": ["base_link", "left_arm_link1"],
        "collision_spheres": {},
        "collision_sphere_buffer": 0.002,
        "self_collision_buffer": {"base_link": 0.02},
        "self_collision_ignore": {},
        "extra_links": {},
        "extra_collision_spheres": {},
        "cspace": {
            "joint_names": names,
            "cspace_distance_weight": [1] * 14,
            "null_space_weight": [1] * 14,
        },
    }
    config_file = tmp_path / "collision.yaml"
    config_file.write_text(yaml.safe_dump({"robot_cfg": {"kinematics": source}}))
    result = build_robot_config(
        {
            "collision_config_path": str(config_file),
            "urdf_path": str(urdf),
            "joint_positions": dict.fromkeys(names, 0),
        }
    )["kinematics"]
    assert result["collision_sphere_buffer"] == 0.002
    assert result["self_collision_buffer"]["base_link"] == pytest.approx(0.018)
    assert result["self_collision_buffer"]["left_arm_link1"] == -0.002
    assert result["self_collision_ignore"] == source["self_collision_ignore"]


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (
            [
                "--explore-attempts-per-session",
                "1",
            ],
            "BEHAVIOR explore runs one attempt per session; use --explore-sessions",
        ),
        (
            ["--env-endpoint", "127.0.0.1:1"],
            "BEHAVIOR explore requires an owned env sidecar; omit --env-endpoint",
        ),
    ],
)
def test_behavior_explore_rejects_incompatible_options(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra_args: list[str],
    message: str,
) -> None:
    from rpent.cli import main as cli

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rpent",
            "--robot",
            "behavior",
            "--task-name",
            "turning_on_radio",
            "--public-seed",
            "0",
            "--explore",
            *extra_args,
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_behavior_external_sidecar_python_gets_source_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robots.behavior import runtime

    monkeypatch.setenv("PYTHONPATH", "/tmp/existing-pythonpath")

    env = runtime._behavior_subprocess_env(
        cuda_device="2",
        ROBOT_PLATFORM="BEHAVIOR",
    )

    assert env["ROBOT_PLATFORM"] == "BEHAVIOR"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert "PYTHONPATH" not in env
    assert env["RPENT_REPO_ROOT"] == str(runtime.get_repo_root())
