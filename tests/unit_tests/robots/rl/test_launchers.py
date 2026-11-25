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

"""Exercise portable launchers without importing a GPU runtime."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "tools/rl"


def _environment():
    env = os.environ.copy()
    for key in [
        "RPENT_CLUSTER_ROOT",
        "RPENT_SGLANG_PYTHON",
        "MODEL_PATH",
        "RPENT_DATA_ROOT",
        "LIBERO_CONFIG_PATH",
        "LIBERO_PRO_ASSET_PATH",
        "SAM3_CHECKPOINT_PATH",
    ]:
        env.pop(key, None)
    return env


def test_serving_requires_an_explicit_checkpoint():
    result = subprocess.run(
        ["bash", str(SCRIPTS / "serve.sh")],
        env=_environment(),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Set MODEL_PATH" in result.stderr


@pytest.mark.parametrize("explicit_python,mtp", [(False, "1"), (True, "0")])
def test_serving_uses_the_selected_interpreter_without_a_site_profile(
    tmp_path, explicit_python, mtp
):
    executable = tmp_path / ("custom python" if explicit_python else "python")
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'gpus': os.environ['CUDA_VISIBLE_DEVICES']}))\n"
    )
    executable.chmod(0o755)
    env = {
        **_environment(),
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "MODEL_PATH": "/weights/model with spaces",
        "SERVE_GPUS": "4,5",
        "ENABLE_MTP": mtp,
    }
    if explicit_python:
        env["RPENT_SGLANG_PYTHON"] = str(executable)
    result = subprocess.run(
        ["bash", str(SCRIPTS / "serve.sh"), "--disable-cuda-graph"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    request = json.loads(result.stdout)
    argv = request["argv"]
    assert argv[:2] == ["-m", "sglang.launch_server"]
    assert "--enable-cache-report" in argv
    assert argv[argv.index("--model-path") + 1] == env["MODEL_PATH"]
    assert argv[argv.index("--tp") + 1] == "2"
    assert argv[argv.index("--context-length") + 1] == "262144"
    assert argv[argv.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert argv[argv.index("--reasoning-parser") + 1] == "qwen3"
    assert ("--speculative-algorithm" in argv) == (mtp == "1")
    assert argv[-1] == "--disable-cuda-graph"
    assert request["gpus"] == "4,5"


@pytest.mark.parametrize("explicit_assets", [False, True])
def test_runtime_paths_are_relocatable_and_explicit_values_win(
    tmp_path, explicit_assets
):
    command = tmp_path / "rpent"
    command.write_text("#!/usr/bin/env bash\nexit 0\n")
    command.chmod(0o755)
    env = {
        **_environment(),
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "XDG_DATA_HOME": str(tmp_path / "user data"),
    }
    expected_root = Path(env["XDG_DATA_HOME"]) / "rpent"
    expected_assets = expected_root / "libero-pro/assets"
    if explicit_assets:
        env["RPENT_DATA_ROOT"] = str(tmp_path / "shared data")
        env["LIBERO_PRO_ASSET_PATH"] = str(tmp_path / "existing assets")
        expected_root = Path(env["RPENT_DATA_ROOT"])
        expected_assets = Path(env["LIBERO_PRO_ASSET_PATH"])
    result = subprocess.run(
        [
            "bash",
            "-c",
            "source "
            + shlex.quote(str(SCRIPTS / "env.sh"))
            + '; printf "%s\\n" "$RPENT_DATA_ROOT" "$LIBERO_PRO_ASSET_PATH" "$SAM3_CHECKPOINT_PATH" "$LIBERO_CONFIG_PATH"',
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == [
        str(expected_root),
        str(expected_assets),
        str(expected_root / "models/sam3/sam3.pt"),
        str(expected_root / "libero-config"),
    ]


@pytest.mark.parametrize(
    "simulator,task_args",
    [
        ("libero", ["--suite", "libero_object", "--task", "0"]),
        ("robocasa", ["--task-name", "PickPlaceCounterToCabinet"]),
        ("robotwin", ["--task-name", "click_alarmclock"]),
        ("robodojo", ["--task-name", "put_bottles_into_dustbin"]),
    ],
)
def test_single_episode_launcher_preserves_the_selected_simulator(
    tmp_path, simulator, task_args, monkeypatch
):
    from robots.rl import rollout_options

    request = {}

    def capture(executable, command):
        request["argv"] = command
        request["libero_type"] = os.environ["LIBERO_TYPE"]

    monkeypatch.setattr(rollout_options.os, "execv", capture)
    monkeypatch.setattr(sys, "argv", ["run", "--simulator", simulator, *task_args])
    monkeypatch.setenv("RPENT_MODEL", "openai-chat:qwen3.8-27b")
    monkeypatch.setenv("RPENT_BASE_URL", "http://localhost:30080/v1")
    monkeypatch.setenv("RPENT_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("LIBERO_TYPE", "standard")
    rollout_options.main()
    argv = request["argv"]
    assert argv[argv.index("--robot") + 1] == "rl"
    assert argv[argv.index("--simulator") + 1] == simulator
    assert not {"--workflow", "--prompt", "--observation-mode"}.intersection(argv)
    assert "--libero-type" not in argv
    assert request["libero_type"] == "standard"


@pytest.mark.parametrize("sam3_per_gpu", [1, 2])
def test_services_assign_sam_replicas_and_clean_up_owned_processes(
    tmp_path, monkeypatch, sam3_per_gpu
):
    from robots.rl import services

    daemons = []

    class Daemon:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.stopped = False
            daemons.append(self)

        def start(self):
            pass

        def stop(self):
            self.stopped = True

    checkpoint = tmp_path / "sam3.pt"
    checkpoint.touch()
    output = tmp_path / "services"
    monkeypatch.setattr(services, "ProcessDaemon", Daemon)
    monkeypatch.setattr(
        services, "wait_for_endpoint", lambda daemon, **_: f"http://{daemon.name}"
    )
    monkeypatch.setattr(services.signal, "signal", lambda *_: None)

    def end_run(_):
        raise SystemExit(0)

    monkeypatch.setattr(services.time, "sleep", end_run)
    monkeypatch.setattr(services, "wait_for_ready", lambda *_, **__: None)
    monkeypatch.setattr(Daemon, "poll", lambda self: None, raising=False)
    config = {
        "nodes": {
            "test": {
                "output_dir": str(output),
                "sam3": {
                    "python": sys.executable,
                    "checkpoint": str(checkpoint),
                    "port": 28424,
                    "replicas": {1: sam3_per_gpu, 3: sam3_per_gpu},
                    "max_pending": 512,
                },
                "backends": {
                    "libero": {
                        "python": sys.executable,
                        "port": 28490,
                        "gpu_slots": {1: 60, 3: 60},
                        "sam3_endpoint": "http://127.0.0.1:28424",
                    }
                },
            }
        }
    }
    with pytest.raises(SystemExit, match="0"):
        services.run_node(config, "test")
    pool = next(d for d in daemons if d.name == "sam3-pool")
    assert pool.cmd[pool.cmd.index("--max-pending") + 1] == "512"
    sam = [d for d in daemons if d.name.startswith("sam3-gpu")]
    assert [d.cmd[d.cmd.index("--cuda-device") + 1] for d in sam] == [
        str(gpu) for gpu in (1, 3) for _ in range(sam3_per_gpu)
    ]
    backend = next(d for d in daemons if d.name == "libero-toolkit")
    assert backend.cmd[backend.cmd.index("--gpu-slots") + 1] == "1:60,3:60"
    assert backend.cmd[backend.cmd.index("--max-initializing-per-gpu") + 1] == "2"
    assert all(d.cmd[0] == sys.executable and d.stopped for d in daemons)


def test_example_deployments_compile_native_slots_and_router_capacity():
    import yaml

    from robots.rl.services import local_services

    config = yaml.safe_load((SCRIPTS / "deployment-384.yaml").read_text())
    for node in config["nodes"]:
        assert local_services(config, node)
    routers = {
        name: args
        for name, _, module, args, _ in local_services(config, "model0")
        if module == "robots.rl.router"
    }
    assert set(routers) == {"libero-router", "robotwin-router"}
    for args in routers.values():
        assert (
            sum(
                int(args[i + 1].rsplit("=", 1)[1])
                for i, value in enumerate(args)
                if value == "--backend"
            )
            == 192
        )
    assert sum(config["nodes"]["sim"]["sam3"]["replicas"].values()) == 16

    config = yaml.safe_load((SCRIPTS / "deployment-local.yaml").read_text())
    assert {name for name, *_ in local_services(config, "local")} == {
        "libero-toolkit",
        "robotwin-toolkit",
    }


def test_rollout_cli_and_batch_share_environment_defaults(monkeypatch):
    import argparse

    from robots.rl.rollout_options import add_options, episode_options, resolve_options

    parser = argparse.ArgumentParser()
    add_options(parser)
    monkeypatch.setenv("RPENT_MODEL", "openai-chat:qwen3.5-4b")
    monkeypatch.setenv("RPENT_MAX_TURNS", "40")
    options = parser.parse_args(["--max-turns", "32", "--no-thinking"])
    resolve_options(options)
    assert options.max_turns == 32
    assert options.reasoning_effort == "none"
    assert episode_options(options)[-1] == "--no-thinking"
    options = parser.parse_args([])
    resolve_options(options)
    assert options.max_turns == 40
