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
    "simulator,workflow,task_args",
    [
        ("libero", "episode", ["--suite", "libero_object", "--task", "0"]),
        ("libero", "zap_turn", ["--suite", "libero_object", "--task", "0"]),
        ("robocasa", "episode", ["--task-name", "PickPlaceCounterToCabinet"]),
        ("robotwin", "episode", ["--task-name", "click_alarmclock"]),
    ],
)
def test_single_episode_launcher_preserves_the_selected_simulator_and_workflow(
    tmp_path, simulator, workflow, task_args
):
    executable = tmp_path / "rpent"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'libero_type': os.environ['LIBERO_TYPE']}))\n"
    )
    executable.chmod(0o755)
    (tmp_path / "python").symlink_to(executable)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPTS / "run.sh"),
            "--simulator",
            simulator,
            "--workflow",
            workflow,
            *task_args,
        ],
        env={
            **_environment(),
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "RPENT_MODEL": "openai-chat:qwen3.8-27b",
            "RPENT_BASE_URL": "http://localhost:30080/v1",
            "LIBERO_TYPE": "standard",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    request = json.loads(result.stdout)
    argv = request["argv"]
    assert argv[argv.index("--robot") + 1] == "rl"
    assert argv[argv.index("--simulator") + 1] == simulator
    assert argv[argv.index("--workflow") + 1] == workflow
    assert "--libero-type" not in argv
    assert request["libero_type"] == "standard"
