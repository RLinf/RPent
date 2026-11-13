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

"""Opt in with RPENT_EFE_TASK_PATH and RPENT_EFE_MODEL to run one Docker task.

This policy-chain test uses the configured model. The loopback test
``tests/unit_tests/robots/efe/test_runner_loopback.py`` does not.
Harbor's verifier owns the reward. A zero reward does not fail this test.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import pytest

# A task may declare ``[verifier.env]`` entries that read host variables. Harbor
# asks for confirmation before forwarding them, so a non-interactive run must
# accept on the command line or it hangs under ``pytest -s`` and raises
# ``EOFError`` under captured stdin.
_HOST_ENV_CONFIRM_FLAG = "--yes"


@pytest.fixture
def harbor_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    task_value = os.environ.get("RPENT_EFE_TASK_PATH")
    if not task_value:
        pytest.skip("Set RPENT_EFE_TASK_PATH to opt in to the Docker/Harbor task.")
    model = os.environ.get("RPENT_EFE_MODEL")
    if not model:
        pytest.fail("Set RPENT_EFE_MODEL for the opted-in Harbor task.")
    task_path = Path(task_value).expanduser().resolve(strict=True)
    if not (task_path / "task.toml").is_file():
        pytest.fail(f"Select one Harbor task directory with task.toml: {task_path}")
    harbor_cli = shutil.which("harbor")
    docker_cli = shutil.which("docker")
    if not harbor_cli or not docker_cli:
        pytest.fail("The opted-in task requires Harbor CLI and Docker on PATH.")
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(filter(None, (str(repo_root), os.environ.get("PYTHONPATH")))),
    )
    importlib.import_module("robots.efe.harbor_agent").RPentEfeAgent
    subprocess.run(
        [docker_cli, "info"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    command = [
        harbor_cli,
        "run",
        "--path",
        str(task_path),
        "-a",
        "robots.efe.harbor_agent:RPentEfeAgent",
        "--model",
        model,
        "-e",
        "docker",
        "--n-attempts",
        "1",
        "--n-concurrent",
        "1",
        "--jobs-dir",
        str(tmp_path / "jobs"),
        "--job-name",
        "rpent-efe-smoke",
        _HOST_ENV_CONFIRM_FLAG,
    ]
    # EFE tasks run the agent under ``network_mode = "allowlist"``, and this
    # agent declares no Harbor ``model_connection``. A custom endpoint is only
    # reachable when its host is added explicitly.
    agent_host = _custom_agent_host()
    if agent_host is not None:
        command.extend(["--allow-agent-host", agent_host])
    return command


def _custom_agent_host() -> str | None:
    """Return the host a custom Codex endpoint needs allowlisted, if any."""
    for name in ("CODEX_BASE_URL", "OPENAI_BASE_URL"):
        value = os.environ.get(name)
        if not value:
            continue
        host = urlsplit(value).hostname
        if host:
            return host
    return None


@pytest.mark.timeout(14400)
def test_harbor_task_records_agent_session_and_verifier_result(
    tmp_path: Path, harbor_command: list[str]
) -> None:
    job_dir = tmp_path / "jobs" / "rpent-efe-smoke"

    completed = subprocess.run(
        harbor_command,
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=14300,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (job_dir / "result.json").is_file(), completed.stdout + completed.stderr
    assert len(list(job_dir.glob("*/result.json"))) == 1
    assert (
        json.loads(next(job_dir.glob("*/result.json")).read_text())["exception_info"]
        is None
    )
    assert isinstance(
        json.loads(next(job_dir.glob("*/result.json")).read_text())["verifier_result"][
            "rewards"
        ],
        dict,
    )
    assert len(list(job_dir.glob("*/agent/**/session.json"))) == 1
    assert (
        json.loads(next(job_dir.glob("*/agent/**/session.json")).read_text())["error"]
        is None
    )
    assert "reward" not in json.loads(
        next(job_dir.glob("*/agent/**/session.json")).read_text()
    )
    assert (
        next(job_dir.glob("*/agent/**/session.json")).parent / "codex.txt.stream.jsonl"
    ).is_file()
