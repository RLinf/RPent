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

"""Shared assertions and lifecycle helpers for checkpoint-backed GPU tests."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from argparse import ArgumentParser, Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from rpent.dashboard.events import NullDashboardEventSink
from rpent.robots.robot_spec import RobotSpec
from tests.e2e_tests.offline_planner_server import (
    OFFLINE_MODEL_NAME,
    OfflinePlannerServer,
    ScriptedToolCall,
)


def run_scripted_policy_chain(
    *,
    robot: str,
    robot_argv: list[str],
    output_dir: Path,
    action: ScriptedToolCall,
    action_count_field: str,
) -> dict[str, Any]:
    """Run the public CLI against a deterministic loopback planner."""
    memory_dir = output_dir.parent / "offline-memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "MEMORY.md").write_text(
        "# Offline GPU E2E memory\n",
        encoding="utf-8",
    )
    script = (
        action,
        ScriptedToolCall(
            "finish",
            {
                "status": "stuck",
                "summary": "bounded GPU E2E completed one VLA action call",
            },
        ),
    )

    executable = Path(sys.executable).with_name("rpent")
    if not executable.is_file():
        raise RuntimeError(f"RPent console script not found: {executable}")
    command = [
        str(executable),
        "--robot",
        robot,
        "--planner",
        "api",
        "--model",
        f"openai-chat:{OFFLINE_MODEL_NAME}",
        "--max-turns",
        str(len(script)),
        "--no-images",
        "--memory-profile",
        "local",
        "--memory-dir",
        str(memory_dir),
        "--output-dir",
        str(output_dir),
        *robot_argv,
    ]
    env = {**os.environ, "OPENAI_API_KEY": OFFLINE_MODEL_NAME}

    repo_root = Path(__file__).resolve().parents[2]
    with OfflinePlannerServer(script) as planner_server:
        completed = subprocess.run(
            [*command, "--base-url", planner_server.base_url],
            cwd=repo_root,
            env=env,
            check=False,
        )
    planner_server.assert_complete()
    if completed.returncode != 0:
        raise RuntimeError(
            f"scripted {robot} session failed with exit code {completed.returncode}"
        )

    transcript_paths = list(output_dir.glob("transcript_*.json"))
    if len(transcript_paths) != 1:
        raise RuntimeError(
            f"expected one planner transcript, found {len(transcript_paths)}"
        )
    transcript = json.loads(transcript_paths[0].read_text(encoding="utf-8"))
    finish = transcript.get("finish")
    if not isinstance(finish, dict) or finish.get("_finish") is not True:
        raise RuntimeError(f"scripted {robot} session did not call finish")

    state_path = output_dir / "states.json"
    state_manifest = json.loads(state_path.read_text(encoding="utf-8"))
    action_steps = [
        step
        for step in state_manifest.get("steps", [])
        if (step.get("command") or {}).get("action") == action.name
    ]
    if not action_steps:
        raise RuntimeError(f"scripted {robot} session did not record {action.name}")
    if any((step.get("result") or {}).get("error") for step in action_steps):
        raise RuntimeError(f"scripted {robot} action recorded an error")
    action_count = sum(
        int((step.get("result") or {}).get(action_count_field, 0))
        for step in action_steps
    )
    if action_count <= 0:
        raise RuntimeError(
            f"scripted {robot} action executed no simulator steps "
            f"({action_count_field}={action_count})"
        )

    return {
        "status": "passed",
        "definition": (
            "offline planner -> public CLI -> toolkit -> checkpoint-backed VLA "
            "-> simulator action -> finish"
        ),
        "planner": "offline-scripted",
        "action_tool": action.name,
        "recorded_action_steps": len(action_steps),
        "simulator_action_count": action_count,
        "simulator_action_count_field": action_count_field,
        "finish_status": finish.get("status"),
        "transcript": transcript_paths[0].name,
        "state_manifest": state_path.name,
    }


def selected_cuda_ordinal() -> int:
    """Return the one physical CUDA ordinal selected by the runner."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not value.isdecimal():
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be one physical integer ordinal so the "
            "MuJoCo EGL device can be selected"
        )
    return int(value)


def parse_runtime_args(spec: RobotSpec, argv: list[str]) -> Namespace:
    """Parse robot arguments through the current public CLI definition."""
    parser = ArgumentParser(add_help=False)
    spec.add_cli_args(parser, False)
    return parser.parse_args(argv)


def installation_info(extra: str) -> dict[str, Any]:
    """Record enough installation identity to diagnose a clean-extra run."""
    return {
        "extra": extra,
        "python": platform.python_version(),
        "executable": sys.executable,
        "rpent": version("rpent"),
    }


@contextmanager
def runtime_phase(
    spec: RobotSpec,
    args: Any,
    output_dir: Path,
    components: set[str],
) -> Iterator[dict[str, Any]]:
    """Start selected production runtime components and always stop owned daemons."""
    output_dir.mkdir(parents=True, exist_ok=True)
    daemons = []
    kwargs: dict[str, Any] = {}
    try:
        daemons, kwargs = spec.init_runtime(
            args,
            output_dir,
            NullDashboardEventSink(),
            components,
        )
        yield kwargs
    finally:
        cleanup_errors = []
        for daemon in reversed(daemons):
            try:
                daemon.stop()
                deadline = time.monotonic() + 5.0
                while daemon.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                if daemon.poll() is None:
                    cleanup_errors.append(f"{daemon.name} is still running")
            except Exception as exc:  # pragma: no cover - only reached on GPU runners
                cleanup_errors.append(f"{daemon.name}: {type(exc).__name__}: {exc}")
        if cleanup_errors:
            raise RuntimeError("runtime cleanup failed: " + "; ".join(cleanup_errors))


def require_array(
    value: Any,
    name: str,
    *,
    ndim: int | None = None,
    last_dim: int | None = None,
    dtype: Any | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise RuntimeError(f"{name} must have {ndim} dimensions; got {array.shape}")
    if last_dim is not None and (array.ndim == 0 or array.shape[-1] != last_dim):
        raise RuntimeError(
            f"{name} must end in dimension {last_dim}; got {array.shape}"
        )
    if dtype is not None and array.dtype != np.dtype(dtype):
        raise RuntimeError(
            f"{name} must have dtype {np.dtype(dtype)}; got {array.dtype}"
        )
    if array.size == 0:
        raise RuntimeError(f"{name} must not be empty")
    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
        raise RuntimeError(f"{name} contains non-finite values")
    return array


def require_rgb(value: Any, name: str) -> np.ndarray:
    return require_array(value, name, ndim=3, last_dim=3, dtype=np.uint8)


def require_depth(value: Any, name: str, rgb: np.ndarray) -> np.ndarray:
    depth = require_array(value, name, ndim=2)
    if depth.shape != rgb.shape[:2]:
        raise RuntimeError(
            f"{name} shape {depth.shape} does not match RGB shape {rgb.shape[:2]}"
        )
    return depth


def require_camera_meta(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"{name} must be a non-empty mapping")
    return value


def array_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def synthetic_rgb() -> np.ndarray:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[64:192, 80:176] = np.asarray([180, 100, 40], dtype=np.uint8)
    return image


def prepare_suite(
    *, output_dir: Path, stack: str, extra: str, details: dict[str, Any]
) -> Path:
    """Create one suite output directory and record its reproducible inputs."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "stack": stack,
        "installation": installation_info(extra),
        **details,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    return output_dir


def publish_check(name: str, result: dict[str, Any]) -> None:
    """Emit a compact structured result into the pytest and Actions logs."""
    print(json.dumps({"check": name, **result}, indent=2, default=str), flush=True)
