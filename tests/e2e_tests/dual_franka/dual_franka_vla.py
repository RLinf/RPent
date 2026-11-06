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

"""Operator-triggered VLA deployment diagnostics over the existing RPent RPCs."""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from robots.dual_franka.robot_spec import _add_runtime_cli_args, _init_runtime
from robots.dual_franka.runtime_config import DEFAULT_CONFIG
from robots.franka.runtime_config import (
    _require_mapping,
    load_mapping,
    set_robot_config_path,
)
from rpent.dashboard.events import NullDashboardEventSink
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("dual_franka.vla_test")


def save_record(path: Path, value: Any) -> None:
    """Store arrays losslessly beside JSON metadata, without pickle."""
    arrays = {}

    def encode(item, key="root"):
        if isinstance(item, np.ndarray):
            arrays[key] = item
            return {"array": key, "shape": list(item.shape), "dtype": str(item.dtype)}
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {str(k): encode(v, key + "/" + str(k)) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [encode(v, key + "/" + str(i)) for i, v in enumerate(item)]
        return item

    metadata = encode(value)
    np.savez_compressed(str(path) + ".npz", **arrays)
    Path(str(path) + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2)
    )


def validate_observation(obs: dict[str, Any]) -> None:
    """Reject malformed policy input before inference."""
    state = np.asarray(obs["states"])
    if state.shape != (20,) or not np.isfinite(state).all():
        raise ValueError(f"Expected finite 20-D state, got {state.shape}")
    main, extras = np.asarray(obs["main_images"]), np.asarray(obs["extra_view_images"])
    if main.ndim != 3 or main.shape[-1] != 3:
        raise ValueError(f"Expected left wrist HWC RGB, got {main.shape}")
    if extras.ndim != 4 or extras.shape[0] != 2 or extras.shape[-1] != 3:
        raise ValueError(f"Expected [base, right wrist] RGB, got {extras.shape}")
    if main.dtype != np.uint8 or extras.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB images")


def validate_actions(
    actions: Any, workspace: dict[str, Any], expected_steps: int
) -> np.ndarray:
    """Validate action shape, finiteness, rotations and workspace limits."""
    actions = np.asarray(actions)
    if actions.shape != (expected_steps, 20) or not np.isfinite(actions).all():
        raise ValueError(
            f"Expected finite actions [{expected_steps},20], got {actions.shape}"
        )
    for arm, offset in enumerate((0, 10)):
        xyz = actions[:, offset : offset + 3]
        low = np.asarray(workspace["ee_pose_limit_min"][arm][:3])
        high = np.asarray(workspace["ee_pose_limit_max"][arm][:3])
        if np.any(xyz < low) or np.any(xyz > high):
            raise ValueError(
                f"Arm {arm} predicted TCP outside configured XYZ workspace; rejected, not clipped"
            )
        rot = actions[:, offset + 3 : offset + 9]
        if np.any(np.linalg.norm(np.cross(rot[:, :3], rot[:, 3:]), axis=1) < 1e-6):
            raise ValueError(f"Arm {arm} predicted degenerate rot6d")
    return actions.astype(np.float32)


class DeploymentTest:
    """Record and execute policy chunks without task-specific skill boundaries."""

    def __init__(
        self,
        env: Any,
        model: Any,
        output: Path,
        workspace: dict[str, Any],
        prompt: str,
        expected_steps: int,
        stopped: Callable[[], bool] = lambda: False,
    ) -> None:
        self.env, self.model = env, model
        self.output, self.workspace = Path(output), workspace
        self.prompt = prompt
        self.index = 0
        self.execution_uncertain = False
        self.episode_done = False
        self.stopped = stopped
        if expected_steps <= 0:
            raise ValueError("expected action steps must be positive")
        self.expected_steps = expected_steps

    def chunk(self) -> dict[str, Any]:
        """Record one prediction, then execute it with stop checks between actions."""
        if self.execution_uncertain:
            raise RuntimeError("Execution state uncertain; check the robot and restart")
        if self.episode_done:
            raise RuntimeError("Episode ended; reset before running again")
        if self.stopped():
            raise KeyboardInterrupt
        self.index += 1
        path = self.output / f"vla_{self.index:06d}"
        record = {"prompt": self.prompt, "timestamp": time.time(), "steps_executed": 0}
        try:
            obs = dict(self.env.get_observation())
            obs["task_descriptions"] = self.prompt
            record["input"] = obs
            validate_observation(obs)
            start = time.monotonic()
            actions = self.model.predict(obs, options={"mode": "eval"})
            record["inference_seconds"] = time.monotonic() - start
            record["actions"] = np.asarray(actions)
            actions = validate_actions(actions, self.workspace, self.expected_steps)
            save_record(path, record)
            if self.stopped():
                raise KeyboardInterrupt
            try:
                for action in actions:
                    if self.stopped():
                        raise KeyboardInterrupt
                    result = self.env.chunk_step(action[None, :])
                    record["steps_executed"] += 1
                    record["result"] = result
                    if result.get("error") or result.get("ok") is False:
                        raise RuntimeError(f"Environment action failed: {result}")
                    self.episode_done = bool(
                        result.get("terminated") or result.get("truncated")
                    )
                    if self.episode_done:
                        break
                record["robot_state_after"] = self.env.get_robot_state()
            except BaseException:
                self.execution_uncertain = True
                raise
            record["ok"] = True
            return {
                "ok": True,
                "actions": record["steps_executed"],
                "terminated": bool(record["result"].get("terminated")),
                "truncated": bool(record["result"].get("truncated")),
                "record": str(path),
            }
        except BaseException as exc:
            record["ok"], record["error"] = False, f"{type(exc).__name__}: {exc}"
            raise
        finally:
            save_record(path, record)

    def reset(self) -> dict[str, Any]:
        """Ask the operator to restore the scene before resetting the same Env."""
        from robots.dual_franka.dual_franka_manual_call import _call_toolkit_tool

        if self.execution_uncertain:
            raise RuntimeError("Execution state uncertain; check the robot and restart")
        self.episode_done = True
        result = _call_toolkit_tool(
            "request_scene_reset",
            {"reason": "Prepare for another VLA rollout"},
            env=self.env,
            model=self.model,
            sam3_client=None,
            output_dir=self.output / f"reset_{time.time_ns()}",
            instruction=self.prompt,
        )
        if (
            result.get("error")
            or result.get("state_capture_error")
            or result.get("result", {}).get("ok") is not True
        ):
            raise RuntimeError(f"Reset not completed: {result}")
        self.episode_done = False
        return result


def run_console(
    session: DeploymentTest,
    read: Callable[[str], str] = input,
    emit: Callable[[str], None] = print,
) -> None:
    """Run finite or continuous rollouts on explicit operator commands."""
    help_text = "run N | run (continuous) | reset | quit"
    emit(help_text)
    emit("Ctrl-C or the STOP file exits; neither is a hardware emergency stop.")
    while not session.stopped():
        try:
            line = read("vla> ").strip()
            if not line:
                continue
            command, _, arg = line.partition(" ")
            if command == "quit" and not arg:
                break
            if command == "reset" and not arg:
                emit(json.dumps(session.reset(), ensure_ascii=False))
            elif command == "run":
                remaining = int(arg) if arg.strip() else None
                if remaining is not None and remaining <= 0:
                    raise ValueError("run N requires a positive integer")
                while remaining is None or remaining > 0:
                    result = session.chunk()
                    emit(json.dumps(result, ensure_ascii=False))
                    if result["terminated"] or result["truncated"]:
                        break
                    if remaining is not None:
                        remaining -= 1
            else:
                raise ValueError(help_text)
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as exc:
            emit(f"Failed: {exc}; no automatic retry")


def run_session(args: argparse.Namespace) -> int:
    """Own Env/VLA services and operator-controlled rollout without an initial reset."""
    if args.expected_action_steps is not None and args.expected_action_steps <= 0:
        raise ValueError("expected action steps must be positive")
    args.robot_config = args.robot_config or str(DEFAULT_CONFIG)
    data = load_mapping(args.robot_config)
    instruction = args.instruction or _require_mapping(data.get("vla", {}), "vla").get(
        "instruction"
    )
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Set --instruction or vla.instruction in the robot config")
    set_robot_config_path(args.robot_config)
    output = Path(
        args.output_dir
        or Path("logs") / datetime.now().strftime("%Y%m%d-%H%M%S-vla-test")
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    init_output_dir(output)
    stop_file = Path(args.stop_file or output / "STOP").resolve()
    if stop_file.exists():
        raise ValueError(f"stop file already exists: {stop_file}")
    stopped = threading.Event()

    def request_stop(signum, frame):
        stopped.set()
        raise KeyboardInterrupt

    with ExitStack() as stack:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous = signal.signal(signum, request_stop)
            stack.callback(signal.signal, signum, previous)
        daemons, runtime = _init_runtime(
            args,
            output,
            NullDashboardEventSink(),
            {"env", "vla"},
            task_description="",
            vla_instruction=instruction,
        )
        for daemon in daemons:
            stack.callback(daemon.stop)
        model = runtime["model"]
        expected_steps = args.expected_action_steps
        if expected_steps is None:
            expected_steps = int(model.status(timeout_s=10)["action_chunk"])
        session = DeploymentTest(
            runtime["env"],
            model,
            output,
            data["workspace"],
            instruction,
            expected_steps,
            stopped=lambda: stopped.is_set() or stop_file.exists(),
        )
        save_record(
            output / "deployment",
            {
                "instruction": instruction,
                "expected_action_steps": expected_steps,
                "robot_config": data,
                "camera_meta": runtime["env"].get_camera_meta(),
            },
        )
        logger.info("No initial reset. Logs: %s; stop file: %s", output, stop_file)
        run_console(session)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the task-independent VLA test command."""
    parser = argparse.ArgumentParser(description=__doc__)
    _add_runtime_cli_args(parser)
    parser.add_argument(
        "--instruction", help="Override vla.instruction from the robot config"
    )
    parser.add_argument(
        "--expected-action-steps",
        type=int,
        default=None,
        help="Override the prediction chunk length reported by vla.status",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stop-file", type=Path)
    return parser


def main() -> int:
    """Run the diagnostic command-line entry point."""
    return run_session(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
