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
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robots.dual_franka.robot_spec import get_robot_spec
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
    """Record explicitly requested inference and execution chunks."""

    def __init__(
        self,
        env: Any,
        model: Any,
        output: Path,
        workspace: dict[str, Any],
        prompt: str,
        expected_steps: int,
    ) -> None:
        self.env, self.model = env, model
        self.output, self.workspace = Path(output), workspace
        self.prompt = prompt
        self.index = 0
        self.execution_uncertain = False
        self.episode_done = False
        if expected_steps <= 0:
            raise ValueError("expected action steps must be positive")
        self.expected_steps = expected_steps

    def chunk(self, execute: bool = False) -> dict[str, Any]:
        """Infer from fresh observations and optionally execute the prediction."""
        if self.execution_uncertain:
            raise RuntimeError(
                "Previous execution failed; restart the session after checking robot state"
            )
        if execute and self.episode_done:
            raise RuntimeError("Episode ended; reset before executing another chunk")
        if not self.prompt.strip():
            raise ValueError("Set a non-empty prompt first")
        self.index += 1
        path = self.output / f"vla_{self.index:06d}"
        record = {"prompt": self.prompt, "execute": execute, "timestamp": time.time()}
        try:
            start = time.monotonic()
            obs = dict(self.env.get_observation())
            obs["task_descriptions"] = self.prompt
            record["input"] = obs
            record["observation_seconds"] = time.monotonic() - start
            validate_observation(obs)
            start = time.monotonic()
            actions = self.model.predict(obs, options={"mode": "eval"})
            record["inference_seconds"] = time.monotonic() - start
            record["actions"] = np.asarray(actions)
            actions = validate_actions(actions, self.workspace, self.expected_steps)
            # Persist the input and prediction before any physical execution.
            save_record(path, record)
            if execute:
                start = time.monotonic()
                try:
                    result = self.env.chunk_step(actions)
                    record["execution_seconds"] = time.monotonic() - start
                    record["result"] = result
                    self.episode_done = bool(
                        result.get("terminated") or result.get("truncated")
                    )
                    record["robot_state_after"] = self.env.get_robot_state()
                    if result.get("ok") is False:
                        raise RuntimeError(f"Environment reported failure: {result}")
                except BaseException:
                    self.execution_uncertain = True
                    raise
            record["ok"] = True
            return {
                "ok": True,
                "executed": execute,
                "actions": len(actions),
                "inference_seconds": record["inference_seconds"],
                "execution_seconds": record.get("execution_seconds", 0),
                "terminated": record.get("result", {}).get("terminated", False),
                "truncated": record.get("result", {}).get("truncated", False),
                "record": str(path),
            }
        except BaseException as exc:
            record["ok"], record["error"] = False, str(exc)
            raise
        finally:
            save_record(path, record)


def run_console(
    session: DeploymentTest,
    read: Callable[[str], str] = input,
    emit: Callable[[str], None] = print,
) -> None:
    """Read explicit commands without invoking an Agent."""
    help_text = (
        "prompt <instruction> | infer | step | run N (1..20 chunks) | reset | quit"
    )
    emit(help_text)
    emit("infer 不下发预测动作；step 执行一个动作块。Ctrl-C 退出会话，不等同硬件急停。")
    while True:
        try:
            line = read("vla> ").strip()
            if not line:
                continue
            command, _, arg = line.partition(" ")
            if command in ("quit", "exit") and not arg:
                break
            if command == "prompt":
                if not arg.strip():
                    raise ValueError("prompt cannot be empty")
                session.prompt = arg.strip()
                emit("Prompt: " + session.prompt)
            elif command == "help" and not arg:
                emit(help_text)
            elif command == "reset" and not arg:
                if session.execution_uncertain:
                    raise RuntimeError(
                        "Execution state uncertain; restart session first"
                    )
                try:
                    result = session.env.reset()
                    session.episode_done = False
                    save_record(session.output / f"reset_{time.time_ns()}", result)
                    emit("Reset complete")
                except BaseException:
                    session.execution_uncertain = True
                    raise
            elif command in ("infer", "step", "run"):
                count = int(arg) if command == "run" else 1
                if (command != "run" and arg) or not 1 <= count <= 20:
                    raise ValueError("Usage: infer | step | run N (1..20)")
                for _ in range(count):
                    result = session.chunk(execute=command != "infer")
                    emit(json.dumps(result, ensure_ascii=False))
                    if result["terminated"] or result["truncated"]:
                        break
            else:
                raise ValueError(help_text)
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as exc:
            emit(f"Failed: {exc}; no automatic retry")


def run_session(args: argparse.Namespace) -> int:
    """Own the diagnostic runtime and close it when the console exits."""
    if args.expected_action_steps <= 0:
        raise ValueError("expected action steps must be positive")
    spec = get_robot_spec()
    from robots.dual_franka.runtime_config import DEFAULT_CONFIG
    from robots.dual_franka.tasks import get_dual_franka_task

    args.robot_config = args.robot_config or str(DEFAULT_CONFIG)
    data = yaml.safe_load(Path(args.robot_config).read_text())
    args.instruction = (
        args.instruction or get_dual_franka_task(args.task_id).vla_instruction
    )
    if not args.instruction:
        raise ValueError("Select a VLA task profile or supply --instruction")
    if get_dual_franka_task(args.task_id).vla_instruction is None and not getattr(
        args, "vla_endpoint", None
    ):
        raise ValueError("Select a VLA task profile or supply --vla-endpoint")
    config = spec.parse_config(args)
    output = init_output_dir(config.output_dir)
    daemons = []
    logger.info(
        "Initializing VLA test: environment may reset robots; predictions execute only on step/run.",
    )
    try:
        daemons, kwargs = spec.init_runtime(
            args, output, NullDashboardEventSink(), {"env", "vla"}
        )
        model = kwargs["model"]
        save_record(
            output / "deployment",
            {
                "expected_action_steps": args.expected_action_steps,
                "robot_config": data,
                "camera_meta": kwargs["env"].get_camera_meta(),
            },
        )
        session = DeploymentTest(
            kwargs["env"],
            model,
            output,
            data["workspace"],
            args.instruction,
            args.expected_action_steps,
        )
        logger.info("Records: %s", output)
        run_console(session)
    finally:
        for daemon in reversed(daemons):
            daemon.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build a standalone diagnostic parser with the complete config contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    get_robot_spec().add_cli_args(parser, use_dashboard=False)
    parser.set_defaults(task_id=1, explore=False, memory_dir=None, memory_profile=None)
    parser.add_argument(
        "--instruction",
        default=None,
        help="Policy instruction override; defaults to the selected task VLA instruction",
    )
    parser.add_argument(
        "--expected-action-steps",
        type=int,
        default=20,
        help="Expected prediction chunk length for action validation (default: 20)",
    )
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> int:
    """Run the diagnostic command-line entry point."""
    return run_session(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
