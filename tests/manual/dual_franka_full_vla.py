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

"""Manual-only continuous VLA rollout; never registered with the agent toolkit."""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import yaml

from robots.dual_franka.env_client import DualFrankaEnvClient
from robots.dual_franka.runtime_config import DEFAULT_CONFIG
from robots.dual_franka.tasks import get_dual_franka_task
from rpent.robots.components.pi05_vla_client import Pi05VLAClient
from rpent.utils.rpc import make_rpc_client
from tests.e2e_tests.dual_franka.dual_franka_vla import (
    validate_actions,
    validate_observation,
)


def run_full_vla(env, model, *, prompt, workspace, expected_steps, stopped, emit):
    """Run indefinitely, checking cancellation before each single-action RPC.

    An in-flight inference/RPC is allowed to return; its remaining actions are
    discarded on cancellation. No reset, recovery or gripper command is added.
    """
    if not prompt.strip():
        raise ValueError("policy instruction must not be empty")
    chunks = steps = 0
    started = time.monotonic()
    summary = {"status": "stopped", "reason": "operator_stop"}
    try:
        while not stopped():
            observation = dict(env.get_observation())
            validate_observation(observation)
            if stopped():
                break
            observation["task_descriptions"] = prompt
            prediction = model.predict(observation, options={"mode": "eval"})
            if stopped():
                break
            actions = validate_actions(prediction, workspace, expected_steps)
            chunks += 1
            emit({"kind": "prediction", "chunk": chunks, "actions": actions.tolist()})
            for index, action in enumerate(actions):
                if stopped():
                    return summary
                # Do not queue a whole prediction chunk on the robot server.
                result = env.chunk_step(action[None, :])
                steps += 1
                emit(
                    {
                        "kind": "action",
                        "chunk": chunks,
                        "index": index,
                        "steps": steps,
                        "terminated": bool(result.get("terminated")),
                        "truncated": bool(result.get("truncated")),
                    }
                )
                if result.get("error") or result.get("ok") is False:
                    raise RuntimeError(f"environment action failed: {result}")
                if result.get("terminated") or result.get("truncated"):
                    summary.update(
                        status="environment_stopped", reason="terminated_or_truncated"
                    )
                    return summary
            emit(
                {
                    "kind": "chunk_complete",
                    "chunks": chunks,
                    "steps": steps,
                    "elapsed_s": time.monotonic() - started,
                }
            )
        return summary
    except BaseException as exc:
        summary.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary.update(chunks=chunks, steps=steps, elapsed_s=time.monotonic() - started)
        emit({"kind": "finish", **summary})


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skill", choices=["vla_full_rollout"], default="vla_full_rollout"
    )
    parser.add_argument("--env-endpoint", default="http://127.0.0.1:6002")
    parser.add_argument("--vla-endpoint", default="http://127.0.0.1:6000")
    parser.add_argument("--task-id", type=int, default=6)
    parser.add_argument(
        "--instruction", help="Override task-configured policy instruction"
    )
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stop-file", type=Path)
    return parser


def main():
    args = build_parser().parse_args()
    output = args.output_dir or Path("logs") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-manual-full-vla"
    )
    output = output.resolve()
    # Never overwrite an earlier experiment's logs or remove its stop signal.
    output.mkdir(parents=True, exist_ok=False)
    stop_file = (args.stop_file or output / "STOP").resolve()
    if stop_file.exists():
        raise RuntimeError(f"stop file already exists: {stop_file}; use a new path")
    prompt = args.instruction or get_dual_franka_task(args.task_id).vla_instruction
    if not prompt:
        raise ValueError("selected task has no VLA instruction")
    config = yaml.safe_load(args.robot_config.read_text())
    event = threading.Event()
    signals = []

    def request_stop(signum, _frame):
        signals.append(signal.Signals(signum).name)
        event.set()

    previous = {
        s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)
    }
    print(f"Logs: {output}\nStop file: {stop_file}", flush=True)
    print(
        "No reset. No duration/chunk/semantic-boundary limit. Ctrl+C or create STOP to stop.",
        flush=True,
    )
    print(
        "An in-flight RPC may finish; this is NOT a hardware emergency stop.",
        flush=True,
    )
    try:
        with ExitStack() as stack:
            env_rpc = make_rpc_client(args.env_endpoint)
            stack.callback(env_rpc.close)
            model_rpc = make_rpc_client(args.vla_endpoint)
            stack.callback(model_rpc.close)
            env = DualFrankaEnvClient(env_rpc, reset_on_connect=False)
            model = Pi05VLAClient(model_rpc, embodiment="dual_franka")
            status = model.status(timeout_s=10)
            (output / "deployment.json").write_text(
                json.dumps(
                    {
                        "skill": args.skill,
                        "task_id": args.task_id,
                        "prompt": prompt,
                        "model": status,
                        "robot_config": str(args.robot_config.resolve()),
                        "stop_file": str(stop_file),
                        "reset_on_connect": False,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            with (output / "events.jsonl").open("x", buffering=1) as log:

                def emit(record):
                    log.write(
                        json.dumps(
                            {"timestamp": time.time(), **record}, ensure_ascii=False
                        )
                        + "\n"
                    )
                    if record["kind"] == "finish":
                        (output / "outcome.json").write_text(
                            json.dumps({**record, "signals": signals}, indent=2)
                        )
                    if record["kind"] in {"chunk_complete", "finish"}:
                        print(json.dumps(record, ensure_ascii=False), flush=True)

                result = run_full_vla(
                    env,
                    model,
                    prompt=prompt,
                    workspace=config["workspace"],
                    expected_steps=int(status["config"]["openpi"]["action_chunk"]),
                    stopped=lambda: event.is_set() or stop_file.exists(),
                    emit=emit,
                )
                return 0 if result["status"] == "stopped" else 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
