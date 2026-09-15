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

"""Generate a visual Franka card from states.json and step intent annotations.

No robot connection is made. Annotations may be authored by the recording agent
from its transcript, then reviewed by the operator. Molmo only supplies points;
it does not infer the purpose of a motion or certify task success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robots.franka.task_card.common import (
    ACTIONS,
    RecordedState,
    checked_result,
    fingerprint,
    localize,
    tcp_pose,
    validate_card,
)


def generate(
    run_dir: str | Path,
    annotations: dict,
    *,
    robot: str,
    task: str,
    molmo: Any,
    human_verdict: str,
) -> dict:
    """Ground annotated source moves and retain the recorded primitive order."""
    if human_verdict != "success":
        raise ValueError("only human-confirmed success may generate a task card")
    state = RecordedState(run_dir)
    outcome_path = Path(run_dir) / "task_card_outcome.json"
    if (
        outcome_path.exists()
        and json.loads(outcome_path.read_text()).get("done") is not True
    ):
        raise ValueError("source replay did not succeed")
    if robot not in {"franka", "dual_franka"}:
        raise ValueError("unsupported robot")
    transcript = list(Path(run_dir).glob("transcript_*.json"))
    for path in transcript:
        record = json.loads(path.read_text())
        if (record.get("finish") or {}).get("status") in {"failure", "stuck"}:
            raise ValueError("agent reported failure; select a successful attempt")
    plan = []
    for row in state.steps[1:]:
        command = row.get("command")
        if not command:
            raise ValueError(f"missing command at step {row['step_idx']}")
        action = command["action"]
        if action == "request_operator_verdict":
            checked_result(row.get("result"))
            if row["result"].get("status") == "failure":
                raise ValueError("source operator reported failure")
            continue
        if action == "task_card_observe":
            checked_result(row.get("result"))
            continue
        if action not in ACTIONS:
            raise ValueError(
                f"unsupported source action {action}; select one clean attempt"
            )
        checked_result(row.get("result"))
        step = row["step_idx"]
        args = {key: value for key, value in command.items() if key != "action"}
        entry = {"source_step": step, "action": action, "arguments": args}
        if action in {"move_delta", "rotate_delta"}:
            annotation = annotations.get(str(step))
            if not annotation:
                raise ValueError(f"step {step} requires a motion intent annotation")
            before = tcp_pose(state.get(step - 1).state, robot, args.get("arm"))
            after = tcp_pose(row["state"], robot, args.get("arm"))
            entry["intent"] = annotation.get("intent", "")
            if action == "move_delta":
                anchor = {
                    "phrase": annotation["phrase"],
                    "camera": annotation["camera"],
                }
                point = localize(
                    state, robot, anchor, molmo, step=step - 1, arm=args.get("arm")
                )
                entry.update(
                    anchor=anchor,
                    offset_xyz=(after[:3] - point).tolist(),
                    reference_xyz=point.tolist(),
                )
            else:
                entry.update(
                    rotation_mode=annotation["rotation_mode"],
                    reference_quaternion=before[3:].tolist(),
                    target_quaternion=after[3:].tolist(),
                )
        plan.append(entry)
    card = {
        "version": 1,
        "robot": robot,
        "task": task,
        "human_verdict": human_verdict,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "run_dir": str(Path(run_dir).resolve()),
            "states_sha256": hashlib.sha256(
                (Path(run_dir) / "states.json").read_bytes()
            ).hexdigest(),
        },
        "requirements": fingerprint(),
        "plan": plan,
    }
    return validate_card(card)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--robot", choices=["franka", "dual_franka"], required=True)
    parser.add_argument(
        "--task", required=True, help="CLI cell tag, e.g. dual_franka_t1"
    )
    parser.add_argument("--robot-config", required=True)
    parser.add_argument("--calibration-path", required=True)
    parser.add_argument("--molmo-endpoint", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    from robots.franka.runtime_config import set_calibration_path, set_robot_config_path
    from robots.franka.task_card.replay import ask_human
    from rpent.robots.components.molmo_client import MolmoClient
    from rpent.utils.rpc import make_rpc_client

    if args.destination.exists():
        parser.error("destination already exists; select a new card path")
    verdict = ask_human(
        f"Source run: {args.run_dir}. Did this attempt complete the task?",
        ("success", "failure"),
    )
    set_robot_config_path(args.robot_config)
    set_calibration_path(args.calibration_path)
    rpc = make_rpc_client(args.molmo_endpoint)
    try:
        card = generate(
            args.run_dir,
            json.loads(args.annotations.read_text()),
            robot=args.robot,
            task=args.task,
            molmo=MolmoClient(rpc),
            human_verdict=verdict,
        )
    finally:
        rpc.close()
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open("x") as file:
        json.dump(card, file, indent=2, allow_nan=False)
    print(args.destination)


if __name__ == "__main__":
    main()
