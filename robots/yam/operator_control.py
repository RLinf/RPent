# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""On-site operator receipts; this is not an Agent tool or a verdict RPC.

Run on the control machine as the owner of the local operator receipt file.
The endpoint is used only to read the episode ID and to consume a ready
receipt via start (bookkeeping, no home motion). Verdicts remain local files.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import uuid
from pathlib import Path


def write_receipt(path: str | Path, *, episode_id: str, event: str, note: str) -> dict:
    if event not in {
        "ready",
        "success",
        "failure",
        "abort",
        "release_left",
        "release_right",
    }:
        raise ValueError(
            "event must be ready, success, failure, abort, release_left, or release_right"
        )
    if not episode_id or not note.strip():
        raise ValueError("episode_id and operator note are required")
    receipt = {
        "episode_id": episode_id,
        "event": event,
        "source": "operator_local",
        "request_id": uuid.uuid4().hex,
        "created_at_s": time.time(),
        "note": note,
    }
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=".receipt-")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(receipt, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return receipt


def read_receipt(path: str | Path | None) -> dict | None:
    if not path:
        return None
    try:
        value = json.loads(Path(path).expanduser().read_text())
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError("operator receipt must be a JSON object")
    if value.get("event") not in {
        "ready",
        "success",
        "failure",
        "abort",
        "release_left",
        "release_right",
    }:
        raise ValueError("operator receipt has invalid event")
    for key in ("episode_id", "request_id", "note"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"operator receipt.{key} must be a nonempty string")
    if value.get("source") != "operator_local":
        raise ValueError("operator receipt.source must be operator_local")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8110")
    parser.add_argument(
        "--episode-id", help="Current ID from status; required for mutations"
    )
    parser.add_argument(
        "--event",
        default="status",
        choices=(
            "status",
            "ready",
            "start",
            "success",
            "failure",
            "abort",
            "recover_up",
            "release_gripper",
            "reset_pose",
            "shutdown",
        ),
    )
    parser.add_argument(
        "--command",
        choices=("/done", "/success", "/failure", "/abort"),
        help="PR182 operator command aliases; /done means scene ready, not motion",
    )
    parser.add_argument("--note", default="")
    parser.add_argument("--arm", choices=("left", "right"))
    parser.add_argument("--distance-m", type=float, default=0.04)
    args = parser.parse_args()
    if args.command:
        if args.event != "status":
            parser.error("use --command or --event, not both")
        args.event = {
            "/done": "ready",
            "/success": "success",
            "/failure": "failure",
            "/abort": "abort",
        }[args.command]
    from rpent.utils.rpc import make_rpc_client

    client = make_rpc_client(args.endpoint)
    if args.event == "status":
        _, info = client.call("env.read_control_state", timeout_s=120)
        print(json.dumps(info["episode_status"], default=str, ensure_ascii=False))
        return
    if args.event in {"reset_pose", "shutdown"}:
        method = (
            "env.reset_to_configured_qpos" if args.event == "reset_pose" else "shutdown"
        )
        print(json.dumps(client.call(method, timeout_s=180), ensure_ascii=False))
        return
    if not args.episode_id or not args.note.strip():
        parser.error("--episode-id and --note are required for operator events")
    config = json.loads(Path(args.config).read_text())
    path = config.get("operator_receipt_path")
    if not path:
        parser.error("config.operator_receipt_path is required")
    if args.event in {"recover_up", "release_gripper"} and args.arm is None:
        parser.error("recovery requires --arm")
    receipt_event = "ready" if args.event in {"start", "recover_up"} else args.event
    if args.event == "release_gripper":
        receipt_event = f"release_{args.arm}"
    receipt = write_receipt(
        path, episode_id=args.episode_id, event=receipt_event, note=args.note
    )
    if args.event == "release_gripper":
        print(
            json.dumps(
                client.call(
                    "env.release_gripper",
                    args=[args.arm],
                    kwargs={"expected_episode_id": args.episode_id},
                    timeout_s=15,
                ),
                ensure_ascii=False,
            )
        )
    elif args.event == "recover_up":
        print(
            json.dumps(
                client.call(
                    "env.recover_up",
                    args=[args.arm, args.distance_m],
                    kwargs={"expected_episode_id": args.episode_id},
                    timeout_s=120,
                ),
                ensure_ascii=False,
            )
        )
    elif args.event == "start":
        result = client.call("env.reset", timeout_s=120)
        print(json.dumps(result[1]["episode_status"], default=str, ensure_ascii=False))
    else:
        if args.event == "abort":
            client.call("env.request_stop", timeout_s=5)
        print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
