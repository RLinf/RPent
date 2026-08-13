#!/usr/bin/env python3
"""Run one saved dual-Franka manual VLA request through current RPent RPCs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rpent.utils.http_rpc import HttpRpcClient
from rpent.utils.socket_rpc import SocketRpcClient
from rpent.utils.vla_client import VLAClient


def _client(endpoint: str):
    if endpoint.startswith("http://"):
        return HttpRpcClient(endpoint)
    value = endpoint.removeprefix("socket://")
    host, port = value.rsplit(":", 1)
    return SocketRpcClient(host, int(port))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--env-endpoint", default="socket://127.0.0.1:5556")
    parser.add_argument("--vla-endpoint", default="http://127.0.0.1:6000")
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    request = json.loads(Path(args.json).read_text())
    params: dict[str, Any] = dict(request.get("params") or {})
    prompt = str(params.get("instruction") or "").strip()
    if not prompt:
        raise ValueError("manual VLA request requires params.instruction")
    max_chunks = int(params.get("max_chunks", 1))
    env_client = _client(args.env_endpoint)
    vla = VLAClient(_client(args.vla_endpoint))
    results = []
    for _ in range(max_chunks):
        observation = env_client.call("env.get_observation", timeout_s=args.timeout_s)
        observation = dict(observation)
        observation["task_descriptions"] = prompt
        actions, _ = vla.predict_action_batch(observation, mode=str(params.get("mode", "eval")))
        result = env_client.call(
            "env.step_chunk",
            kwargs={"actions": np.asarray(actions), "return_all_frames": False},
            timeout_s=args.timeout_s,
        )
        results.append(result)
        if result.get("terminated") or result.get("truncated"):
            break
    print(json.dumps({"chunks_executed": len(results), "last_chunk": results[-1] if results else None}, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
