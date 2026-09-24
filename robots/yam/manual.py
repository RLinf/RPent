# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Operator-only diagnostic tasks; no planner, implicit reset or home."""

import json
from pathlib import Path

from robots.yam.primitives import YamPrimitives
from robots.yam.tasks import DIAGNOSTIC_TASKS
from robots.yam.vla_test import VLATest
from rpent.dashboard.events import NullDashboardEventSink


def run_session(args):
    from robots.yam.robot_spec import _init_runtime, _parse_config
    from rpent.utils.rpc import make_rpc_client

    if not args.env_endpoint:
        raise ValueError("--env-endpoint is required")
    rpc = make_rpc_client(args.env_endpoint)
    if rpc.call("env.is_started", timeout_s=5) is not True:
        raise RuntimeError(
            "Start hardware in the operator terminal first; diagnostics never power on"
        )
    if args.task_id == 104 and (not args.vla_endpoint or args.without_vla):
        raise ValueError("task 104 requires --vla-endpoint")
    config = _parse_config(args)
    selected = {"env", "vla"} if args.task_id == 104 else {"env"}
    _, kwargs = _init_runtime(
        args, config.output_dir, NullDashboardEventSink(), selected
    )
    p = YamPrimitives(**kwargs, check_cancelled=lambda: None)
    test = VLATest(p, config.output_dir)
    log = Path(config.output_dir) / "diagnostics.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(DIAGNOSTIC_TASKS[args.task_id], flush=True)
    print(
        'One JSON command per line: {"tool":"status"}; task103: move_to/rotate_wrist/set_gripper/release; task104: infer, execute with use_length. quit exits and holds. Stop the Dashboard Agent before using this console.',
        flush=True,
    )
    uncertain = False
    try:
        while True:
            try:
                line = input("yam> ").strip()
            except EOFError:
                break
            if line in {"quit", "exit"}:
                break
            event = {"input": line, "diagnostic_only": True}
            try:
                request = json.loads(line)
                name = request["tool"]
                arguments = request.get("arguments", {})
                allowed = {"status", "stop"} | (
                    {"move_to", "rotate_wrist", "set_gripper", "release"}
                    if args.task_id == 103
                    else {"infer", "execute"}
                )
                if name not in allowed:
                    raise ValueError(f"tool not allowed for task {args.task_id}")
                if name == "status":
                    result = p.status()
                elif name == "stop":
                    result = p.env.request_stop()
                elif uncertain:
                    raise RuntimeError(
                        "Previous operation failed; reconcile live state and restart console"
                    )
                else:
                    try:
                        if name in {"infer", "execute"}:
                            result = getattr(test, name)(**arguments)
                        else:
                            if not p.status()["can_continue"]:
                                raise RuntimeError(
                                    "Control not ready; use the operator receipt protocol"
                                )
                            result = getattr(p, name)(**arguments)
                    except Exception:
                        uncertain = True
                        raise
                event["result"] = result
            except Exception as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                p.stop_recording()  # no growing in-memory video buffer
            with log.open("a") as stream:
                stream.write(json.dumps(event, default=str) + "\n")
            print(json.dumps(event, default=str), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        p.env.request_stop()
        print(
            "Stopped and holding. No home, release, shutdown or success memory was generated.",
            flush=True,
        )
    return 0
