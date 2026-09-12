#!/usr/bin/env python3
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

# ruff: noqa: E402
"""Run one Dual-Franka primitive manually against live RPent services.

This is the RPent-side replacement for the old PhysicalAgent
``cli.dual_franka_manual`` helper.  It intentionally bypasses the planner: use it
for live robot bring-up, single-skill VLA checks, perception checks, and reset
or gripper sanity tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robots.dual_franka import perception as dual_franka_perception
from robots.dual_franka.tools import (
    TOOLS_SPEC,
    DualFrankaPrimitives,
    dump_state,
    view_env_state,
)
from robots.franka.tools import view_camera_meta
from rpent.robots.components.pi05_vla_client import Pi05VLAClient
from rpent.robots.components.sam3_client import Sam3Client
from rpent.session import EnvState
from rpent.utils.config import get_repo_root
from rpent.utils.rpc import make_rpc_client, wait_for_ready

_RESET_SPEC: dict[str, Any] = {
    "name": "reset",
    "description": (
        "Manual-only full environment reset to the configured initial posture. "
        "This is intentionally not exposed as a planner primitive."
    ),
    "input_schema": {"type": "object", "properties": {}},
}


def _tool_spec_map(*, include_manual: bool = False) -> dict[str, dict[str, Any]]:
    specs = {str(spec["name"]): spec for spec in TOOLS_SPEC}
    if include_manual:
        specs[str(_RESET_SPEC["name"])] = _RESET_SPEC
    return specs


def _registered_tool_names() -> set[str]:
    """Return the live dual-Franka tool names registered by this checkout."""
    return set(_tool_spec_map())


def _manual_primitive_names() -> set[str]:
    """Return registered tools plus manual-only helpers supported by this script."""
    return set(_tool_spec_map(include_manual=True))


_VLA_PRIMITIVES = {"vla_right_grasp", "vla_handoff", "vla_left_place"}
_SAM3_PRIMITIVES = {"segment"}


def _example_value(schema: dict[str, Any], name: str) -> Any:
    if "default" in schema:
        return schema["default"]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    schema_type = schema.get("type")
    if schema_type == "string":
        return name
    if schema_type == "integer":
        return int(schema.get("minimum", 0))
    if schema_type == "number":
        return float(schema.get("minimum", 0.0))
    if schema_type == "boolean":
        return False
    if schema_type == "array":
        min_items = int(schema.get("minItems", 1))
        item_schema = schema.get("items")
        item_schema = item_schema if isinstance(item_schema, dict) else {}
        return [_example_value(item_schema, name) for _ in range(min_items)]
    if schema_type == "object":
        return _example_params({"input_schema": schema})
    return None


def _example_params(spec: dict[str, Any]) -> dict[str, Any]:
    input_schema = spec.get("input_schema")
    if not isinstance(input_schema, dict):
        return {}
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        return {}
    required = input_schema.get("required")
    names = required if isinstance(required, list) and required else properties.keys()
    return {
        str(name): _example_value(properties.get(name, {}), str(name))
        for name in names
        if isinstance(properties.get(name, {}), dict)
    }


def _schema_payload(name: str) -> dict[str, Any]:
    specs = _tool_spec_map(include_manual=True)
    try:
        spec = specs[name]
    except KeyError as exc:
        known = sorted(specs)
        raise SystemExit(f"unknown primitive {name!r}; known={known}") from exc
    return {
        "primitive": name,
        "description": spec.get("description", ""),
        "input_schema": spec.get("input_schema", {}),
    }


def _example_payload(name: str) -> dict[str, Any]:
    spec = _schema_payload(name)
    if name == "segment":
        return {
            "primitive": name,
            "params": {
                "camera": "d455",
                "prompt": "target object",
                "target_name": "target",
            },
        }
    return {
        "primitive": name,
        "params": _example_params(spec),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{type(value).__name__} len={len(value)}>"
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _strip_for_json(value: Any) -> Any:
    """Remove large/private binary fields before writing human-readable JSON."""
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and key.startswith("_image_"):
                output[key] = _json_default(item)
            else:
                output[key] = _strip_for_json(item)
        return output
    if isinstance(value, list):
        return [_strip_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_strip_for_json(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _json_default(value)
    return value


class ManualDualFrankaEnv:
    """Small env client that does not reset on construction."""

    def __init__(self, endpoint: str):
        self._client = make_rpc_client(endpoint)
        self.meta = self._client.call("env.get_env_meta", timeout_s=30.0)

    def reset(self) -> dict[str, Any]:
        return self._client.call("env.reset", timeout_s=180.0)

    def get_robot_state(self) -> dict[str, Any]:
        return self._client.call("env.get_robot_state", timeout_s=30.0)

    def get_observation(self) -> dict[str, Any]:
        return self._client.call("env.get_observation", timeout_s=30.0)

    def get_camera_meta(self) -> dict[str, Any] | None:
        return self._client.call("env.get_camera_meta", timeout_s=30.0)

    def move_delta(
        self,
        arm: str,
        delta_xyz: np.ndarray | list[float],
    ) -> dict[str, Any]:
        return self._client.call(
            "env.move_delta",
            kwargs={"arm": arm, "delta_xyz": np.asarray(delta_xyz, dtype=np.float32)},
            timeout_s=120.0,
        )

    def rotate_delta(
        self,
        arm: str,
        delta_rpy: np.ndarray | list[float],
    ) -> dict[str, Any]:
        return self._client.call(
            "env.rotate_delta",
            kwargs={"arm": arm, "delta_rpy": np.asarray(delta_rpy, dtype=np.float32)},
            timeout_s=120.0,
        )

    def set_gripper(self, arm: str, *, open: bool) -> dict[str, Any]:
        return self._client.call(
            "env.set_gripper",
            kwargs={"arm": arm, "open": bool(open)},
            timeout_s=120.0,
        )

    def recover_joint_posture(
        self,
        *,
        reason: str = "",
        return_to_start: bool = True,
    ) -> dict[str, Any]:
        return self._client.call(
            "env.recover_joint_posture",
            kwargs={"reason": reason, "return_to_start": bool(return_to_start)},
            timeout_s=240.0,
        )

    def chunk_step(self, actions: np.ndarray) -> dict[str, Any]:
        return self._client.call(
            "env.chunk_step",
            kwargs={"actions": np.asarray(actions, dtype=np.float32)},
            timeout_s=300.0,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one manual Dual-Franka RPent primitive/tool call."
    )
    parser.add_argument("--env-endpoint", default="http://127.0.0.1:6001")
    parser.add_argument("--vla-endpoint", default=None)
    parser.add_argument("--sam3-endpoint", default=None)
    parser.add_argument("--primitive", default=None)
    parser.add_argument(
        "--params",
        default="{}",
        help="JSON object passed as primitive/tool params.",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Print the registered input schema for one primitive and exit.",
    )
    parser.add_argument(
        "--example",
        default=None,
        help="Print a minimal example payload for one primitive and exit.",
    )
    parser.add_argument("--list-primitives", action="store_true")
    parser.add_argument("--no-ready-check", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for result.json and captured states/images.",
    )
    parser.add_argument(
        "--no-dump-state",
        action="store_true",
        help="Do not capture a post-call state/images manifest.",
    )
    return parser


def _load_payload(args: argparse.Namespace) -> dict[str, Any]:
    if not args.primitive:
        raise ValueError("provide --primitive, --schema, --example, or --list-primitives")
    payload = {"primitive": args.primitive, "params": json.loads(args.params)}
    primitive = payload.get("primitive")
    params = payload.get("params", {})
    if not isinstance(primitive, str) or not primitive:
        raise ValueError("payload.primitive must be a non-empty string")
    if not isinstance(params, dict):
        raise ValueError("payload.params must be a JSON object")
    return {"primitive": primitive, "params": params}


def _default_output_dir(primitive: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return get_repo_root() / "logs" / f"{stamp}-manual-{primitive}"


def _call_readonly_tool(
    primitive: str,
    params: dict[str, Any],
    *,
    primitives: DualFrankaPrimitives,
    state: EnvState,
    sam3_client: Sam3Client | None,
    dump_state_enabled: bool,
) -> dict[str, Any]:
    if primitive == "describe_dual_franka_setup":
        return primitives.describe_dual_franka_setup()
    if primitive == "view_env_state":
        if dump_state_enabled:
            dump_state(
                primitives,
                state,
                command={"action": "view_env_state", "params": params},
                result=None,
                elapsed_s=None,
            )
            return view_env_state(state=state, **params)
        return {
            "step_idx": None,
            "state": primitives.env.get_robot_state(),
            "camera_meta": primitives.env.get_camera_meta() or {},
            "images": [],
            "artifact_images": [],
            "note": (
                "--no-dump-state skips observation/image capture; omit it to "
                "record camera artifacts for view_env_state/back_project/segment."
            ),
        }
    if primitive == "view_camera_meta":
        if dump_state_enabled:
            dump_state(
                primitives,
                state,
                command={"action": "view_camera_meta", "params": params},
                result=None,
                elapsed_s=None,
            )
            return view_camera_meta(state=state, **params)
        return {"step": None, "camera_meta": primitives.env.get_camera_meta() or {}}
    if primitive == "back_project":
        if state.latest_step is None:
            if not dump_state_enabled:
                raise RuntimeError(
                    "back_project requires a recorded RGBD snapshot; omit "
                    "--no-dump-state so the manual tool can capture one first."
                )
            dump_state(
                primitives,
                state,
                command={"action": "snapshot_before_back_project", "params": {}},
                result=None,
                elapsed_s=None,
            )
        return dual_franka_perception.back_project(state=state, **params)
    if primitive == "segment":
        if state.latest_step is None:
            if not dump_state_enabled:
                raise RuntimeError(
                    "segment requires a recorded RGBD snapshot; omit "
                    "--no-dump-state so the manual tool can capture one first."
                )
            dump_state(
                primitives,
                state,
                command={"action": "snapshot_before_segment", "params": {}},
                result=None,
                elapsed_s=None,
            )
        return dual_franka_perception.segment(
            state=state,
            sam3_client=sam3_client,
            **params,
        )
    raise KeyError(primitive)


def _call_mutating_primitive(
    primitive: str,
    params: dict[str, Any],
    *,
    env: ManualDualFrankaEnv,
    primitives: DualFrankaPrimitives,
) -> dict[str, Any]:
    if primitive == "reset":
        return env.reset()
    if primitive == "move_delta":
        return primitives.move_delta(**params)
    if primitive == "rotate_delta":
        return primitives.rotate_delta(**params)
    if primitive == "open_gripper":
        return primitives.open_gripper(**params)
    if primitive == "close_gripper":
        return primitives.close_gripper(**params)
    if primitive == "recover_joint_posture":
        return primitives.recover_joint_posture(**params)
    if primitive in _registered_tool_names() and hasattr(primitives, primitive):
        handler = getattr(primitives, primitive)
        return handler(**params)
    raise KeyError(primitive)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_primitives:
        names = sorted(_manual_primitive_names())
        print(json.dumps(names, indent=2))
        return 0
    if args.schema:
        print(json.dumps(_schema_payload(args.schema), indent=2))
        return 0
    if args.example:
        print(json.dumps(_example_payload(args.example), indent=2))
        return 0

    payload = _load_payload(args)
    primitive = payload["primitive"]
    params = payload["params"]
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(primitive)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_rpc = make_rpc_client(args.env_endpoint)
    if not args.no_ready_check:
        wait_for_ready(env_rpc, timeout_s=min(args.timeout_s, 60.0))
    env = ManualDualFrankaEnv(args.env_endpoint)

    needs_vla = primitive in _VLA_PRIMITIVES
    needs_sam3 = primitive in _SAM3_PRIMITIVES

    model = None
    if args.vla_endpoint and needs_vla:
        vla_rpc = make_rpc_client(args.vla_endpoint)
        if not args.no_ready_check:
            wait_for_ready(vla_rpc, timeout_s=min(args.timeout_s, 60.0))
        model = Pi05VLAClient(vla_rpc, embodiment="dual_franka")

    sam3_client = None
    if args.sam3_endpoint and needs_sam3:
        sam3_rpc = make_rpc_client(args.sam3_endpoint)
        if not args.no_ready_check:
            wait_for_ready(sam3_rpc, timeout_s=min(args.timeout_s, 60.0))
        sam3_client = Sam3Client(sam3_rpc)

    state = EnvState(output_dir)
    primitives = DualFrankaPrimitives(
        env=env,
        model=model,
        task_description="manual dual-Franka primitive call",
        check_cancelled=lambda: None,
        sam3_client=sam3_client,
    )

    started = time.perf_counter()
    readonly = {
        "describe_dual_franka_setup",
        "view_env_state",
        "view_camera_meta",
        "back_project",
        "segment",
    }
    try:
        if primitive in readonly:
            result = _call_readonly_tool(
                primitive,
                params,
                primitives=primitives,
                state=state,
                sam3_client=sam3_client,
                dump_state_enabled=not args.no_dump_state,
            )
        else:
            result = _call_mutating_primitive(
                primitive,
                params,
                env=env,
                primitives=primitives,
            )
    except KeyError:
        known = sorted(_manual_primitive_names())
        raise SystemExit(f"unknown primitive {primitive!r}; known={known}")
    elapsed_s = time.perf_counter() - started

    if not args.no_dump_state and primitive not in {
        "view_env_state",
        "view_camera_meta",
        "back_project",
        "segment",
    }:
        try:
            dump_state(
                primitives,
                state,
                command={"action": primitive, "params": params},
                result=result,
                elapsed_s=elapsed_s,
            )
        except Exception as exc:  # keep manual action result visible
            result = {
                "ok": bool(result.get("ok")) if isinstance(result, dict) else None,
                "primitive_result": result,
                "state_dump_error": f"{type(exc).__name__}: {exc}",
            }

    out = _strip_for_json({
        "primitive": primitive,
        "params": params,
        "elapsed_s": elapsed_s,
        "output_dir": str(output_dir),
        "result": result,
    })
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(out, indent=2, default=_json_default))
    print(json.dumps(out, indent=2, default=_json_default))
    print(f"\n[result] {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
