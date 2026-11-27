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

"""E2E runner that drives ``rpent.cli.main`` with a tool-traversing FunctionModel."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tests.e2e_tests.coverage import inspect_trajectory

E2E_MODEL_NAME = "e2e-tool-walker"
PLANNER_TIMEOUT_SECONDS = 35 * 60

# Observe and inspect before mutating state; leave episode resets until last.
_FIRST_TOOLS = (
    "view_env_state",
    "list_dir",
    "write_text_file",
    "read_text_file",
    "read_image",
    "segment",
    "sample_world_xyz",
    "query_world_map",
    "set_gripper",
    "release",
)

_BOUNDED_STEP_KEYS = ("max_chunks", "chunks", "n_action_steps", "steps", "max_steps")


def selected_cuda_ordinals() -> list[int]:
    """Return physical CUDA ordinals exposed via CUDA_VISIBLE_DEVICES."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    tokens = [part.strip() for part in raw.split(",") if part.strip()]
    if not tokens or not all(token.isdecimal() for token in tokens):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must list one or more physical integer "
            f"ordinals so MuJoCo EGL devices can be selected (got {raw!r})"
        )
    return [int(token) for token in tokens]


def selected_cuda_ordinal() -> int:
    """Return the primary physical CUDA ordinal selected by the runner."""
    return selected_cuda_ordinals()[0]


def _sample_from_schema(prop: dict[str, Any], key: str) -> Any:
    """Synthesize a minimal valid value from a JSON Schema property."""
    if "enum" in prop and prop["enum"]:
        return prop["enum"][0]
    if "default" in prop and prop["default"] is not None:
        return prop["default"]
    if "anyOf" in prop:
        for branch in prop["anyOf"]:
            if isinstance(branch, dict) and branch.get("type") != "null":
                return _sample_from_schema(branch, key)
    kind = prop.get("type")
    if kind == "integer":
        return int(prop.get("minimum", 1))
    if kind == "number":
        return float(prop.get("minimum", 0.05))
    if kind == "boolean":
        return False
    if kind == "string":
        return "agentview" if "camera" in key else "test"
    if kind == "array":
        if key == "xyz":
            return [0.0, 0.0, 0.2]
        items = prop.get("items") or {}
        if items.get("type") in ("integer", "number"):
            return [0.0] * int(prop.get("minItems", 3))
        if items.get("type") == "array":
            return [[128, 128]]
        return []
    if kind == "object":
        return {}
    return "test"


class ToolTraversalPolicy:
    """Select each advertised tool once and build bounded deterministic arguments."""

    def __init__(
        self,
        *,
        output_dir: Path,
        examples: dict[str, dict[str, Any]] | None = None,
        skip_tools: list[str] | set[str] | tuple[str, ...] | None = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._examples = dict(examples or {})
        self._skip = set(skip_tools or ())
        self._requested: set[str] = set()

    def next_call(self, tools: list[Any]) -> tuple[str, dict[str, Any]] | None:
        """Pick the next uncalled ToolDefinition and return ``(name, args)``."""
        available = {
            tool.name: tool.parameters_json_schema or {}
            for tool in tools
            if tool.name not in self._skip
        }
        if not available:
            return None
        if not self._requested:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            (self._output_dir / "tool-inventory.json").write_text(
                json.dumps(sorted(available)),
                encoding="utf-8",
            )
        ordered = [
            *_FIRST_TOOLS,
            *(n for n in available if n not in ("reset", "finish")),
            "reset",
            "finish",
        ]
        name = next(
            (n for n in ordered if n in available and n not in self._requested),
            "finish" if "finish" in available else next(iter(available)),
        )
        self._requested.add(name)
        return name, self._build_arguments(name, available[name])

    def _build_arguments(self, name: str, schema: dict[str, Any]) -> dict[str, Any]:
        if name in self._examples:
            return dict(self._examples[name])
        if name == "list_dir":
            return {"path": str(self._output_dir)}
        if name == "write_text_file":
            return {"path": f"{self._output_dir}/probe.txt", "content": "ok"}
        if name == "read_text_file":
            return {"path": f"{self._output_dir}/probe.txt"}
        if name == "read_image":
            images = sorted(self._output_dir.glob("*.png/00.png"))
            if images:
                return {"name": images[0].parent.name, "step": 0}
        if name == "finish":
            return {
                "status": "stuck",
                "summary": "E2E tool walker exercised all advertised tools",
            }

        props = schema.get("properties") or {}
        args: dict[str, Any] = {}
        for key in schema.get("required") or []:
            if key in props and isinstance(props[key], dict):
                args[key] = _sample_from_schema(props[key], key)
        for key in _BOUNDED_STEP_KEYS:
            if key in props and key not in args:
                args[key] = 1
        return args


def build_tool_walker_model(
    *,
    output_dir: Path,
    examples: dict[str, dict[str, Any]] | None = None,
    skip_tools: list[str] | set[str] | tuple[str, ...] | None = None,
) -> Any:
    """Return a pydantic_ai FunctionModel backed by ``ToolTraversalPolicy``."""
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel

    policy = ToolTraversalPolicy(
        output_dir=output_dir, examples=examples, skip_tools=skip_tools
    )

    def _call(_messages: Any, info: Any) -> ModelResponse:
        choice = policy.next_call([*info.function_tools, *info.output_tools])
        if choice is None:
            return ModelResponse(parts=[TextPart("done")])
        name, args = choice
        return ModelResponse(parts=[ToolCallPart(tool_name=name, args=args)])

    async def _stream(_messages: Any, info: Any):
        choice = policy.next_call([*info.function_tools, *info.output_tools])
        if choice is None:
            yield "done"
            return
        name, args = choice
        yield {0: DeltaToolCall(name=name, json_args=json.dumps(args))}

    return FunctionModel(
        function=_call,
        stream_function=_stream,
        model_name=E2E_MODEL_NAME,
    )


def run_case(
    *,
    robot: str,
    robot_argv: list[str],
    output_dir: Path,
    action_tool: str | None = None,
    action_count_field: str | None = None,
    tool_args: dict[str, dict[str, Any]] | None = None,
    skip_tools: list[str] | None = None,
    require_post_action_image: bool = False,
) -> dict[str, Any]:
    """Exercise every advertised tool via ``rpent.cli.main`` and ``FunctionModel``."""
    memory_dir = output_dir.parent / "offline-memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "MEMORY.md").write_text(
        "# Offline GPU E2E memory\n", encoding="utf-8"
    )
    global_dir = memory_dir / "global"
    global_dir.mkdir(exist_ok=True)
    (global_dir / "GLOBAL_MEMORY.md").write_text(
        "# Offline GPU E2E memory\n\nNo task-specific guidance is provided.\n",
        encoding="utf-8",
    )

    rpent_argv = [
        "--robot",
        robot,
        "--planner",
        "api",
        "--model",
        f"openai-chat:{E2E_MODEL_NAME}",
        "--max-turns",
        "64",
        "--planner-timeout-s",
        str(PLANNER_TIMEOUT_SECONDS),
        "--record-trajectory",
        "--memory-profile",
        "local",
        "--memory-dir",
        str(memory_dir),
        "--output-dir",
        str(output_dir),
        *robot_argv,
    ]
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.e2e_tests.runner",
            str(output_dir),
            json.dumps({"examples": tool_args or {}, "skip_tools": skip_tools or []}),
            "--",
            *rpent_argv,
        ],
        cwd=repo_root,
        env=os.environ.copy(),
        check=False,
    )
    coverage = inspect_trajectory(output_dir, action_tool, skip_tools=skip_tools)
    if completed.returncode or not coverage["passed"]:
        raise RuntimeError(
            f"{robot} planner exit={completed.returncode}, "
            f"missing={coverage['missing']}, failures={coverage['failures']}. "
            f"See {output_dir / 'tool-coverage.json'}"
        )
    if require_post_action_image and not coverage["post_action_new_image_count"]:
        raise RuntimeError("post-action planner request contained no new image")

    transcript_paths = list(output_dir.glob("transcript_*.json"))
    if len(transcript_paths) != 1:
        raise RuntimeError(
            f"expected one planner transcript, found {len(transcript_paths)}"
        )
    transcript = json.loads(transcript_paths[0].read_text(encoding="utf-8"))
    finish = transcript.get("finish")
    if transcript.get("error"):
        raise RuntimeError(f"{robot} planner error: {transcript['error']}")
    if not isinstance(finish, dict) or finish.get("_finish") is not True:
        raise RuntimeError(f"{robot} session did not call finish")

    state_path = output_dir / "states.json"
    action_steps: list[dict[str, Any]] = []
    action_count = 0
    if action_tool:
        if not state_path.is_file():
            raise RuntimeError(f"{robot} session did not write {state_path}")
        state_manifest = json.loads(state_path.read_text(encoding="utf-8"))
        action_steps = [
            step
            for step in state_manifest.get("steps", [])
            if (step.get("command") or {}).get("action") == action_tool
        ]
        if not action_steps:
            raise RuntimeError(f"{robot} session did not record {action_tool}")
        if any((step.get("result") or {}).get("error") for step in action_steps):
            raise RuntimeError(f"{robot} action recorded an error")
        if action_count_field:
            action_count = sum(
                int((step.get("result") or {}).get(action_count_field, 0))
                for step in action_steps
            )
            if action_count <= 0:
                raise RuntimeError(
                    f"{robot} action executed no simulator steps "
                    f"({action_count_field}={action_count})"
                )

    return {
        "status": "passed",
        "planner": E2E_MODEL_NAME,
        "action_tool": action_tool,
        "recorded_action_steps": len(action_steps),
        "simulator_action_count": action_count,
        "finish_status": finish.get("status"),
        "post_action_new_image_count": coverage["post_action_new_image_count"],
        "tools_executed": coverage["executed"],
    }


@contextmanager
def run_sidecar(
    sidecar: dict[str, Any],
    *,
    output_dir: Path,
) -> Iterator[str]:
    """Launch an ephemeral per-case sidecar via ProcessDaemon and stop it on exit."""
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.rpc import make_rpc_client, wait_for_ready

    port = str(pick_free_port())
    cmd = [token.replace("{port}", port) for token in sidecar["cmd"]]
    endpoint = str(sidecar.get("endpoint", "http://127.0.0.1:{port}")).replace(
        "{port}", port
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    daemon = ProcessDaemon(
        "sidecar",
        cmd,
        env_overrides=sidecar.get("env") or None,
        log_path=str(output_dir / "sidecar-server.log"),
    )
    client = make_rpc_client(endpoint)
    try:
        daemon.start()
        wait_for_ready(
            client,
            timeout_s=float(sidecar.get("ready_timeout_s", 300)),
            daemon=daemon,
        )
        yield endpoint
    finally:
        client.close()
        daemon.stop()


def publish_check(name: str, result: dict[str, Any]) -> None:
    """Emit a compact structured result into the pytest and Actions logs."""
    print(json.dumps({"check": name, **result}, indent=2, default=str), flush=True)


def main(argv: list[str] | None = None) -> int:
    """Run ``rpent.cli.main.main`` with ``build_api_model`` patched to FunctionModel."""
    raw = list(sys.argv[1:] if argv is None else argv)
    sep = raw.index("--")
    output_dir = Path(raw[0])
    payload = json.loads(raw[1]) if sep > 1 else {}
    if isinstance(payload, dict) and "examples" in payload:
        examples = payload.get("examples") or {}
        skip_tools = payload.get("skip_tools") or []
    else:
        examples = payload
        skip_tools = []
    rpent_argv = raw[sep + 1 :]

    import rpent.planner.base as planner_base
    from rpent.cli import main as rpent_main

    planner_base.build_api_model = lambda *_a, **_kw: build_tool_walker_model(
        output_dir=output_dir,
        examples=examples,
        skip_tools=skip_tools,
    )
    sys.argv = ["rpent", *rpent_argv]
    return int(rpent_main.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
