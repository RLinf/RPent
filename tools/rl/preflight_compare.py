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

"""Verify native workflow boundaries and real VLM image requests before scoring."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from robots.rl.rl_episode import render_prompts, start_environment, write_transcript
from robots.rl.robot_spec import configure_planner, get_robot_spec
from rpent.dashboard.events import NullDashboardEventSink
from rpent.evaluation import write_json_atomic as atomic
from rpent.evaluation.trajectory import audit_episode
from rpent.planner.api_loop import ApiAgentLoop
from rpent.utils.logging import init_output_dir
from tools.rl.compare_workflows import WORKFLOWS
from tools.rl.run_batch import source_identity


def scripted_native(args, workflow: str) -> dict:
    """Exercise genuine native tools with a test planner, outside benchmark tasks."""
    output = args.output / f"scripted-{workflow}"
    init_output_dir(output)
    parser = argparse.ArgumentParser()
    get_robot_spec().add_cli_args(parser, use_dashboard=False)
    runtime = parser.parse_args(
        [
            "--workflow",
            workflow,
            "--sam3-endpoint",
            args.sam3_endpoint,
            "--cuda-device",
            "0",
            "--libero-type",
            "pro",
        ]
    )
    runtime.output_dir = output
    with contextlib.ExitStack() as release:
        toolkit, config = start_environment(
            runtime, {"suite": "libero_object", "task": 0, "seed": 1}, release
        )
        pose = toolkit._primitives.measure().state()["pose"]["xyz"]
        query = ("change_query", {"queries": ["object", "basket"]})
        move = ("move_to", {"xyz": [pose[0], pose[1], pose[2] + 0.02]})
        grip = ("set_gripper", {"opening": 0.9})
        finish = ("finish", {})
        if workflow == "episode":
            script = [
                [("view_env", {})],
                [("segment", {"prompt": "object"})],
                [
                    (
                        "back_project",
                        {"pixel": [512, 512], "camera": "agentview", "step": 0},
                    )
                ],
                [move],
                [grip],
                [finish],
            ]
        elif workflow == "zap_turn":
            script = [
                [query],
                [("update_todo", {"items": ["verify motion, then gripper"]})],
                [move],
                [grip],
                [finish],
            ]
        elif workflow == "zap_episode":
            script = [[query], [move], [grip], [finish]]
        else:
            script = [
                [
                    query,
                    (
                        "update_todo",
                        {
                            "items": [
                                {"text": "verify motion", "status": "in_progress"},
                                {"text": "verify gripper", "status": "pending"},
                            ]
                        },
                    ),
                ],
                [move],
                [
                    (
                        "update_todo",
                        {
                            "items": [
                                {"text": "verify motion", "status": "done"},
                                {"text": "verify gripper", "status": "in_progress"},
                            ]
                        },
                    )
                ],
                [grip],
                [finish],
            ]
        position = 0

        async def stream(messages, info):
            nonlocal position
            calls = script[position]
            position += 1
            yield {
                i: DeltaToolCall(name=name, json_args=json.dumps(arguments))
                for i, (name, arguments) in enumerate(calls)
            }

        planner = configure_planner(
            ApiAgentLoop(
                model=FunctionModel(stream_function=stream),
                dashboard_events=NullDashboardEventSink(),
                timeout_s=600,
                sequential_tools=True,
            ),
            config,
        )
        system, user = render_prompts(config)
        result = planner.solve(
            system_prompt=system, user_message=user, toolkit=toolkit, max_turns=20
        )
        write_transcript(output, result)
        assert result.error is None and result.finish_result is not None
        assert toolkit._primitives.env_steps > 0
        if workflow == "zap_subtask":
            assert toolkit.workflow_stats["context_resets"] == 2
        evidence = {
            "passed": True,
            "requests": result.stats["turns_used"],
            "tool_calls": result.stats["tool_calls"],
            "env_steps": toolkit._primitives.env_steps,
            "workflow_stats": dict(toolkit.workflow_stats),
            "native_success": toolkit.solved(),
        }
    atomic(output / "verified.json", evidence)
    return evidence


def real_smoke(args, workflow: str, size: int, index: int) -> dict:
    """Make bounded real Qwen requests and audit exact wire images and penalties."""
    output = args.output / f"vlm-{workflow}-{size}"
    endpoint = args.endpoints.split(",")[index % len(args.endpoints.split(","))]
    environment = {
        **os.environ,
        "RPENT_MODEL": "openai-chat:qwen3.8-27b",
        "RPENT_BASE_URL": endpoint,
        "RPENT_SAM3_ENDPOINT": args.sam3_endpoint,
        "RPENT_MAX_TURNS": "6",
        "RPENT_OUTPUT_DIR": str(output),
        "RPENT_IMAGE_SIZE": str(size),
        "RPENT_PRESENCE_PENALTY": "0",
        "RPENT_REPETITION_PENALTY": "1",
        "RPENT_RECORD_MODEL_REQUESTS": "1",
        "LIBERO_TYPE": "pro",
    }
    command = [
        "bash",
        "tools/rl/run.sh",
        "--simulator",
        "libero",
        "--workflow",
        workflow,
        "--suite",
        "libero_object",
        "--task",
        "0",
        "--seed",
        "1",
        "--cuda-device",
        str(index % 2),
    ]
    with (args.output / f"vlm-{workflow}-{size}.log").open("w") as log:
        subprocess.run(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=900,
        )
    result, turns = audit_episode(output, verify_assets=True)
    for directory, _, _ in turns:
        wire = json.loads((directory / "request.json").read_text())
        assert wire["presence_penalty"] == 0
        images = []

        def visit(item):
            if isinstance(item, dict):
                if "$asset" in item and item.get("media_type", "").startswith("image/"):
                    with Image.open(output / "api_trace" / item["$asset"]) as image:
                        images.append(image.size)
                else:
                    for value in item.values():
                        visit(value)
            elif isinstance(item, list):
                for value in item:
                    visit(value)

        visit(wire)
        assert all(shape == (size, size) for shape in images)
    first = json.loads((output / "api_trace/responses/000001.json").read_text())[0]
    evidence = {
        "passed": True,
        "result": result,
        "first_input_tokens": first["usage"]["input_tokens"],
        "wire_turns": len(turns),
    }
    atomic(output / "verified.json", evidence)
    return evidence


def main() -> None:
    """Combine exact-source CPU, capacity, native-tool and model-request proofs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument("--cpu-proof", type=Path)
    parser.add_argument("--sam3-endpoint", required=True)
    parser.add_argument("--endpoints", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    identity = source_identity()
    capacity = json.loads(args.capacity.read_text())
    assert capacity["passed"] and capacity["source"] == identity
    if args.cpu_proof:
        cpu = json.loads(args.cpu_proof.read_text())
        assert cpu["passed"] and cpu["source"] == identity
    else:
        for command in (
            [sys.executable, "-m", "pytest", "tests/unit_tests", "-q"],
            [sys.executable, "-m", "ruff", "check", "--preview", "."],
            [sys.executable, "-m", "ruff", "format", "--check", "."],
        ):
            subprocess.run(command, check=True)
    scripted = {name: scripted_native(args, name) for name in WORKFLOWS}
    panels = [(name, 1024) for name in WORKFLOWS] + [
        ("zap_episode", 512),
        ("zap_episode", 256),
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            f"{name}-{size}": executor.submit(real_smoke, args, name, size, index)
            for index, (name, size) in enumerate(panels)
        }
        vlm = {label: future.result() for label, future in futures.items()}
    token_counts = [
        vlm[f"zap_episode-{size}"]["first_input_tokens"] for size in (1024, 512, 256)
    ]
    assert token_counts[0] > token_counts[1] > token_counts[2]
    from transformers import Qwen2VLImageProcessorFast

    processor = Qwen2VLImageProcessorFast.from_pretrained(
        str(args.model_path), local_files_only=True
    )
    grids = {
        size: processor(images=Image.new("RGB", (size, size)), return_tensors="pt")[
            "image_grid_thw"
        ].tolist()
        for size in (1024, 512, 256)
    }
    assert source_identity() == identity
    atomic(
        args.output / "verified.json",
        {
            "passed": True,
            "source": identity,
            "scripted": scripted,
            "vlm": vlm,
            "image_grids": grids,
            "first_input_tokens": token_counts,
            "capacity": str(args.capacity),
            "checks": {
                "cpu": True,
                "capacity": True,
                "native_chain": True,
                "image_sizes": True,
            },
        },
    )


if __name__ == "__main__":
    main()
