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

"""Two official episodes through the shipped XPolicyLab launcher."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e_tests.offline_planner_server import (
    OFFLINE_MODEL_NAME,
    OfflinePlannerServer,
    ScriptedToolCall,
)

pytestmark = [
    pytest.mark.timeout(1800),
    pytest.mark.skipif(
        os.environ.get("RPENT_RUN_XPOLICYLAB_INTEGRATION") != "1",
        reason="requires official runtime, a reserved GPU and a VPP2 endpoint",
    ),
]


def test_two_official_episodes():
    runtime = Path(os.environ["RPENT_XPOLICYLAB_ROOT"])
    output = Path(os.environ["RPENT_E2E_OUTPUT_DIR"]) / "xpolicylab"
    memory = output.parent / "offline-memory"
    memory.mkdir(parents=True, exist_ok=True)
    (memory / "MEMORY.md").write_text("# Integration check: no task memory\n")
    sequence = (
        ScriptedToolCall("view_env_state", {"step": -1}),
        ScriptedToolCall("set_gripper", {"arm": "left", "val": 0.5, "steps": 2}),
        ScriptedToolCall(
            "rotate_wrist", {"arm": "left", "delta_yaw_deg": 1, "substeps": 5}
        ),
        ScriptedToolCall(
            "vpp2_act", {"instruction": "Keep both hands above the table.", "chunks": 1}
        ),
        ScriptedToolCall("set_gripper", {"arm": "right", "val": 0.5, "steps": 2}),
        ScriptedToolCall("view_env_state", {"step": -1}),
        ScriptedToolCall(
            "finish",
            {"status": "stuck", "summary": "Bounded integration check completed."},
        ),
    )
    with OfflinePlannerServer(sequence * 2) as server:
        env = dict(
            os.environ,
            OPENAI_API_KEY=OFFLINE_MODEL_NAME,
            OPENAI_BASE_URL=server.base_url,
        )
        env.update(NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
        command = [
            sys.executable,
            "-m",
            "robots.robodojo.xpolicylab",
            "--runtime-root",
            str(runtime),
            "--task",
            "stack_blocks",
            "--seed",
            "1",
            "--layouts",
            "2,7",
            "--am-endpoint",
            os.environ["RPENT_ROBODOJO_VLA_ENDPOINT"],
            "--memory-dir",
            str(memory),
            "--output-dir",
            str(output),
            "--planner",
            "api",
            "--model",
            f"openai-chat:{OFFLINE_MODEL_NAME}",
            "--timeout",
            "600",
            "--env-cuda-device",
            os.environ.get("RPENT_XPOLICYLAB_ENV_GPU", "0"),
            "--policy-cuda-device",
            os.environ.get("RPENT_XPOLICYLAB_POLICY_GPU", "0"),
        ]
        if catalog := os.environ.get("RPENT_OFFICIAL_DESCRIPTIONS"):
            command += ["--official-descriptions", catalog]
        subprocess.run(command, env=env, check=True)
        server.assert_complete()
    reports = [
        json.loads(p.read_text())
        for p in sorted((output / "episodes").glob("*/report.json"))
    ]
    assert [r["episode"]["layout"] for r in reports] == [2, 7]
    assert all(r["status"] == "completed" for r in reports)
    assert all(r["native_steps"] == r["history_steps"] == 33 for r in reports)
    assert all(r["inference_count"] == 1 for r in reports)
    result_dir = Path(json.loads((output / "manifest.json").read_text())["result_dir"])
    # Verify partial batches too: official aggregate tables require full counts.
    spec = importlib.util.spec_from_file_location(
        "official_summary", runtime / "scripts/internal/summarize_result.py"
    )
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    entries = summary.load_entries(str(result_dir))
    cameras, count = summary.video_inventory(str(result_dir))
    assert len(entries) == 2
    details = json.loads((result_dir / "_result.json").read_text())["details"]
    assert {entry["layout_id"] for entry in details.values()} == {2, 7}
    # Official video names use record IDs; details map them to original layouts.
    assert count == 6 and all(
        cameras[str(int(record))] == summary.EXPECTED_CAMERAS for record in details
    )
    assert any(server.image_digests), "planner must receive camera images"
