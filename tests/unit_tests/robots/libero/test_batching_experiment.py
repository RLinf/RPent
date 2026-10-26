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

"""Offline behavior checks for the opt-in tool batching experiment."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from robots.libero.experiments.batching import BatchToolkit, execution_policy
from robots.libero.experiments.benchmark_batching import summarize
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.tools import ToolResult, tool


class FakeBatchToolkit(BatchToolkit):
    def __init__(self, path: Path, size: int = 4):
        self.calls = []
        self.outcomes = []
        self.entered = threading.Event()
        self.cancel_seen = threading.Event()
        self.block = False
        super().__init__(
            batch_size=size,
            event_path=path / "tools.jsonl",
            runtime_kwargs={},
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(path / "memory"),
            state_output_dir=path,
        )

    def init_primitives(self, *, runtime_kwargs: dict[str, Any]) -> None:
        with self.state.record_step(state={}, terminated=False):
            pass

    def _register_libero_tools(self) -> None:
        self.add_tool(self.move_to)
        self.add_tool(self.pi0_pick)
        self.add_tool(self.pi0_doubled)

    def action(self, name):
        self.calls.append(name)
        if self.block:
            self.entered.set()
            assert self.cancel_seen.wait(3)
            self.raise_if_cancelled()
        return ToolResult(data=self.outcomes.pop(0))

    @tool
    def move_to(self, xyz: list[float], tol: float = 0.012) -> ToolResult:
        """Move to a test waypoint."""
        return self.action("move_to")

    @tool
    def pi0_pick(self, prompt: str) -> ToolResult:
        """Pick a test object."""
        return self.action("pi0_pick")

    @tool
    def pi0_doubled(self, prompt: str) -> ToolResult:
        """Execute a test contact skill."""
        return self.action("pi0_doubled")

    def get_env_state(self, *, command, result, elapsed_s):
        terminated = result.get("terminated", False)
        self._solved |= terminated
        with self.state.record_step(
            state={},
            command=command,
            result=result,
            elapsed_s=elapsed_s,
            terminated=terminated,
            truncated=result.get("truncated", False),
        ):
            pass
        return ToolResult(data={"observed": len(self.calls)})

    def cancel_active_and_wait(self):
        # Release the fake primitive only after the scheduler has paused admission.
        self._batch_cancelled.set()
        self._scheduler.cancel()
        self.cancel_seen.set()
        super().cancel_active_and_wait()


def move():
    return {"action": "move_to", "arguments": {"xyz": [0.0, 0.0, 0.2]}}


def pick(name="pi0_pick"):
    return {"action": name, "arguments": {"prompt": "pick"}}


def test_batch_reuses_child_state_capture_and_final_observation(tmp_path):
    toolkit = FakeBatchToolkit(tmp_path)
    toolkit.outcomes = [{"success": True}, {"final_dist_m": 0.001, "terminated": True}]
    result = toolkit.execute_tool(
        "execute_sequence", {"steps": [pick(), move(), move()]}
    )
    assert toolkit.calls == ["pi0_pick", "move_to"]
    assert result.data["observed"] == 2
    assert result.data["stop_reason"] == "native_success"
    assert [s["step_idx"] for s in result.data["sequence"]] == [1, 2]
    assert toolkit.state.latest_record().command["action"] == "move_to"


@pytest.mark.parametrize(
    "steps",
    [
        [],
        [move()] * 5,
        [move(), {"action": "reset", "arguments": {}}],
        [move(), {"action": "move_to", "arguments": {"xyz": "invalid"}}],
        [{"action": "execute_sequence", "arguments": {"steps": []}}],
    ],
)
def test_invalid_later_step_prevents_all_motion(tmp_path, steps):
    toolkit = FakeBatchToolkit(tmp_path)
    result = toolkit.execute_tool("execute_sequence", {"steps": steps})
    assert result.is_error
    assert toolkit.calls == []


@pytest.mark.parametrize(
    ("first", "outcome", "reason"),
    [
        (pick(), {"success": False}, "pick_unconfirmed"),
        (pick(), {}, "pick_unconfirmed"),
        (move(), {"final_dist_m": 0.1}, "position_not_reached"),
        (move(), {"final_dist_m": 0.0, "position_hold_ok": False}, "action_failed"),
        (move(), {"final_dist_m": 0.0, "truncated": True}, "environment_truncated"),
        (move(), {"error": "RPC failed"}, "tool_error"),
        (pick("pi0_doubled"), {"success": False}, "contact_requires_observation"),
    ],
)
def test_sequence_returns_before_dependent_action(tmp_path, first, outcome, reason):
    toolkit = FakeBatchToolkit(tmp_path)
    toolkit.outcomes = [outcome]
    result = toolkit.execute_tool("execute_sequence", {"steps": [first, move()]})
    assert len(toolkit.calls) == 1
    assert result.data["stop_reason"] == reason


def test_ordinary_arm_has_no_sequence_and_batch_limit_is_configurable(tmp_path):
    ordinary = FakeBatchToolkit(tmp_path, 0)
    assert ordinary.execute_tool("execute_sequence", {"steps": [move()]}).is_error
    assert "execute_sequence" not in execution_policy(0)
    batched = FakeBatchToolkit(tmp_path, 8)
    batched.outcomes = [{"final_dist_m": 0.0}] * 8
    result = batched.execute_tool("execute_sequence", {"steps": [move()] * 8})
    assert result.data["actions_executed"] == 8
    assert len(batched.calls) == 8
    assert "8" in execution_policy(8)


def test_cancellation_drains_batch_without_running_next_child(tmp_path):
    toolkit = FakeBatchToolkit(tmp_path)
    toolkit.block = True
    results = []
    worker = threading.Thread(
        target=lambda: results.append(
            toolkit.execute_tool("execute_sequence", {"steps": [move(), move()]})
        )
    )
    worker.start()
    assert toolkit.entered.wait(3)
    toolkit.cancel_active_and_wait()
    worker.join(3)
    assert not worker.is_alive()
    assert toolkit.calls == ["move_to"]
    assert results[0].is_error
    toolkit.block = False
    toolkit.resume_calls()
    toolkit.outcomes = [{"final_dist_m": 0.0}]
    assert not toolkit.execute_tool("move_to", move()["arguments"]).is_error


def row(task, size, success, seconds):
    return {
        "cell": f"{task}-{size}",
        "backend": "apxinf",
        "batch_size": size,
        "task": task,
        "seed": 0,
        "native_success": success,
        "wall_s": seconds,
        "infrastructure_ok": True,
        "vla_rpc_s": 2.0,
        "vla_calls": 10,
        "llm_responses": 20,
        "sequence_calls": 0,
        "sequence_actions": 0,
    }


def test_failed_short_episode_is_not_reported_as_speedup():
    rows = [
        row(0, 0, True, 100),
        row(0, 4, False, 10),
        row(1, 0, True, 100),
        row(1, 4, True, 80),
    ]
    rows.append({**row(2, 4, False, 1), "infrastructure_ok": False})
    report = summarize(rows)
    pair = report["pairs"][0]
    assert pair["complete_pairs"] == 2
    assert pair["success_loss_pairs"] == 1
    assert pair["both_success_pairs"] == 1
    assert pair["both_success_mean_saved_s"] == 20
    assert report["infrastructure_excluded"] == ["2-4"]
    assert report["groups"][1]["wall_mean_all_s"] == 45


def test_sweep_resume_preserves_results_and_rejects_changed_protocol(
    tmp_path, monkeypatch
):
    import json

    from robots.libero.experiments import benchmark_batching as benchmark

    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "MEMORY.md").write_text("Frozen task memory")
    provenance = tmp_path / "provenance.json"
    provenance.write_text('{"dataset": "test"}')
    output = tmp_path / "run"
    arguments = [
        "--output-dir",
        str(output),
        "--memory-dir",
        str(memory),
        "--provenance",
        str(provenance),
        "--backend",
        "apxinf=http://localhost:1",
        "--sam3-endpoint",
        "http://localhost:2",
        "--model",
        "test",
        "--tasks",
        "0",
        "--seeds",
        "0",
        "--batch-sizes",
        "0",
        "4",
    ]
    launches = []

    class Worker:
        returncode = 0

        def __init__(self, command, **kwargs):
            directory = Path(command[-2]) / command[-1]
            cell = json.loads((directory / "cell.json").read_text())
            result = row(cell["task"], cell["batch_size"], True, 100)
            result.update(cell, initial_image_sha256="same")
            benchmark.write_json(directory / "result.json", result)
            launches.append(command[-1])

        def wait(self, timeout):
            return self.returncode

        def poll(self):
            return self.returncode

    monkeypatch.setattr(benchmark.subprocess, "Popen", Worker)
    monkeypatch.setattr(
        benchmark.subprocess, "check_output", lambda *a, **k: "revision"
    )
    benchmark.sweep(benchmark.parser().parse_args(arguments))
    assert len(launches) == 2
    benchmark.sweep(benchmark.parser().parse_args(arguments + ["--resume"]))
    assert len(launches) == 2
    assert json.loads((output / "completed.json").read_text())["episodes"] == 2
    (memory / "MEMORY.md").write_text("changed")
    with pytest.raises(ValueError, match="Resume inputs differ"):
        benchmark.sweep(benchmark.parser().parse_args(arguments + ["--resume"]))


@pytest.mark.parametrize(
    "planner_error",
    [None, "Selected model is at capacity.", "Codex SDK timed out after 900s"],
)
def test_episode_owns_cleanup_and_distinguishes_budget_from_service_failure(
    tmp_path, monkeypatch, planner_error
):
    from types import SimpleNamespace

    from robots.libero import env_client
    from robots.libero.experiments import batching
    from robots.libero.experiments import benchmark_batching as benchmark
    from rpent.planner import codex
    from rpent.robots.components import pi05_vla_client, sam3_client
    from rpent.utils import daemon, rpc

    output = tmp_path / "cell"
    output.mkdir()
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "MEMORY.md").write_text("frozen")
    closed = []
    rpc_calls = []

    class Client:
        def __init__(self, endpoint):
            self.endpoint = endpoint

        def call(self, method, *args, **kwargs):
            rpc_calls.append(method)

        def close(self):
            closed.append(self.endpoint)

    class Daemon:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            closed.append("daemon")

    class FakeToolkit:
        def __init__(self, **kwargs):
            self.runtime = kwargs["runtime_kwargs"]
            images = output / "agentview.png"
            images.mkdir()
            (images / "00.png").write_bytes(b"same image")

        def close(self):
            closed.append("toolkit")

    class Planner:
        def __init__(self, **kwargs):
            pass

        def solve(self, **kwargs):
            toolkit = kwargs["toolkit"]
            toolkit.runtime["model"].call("vla.predict")
            return SimpleNamespace(
                error=planner_error,
                finish_result={"status": "success"},
                stats={"turns_used": 2},
            )

    monkeypatch.setattr(rpc, "make_rpc_client", Client)
    monkeypatch.setattr(rpc, "wait_for_ready", lambda *a, **k: None)
    monkeypatch.setattr(daemon, "ProcessDaemon", Daemon)
    monkeypatch.setattr(daemon, "pick_free_port", lambda: 12345)
    # A claimed planner success must not override the native false flag.
    monkeypatch.setattr(
        env_client, "LiberoEnvClient", lambda *a, **k: SimpleNamespace(terminated=False)
    )
    monkeypatch.setattr(pi05_vla_client, "Pi05VLAClient", lambda client, **k: client)
    monkeypatch.setattr(sam3_client, "Sam3Client", lambda client: client)
    monkeypatch.setattr(batching, "BatchToolkit", FakeToolkit)
    monkeypatch.setattr(codex, "CodexPlanner", Planner)
    config = {
        "backends": {"apxinf": "http://localhost:1"},
        "sam3_endpoint": "http://localhost:2",
        "suite": "libero_object_task",
        "max_episode_steps": 10000,
        "cuda_device": None,
        "memory_dir": str(memory),
        "memory_sha256": benchmark.file_hashes(memory),
        "model": "test",
        "reasoning_effort": "low",
        "planner_timeout_s": 900,
        "max_turns": 100,
    }
    cell = {"cell": "test", "backend": "apxinf", "task": 0, "seed": 0, "batch_size": 4}
    result = benchmark.run_episode(config, cell, output)
    assert result["native_success"] is False
    assert result["infrastructure_ok"] is (
        planner_error != "Selected model is at capacity."
    )
    assert result["vla_calls"] == 1
    assert result["vla_rpc_s"] >= 0
    assert closed == [
        "toolkit",
        "http://127.0.0.1:12345",
        "daemon",
        "http://localhost:2",
        "http://localhost:1",
    ]
    assert rpc_calls == ["vla.predict"]
    assert (output / "result.json").exists()
