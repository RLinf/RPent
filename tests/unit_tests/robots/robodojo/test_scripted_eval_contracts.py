# Copyright 2026 The RPent Authors.
"""Offline freeze, reward boundary and subprocess evaluation contracts."""

import ast
import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from robots.robodojo.scripted import eval as runner
from robots.robodojo.scripted.hashing import sha256_file, sha256_sources
from robots.robodojo.scripted.reward_audit import check_payload, strict_audit
from rpent.utils.rpc.http_rpc import HttpRpcServer


@pytest.fixture
def recipe(tmp_path):
    path = tmp_path / "recipe.py"
    path.write_text("def main(env, output):\n    env.get_obs()\n    env.step({})\n")
    return path


@pytest.fixture
def frozen(tmp_path, recipe):
    source = tmp_path / "sim"
    (source / "task").mkdir(parents=True)
    (source / "task/task.yml").write_text("task: general_pickup\n")
    output = tmp_path / "release"
    runner.freeze(
        output, source, recipe, task="general_pickup", layout=0, step_limit=200
    )
    return output


def test_worker_gate_and_no_repository_imports(tmp_path, recipe):
    worker = tmp_path / "worker.py"
    worker.write_text(runner.generated_worker(recipe))
    assert strict_audit(worker)["passed"]
    assert "robots.robodojo" not in worker.read_text()
    assert "rpent" not in worker.read_text()


def test_numerical_primitives_are_identical_ast(recipe):
    original = ast.parse(
        (runner.ROOT / "robots/robodojo/scripted/primitives.py").read_text()
    )
    generated = ast.parse(runner.generated_worker(recipe))
    # This branch owns move_to/set_gripper, not the research move_to_pose.
    for name in runner.PRIMITIVES:
        a = next(n for n in original.body if getattr(n, "name", "") == name)
        b = next(n for n in generated.body if getattr(n, "name", "") == name)
        assert ast.dump(a) == ast.dump(b)


@pytest.mark.parametrize(
    "code",
    [
        "env.get_reward_details()",
        "env.get_score()",
        "env.is_success()",
        "x['official_success']",
        "dict(reward=1)",
        "x.completed_predicates",
        "import robots.robodojo.env_client",
        "from rpent.utils import rpc",
        "import socket",
        "from . import local",
        "__import__('socket')",
        "x = [{'prompt': 'a', 'stage': 1}, {'prompt': 'b', 'stage': 2}]",
        "def broken(:",
    ],
)
def test_forbidden_reads_and_paths_fail(tmp_path, code):
    worker = tmp_path / "worker.py"
    worker.write_text(code)
    assert not strict_audit(worker)["passed"]


def test_nested_privileged_field_denied():
    with pytest.raises(RuntimeError, match="privileged"):
        check_payload({"obs": [{"state": {"official_success": True}}]})


def test_status_is_allowlisted():
    safe = runner.safe_status(
        {
            "step": 0,
            "step_limit": 200,
            "success": True,
            "horizon": {"official_success": True},
            "safety": {"bottle_pose": [1, 2, 3]},
        }
    )
    check_payload(safe)
    assert safe == {"step": 0, "step_limit": 200}


@pytest.mark.parametrize(
    "method",
    ["is_success", "get_reward_details", "reset", "close", "solve_ik_pose", "env.step"],
)
def test_bridge_rejects_arbitrary_rpc(method):
    with pytest.raises(RuntimeError, match="RPC denied"):
        runner.bridge_call(None, {"method": method, "args": []})


class FakeRpc:
    def __init__(self):
        self.calls = []
        self.step = 0

    def call(self, method, args=(), kwargs=None, **options):
        self.calls.append(method)
        status = {
            "step": self.step,
            "step_limit": 200,
            "success": True,
            "safety": {"reward": 1},
        }
        obs = {
            "vision": {
                "cam_head": {"color": np.zeros((1, 1, 3), dtype=np.uint8), "score": 1}
            },
            "state": {"left_ee_pose": [0] * 7, "reward": 1},
            "instruction": "test",
            "reward": 1,
        }
        if method == "env.get_env_meta":
            return {"task": "general_pickup", "layout": 0, "random": False}
        if method == "env.get_status":
            return status
        if method in {"env.get_obs", "env.reset"}:
            return obs
        if method == "env.step":
            self.step += 1
            return (
                obs,
                1.0,
                True,
                {"status": {**status, "step": self.step}, "reward": 1},
            )
        if method == "env.is_success":
            return False
        if method == "env.get_reward_details":
            return {
                "reward": 0.25,
                "success": True,
                "horizon": {"official_success": True},
                "process_score": 99,
            }
        raise AssertionError(method)

    def close(self):
        self.calls.append("client.close")


def test_step_filters_reward_done_and_nested_payload():
    result = runner.bridge_call(FakeRpc(), {"method": "step", "args": [{}]})
    check_payload(result)
    assert set(result) == {"obs", "status"}
    assert set(result["obs"]["state"]) == {"left_ee_pose"}
    assert set(result["obs"]["vision"]["cam_head"]) == {"color"}
    assert result["status"] == {"step": 1, "step_limit": 200}


def test_action_budget_blocks_before_step():
    rpc = FakeRpc()
    rpc.step = 180
    with pytest.raises(RuntimeError, match="budget"):
        runner.bridge_call(rpc, {"method": "step", "args": [{}]})
    assert "env.step" not in rpc.calls


def test_freeze_verify_and_refreeze_archive(frozen):
    release = runner.verify(frozen)
    assert release["official_success_field"] == "env.is_success()"
    assert all(Path(p).is_file() for p in release["source_sha256"])
    digest = sha256_file(frozen / "frozen_release.json")
    runner.freeze(
        frozen,
        Path(release["source_root"]),
        Path(release["recipe"]),
        task="general_pickup",
        layout=0,
        step_limit=200,
    )
    assert (frozen / "preflight_releases" / f"{digest}.json").exists()
    runner.verify(frozen)


@pytest.mark.parametrize("target", ["recipe", "worker", "sim", "new_sim", "digest"])
def test_verify_rejects_changes(frozen, target):
    release = runner.verify(frozen)
    paths = {
        "recipe": Path(release["recipe"]),
        "worker": frozen / "release/worker.py",
        "sim": Path(release["simulator_sources"][0]),
        "new_sim": Path(release["source_root"]) / "task/new.py",
    }
    if target == "digest":
        release["harness_sha256"] = "bad"
        runner.write(frozen / "frozen_release.json", release)
    else:
        with paths[target].open("a") as stream:
            stream.write("\n# changed\n")
    with pytest.raises(RuntimeError, match="changed|digest"):
        runner.verify(frozen)


def test_claim_blocks_refreeze_and_retry(frozen):
    release = runner.verify(frozen)
    (frozen / "formal_attempt_claim.json").write_text("{}")
    with pytest.raises(RuntimeError, match="refreeze"):
        runner.freeze(
            frozen,
            Path(release["source_root"]),
            Path(release["recipe"]),
            task="general_pickup",
            layout=0,
            step_limit=200,
        )
    with pytest.raises(FileExistsError):
        runner.run(frozen, "unused")


def test_hashes_are_order_independent(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("a")
    b.write_text("b")
    assert sha256_sources([a, b]) == sha256_sources([b, a])


def test_native_success_is_only_success_source(tmp_path, monkeypatch):
    rpc = FakeRpc()
    monkeypatch.setattr(runner, "HttpRpcClient", lambda endpoint: rpc)
    assert runner.evaluate("unused", tmp_path) == {
        "official_success": False,
        "reward": 0.25,
    }
    assert rpc.calls == ["env.is_success", "env.get_reward_details", "client.close"]


def test_real_worker_exits_before_real_evaluator(frozen, monkeypatch):
    rpc = FakeRpc()
    events = []
    workers = []
    original_popen = subprocess.Popen

    def popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        if "-I" in args[0]:
            workers.append(process)
            assert "PYTHONPATH" not in kwargs["env"]
        return process

    def dispatch(method, args, kwargs, **options):
        if method in {"env.is_success", "env.get_reward_details"}:
            assert workers[0].poll() == 0
            assert workers[0].stdin.closed and workers[0].stdout.closed
            events.append(method)
        return rpc.call(method, args, kwargs)

    server = HttpRpcServer(("localhost", 0), dispatch)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(subprocess, "Popen", popen)
    try:
        result = runner.run(frozen, f"http://localhost:{server.server_port}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result == {
        "official_success": False,
        "reward": 0.25,
        "valid": True,
        "no_retry": True,
    }
    assert events == ["env.is_success", "env.get_reward_details"]
    audit = json.loads((frozen / "trial/audit.json").read_text())
    assert audit["action_channel_closed_before_evaluation"]
    assert audit["requests"] == 2


def test_cli_help_without_simulator():
    result = subprocess.run(
        [sys.executable, "-m", "robots.robodojo.scripted.eval", "--help"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "--source-root" in result.stdout
