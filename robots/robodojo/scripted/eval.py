# Copyright 2026 The RPent Authors.
"""Freeze and run one trusted scripted recipe against an exclusive dev server.

Use ``python -m robots.robodojo.scripted.eval freeze --output DIR
--source-root ROOT --recipe FILE --task TASK --layout N --step-limit N``;
then ``verify --output DIR`` or ``run --output DIR --endpoint URL``.

Recipes define ``main(env, output)`` and may use the embedded ``move_to`` and
``set_gripper`` primitives with their existing primitives-object interface.
``env.step`` supplies filtered observations/status and None reward/done slots.
Recipes must keep stdout reserved for the bridge protocol. No executable task
recipe is bundled; historical JSON plans are reference material only.
The server must be exclusively owned by the operator, not shared with other
clients. This harness does not authenticate the server's loaded source or act
as an OS sandbox. It does not start, stop, or reconfigure the borrowed server.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import selectors
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robots.robodojo.access import public_observation
from robots.robodojo.scripted.hashing import sha256_file, sha256_sources
from robots.robodojo.scripted.reward_audit import check_payload, strict_audit
from rpent.utils.logging import get_logger
from rpent.utils.rpc.http_rpc import HttpRpcClient, _NumpyEncoder

ROOT = Path(__file__).resolve().parents[3]
logger = get_logger(__name__)
PRIMITIVES = {
    "_arm_ee_pose_key",
    "_arm_ee_joint_key",
    "_refresh_obs",
    "move_to",
    "set_gripper",
}
SOURCE_DIRS = (
    "env",
    "src",
    "task",
    "env_cfg",
    "utils",
    "Assets/Robots/x5",
    "third_party/curobo/curobo",
)
SOURCE_SUFFIXES = {".py", ".json", ".yml", ".yaml", ".urdf", ".usd"}

# Worker transport has no endpoint, environment inheritance or repository imports.
HEADER = """from __future__ import annotations
import base64
import json
import sys
from pathlib import Path
import numpy as np

def encode(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)

def decode(obj):
    if isinstance(obj, dict):
        if '__ndarray__' in obj:
            return np.frombuffer(base64.b64decode(obj['__ndarray__']), dtype=obj['dtype']).reshape(obj['shape']).copy()
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list): return [decode(v) for v in obj]
    return obj

class Bridge:
    def call(self, method, *args):
        print(json.dumps(dict(method=method, args=args), default=encode), flush=True)
        return decode(json.loads(sys.stdin.readline()))['result']
    def get_obs(self): return self.call('get_obs')
    def get_status(self): return self.call('get_status')
    def solve_ik_position(self, arm, xyz): return self.call('solve_ik_position', arm, xyz)
    def step(self, action):
        result = self.call('step', action)
        return result['obs'], None, None, {'status': result['status']}
"""


def write(path: Path, obj: Any) -> None:
    """Write a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def generated_worker(recipe: Path) -> str:
    """Embed a recipe and unchanged current numerical primitives, without imports."""
    tree = ast.parse(recipe.read_text())
    if not any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body):
        raise ValueError("recipe must define main(env, output)")
    primitives = ast.parse((Path(__file__).with_name("primitives.py")).read_text())
    extracted = [
        n
        for n in primitives.body
        if isinstance(n, ast.FunctionDef) and n.name in PRIMITIVES
    ]
    if len(extracted) != len(PRIMITIVES):
        raise RuntimeError("numerical primitive inventory changed")
    # Future imports must precede the generated transport definitions.
    tree.body = [
        n
        for n in tree.body
        if not (isinstance(n, ast.ImportFrom) and n.module == "__future__")
    ]
    module = ast.Module(
        body=ast.parse(HEADER).body + extracted + tree.body, type_ignores=[]
    )
    return (
        ast.unparse(ast.fix_missing_locations(module))
        + "\n\nif __name__ == '__main__':\n    main(Bridge(), Path(sys.argv[1]))\n"
    )


def safe_status(status: dict) -> dict:
    """Expose budgets only; safety alarms in dev mode contain simulator truth."""
    return {key: status[key] for key in ("step", "step_limit")}


def safe_obs(obs: dict) -> dict:
    """Reuse RoboDojo's public RGB-D/proprioception allowlist."""
    return public_observation(obs)


def bridge_call(rpc: Any, request: dict, action_budget: int = 180) -> dict:
    """Dispatch only supported worker calls and strip privileged return values."""
    method, args = request["method"], request["args"]
    arity = {"get_obs": 0, "get_status": 0, "solve_ik_position": 2, "step": 1}
    if method not in arity:
        raise RuntimeError(f"RPC denied: {method}")
    if not isinstance(args, list) or len(args) != arity[method]:
        raise ValueError(f"invalid arguments for {method}")
    if method == "get_obs":
        result = safe_obs(rpc.call("env.get_obs"))
    elif method == "get_status":
        result = safe_status(rpc.call("env.get_status"))
    elif method == "solve_ik_position":
        ik = rpc.call(
            "env.solve_ik_position",
            kwargs={"arm": args[0], "xyz": args[1]},
            timeout_s=60,
        )
        result = {
            k: ik[k]
            for k in (
                "status",
                "error",
                "arm",
                "joint_value",
                "joint_displacement",
                "target_pose_wxyz",
                "quaternion_input_norm",
            )
            if k in ik
        }
    else:
        status = rpc.call("env.get_status")
        if status["step"] >= min(action_budget, status["step_limit"]):
            raise RuntimeError("outer action budget guard")
        obs, _reward, _done, info = rpc.call("env.step", args=(args[0],), timeout_s=60)
        result = {"obs": safe_obs(obs), "status": safe_status(info["status"])}
    check_payload(result)
    return result


def simulator_sources(source_root: Path) -> list[Path]:
    """Inventory simulator code/config and robot assets covered by the release."""
    return sorted(
        {
            p.resolve()
            for base in SOURCE_DIRS
            for p in (source_root / base).rglob("*")
            if p.is_file() and p.suffix in SOURCE_SUFFIXES
        }
    )


def freeze(
    output: Path,
    source_root: Path,
    recipe: Path,
    *,
    task: str,
    layout: int,
    step_limit: int,
    action_budget: int = 180,
) -> dict:
    """Freeze sources, dependencies, recipe and one fixed-layout attempt contract."""
    output, source_root, recipe = (
        output.resolve(),
        source_root.resolve(),
        recipe.resolve(),
    )
    if not task or layout < 0 or not 0 < action_budget <= step_limit:
        raise ValueError(
            "require a task, nonnegative layout and 0 < action budget <= step limit"
        )
    sim_paths = simulator_sources(source_root)
    if not source_root.is_dir() or not sim_paths:
        raise ValueError("source-root must contain simulator source files")
    if (output / "formal_attempt_claim.json").exists():
        raise RuntimeError("cannot refreeze after formal attempt claim")
    manifest = output / "frozen_release.json"
    if manifest.exists():
        write(
            output / "preflight_releases" / f"{sha256_file(manifest)}.json",
            json.loads(manifest.read_text()),
        )
    worker = output / "release/worker.py"
    worker.parent.mkdir(parents=True, exist_ok=True)
    worker.write_text(generated_worker(recipe))
    audit = strict_audit(worker)
    write(output / "audit_static.json", audit)
    if not audit["passed"]:
        raise RuntimeError("worker isolation gate failed")
    dependencies = output / "release/dependencies.json"
    write(
        dependencies,
        {p.metadata["Name"]: p.version for p in importlib.metadata.distributions()},
    )
    paths = [
        worker,
        recipe,
        dependencies,
        Path(sys.executable).resolve(),
        ROOT / "pyproject.toml",
    ]
    for directory in ("robots/robodojo", "rpent/utils/rpc", "rpent/robots/components"):
        paths.extend((ROOT / directory).rglob("*.py"))
    paths.extend(
        [
            ROOT / "rpent/utils/logging.py",
            ROOT / "tests/unit_tests/robots/robodojo/test_scripted_eval_contracts.py",
            *sim_paths,
        ]
    )
    paths = sorted({p.resolve() for p in paths})
    release = {
        "release_version": "robodojo-scripted-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": sys.version,
        "source_root": str(source_root),
        "simulator_sources": [str(p) for p in sim_paths],
        "recipe": str(recipe),
        "cell": {"task": task, "layout": layout, "step_limit": step_limit},
        "action_budget": action_budget,
        "max_attempts": 1,
        "evaluator": "robots.robodojo.scripted.eval",
        "official_success_field": "env.is_success()",
        "reward_boundary": {
            "endpoint_in_worker": False,
            "evaluator_process": "after worker exit and pipe closure",
        },
        "source_sha256": {str(p): sha256_file(p) for p in paths},
        "harness_sha256": sha256_sources(paths),
    }
    write(manifest, release)
    return release


def verify(output: Path) -> dict:
    """Verify source contents, simulator inventory and worker audit before use."""
    release = json.loads((output / "frozen_release.json").read_text())
    for path, expected in release["source_sha256"].items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"frozen source changed: {path}")
    if sha256_sources(release["source_sha256"]) != release["harness_sha256"]:
        raise RuntimeError("frozen harness digest mismatch")
    if [str(p) for p in simulator_sources(Path(release["source_root"]))] != release[
        "simulator_sources"
    ]:
        raise RuntimeError("simulator source inventory changed")
    if not strict_audit(output / "release/worker.py")["passed"]:
        raise RuntimeError("isolation audit failed")
    return release


def evaluate(endpoint: str, output: Path) -> dict:
    """Read native success and reward once in the final evaluator process."""
    rpc = HttpRpcClient(endpoint)
    try:
        official = bool(rpc.call("env.is_success"))
        details = rpc.call("env.get_reward_details")
        result = {"official_success": official, "reward": details["reward"]}
        write(output / "external_evaluation.json", result)
        return result
    finally:
        rpc.close()


def run_worker(output: Path, rpc: Any, action_budget: int) -> dict:
    """Serve a bounded worker and close all action pipes before returning."""
    cell = output / "trial"
    agent = cell / "agent"
    agent.mkdir()
    error, requests = None, 0
    env = {
        "PATH": os.environ.get("PATH", ""),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    with (
        (cell / "worker.stderr.log").open("w") as err,
        (cell / "bridge_audit.jsonl").open("w") as log,
    ):
        worker = subprocess.Popen(
            [sys.executable, "-I", str(output / "release/worker.py"), str(agent)],
            cwd=agent,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
        )
        selector = selectors.DefaultSelector()
        selector.register(worker.stdout, selectors.EVENT_READ)
        os.set_blocking(worker.stdout.fileno(), False)
        pending = b""
        deadline = time.monotonic() + 540
        try:
            while time.monotonic() < deadline:
                if not selector.select(timeout=1):
                    continue
                chunk = os.read(worker.stdout.fileno(), 65536)
                if not chunk:
                    if pending:
                        raise RuntimeError("incomplete worker request")
                    break
                pending += chunk
                if len(pending) > 1024 * 1024:
                    raise RuntimeError("worker request too large")
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    request = json.loads(line)
                    result = bridge_call(rpc, request, action_budget)
                    encoded = json.dumps({"result": result}, cls=_NumpyEncoder).encode()
                    requests += 1
                    log.write(
                        json.dumps(
                            {
                                "index": requests,
                                "method": request["method"],
                                "delivered_sha256": hashlib.sha256(encoded).hexdigest(),
                            }
                        )
                        + "\n"
                    )
                    log.flush()
                    worker.stdin.write(encoded + b"\n")
                    worker.stdin.flush()
            else:
                raise TimeoutError("formal worker deadline")
            worker.wait(timeout=10)
        except Exception as exc:
            error = str(exc)
        finally:
            selector.close()
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
            worker.stdin.close()
            worker.stdout.close()
    return {
        "worker_returncode": worker.returncode,
        "error": error,
        "requests": requests,
        "action_channel_closed_before_evaluation": True,
    }


def run(output: Path, endpoint: str) -> dict:
    """Run one claimed attempt, close the action channel, then spawn evaluation."""
    output = output.resolve()
    release = verify(output)
    manifest_hash = sha256_file(output / "frozen_release.json")
    with (output / "formal_attempt_claim.json").open("x") as stream:
        json.dump(
            {
                "start": datetime.now(timezone.utc).isoformat(),
                "release_sha256": manifest_hash,
            },
            stream,
        )
    rpc = HttpRpcClient(endpoint)
    cell = output / "trial"
    try:
        meta = rpc.call("env.get_env_meta")
        expected = release["cell"]
        if (
            meta["task"] != expected["task"]
            or meta["layout"] != expected["layout"]
            or meta["random"]
            or meta.get("mode", "dev") != "dev"
        ):
            raise RuntimeError("requires matching fixed-layout dev environment")
        status = rpc.call("env.get_status")
        if status["step"] != 0 or status["step_limit"] != expected["step_limit"]:
            raise RuntimeError("fresh episode with frozen step limit required")
        rpc.call("env.reset", timeout_s=60)
        cell.mkdir(exist_ok=True)
        write(cell / "release.json", release)
        audit = run_worker(output, rpc, release["action_budget"])
    finally:
        rpc.close()
    try:
        verify(output)
        if sha256_file(output / "frozen_release.json") != manifest_hash:
            raise RuntimeError("release manifest changed during run")
        audit["frozen_unchanged"] = True
    except Exception as exc:
        audit.update(frozen_unchanged=False, error=str(exc))
    write(cell / "audit.json", audit)
    # Worker has exited and both pipes are closed; do not close the server.
    evaluated = subprocess.run(
        [
            sys.executable,
            "-m",
            "robots.robodojo.scripted.eval",
            "evaluate",
            "--endpoint",
            endpoint,
            "--output",
            str(cell),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
        timeout=60,
    )
    (cell / "evaluator.stdout.log").write_text(evaluated.stdout)
    (cell / "evaluator.stderr.log").write_text(evaluated.stderr)
    if evaluated.returncode:
        raise RuntimeError("outer evaluator failed; see evaluator.stderr.log")
    result = json.loads((cell / "external_evaluation.json").read_text())
    result.update(
        valid=audit["frozen_unchanged"]
        and audit["worker_returncode"] == 0
        and audit["error"] is None,
        no_retry=True,
    )
    write(cell / "result.json", result)
    return result


def main() -> None:
    """Dispatch freeze, verify, run, or the runner's evaluator subprocess."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify", "run", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--layout", type=int)
    parser.add_argument("--step-limit", type=int)
    parser.add_argument("--action-budget", type=int, default=180)
    args = parser.parse_args()
    if args.command == "freeze":
        if any(
            getattr(args, key) is None
            for key in ("source_root", "recipe", "task", "layout", "step_limit")
        ):
            parser.error(
                "freeze requires --source-root, --recipe, --task, --layout and --step-limit"
            )
        result = freeze(
            args.output,
            args.source_root,
            args.recipe,
            task=args.task,
            layout=args.layout,
            step_limit=args.step_limit,
            action_budget=args.action_budget,
        )
    elif args.command == "verify":
        result = verify(args.output)
    else:
        if not args.endpoint:
            parser.error("run/evaluate requires --endpoint")
        if args.command == "run":
            result = run(args.output, args.endpoint)
        else:
            result = evaluate(args.endpoint, args.output)
    logger.info(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
