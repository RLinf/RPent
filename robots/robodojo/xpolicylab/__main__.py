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

"""Launch RPent through RoboDojo's official XPolicyLab evaluator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from rpent.evaluation import write_json_atomic
from rpent.utils.config import get_repo_root
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger(__name__)


def prepare(args: argparse.Namespace) -> tuple[dict, dict[str, str], Path]:
    """Register the policy and freeze inputs without changing task configuration."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required on PATH for official episode videos")
    root, output = args.runtime_root.resolve(), args.output_dir.resolve()
    layouts = selected_layouts(root, args.task, args.layouts, args.episodes)
    repo = get_repo_root()
    observations = yaml.safe_load((root / "env_cfg/arx_x5.yml").read_text())[
        "observation"
    ]
    for field in ("depth", "intrinsic_matrix", "extrinsic_matrix"):
        if not observations["vision"].get(field):
            raise ValueError(f"Enable observation.vision.{field} in env_cfg/arx_x5.yml")
    if observations["vision"].get("approximate_depth", True):
        raise ValueError(
            "Set observation.vision.approximate_depth=false for metric depth"
        )
    cameras = yaml.safe_load((root / "env_cfg/camera/camera_config.yml").read_text())[
        "annotator"
    ]
    for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        capture = cameras[camera]
        if not capture.get("enabled") or not any(
            isinstance(value, dict)
            and value.get("type") == "distance_to_image_plane"
            and value.get("device") == "cpu"
            for value in capture.values()
        ):
            raise ValueError(f"Enable CPU distance_to_image_plane capture for {camera}")
    if not args.memory_dir.is_dir():
        raise ValueError(f"Memory directory not found: {args.memory_dir}")
    package = Path(__file__).resolve().parent
    link = root / "XPolicyLab/policy/RPent"
    try:
        link.symlink_to(package, target_is_directory=True)
    except FileExistsError:
        if link.resolve() != package:
            raise FileExistsError(
                f"Another RPent adapter already exists at {link}"
            ) from None
    output.mkdir(parents=True, exist_ok=False)
    init_output_dir(output)
    config = yaml.safe_load((package / "deploy.yml").read_text())
    config.update(
        port=pick_free_port(),
        output_dir=str(output),
        memory_dir=str(args.memory_dir.resolve()),
        am_endpoint=args.am_endpoint,
        sam3_endpoint=args.sam3_endpoint,
        planner=args.planner,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout=args.timeout,
    )
    (output / "policy.yml").write_text(yaml.safe_dump(config))
    sources = [
        repo / "robots/robodojo/prompts",
        repo / "robots/robodojo/guides",
        args.memory_dir,
    ]
    hashes = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
        for source in sources
        for p in sorted(source.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    names = {
        d.metadata["Name"]
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }
    write_json_atomic(
        output / "dependencies.json",
        {
            "python": sys.version,
            "packages": {
                name: importlib.metadata.version(name) for name in sorted(names)
            },
        },
    )
    revisions = {}
    for name, checkout in (
        ("rpent", repo),
        ("robodojo", root),
        ("xpolicylab", root / "XPolicyLab"),
    ):
        revisions[name] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip()
        (output / f"{name}.patch").write_bytes(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=checkout)
        )
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_root = (
        root
        / "eval_result/RoboDojo"
        / args.task
        / "RPent/arx_x5"
        / f"{args.seed}_{output.name}"
    )
    write_json_atomic(
        output / "manifest.json",
        {
            "revisions": revisions,
            "sha256": hashes,
            "task": args.task,
            "seed": args.seed,
            "episodes": len(layouts),
            "layouts": layouts,
            "run_id": run_id,
            "result_dir": str(result_root / run_id),
            "env_cuda_device": args.env_cuda_device,
            "policy_cuda_device": args.policy_cuda_device,
        },
    )
    env = dict(
        os.environ,
        RPENT_REPO_ROOT=str(repo),
        ROBODOJO_RUN_ID=run_id,
        EVAL_NUM=str(len(layouts)),
        RPENT_XPL_LAYOUTS=json.dumps(layouts),
        PYTHONUNBUFFERED="1",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo), str(root), str(root / "XPolicyLab"), env.get("PYTHONPATH", "")]
    )
    return config, env, result_root / run_id


def run(args: argparse.Namespace) -> None:
    """Own the policy/evaluator processes, borrowing existing VPP2 and SAM3 services."""
    config, env, result_dir = prepare(args)
    output, root = args.output_dir.resolve(), args.runtime_root.resolve()
    # Import the official transport only after its checkout is on the path.
    sys.path[:0] = [str(root), str(root / "XPolicyLab")]
    from client_server.ws import WsModelClient

    policy = ProcessDaemon(
        "xpolicylab-policy",
        [
            sys.executable,
            "-u",
            "-m",
            "client_server.ws",
            "--config-path",
            str(output / "policy.yml"),
        ],
        cwd=str(root),
        env_overrides=dict(env, CUDA_VISIBLE_DEVICES=args.policy_cuda_device),
        log_path=str(output / "policy.log"),
    )
    evaluator = ProcessDaemon(
        "xpolicylab-evaluator",
        [
            sys.executable,
            "-u",
            "-m",
            "robots.robodojo.xpolicylab.evaluator",
            "--task_name",
            args.task,
            "--num_envs",
            "1",
            "--env_cfg_type",
            "arx_x5",
            "--device_id",
            "0",
            "--policy_name",
            "RPent",
            "--port",
            str(config["port"]),
            "--additional_info",
            output.name,
            "--seed",
            str(args.seed),
            "--headless",
            "--enable_cameras",
        ],
        cwd=str(root),
        env_overrides=dict(env, CUDA_VISIBLE_DEVICES=args.env_cuda_device),
        log_path=str(output / "environment.log"),
    )
    # This integration loads the official checkout, not an RLinf environment.
    policy.subprocess_env["PYTHONPATH"] = env["PYTHONPATH"]
    evaluator.subprocess_env["PYTHONPATH"] = env["PYTHONPATH"]
    try:
        policy.start()
        deadline = time.monotonic() + 120
        while True:
            client = None
            try:
                client = WsModelClient(
                    url=f"ws://127.0.0.1:{config['port']}",
                    evaluation_id="rpent-startup",
                    trial_id="health",
                    connect_timeout_s=2,
                    request_timeout_s=5,
                    max_connect_attempts=1,
                )
                client.call(func_name="episode_report")
                break
            except Exception:
                if policy.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Policy server did not start; see {output / 'policy.log'}"
                    ) from None
                time.sleep(1)
            finally:
                if client is not None:
                    client.close()
        evaluator.start()
        while evaluator.poll() is None:
            if policy.poll() is not None:
                raise RuntimeError("Policy server exited during evaluation")
            time.sleep(1)
        if evaluator.poll() != 0:
            raise RuntimeError(f"Evaluation stopped; see {output / 'environment.log'}")
        verify_results(root, result_dir, output, json.loads(env["RPENT_XPL_LAYOUTS"]))
        if not os.environ.get("RPENT_XPL_BATCH"):
            with (output / "summarize.log").open("w") as summary_log:
                subprocess.run(
                    ["bash", "scripts/robodojo.sh", "summarize"],
                    cwd=root,
                    env=env,
                    check=True,
                    stdout=summary_log,
                    stderr=subprocess.STDOUT,
                )
        logger.info("Official results and videos: %s", result_dir)
    finally:
        evaluator.stop()
        policy.stop()


def episode_count(root: Path, task: str, requested: int | None) -> int:
    """Match the official per-task evaluation count and EVAL_NUM clipping."""
    cfg = yaml.safe_load((root / "task/RoboDojo/config/_task.yml").read_text())
    native = int(
        cfg["tasks"].get(task, {}).get("eval_nums", cfg["common"]["eval_nums"])
    )
    return min(native, requested) if requested is not None else native


def selected_layouts(
    root: Path, task: str, selection: str | None, episodes: int | None = None
) -> list[int]:
    """Resolve explicit layout IDs within the official evaluation range."""
    native = episode_count(root, task, None)
    if selection in (None, "all"):
        return list(range(episode_count(root, task, episodes)))
    layouts = []
    for part in selection.split(","):
        bounds = part.split("-")
        if len(bounds) not in (1, 2) or not all(b.isdigit() for b in bounds):
            raise ValueError("Use --layouts all, 0-9, or 2,7,9")
        first, last = int(bounds[0]), int(bounds[-1])
        if not 0 <= first <= last < native:
            raise ValueError(f"{task}: layout range must be within 0–{native - 1}")
        layouts.extend(range(first, last + 1))
    if len(layouts) != len(set(layouts)):
        raise ValueError("Duplicate layout IDs")
    return sorted(layouts)


def verify_results(
    root: Path, result_dir: Path, output: Path, expected: list[int] | None = None
) -> None:
    """Check native results and video completeness using the official parser."""
    spec = importlib.util.spec_from_file_location(
        "robodojo_summary", root / "scripts/internal/summarize_result.py"
    )
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    entries = summary.load_entries(str(result_dir))
    if not entries:
        errors = list((result_dir / "rpent").glob("*/infrastructure-error.json"))
        if errors:
            error = json.loads(errors[0].read_text())["error"]
            raise RuntimeError(f"Official evaluation failed: {error}; see {errors[0]}")
        raise RuntimeError(f"Official results missing or empty: {result_dir}")
    cameras, _ = summary.video_inventory(str(result_dir))
    details = json.loads((result_dir / "_result.json").read_text())["details"]
    if expected is not None and (
        len(entries) != len(expected)
        or {entry["layout_id"] for entry in details.values()} != set(expected)
    ):
        raise RuntimeError(f"Expected layouts {expected}; results are incomplete")
    reports = {}
    for path in (output / "episodes").glob("*/report.json"):
        report = json.loads(path.read_text())
        reports[report["episode"]["layout"]] = report
    for record, entry in details.items():
        layout = entry["layout_id"]
        report = reports.get(layout)
        if report is None or report["status"] != "completed":
            raise RuntimeError(f"Missing completed planner report for layout {layout}")
        if cameras.get(str(int(record))) != summary.EXPECTED_CAMERAS:
            raise RuntimeError(f"Missing official camera videos for record {record}")
    write_json_atomic(
        output / "result-check.json",
        {
            "episodes": len(entries),
            "successes": sum(success for success, _ in entries),
            "result_dir": str(result_dir),
            "videos_complete": True,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    """Define shared single-task and batch evaluation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        required=True,
        help="Official RoboDojo checkout with XPolicyLab and assets",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task")
    selection.add_argument(
        "--dimensions",
        nargs="+",
        choices=("generalization", "memory", "long-horizon", "precision"),
        help="Use the official inventory for the selected capability groups",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Limit a batch to named tasks within the selected dimensions",
    )
    parser.add_argument(
        "--worker",
        action="append",
        metavar="ENV_GPU:POLICY_GPU",
        help="Batch worker GPU pair; repeat for parallel workers",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the batch plan without starting processes",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume unstarted tasks with unchanged inputs",
    )
    parser.add_argument("--seed", type=int, default=1)
    layouts = parser.add_mutually_exclusive_group()
    layouts.add_argument(
        "--layouts", help="Layout IDs: all, 0-49, or 2,7,9 (default: all)"
    )
    layouts.add_argument(
        "--episodes",
        type=int,
        help="Run the first N layouts; omitted uses official task count",
    )
    parser.add_argument(
        "--am-endpoint", required=True, help="Existing RPent VPP2 RPC endpoint"
    )
    parser.add_argument("--sam3-endpoint", help="Existing RPent SAM3 RPC endpoint")
    parser.add_argument(
        "--memory-dir", type=Path, default=get_repo_root() / "memory/robodojo"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for frozen inputs and planner traces",
    )
    parser.add_argument("--planner", default="codex")
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout", type=int, default=4800)
    parser.add_argument("--env-cuda-device", default="0")
    parser.add_argument(
        "--policy-cuda-device", default="0", help="GPU for policy-side cuRobo"
    )
    return parser


def main() -> None:
    """Launch a single task or a frozen multi-task batch."""
    parser = build_parser()
    args = parser.parse_args()
    if (
        args.seed < 0
        or args.timeout <= 0
        or (args.episodes is not None and args.episodes <= 0)
    ):
        parser.error("seed must be nonnegative; timeout and episodes must be positive")

    def interrupt_run(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt_run)
    signal.signal(signal.SIGTERM, interrupt_run)
    if args.dimensions:
        from robots.robodojo.xpolicylab.batch import run_batch

        run_batch(args)
    else:
        if args.worker or args.tasks or args.dry_run or args.resume:
            parser.error(
                "--tasks, --worker, --dry-run and --resume require --dimensions"
            )
        run(args)


if __name__ == "__main__":
    main()
