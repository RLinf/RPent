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

"""Fixed-observation checklist experiment: collect, infer and independently evaluate.

Run ``python -m toolkits.geometry.evaluation.localization_experiment --help``. Collection needs
only ENV; inference needs only SAM3 and the explicit visual grounding model.
Neither stage starts a planner/action model or executes a manipulation action.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rpent.session import EnvState
from rpent.utils.logging import get_logger
from toolkits.geometry.evaluation.localization_checklist import (
    build_manifest,
    frame_id,
    load_manifest,
    manifest_digest,
    select_cases,
)

logger = get_logger("localization_experiment")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def _fresh_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output must be a new or empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_header(
    manifest: dict[str, Any], cases: list[dict[str, Any]], stage: str
) -> dict[str, Any]:
    from dataclasses import asdict

    from toolkits.geometry.geometry import FitOptions

    geometry_dir = Path(__file__).resolve().parents[1]
    evaluation_dir = Path(__file__).resolve().parent
    sources = [
        geometry_dir / "geometry.py",
        geometry_dir / "reference_geometry.py",
        geometry_dir / "candidate_geometry.py",
        geometry_dir / "geometry_agent.py",
        evaluation_dir / "localization_checklist.py",
        evaluation_dir / "localization_experiment.py",
        evaluation_dir / "localization_evaluation.py",
    ]
    return {
        "schema_version": 1,
        "stage": stage,
        "manifest_sha256": manifest_digest(manifest),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "planned_cases": 80,
        "planned_requests": 160,
        "selected_cases": [case["case_id"] for case in cases],
        "selected_requests": 2 * len(cases),
        "protocol": manifest["protocol"],
        "fit_options": asdict(FitOptions()),
        "source_sha256": {path.name: _hash(path) for path in sources},
    }


def _runtime_args(args: argparse.Namespace, case: dict[str, Any]) -> argparse.Namespace:
    from robots.libero.robot_spec import get_robot_spec

    defaults = argparse.ArgumentParser(add_help=False)
    get_robot_spec().add_cli_args(defaults, use_dashboard=True)
    runtime = defaults.parse_args([])
    for key in (
        "env_endpoint",
        "sam3_endpoint",
        "cuda_device",
        "env_cuda_device",
        "sam3_cuda_device",
        "max_episode_steps",
        "evaluation_references",
    ):
        if hasattr(args, key):
            setattr(runtime, key, getattr(args, key))
    runtime.suite, runtime.task, runtime.seed = (
        case["suite"],
        case["task"],
        case["seed"],
    )
    runtime.libero_type = "pro"
    return runtime


def _start_runtime(
    args: argparse.Namespace, case: dict[str, Any], output: Path, components: set[str]
) -> tuple[list, dict[str, Any]]:
    from robots.libero.robot_spec import get_robot_spec
    from rpent.dashboard.events import NullDashboardEventSink

    output.mkdir(parents=True, exist_ok=True)
    return get_robot_spec().init_runtime(
        _runtime_args(args, case), output, NullDashboardEventSink(), components
    )


def _stop_owned(daemons: list) -> None:
    for daemon in reversed(daemons):
        try:
            daemon.stop()
        except Exception:
            logger.exception("failed to stop owned localization component")


def _save_native_high(env: Any, state: EnvState) -> None:
    """Preserve paired native RGB-D at float32 precision in experiment snapshots."""
    from robots.libero.tools import _metric_depth, _world_from_depth

    rgb, raw_depth = env.render_camera("agentview", height=1024, width=1024, depth=True)
    meta = env.get_camera_meta("agentview", height=1024, width=1024)
    if not meta:
        raise ValueError("native agentview camera metadata missing")
    rgb = np.asarray(rgb)[::-1]
    depth = _metric_depth(raw_depth, meta)[::-1]
    if rgb.shape != (1024, 1024, 3) or depth.shape != (1024, 1024):
        raise ValueError("capture requires native matching 1024x1024 agentview RGB-D")
    world = _world_from_depth(depth, meta).astype(np.float32)
    valid = (
        np.isfinite(world).all(axis=2)
        & np.isfinite(depth)
        & (depth > 0)
        & (depth <= 10)
    )
    world[~valid] = np.nan
    for name, value in {
        "agentview_high.png": rgb,
        "agentview_depth_high.npz": depth,
        "agentview_world_high.npz": world,
        "agentview_valid_high.npz": valid,
        "agentview_metadata_high.json": meta,
    }.items():
        if state.save(name, value, step=0) is None:
            raise RuntimeError(f"failed to save native observation: {name}")


def collect(
    args: argparse.Namespace, manifest: dict[str, Any], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Capture step zero for each fixed seed without any policy or perception model."""
    from robots.libero.tools import LiberoPrimitives, dump_state

    if args.env_endpoint and len(cases) != 1:
        raise ValueError(
            "an ENV endpoint is pinned to one suite/task/seed; select exactly one --case"
        )
    _fresh_output(args.output)
    header = _run_header(manifest, cases, "collect")
    _write_json(args.output / "run_manifest.json", header)
    rows = []
    for case in cases:
        case_id = case["case_id"]
        output = args.output / case_id
        daemons = []
        row = {"case_id": case_id, "status": "error"}
        try:
            daemons, runtime = _start_runtime(args, case, output / "runtime", {"env"})
            primitive = LiberoPrimitives(
                env=runtime["env"],
                model=None,
                sam3_client=None,
                check_cancelled=lambda: None,
            )
            primitive.reset()
            state = EnvState(output / "episode")
            dump_state(primitive, state)
            _save_native_high(runtime["env"], state)
            if getattr(args, "evaluation_references", False):
                snapshot = runtime["env"]._client.call(
                    "evaluation.reference_snapshot", timeout_s=120
                )
                snapshot.update(
                    case_id=case_id,
                    frame_id=frame_id(case_id),
                    step=0,
                    camera="agentview",
                )
                _write_json(output / "evaluator_snapshot.json", snapshot)
            artifacts = {
                name: _hash(state.artifact_path(name, step=0))
                for name in state.get(0).artifacts
            }
            _write_json(
                output / "observation.json",
                {
                    "schema_version": 1,
                    "manifest_sha256": manifest_digest(manifest),
                    "case_id": case_id,
                    "frame_id": frame_id(case_id),
                    "suite": case["suite"],
                    "task": case["task"],
                    "seed": case["seed"],
                    "step": 0,
                    "camera": "agentview",
                    "units": "m",
                    "artifact_sha256": artifacts,
                },
            )
            row["status"] = "captured"
        except Exception as exc:
            row["error"] = str(exc)
            logger.exception("capture failed: %s", case_id)
        finally:
            _stop_owned(daemons)
        rows.append(row)
        _write_json(args.output / "collection.json", rows)
        logger.info("capture %s: %s", case_id, row["status"])
    summary = {
        "selected_cases": len(cases),
        "captured": sum(row["status"] == "captured" for row in rows),
        "failed": sum(row["status"] != "captured" for row in rows),
    }
    _write_json(args.output / "summary.json", summary)
    return summary


def copy_observation(
    source: Path, destination: Path, manifest: dict[str, Any], case: dict[str, Any]
) -> tuple[EnvState, dict[str, str]]:
    """Copy only sensor artifacts to a fresh agent state; never copy annotations/state extras."""
    if not source.is_dir():
        raise ValueError(f"input observation missing: {source}")
    identity = json.loads((source / "observation.json").read_text())
    case_id = case["case_id"]
    for key, value in {
        "manifest_sha256": manifest_digest(manifest),
        "case_id": case_id,
        "frame_id": frame_id(case_id),
        "step": 0,
        "camera": "agentview",
        "units": "m",
    }.items():
        if identity.get(key) != value:
            raise ValueError(f"observation {key} mismatch: {case_id}")
    records = json.loads((source / "episode" / "states.json").read_text())["steps"]
    if len(records) != 1 or records[0]["step_idx"] != 0:
        raise ValueError("checklist requires exactly the initial step-zero observation")
    allowed = {
        f"agentview{suffix}"
        for suffix in (
            ".png",
            "_world.npz",
            "_depth.npz",
            "_metadata.json",
            "_high.png",
            "_world_high.npz",
            "_depth_high.npz",
            "_metadata_high.json",
            "_valid_high.npz",
        )
    }
    names = allowed.intersection(records[0]["artifacts"])
    if (
        not {
            "agentview_high.png",
            "agentview_world_high.npz",
            "agentview_depth_high.npz",
            "agentview_metadata_high.json",
        }
        <= names
    ):
        raise ValueError(
            "native high-resolution RGB-D/world/calibration incomplete; recollect this case"
        )
    original = EnvState(source / "episode")
    hashes = {}
    for name in sorted(names):
        digest = _hash(original.artifact_path(name, step=0))
        if identity["artifact_sha256"].get(name) != digest:
            raise ValueError(f"observation artifact changed after collection: {name}")
        hashes[name] = digest
    state = EnvState(destination)
    with state.record_step(state={}):
        for name in sorted(names):
            value = original.load(name, step=0)
            if name.endswith(".json"):
                value = {
                    key: val
                    for key, val in value.items()
                    if key
                    in {
                        "intrinsic_K",
                        "extrinsic_cam2world",
                        "depth_near",
                        "depth_far",
                        "height",
                        "width",
                    }
                }
            if state.save(name, value) is None:
                raise RuntimeError(f"failed to copy sensor artifact: {name}")
    return state, hashes


def _prediction_base(
    manifest: dict[str, Any], case: dict[str, Any], request: dict[str, Any]
) -> dict[str, Any]:
    reference = manifest["references"][request["reference"]]
    return {
        **request,
        "case_id": case["case_id"],
        "suite": case["suite"],
        "task": case["task"],
        "seed": case["seed"],
        "query": reference["query"],
        "quantity": reference["quantity"],
        "definition": reference["definition"],
        "frame_id": frame_id(case["case_id"]),
        "manifest_sha256": manifest_digest(manifest),
        "camera": "agentview",
        "step": 0,
        "units": "m",
        "status": "error",
        "world_xyz": None,
    }


def infer_cases(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
    sam3_client: Any,
    *,
    grounder: Any = None,
) -> list[dict[str, Any]]:
    """Run both fixed references per case; one failure never skips the other target."""
    from robots.libero.tools import LiberoPrimitives
    from toolkits.geometry.geometry_agent import GeometryAgent

    predictions = []
    for case in cases:
        output = args.output / case["case_id"]
        try:
            state, hashes = copy_observation(
                args.inputs / case["case_id"], output / "episode", manifest, case
            )
            primitive = LiberoPrimitives(
                env=None,
                model=None,
                sam3_client=sam3_client,
                check_cancelled=lambda: None,
            )
            agent = GeometryAgent(
                state=state,
                segment=primitive.segment,
                check_cancelled=lambda: None,
                model=args.geometry_model,
                base_url=args.geometry_base_url,
                grounder=grounder,
                permissive=getattr(args, "permissive", False),
            )
            error = None
        except Exception as exc:
            error, hashes = str(exc), {}
            logger.exception("observation unavailable: %s", case["case_id"])
        for request in case["requests"]:
            row = _prediction_base(manifest, case, request)
            if error is not None:
                row["reason"] = error
            else:
                try:
                    data = agent.localize(
                        row["query"],
                        camera="agentview",
                        step=0,
                        required_geometry=[row["quantity"]],
                    ).data
                    field = data.get("geometry", {}).get(row["quantity"], {})
                    # A material point or region median is never a reference-center fallback.
                    row.update(
                        status=data["status"],
                        world_xyz=field.get("value"),
                        reason=field.get("reason")
                        or data.get("error")
                        or data.get("reason"),
                        calls=data.get("calls"),
                        elapsed_ms=data.get("elapsed_ms"),
                        geometry_artifact=data.get("geometry_artifact"),
                        artifact_sha256=hashes,
                        permissive=getattr(args, "permissive", False),
                        strict_accepted=field.get("strict_accepted"),
                        estimate_method=field.get("estimate_method"),
                        strict_rejection=field.get("strict_rejection"),
                    )
                    _write_json(output / f"result_{request['slot']}.json", data)
                except Exception as exc:
                    row["reason"] = str(exc)
                    logger.exception("localization failed: %s", row["request_id"])
            predictions.append(row)
            with (args.output / "results.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
            logger.info("localization %s: %s", row["request_id"], row["status"])
    return predictions


def infer(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
    *,
    grounder: Any = None,
) -> dict[str, Any]:
    """Start only SAM3, replay immutable captures and persist every fixed request."""
    model = args.geometry_model.strip() if args.geometry_model else ""
    if not model or (
        model != "codex" and (":" not in model or not model.split(":", 1)[1].strip())
    ):
        raise ValueError("--geometry-model must be provider:model or codex[:model]")
    args.geometry_model = model
    if not args.inputs.is_dir():
        raise ValueError(f"input collection missing: {args.inputs}")
    _fresh_output(args.output)
    header = _run_header(manifest, cases, "infer")
    # Provider credentials/URLs are deliberately excluded from the run manifest.
    header["geometry_model"] = args.geometry_model
    header["permissive"] = getattr(args, "permissive", False)
    _write_json(args.output / "run_manifest.json", header)
    daemons = []
    try:
        daemons, runtime = _start_runtime(
            args, cases[0], args.output / "runtime", {"sam3"}
        )
        if grounder is None:
            predictions = infer_cases(args, manifest, cases, runtime["sam3_client"])
        else:
            predictions = infer_cases(
                args, manifest, cases, runtime["sam3_client"], grounder=grounder
            )
    finally:
        _stop_owned(daemons)
    latencies = [
        row["elapsed_ms"] for row in predictions if row.get("elapsed_ms") is not None
    ]
    summary = {
        "selected_requests": 2 * len(cases),
        "recorded": len(predictions),
        "status_counts": dict(Counter(row["status"] for row in predictions)),
        "returned_centers": sum(row["world_xyz"] is not None for row in predictions),
        "calls": {
            name: sum((row.get("calls") or {}).get(name, 0) for row in predictions)
            for name in ("grounding", "segment", "back_project")
        },
        "latency_ms": {
            "mean": float(np.mean(latencies)) if latencies else None,
            "p95": float(np.quantile(latencies, 0.95)) if latencies else None,
        },
        "accuracy": None,
    }
    _write_json(args.output / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in (
        "write-manifest",
        "validate-manifest",
        "collect",
        "infer",
        "truth-template",
        "evaluate",
    ):
        sub = commands.add_parser(command)
        sub.add_argument(
            "--manifest",
            type=Path,
            required=command == "write-manifest",
            help="Fixed checklist JSON; defaults to the built-in checklist except when exporting it.",
        )
        if command in ("collect", "infer", "evaluate"):
            sub.add_argument(
                "--case",
                action="append",
                dest="case_ids",
                help="Fixed case ID, repeatable (default all 80).",
            )
            sub.add_argument(
                "--limit", type=int, help="First N selected cases for a pilot run."
            )
        if command in ("collect", "infer"):
            sub.add_argument(
                "--output",
                type=Path,
                required=True,
                help="New/empty experiment directory.",
            )
            sub.add_argument("--cuda-device", type=int)
        if command == "collect":
            sub.add_argument(
                "--with-references",
                dest="evaluation_references",
                action="store_true",
                help="Export compiled geometry and body poses for the independent evaluator only.",
            )
            sub.add_argument(
                "--env-endpoint",
                help="Borrow an ENV pinned to exactly one selected case.",
            )
            sub.add_argument("--env-cuda-device", type=int)
            sub.add_argument("--max-episode-steps", type=int, default=10000)
        elif command == "infer":
            sub.add_argument(
                "--inputs", type=Path, required=True, help="Output of collect."
            )
            sub.add_argument("--geometry-model", required=True)
            sub.add_argument("--geometry-base-url")
            sub.add_argument(
                "--permissive",
                action="store_true",
                help="Return observation-only approximate reference centers for error evaluation when strict fitting rejects them.",
            )
            sub.add_argument(
                "--sam3-endpoint",
                help="Borrow an existing SAM3 service; otherwise start one local service.",
            )
            sub.add_argument("--sam3-cuda-device", type=int)
        elif command == "truth-template":
            sub.add_argument(
                "--output", type=Path, required=True, help="New annotation JSON file."
            )
        elif command == "evaluate":
            sub.add_argument("--truth", type=Path, required=True)
            sub.add_argument(
                "--predictions",
                type=Path,
                required=True,
                help="Inference results.jsonl.",
            )
            sub.add_argument(
                "--output",
                type=Path,
                required=True,
                help="New/empty evaluation directory.",
            )
            sub.add_argument("--tolerance-m", type=float, default=0.01)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one stage; never import simulator/model servers for help or evaluation."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "write-manifest":
            _write_json(args.manifest, build_manifest())
            logger.info("wrote checklist: %s", args.manifest)
            return 0
        manifest = (
            load_manifest(args.manifest)
            if args.manifest is not None
            else build_manifest()
        )
        cases = select_cases(
            manifest, getattr(args, "case_ids", None), getattr(args, "limit", None)
        )
        if args.command == "validate-manifest":
            logger.info("checklist valid: 16 task rows, 80 cases, 160 requests")
        elif args.command == "collect":
            summary = collect(args, manifest, cases)
            return int(summary["failed"] > 0)
        elif args.command == "infer":
            summary = infer(args, manifest, cases)
            return int(summary["status_counts"].get("error", 0) > 0)
        elif args.command == "truth-template":
            from toolkits.geometry.evaluation.localization_evaluation import (
                truth_template,
            )

            if args.output.exists():
                raise ValueError(f"annotation file already exists: {args.output}")
            _write_json(args.output, truth_template(manifest))
        elif args.command == "evaluate":
            from toolkits.geometry.evaluation.localization_evaluation import evaluate

            predictions = [
                json.loads(line)
                for line in args.predictions.read_text().splitlines()
                if line.strip()
            ]
            report = evaluate(
                manifest,
                json.loads(args.truth.read_text()),
                predictions,
                tolerance_m=args.tolerance_m,
                cases=cases,
            )
            _fresh_output(args.output)
            _write_json(args.output / "metrics.json", report)
            with (args.output / "errors.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(report["rows"][0]))
                writer.writeheader()
                writer.writerows(report["rows"])
            logger.info("localization evaluation: %s", report["overall"])
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
