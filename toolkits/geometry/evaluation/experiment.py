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

"""Collect, replay and evaluate LIBERO initial-frame geometry independently."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rpent.evaluation.result import write_json_atomic
from rpent.session import EnvState
from rpent.utils.logging import get_logger
from toolkits.geometry.evaluation.checklist import (
    build_manifest,
    frame_id,
    load_manifest,
    manifest_digest,
    select_cases,
)
from toolkits.geometry.evaluation.observation import (
    FrozenObservationProvider,
    file_digest,
    load_observation,
    save_observation,
)
from toolkits.geometry.evaluation.replay import append_jsonl, read_jsonl
from toolkits.geometry.geometry_tool import GeometryTool
from toolkits.geometry.perception import Segmenter, VisualGrounder

logger = get_logger("geometry_evaluation")


def fresh_output(path: Path) -> None:
    """Require a fresh destination; do not overwrite an earlier experiment."""
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output must be a new or empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def run_header(
    manifest: dict[str, Any], cases: list[dict[str, Any]], stage: str
) -> dict[str, Any]:
    """Record the exact source, checklist and observation protocol for this run."""
    directory = Path(__file__).resolve().parent
    sources = sorted(directory.glob("*.py")) + sorted(directory.parent.glob("*.py"))
    revision, dirty = None, None
    try:
        top = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        if directory == Path(top) / "toolkits/geometry/evaluation":
            revision = subprocess.check_output(
                ["git", "-C", top, "rev-parse", "HEAD"], text=True, timeout=10
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    [
                        "git",
                        "-C",
                        top,
                        "status",
                        "--porcelain",
                        "--",
                        "toolkits/geometry",
                    ],
                    text=True,
                    timeout=10,
                ).strip()
            )
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "schema_version": 1,
        "stage": stage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": revision,
        "git_dirty": dirty,
        "manifest_sha256": manifest_digest(manifest),
        "protocol": manifest["protocol"],
        "selected_cases": [case["case_id"] for case in cases],
        "selected_requests": sum(len(case["requests"]) for case in cases),
        "source_sha256": {
            str(path.relative_to(directory.parent)): file_digest(path)
            for path in sources
        },
    }


def collect_cases(
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
    output: Path,
    *,
    height: int = 1024,
    width: int = 1024,
) -> dict[str, Any]:
    """Own a capture-only simulator per case and persist independent model state."""
    from toolkits.geometry.evaluation.libero import capture_case

    fresh_output(output)
    write_json_atomic(
        output / "run_manifest.json", run_header(manifest, cases, "collect")
    )
    rows = []
    for case in cases:
        row = {"case_id": case["case_id"], "status": "error"}
        try:
            observation, snapshot = capture_case(case, height=height, width=width)
            root = output / case["case_id"]
            save_observation(root, manifest, case, observation)
            write_json_atomic(root / "evaluator_snapshot.json", snapshot)
            row["status"] = "captured"
        except Exception as exc:
            row["reason"] = str(exc)
            logger.exception("initial-frame capture failed: %s", case["case_id"])
        rows.append(row)
        write_json_atomic(output / "collection.json", {"cases": rows})
    summary = {
        "selected_cases": len(cases),
        "captured": sum(row["status"] == "captured" for row in rows),
        "failed": sum(row["status"] == "error" for row in rows),
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def prediction_base(
    manifest: dict[str, Any], case: dict[str, Any], request: dict[str, Any]
) -> dict[str, Any]:
    """Describe one unchanged checklist target, including its point definition."""
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
        "fit_accepted": False,
        "warnings": [],
    }


def infer_cases(
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
    inputs: Path,
    output: Path,
    *,
    grounder: VisualGrounder,
    segmenter: Segmenter,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Call the current GeometryTool for every request, without opening references."""
    predictions = []
    for case in cases:
        root = output / case["case_id"]
        error = None
        try:
            observation = load_observation(inputs / case["case_id"], manifest, case)
            state = EnvState(root / "episode")
            with state.record_step(state={}):
                pass
            geometry_tool = GeometryTool(
                state=state,
                observations=FrozenObservationProvider(observation),
                grounder=grounder,
                segmenter=segmenter,
                model=model,
            )
        except Exception as exc:
            error = str(exc)
        for request in case["requests"]:
            row = prediction_base(manifest, case, request)
            if error is not None:
                row["reason"] = error
            else:
                try:
                    data = geometry_tool.geometry(
                        row["query"],
                        camera="agentview",
                        step=0,
                        required_geometry=[row["quantity"]],
                    ).data
                    write_json_atomic(root / f"result_{request['slot']}.json", data)
                    field = data.get("geometry", {}).get(row["quantity"], {})
                    point = field.get("value")
                    if point is not None:
                        for key, expected in {
                            "world_frame_id": row["frame_id"],
                            "definition": row["definition"],
                            "source_step": 0,
                            "unit": "m",
                        }.items():
                            if field.get(key) != expected:
                                raise ValueError(f"returned field {key} mismatch")
                        xyz = np.asarray(point, dtype=float)
                        if xyz.shape != (3,) or not np.isfinite(xyz).all():
                            raise ValueError("geometry did not return a finite center")
                    row.update(
                        status=data["status"],
                        world_xyz=point,
                        fit_accepted=field.get("fit_accepted", False),
                        warnings=field.get("warnings", []),
                        reason=field.get("reason")
                        or data.get("error")
                        or data.get("reason"),
                        calls=data.get("calls"),
                        elapsed_ms=data.get("elapsed_ms"),
                        geometry_artifact=data.get("geometry_artifact"),
                        artifact_sha256=dict(observation.source_refs),
                    )
                except Exception as exc:
                    row["reason"] = str(exc)
            append_jsonl(output / "results.jsonl", row)
            predictions.append(row)
            logger.info("geometry %s: %s", row["request_id"], row["status"])
    return predictions


def infer(
    args: argparse.Namespace, manifest: dict[str, Any], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Run live clients with recording, or replay without contacting any service."""
    from toolkits.geometry.evaluation.replay import (
        RecordingGrounder,
        RecordingSegmenter,
        ReplayGrounder,
        ReplaySegmenter,
    )

    if not args.inputs.is_dir():
        raise ValueError(f"observation collection missing: {args.inputs}")
    if args.replay:
        if args.molmo_endpoint or args.sam3_endpoint:
            raise ValueError("--replay cannot be combined with live model arguments")
        grounder, segmenter = ReplayGrounder(args.replay), ReplaySegmenter(args.replay)
    else:
        if not args.molmo_endpoint or not args.sam3_endpoint:
            raise ValueError(
                "live inference needs --molmo-endpoint and --sam3-endpoint"
            )
        from rpent.robots.components.molmo_client import MolmoClient
        from rpent.robots.components.sam3_client import Sam3Client
        from rpent.utils.rpc import make_rpc_client

        grounder = MolmoClient(
            make_rpc_client(args.molmo_endpoint), point_mode="geometry"
        )
        segmenter = Sam3Client(make_rpc_client(args.sam3_endpoint))
    fresh_output(args.output)
    header = run_header(manifest, cases, "infer")
    header.update(
        perception_mode="frozen" if args.replay else "live",
        geometry_model=None if args.replay else "Molmo2-8B",
        fit_profile="current_GeometryTool_defaults",
    )
    if args.replay:
        header["replay_sha256"] = {
            path.name: file_digest(path)
            for path in args.replay.iterdir()
            if path.is_file()
        }
    else:
        grounder = RecordingGrounder(grounder, args.output / "perception")
        segmenter = RecordingSegmenter(segmenter, args.output / "perception")
    write_json_atomic(args.output / "run_manifest.json", header)
    rows = infer_cases(
        manifest,
        cases,
        args.inputs,
        args.output,
        grounder=grounder,
        segmenter=segmenter,
        model=header["geometry_model"],
    )
    summary = {
        "planned": sum(len(case["requests"]) for case in cases),
        "recorded": len(rows),
        "returned": sum(row["world_xyz"] is not None for row in rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "perception_mode": header["perception_mode"],
        "fresh_grounding_calls": 0
        if args.replay
        else sum((row.get("calls") or {}).get("grounding", 0) for row in rows),
        "fresh_segmentation_calls": 0
        if args.replay
        else sum((row.get("calls") or {}).get("segment", 0) for row in rows),
        "accuracy": None,
    }
    write_json_atomic(args.output / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in (
        "write-manifest",
        "truth-template",
        "collect",
        "build-truth",
        "infer",
        "evaluate",
        "freeze-perception",
    ):
        sub = commands.add_parser(command)
        sub.add_argument("--output", type=Path, required=True)
        sub.set_defaults(manifest=None, case=None, limit=None)
        if command not in ("write-manifest", "freeze-perception"):
            sub.add_argument("--manifest", type=Path)
            sub.add_argument("--case", action="append")
            sub.add_argument("--limit", type=int)
        if command in ("infer", "build-truth"):
            sub.add_argument("--inputs", type=Path, required=True)
        if command == "collect":
            sub.add_argument("--height", type=int, default=1024)
            sub.add_argument("--width", type=int, default=1024)
            sub.add_argument("--cuda-device", type=int)
        elif command == "build-truth":
            sub.add_argument("--profiles", type=Path)
        elif command == "infer":
            sub.add_argument(
                "--molmo-endpoint",
                help="RPC endpoint of an existing Molmo2-8B service.",
            )
            sub.add_argument("--sam3-endpoint")
            sub.add_argument("--replay", type=Path)
        elif command == "evaluate":
            sub.add_argument("--predictions", type=Path, required=True)
            sub.add_argument("--truth", type=Path, required=True)
        elif command == "freeze-perception":
            sub.add_argument(
                "--predictions",
                type=Path,
                required=True,
                help="Legacy directory containing results.jsonl and per-case evidence.",
            )
            sub.add_argument("--grounding", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one explicit stage; imports and help require no simulator or model stack."""
    parser = _parser()
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest) if args.manifest else build_manifest()
    cases = select_cases(manifest, args.case, args.limit)
    if args.command in ("write-manifest", "truth-template", "build-truth"):
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.command == "write-manifest":
            result = manifest
        elif args.command == "truth-template":
            from toolkits.geometry.evaluation.metrics import truth_template

            result = truth_template(manifest, cases)
        else:
            from toolkits.geometry.evaluation.references import build_truth

            profiles = json.loads(args.profiles.read_text()) if args.profiles else None
            result = build_truth(manifest, cases, args.inputs, profiles=profiles)
        write_json_atomic(args.output, result)
        return 0
    if args.command == "collect":
        if args.height <= 0 or args.width <= 0:
            raise ValueError("image dimensions must be positive")
        if args.cuda_device is not None:
            if args.cuda_device < 0 or os.environ.get("CUDA_VISIBLE_DEVICES"):
                raise ValueError(
                    "use a nonnegative --cuda-device without a conflicting CUDA_VISIBLE_DEVICES mask"
                )
            from rpent.utils.egl import configure_egl_device

            configure_egl_device(args.cuda_device)
            import torch

            torch.cuda.set_device(args.cuda_device)
        result = collect_cases(
            manifest, cases, args.output, height=args.height, width=args.width
        )
        return int(result["failed"] > 0)
    if args.command == "infer":
        result = infer(args, manifest, cases)
        return int(result["status_counts"].get("error", 0) > 0)
    fresh_output(args.output)
    if args.command == "freeze-perception":
        from toolkits.geometry.evaluation.replay import freeze_legacy

        freeze_legacy(args.predictions, args.grounding, args.output)
        return 0
    from toolkits.geometry.evaluation.metrics import evaluate, write_report

    report = evaluate(
        manifest,
        json.loads(args.truth.read_text()),
        read_jsonl(args.predictions),
        cases=cases,
    )
    provenance = {
        "predictions_sha256": file_digest(args.predictions),
        "truth_sha256": file_digest(args.truth),
        "evaluation": run_header(manifest, cases, "evaluate"),
    }
    run_path = args.predictions.parent / "run_manifest.json"
    if run_path.is_file():
        original = json.loads(run_path.read_text())
        provenance["prediction_run"] = {
            key: original[key]
            for key in (
                "commit",
                "git_tree",
                "git_commit",
                "git_dirty",
                "geometry_model",
                "perception_mode",
                "permissive",
                "fit_profile",
                "stage",
                "created_at",
                "result_kind",
            )
            if key in original
        }
    write_report(manifest, report, args.output, provenance=provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
