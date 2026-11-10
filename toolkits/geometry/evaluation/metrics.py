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

"""Independent model references and position errors; never perception inputs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from rpent.evaluation.result import write_json_atomic
from toolkits.geometry.evaluation.checklist import (
    frame_id,
    manifest_digest,
)


def truth_template(
    manifest: dict[str, Any], cases: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Create unlabelled references; null coordinates never mean a zero point."""
    return {
        "schema_version": 1,
        "manifest_sha256": manifest_digest(manifest),
        "units": "m",
        "annotations": [
            {
                "request_id": request["request_id"],
                "reference": request["reference"],
                "frame_id": frame_id(case["case_id"]),
                "definition": manifest["references"][request["reference"]][
                    "definition"
                ],
                "evaluable": None,
                "world_xyz": None,
                "annotation_source": None,
                "not_evaluable_reason": None,
            }
            for case in (manifest["cases"] if cases is None else cases)
            for request in case["requests"]
        ],
    }


def _xyz(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be three finite metre coordinates")
    return result


def _annotation_point(annotation: dict[str, Any]) -> list[float]:
    if annotation.get("world_xyz") is not None:
        if (
            annotation.get("model_local_xyz") is not None
            or annotation.get("barrel_profile") is not None
        ):
            raise ValueError(
                "use exactly one world, model-local or barrel-profile truth definition"
            )
        return _xyz(annotation["world_xyz"], "world_xyz").tolist()
    local = annotation.get("model_local_xyz")
    profile = annotation.get("barrel_profile")
    if profile is not None:
        if local is not None or "barrel_excluding_neck" not in annotation["definition"]:
            raise ValueError(
                "barrel_profile applies only to the fixed bottle barrel center"
            )
        axis = _xyz(profile["axis"], "barrel axis")
        if not np.isclose(np.linalg.norm(axis), 1, atol=1e-6):
            raise ValueError("barrel axis must be a unit vector in model coordinates")
        bounds = np.asarray(profile["bounds_m"], dtype=float)
        if (
            bounds.shape != (2,)
            or not np.isfinite(bounds).all()
            or bounds[0] >= bounds[1]
        ):
            raise ValueError("barrel bounds must be ordered finite metre coordinates")
        local = (
            _xyz(profile["axis_origin"], "barrel axis origin") + axis * bounds.mean()
        )
    point = _xyz(local, "model_local_xyz")
    pose = np.asarray(annotation.get("model_to_world"), dtype=float)
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-6)
    ):
        raise ValueError("model_to_world must be a finite rigid transform in metres")
    return (pose[:3, :3] @ point + pose[:3, 3]).tolist()


def validate_truth(
    manifest: dict[str, Any],
    truth: dict[str, Any],
    cases: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate definitions, transforms and independent labels for selected cases."""
    if (
        truth.get("schema_version") != 1
        or truth.get("units") != "m"
        or truth.get("manifest_sha256") != manifest_digest(manifest)
    ):
        raise ValueError("truth schema, metre units or manifest hash mismatch")
    expected = {
        row["request_id"]: row for row in truth_template(manifest)["annotations"]
    }
    result = {}
    for row in truth.get("annotations", []):
        request_id = row.get("request_id")
        if request_id not in expected or request_id in result:
            raise ValueError(f"unknown or duplicate truth request: {request_id}")
        for key in ("reference", "frame_id", "definition"):
            if row.get(key) != expected[request_id][key]:
                raise ValueError(f"truth {key} mismatch: {request_id}")
        if (
            type(row.get("evaluable")) is not bool
            or not isinstance(row.get("annotation_source"), str)
            or not row["annotation_source"].strip()
        ):
            raise ValueError(
                f"independent evaluable label and annotation_source required: {request_id}"
            )
        resolved = dict(row)
        if row["evaluable"]:
            resolved["world_xyz"] = _annotation_point(row)
        elif (
            not isinstance(row.get("not_evaluable_reason"), str)
            or not row["not_evaluable_reason"].strip()
        ):
            raise ValueError(f"not_evaluable_reason required: {request_id}")
        elif any(
            row.get(key) is not None
            for key in ("world_xyz", "model_local_xyz", "barrel_profile")
        ):
            raise ValueError(
                f"not-evaluable reference must not carry a truth point: {request_id}"
            )
        if not row["evaluable"]:
            resolved["world_xyz"] = None
        result[request_id] = resolved
    required = {
        row["request_id"] for row in truth_template(manifest, cases)["annotations"]
    }
    if not required <= result.keys():
        raise ValueError(
            f"truth annotations incomplete: {len(required - result.keys())} selected requests lack references"
        )
    return result


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in rows if row["evaluable"]]
    errors = np.asarray(
        [row["error_m"] for row in evaluable if row["error_m"] is not None]
    )
    n_eval, n_return = len(evaluable), len(errors)
    returned = sum(row["returned"] for row in rows)
    return {
        "planned": len(rows),
        "evaluable": n_eval,
        "not_evaluable": len(rows) - n_eval,
        "attempted": sum(row["prediction_status"] != "unrun" for row in rows),
        "unrun": sum(row["prediction_status"] == "unrun" for row in rows),
        "returned": returned,
        "return_rate": returned / len(rows) if rows else None,
        "returned_on_evaluable": n_return,
        "refused_or_failed_on_evaluable": sum(
            row["error_m"] is None and row["prediction_status"] != "unrun"
            for row in evaluable
        ),
        "returned_on_not_evaluable": sum(
            row["returned"] and not row["evaluable"] for row in rows
        ),
        "coverage_on_evaluable": n_return / n_eval if n_eval else None,
        "conditional_error_mm_on_returned_only": {
            "mean": float(errors.mean() * 1000) if n_return else None,
            "median": float(np.median(errors) * 1000) if n_return else None,
            "p95": float(np.quantile(errors, 0.95) * 1000) if n_return else None,
            "max": float(errors.max() * 1000) if n_return else None,
            "rmse": float(np.sqrt(np.mean(errors**2)) * 1000) if n_return else None,
        },
    }


def evaluate(
    manifest: dict[str, Any],
    truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare corresponding centers and retain missing/error rows in coverage."""
    annotations = validate_truth(manifest, truth, cases)
    case_by_id = {case["case_id"]: case for case in manifest["cases"]}
    indexed = {}
    for prediction in predictions:
        key = prediction.get("request_id")
        if key not in annotations or key in indexed:
            raise ValueError(f"unknown or duplicate prediction: {key}")
        expected = annotations[key]
        case_id = key.rsplit("/", 1)[0]
        ref = manifest["references"][expected["reference"]]
        for name, value in {
            "reference": expected["reference"],
            "frame_id": expected["frame_id"],
            "units": "m",
            "manifest_sha256": manifest_digest(manifest),
            "quantity": ref["quantity"],
            "definition": ref["definition"],
            "step": 0,
            "camera": "agentview",
            "case_id": case_id,
        }.items():
            if prediction.get(name) != value:
                raise ValueError(f"prediction {name} mismatch: {key}")
        point = prediction.get("world_xyz")
        if prediction.get("status") not in (
            "usable",
            "partial",
            "need_observation",
            "error",
        ):
            raise ValueError(f"unknown prediction status: {key}")
        if point is not None:
            _xyz(point, "prediction world_xyz")
            if prediction.get("status") not in ("usable", "partial"):
                raise ValueError(f"failed prediction carries a point: {key}")
        elif prediction.get("status") == "usable":
            raise ValueError(f"usable prediction has no point: {key}")
        indexed[key] = prediction
    selected_ids = {
        case["case_id"] for case in (manifest["cases"] if cases is None else cases)
    }
    if not selected_ids <= case_by_id.keys():
        raise ValueError("evaluation contains an unknown case")
    rows, groups = [], defaultdict(list)
    for key, annotation in annotations.items():
        case_id = key.rsplit("/", 1)[0]
        if case_id not in selected_ids:
            continue
        prediction = indexed.get(key, {})
        point = prediction.get("world_xyz")
        error = (
            float(
                np.linalg.norm(
                    _xyz(point, "prediction") - _xyz(annotation["world_xyz"], "truth")
                )
            )
            if annotation["evaluable"] and point is not None
            else None
        )
        row = {
            "request_id": key,
            "reference": annotation["reference"],
            "suite": case_by_id[case_id]["suite"],
            "case_id": case_id,
            "quantity": manifest["references"][annotation["reference"]]["quantity"],
            "definition": annotation["definition"],
            "frame_id": annotation["frame_id"],
            "evaluable": annotation["evaluable"],
            "not_evaluable_reason": annotation["not_evaluable_reason"],
            "prediction_status": prediction.get("status", "unrun"),
            "prediction_reason": prediction.get("reason"),
            "returned": point is not None,
            "world_xyz": point,
            "truth_world_xyz": annotation["world_xyz"],
            "error_m": error,
            "error_mm": None if error is None else error * 1000,
        }
        rows.append(row)
        groups[("by_reference", row["reference"])].append(row)
        groups[("by_suite", row["suite"])].append(row)
    report = {
        "schema_version": 1,
        "manifest_sha256": manifest_digest(manifest),
        "units": "m",
        "scope": "initial-observation reference-point geometry; not manipulation task success",
        "overall": _metrics(rows),
        "by_reference": {},
        "by_suite": {},
        "rows": rows,
    }
    for (group, name), values in groups.items():
        report[group][name] = _metrics(values)
    return report


def write_report(
    manifest: dict[str, Any],
    report: dict[str, Any],
    output: Path,
    *,
    provenance: dict[str, Any],
) -> None:
    """Write machine-readable errors and a bilingual report with every request."""
    report = {**report, "provenance": provenance}
    write_json_atomic(output / "metrics.json", report)
    columns = [
        "request_id",
        "label",
        "quantity",
        "definition",
        "frame_id",
        "prediction_status",
        "prediction_reason",
        "evaluable",
        "not_evaluable_reason",
        "predicted_x_m",
        "predicted_y_m",
        "predicted_z_m",
        "reference_x_m",
        "reference_y_m",
        "reference_z_m",
        "error_mm",
    ]
    with (output / "errors.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in report["rows"]:
            point = row["world_xyz"] or [None] * 3
            truth = row["truth_world_xyz"] or [None] * 3
            writer.writerow(
                {
                    **{key: row[key] for key in columns if key in row},
                    "label": manifest["references"][row["reference"]]["label"],
                    **{
                        f"predicted_{axis}_m": value
                        for axis, value in zip("xyz", point, strict=True)
                    },
                    **{
                        f"reference_{axis}_m": value
                        for axis, value in zip("xyz", truth, strict=True)
                    },
                }
            )
    overall = report["overall"]
    run = provenance.get("prediction_run", {})
    lines = [
        "# LIBERO initial-frame geometry / LIBERO 起始帧几何评估",
        "",
        "Coordinates are metres; position errors are millimetres. "
        "坐标单位为米，位置误差单位为毫米。",
        "",
        f"Planned / 计划项数: **{overall['planned']}**; "
        f"attempted / 已运行: **{overall['attempted']}**; "
        f"returned / 返回坐标: **{overall['returned']}**.",
        "",
        f"Prediction revision / 预测版本: `{run.get('git_commit', run.get('commit', 'not recorded / 未记录'))}`; "
        f"perception / 感知输入: `{run.get('perception_mode', run.get('stage', 'not recorded / 未记录'))}`; "
        f"permissive / 历史宽松模式: `{run.get('permissive', False)}`.",
        "",
        "Errors below are conditional on an evaluable reference and a returned point. "
        "Missing, failed and unrun requests remain in coverage counts. "
        "以下误差只统计有参考点且返回坐标的项；未返回、失败和未运行项保留在返回率统计中。",
        "",
        "| Target / 部位 | Planned / 计划 | Returned / 返回 | Mean mm / 均值 | Median mm / 中位数 | P95 mm | Max mm / 最大值 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def number(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f}"

    if run.get("result_kind") == "historical":
        lines[2:2] = [
            "**Historical measurements, re-scored for reproducibility; not a new run of the current GeometryTool. "
            "历史测量结果，仅重新计算误差以复现统计，不是当前 GeometryTool 的新实验。**",
            "",
        ]

    for name, metrics in [("all", overall), *report["by_reference"].items()]:
        label = "All / 全部" if name == "all" else manifest["references"][name]["label"]
        errors = metrics["conditional_error_mm_on_returned_only"]
        lines.append(
            f"| {label} | {metrics['planned']} | {metrics['returned']} | "
            + " | ".join(
                number(errors[key]) for key in ("mean", "median", "p95", "max")
            )
            + " |"
        )
    lines += [
        "",
        "## Per-request coordinates / 逐项坐标",
        "",
        "| Request | Predicted XYZ m / 测量坐标 | Reference XYZ m / 参考坐标 | Error mm / 误差 | Status / 状态 |",
        "| --- | --- | --- | ---: | --- |",
    ]

    def coordinates(point: list[float] | None) -> str:
        return "—" if point is None else ", ".join(f"{value:.6f}" for value in point)

    for row in report["rows"]:
        lines.append(
            f"| {row['request_id']} | {coordinates(row['world_xyz'])} | "
            f"{coordinates(row['truth_world_xyz'])} | {number(row['error_mm'])} | "
            f"{row['prediction_status']} |"
        )
    lines += [
        "",
        "Full-precision coordinates and failure reasons are in `errors.csv`. "
        "Source hashes, frame definitions and grouped statistics are in `metrics.json`. "
        "完整精度坐标及失败原因见 `errors.csv`；来源哈希、坐标系和分组统计见 `metrics.json`。",
        "",
        "This evaluates initial-frame geometry, not manipulation success or calibrated uncertainty. "
        "本报告评估起始帧几何误差，不代表操作任务成功率或经过标定的不确定度。",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
